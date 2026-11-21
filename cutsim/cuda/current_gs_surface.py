"""Material reconstruction from the CURRENT finite, clipped, oriented GS only.

This is a sampled surface estimator, not an exact solid oracle. Intersections
of overlapping, equally oriented surfels are consolidated; opposite crossings
are never distance-merged. Missing ray coverage uses the nearest finite patch's
orientation, explicitly counted in diagnostics. No cutter/program is accepted.
"""
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from cutsim.rendering.gs_extent import support_factor, support_radius
from scipy.spatial import cKDTree
from numba import njit

from cutsim.rendering.profile_cut_clip import surviving_support
from cutsim.cuda.profile_ray_domain import WINDOW_BOUNDARY_OWNER, ray_bounds


@njit(cache=True)
def _distance(q,g):
    delta=q-g[:3]
    x=np.dot(delta,g[3:6]);y=np.dot(delta,g[6:9]);h=np.dot(delta,g[9:12]);r=g[12]
    a,b,d,e,f,k=g[13:19]
    best=np.inf
    if x*x+y*y<=r*r and a*x+b*y+d>=0 and e*x+f*y+k>=0:best=0.
    length=np.hypot(x,y)
    if length>0:
        xx=x*r/length;yy=y*r/length
        if a*xx+b*yy+d>=-1e-12 and e*xx+f*yy+k>=-1e-12:
            best=min(best,(xx-x)**2+(yy-y)**2)
    for slot in range(2):
        nx,ny,dd=(a,b,d) if slot==0 else (e,f,k)
        ox,oy,od=(e,f,k) if slot==0 else (a,b,d)
        if not np.isfinite(dd) or abs(dd)>r or nx*nx+ny*ny<.5:continue
        cx=-dd*nx;cy=-dd*ny;tx=-ny;ty=nx
        extent=np.sqrt(max(0.,r*r-dd*dd));lo=-extent;hi=extent
        slope=ox*tx+oy*ty;value=ox*cx+oy*cy+od
        if slope>1e-14:lo=max(lo,-value/slope)
        elif slope<-1e-14:hi=min(hi,-value/slope)
        elif value<0:continue
        if lo>hi:continue
        t=min(hi,max(lo,(x-cx)*tx+(y-cy)*ty))
        best=min(best,(cx+t*tx-x)**2+(cy+t*ty-y)**2)
    return np.sqrt(h*h+best),h


@njit(cache=True,nogil=True)
def _nearest_cpu(points,geometry,rows,offsets):
    values=np.full(len(points),np.inf);chosen=np.full(len(points),-1,np.int64)
    for i in range(len(points)):
        best=np.inf
        for j in range(offsets[i],offsets[i+1]):
            row=rows[j];distance,height=_distance(points[i],geometry[row])
            if distance<best or (distance==best and row<chosen[i]):
                best=distance;chosen[i]=row;values[i]=distance*(-1 if height<0 else 1)
    return values,chosen


@njit(cache=True,nogil=True)
def _hits_cpu(origins,directions,domain,geometry,rows,rays,weight_coefficient=2.):
    result=np.full((len(rows),3),np.nan)
    for i in range(len(rows)):
        ray=rays[i];g=geometry[rows[i]];slope=np.dot(directions[ray],g[9:12])
        if abs(slope)<1e-10:continue
        t=np.dot(g[:3]-origins[ray],g[9:12])/slope
        if t<domain[ray,0] or t>domain[ray,1]:continue
        delta=origins[ray]+t*directions[ray]-g[:3]
        x=np.dot(delta,g[3:6]);y=np.dot(delta,g[6:9]);r=g[12]
        if x*x+y*y>r*r+1e-12:continue
        if g[13]*x+g[14]*y+g[15]<-1e-12 or g[16]*x+g[17]*y+g[18]<-1e-12:continue
        result[i,0]=t;result[i,1]=1 if slope>0 else -1
        result[i,2]=abs(slope)*np.exp(-weight_coefficient*(x*x+y*y)/(r*r))
    return result


_CUDA=r'''
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
#define NAN (__longlong_as_double(0x7ff8000000000000LL))
__device__ double dot(const double* a,const double* b){return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2];}
__device__ double dist(const double* q,const double* g,double* height){
 double z[3]={q[0]-g[0],q[1]-g[1],q[2]-g[2]};
 double x=dot(z,g+3),y=dot(z,g+6),h=dot(z,g+9),r=g[12];*height=h;
 double a=g[13],b=g[14],d=g[15],e=g[16],f=g[17],k=g[18],best=INFINITY;
 if(x*x+y*y<=r*r && a*x+b*y+d>=0 && e*x+f*y+k>=0)best=0;
 double length=hypot(x,y);
 if(length>0){double xx=x*r/length,yy=y*r/length;
  if(a*xx+b*yy+d>=-1e-12 && e*xx+f*yy+k>=-1e-12)best=fmin(best,(xx-x)*(xx-x)+(yy-y)*(yy-y));}
 for(int slot=0;slot<2;slot++){
  double nx=slot?e:a,ny=slot?f:b,dd=slot?k:d,ox=slot?a:e,oy=slot?b:f,od=slot?d:k;
  if(!isfinite(dd)||fabs(dd)>r||nx*nx+ny*ny<.5)continue;
  double cx=-dd*nx,cy=-dd*ny,tx=-ny,ty=nx,extent=sqrt(fmax(0.,r*r-dd*dd));
  double lo=-extent,hi=extent,slope=ox*tx+oy*ty,value=ox*cx+oy*cy+od;
  if(slope>1e-14)lo=fmax(lo,-value/slope);else if(slope<-1e-14)hi=fmin(hi,-value/slope);else if(value<0)continue;
  if(lo>hi)continue;
  double t=fmin(hi,fmax(lo,(x-cx)*tx+(y-cy)*ty));
  best=fmin(best,(cx+t*tx-x)*(cx+t*tx-x)+(cy+t*ty-y)*(cy+t*ty-y));
 }return sqrt(h*h+best);
}
extern "C" __global__ void nearest(const double* points,const double* geometry,const long long* rows,
 const long long* offsets,long long count,double* value,long long* chosen){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 double best=INFINITY;value[i]=INFINITY;chosen[i]=-1;
 for(long long j=offsets[i];j<offsets[i+1];j++){long long row=rows[j];double h;
  double dd=dist(points+3*i,geometry+19*row,&h);
  if(dd<best||(dd==best&&row<chosen[i])){best=dd;chosen[i]=row;value[i]=dd*(h<0?-1.:1.);}}
}
extern "C" __global__ void hits(const double* origins,const double* directions,const double* domain,
 const double* geometry,const long long* rows,const long long* rays,long long count,double* result){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* g=geometry+19*rows[i];long long ray=rays[i];const double* o=origins+3*ray;const double* d=directions+3*ray;
 result[3*i]=NAN;double slope=dot(d,g+9);if(fabs(slope)<1e-10)return;
 double delta[3]={g[0]-o[0],g[1]-o[1],g[2]-o[2]},t=dot(delta,g+9)/slope;
 if(t<domain[2*ray]||t>domain[2*ray+1])return;
 double p[3]={o[0]+t*d[0]-g[0],o[1]+t*d[1]-g[1],o[2]+t*d[2]-g[2]};
 double x=dot(p,g+3),y=dot(p,g+6),r=g[12];if(x*x+y*y>r*r+1e-12)return;
 if(g[13]*x+g[14]*y+g[15]<-1e-12||g[16]*x+g[17]*y+g[18]<-1e-12)return;
 result[3*i]=t;result[3*i+1]=slope>0?1:-1;result[3*i+2]=fabs(slope)*exp(-2*(x*x+y*y)/(r*r));
}
'''


def _csr(lists):
    counts=np.fromiter(map(len,lists),np.int64,count=len(lists))
    return (np.concatenate(lists).astype(np.int64) if counts.sum() else np.empty(0,np.int64),
            np.r_[0,np.cumsum(counts)])


class CurrentGSSurface:
    def __init__(self,snapshot,backend='cpu',*,cache=None,spatial_index=True,selection=None):
        if backend not in ('cpu','cuda'):raise ValueError('Resolved CPU/CUDA backend required')
        self.snapshot=snapshot;self.backend=backend;self.rt=None;self.device_geometry=None
        self.readback_bytes=0
        self.cache=cache
        if cache is not None:cache.bind(snapshot,backend)
        p=snapshot.endpoints['positions'];n=snapshot.endpoints['normals']
        certificate=getattr(snapshot,'material_clipping',None) or snapshot.clipping
        clip=certificate.planes if certificate is not None else np.tile([0,0,np.inf,0,0,np.inf,np.inf],(len(p),1))
        radius=np.minimum(support_factor(snapshot)*snapshot.gaussian.attributes['scale'][:,0],clip[:,6])
        keep=surviving_support(clip,radius)
        if selection is not None:keep &= selection.mask
        self.rows=np.flatnonzero(keep)
        self.geometry=np.ascontiguousarray(np.column_stack((p,snapshot.gaussian.u,snapshot.gaussian.v,n,radius,clip[:,:6]))[keep])
        self.source=SimpleNamespace(positions=p,normals=n,tool_ids=snapshot.endpoints['tool_ids'])
        self.max_radius=float(radius[keep].max(initial=0))
        self.tree=cKDTree(p[keep]) if len(self.rows) and spatial_index else None
        self.xy_tree=None
        self.stats=dict(source_gs=len(p),active_gs=len(self.rows),query_points=0,query_pairs=0,ray_pairs=0,
            endpoint_edge_queries=0,endpoint_edge_resolved=0,endpoint_side_changes=0,
            material_edge_queries=0,material_edge_resolved=0,material_side_changes=0,
            material_confirm_queries=0,material_conflicts=0)
        if backend=='cuda':
            from cutsim.cuda.profile_dexel_cuda import runtime
            self.rt=runtime()
            with self.rt.context():
                cp=self.rt.cp;module=cp.RawModule(code=_CUDA.replace("exp(-2*",f"exp(-{.5*support_factor(snapshot)**2}*"),options=('--fmad=false',))
                self.kernels={name:module.get_function(name) for name in ('nearest','hits')}
                self.device_geometry=cp.asarray(self.geometry)

    def _nearest(self,points,rows,offsets):
        self.stats['query_pairs']+=len(rows)
        if self.backend=='cpu':return _nearest_cpu(points,self.geometry,rows,offsets)
        rt=self.rt;cp=rt.cp
        with rt.context():
            value=cp.empty(len(points),cp.float64);chosen=cp.empty(len(points),cp.int64)
            if len(points):self.kernels['nearest'](((len(points)+255)//256,),(256,),
                (cp.asarray(points),self.device_geometry,cp.asarray(rows),cp.asarray(offsets),np.int64(len(points)),value,chosen))
            self.readback_bytes+=value.nbytes+chosen.nbytes
            return cp.asnumpy(value,stream=rt.stream),cp.asnumpy(chosen,stream=rt.stream)

    def query(self,points,*,use_cache=True):
        """Exact nearest clipped disc; its oriented normal estimates material sign.

        The search radius is certified from a first finite-patch distance, so
        this does not silently cap the number of neighbouring GS at k samples.
        Returned owners index the input snapshot, not a machining operation.
        """
        points=np.ascontiguousarray(points,dtype=float).reshape(-1,3)
        self.stats['query_points']+=len(points)
        if self.cache is not None and use_cache:return self.cache.query(points,self._query_uncached)
        return self._query_uncached(points)

    def _query_uncached(self,points):
        if self.tree is None:return np.full(len(points),np.inf),np.full(len(points),-1,np.int64)
        values=[];owners=[]
        for start in range(0,len(points),2048):
            q=points[start:start+2048];_,near=self.tree.query(q)
            first,_=self._nearest(q,np.asarray(near,np.int64),np.arange(len(q)+1,dtype=np.int64))
            lists=self.tree.query_ball_point(q,np.abs(first)+self.max_radius+1e-10)
            rows,offsets=_csr(lists);value,chosen=self._nearest(q,rows,offsets)
            values.append(value);owners.append(np.where(chosen>=0,self.rows[np.maximum(chosen,0)],-1))
        return (np.concatenate(values),np.concatenate(owners)) if values else (np.empty(0),np.empty(0,np.int64))

    def patch_crossings(self,inside,outside,owners):
        """Certify an inside-to-outside segment crosses the selected finite GS.

        A nearest-patch sign at the outside probe alone is insufficient: its
        supporting plane can intersect the segment beyond the clipped disc.
        This uses exactly the material footprint used by ray intersections.
        """
        inside=np.asarray(inside,float).reshape(-1,3);outside=np.asarray(outside,float).reshape(-1,3)
        owners=np.asarray(owners,np.int64).reshape(-1)
        if len(inside)!=len(outside) or len(inside)!=len(owners):raise ValueError('Matching segment/GS counts required')
        valid=np.zeros(len(owners),bool)
        if not len(self.rows) or not len(owners):return valid
        index=np.searchsorted(self.rows,owners);found=index<len(self.rows)
        found &= self.rows[np.minimum(index,len(self.rows)-1)]==owners
        ids=np.flatnonzero(found);g=self.geometry[index[ids]]
        a=inside[ids]-g[:,:3];b=outside[ids]-g[:,:3]
        ha=np.sum(a*g[:,9:12],axis=1);hb=np.sum(b*g[:,9:12],axis=1)
        # The centre was accepted as material. Never apply an extrapolated
        # constraint which declares that same centre to be outside.
        crossing=(ha<=0.) & (hb>0.)
        t=np.divide(-ha,hb-ha,out=np.zeros(len(ids)),where=crossing)
        hit=a+t[:,None]*(b-a)
        x=np.sum(hit*g[:,3:6],axis=1);y=np.sum(hit*g[:,6:9],axis=1)
        crossing &= x*x+y*y<=g[:,12]**2+1e-12
        crossing &= g[:,13]*x+g[:,14]*y+g[:,15]>=-1e-12
        crossing &= g[:,16]*x+g[:,17]*y+g[:,18]>=-1e-12
        valid[ids]=crossing
        return valid

    def intersections(self,grid):
        domain=ray_bounds(grid)
        if domain is None:raise ValueError('Current-GS reconstruction requires finite local ray windows')
        origins=np.ascontiguousarray(grid.origins);directions=np.ascontiguousarray(grid.directions)
        chunks=[]
        if self.tree is None:return np.empty((0,5))
        for start in range(0,len(origins),128):
            stop=min(start+128,len(origins));o=origins[start:stop];d=directions[start:stop];b=domain[start:stop]
            if np.all(d[:,:2]==0):
                # A vertical line can only hit discs whose XY projections
                # reach that line. Avoid a large 3-D search sphere spanning
                # the whole stock depth for material confirmation rays.
                if self.xy_tree is None:self.xy_tree=cKDTree(self.geometry[:,:2])
                rows,offsets=_csr(self.xy_tree.query_ball_point(o[:,:2],self.max_radius+1e-10))
            else:
                middle=o+np.mean(b,axis=1)[:,None]*d
                reach=(b[:,1]-b[:,0])*.5+self.max_radius+1e-10
                rows,offsets=_csr(self.tree.query_ball_point(middle,reach))
            rays=np.repeat(np.arange(len(o),dtype=np.int64),np.diff(offsets));self.stats['ray_pairs']+=len(rows)
            if self.backend=='cpu':result=_hits_cpu(o,d,b,self.geometry,rows,rays,.5*support_factor(self.snapshot)**2)
            else:
                rt=self.rt;cp=rt.cp
                with rt.context():
                    result=cp.empty((len(rows),3),cp.float64)
                    if len(rows):self.kernels['hits'](((len(rows)+255)//256,),(256,),
                        (cp.asarray(o),cp.asarray(d),cp.asarray(b),self.device_geometry,cp.asarray(rows),cp.asarray(rays),np.int64(len(rows)),result))
                    valid=cp.flatnonzero(cp.isfinite(result[:,0]))
                    self.readback_bytes+=valid.nbytes+len(valid)*3*8
                    ids=cp.asnumpy(valid,stream=rt.stream);result=cp.asnumpy(result[valid],stream=rt.stream)
                    rows=rows[ids];rays=rays[ids]
            valid=np.isfinite(result[:,0]);result=result[valid];rows=rows[valid];rays=rays[valid]
            chunks.append(np.column_stack((rays+start,result[:,:2],-self.rows[rows]-1,result[:,2])))
        hits=np.concatenate(chunks) if chunks else np.empty((0,5))
        return hits[np.lexsort((hits[:,3],hits[:,1],hits[:,0]))] if len(hits) else hits

    def segment_boundaries(self,inside,outside):
        """Select the first oriented, finite GS crossing on each segment.

        The nearest GS at the far end need not be the surface crossed by the
        segment (for example, a groove floor instead of its side wall).
        Opposite orientations are kept separate, including thin material.
        """
        inside=np.asarray(inside,float).reshape(-1,3);outside=np.asarray(outside,float).reshape(-1,3)
        if len(inside)!=len(outside):raise ValueError('Matching segment endpoint counts required')
        direction=outside-inside;length=np.linalg.norm(direction,axis=1)
        chosen=np.full(len(inside),-1,np.int64);valid=np.flatnonzero(length>1e-12)
        if not len(valid):return chosen
        grid=SimpleNamespace(origins=inside[valid],directions=direction[valid]/length[valid,None],
            ray_bounds=np.column_stack((np.zeros(len(valid)),length[valid])))
        hits=self.intersections(grid)
        if not len(hits):return chosen
        starts=np.r_[0,1+np.flatnonzero(np.diff(hits[:,0]))];ends=np.r_[starts[1:],len(hits)]
        for start,end in zip(starts,ends):
            group=hits[start:end];opposite=np.flatnonzero(group[:,2]!=group[0,2])
            if len(opposite):group=group[:opposite[0]]
            hit=group[np.argmax(group[:,4])]
            chosen[valid[int(hit[0])]]=-int(hit[3])-1
        return chosen

    def material_query(self,points,*,use_cache=True,stats_prefix='material'):
        """Nearest finite distance with an oriented-crossing material sign.

        A point below the rim of a groove can be outside the material although
        it is below the nearest top GS's infinite plane. For edge-nearest
        queries, use the first oriented downward crossing of CURRENT GS.
        No initial stock, cutter history, or persistent volume is consulted.
        Missing ray coverage retains the explicitly approximate nearest sign.

        The returned distance magnitude and owner are still those of the
        nearest finite patch. Only the material sign is resolved by a ray.
        Cache only nearest-patch distances: a remote crossing can change even
        when the nearest patch and its distance ball are unchanged.
        """
        points=np.asarray(points,float).reshape(-1,3)
        signed,owners=self.query(points,use_cache=use_cache);signed=signed.copy();inside=signed<0
        valid=np.flatnonzero((owners>=0)&np.isfinite(signed))
        if not len(valid):return signed,owners
        height=np.sum((points[valid]-self.source.positions[owners[valid]])*
            self.source.normals[owners[valid]],axis=1)
        edge=valid[np.abs(signed[valid])>np.abs(height)+1e-7]
        self.stats[stats_prefix+'_edge_queries']+=len(edge)
        if len(edge):
            end=points[edge].copy()
            end[:,2]=np.minimum(end[:,2]-1e-6,np.min(self.geometry[:,2]-self.geometry[:,12])-1e-6)
            hit=self.segment_boundaries(points[edge],end);found=hit>=0
            # Downward exit (normal.z < 0) starts in material; entrance starts
            # in air. Keep zero-distance boundary samples excluded.
            resolved=self.source.normals[hit[found],2]<0
            # A downward ray through a missing floor may reach the stock's
            # bottom and falsely suggest solid material. Before changing air
            # to material require an upward EXIT as independent evidence.
            ids=edge[found];confirm=np.flatnonzero(resolved & ~inside[ids])
            if len(confirm):
                top=points[ids[confirm]].copy()
                top[:,2]=np.maximum(top[:,2]+1e-6,np.max(self.geometry[:,2]+self.geometry[:,12])+1e-6)
                upper=self.segment_boundaries(points[ids[confirm]],top)
                agrees=(upper>=0)&(self.source.normals[np.maximum(upper,0),2]>0)
                resolved[confirm[~agrees]]=False
                self.stats['material_confirm_queries']+=len(confirm)
                self.stats['material_conflicts']+=int(np.count_nonzero(~agrees))
            self.stats[stats_prefix+'_edge_resolved']+=int(np.count_nonzero(found))
            self.stats[stats_prefix+'_side_changes']+=int(np.count_nonzero(inside[edge[found]]!=resolved))
            signed[edge[found]]=np.abs(signed[edge[found]])*np.where(resolved,-1.,1.)
        return signed,owners

    def endpoint_inside(self,points,*,ray_has_hit=None):
        points=np.asarray(points,float).reshape(-1,3)
        signed,owners=self.material_query(points,stats_prefix='endpoint');inside=signed < -1e-8
        ids=np.flatnonzero(inside & (owners>=0))
        height=np.sum((points[ids]-self.source.positions[owners[ids]])*self.source.normals[owners[ids]],axis=1)
        if ray_has_hit is not None:
            evidence=np.asarray(ray_has_hit,bool)
            if evidence.shape!=(len(points),):raise ValueError('One hit-evidence flag per endpoint required')
            # A distant vertical confirmation plus a wholly inferred Dexel
            # interval must not regrow a surface through a sampling gap. When
            # the nearest patch says air, require an actual hit on that Dexel.
            inside[ids[(height>=0)&~evidence[ids]]]=False
        return inside

    def intervals(self,grid):
        started=perf_counter();domain=ray_bounds(grid);hits=self.intersections(grid)
        points=grid.origins+domain[:,0,None]*grid.directions
        counts=np.bincount(hits[:,0].astype(np.int64),minlength=len(points))
        offsets=np.r_[0,np.cumsum(counts)];lines=[];merged=0
        missing=np.flatnonzero(counts==0);inferred={}
        seed=np.full(len(points),np.nan)
        if len(missing):
            seed[missing],_=self.material_query(points[missing])
            tip=grid.origins[missing]+domain[missing,1,None]*grid.directions[missing]
            end,_=self.material_query(tip)
            crossing=missing[(seed[missing]<0)!=(end<0)]
            lo=domain[crossing,0].copy();hi=domain[crossing,1].copy()
            # No finite ray hit, but opposite signs at the window ends:
            # explicitly reconstruct a crossing in the nearest-patch field.
            for _ in range(24 if len(crossing) else 0):
                mid=(lo+hi)*.5;q=grid.origins[crossing]+mid[:,None]*grid.directions[crossing]
                value,owner=self.material_query(q);same=(value<0)==(seed[crossing]<0)
                lo=np.where(same,mid,lo);hi=np.where(same,hi,mid)
            if len(crossing):
                for ray,t,row in zip(crossing,(lo+hi)*.5,owner):inferred[int(ray)]=(float(t),-int(row)-1)
        # Same-orientation runs are redundant samples of a crossing. Use their
        # highest-confidence actual hit (rather than averaging away thin gaps).
        for ray,(low,high) in enumerate(domain):
            h=hits[offsets[ray]:offsets[ray+1]];groups=[];i=0
            while i<len(h):
                j=i+1
                while j<len(h) and h[j,2]==h[i,2]:j+=1
                groups.append(h[i+int(np.argmax(h[i:j,4]))]);merged+=j-i-1;i=j
            if ray in inferred:
                t,who=inferred[ray];direction=1 if seed[ray]<0 else -1
                groups=[np.array([ray,t,direction,who,1.])]
            inside=bool(seed[ray]<0) if not groups else bool(groups[0][2]>0)
            entry=low;owner=WINDOW_BOUNDARY_OWNER;line=[]
            for hit in groups:
                t=float(hit[1]);who=int(hit[3])
                if hit[2]<0:
                    if not inside:entry=t;owner=who
                    inside=True
                else:
                    if inside and entry<t:line.append((entry,t,owner,who))
                    inside=False
            if inside and entry<high:line.append((entry,float(high),owner,WINDOW_BOUNDARY_OWNER))
            lines.append(tuple(line))
        stats=dict(self.stats,hits=len(hits),merged_hits=merged,nearest_seed_rays=int(np.count_nonzero(counts==0)),inferred_crossings=len(inferred),
                   initialization_s=perf_counter()-started,material_source='current_gs')
        return lines,hits[:,:4],stats

    def close(self):
        self.device_geometry=None;self._curved_clip_edges=None
