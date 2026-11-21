"""Incremental, bidirectional finite-GS clipping, without cutter geometry.

Each unordered pair involving a new GS is visited once. Its common finite
intersection segment within 2 sigma must survive BOTH prior clip planes.
Each resulting edge is admitted independently within its owner's 2 sigma.
A physical ridge
or valley contributes one halfplane to each GS; same-surface, parallel and
mixed-sign (ambiguous) pairs contribute none. In the standard hybrid mode,
retained GS instead receive current-sweep analytic edges; new GS keep pairs.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.clip_edge_batches import EdgeBatches
from cutsim.rendering.profile_cut_clip import ProfileClip

PAIR_SEARCH_SIGMA = 2.0


CUDA=r'''
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
__device__ double pdot(const double* a,const double* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ unsigned int pbits(unsigned int v){
 v=(v*0x00010001u)&0xFF0000FFu;v=(v*0x00000101u)&0x0F00F00Fu;
 v=(v*0x00000011u)&0xC30C30C3u;v=(v*0x00000005u)&0x49249249u;return v;
}
extern "C" __global__ void pair_codes(const double* g,int n,const double* lo,const double* hi,unsigned int* codes){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int q[3];
 for(int k=0;k<3;k++)q[k]=(unsigned int)fmin(1023.,fmax(0.,1023.*(g[25*i+k]-lo[k])/fmax(hi[k]-lo[k],1e-30)));
 codes[i]=(pbits(q[0])<<2)|(pbits(q[1])<<1)|pbits(q[2]);
}
extern "C" __global__ void pair_leaves(const double* g,const long long* order,int n,int leaves,double* boxes){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=leaves)return;
 for(int k=0;k<3;k++){
  boxes[6*(leaves+i)+k]=i<n?g[25*order[i]+k]-g[25*order[i]+12]:INFINITY;
  boxes[6*(leaves+i)+k+3]=i<n?g[25*order[i]+k]+g[25*order[i]+12]:-INFINITY;
 }
}
extern "C" __global__ void pair_parents(double* b,int start){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=start)return;int node=start+i;
 for(int k=0;k<3;k++){b[6*node+k]=fmin(b[12*node+k],b[12*node+6+k]);b[6*node+k+3]=fmax(b[12*node+k+3],b[12*node+6+k+3]);}
}
__device__ bool pair_overlap(const double* a,const double* b){
 if(b[0]>b[3])return false;
 for(int k=0;k<3;k++)if(a[k]+a[12]<b[k]-1e-9||a[k]-a[12]>b[k+3]+1e-9)return false;
 return true;
}
__device__ bool pair_trim(const double* g,const double* foot,const double* line,double* lo,double* hi){
 double delta[3]={foot[0]-g[0],foot[1]-g[1],foot[2]-g[2]};
 double x=pdot(delta,g+3),y=pdot(delta,g+6),dx=pdot(line,g+3),dy=pdot(line,g+6);
 for(int k=13;k<25;k+=3){
  double value=g[k]*x+g[k+1]*y+g[k+2],slope=g[k]*dx+g[k+1]*dy;
  if(fabs(slope)<1e-12){if(value<-1e-9)return false;}
  else if(slope>0.)*lo=fmax(*lo,(-value-1e-9)/slope);
  else *hi=fmin(*hi,(-value-1e-9)/slope);
 }
 return *hi-*lo>1e-8;
}
// Return 1=ridge, 2=valley, 0=no finite edge. Output normals point to kept side.
__device__ int pair_edge(const double* a,const double* b,double* edges){
 if(!(a[12]>0.)||!(b[12]>0.))return 0;
 double delta[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
 if(pdot(delta,delta)>(a[12]+b[12])*(a[12]+b[12])+1e-12)return 0;
 double c=fmax(-1.,fmin(1.,pdot(a+9,b+9))),s2=1.-c*c;
 if(s2<0.00759612349389597)return 0; // sin(5 degrees)^2
 double s=sqrt(s2),da=pdot(delta,a+9),db=-pdot(delta,b+9),sign;
 if(da>1e-10&&db>1e-10)sign=1.;
 else if(da<-1e-10&&db<-1e-10)sign=-1.;
 else return 0;
 double ca=db/s,cb=da/s;
 if(fabs(ca)>=a[12]||fabs(cb)>=b[12])return 0;
 double line[3]={(a[10]*b[11]-a[11]*b[10])/s,(a[11]*b[9]-a[9]*b[11])/s,(a[9]*b[10]-a[10]*b[9])/s};
 double foot[3];for(int k=0;k<3;k++)foot[k]=a[k]-db/s2*(b[9+k]-c*a[9+k]);
 double along=pdot(delta,line),ha=sqrt(fmax(0.,a[12]*a[12]-ca*ca)),hb=sqrt(fmax(0.,b[12]*b[12]-cb*cb));
 double lo=fmax(-ha,along-hb),hi=fmin(ha,along+hb);
 if(hi-lo<=1e-8||!pair_trim(a,foot,line,&lo,&hi)||!pair_trim(b,foot,line,&lo,&hi))return 0;
 edges[0]=sign*pdot(b+9,a+3)/s;edges[1]=sign*pdot(b+9,a+6)/s;edges[2]=sign*ca;
 edges[3]=sign*pdot(a+9,b+3)/s;edges[4]=sign*pdot(a+9,b+6)/s;edges[5]=sign*cb;
 return sign>0.?2:1;
}
extern "C" __global__ void pair_query(const double* g,const long long* ids,int n,int old,
 const double* boxes,const long long* order,int leaves,const long long* offsets,int* counts,
 long long* pairs,double* edges,unsigned char* kinds,int write){
 int i=old+blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const double* a=g+25*i;int stack[64],top=0,found=0;stack[top++]=1;
 while(top){int node=stack[--top];if(!pair_overlap(a,boxes+6*node))continue;
  if(node<leaves){stack[top++]=node*2+1;stack[top++]=node*2;continue;}
  int leaf=node-leaves;if(leaf>=n)continue;long long j=order[leaf];
  if(j>=i||ids[i]==ids[j])continue;
  double e[6];int kind=pair_edge(a,g+25*j,e);if(!kind)continue;
  if(write){long long at=offsets[i-old]+found;pairs[2*at]=i;pairs[2*at+1]=j;kinds[at]=kind;for(int k=0;k<6;k++)edges[6*at+k]=e[k];}
  found++;
 }
 if(!write)counts[i-old]=found;
}
'''


def _edge_cpu(a,b,*,return_segment=False):
    """Independent Float64 reference for the finite line segment predicate."""
    if min(a[12],b[12])<=0:return None
    delta=b[:3]-a[:3];c=np.clip(a[9:12]@b[9:12],-1.,1.);s2=1-c*c
    if delta@delta>(a[12]+b[12])**2+1e-12 or s2<np.sin(np.deg2rad(5.))**2:return None
    s=np.sqrt(s2);da=delta@a[9:12];db=-delta@b[9:12]
    if da>1e-10 and db>1e-10:sign=1.
    elif da<-1e-10 and db<-1e-10:sign=-1.
    else:return None
    ca,cb=db/s,da/s
    if abs(ca)>=a[12] or abs(cb)>=b[12]:return None
    line=np.cross(a[9:12],b[9:12])/s;foot=a[:3]-db/s2*(b[9:12]-c*a[9:12])
    along=delta@line;ha=np.sqrt(a[12]**2-ca**2);hb=np.sqrt(b[12]**2-cb**2)
    lo,hi=max(-ha,along-hb),min(ha,along+hb)
    for g in (a,b):
        xy=np.array([(foot-g[:3])@g[3:6],(foot-g[:3])@g[6:9]])
        direction=np.array([line@g[3:6],line@g[6:9]])
        for x,y,d in g[13:].reshape(4,3):
            value=np.array([x,y])@xy+d;slope=np.array([x,y])@direction
            if abs(slope)<1e-12:
                if value<-1e-9:return None
            elif slope>0:lo=max(lo,(-value-1e-9)/slope)
            else:hi=min(hi,(-value-1e-9)/slope)
        if hi-lo<=1e-8:return None
    e=sign*np.array([[b[9:12]@a[3:6]/s,b[9:12]@a[6:9]/s,ca],
                     [a[9:12]@b[3:6]/s,a[9:12]@b[6:9]/s,cb]])
    kind=2 if sign>0 else 1
    if return_segment:return e,kind,(foot,line,lo,hi)
    return e,kind


def intersect_pairs(geometry,identity,nold,backend,*,max_pairs=2_000_000,evaluation_radii=None):
    """Count/scan allocation on CUDA; never truncate an overflowing edge set."""
    n=len(geometry)
    if n<=nold:return np.empty((0,2),np.int64),np.empty((0,2,3)),np.empty(0,np.uint8),0.
    if backend!='cuda':
        from scipy.spatial import cKDTree
        tree=cKDTree(geometry[:,:3]);largest=geometry[:,12].max(initial=0)
        pairs=[];edges=[];kinds=[]
        for i in range(nold,n):
            for j in sorted(tree.query_ball_point(geometry[i,:3],geometry[i,12]+largest)):
                if j>=i or identity[i]==identity[j]:continue
                hit=_edge_cpu(geometry[i],geometry[j])
                if hit is not None:
                    pairs.append((i,j));edges.append(hit[0]);kinds.append(hit[1])
                    if len(pairs)>max_pairs:raise RuntimeError('GS pair clipping exceeds pair budget; block not published')
        pairs=np.array(pairs,np.int64).reshape(-1,2);edges=np.array(edges).reshape(-1,2,3);kinds=np.array(kinds,np.uint8)
        if evaluation_radii is not None:
            active=np.abs(edges[:,:,2])<=np.asarray(evaluation_radii)[pairs]
            edges[~active]=[0.,0.,np.inf];keep=active.any(axis=1)
            pairs,edges,kinds=pairs[keep],edges[keep],kinds[keep]
        return pairs,edges,kinds,0.
    from cutsim.cuda.profile_dexel_cuda import runtime
    rt=runtime();cp=rt.cp
    with rt.context():
        if not hasattr(rt,'gs_pair_clip_kernels'):
            module=cp.RawModule(code=CUDA,options=('--fmad=false',))
            rt.gs_pair_clip_kernels=[module.get_function('pair_'+s) for s in ('codes','leaves','parents','query')]
        codes_kernel,leaves_kernel,parents_kernel,query=rt.gs_pair_clip_kernels
        g=cp.asarray(geometry);ids=cp.asarray(identity,dtype=cp.int64);leaves=1<<(n-1).bit_length()
        codes=cp.empty(n,cp.uint32);boxes=cp.empty((2*leaves,6),cp.float64)
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        codes_kernel(((n+127)//128,),(128,),(g,np.int32(n),cp.min(g[:,:3],axis=0),cp.max(g[:,:3],axis=0),codes))
        order=cp.argsort(codes).astype(cp.int64)
        leaves_kernel(((leaves+127)//128,),(128,),(g,order,np.int32(n),np.int32(leaves),boxes))
        level=leaves//2
        while level:
            parents_kernel(((level+127)//128,),(128,),(boxes,np.int32(level)));level//=2
        counts=cp.empty(n-nold,cp.int32);dummy=cp.empty(1,cp.int64);de=cp.empty(1,cp.float64);dk=cp.empty(1,cp.uint8)
        args=(g,ids,np.int32(n),np.int32(nold),boxes,order,np.int32(leaves))
        query(((n-nold+127)//128,),(128,),(*args,dummy,counts,dummy,de,dk,np.int32(0)))
        offsets=cp.cumsum(counts,dtype=cp.int64);total=int(offsets[-1].item());offsets-=counts
        if total>max_pairs:raise RuntimeError(f'GS pair clipping exceeds {max_pairs} pairs; block not published')
        pairs=cp.empty((total,2),cp.int64);edges=cp.empty((total,2,3),cp.float64);kinds=cp.empty(total,cp.uint8)
        query(((n-nold+127)//128,),(128,),(*args,offsets,counts,pairs,edges,kinds,np.int32(1)))
        if evaluation_radii is not None:
            active=cp.abs(edges[:,:,2])<=cp.asarray(evaluation_radii)[pairs]
            edges[~active]=cp.asarray([0.,0.,cp.inf]);keep=cp.any(active,axis=1)
            pairs,edges,kinds=pairs[keep],edges[keep],kinds[keep]
        end.record();end.synchronize();ms=float(cp.cuda.get_elapsed_time(begin,end))
        pairs=cp.asnumpy(pairs,stream=rt.stream);edges=cp.asnumpy(edges,stream=rt.stream);kinds=cp.asnumpy(kinds,stream=rt.stream)
    # Deterministic edge order for CPU/GPU fitting and replay.
    order=np.lexsort((pairs[:,1],pairs[:,0]))
    return pairs[order],edges[order],kinds[order],ms


def pair_edge_groups(pairs, edges):
    """Group the admitted sides of each finite pair by receiving GS."""
    active = np.isfinite(edges[:, :, 2])
    owners = pairs[active]
    order = np.argsort(owners, kind='stable')
    rows, starts, counts = np.unique(owners[order], return_index=True, return_counts=True)
    return rows, starts, counts, edges[active][order], np.nonzero(active)[0][order]


def pair_importance(geometry, radii, pairs, *, return_segments=False):
    """Peak product of the two GS densities along their finite shared segment."""
    weights = np.full(len(pairs), np.nan)
    endpoints = np.full((len(pairs), 2, 3), np.nan) if return_segments else None
    for i, (a, b) in enumerate(pairs):
        segment = _edge_cpu(geometry[a], geometry[b], return_segment=True)
        if segment is None:
            continue
        foot, tangent, lo, hi = segment[2]
        centers = geometry[[a, b], :3]
        inverse_variance = (CLIP_SIGMA / radii[[a, b]]) ** 2
        projected = (centers - foot) @ tangent
        position = np.clip(np.dot(projected, inverse_variance) / inverse_variance.sum(), lo, hi)
        delta = foot + position * tangent - centers
        weights[i] = np.exp(-.5 * np.dot(np.sum(delta**2, axis=1), inverse_variance))
        if endpoints is not None:
            endpoints[i] = foot + np.array([[lo], [hi]]) * tangent
    return (weights, endpoints) if return_segments else weights


def pair_seams(geometry, receivers, segments):
    """Express a finite world-space seam in each receiving GS's tangent axes."""
    delta = segments - geometry[receivers, None, :3]
    basis = geometry[receivers, 3:9].reshape(-1, 2, 3)
    return delta @ basis.transpose(0, 2, 1)


def _append_pair_edges(raw, preview, rows, planes):
    if not len(rows):
        return
    raw.append(rows, planes)
    # Keep the two-edge preview for pre-publication consumers. The full list
    # still drives the exact clipping/subdivision publication step.
    local = preview[rows].copy()
    for edge in planes.transpose(1, 0, 2):
        carry = edge.copy()
        for k in (0, 3):
            swap = carry[:, 2] < local[:, k + 2]
            old = local[swap, k:k + 3].copy()
            local[swap, k:k + 3] = carry[swap]
            carry[swap] = old
    preview[rows] = local


def _project_pair_edges(projection, draw, material, radius, rows, starts, counts, values, importance, seams):
    fitted = np.zeros(len(rows), bool)
    planes = np.tile([0., 0., np.inf], (2, len(rows), 2, 1))
    one = two = 0
    seam_rows = seam_limited = 0
    target_weight = missed_weight = 0.
    for i, (row, start, count) in enumerate(zip(rows, starts, counts)):
        part = slice(start, start + count)
        ref = values[part]
        weights = importance[part]
        if not np.isfinite(weights).all():
            continue
        draw_incoming, material_incoming = draw[row, :6], material[row, :6]
        d = projection.fit_world(draw_incoming, ref, weights, radius[row], seams=seams[part])
        seam_rows += bool(d.seam_points)
        seam_limited += bool(d.seam_limited_angles)
        if d.chosen_edges is None:
            continue
        m = d if np.array_equal(draw_incoming, material_incoming) else projection.fit_world(
            material_incoming, ref, weights, radius[row], seams=seams[part])
        if m.chosen_edges is None:
            continue
        fitted[i] = True
        for target, result in zip(planes, (d, m)):
            n = len(result.chosen_edges)
            target[i, :n] = result.chosen_edges
            target[i, :n, 2] *= radius[row]
        if max(len(d.chosen_edges), len(m.chosen_edges)) == 1:
            one += 1
        else:
            two += 1
        area = radius[row] ** 2
        target_weight += area * d.target_weight
        missed_weight += area * d.chosen_error.missed_weight
    return fitted, planes, dict(clip_projection_checked=len(rows), clip_projection_one=one,
        clip_projection_two=two, clip_projection_fallback=len(rows)-one-two,
        clip_projection_seam_rows=seam_rows, clip_projection_seam_limited_rows=seam_limited,
        clip_projection_weighted_miss_percent=100 * missed_weight / target_weight if target_weight else None)


def update_pair_clipping(snapshot,previous,kept,backend,*,collect_edges=True,gpu_prepare=True,projection=None,
                         retained_sweep=None,stock_bounds=None,current_surface_id=None,sweep_workspace=None):
    started=perf_counter();p=snapshot.endpoints['positions'];n=len(p);nold=int(np.count_nonzero(kept))
    draw=np.tile([0.,0.,np.inf,0.,0.,np.inf,np.inf],(n,1));material=draw.copy()
    draw[:nold]=previous.clipping.planes[kept]
    material[:nold]=(getattr(previous,'material_clipping',None) or previous.clipping).planes[kept]
    u=snapshot.gaussian.u;v=snapshot.gaussian.v;nrm=np.cross(u,v)
    sigma=snapshot.gaussian.attributes['scale'][:,0].astype(float)
    radius=CLIP_SIGMA*sigma
    # Published clip radius is a 2-sigma evaluation certificate, not the
    # neighbour-search radius. Respect empty patches and prior halfplanes,
    # without restricting the search to an older, narrower certificate.
    search_radius=np.where((draw[:,6]>0)&(material[:,6]>0),PAIR_SEARCH_SIGMA*sigma,0.)
    g=np.ascontiguousarray(np.column_stack((p,u,v,nrm,search_radius,draw[:,:6],material[:,:6])),dtype=float)
    identity=getattr(snapshot,'material_surface_ids',None)
    if identity is None:identity=snapshot.gaussian.attributes.get('surface_id',np.arange(n)[:,None]).reshape(-1)
    pairs,edges,kinds,kernel_ms=intersect_pairs(g,identity,nold,backend,evaluation_radii=radius)
    sweep_rows=np.empty(0,np.int64);sweep_clip=None;sweep_stats={}
    if retained_sweep is not None:
        from cutsim.rendering.retained_sweep_clip import retained_sweep_edges
        sweep_rows,sweep_clip,sweep_stats=retained_sweep_edges(previous,kept,retained_sweep,backend,
            stock_bounds=stock_bounds,current_surface_id=current_surface_id,workspace=sweep_workspace)
        # Old GS receive only the current analytic sweep, never the sampled
        # peer edge as well. New GS keep the existing finite-pair path.
        edges=edges.copy();edges[pairs<nold]=[0.,0.,np.inf]
    rawd=EdgeBatches();rawm=EdgeBatches()
    edge_rows, starts, counts, values, pair_ids = pair_edge_groups(pairs, edges)
    rows=np.union1d(np.union1d(edge_rows,sweep_rows),np.arange(nold,n))
    rawd.append(rows,draw[rows,:6].reshape(-1,2,3));rawm.append(rows,material[rows,:6].reshape(-1,2,3))
    if sweep_clip is not None:
        # Keep every edge for final 2edge publication; do NOT propagate the
        # preview's third-edge radius reduction into the GS geometry.
        _append_pair_edges(rawd,draw,sweep_rows,sweep_clip.constraints)
        _append_pair_edges(rawm,material,sweep_rows,sweep_clip.constraints)
    fitted = np.zeros(len(edge_rows), bool)
    projection_stats = {}
    if projection is not None:
        start = perf_counter()
        importance, segments = pair_importance(g, radius, pairs, return_segments=True)
        seams = pair_seams(g, np.repeat(edge_rows, counts), segments[pair_ids])
        fitted, planes, projection_stats = _project_pair_edges(projection, draw, material, radius,
            edge_rows, starts, counts, values, importance[pair_ids], seams)
        _append_pair_edges(rawd, draw, edge_rows[fitted], planes[0, fitted])
        _append_pair_edges(rawm, material, edge_rows[fitted], planes[1, fitted])
        projection_stats['clip_projection_ms'] = (perf_counter() - start) * 1000
    # Each batch has unique row IDs, as required by the GPU edge packer.
    for depth in range(int(counts.max(initial=0))):
        take = (counts > depth) & ~fitted
        row, e = edge_rows[take], values[starts[take]+depth, None, :]
        _append_pair_edges(rawd, draw, row, e)
        _append_pair_edges(rawm, material, row, e)
    if not collect_edges:
        # The production centre-ray path collects all constraints. A caller
        # without publication must not silently publish a lossy 2-edge preview.
        if len(pairs) or len(sweep_rows):raise ValueError('GS clipping requires 2edge publication / collect_edges')
    if not gpu_prepare:
        def as_dict(raw):
            result={}
            for rr,ee in raw.blocks:
                for row,e in zip(rr,ee):result.setdefault(int(row),[]).extend(e)
            return result
        rawd,rawm=as_dict(rawd),as_dict(rawm)
    draw.flags.writeable=False;material.flags.writeable=False
    stats=dict(backend=backend,mode='gs_pairs',material_source='current_gs',clipping_source='finite_gs_pairs',
        update_candidate_sigma=PAIR_SEARCH_SIGMA,boundary_search_sigma=PAIR_SEARCH_SIGMA,edge_evaluation_sigma=CLIP_SIGMA,input_gs=n,
        total_ms=(perf_counter()-started)*1000,kernel_ms=kernel_ms,recomputed_gs=len(rows),reused_gs=n-len(rows),
        gs_pair_count=len(pairs),gs_pair_ridges=int(np.count_nonzero(kinds==1)),gs_pair_valleys=int(np.count_nonzero(kinds==2)),
        analytic_sweep_clip_calls=0,boundary_probes=0,accepted_boundary_planes=len(values),
        rejected_extrapolated_boundaries=0,accepted_finite_boundaries=len(pairs),boundary_segment_queries=0,reselected_boundaries=0,
        continuous_surface_probes_skipped=0,same_surface_old_cutter_skips=0,
        clipped_gs=int(np.count_nonzero(np.minimum(draw[:,2],draw[:,5])<radius)),radius_limited_gs=0,
        clip_edge_method='projection' if projection is not None else 'gs-pairs',**projection_stats)
    if retained_sweep is not None:
        stats.update(sweep_stats,mode='sweep_old_gs_new',clipping_source='sweep_old_gs_new',
            analytic_sweep_clip_calls=int(sweep_clip is not None and not sweep_stats['sweep_clip_cache_reused']),
            kernel_ms=kernel_ms+sweep_stats['sweep_clip_kernel_ms'])
    return ProfileClip(draw,stats,rawd),ProfileClip(material,dict(stats),rawm)
