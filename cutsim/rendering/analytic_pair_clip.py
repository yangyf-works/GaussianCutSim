"""CUDA local intersections of TWO finite generating analytic surfaces.

Solve both implicit equations, then project the SAME intersection tangent to
each GS. This is a bounded local numerical solve, not a closed-form global
quadric intersection. Failed/tangent roots never become extrapolated edges.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.gs_pair_clip import CUDA as PAIR_CUDA, pair_edge_groups, _append_pair_edges
from cutsim.rendering.clip_edge_batches import EdgeBatches
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip
from cutsim.rendering.analytic_surfaces import pack_snapshot


CUDA=PAIR_CUDA[:PAIR_CUDA.index('extern "C" __global__ void pair_query')]+r'''
__device__ double equation(const double* e,const double* p,double* grad){
 double d[3]={p[0]-e[0],p[1]-e[1],p[2]-e[2]},f=e[15];
 for(int a=0;a<3;a++){
  double q=0.;for(int b=0;b<3;b++)q+=e[3+3*a+b]*d[b];
  grad[a]=2*q+e[12+a];f+=d[a]*(q+e[12+a]);
 }
 return f;
}
__device__ bool root(const double* a,const double* b,double* p,double* ga,double* gb){
 for(int iter=0;iter<16;iter++){
  double fa=equation(a,p,ga),fb=equation(b,p,gb);
  double aa=pdot(ga,ga),bb=pdot(gb,gb),ab=pdot(ga,gb),det=aa*bb-ab*ab;
  if(aa<1e-20||bb<1e-20||det<1e-10*aa*bb)return false;
  if(fabs(fa)/sqrt(aa)<1e-8&&fabs(fb)/sqrt(bb)<1e-8)return true;
  double la=(bb*fa-ab*fb)/det,lb=(aa*fb-ab*fa)/det;
  for(int k=0;k<3;k++)p[k]-=la*ga[k]+lb*gb[k];
 }
 return false;
}
__device__ bool primitive_inside(int i,const double* p,const double* rec,const long long* spans,
 const double* planes,double tolerance){
 for(long long j=spans[2*i];j<spans[2*i+1];j++){
  const double* h=planes+5*j;double len=sqrt(pdot(h,h));
  if(pdot(h,p)+h[3]<-tolerance*len)return false;
 }
 const double* r=rec+14*i;
 if(r[13]){
  double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
  for(int a=0;a<3;a++)for(int b=0;b<3;b++)q+=d[a]*r[3+3*a+b]*d[b];
  if(q-r[12]*r[12]>tolerance*fmax(1.,2*r[12]))return false;
 }
 return true;
}
__device__ bool domain(const long long* m,const double* p,const double* rec,const long long* spans,const double* planes){
 if(!primitive_inside(m[2],p,rec,spans,planes,2e-6))return false;
 // A primitive's internal partition face is NOT an exposed sweep surface.
 if(!m[3])for(int i=m[0];i<m[0]+m[1];i++)
  if(i!=m[2]&&primitive_inside(i,p,rec,spans,planes,-2e-6))return false;
 return true;
}
__device__ bool admitted(const double* g,const double* q){
 double d[3]={q[0]-g[0],q[1]-g[1],q[2]-g[2]};
 if(pdot(d,d)>g[12]*g[12]+1e-7)return false;
 double x=pdot(d,g+3),y=pdot(d,g+6);
 for(int k=13;k<25;k+=3)if(g[k]*x+g[k+1]*y+g[k+2]<-2e-6)return false;
 return true;
}
__device__ int analytic_edge(const double* a,const double* b,long long ida,long long idb,
 const double* ea,const double* eb,const long long* ma,const long long* mb,
 const double* rec,const long long* spans,const double* planes,double* edges,double* point){
 if(!(a[12]>0.)||!(b[12]>0.))return 0;
 double delta[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
 if(pdot(delta,delta)>(a[12]+b[12])*(a[12]+b[12]))return 0;
 bool same_family=ida>=7&&idb>=7&&(ida-7)/3==(idb-7)/3;
 double ga[3],gb[3],q[3];bool ok=false;
 for(int seed=0;seed<3;seed++){
  for(int k=0;k<3;k++)q[k]=seed==0?.5*(a[k]+b[k]):(seed==1?a[k]:b[k]);
  if(!root(ea,eb,q,ga,gb))continue;
  if(!domain(ma,q,rec,spans,planes)||!domain(mb,q,rec,spans,planes))continue;
  if(!admitted(a,q)||!admitted(b,q))continue;
  ok=true;break;
 }
 if(!ok)return 0;
 double line[3]={ga[1]*gb[2]-ga[2]*gb[1],ga[2]*gb[0]-ga[0]*gb[2],ga[0]*gb[1]-ga[1]*gb[0]};
 double norm=sqrt(pdot(line,line));
 if(norm*norm<0.00759612349389597*pdot(ga,ga)*pdot(gb,gb))return 0;
 for(int k=0;k<3;k++){line[k]/=norm;point[k]=q[k];}
 for(int side=0;side<2;side++){
  const double* g=side?b:a;const double* peer=side?ga:gb;const long long* pm=side?ma:mb;
  double x=-pdot(line,g+6),y=pdot(line,g+3),len=hypot(x,y);
  if(len<1e-10)return 0;x/=len;y/=len;
  // Stock retains its interior. Different cuts retain the exterior of each
  // other. Floor/wall of a continuous sweep family retain the finite domain.
  double sign=pm[3]||same_family?-1.:1.;
  if(sign*(x*pdot(peer,g+3)+y*pdot(peer,g+6))<0.){x=-x;y=-y;}
  double d[3]={q[0]-g[0],q[1]-g[1],q[2]-g[2]};
  edges[3*side]=x;edges[3*side+1]=y;edges[3*side+2]=-x*pdot(d,g+3)-y*pdot(d,g+6);
 }
 return same_family?2:1;
}
extern "C" __global__ void analytic_query(const double* g,const long long* ids,int n,int old,
 const double* boxes,const long long* order,int leaves,const long long* offsets,int* counts,
 long long* pairs,double* edges,unsigned char* kinds,double* roots,int write,
 const long long* patch_ids,const double* equations,const long long* meta,const double* rec,
 const long long* spans,const double* planes){
 int i=old+blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const double* a=g+25*i;int stack[64],top=0,found=0;stack[top++]=1;
 long long ai=patch_ids[i];
 while(top){int node=stack[--top];if(!pair_overlap(a,boxes+6*node))continue;
  if(node<leaves){stack[top++]=node*2+1;stack[top++]=node*2;continue;}
  int leaf=node-leaves;if(leaf>=n)continue;long long j=order[leaf];
  if(j>=i||ids[i]==ids[j])continue;long long bi=patch_ids[j];double e[6],q[3];
  int kind=analytic_edge(a,g+25*j,ids[i],ids[j],equations+16*ai,equations+16*bi,
   meta+4*ai,meta+4*bi,rec,spans,planes,e,q);if(!kind)continue;
  if(write){long long at=offsets[i-old]+found;pairs[2*at]=i;pairs[2*at+1]=j;kinds[at]=kind;
   for(int k=0;k<6;k++)edges[6*at+k]=e[k];for(int k=0;k<3;k++)roots[3*at+k]=q[k];}
  found++;
 }
 if(!write)counts[i-old]=found;
}
'''


def intersect_analytic(geometry,identity,nold,packed,evaluation_radii,*,max_pairs=2_000_000):
    """GPU BVH + bounded dual-equation Newton solve + per-owner deduplication."""
    from cutsim.cuda.profile_dexel_cuda import runtime
    n=len(geometry)
    if n<=nold:return np.empty((0,2),np.int64),np.empty((0,2,3)),np.empty(0,np.uint8),np.empty((0,3)),0.
    rt=runtime();cp=rt.cp
    with rt.context():
        if not hasattr(rt,'analytic_clip_kernels'):
            module=cp.RawModule(code=CUDA,options=('--fmad=false',))
            rt.analytic_clip_kernels=[module.get_function(s) for s in ('pair_codes','pair_leaves','pair_parents','analytic_query')]
        codes_kernel,leaves_kernel,parents_kernel,query=rt.analytic_clip_kernels
        g=cp.asarray(geometry);ids=cp.asarray(identity,dtype=cp.int64);data=tuple(cp.asarray(a) for a in packed)
        leaves=1<<(n-1).bit_length();codes=cp.empty(n,cp.uint32);boxes=cp.empty((2*leaves,6),cp.float64)
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        codes_kernel(((n+127)//128,),(128,),(g,np.int32(n),cp.min(g[:,:3],axis=0),cp.max(g[:,:3],axis=0),codes))
        order=cp.argsort(codes).astype(cp.int64)
        leaves_kernel(((leaves+127)//128,),(128,),(g,order,np.int32(n),np.int32(leaves),boxes))
        level=leaves//2
        while level:parents_kernel(((level+127)//128,),(128,),(boxes,np.int32(level)));level//=2
        counts=cp.empty(n-nold,cp.int32);di=cp.empty(1,cp.int64);dd=cp.empty(1,cp.float64);db=cp.empty(1,cp.uint8)
        args=(g,ids,np.int32(n),np.int32(nold),boxes,order,np.int32(leaves))
        query(((n-nold+127)//128,),(128,),(*args,di,counts,di,dd,db,dd,np.int32(0),*data))
        offsets=cp.cumsum(counts,dtype=cp.int64);total=int(offsets[-1].item());offsets-=counts
        if total>max_pairs:raise RuntimeError(f'Analytic clipping exceeds {max_pairs} pairs; block not published')
        pairs=cp.empty((total,2),cp.int64);edges=cp.empty((total,2,3),cp.float64);kinds=cp.empty(total,cp.uint8);roots=cp.empty((total,3),cp.float64)
        query(((n-nold+127)//128,),(128,),(*args,offsets,counts,pairs,edges,kinds,roots,np.int32(1),*data))
        active=edges[:,:,2]<=cp.asarray(evaluation_radii)[pairs]
        # One nearest local tangent per receiver + analytic peer patch. Many
        # GS sampling the same face must not cause duplicate subdivisions.
        receiver=pairs.ravel();peer=data[0][pairs[:,::-1]].ravel()
        distance=cp.sum((cp.repeat(roots,2,axis=0)-g[receiver,:3])**2,axis=1)
        distance=cp.where(active.ravel(),distance,cp.inf)
        order=cp.lexsort(cp.stack((distance,peer,receiver)));rr=receiver[order];pp=peer[order]
        first=cp.ones(len(order),cp.bool_)
        if len(order)>1:first[1:]=(rr[1:]!=rr[:-1])|(pp[1:]!=pp[:-1])
        selected=cp.zeros(total*2,cp.bool_);selected[order[first]]=True
        active &= selected.reshape(-1,2)
        edges[~active]=cp.asarray([0.,0.,cp.inf]);keep=cp.any(active,axis=1)
        pairs,edges,kinds,roots=pairs[keep],edges[keep],kinds[keep],roots[keep]
        end.record();end.synchronize();ms=float(cp.cuda.get_elapsed_time(begin,end))
        result=[cp.asnumpy(a,stream=rt.stream) for a in (pairs,edges,kinds,roots)]
    order=np.lexsort((result[0][:,1],result[0][:,0]))
    return *(a[order] for a in result),ms


def update_analytic_clipping(snapshot,previous,kept,*,collect_edges=True,gpu_prepare=True):
    if not collect_edges:raise ValueError('Analytic-pair clipping requires 2edge publication')
    started=perf_counter();p=snapshot.endpoints['positions'];n=len(p);nold=int(np.count_nonzero(kept))
    draw=np.tile([0.,0.,np.inf,0.,0.,np.inf,np.inf],(n,1));material=draw.copy()
    draw[:nold]=previous.clipping.planes[kept]
    material[:nold]=(getattr(previous,'material_clipping',None) or previous.clipping).planes[kept]
    u,v=snapshot.gaussian.u,snapshot.gaussian.v;sigma=snapshot.gaussian.attributes['scale'][:,0].astype(float)
    radius=CLIP_SIGMA*sigma;search=np.where((draw[:,6]>0)&(material[:,6]>0),2*sigma,0.)
    g=np.ascontiguousarray(np.column_stack((p,u,v,np.cross(u,v),search,draw[:,:6],material[:,:6])))
    packed=pack_snapshot(snapshot)
    pairs,edges,kinds,roots,ms=intersect_analytic(g,snapshot.material_surface_ids,nold,packed,radius)
    edge_rows,starts,counts,values,_=pair_edge_groups(pairs,edges)
    rows=np.union1d(edge_rows,np.arange(nold,n));rawd=EdgeBatches();rawm=EdgeBatches()
    rawd.append(rows,draw[rows,:6].reshape(-1,2,3));rawm.append(rows,material[rows,:6].reshape(-1,2,3))
    for depth in range(int(counts.max(initial=0))):
        take=counts>depth;row=edge_rows[take];e=values[starts[take]+depth,None,:]
        _append_pair_edges(rawd,draw,row,e);_append_pair_edges(rawm,material,row,e)
    if not gpu_prepare:
        def as_dict(raw):
            result={}
            for rr,ee in raw.blocks:
                for row,e in zip(rr,ee):result.setdefault(int(row),[]).extend(e)
            return result
        rawd,rawm=as_dict(rawd),as_dict(rawm)
    stats=dict(backend='cuda',mode='analytic_pairs',clipping_source='analytic_pairs',material_source='current_gs',
        total_ms=(perf_counter()-started)*1000,kernel_ms=ms,recomputed_gs=len(rows),reused_gs=n-len(rows),
        boundary_search_sigma=2.,edge_evaluation_sigma=CLIP_SIGMA,analytic_patch_count=len(snapshot.analytic_patches),
        gs_pair_count=len(pairs),gs_pair_ridges=int(np.count_nonzero(kinds==1)),gs_pair_valleys=int(np.count_nonzero(kinds==2)),
        clipped_gs=int(np.count_nonzero(np.minimum(draw[:,2],draw[:,5])<radius)),radius_limited_gs=0,
        rejected_extrapolated_boundaries=0,accepted_finite_boundaries=len(pairs),boundary_segment_queries=0,reselected_boundaries=0,
        analytic_sweep_clip_calls=0,boundary_probes=0,accepted_boundary_planes=len(values),clip_edge_method='analytic-pairs')
    draw.flags.writeable=False;material.flags.writeable=False
    return ProfileClip(draw,stats,rawd),ProfileClip(material,dict(stats),rawm)
