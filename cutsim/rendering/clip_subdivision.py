"""Two-edge publication without a clipping-induced GS radius fallback.

All halfplanes are retained until fitting. A child is a real primitive with
its own centre/scale, not a hidden radius cap on the original GS. CUDA chooses
a bounded greedy cover. The standard sample-first path retains the parent
when its capped split cannot pass validation. Publication rejects remaining
clipping-induced radius caps with ClippingSubdivisionError.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip


class ClippingSubdivisionError(RuntimeError):
    pass


CUDA=r'''
#define INF (__longlong_as_double(0x7ff0000000000000LL))
__device__ void line_at(const double* e,int n,const double* extra,int ne,int j,double cx,double cy,double* q){
 const double* a=j<ne?extra+3*j:e+3*(j-ne);
 q[0]=a[0];q[1]=a[1];q[2]=a[2]+a[0]*cx+a[1]*cy;
}
// Exact active-edge reduction of a circle intersected with halfplanes.
// Draw and material certificates must EACH be valid in isolation.
__device__ int reduce(const double* e,int n,const double* extra,int ne,
 double cx,double cy,double r,double* result,bool count_extra=false){
 for(int k=0;k<6;k++)result[k]=0.;result[2]=result[5]=INF;
 bool origin=true;int active=0,all=0;
 for(int j=0;j<n+ne;j++){
  double a[3];line_at(e,n,extra,ne,j,cx,cy,a);
  if(!isfinite(a[2]))continue;
  if(a[2]<0.)origin=false;
  double norm=hypot(a[0],a[1]);
  if(norm<1e-12){if(a[2]<0.)return -1;continue;}
  if(a[2]<-r-1e-10)return -1;
  if(a[2]>=r)continue;
  bool duplicate=false;
  for(int k=0;k<n+ne;k++)if(k!=j){
   double b[3];line_at(e,n,extra,ne,k,cx,cy,b);
   if(a[0]*b[0]+a[1]*b[1]>1.-1e-12 &&
      (b[2]<a[2]-1e-12 || (fabs(b[2]-a[2])<=1e-12&&k<j))){duplicate=true;break;}
  }
  if(duplicate)continue;
  double x=-a[2]*a[0],y=-a[2]*a[1],tx=-a[1],ty=a[0];
  double lo=-sqrt(fmax(0.,r*r-a[2]*a[2])),hi=-lo;
  for(int k=0;k<n+ne;k++)if(k!=j){
   double b[3];line_at(e,n,extra,ne,k,cx,cy,b);if(!isfinite(b[2]))continue;
   double slope=b[0]*tx+b[1]*ty,value=b[0]*x+b[1]*y+b[2];
   if(fabs(slope)<1e-12){if(value<-1e-10){hi=-INF;break;}}
   else if(slope>0.)lo=fmax(lo,-value/slope);else hi=fmin(hi,-value/slope);
  }
  if(hi>=lo-1e-10){
   all++;
   if(j>=ne||count_extra){if(active<2)for(int k=0;k<3;k++)result[3*active+k]=a[k];active++;}
  }
 }
 if(!origin&&!all)return -1;
 return active;
}
__device__ bool inside(const double* e,int n,double x,double y){
 for(int j=0;j<n;j++)if(e[3*j]*x+e[3*j+1]*y+e[3*j+2]<-1e-9)return false;
 return true;
}
__device__ bool fit(const double* d,const double* m,int n,double cx,double cy,double r,
 const double* cap,int nc,double* out){
 double a[6],b[6];int nd=reduce(d,n,cap,nc,cx,cy,r,a,true);
 if(nd<0||nd>2)return false;
 // Material queries consume their own certificate, without draw edges.
 int nm=reduce(m,n,cap,nc,cx,cy,r,b,true);if(nm>2)return false;
 out[0]=cx;out[1]=cy;out[2]=r;
 for(int k=0;k<6;k++){out[3+k]=a[k];out[9+k]=b[k];}
 if(nm<0){out[9]=out[10]=0.;out[11]=-INF;}
 return true;
}
extern "C" __global__ void roots(const double* d,const double* m,const double* r,int count,int n,
 unsigned char* flags,double* result){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 double a[6];int nd=reduce(d+3*n*i,n,nullptr,0,0.,0.,r[i],a);
 if(nd<0){flags[i]=0;return;}
 flags[i]=fit(d+3*n*i,m+3*n*i,n,0,0,r[i],nullptr,0,result+15*i)?1:2;
}
extern "C" __global__ void probes(const double* d,const double* m,const double* r,int count,int n,
 const double* samples,int ns,double* xy,unsigned char* valid){
 int i=blockIdx.x*blockDim.x+threadIdx.x;int pair=n*(n-1)/2,extra=2*n+pair,total=ns+2*extra;
 if(i>=count*total)return;int parent=i/total,j=i%total;double R=r[parent],x=0.,y=0.;valid[i]=0;
 const double* dd=d+3*n*parent;const double* mm=m+3*n*parent;
 if(j<ns){x=R*samples[2*j];y=R*samples[2*j+1];}
 else{
  int q=j-ns;const double* e=q<extra?dd:mm;q%=extra;
  if(q<2*n){int k=q/2;const double* a=e+3*k;if(!isfinite(a[2])||hypot(a[0],a[1])<.5||fabs(a[2])>R)return;
   double t=sqrt(fmax(0.,R*R-a[2]*a[2]))*(q%2?1.:-1.);x=-a[2]*a[0]-t*a[1];y=-a[2]*a[1]+t*a[0];}
  else{q-=2*n;int a=0;while(q>=n-a-1){q-=n-a-1;a++;}int b=a+q+1;
   const double* A=e+3*a;const double* B=e+3*b;double det=A[0]*B[1]-A[1]*B[0];
   if(fabs(det)<1e-12||!isfinite(A[2])||!isfinite(B[2]))return;
   x=(A[1]*B[2]-A[2]*B[1])/det;y=(A[2]*B[0]-A[0]*B[2])/det;
  }
 }
 if(x*x+y*y>R*R+1e-9||!inside(dd,n,x,y))return;
 xy[2*i]=x;xy[2*i+1]=y;valid[i]=1;
}
extern "C" __global__ void candidates(const double* d,const double* m,const double* r,int count,int n,
 const double* xy,const int* lengths,int ns,double* output,unsigned char* valid){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*ns*2)return;
 int parent=i/(ns*2),j=(i/2)%ns,kind=i%2;valid[i]=0;if(j>=lengths[parent])return;
 double R=r[parent],px=xy[2*(parent*ns+j)],py=xy[2*(parent*ns+j)+1],length=hypot(px,py);
 const double* dd=d+3*n*parent;const double* mm=m+3*n*parent;double* out=output+15*i;
 if(!kind){
  if(length<1e-10)return;double cap[3]={px/length,py/length,-length};
  if(!fit(dd,mm,n,0,0,R,cap,1,out))return;
  double lo=-R,hi=length;
  for(int k=0;k<10;k++){double h=(lo+hi)*.5,tmp[15];cap[2]=-h;
   if(fit(dd,mm,n,0,0,R,cap,1,tmp)){hi=h;for(int a=0;a<15;a++)out[a]=tmp[a];}else lo=h;}
  valid[i]=1;
 }else{
  double cx=px,cy=py;
  if(length>R*(1.-1e-8)){cx=.75*px;cy=.75*py;}
  double limit=fmax(0.,R-hypot(cx,cy)),rad=limit;
  for(int k=0;k<20;k++){
   if(rad<1e-7)return;
   if(fit(dd,mm,n,cx,cy,rad,nullptr,0,out)){
    double lo=rad,hi=fmin(limit,2.*rad);
    for(int t=0;t<8;t++){double mid=(lo+hi)*.5,tmp[15];
     if(fit(dd,mm,n,cx,cy,mid,nullptr,0,tmp)){lo=mid;for(int a=0;a<15;a++)out[a]=tmp[a];}else hi=mid;}
    double dx=px-cx,dy=py-cy;if(dx*dx+dy*dy<=out[2]*out[2]+1e-9)valid[i]=1;return;
   }
   rad*=.5;
  }
 }
}
__device__ bool covers(const double* c,double x,double y){
 x-=c[0];y-=c[1];if(x*x+y*y>c[2]*c[2]+1e-9)return false;
 return c[3]*x+c[4]*y+c[5]>=-1e-9&&c[6]*x+c[7]*y+c[8]>=-1e-9;
}
extern "C" __global__ void select_cover(const double* xy,const int* lengths,int ns,
 const double* candidates,const unsigned char* valid,int limit,double* result,int* counts,int* missed){
 int parent=blockIdx.x,lane=threadIdx.x;extern __shared__ unsigned char uncovered[];
 __shared__ int scores[128],indices[128],chosen,remaining;
 int len=lengths[parent];for(int j=lane;j<len;j+=128)uncovered[j]=1;
 if(!lane){counts[parent]=0;remaining=0;}__syncthreads();
 for(int step=0;step<limit;step++){
  int score=0,best=-1;
  for(int j=lane;j<2*len;j+=128)if(valid[parent*ns*2+j]){
   const double* c=candidates+15*(parent*ns*2+j);int value=0;
   for(int k=0;k<len;k++)if(uncovered[k]&&covers(c,xy[2*(parent*ns+k)],xy[2*(parent*ns+k)+1]))value++;
   if(value>score){score=value;best=j;}
  }
  scores[lane]=score;indices[lane]=best;__syncthreads();
  for(int stride=64;stride;stride/=2){if(lane<stride&&(scores[lane+stride]>scores[lane]||
   (scores[lane+stride]==scores[lane]&&indices[lane+stride]>=0&&(indices[lane]<0||indices[lane+stride]<indices[lane])))){
    scores[lane]=scores[lane+stride];indices[lane]=indices[lane+stride];}__syncthreads();}
  if(!lane)chosen=scores[0]?indices[0]:-1;__syncthreads();if(chosen<0)break;
  const double* c=candidates+15*(parent*ns*2+chosen);
  if(!lane){for(int j=0;j<15;j++)result[15*(parent*limit+step)+j]=c[j];counts[parent]++;}
  for(int j=lane;j<len;j+=128)if(covers(c,xy[2*(parent*ns+j)],xy[2*(parent*ns+j)+1]))uncovered[j]=0;
  __syncthreads();
 }
 int left=0;for(int j=lane;j<len;j+=128)left+=uncovered[j];if(left)atomicAdd(&remaining,left);
 __syncthreads();if(!lane)missed[parent]=remaining;
}
'''


def _edges(clip,row,radius):
    raw=getattr(clip,'constraints',None)
    values=np.asarray(raw[row] if raw is not None and row in raw else clip.planes[row,:6].reshape(2,3),float).reshape(-1,3)
    norm=np.linalg.norm(values[:,:2],axis=1)
    invalid=(norm<1e-12)&(values[:,2]<0)
    if invalid.any():return np.array([[0.,0.,-1.]])
    keep=(norm>1e-12)&np.isfinite(values[:,2])
    values=values[keep]/norm[keep,None]
    return values[values[:,2]<radius+1e-9]


def partition(snapshot,*,max_children=4096,per_parent=32):
    """Return keep mask, child source rows, and 15-column child certificates."""
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.rendering.gs_area_retirement import _disc
    started=perf_counter();rt=runtime();cp=rt.cp
    n=len(snapshot.endpoints['positions']);radius=CLIP_SIGMA*snapshot.gaussian.attributes['scale'][:,0].astype(float)
    draw=snapshot.clipping.planes.copy();material=snapshot.material_clipping.planes.copy()
    rawd=getattr(snapshot.clipping,'constraints',None) or {};rawm=getattr(snapshot.material_clipping,'constraints',None) or {}
    rows=np.array(sorted(rawd.keys()|rawm.keys()),dtype=np.int64)
    groups={}
    for row in rows:
        d=_edges(snapshot.clipping,row,radius[row]);m=_edges(snapshot.material_clipping,row,radius[row])
        size=max(len(d),len(m),1);bucket=2**int(np.ceil(np.log2(size)))
        if bucket>256:raise ClippingSubdivisionError(f'Too many clipping edges at GS {row}: {size}')
        groups.setdefault(bucket,[]).append((row,d,m))
    keep=np.ones(n,bool);parents=[];children=[];fitted=0;split_count=0
    with rt.context():
        if not hasattr(rt,'clip_partition_kernels'):
            mod=cp.RawModule(code=CUDA,options=('--fmad=false',))
            rt.clip_partition_kernels=[mod.get_function(x) for x in ('roots','probes','candidates','select_cover')]
        root,probe,candidate,select=rt.clip_partition_kernels
        for width,items in groups.items():
            # Bounded work batches; unfinished families are never published.
            for start in range(0,len(items),512):
                batch=items[start:start+512];ids=np.array([x[0] for x in batch]);count=len(ids)
                d=np.zeros((count,width,3));d[:,:,2]=np.inf;m=d.copy()
                for i,(_,dd,mm) in enumerate(batch):d[i,:len(dd)]=dd;m[i,:len(mm)]=mm
                dd=cp.asarray(d);mm=cp.asarray(m);rr=cp.asarray(radius[ids]);flags=cp.empty(count,cp.uint8);out=cp.empty((count,15),cp.float64)
                root(((count+127)//128,),(128,),(dd,mm,rr,np.int32(count),np.int32(width),flags,out))
                flag=cp.asnumpy(flags,stream=rt.stream);cert=cp.asnumpy(out,stream=rt.stream)
                keep[ids[flag==0]]=False
                good=flag==1;fitted+=int(good.sum())
                for target,sl in ((draw,slice(3,9)),(material,slice(9,15))):
                    target[ids[good],:6]=cert[good,sl];target[ids[good],6]=radius[ids[good]]
                split=np.flatnonzero(flag==2)
                if not len(split):continue
                sid=ids[split];dd=dd[split];mm=mm[split];rr=rr[split];count=len(split)
                angles=(np.arange(128)+.5)*np.pi/64
                samples=np.vstack((_disc(16,32),np.column_stack((np.cos(angles),np.sin(angles)))))
                ns=len(samples);total=ns+2*(2*width+width*(width-1)//2)
                xy=cp.empty((count,total,2),cp.float64);valid=cp.zeros((count,total),cp.uint8)
                probe(((count*total+127)//128,),(128,),(dd,mm,rr,np.int32(count),np.int32(width),cp.asarray(samples),np.int32(ns),xy,valid))
                lengths=cp.sum(valid,axis=1,dtype=cp.int32);maxlen=int(cp.asnumpy(lengths.max(),stream=rt.stream))
                if not maxlen:raise ClippingSubdivisionError('Unresolved clipping domain has no witness samples')
                packed=cp.zeros((count,maxlen,2),cp.float64);a,b=cp.nonzero(valid)
                offsets=cp.cumsum(valid,axis=1,dtype=cp.int32)-1;packed[a,offsets[a,b]]=xy[a,b]
                cand=cp.empty((count,2*maxlen,15),cp.float64);cv=cp.empty((count,2*maxlen),cp.uint8)
                candidate(((count*2*maxlen+127)//128,),(128,),
                    (dd,mm,rr,np.int32(count),np.int32(width),packed,lengths,np.int32(maxlen),cand,cv))
                result=cp.empty((count,per_parent,15),cp.float64);counts=cp.empty(count,cp.int32);miss=cp.empty(count,cp.int32)
                select((count,),(128,),(packed,lengths,np.int32(maxlen),cand,cv,np.int32(per_parent),result,counts,miss),shared_mem=maxlen)
                missed=cp.asnumpy(miss,stream=rt.stream);sizes=cp.asnumpy(counts,stream=rt.stream)
                if np.any(missed):raise ClippingSubdivisionError(f'Clipping needs more subdivision: GS {sid[missed>0][:8].tolist()}, uncovered witnesses {missed[missed>0][:8].tolist()}; block not published')
                if len(children)+int(sizes.sum())>max_children:raise ClippingSubdivisionError(f'Clipping subdivision budget {max_children} exceeded; block not published')
                values=cp.asnumpy(result,stream=rt.stream)
                for row,size,family in zip(sid,sizes,values):
                    keep[row]=False;parents.extend([int(row)]*int(size));children.extend(family[:size]);split_count+=1
    # Empty material is encoded explicitly, never as a shrunken positive disc.
    empty=~np.isfinite(material[:,2])&(material[:,2]<0)
    material[empty]=[0,0,-1,0,0,np.inf,0]
    stats=dict(snapshot.clipping.stats,clip_subdivision=True,clip_gpu_prepare=False,clip_partition_checked=len(rows),
        clip_partition_kept=fitted,clip_partition_parents=split_count,clip_partition_children=len(children),
        clip_partition_ms=(perf_counter()-started)*1000,radius_limited_gs=0)
    snapshot.clipping=ProfileClip(draw,stats);snapshot.material_clipping=ProfileClip(material,dict(stats))
    return keep,np.asarray(parents,np.int64),np.asarray(children,float).reshape(-1,15),stats


def publish(snapshot,added,removed,block,*,max_children=4096,gpu_prepare=False,debug=None,subdivide=True,trim_cache=None):
    from cutsim.cutting.tool_profile_simulator import _select
    from cutsim.cutting.tool_profile_state import combine_surfaces,RAY_STRIDE
    region=getattr(snapshot.clipping.constraints,'regions',None)
    untouched_stock=getattr(region,'untouched_stock',np.zeros(len(added),bool))
    preserve_input_clip=untouched_stock.copy()
    if region is not None and hasattr(region,'twoedge_fallback'):
        from cutsim.cuda.profile_dexel_cuda import runtime
        rt=runtime()
        with rt.context():
            rolled_back=rt.cp.asnumpy(region.rows[region.twoedge_fallback.astype(bool)])
        preserve_input_clip[rolled_back]=True
    inherited=[clip.planes.copy() for clip in (snapshot.clipping,snapshot.material_clipping)] if np.any(preserve_input_clip) else None
    if hasattr(snapshot.clipping.constraints,'regions'):
        from cutsim.rendering.sampled_analytic_clip import partition_sampled
        keep,parents,c,stats=partition_sampled(snapshot,max_children=max_children,subdivide=subdivide,trim_cache=trim_cache)
    elif gpu_prepare:
        from cutsim.rendering.clip_subdivision_gpu import partition_gpu
        keep,parents,c,stats=partition_gpu(snapshot,max_children=max_children)
    else:keep,parents,c,stats=partition(snapshot,max_children=max_children)
    if debug is not None:
        debug['split_parents']=parents.copy()
        debug['split_parents'].flags.writeable=False
        if debug.get('track_sources', False):
            debug['published_sources']=np.r_[np.flatnonzero(keep),parents]
            debug['published_sources'].flags.writeable=False
    old_sources=np.flatnonzero(~removed);old_published=np.flatnonzero(~added)
    removed=removed.copy();removed[old_sources[~keep[old_published]]]=True
    result=_select(snapshot,keep);new_added=added[keep]
    if len(parents):
        child=_select(snapshot,parents)
        p=child.endpoints['positions']+c[:,0,None]*child.gaussian.u+c[:,1,None]*child.gaussian.v
        child.endpoints['positions']=p;child.gaussian.attributes['positions']=p.astype(np.float32)
        sigma=(c[:,2]/CLIP_SIGMA).astype(np.float32)
        # Round inward only when fitting NEW child geometry to its domain.
        sigma=np.where(CLIP_SIGMA*sigma.astype(float)>c[:,2],np.nextafter(sigma,np.float32(0)),sigma)
        child.gaussian.attributes['scale']=sigma[:,None];r=CLIP_SIGMA*sigma.astype(float)
        child.endpoints['owners']=np.full(len(parents),block,np.int64)
        child.endpoints['ray_ids']=-(block+1)*RAY_STRIDE-200_000-np.arange(len(parents),dtype=np.int64)
        d=np.column_stack((c[:,3:9],r));m=np.column_stack((c[:,9:15],r))
        empty=~np.isfinite(m[:,2])&(m[:,2]<0);m[empty]=[0,0,-1,0,0,np.inf,0]
        child.clipping=ProfileClip(d,stats);child.material_clipping=ProfileClip(m,stats)
        source=result;result=combine_surfaces(source,np.ones(len(new_added),bool),child)
        result.material_surface_ids=np.r_[source.material_surface_ids,child.material_surface_ids]
        result.clipping=ProfileClip(np.concatenate((source.clipping.planes,d)),stats)
        result.material_clipping=ProfileClip(np.concatenate((source.material_clipping.planes,m)),dict(stats))
        new_added=np.r_[new_added,np.ones(len(parents),bool)]
    stats['clip_partition_retired_old']=int(np.count_nonzero(~keep[old_published]))
    if debug is not None:
        # Explicit publication provenance, independent of synthetic ray IDs.
        debug['split_children']=np.r_[np.zeros(int(keep.sum()),bool),np.ones(len(parents),bool)]
        debug['split_children'].flags.writeable=False
    radius=CLIP_SIGMA*result.gaussian.attributes['scale'][:,0].astype(float)
    for side,clip in enumerate((result.clipping,result.material_clipping)):
        limited=(clip.planes[:,6]>0)&(clip.planes[:,6]<radius-1e-10)
        if inherited is not None:
            # Narrow initial box faces already have a finite radius. A stock
            # GS excluded from this partial-cut pass must retain its entire
            # existing certificate, rather than being expanded or re-fitted.
            unchanged=preserve_input_clip[keep]&np.all(clip.planes[:int(keep.sum())]==inherited[side][keep],axis=1)
            limited[:len(unchanged)] &= ~unchanged
        if np.any(limited):
            raise ClippingSubdivisionError('Published clipping must not reduce the GS support radius; reset and recompute')
    stats['clipped_gs']=int(np.count_nonzero(np.minimum(result.clipping.planes[:,2],result.clipping.planes[:,5])<radius))
    result.clipping.stats.update(clipped_gs=stats['clipped_gs'],radius_limited_gs=0)
    return result,new_added,removed,{k:v for k,v in stats.items() if k.startswith('clip_')}


def section_snapshot(snapshot,selected,y):
    """Render-only section: resolve overflow without changing material state."""
    from cutsim.cutting.tool_profile_simulator import _select
    s=_select(snapshot,selected)
    p=s.endpoints['positions'];u=s.gaussian.u;v=s.gaussian.v
    length=np.hypot(u[:,1],v[:,1]);r=CLIP_SIGMA*s.gaussian.attributes['scale'][:,0]
    d=np.divide(y-p[:,1],length,out=np.full(len(p),np.inf),where=length>1e-10)
    rows=np.flatnonzero(d<r)
    if not len(rows):return s
    raw={}
    for i in rows:
        raw[int(i)]=np.vstack((s.clipping.planes[i,:6].reshape(2,3),[-u[i,1]/length[i],-v[i,1]/length[i],d[i]]))
    s.clipping=ProfileClip(s.clipping.planes,s.clipping.stats,raw)
    # The rendering section does not change the material certificate.
    result,_,_,_=publish(s,np.zeros(len(p),bool),np.zeros(len(p),bool),0)
    return result
