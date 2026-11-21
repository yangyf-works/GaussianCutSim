"""Conservative whole-support removal using only the current convex cutters.

A GS is bounded by its tangent square at the requested radius (default
4 sigma; standard retained-region retirement explicitly requests 2 sigma).
All four vertices must be strictly inside ONE convex primitive.
Its convex hull then contains the entire disc, including any clipped subset.
For known convex CuttingSweep/FlatSweep types, a second proof encloses the surviving
clipped disk with a circumscribed polygon and checks its vertices against the
whole sweep. Arbitrary unions cannot use this convex-hull certificate.
An additional disk/chord extremum certificate resolves polygon overestimates
and line-only boundary contact. Failure to certify means KEEP.
"""
import numpy as np
from cutsim.core.cutting_sweep import CuttingSweep
from cutsim.core.sweep import FlatSweep
from cutsim.cutting.local_dexel import bounds
from cutsim.rendering.profile_cut_clip import pack_cutters


_KERNEL=r'''
extern "C" __global__ void full_support(const double* geometry,long long count,
 const double* records,const long long* spans,const double* planes,int parts,unsigned char* removed){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* g=geometry+10*i;removed[i]=0;
 for(int part=0;part<parts;part++){
  const double* rec=records+14*part;bool contained=true;
  for(int corner=0;corner<4&&contained;corner++){
   double x[3];for(int k=0;k<3;k++)x[k]=g[k]+g[9]*((corner&1?1.:-1.)*g[3+k]+(corner&2?1.:-1.)*g[6+k]);
   for(long long j=spans[2*part];j<spans[2*part+1];j++){
    const double* h=planes+5*j;double norm=sqrt(h[0]*h[0]+h[1]*h[1]+h[2]*h[2]);
    double value=h[0]*x[0]+h[1]*x[1]+h[2]*x[2]+h[3];
    double tol=1e-9*(1.+fabs(h[3])+norm*(1.+fabs(x[0])+fabs(x[1])+fabs(x[2])));
    if(!isfinite(value)||value<=tol){contained=false;break;}
   }
   if(contained&&rec[13]){
    double d[3]={x[0]-rec[0],x[1]-rec[1],x[2]-rec[2]},q=0.;
    for(int k=0;k<3;k++)for(int h=0;h<3;h++)q+=d[k]*rec[3+3*k+h]*d[h];
    double r2=rec[12]*rec[12],tol=1e-9*(1.+r2+fabs(q));
    if(!isfinite(q)||q>=r2-tol)contained=false;
   }
  }
  if(contained){removed[i]=1;return;}
 }
}
'''


_POLYGON_KERNEL=r'''
__device__ bool in_convex_sweep(const double* x,const double* records,
 const long long* spans,const double* planes,int parts){
 for(int part=0;part<parts;part++){
  bool inside=true;const double* rec=records+14*part;
  for(long long j=spans[2*part];j<spans[2*part+1];j++){
   const double* h=planes+5*j;
   double value=h[0]*x[0]+h[1]*x[1]+h[2]*x[2]+h[3];
   double norm=sqrt(h[0]*h[0]+h[1]*h[1]+h[2]*h[2]);
   // Internal partition planes may be touched, external boundaries may not.
   double tol=h[4]?1e-9*(1.+fabs(h[3])+norm*(1.+fabs(x[0])+fabs(x[1])+fabs(x[2]))):0.;
   if(!isfinite(value)||value<tol){inside=false;break;}
  }
  if(inside&&rec[13]){
   double d[3]={x[0]-rec[0],x[1]-rec[1],x[2]-rec[2]},q=0.;
   for(int k=0;k<3;k++)for(int h=0;h<3;h++)q+=d[k]*rec[3+3*k+h]*d[h];
   double r2=rec[12]*rec[12];
   if(!isfinite(q)||q>=r2-1e-9*(1.+r2+fabs(q)))inside=false;
  }
  if(inside)return true;
 }
 return false;
}
extern "C" __global__ void clipped_convex_support(const double* geometry,long long count,
 const double* records,const long long* spans,const double* planes,int parts,
 const double* clips,const double* circle,unsigned char* removed){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||removed[i])return;
 const double* g=geometry+10*i;const double* clip=clips+7*i;
 double r=fmin(g[9],clip[6]);if(!(r>0.)||!isfinite(r))return;
 double eps=1e-8*(1.+fabs(g[0])+fabs(g[1])+fabs(g[2])+r);
 double a[80],b[80];int n=32;
 for(int j=0;j<n;j++){a[2*j]=(r+eps)*circle[2*j];a[2*j+1]=(r+eps)*circle[2*j+1];}
 for(int slot=0;slot<6;slot+=3){
  if(isinf(clip[slot+2])&&clip[slot+2]>0.)continue;
  if(!isfinite(clip[slot])||!isfinite(clip[slot+1])||!isfinite(clip[slot+2]))return;
  int m=0;
  for(int j=0;j<n;j++){
   int k=(j+1)%n;double x=a[2*j],y=a[2*j+1],xx=a[2*k],yy=a[2*k+1];
   double v=clip[slot]*x+clip[slot+1]*y+clip[slot+2]+eps;
   double w=clip[slot]*xx+clip[slot+1]*yy+clip[slot+2]+eps;
   if(v>=0.){if(m>=40)return;b[2*m]=x;b[2*m+1]=y;m++;}
   if((v>=0.)!=(w>=0.)){
    if(m>=40)return;double t=v/(v-w);b[2*m]=x+t*(xx-x);b[2*m+1]=y+t*(yy-y);m++;
   }
  }
  n=m;if(n<3)return;for(int j=0;j<2*n;j++)a[j]=b[j];
 }
 for(int j=0;j<n;j++){
  double x[3];for(int k=0;k<3;k++)x[k]=g[k]+a[2*j]*g[3+k]+a[2*j+1]*g[6+k];
  if(!in_convex_sweep(x,records,spans,planes,parts))return;
 }
 removed[i]=1;
}
'''

_ANGLES=np.arange(32)*2*np.pi/32
_CIRCLE=np.ascontiguousarray(np.column_stack((np.cos(_ANGLES),np.sin(_ANGLES)))/np.cos(np.pi/32))


def _clipped_polygon(g,clip):
    """An outward bound, not boundary samples or a split of the actual GS."""
    radius=min(g[9],clip[6])
    if not np.isfinite(radius) or radius<=0:return np.empty((0,2))
    eps=1e-8*(1+abs(g[:3]).sum()+radius);poly=(radius+eps)*_CIRCLE
    for slot in (0,3):
        h=clip[slot:slot+3]
        if np.isposinf(h[2]):continue
        if not np.isfinite(h).all():return np.empty((0,2))
        result=[]
        for x,y in zip(poly,np.roll(poly,-1,axis=0)):
            a=x@h[:2]+h[2]+eps;b=y@h[:2]+h[2]+eps
            if a>=0:result.append(x)
            if (a>=0)!=(b>=0):result.append(x+a/(a-b)*(y-x))
        poly=np.asarray(result).reshape(-1,2)
        if len(poly)<3:return np.empty((0,2))
    return poly


def _points_in_convex_sweep(points,packed):
    inside=np.zeros(len(points),bool)
    for rec,span in zip(packed[0],packed[1]):
        good=np.ones(len(points),bool)
        for h in packed[2][span[0]:span[1]]:
            value=points@h[:3]+h[3]
            tol=1e-9*(1+abs(h[3])+np.linalg.norm(h[:3])*(1+abs(points).sum(axis=1))) if h[4] else 0.
            good &= np.isfinite(value)&(value>=tol)
        if rec[13]:
            delta=points-rec[:3];q=np.einsum('ni,ij,nj->n',delta,rec[3:12].reshape(3,3),delta)
            r2=rec[12]**2;good &= np.isfinite(q)&(q<r2-1e-9*(1+r2+abs(q)))
        inside |= good
    return inside


def _clipped_convex_cpu(geometry,clips,packed,removed):
    result=removed.copy()
    for i in np.flatnonzero(~result):
        g=geometry[i];poly=_clipped_polygon(g,clips[i])
        if len(poly)<3:continue
        points=g[:3]+poly[:,0,None]*g[3:6]+poly[:,1,None]*g[6:9]
        result[i]=_points_in_convex_sweep(points,packed).all()
    return result


def _contained_cpu(geometry,packed):
    records,spans,planes,_,_=packed
    corners=np.stack([geometry[:,:3]+geometry[:,9,None]*(sx*geometry[:,3:6]+sy*geometry[:,6:9])
                      for sx,sy in ((-1,-1),(1,-1),(-1,1),(1,1))],axis=1)
    removed=np.zeros(len(geometry),bool)
    for rec,span in zip(records,spans):
        inside=np.ones(corners.shape[:2],bool)
        for h in planes[span[0]:span[1]]:
            value=corners@h[:3]+h[3]
            tol=1e-9*(1+abs(h[3])+np.linalg.norm(h[:3])*(1+abs(corners).sum(axis=2)))
            inside &= np.isfinite(value)&(value>tol)
        if rec[13]:
            delta=corners-rec[:3];q=np.einsum('...i,ij,...j->...',delta,rec[3:12].reshape(3,3),delta)
            r2=rec[12]**2;tol=1e-9*(1+r2+abs(q))
            inside &= np.isfinite(q)&(q<r2-tol)
        removed |= inside.all(axis=1)
    return removed


def fully_cut_support_keep(previous,sweep,backend,*,hit_evidence=None,radius_sigma=4.,device_mask=False,excluded=None,selection=None):
    """Delete only proven fully cut draw AND material supports.

    A proof using inherited draw clipping cannot retire a wider material
    footprint. Identical clips share one proof; the unclipped-square proof
    covers both footprints without another kernel. Uncertain rows survive.
    IDs and hit counts do not authorize deletion.

    device_mask returns a CUDA uint32 DELETE mask, without a host readback.
    Otherwise return a host boolean KEEP mask.
    """
    p=previous.endpoints['positions'];keep=np.ones(len(p),bool);lo,hi=bounds(sweep)
    if not np.isfinite(radius_sigma) or radius_sigma<=0:raise ValueError('Positive finite support radius required')
    if device_mask and backend!='cuda':raise ValueError('Device mask requires CUDA')
    radius=radius_sigma*previous.gaussian.attributes['scale'][:,0].astype(float)
    # A clipped support may be contained even when its centre is outside.
    ids=(selection.rows if selection is not None else
         np.flatnonzero(np.all((p+radius[:,None]>=lo)&(p-radius[:,None]<=hi),axis=1)&np.isfinite(radius)&(radius>0)))
    if not len(ids):
        if device_mask:
            from cutsim.cuda.profile_dexel_cuda import runtime
            rt=runtime()
            with rt.context():return rt.cp.zeros(len(p),rt.cp.uint32)
        return keep
    geometry=np.ascontiguousarray(np.column_stack((p[ids],previous.gaussian.u[ids],previous.gaussian.v[ids],radius[ids])),dtype=float)
    packed=selection.packed if selection is not None else pack_cutters((sweep,))
    # A flat/ball cutting tool restricted by two axial planes is convex.
    # Translating it along ONE straight segment preserves convexity. This is
    # a type-specific proof; never infer convexity for a generic primitive union.
    # FlatSweep is exactly a convex capsule extruded in Z.
    # Its internal primitive seams are not uncut walls; a clipped disk may
    # cross them and still be entirely inside the whole cutter volume.
    convex=type(sweep) in (CuttingSweep,FlatSweep)
    clipping=getattr(previous,'clipping',None)
    clips=np.tile([0,0,np.inf,0,0,np.inf,np.inf],(len(ids),1)) if clipping is None else np.ascontiguousarray(clipping.planes[ids])
    material=getattr(previous,'material_clipping',None)
    material_clips=clips if material is None else np.ascontiguousarray(material.planes[ids])
    different=np.any(clips!=material_clips,axis=1)
    if backend=='cuda':
        from cutsim.cuda.profile_dexel_cuda import runtime
        rt=runtime();cp=rt.cp
        with rt.context():
            if not hasattr(rt,'full_support_kernel'):
                rt.full_support_kernel=cp.RawKernel(_KERNEL,'full_support',options=('--fmad=false',))
            # An observed point outside the current cut vetoes whole-support
            # deletion. No-hit rows still use the geometric proof: absence of
            # a hit never becomes permission to remove a GS.
            eligible=cp.ones(len(ids),cp.bool_)
            if hit_evidence is not None:eligible &= (hit_evidence.flags[cp.asarray(ids)]&4)==0
            if excluded is not None:eligible &= excluded[cp.asarray(ids)]==0
            active=cp.flatnonzero(eligible)
            result=cp.empty(len(active),cp.uint8)
            geometry_gpu=cp.asarray(geometry)[active];packed_gpu=[cp.asarray(x) for x in packed[:3]]
            if len(active):rt.full_support_kernel(((len(active)+127)//128,),(128,),
                (geometry_gpu,np.int64(len(active)),*packed_gpu,np.int32(len(packed[0])),result))
            # The unclipped square certificate covers both domains. Only a
            # clipping-dependent deletion needs a second material certificate.
            whole=result.copy() if different.any() else None
            if convex and len(active):
                if not hasattr(rt,'clipped_convex_support_kernel'):
                    rt.clipped_convex_support_kernel=cp.RawKernel(_POLYGON_KERNEL,'clipped_convex_support',options=('--fmad=false',))
                rt.clipped_convex_support_kernel(((len(active)+127)//128,),(128,),
                    (geometry_gpu,np.int64(len(active)),*packed_gpu,np.int32(len(packed[0])),cp.asarray(clips)[active],cp.asarray(_CIRCLE),result))
            if len(active):
                from cutsim.rendering.analytic_disk_containment import CUDA
                if not hasattr(rt,'analytic_disk_support_kernel'):
                    rt.analytic_disk_support_kernel=cp.RawKernel(CUDA,'analytic_disk_support',options=('--fmad=false',))
                rt.analytic_disk_support_kernel(((len(active)+127)//128,),(128,),
                    (geometry_gpu,np.int64(len(active)),*packed_gpu,np.int32(len(packed[0])),cp.asarray(clips)[active],result))
            if different.any():
                check=cp.flatnonzero(cp.asarray(different)[active]&(result!=0)&(whole==0))
                if len(check):
                    mg=cp.ascontiguousarray(geometry_gpu[check])
                    mc=cp.ascontiguousarray(cp.asarray(material_clips)[active[check]])
                    confirmed=cp.zeros(len(check),cp.uint8)
                    if convex:rt.clipped_convex_support_kernel(((len(check)+127)//128,),(128,),
                        (mg,np.int64(len(check)),*packed_gpu,np.int32(len(packed[0])),mc,cp.asarray(_CIRCLE),confirmed))
                    rt.analytic_disk_support_kernel(((len(check)+127)//128,),(128,),
                        (mg,np.int64(len(check)),*packed_gpu,np.int32(len(packed[0])),mc,confirmed))
                    result[check]=confirmed
            all_removed=cp.zeros(len(ids),cp.uint8);all_removed[active]=result
            if device_mask:
                mask=cp.zeros(len(p),cp.uint32);mask[cp.asarray(ids)]=all_removed
                return mask
            removed=cp.asnumpy(all_removed,stream=rt.stream).astype(bool)
    elif backend=='cpu':
        removed=_contained_cpu(geometry,packed)
        whole=removed.copy()
        if convex:removed=_clipped_convex_cpu(geometry,clips,packed,removed)
        from cutsim.rendering.analytic_disk_containment import certify_cpu
        removed=certify_cpu(geometry,clips,packed,removed)
        check=np.flatnonzero(different&removed&~whole)
        if len(check):
            confirmed=np.zeros(len(check),bool)
            if convex:confirmed=_clipped_convex_cpu(geometry[check],material_clips[check],packed,confirmed)
            removed[check]=certify_cpu(geometry[check],material_clips[check],packed,confirmed)
    else:raise ValueError('Support retirement requires cpu or cuda')
    keep[ids]=~removed
    return keep
