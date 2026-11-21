"""Reuse certified live planar coverage across adjacent cutting blocks.

An old end-cap clip can become an interior seam of a continuous floor. Only
coverage already admitted by a newly generated coplanar GS may release it.
No historic cutter union, unbounded plane extension, or extra Dexel is used.
"""
import numpy as np
from cutsim.rendering.planar_floor_domain import CUDA as FLOOR_CUDA

CUDA=FLOOR_CUDA+r'''
extern "C" __global__ void shared_planar_coverage(const double* g,const double* radii,
 const long long* rows,const long long* starts,const long long* counts,const long long* sources,
 const long long* patches,const long long* meta,const double* rec,const long long* spans,const double* planes,
 const double* samples,int ns,int nr,unsigned char* truth,unsigned int* changed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns*nr||truth[i])return;
 int row=i/ns,j=i%ns;long long target=rows[row];const double* a=g+25*target;
 double x=radii[target]*samples[2*j],y=radii[target]*samples[2*j+1],p[3];
 for(int k=0;k<3;k++)p[k]=a[k]+x*a[3+k]+y*a[6+k];
 for(long long h=starts[row];h<starts[row]+counts[row];h++){
  long long source=sources[h];const double* b=g+25*source;double u=0.,v=0.;
  for(int k=0;k<3;k++){double d=p[k]-b[k];u+=d*b[3+k];v+=d*b[6+k];}
  if(u*u+v*v>radii[source]*radii[source])continue;
  bool yes=true;
  for(int k=13;k<25;k+=3)if(b[k]*u+b[k+1]*v+b[k+2]<-1e-10){yes=false;break;}
  if(yes&&floor_domain_inside(meta+4*patches[source],p,rec,spans,planes)){
   truth[i]=1;atomicOr(changed+row,1u);return;
  }
 }
}
'''


def extend_coverage(rt,region,nold):
    """Expand old residual truth only inside accepted current planar disks."""
    from cutsim.rendering.sampled_analytic_clip import fit
    cp=rt.cp;pairs=region.coplanar_pairs
    stats=dict(coplanar_support_pairs=0,coplanar_support_updated=0)
    if not len(pairs):return stats
    accepted=cp.ones(len(region.geometry),cp.bool_)
    accepted[region.rows]=region.accepted.astype(cp.bool_)
    pairs=pairs[accepted[pairs[:,0]]]
    if not len(pairs):return stats
    stats['coplanar_support_pairs']=len(pairs)
    order=cp.argsort(pairs[:,1]);pairs=pairs[order]
    rows=region.rows;nr=len(rows)
    starts=cp.searchsorted(pairs[:,1],rows,side='left').astype(cp.int64)
    counts=cp.searchsorted(pairs[:,1],rows,side='right').astype(cp.int64)-starts
    geometry=region.geometry.copy()
    new=region.rows>=nold;newrows=region.rows[new]
    geometry[newrows,13:19]=region.preview[new,3:9]
    geometry[newrows,19:25]=region.preview[new,9:15]
    if not hasattr(rt,'shared_planar_coverage_kernel'):
        rt.shared_planar_coverage_kernel=cp.RawKernel(CUDA,'shared_planar_coverage',options=('--fmad=false',))
    changed=cp.zeros(nr,cp.uint32)
    for samples,truth in ((region.sample_xy,region.sample_truth),(region.validation_samples,region.validation_truth)):
        ns=len(samples)
        if nr:rt.shared_planar_coverage_kernel(((nr*ns+127)//128,),(128,),
            (geometry,region.radii,rows,starts,counts,cp.ascontiguousarray(pairs[:,0]),*region.floor_domain_data,
             samples,np.int32(ns),np.int32(nr),truth,changed))
    dirty=cp.flatnonzero(changed)
    stats['coplanar_support_updated']=len(dirty)
    if not len(dirty):return stats
    if getattr(region,'sample_first',False):
        from cutsim.rendering.sample_priority_clip import refit
        refit(rt,region,dirty)
        return stats
    disks=cp.zeros((len(dirty),5));disks[:,0]=dirty;disks[:,3]=region.radii[rows[dirty]];disks[:,4]=-1.
    result=cp.empty((len(dirty),15));error=cp.empty((len(dirty),2));valid=cp.empty(len(dirty),cp.uint8)
    fit(rt,region,disks,result,error,valid,alternatives=8)
    region.preview[dirty]=result;region.error[dirty]=error;region.accepted[dirty]=valid
    return stats
