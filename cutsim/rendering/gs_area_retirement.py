"""GPU area/opacity-fraction retirement of finite, already-clipped GS.

Uniform-area disc quadrature: 512 samples, refined to 4096 near the selected
threshold or when the surviving clip is small. This is an estimate, not a proof.
The footprint uses the snapshot finite radius, never UI zoom/size.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.gs_extent import support_factor, support_radius
from cutsim.cutting.local_dexel import bounds
from cutsim.rendering.profile_cut_clip import pack_cutters

from cutsim.rendering.gs_sample_evaluation import CUDA as SAMPLE_CUDA, disc as _disc, device_disc

# R measures the central opaque footprint; clipping covers a wider domain.
RESIDUAL_SIGMA = 1.17
DEFAULT_RESIDUAL_THRESHOLD = .4

KERNEL=SAMPLE_CUDA+r'''
__device__ bool cut_inside(const double* p,const double* records,const long long* spans,
 const double* planes,int parts){
 for(int part=0;part<parts;part++){
  bool yes=true;const double* r=records+14*part;
  for(long long j=spans[2*part];j<spans[2*part+1];j++){
   const double* h=planes+5*j;double v=h[0]*p[0]+h[1]*p[1]+h[2]*p[2]+h[3];
   if(v<(h[4]?1e-8:-1e-10)){yes=false;break;}
  }
  if(yes&&r[13]){double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
   q=gs_quadratic_form(r+3,d);
   yes=q<r[12]*r[12]-1e-8;
  }
  if(yes)return true;
 }
 return false;
}
extern "C" __global__ void area_retire(const double* geometry,const double* clips,long long count,
 const double* records,const long long* spans,const double* planes,int parts,
 const double* coarse,const double* fine,double threshold,double weight_scale,unsigned char* removed,
 double* fractions,unsigned char* refined){
 long long i=blockIdx.x;if(i>=count)return;
 int lane=threadIdx.x;__shared__ int totals[128],again;
 __shared__ double weights[128],insides[128];
 const double* g=geometry+10*i;const double* c=clips+7*i;
 double radius=fmin(g[9],c[6]);
 if(!lane){removed[i]=0;fractions[i]=0.;refined[i]=0;}
 if(!(radius>0.)||!isfinite(radius))return;
 for(int pass=0;pass<2;pass++){
  int valid=0,n=pass?4096:512;double total_weight=0.,cut=0.;const double* samples=pass?fine:coarse;
  for(int j=lane;j<n;j+=blockDim.x){
   double x=radius*samples[2*j],y=radius*samples[2*j+1];
   if(!gs_sample_admitted(c,x,y))continue;
   // g[9] is factor*sigma, not the clipped radius. alpha0 cancels in R.
   double weight=weight_scale>0.?gs_sample_weight(x,y,g[9]/sqrt(2.*weight_scale)):1.;
   valid++;total_weight+=weight;
   double p[3];gs_sample_point(g,x,y,p);
   if(cut_inside(p,records,spans,planes,parts))cut+=weight;
  }
  totals[lane]=valid;weights[lane]=total_weight;insides[lane]=cut;__syncthreads();
  for(int stride=64;stride;stride/=2){
   if(lane<stride){totals[lane]+=totals[lane+stride];weights[lane]+=weights[lane+stride];insides[lane]+=insides[lane+stride];}
   __syncthreads();
  }
  if(!lane){
   double ratio=weights[0]>0.?insides[0]/weights[0]:0.;
   again=!pass&&(totals[0]<32||fabs(ratio-threshold)<.1);
   if(again)refined[i]=1;
   else{fractions[i]=ratio;removed[i]=totals[0]>0&&ratio>=threshold;}
  }
  __syncthreads();if(!again)return;
 }
}
'''


def area_candidates_device(snapshot,sweep,*,threshold=.5,capture=False,radius_sigma=None,opacity_weighted=False,workspace=None,excluded=None):
    """Return a GPU uint32 area mask; the caller chooses OR, AND or area-only.

    The caller owns this transient mask. Only diagnostic capture downloads
    fractions; production retains the per-GS decisions on the GPU.
    `excluded` is a device mask of prior ray deletions. Compact candidates on
    device before quadrature; excluded rows receive no sampling work.
    """
    if not np.isfinite(threshold) or not 0<threshold<=1:raise ValueError('Area threshold must be in (0,1]')
    from cutsim.cuda.profile_dexel_cuda import runtime
    started=perf_counter();rt=runtime();cp=rt.cp;p=snapshot.endpoints['positions']
    factor=support_factor(snapshot) if radius_sigma is None else float(radius_sigma)
    if not np.isfinite(factor) or factor<=0:raise ValueError('Positive finite sampling radius required')
    radius=factor*snapshot.gaussian.attributes['scale'][:,0].astype(float)
    clips=snapshot.clipping.planes
    effective=np.minimum(radius,clips[:,6]);low,high=bounds(sweep)
    # Overlap of the whole footprint: centres outside the box can still cut.
    ids=np.flatnonzero((effective>0)&np.all((p+effective[:,None]>=low)&(p-effective[:,None]<=high),axis=1))
    if workspace is not None:
        workspace.clear()
        workspace.update(snapshot=snapshot,sweep=sweep,ids=ids,radius_sigma=factor)
    with rt.context():candidates=cp.zeros(len(p),cp.uint32)
    fractions=np.zeros(len(p)) if capture else None
    eligible_count=0
    refined_count=0;kernel_ms=0.;evaluated_count=0
    if len(ids):
        geometry=np.ascontiguousarray(np.column_stack((p[ids],snapshot.gaussian.u[ids],snapshot.gaussian.v[ids],radius[ids])))
        packed=pack_cutters((sweep,))
        with rt.context():
            if not hasattr(rt,'area_retirement_kernel'):
                rt.area_retirement_kernel=cp.RawKernel(KERNEL,'area_retire',options=('--fmad=false',))
            device_cutters=tuple(cp.asarray(x) for x in (packed if workspace is not None else packed[:3]))
            device_ids=cp.asarray(ids)
            active=cp.ones(len(ids),cp.bool_) if excluded is None else excluded[device_ids]==0
            device_geometry=cp.ascontiguousarray(cp.asarray(geometry)[active])
            device_clips=cp.ascontiguousarray(cp.asarray(clips[ids])[active])
            device_ids=device_ids[active];evaluated_count=len(device_ids)
            args=[device_geometry,device_clips,*device_cutters[:3],device_disc(rt,16,32),device_disc(rt,64,64)]
            if workspace is not None:
                workspace.update(packed_cutters=packed,device_cutters=device_cutters)
            removed=cp.empty(evaluated_count,cp.uint8);ratio=cp.empty(evaluated_count,cp.float64);refined=cp.empty(evaluated_count,cp.uint8)
            begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
            if evaluated_count:rt.area_retirement_kernel((evaluated_count,),(128,),
                (args[0],args[1],np.int64(evaluated_count),*args[2:5],np.int32(len(packed[0])),*args[5:],np.float64(threshold),np.float64(.5*factor*factor if opacity_weighted else 0.),removed,ratio,refined))
            end.record();end.synchronize();kernel_ms=float(cp.cuda.get_elapsed_time(begin,end))
            candidates[device_ids]=removed
            eligible_count=int(cp.asnumpy(cp.count_nonzero(removed),stream=rt.stream))
            refined_count=int(cp.asnumpy(cp.count_nonzero(refined),stream=rt.stream))
            if capture:fractions[cp.asnumpy(device_ids,stream=rt.stream)]=cp.asnumpy(ratio,stream=rt.stream)
    stats=dict(area_retirement_candidates=evaluated_count,area_retirement_excluded=len(ids)-evaluated_count,area_retirement_eligible=eligible_count,
        area_retirement_refined=refined_count,area_retirement_threshold=threshold,
        area_retirement_kernel_ms=kernel_ms,area_retirement_ms=(perf_counter()-started)*1000,
        area_retirement_samples=512,area_retirement_refined_samples=4096)
    return candidates,stats,fractions


def residual_candidates_device(snapshot,sweep,*,threshold=DEFAULT_RESIDUAL_THRESHOLD,capture=False,workspace=None,excluded=None):
    """R = surviving opacity / original opacity in the clipped 1.17-sigma disc.

    Equal-area samples carry exp(-r^2/(2*sigma^2)); the common positive
    centre alpha cancels. Prior clipping excludes points from BOTH sums.
    R <= threshold qualifies independently of ray hits. R=0 is a sampled
    estimate, not a whole-support proof. No valid samples never qualifies;
    its reported R is conservatively 1, as for an unvisited external GS.
    """
    if not np.isfinite(threshold) or not 0<=threshold<1:
        raise ValueError('Residual R threshold must be in [0,1)')
    mask,detail,fraction=area_candidates_device(snapshot,sweep,threshold=1.-threshold,
        capture=capture,radius_sigma=RESIDUAL_SIGMA,opacity_weighted=True,workspace=workspace,excluded=excluded)
    stats=dict(residual_r_sigma=RESIDUAL_SIGMA,residual_r_threshold=float(threshold),residual_r_metric='opacity-weighted',
        residual_r_candidates=detail['area_retirement_candidates'],
        residual_r_skipped_ray=detail['area_retirement_excluded'],
        residual_r_eligible=detail['area_retirement_eligible'],
        residual_r_refined=detail['area_retirement_refined'],
        residual_r_samples=detail['area_retirement_samples'],
        residual_r_refined_samples=detail['area_retirement_refined_samples'],
        residual_r_kernel_ms=detail['area_retirement_kernel_ms'],
        residual_r_ms=detail['area_retirement_ms'])
    return mask,stats,None if fraction is None else 1.-fraction
