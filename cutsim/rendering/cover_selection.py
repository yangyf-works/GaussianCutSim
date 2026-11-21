"""Bounded greedy child-cover selection without per-child host round trips."""
import numpy as np

CUDA = r'''
__device__ double warp_sum(double x){
 for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;
}
extern "C" __global__ void select_cover(const unsigned char* cover,
 const unsigned char* wanted,const double* weights,unsigned char* valid,
 unsigned char* united,int ns,int nc,int limit,const double* denominator,const double* weight_total,
 double tolerance,double* output){
 __shared__ unsigned char target[896],occupied[896];
 __shared__ double w[896],scores[8],missing[8],extra[8];
 __shared__ int choices[8],winner,chosen_count,stop;
 double missing_total=*denominator,total=*weight_total;
 int t=threadIdx.x,lane=t%32,warp=t/32;
 for(int j=t;j<ns;j+=256){target[j]=wanted[j];occupied[j]=united[j];w[j]=weights[j];}
 if(!t){chosen_count=0;stop=0;}__syncthreads();
 for(int step=0;step<limit;step++){
  double best=-1.;int at=-1;
  for(int c=warp;c<nc;c+=8){
   if(!valid[c])continue;
   double score=0.,over=0.;
   for(int j=lane;j<ns;j+=32){
    bool yes=cover[(long long)c*ns+j];
    if(yes&&target[j]&&!occupied[j])score+=w[j];
    if((yes||occupied[j])&&!target[j])over+=w[j];
   }
   score=warp_sum(score);over=warp_sum(over);
   if(!lane&&over/total<=tolerance&&(score>best||(score==best&&(at<0||c<at)))){best=score;at=c;}
  }
  if(!lane){scores[warp]=best;choices[warp]=at;}__syncthreads();
  if(!t){
   double best=-1.;winner=-1;
   for(int k=0;k<8;k++)if(scores[k]>best||(scores[k]==best&&choices[k]>=0&&(winner<0||choices[k]<winner))){best=scores[k];winner=choices[k];}
   if(best<=0.)stop=1;
   else{output[3+chosen_count++]=winner;valid[winner]=0;}
  }
  __syncthreads();if(stop)break;
  for(int j=t;j<ns;j+=256)occupied[j]|=cover[(long long)winner*ns+j];
  __syncthreads();
  double left=0.;for(int j=t;j<ns;j+=256)if(target[j]&&!occupied[j])left+=w[j];
  left=warp_sum(left);if(!lane)missing[warp]=left;__syncthreads();
  if(!t){double sum=0.;for(int k=0;k<8;k++)sum+=missing[k];if(sum/missing_total<=tolerance)stop=1;}
  __syncthreads();if(stop)break;
 }
 double left=0.,over=0.;
 for(int j=t;j<ns;j+=256){
  united[j]=occupied[j];if(target[j]&&!occupied[j])left+=w[j];if(!target[j]&&occupied[j])over+=w[j];
 }
 left=warp_sum(left);over=warp_sum(over);
 if(!lane){missing[warp]=left;extra[warp]=over;}__syncthreads();
 if(!t){
  double a=0.,b=0.;for(int k=0;k<8;k++){a+=missing[k];b+=extra[k];}
  output[0]=chosen_count;output[1]=a/missing_total;output[2]=b/total;
 }
}
'''


OPACITY_CUDA = r'''
__device__ double opacity_sum(double x){
 for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;
}
extern "C" __global__ void refine_opacity(const double* alpha,const unsigned char* coverage,
 const unsigned char* wanted,const double* weights,const double* target,
 const unsigned char* valid,const double* initial_current,const unsigned char* initial_union,
 const long long* initial_indices,int initial_count,int ns,int dense,int nc,int limit,
 const double* weight_total,double tolerance,double* output){
 __shared__ double current[896],w[896],goal[896],scores[8],loss_a[8],loss_b[8],den_a[8],den_b[8];
 __shared__ unsigned char occupied[896],desired[896],available[128];
 __shared__ int choices[8],winner,count,stop;
 __shared__ double lost;
 int t=threadIdx.x,lane=t%32,warp=t/32;
 for(int j=t;j<ns;j+=256){current[j]=initial_current[j];occupied[j]=initial_union[j];
  w[j]=weights[j];goal[j]=target[j];desired[j]=wanted[j];}
 for(int j=t;j<nc;j+=256)available[j]=valid[j];
 __syncthreads();
 if(!t){count=initial_count;stop=0;for(int j=0;j<initial_count;j++){
  output[2+j]=initial_indices[j];available[initial_indices[j]]=0;}}
 __syncthreads();
 // Check loss before each choice and after the final allowed child.
 for(int step=initial_count;step<=limit;step++){
  double a=0.,b=0.,da=0.,db=0.;
  for(int j=t;j<ns;j+=256){double v=current[j]<.5?goal[j]:0.;
   if(j<dense){a+=v;da+=goal[j];}else{b+=v;db+=goal[j];}}
  a=opacity_sum(a);b=opacity_sum(b);da=opacity_sum(da);db=opacity_sum(db);
  if(!lane){loss_a[warp]=a;loss_b[warp]=b;den_a[warp]=da;den_b[warp]=db;}__syncthreads();
  if(!t){a=b=da=db=0.;for(int k=0;k<8;k++){a+=loss_a[k];b+=loss_b[k];da+=den_a[k];db+=den_b[k];}
   lost=fmax(a/fmax(da,1e-30),b/fmax(db,1e-30));stop=lost<=tolerance||step==limit;}
  __syncthreads();if(stop)break;
  double best=-1.;int at=-1;
  for(int c=warp;c<nc;c+=8){
   if(!available[c])continue;
   double gain=0.,over=0.;
   for(int j=lane;j<ns;j+=32){double combined=current[j]+(1-current[j])*alpha[(long long)c*ns+j];
    gain+=goal[j]*(fmax(.5-current[j],0.)-fmax(.5-combined,0.));
    if((coverage[(long long)c*ns+j]||occupied[j])&&!desired[j])over+=w[j];}
   gain=opacity_sum(gain);over=opacity_sum(over);
   if(!lane&&over/(*weight_total)<=tolerance&&(gain>best||(gain==best&&(at<0||c<at)))){best=gain;at=c;}
  }
  if(!lane){scores[warp]=best;choices[warp]=at;}__syncthreads();
  if(!t){double best=-1.;winner=-1;
   for(int k=0;k<8;k++)if(scores[k]>best||(scores[k]==best&&choices[k]>=0&&(winner<0||choices[k]<winner))){best=scores[k];winner=choices[k];}
   stop=best<=0.;if(!stop){output[2+count++]=winner;available[winner]=0;}}
  __syncthreads();if(stop)break;
  for(int j=t;j<ns;j+=256){current[j]+=(1-current[j])*alpha[(long long)winner*ns+j];occupied[j]|=coverage[(long long)winner*ns+j];}
  __syncthreads();
 }
 if(!t){output[0]=count;output[1]=lost;}
}
'''


def refine_opacity(rt, alpha, coverage, wanted, weights, target, valid, current, united,
                   selected, *, dense_count, per_parent, total, tolerance):
    """Finish opacity-aware selection on GPU, with a single result readback."""
    cp=rt.cp;ns=len(wanted);nc=len(coverage)
    if ns>896 or nc>128:raise ValueError('Opacity selector capacity exceeded')
    if not 0<=dense_count<=ns or not len(selected)<=per_parent:
        raise ValueError('Invalid opacity selection limits')
    if not hasattr(rt,'refine_opacity_kernel'):
        rt.refine_opacity_kernel=cp.RawKernel(OPACITY_CUDA,'refine_opacity',options=('--fmad=false',))
    output=cp.empty(2+per_parent,cp.float64)
    rt.refine_opacity_kernel((1,),(256,),(
        cp.ascontiguousarray(alpha,dtype=cp.float64),cp.ascontiguousarray(coverage,dtype=cp.uint8),
        cp.ascontiguousarray(wanted,dtype=cp.uint8),weights,target,
        cp.ascontiguousarray(valid,dtype=cp.uint8),current,cp.ascontiguousarray(united,dtype=cp.uint8),
        cp.asarray(selected,dtype=cp.int64),np.int32(len(selected)),np.int32(ns),np.int32(dense_count),
        np.int32(nc),np.int32(per_parent),cp.asarray(total,dtype=cp.float64),np.float64(tolerance),output))
    result=cp.asnumpy(output,stream=rt.stream)
    selected[:]=result[2:2+int(result[0])].astype(np.int64).tolist()
    return float(result[1])


def select_cover(rt, cover, wanted, weights, valid, united, selected, *,
                 per_parent, missing_total, total, tolerance):
    """Append the best legal children; preserve the caller's cover and budget.

    A single CTA keeps all validation witnesses and the evolving union in
    shared memory. Only selected indices and final error metrics return to CPU.
    The caller handles refinement/fitting and the final missing/excess checks.
    """
    cp=rt.cp;ns=len(wanted);limit=max(0,per_parent-len(selected))
    if ns>896:raise ValueError('Cover selector supports at most 896 witnesses')
    if not hasattr(rt,'select_cover_kernel'):
        rt.select_cover_kernel=cp.RawKernel(CUDA,'select_cover',options=('--fmad=false',))
    # Scalar totals are already GPU results. Passing them by pointer avoids
    # an otherwise hidden host synchronization for each parent.
    output=cp.empty(3+limit,cp.float64)
    rt.select_cover_kernel((1,),(256,),
        (cover,wanted,weights,valid,united,np.int32(ns),np.int32(len(cover)),np.int32(limit),
         cp.asarray(missing_total,dtype=cp.float64),cp.asarray(total,dtype=cp.float64),np.float64(tolerance),output))
    host=cp.asnumpy(output,stream=rt.stream)
    selected.extend(host[3:3+int(host[0])].astype(np.int64).tolist())
    return float(host[1]),float(host[2])
