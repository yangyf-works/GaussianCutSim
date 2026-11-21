"""GPU checks for two halfplanes in native angle/split Float32 form.

Report approximation loss separately from structural drawability. A valid
best fit outside tolerance remains a fallback until bounded subdivision passes.
Invalid/empty fits with surviving witnesses restore both inherited clips atomically;
failure of two-edge approximation alone never deletes a GS.
The shared residual path uses the same 128 points, not an independent holdout.
"""
import numpy as np

CUDA=r'''
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
extern "C" __global__ void validate_edges(const double* geometry,const long long* rows,const double* radii,
 const double* xy,const double* weights,const unsigned char* truth,int ns,int n,double tolerance,
 double* preview,double* errors,unsigned char* valid,unsigned char* fallback,unsigned char* accepted){
 int i=blockIdx.x,lane=threadIdx.x;if(i>=n)return;
 long long row=rows[i];double R=radii[row];double* out=preview+15*i;
 __shared__ double line[6],native[6],sum[128*5];__shared__ int structural,bad;
 if(!lane){bad=0;errors[2*i]=errors[2*i+1]=0.;}
 for(int side=0;side<2;side++){
  if(!lane){structural=isfinite(R)&&R>0.;bool empty=false;
   for(int k=0;k<6;k++)line[k]=out[3+6*side+k];
   for(int k=0;k<6;k+=3){double len=hypot(line[k],line[k+1]);
    bool finite=isfinite(line[k])&&isfinite(line[k+1])&&isfinite(line[k+2]);
    if(finite&&len>1e-15)for(int j=0;j<3;j++)line[k+j]/=len;
    else if(finite&&len<=1e-15){if(line[k+2]<0.)empty=true;else{line[k]=line[k+1]=0.;line[k+2]=INFINITY;}}
   }
   // Native angle/split cannot encode a zero-normal negative constant.
   // Two opposed halfplanes express empty without changing GS geometry.
   if(empty){line[0]=1.;line[1]=line[4]=0.;line[3]=-1.;line[2]=line[5]=-fmax(R*1e-6,1e-10);}
   for(int k=0;k<6;k+=3){
    bool disabled=line[k]==0.&&line[k+1]==0.&&isinf(line[k+2])&&line[k+2]>0.;
    float angle=(float)atan2(line[k+1],line[k]),split=(float)(-line[k+2]/(R/2));
    if(angle>=(float)3.141592653589793)angle=-(float)3.141592653589793;
    if(disabled){native[k]=native[k+1]=0.;native[k+2]=INFINITY;}
    else{if(!isfinite(line[k])||!isfinite(line[k+1])||!isfinite(line[k+2])||!isfinite(angle)||!isfinite(split))structural=0;
     native[k]=cosf(angle);native[k+1]=sinf(angle);native[k+2]=-(double)split*R/2.;}
   }
  }__syncthreads();
  double lost=0.,over=0.,target=0.,total=0.,predicted=0.;
  for(int j=lane;j<ns;j+=128){double x=R*xy[2*j],y=R*xy[2*j+1],w=weights[j]*(ns>128&&j>=ns-128?.125:1.);
   bool wanted=truth[i*ns+j],yes=native[0]*x+native[1]*y+native[2]>=-1e-10&&native[3]*x+native[4]*y+native[5]>=-1e-10;
   total+=w;if(wanted)target+=w;if(yes)predicted+=w;if(wanted&&!yes)lost+=w;if(!wanted&&yes)over+=w;
  }
  sum[5*lane]=lost;sum[5*lane+1]=over;sum[5*lane+2]=target;sum[5*lane+3]=total;sum[5*lane+4]=predicted;
  __syncthreads();for(int step=64;step;step/=2){if(lane<step)for(int k=0;k<5;k++)sum[5*lane+k]+=sum[5*(lane+step)+k];__syncthreads();}
  if(!lane){errors[2*i]=fmax(errors[2*i],sum[0]/fmax(sum[2],1e-30));errors[2*i+1]=fmax(errors[2*i+1],sum[1]/fmax(sum[3],1e-30));
   bool unsafe=!structural||(sum[2]>0.&&sum[4]==0.);bad|=unsafe;
   for(int k=0;k<6;k++)out[3+6*side+k]=line[k];
  }__syncthreads();
 }
 if(!lane){
  // Commit a GS as one unit: never mix a new draw clip with an inherited
  // material clip (or vice versa) after either representation fails.
  if(bad)for(int k=0;k<12;k++)out[3+k]=geometry[25*row+13+k];
  fallback[i]=bad;valid[i]=!bad;accepted[i]=!bad&&errors[2*i]<=tolerance&&errors[2*i+1]<=tolerance;
 }
}
'''


def validate_twoedge(rt,region,*,tolerance=.02):
    cp=rt.cp;n=len(region.rows)
    region.twoedge_validation_error=cp.zeros((n,2))
    region.twoedge_drawable=cp.ones(n,cp.uint8)
    region.twoedge_fallback=cp.zeros(n,cp.uint8)
    region.accepted=cp.zeros(n,cp.uint8)
    if not hasattr(rt,'validate_twoedge_kernel'):
        rt.validate_twoedge_kernel=cp.RawKernel(CUDA,'validate_edges',options=('--fmad=false',))
    if n:rt.validate_twoedge_kernel((n,),(128,),(
        getattr(region,'rollback_geometry',region.geometry),region.rows,region.radii,region.validation_samples,region.validation_weights,
        region.validation_truth,np.int32(len(region.validation_samples)),np.int32(n),np.float64(tolerance),
        region.preview,region.twoedge_validation_error,region.twoedge_drawable,region.twoedge_fallback,region.accepted))
    return dict(twoedge_checked=n,twoedge_holdout_points=len(region.validation_samples) if getattr(region,'validation_is_holdout',True) else 0,
        twoedge_validation_points=len(region.validation_samples),twoedge_tolerance=tolerance,
        twoedge_unrepresentable=int(cp.count_nonzero(region.accepted==0).item()),
        twoedge_invalid_or_empty=int(cp.count_nonzero(region.twoedge_fallback).item()),twoedge_failure_deletes_gs=False)
