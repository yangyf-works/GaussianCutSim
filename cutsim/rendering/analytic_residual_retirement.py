"""Approximate Gaussian residual retirement, separate from full-cut proofs.

Reuse Dexel's live analytic table/BVH. Contact and missing local evidence keep
support. Evaluate one 1.5-sigma domain: the union of inherited draw/material
clipping. Retire below 10% remaining weight. Newly fitted edges are not truth.
"""
from time import perf_counter
import numpy as np
from types import SimpleNamespace

RESIDUAL_THRESHOLD = .10
RESIDUAL_SIGMA = 1.5
RESIDUAL_RADIAL = 4
RESIDUAL_ANGULAR = 32
RESIDUAL_SAMPLES = RESIDUAL_RADIAL * RESIDUAL_ANGULAR


CUDA = r'''
__device__ bool residual_peer_cut(const double* p,const double* own,const long long* om,
 long long own_id,const double* g,const double* boxes,const long long* order,
 long long n,long long leaves,const long long* ids,const double* eq,const long long* meta,
 const double* rec,const long long* spans,const double* planes,
 const long long* surface_ids,long long protected_id){
 int stack[64],top=0;stack[top++]=1;
 while(top){int node=stack[--top];const double* b=boxes+6*node;
  if(p[0]<b[0]||p[0]>b[3]||p[1]<b[1]||p[1]>b[4]||p[2]<b[2]||p[2]>b[5])continue;
  if(node<leaves){stack[top++]=2*node+1;stack[top++]=2*node;continue;}
  long long leaf=node-leaves;if(leaf>=n)continue;long long row=order[leaf],patch=ids[row];
  if(protected_id>=0&&surface_ids[row]==protected_id)continue;
  const long long* m=meta+4*patch;
  if(patch==own_id||m[3]||m[0]==om[0])continue;
  if(!ar_domain(p,g+19*row,m,rec,spans,planes))continue;
  if(!finite_cut_inside(p,m,rec,spans,planes))continue;
  // Evaluate the more expensive analytic contact certificate only after
  // local support and strict finite membership have admitted the point.
  if(surface_clear_of_solid(own,m,rec,spans,planes))continue;
  return true;
 }return false;
}
__device__ bool residual_clip(double x,double y,const double* clip){
 if(x*x+y*y>clip[6]*clip[6]||clip[6]<=0.)return false;
 for(int k=0;k<6;k+=3)if(clip[k]*x+clip[k+1]*y+clip[k+2]<-1e-10)return false;
 return true;
}
__device__ bool residual_before(const double* p,
 const long long* m,const long long* spans,const double* planes,const double* stock){
 for(int k=0;k<3;k++)if(p[k]<stock[k]-1e-9||p[k]>stock[k+3]+1e-9)return false;
 // Union of generating halfspaces bounds a tangent disk. Do not intersect
 // it with its own curved quadric (that would erase cylinder tangent disks).
 for(int part=m[0];part<m[0]+m[1];part++){
  bool yes=true;for(long long j=spans[2*part];j<spans[2*part+1];j++)
   if(dot(p,planes+5*j)+planes[5*j+3]<-1e-9){yes=false;break;}
  if(yes)return true;
 }return false;
}
extern "C" __global__ void residual_ratio(const double* geometry,const double* clips,
 const long long* rows,const long long* own_ids,int count,const double* samples,const double* weights,int ns,
 const double* stock,const double* cr,const long long* cs,const double* cplanes,int parts,
 const double* g,const double* boxes,const long long* order,long long n,long long leaves,
 const long long* ids,const double* eq,const long long* meta,const double* rec,
 const long long* spans,const double* planes,const long long* surface_ids,const long long* own_surface_ids,
 double support_sigma,double threshold,int current_cut,
 unsigned int* removed,double* ratios,unsigned char* before_truth,unsigned char* after_truth){
 int i=blockIdx.x,lane=threadIdx.x;if(i>=count)return;
 long long row=rows[i],patch=own_ids[row];const double* a=geometry+10*row;
 const long long* m=meta+4*patch;const double* own=eq+16*patch;
 long long cm[4]={0,parts,0,0};
 bool contact=surface_clear_of_solid(own,cm,cr,cs,cplanes);
 __shared__ double sum[128*2];
  double before=0.,after=0.;double R=a[9]*support_sigma;
  const double* clip=clips+14*row;
  for(int j=lane;j<ns;j+=128){double x=R*samples[2*j],y=R*samples[2*j+1],p[3];
   before_truth[i*ns+j]=after_truth[i*ns+j]=0;
   for(int k=0;k<3;k++)p[k]=a[k]+x*a[3+k]+y*a[6+k];
   // One truth evaluation for the union, not two independent R decisions.
   if(!residual_clip(x,y,clip)&&!residual_clip(x,y,clip+7))continue;
   if(!residual_before(p,m,spans,planes,stock))continue;
   double w=weights[j]*exp(-.5*support_sigma*support_sigma*(samples[2*j]*samples[2*j]+samples[2*j+1]*samples[2*j+1]));
   before+=w;
   before_truth[i*ns+j]=1;
   bool cut=current_cut&&!contact&&finite_cut_inside(p,cm,cr,cs,cplanes);
   // Newly generated clipping must obey the same continuity exclusion as
   // grouped_candidates. Old-GS deletion still evaluates actual removal.
   if(!cut&&n)cut=residual_peer_cut(p,own,m,patch,g,boxes,order,n,leaves,ids,eq,meta,rec,spans,planes,
                                  surface_ids,current_cut?-1:own_surface_ids[row]);
   if(!cut){after+=w;after_truth[i*ns+j]=1;}
  }
  sum[2*lane]=before;sum[2*lane+1]=after;
 __syncthreads();for(int step=64;step;step/=2){if(lane<step)for(int k=0;k<2;k++)sum[2*lane+k]+=sum[2*(lane+step)+k];__syncthreads();}
 if(!lane){double r=sum[0]>1e-15?sum[1]/sum[0]:1.;
  // Unknown/empty denominators KEEP. This is a threshold decision, not proof.
  ratios[i]=r;removed[row]=r<threshold;
 }
}
'''


def residual_retirement(surface, sweep, already_removed, stock_bounds, *, threshold=RESIDUAL_THRESHOLD):
    from cutsim.cutting.local_dexel import bounds
    if not 0 <= threshold < 1:raise ValueError('Residual threshold must be in [0,1)')
    started=perf_counter();rt=surface.rt;cp=rt.cp;s=surface.snapshot
    p=s.endpoints['positions'];sigma=s.gaussian.attributes['scale'][:,0].astype(float)
    lo,hi=bounds(sweep)
    radius=RESIDUAL_SIGMA*sigma[:,None]
    selection=getattr(surface,'sweep_selection',None)
    candidates=(selection.rows if selection is not None else
                np.flatnonzero(np.all((p+radius>=lo)&(p-radius<=hi),axis=1)))
    with rt.context():
        output=cp.zeros(len(p),cp.uint32)
        rows=cp.asarray(candidates);rows=rows[already_removed[rows]==0]
        stats=dict(residual_threshold=threshold,residual_candidates=len(rows),residual_removed=0,
                   residual_samples=RESIDUAL_SAMPLES,residual_passes=1,residual_sigma=RESIDUAL_SIGMA,
                   residual_domain='draw_material_union',residual_gpu_ms=0.,residual_ms=0.)
        if not len(rows) or not len(surface.rows):
            surface.residual_samples=SimpleNamespace(rows=rows,ratios=cp.ones(len(rows)),
                before=cp.zeros((len(rows),RESIDUAL_SAMPLES),cp.uint8),after=cp.zeros((len(rows),RESIDUAL_SAMPLES),cp.uint8))
            surface.residual_evaluation=(rows,surface.residual_samples.ratios)
            return output,stats
        samples,output,gpu_ms = sample_residual(surface,s,rows,sweep,stock_bounds,threshold=threshold)
        stats.update(residual_removed=int(cp.count_nonzero(output).item()),residual_gpu_ms=gpu_ms)
        surface.residual_samples=samples
        # Keep the existing diagnostic API; point results live only for this block.
        surface.residual_evaluation=(rows,samples.ratios)
    stats['residual_ms']=(perf_counter()-started)*1000
    return output,stats


def sample_residual(surface,snapshot,rows,sweep,stock_bounds,*,threshold=RESIDUAL_THRESHOLD,current_cut=True):
    """One 128-point union evaluation against BEFORE live analytic support.

    New GS use their generating finite domain and old peer cuts, but never
    subtract their own generating sweep from their tangent disks.
    """
    from cutsim.cuda.analytic_ray_generation import evaluator, CUDA as ray_cuda
    from cutsim.cuda.current_gs_surface import _CUDA
    from cutsim.rendering.finite_surface_contact import CUDA as contact_cuda
    from cutsim.rendering.profile_cut_clip import pack_cutters
    from cutsim.rendering.gs_sample_evaluation import disc
    from cutsim.rendering.analytic_surfaces import pack_snapshot
    rt=surface.rt;cp=rt.cp;s=snapshot
    with rt.context():
        rows=cp.asarray(rows,dtype=cp.int64)
        output=cp.zeros(len(s.endpoints['positions']),cp.uint32)
        ratios=cp.ones(len(rows));before=cp.zeros((len(rows),RESIDUAL_SAMPLES),cp.uint8);after=cp.zeros_like(before)
        result=SimpleNamespace(rows=rows,ratios=ratios,before=before,after=after)
        if not len(rows):return result,output,0.
        evidence=evaluator(surface)
        if s is surface.snapshot:
            own_ids=evidence.all_patch_ids;data=evidence.data
        else:
            old=surface.snapshot;count=len(old.analytic_patch_ids)
            merged=SimpleNamespace(analytic_patch_ids=np.r_[old.analytic_patch_ids,s.analytic_patch_ids],
                analytic_patches={**old.analytic_patches,**s.analytic_patches})
            selection=getattr(surface,'sweep_selection',None)
            selected=(None if selection is None else np.r_[selection.rows,count+cp.asnumpy(rows,stream=rt.stream)])
            packed=pack_snapshot(merged,rows=selected)
            own_ids=cp.asarray(packed[0][count:])
            data=(cp.asarray(packed[0][:count][surface.rows]),*(cp.asarray(a) for a in packed[1:]))
        sigma=s.gaussian.attributes['scale'][:,0].astype(float)
        geometry=np.column_stack((s.endpoints['positions'],s.gaussian.u,s.gaussian.v,sigma))
        if s.clipping is None:
            clips=np.tile([0.,0.,np.inf,0.,0.,np.inf,np.inf]*2,(len(sigma),1))
        else:clips=np.column_stack((s.clipping.planes,(s.material_clipping or s.clipping).planes))
        xy=disc(RESIDUAL_RADIAL,RESIDUAL_ANGULAR);w=np.full(RESIDUAL_SAMPLES,1./RESIDUAL_SAMPLES)
        if not hasattr(rt,'analytic_single_residual_kernel'):
            rt.analytic_single_residual_kernel=cp.RawKernel(_CUDA+ray_cuda+contact_cuda+CUDA,'residual_ratio',options=('--fmad=false',))
        cr,cs,ch=(cp.asarray(a) for a in pack_cutters((sweep,))[:3])
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        rt.analytic_single_residual_kernel((len(rows),),(128,),(
            cp.asarray(geometry),cp.asarray(clips),rows,own_ids,np.int32(len(rows)),
            cp.asarray(xy),cp.asarray(w),np.int32(len(xy)),cp.asarray(np.asarray(stock_bounds,dtype=float).reshape(6)),
            cr,cs,ch,np.int32(len(cr)),surface.device_geometry,evidence.boxes,surface.order,
            np.int64(len(surface.rows)),np.int64(surface.leaves),*data,
            cp.asarray((surface.snapshot.material_surface_ids if surface.snapshot.material_surface_ids is not None else
                        surface.snapshot.gaussian.attributes['surface_id'][:,0])[surface.rows],dtype=cp.int64),
            cp.asarray(s.material_surface_ids if s.material_surface_ids is not None else
                       s.gaussian.attributes['surface_id'][:,0],dtype=cp.int64),
            np.float64(RESIDUAL_SIGMA),np.float64(threshold),np.int32(current_cut),output,ratios,before,after))
        end.record();end.synchronize()
        return result,output,float(cp.cuda.get_elapsed_time(begin,end))
