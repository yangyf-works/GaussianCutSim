"""Replace redundant forward cap support at a same-ID ball continuation.

The old tangent disk can lie outside the new barrel even when its spherical
surface is cut. Evaluate that sphere, and require opaque coverage of EVERY
uncut probe by permanent rear-cap GS. Candidates never cover each other.
This is a bounded sampled replacement, not a full-containment certificate.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.finite_surface_contact import CUDA as CONTACT_CUDA

CUDA = CONTACT_CUDA + r'''
__device__ bool cap_clip(const double* c,double x,double y){
 if(c[6]<=0.||x*x+y*y>c[6]*c[6])return false;
 for(int k=0;k<6;k+=3)if(c[k]*x+c[k+1]*y+c[k+2]<-1e-10)return false;
 return true;
}
extern "C" __global__ void cap_replace(const double* g,const double* clips,
 const long long* rows,const long long* donors,const long long* ranges,int count,
 const double* spheres,const double* xy,int ns,const double* stock,double display,
 const double* rec,const long long* spans,const double* planes,int parts,
 const unsigned int* removed,unsigned int* result){
 int i=blockIdx.x,lane=threadIdx.x;if(i>=count)return;
 long long row=rows[i];const double* a=g+10*row;const double* sphere=spheres+4*i;
 long long cut[4]={0,parts,0,0};
 if(removed[row]||!finite_cut_inside(a,cut,rec,spans,planes))return;
 __shared__ unsigned int failed,present;
 if(!lane){failed=0;present=0;}__syncthreads();
 for(int j=lane;j<ns;j+=blockDim.x){
  double x=2.*a[9]*xy[2*j],y=2.*a[9]*xy[2*j+1];
  if(!cap_clip(clips+14*row,x,y)&&!cap_clip(clips+14*row+7,x,y))continue;
  double d[3],len=0.;for(int k=0;k<3;k++){d[k]=a[k]+x*a[3+k]+y*a[6+k]-sphere[k];len+=d[k]*d[k];}
  if(len<=0.){atomicOr(&failed,1u);continue;}
  double p[3];bool in_stock=true;for(int k=0;k<3;k++){
   p[k]=sphere[k]+sphere[3]*d[k]/sqrt(len);
   if(p[k]<stock[k]-1e-9||p[k]>stock[k+3]+1e-9)in_stock=false;
  }
  if(!in_stock)continue;atomicOr(&present,1u);
  if(finite_cut_inside(p,cut,rec,spans,planes))continue;
  double trans[2]={1.,1.};
  for(long long at=ranges[2*i];at<ranges[2*i+1];at++){
   long long donor=donors[at];const double* b=g+10*donor;
   // Only permanent rear support, never another replacement candidate.
   if(removed[donor]||finite_cut_inside(b,cut,rec,spans,planes))continue;
   double u=0.,v=0.;for(int k=0;k<3;k++){u+=(p[k]-b[k])*b[3+k];v+=(p[k]-b[k])*b[6+k];}
   double rho=(u*u+v*v)/(b[9]*b[9]*display*display);if(rho>9.)continue;
   double alpha=.99*exp(-.5*rho);
   for(int side=0;side<2;side++)if(cap_clip(clips+14*donor+7*side,u,v))trans[side]*=1.-alpha;
   if(trans[0]<.5&&trans[1]<.5)break;
  }
  if(trans[0]>=.5||trans[1]>=.5)atomicOr(&failed,1u);
 }
 __syncthreads();if(!lane&&present&&!failed)result[row]=1;
}
'''


def retire_ball_continuation(surface, sweep, removed, current_ids, stock_bounds, *, display_scale=1.):
    """GPU 512-probe check; only exact spherical start faces, same identity.

    A new nonzero horizontal ball move must have the same radius and start
    centre. Different radii, headings/IDs, flats and ramps use normal cutting.
    The test leaves both sigma and inherited clipping unchanged.
    """
    from cutsim.rendering.gs_sample_evaluation import disc
    from cutsim.rendering.profile_cut_clip import pack_cutters
    from cutsim.rendering.opaque_coverage import display_factor
    started=perf_counter();rt=surface.rt;cp=rt.cp;s=surface.snapshot
    stats=dict(ball_continuation_candidates=0,ball_continuation_removed=0,ball_continuation_ms=0.)
    if getattr(sweep,'kind',None)!='ball' or not hasattr(sweep,'a_xyz'):return stats
    a,b=np.asarray(sweep.a_xyz),np.asarray(sweep.b_xyz)
    if a[2]!=b[2] or np.array_equal(a,b):return stats
    center=a+np.array([0.,0.,sweep.radius]);p=s.endpoints['positions']
    identity=s.material_surface_ids;patches=s.analytic_patch_ids
    rows=[];donors=[];ranges=[];spheres=[]
    for key in np.unique(patches[np.isin(identity,np.atleast_1d(current_ids))]):
        patch=s.analytic_patches[int(key)];e=patch.equation
        if patch.stock or not np.allclose(e[:3],center,rtol=0,atol=1e-9):continue
        if not np.allclose(e[3:12].reshape(3,3),np.eye(3),rtol=0,atol=1e-12):continue
        if np.any(e[12:15]) or abs(e[15]+sweep.radius**2)>1e-9:continue
        group=np.flatnonzero((patches==key)&np.isin(identity,np.atleast_1d(current_ids)))
        # Reject off-surface repairs: radial projection is valid for native
        # spherical GS, not arbitrary fragments labelled with that equation.
        on=np.abs(np.linalg.norm(p[group]-center,axis=1)-sweep.radius)<1e-8
        group=group[on]
        forward=group[(p[group]-center)@(b-a)>1e-9]
        selection=getattr(surface,'sweep_selection',None)
        if selection is not None:forward=forward[selection.mask[forward]]
        for row in forward:
            stable=group[identity[group]==identity[row]]
            reach=2*s.gaussian.attributes['scale'][row,0]+3*s.gaussian.attributes['scale'][stable,0]
            stable=stable[np.linalg.norm(p[stable]-p[row],axis=1)<=reach]
            ranges.append((len(donors),len(donors)+len(stable)));donors.extend(stable)
            rows.append(row);spheres.append(np.r_[center,sweep.radius])
    if not rows:return stats
    g=np.column_stack((p,s.gaussian.u,s.gaussian.v,s.gaussian.attributes['scale'][:,0]))
    clips=np.column_stack((s.clipping.planes,(s.material_clipping or s.clipping).planes))
    xy=disc(8,32);theta=np.pi/32
    holdout=xy@np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy=np.ascontiguousarray(np.r_[xy,holdout])
    with rt.context():
        if not hasattr(rt,'ball_continuation_kernel'):
            rt.ball_continuation_kernel=cp.RawKernel(CUDA,'cap_replace',options=('--fmad=false',))
        result=cp.zeros(len(p),cp.uint32)
        rec,spans,planes=(cp.asarray(v) for v in pack_cutters((sweep,))[:3])
        row_ids=cp.asarray(rows,dtype=cp.int64)
        stats['ball_continuation_candidates']=int(cp.count_nonzero(removed[row_ids]==0).item())
        rt.ball_continuation_kernel((len(rows),),(128,),(cp.asarray(g),cp.asarray(clips),row_ids,
            cp.asarray(donors,dtype=cp.int64),cp.asarray(ranges,dtype=cp.int64),np.int32(len(rows)),
            cp.asarray(spheres),cp.asarray(xy),np.int32(len(xy)),cp.asarray(np.asarray(stock_bounds,dtype=float).reshape(6)),
            np.float64(display_factor(display_scale)),rec,spans,planes,np.int32(len(rec)),removed,result))
        stats['ball_continuation_removed']=int(cp.count_nonzero(result).item())
        removed |= result
    stats['ball_continuation_ms']=(perf_counter()-started)*1000
    return stats
