"""Keep child-ray preparation and material admission resident until publication.

The CUDA field is the same finite sweep field used by sweep.sdf, including
candidate-time minimisation for CuttingSweep. It is not a primitive-union
boundary test: internal primitive partitions must never become new surfaces.
"""
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from cutsim.cuda.adaptive_profile import CHARTS


CUDA = r'''
#include <math_constants.h>
#define INFINITY CUDART_INF
__device__ double clamp_time(double t){return fmin(1.,fmax(0.,t));}
__device__ double flat_field(double radial,double z,double lo,double hi){
 double axial=fabs(z-(lo+hi)*.5)-(hi-lo)*.5;
 return hypot(fmax(radial,0.),fmax(axial,0.))+fmin(fmax(radial,axial),0.);
}
__device__ double sweep_field(const double* world,const double* s){
 // Record: local a,b,r,low,high,kind(0 flat,1 ball,2 finite),ball,
 // frame origin, frame basis (row major).
 double p[3],q[3],d[3];
 for(int k=0;k<3;k++){
  p[k]=0.;for(int j=0;j<3;j++)p[k]+=(world[j]-s[11+j])*s[14+3*j+k];
  q[k]=p[k]-s[k];d[k]=s[3+k]-s[k];
 }
 double r=s[6];int kind=(int)s[9];
 if(kind<2){
  double den=d[0]*d[0]+d[1]*d[1];
  double t=clamp_time((q[0]*d[0]+q[1]*d[1])/fmax(den,1e-30));
  double x=p[0]-(s[0]+t*d[0]),y=p[1]-(s[1]+t*d[1]);
  double radial=sqrt(x*x+y*y)-r;
  double value=flat_field(radial,p[2],s[7],s[8]);
  if(kind==1){double z=p[2]-s[2];value=fmin(value,sqrt((x*x+y*y)+z*z)-r);}
  return value;
 }
 double times[16];int n=0;times[n++]=0.;times[n++]=1.;
 double low=s[7]-q[2],high=q[2]-s[8];bool ball=s[10]!=0.;
 if(d[2]!=0.)times[n++]=(q[2]-.5*(s[7]+s[8]))/d[2];
 if(ball&&d[2]!=0.)times[n++]=(q[2]-r)/d[2];
 for(int form=0;form<(ball?2:1);form++){
  double z=form?q[2]-r:0.,dz=form?d[2]:0.;
  double den=(d[0]*d[0]+d[1]*d[1])+dz*dz;
  double dot=(q[0]*d[0]+q[1]*d[1])+z*dz;
  double norm=(q[0]*q[0]+q[1]*q[1])+z*z;
  if(den!=0.)times[n++]=dot/den;
  for(int side=0;side<2;side++){
   double slope=side?-d[2]:d[2],offset=side?high:low;
   double aa=den-slope*slope,bb=-2.*(dot+slope*(r+offset));
   double cc=norm-(r+offset)*(r+offset),first,second;
   if(fabs(aa)<=1e-14*fmax(den+slope*slope,1e-30)){
    first=second=bb!=0.?-cc/bb:0.;
   }else{
    double disc=bb*bb-4.*aa*cc;
    double stable=-.5*(bb+copysign(sqrt(fmax(disc,0.)),bb));
    first=disc>=0.?stable/aa:0.;
    second=disc>=0.?(stable!=0.?cc/stable:-bb/(2.*aa)):0.;
   }
   times[n++]=first;times[n++]=second;
  }
 }
 double result=INFINITY;
 for(int j=0;j<n;j++){
  double t=clamp_time(times[j]),x=q[0]-t*d[0],y=q[1]-t*d[1],z=q[2]-t*d[2];
  double rz=ball?fmin(z-r,0.):0.;
  result=fmin(result,fmax(sqrt((x*x+y*y)+rz*rz)-r,fmax(s[7]-z,z-s[8])));
 }
 return result;
}
extern "C" __global__ void adaptive_prepare(const double* origins,const double* dirs,
 const int* charts,const double* pitches,const double* distances,int count,
 const double* params,double* packet,unsigned char* valid){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 // params: field record 23, world a 3, feed 3, axis 3, ball, box 6,
 // stock box 6, margin (negative means the ray pitch).
 const double* a=params+23;const double* feed=params+26;const double* axis=params+29;
 const double* box=params+33;const double* stock=params+39;
 double o[3],d[3],point[3],t=distances[i];double* out=packet+7*i;
 for(int k=0;k<3;k++){o[k]=origins[3*i+k];d[k]=dirs[3*i+k];point[k]=o[k]+t*d[k];}
 if(params[32]!=0.){
  int chart=charts[i];
  if(chart==2||chart==3||chart==6){
   double dot=0.,length=0.;for(int k=0;k<3;k++){dot+=(point[k]-a[k])*feed[k];length+=feed[k]*feed[k];}
   double along=clamp_time(dot/fmax(length,1e-30)),on_axis[3],height=0.;
   for(int k=0;k<3;k++){on_axis[k]=a[k]+along*feed[k];height+=(o[k]-on_axis[k])*axis[k];}
   for(int k=0;k<3;k++)o[k]=on_axis[k]+height*axis[k];
  }
  double norm=0.;for(int k=0;k<3;k++){d[k]=point[k]-o[k];norm+=d[k]*d[k];}
  t=sqrt(norm);for(int k=0;k<3;k++)d[k]/=fmax(t,1e-30);
 }
 double lo=0.,hi=t+(params[45]<0.?pitches[i]:params[45]);bool ok=true;
 for(int k=0;k<3;k++){
  bool parallel=fabs(d[k])<1e-15;double denom=parallel?1.:d[k];
  double u=(box[k]-o[k])/denom,v=(box[3+k]-o[k])/denom;
  lo=fmax(lo,parallel?-INFINITY:fmin(u,v));hi=fmin(hi,parallel?INFINITY:fmax(u,v));
  ok &= !parallel||(o[k]>=box[k]&&o[k]<=box[3+k]);
 }
 if(!ok)hi=lo-1.;
 ok=t>=lo&&t<=hi;
 for(int k=0;k<3;k++){
  point[k]=o[k]+t*d[k];ok &= point[k]>=stock[k]&&point[k]<=stock[3+k];
  out[k]=d[k];out[3+k]=point[k];
 }
 ok &= fabs(sweep_field(point,params))<=1e-7;
 out[6]=0.;valid[i]=ok;
}
extern "C" __global__ void adaptive_publish(const double* origins,const double* dirs,
 const int* charts,const double* cells,const double* radii,const double* pitches,
 const double* bounds,const double* distances,const double* feeds,const long long* parents,
 const double* packet,int count,unsigned char* bytes){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 // Aligned structure-of-arrays regions: 18 doubles, int64 parent,
 // int32 chart, int8 decision per child. No centred/query vectors read back.
 double* out=(double*)bytes+18*i;
 for(int k=0;k<3;k++){out[k]=origins[3*i+k];out[3+k]=dirs[3*i+k];out[15+k]=feeds[3*i+k];}
 for(int k=0;k<4;k++)out[6+k]=cells[4*i+k];
 out[10]=radii[i];out[11]=pitches[i];out[12]=bounds[2*i];out[13]=bounds[2*i+1];out[14]=distances[i];
 ((long long*)(bytes+144LL*count))[i]=parents[i];
 ((int*)(bytes+152LL*count))[i]=charts[i];
 ((signed char*)(bytes+156LL*count))[i]=(signed char)packet[7*i+6];
}
'''


def sweep_record(sweep):
    """Small CPU constants only; no query or ray arrays cross to the host."""
    from cutsim.core.cutting_sweep import CuttingSweep
    from cutsim.core.sweep import FlatSweep, BallSweep
    local = getattr(sweep, 'local', sweep)
    origin = getattr(sweep, 'origin', np.zeros(3))
    basis = getattr(sweep, 'basis', np.eye(3))
    if isinstance(local, CuttingSweep):
        record = np.r_[local.a_xyz, local.b_xyz, local.radius,
            local.cutting_start, local.cutting_end, 2., float(local.kind == 'ball')]
    elif isinstance(local, (FlatSweep, BallSweep)):
        ball = isinstance(local, BallSweep)
        z = local.center_z if ball else 0.
        record = np.r_[local.a_xy, z, local.b_xy, z, local.radius,
            z if ball else local.z_lo, local.z_hi, float(ball), float(ball)]
    else:
        raise TypeError(f'Unsupported adaptive finite sweep: {type(local)}')
    return np.r_[record, origin, basis.ravel()]


def prepare_device(surface, raw, sweep, box, stock_bounds, ray_margin):
    rt = surface.rt; cp = rt.cp
    a = np.asarray(sweep.a_xyz if hasattr(sweep, 'a_xyz') else [*sweep.a_xy, sweep.z_lo])
    b = np.asarray(sweep.b_xyz if hasattr(sweep, 'b_xyz') else [*sweep.b_xy, sweep.z_lo])
    axis = np.asarray(sweep.basis[:, 2] if hasattr(sweep, 'local') else [0., 0., 1.])
    feed = b-a; feed = feed-axis*(feed@axis)
    stock = (np.full(3, -np.inf), np.full(3, np.inf)) if stock_bounds is None else stock_bounds
    params = np.r_[sweep_record(sweep), a, feed, axis, float(sweep.kind == 'ball'),
                   box[0], box[1], stock[0], stock[1], -1. if ray_margin is None else ray_margin]
    if not hasattr(rt, 'adaptive_prepare_kernel'):
        rt.adaptive_prepare_kernel = cp.RawKernel(CUDA, 'adaptive_prepare', options=('--fmad=false',))
    count = len(raw.origins)
    packet = cp.empty((count, 7), cp.float64); valid = cp.empty(count, cp.uint8)
    if count:
        rt.adaptive_prepare_kernel(((count+127)//128,), (128,),
            (raw.origins, raw.directions, raw.charts, raw.pitches, raw.surface_distance, np.int32(count),
             cp.asarray(params), packet, valid))
    return packet, cp.flatnonzero(valid)


def generate_tool_gs_device(surface, raw, source, command, sweep, block, group,
                            material_group, box, ray_offset, ray_margin=None, *, stock_bounds=None,
                            evaluated_packet=None):
    """One final packed readback for GS fitting and replay, after GPU admission."""
    from cutsim.cutting.tool_profile_state import RAY_STRIDE
    from cutsim.cuda.material_evidence import finite_material_evidence_device
    from cutsim.cuda.profile_program import _snapshot
    from cutsim.cuda.center_ray_update import center_grid
    if ray_offset < 0 or ray_offset+len(raw.origins) > RAY_STRIDE:
        raise ValueError('Additional rays exceed the block ray-ID range')
    started = perf_counter(); rt = surface.rt; cp = rt.cp
    with rt.context():
        if evaluated_packet is None:
            packet, candidates = prepare_device(surface, raw, sweep, box, stock_bounds, ray_margin)
            evidence, _, _ = finite_material_evidence_device(surface,
                packet[candidates, 3:6], packet[candidates, :3])
            # -2 marks points rejected by envelope/ray/stock tests, not material air.
            packet[:, 6] = -2
            packet[candidates, 6] = evidence
            candidate_count = len(candidates)
        else:
            # Already classified and compacted on the GPU by the cell vote.
            packet = evaluated_packet
            if packet.shape != (len(raw.origins), 7):
                raise ValueError('Evaluated packet must match the published rays')
            candidate_count = len(packet)
        if not hasattr(rt, 'adaptive_publish_kernel'):
            rt.adaptive_publish_kernel = cp.RawKernel(CUDA, 'adaptive_publish', options=('--fmad=false',))
        count = len(raw.origins); payload = cp.empty(count*157, cp.uint8)
        if count:
            rt.adaptive_publish_kernel(((count+127)//128,), (128,),
                (raw.origins, raw.directions, raw.charts, raw.cell_bounds, raw.radii, raw.pitches,
                 raw.ray_bounds, raw.surface_distance, raw.feeds, source, packet, np.int32(count), payload))
        host = surface._download(payload)
    values = host[:144*count].view(np.float64).reshape(count,18)
    parents = host[144*count:152*count].view(np.int64).copy()
    charts = host[152*count:156*count].view(np.int32)
    states = host[156*count:].view(np.int8)
    selected = np.flatnonzero(states == 1)
    # CPU publication already needs ray metadata for GS sizing and replay.
    # Reuse its canonical arithmetic to preserve existing snapshot coordinates;
    # it performs no material/Sweep queries and uploads no child arrays.
    descriptor = center_grid(SimpleNamespace(origins=values[:,:3].copy(), directions=values[:,3:6].copy(),
        charts=CHARTS[charts], cell_bounds=values[:,6:10].copy(), radii=values[:,10].copy(),
        pitches=values[:,11].copy(), ray_bounds=values[:,12:14].copy(), surface_distance=values[:,14].copy(),
        feeds=values[:,15:18].copy()), sweep, box, margin=ray_margin)
    del descriptor.center_rays
    points = descriptor.origins[selected]+descriptor.surface_distance[selected,None]*descriptor.directions[selected]
    ep = dict(positions=points, normals=descriptor.tool_normals[selected],
        ray_ids=selected, distances=descriptor.surface_distance[selected],
        owners=np.zeros(len(selected), np.int64), tool_ids=np.full(len(selected), sweep.tool_id, np.int32))
    new = _snapshot(ep, descriptor, [command], surface_groups=np.array([group]))
    new.material_surface_ids = new.gaussian.attributes['surface_id'][:, 0]+3*(material_group-group)
    new.endpoints['owners'] = np.full(len(selected), block, np.int64)
    new.endpoints['ray_ids'] = selected+ray_offset+(block+1)*RAY_STRIDE
    for array in (*new.endpoints.values(), new.material_surface_ids):
        array.flags.writeable = False
    stats = dict(adaptive_analytic_candidates=candidate_count,
        adaptive_analytic_air=int(np.count_nonzero(states == -1)),
        adaptive_analytic_unknown=int(np.count_nonzero(states == 0)),
        adaptive_publication_readback_bytes=host.nbytes, adaptive_intermediate_readback_bytes=0,
        adaptive_total_ms=(perf_counter()-started)*1000)
    return new, descriptor, parents, stats
