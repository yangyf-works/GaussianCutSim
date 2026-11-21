"""Ordered retirement of published children, with geometry and opacity guards.

No uncovered-surface search or GS generation. Only explicitly supplied children
can be retired. Every donor is still present when a deletion is committed.
"""
from time import perf_counter
import numpy as np

OPACITY_CUDA = r'''
extern "C" __global__ void fragment_opacity_loss(const double* points,const double* normals,
 const double* tolerance,int count,const double* gs,int ng,int target,unsigned char* lost){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* p=points+3*i;const double* n=normals+3*i;
 double remaining=1.,alpha=0.;
 for(int j=0;j<ng;j++){
  const double* g=gs+19*j;double x=0.,y=0.,h=0.,dot=0.;
  for(int k=0;k<3;k++){
   double d=p[k]-g[k];x+=d*g[3+k];y+=d*g[6+k];h+=d*g[9+k];dot+=n[k]*g[9+k];
  }
  if(dot<.9961946980917455||fabs(h)>tolerance[i])continue;
  const double* c=g+12;double sigma=c[6];
  if(!(sigma>0.)||x*x+y*y>9.*sigma*sigma+1e-12)continue;
  if(c[0]*x+c[1]*y+c[2]<-1e-9||c[3]*x+c[4]*y+c[5]<-1e-9)continue;
  double a=.99*exp(-.5*(x*x+y*y)/(sigma*sigma));
  if(j==target)alpha=a;else remaining*=1.-a;
 }
 lost[i]=(alpha>0. && remaining*(1.-alpha)<.5+1e-7 && remaining>=.5-1e-7);
}
'''


def _witnesses(snapshot, rows, draw):
    """Fixed shared witness set: two staggered grids, rim, centre and clip corners."""
    angle=(np.arange(32)+.5)*2*np.pi/32
    radius=np.sqrt((np.arange(16)+.5)/16)
    grids=[]
    for shift in (0.,np.pi/32):
        grids.append((radius[:,None,None]*np.column_stack((np.cos(angle+shift),np.sin(angle+shift)))[None]).reshape(-1,2))
    rim=np.arange(128)*2*np.pi/128
    unit=np.vstack((*grids,np.column_stack((np.cos(rim),np.sin(rim))),[[0.,0.]]))
    points=[];normals=[];tolerance=[]
    for row in rows:
        c=draw[row];R=3*c[6]
        if not R>0:continue
        extras=[]
        for h in c[:6].reshape(2,3):
            length=np.linalg.norm(h[:2])
            if length<1e-12 or not np.isfinite(h[2]):continue
            normal=h[:2]/length;offset=h[2]/length
            if abs(offset)<=R:
                tangent=np.array([-normal[1],normal[0]])*np.sqrt(max(0.,R*R-offset*offset))
                extras.extend((-offset*normal+tangent,-offset*normal-tangent))
        lines=c[:6].reshape(2,3)
        if np.isfinite(lines[:,2]).all() and abs(np.linalg.det(lines[:,:2]))>1e-12:
            extras.append(np.linalg.solve(lines[:,:2],-lines[:,2]))
        xy=np.vstack((unit*R,np.asarray(extras).reshape(-1,2)))
        valid=np.sum(xy*xy,axis=1)<=R*R+1e-12
        for h in lines:valid &= xy@h[:2]+h[2]>=-1e-9
        xy=xy[valid]
        points.append(snapshot.endpoints['positions'][row]+xy[:,0,None]*snapshot.gaussian.u[row]+xy[:,1,None]*snapshot.gaussian.v[row])
        normals.append(np.tile(snapshot.endpoints['normals'][row],(len(xy),1)))
        tolerance.append(np.full(len(xy),min(.005,.04*float(snapshot.gaussian.attributes['scale'][row,0]))))
    if not points:return np.empty((0,3)),np.empty((0,3)),np.empty(0)
    return np.concatenate(points),np.concatenate(normals),np.concatenate(tolerance)


def prune(snapshot, rt, candidates, *, display_scale=1., detail=None):
    from cutsim.cutting.tool_profile_simulator import _select
    from cutsim.rendering.opaque_coverage import opaque_geometry
    from cutsim.rendering.surface_replacement import covered_rows
    started=perf_counter()
    rows=np.asarray(candidates,np.int64)
    result=np.zeros(len(rows),bool)
    stats=dict(residual_prune_candidates=len(rows),residual_prune_geometry_rejected=0,
               residual_prune_opacity_rejected=0,residual_prune_opacity_probes=0)
    if len(np.unique(rows))!=len(rows):raise ValueError('Duplicate fragment candidates')
    if len(rows):
        p=snapshot.endpoints['positions'];g=snapshot.gaussian
        if np.any(rows<0) or np.any(rows>=len(p)):raise ValueError('Invalid fragment candidate')
        sigma=g.attributes['scale'][:,0].astype(float)
        draw=None
        alive=np.ones(len(p),bool)
        # Smaller children first, deterministic row order for ties.
        for at in np.lexsort((rows,sigma[rows])):
            row=rows[at]
            # Conservative geometry proof checks BOTH certificates at 1.5/2 sigma.
            reach=2*(sigma+sigma[row])+.005
            near=alive & np.all(np.abs(p-p[row])<=reach[:,None],axis=1)
            near[row]=False
            donors=np.flatnonzero(near)
            local=np.r_[row,donors]
            pairs=np.column_stack((np.zeros(len(donors),np.int64),np.arange(1,len(local))))
            if not covered_rows(rt,_select(snapshot,local),[0],pairs)[0]:
                stats['residual_prune_geometry_rejected']+=1;continue
            if draw is None:
                draw=opaque_geometry(snapshot,display_scale)
                points,normals,tolerance=_witnesses(snapshot,rows,draw)
                stats['residual_prune_opacity_probes']=len(points)
            # All fixed witnesses that this child can affect, including other
            # children's grids. Ordered checks preserve earlier deletion checks.
            affected=np.all(np.abs(points-p[row])<=3*draw[row,6]+.006,axis=1)
            q=points[affected]
            if not len(q):
                stats['residual_prune_opacity_rejected']+=1;continue
            low=q.min(axis=0)-.006;high=q.max(axis=0)+.006
            radius=3*np.maximum(draw[:,6],0.)+1e-7
            ids=np.flatnonzero(alive & np.all((p+radius[:,None]>=low)&(p-radius[:,None]<=high),axis=1))
            geometry=np.ascontiguousarray(np.column_stack((p[ids],g.u[ids],g.v[ids],snapshot.endpoints['normals'][ids],draw[ids])))
            cp=rt.cp
            with rt.context():
                if not hasattr(rt,'fragment_opacity_loss_kernel'):
                    rt.fragment_opacity_loss_kernel=cp.RawKernel(OPACITY_CUDA,'fragment_opacity_loss',options=('--fmad=false',))
                lost=cp.empty(len(q),cp.uint8)
                rt.fragment_opacity_loss_kernel(((len(q)+127)//128,),(128,),
                    (cp.asarray(np.ascontiguousarray(q)),cp.asarray(np.ascontiguousarray(normals[affected])),
                     cp.asarray(tolerance[affected]),np.int32(len(q)),cp.asarray(geometry),np.int32(len(ids)),
                     np.int32(np.searchsorted(ids,row)),lost))
                rejected=bool(cp.any(lost).item())
            if rejected:stats['residual_prune_opacity_rejected']+=1;continue
            alive[row]=False;result[at]=True
    if detail is not None:detail.update(stats)
    return result,(perf_counter()-started)*1000
