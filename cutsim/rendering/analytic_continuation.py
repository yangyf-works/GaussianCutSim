"""Release obsolete round-end floor limits inside a certified planar extension.

Uses live generating-face metadata, not cutter replay. A circular floor's
inherited tangent is replaceable only when the new coplanar prism contains
its entire old clipped disk. Other boundaries remain mandatory. CUDA proves
linear containment from disk/line extrema and publishes at most two edges.
"""
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.clip_subdivision import CUDA as SUBDIVISION_CUDA

CUDA=SUBDIVISION_CUDA[:SUBDIVISION_CUDA.index('extern "C"')]+r'''
__device__ void candidate_min(const double* e,int n,double x,double y,double a,double b,double d,double* best){
 if(inside(e,n,x,y))*best=fmin(*best,a*x+b*y+d);
}
__device__ double minimum(const double* e,int n,double R,double a,double b,double d){
 double best=INF,L=hypot(a,b);
 candidate_min(e,n,0.,0.,a,b,d,&best);
 if(L>1e-12)candidate_min(e,n,-R*a/L,-R*b/L,a,b,d,&best);
 for(int i=0;i<n;i++){
  const double* h=e+3*i;double len=hypot(h[0],h[1]);if(len<1e-12||!isfinite(h[2]))continue;
  double nx=h[0]/len,ny=h[1]/len,c=h[2]/len;if(fabs(c)>R+1e-10)continue;
  double t=sqrt(fmax(0.,R*R-c*c));
  for(int sign=-1;sign<=1;sign+=2)candidate_min(e,n,-c*nx-sign*t*ny,-c*ny+sign*t*nx,a,b,d,&best);
  for(int j=0;j<i;j++){
   const double* q=e+3*j;double det=h[0]*q[1]-h[1]*q[0];if(!isfinite(q[2])||fabs(det)<1e-12)continue;
   double x=(h[1]*q[2]-q[1]*h[2])/det,y=(q[0]*h[2]-h[0]*q[2])/det;
   if(x*x+y*y<=R*R+1e-9)candidate_min(e,n,x,y,a,b,d,&best);
  }
 }
 return best;
}
extern "C" __global__ void continue_floor(const double* geometry,const double* oldclips,
 const double* circles,const double* planes,const long long* spans,int count,
 double* output,unsigned char* changed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* g=geometry+10*i;const double* circle=circles+3*i;double R=g[9];
 double domain[24];int nd=0;
 changed[i]=0;
 for(long long k=spans[2*i];k<spans[2*i+1];k++){
  const double* h=planes+4*k;double a=0.,b=0.,d=h[3];
  for(int j=0;j<3;j++){a+=h[j]*g[3+j];b+=h[j]*g[6+j];d+=h[j]*g[j];}
  double len=hypot(a,b);if(len<1e-10){if(d<-1e-8)return;continue;}
  if(nd>=8)return;domain[3*nd]=a/len;domain[3*nd+1]=b/len;domain[3*nd+2]=d/len;nd++;
 }
 double result[12];int removed=0;
 for(int side=0;side<2;side++){
  const double* old=oldclips+12*i+6*side;double retained[6];int nr=0;
  // Never discard any of the previously represented floor.
  for(int k=0;k<nd;k++){
   double value=minimum(old,2,R,domain[3*k],domain[3*k+1],domain[3*k+2]);
   if(!isfinite(value)||value<-1e-8)return;
  }
  for(int k=0;k<2;k++){
   const double* h=old+3*k;if(!isfinite(h[2]))continue;
   double ideal=circle[2]-h[0]*circle[0]-h[1]*circle[1];
   bool obsolete=fabs(hypot(h[0],h[1])-1.)<1e-8&&fabs(h[2]-ideal)<=.061*R+1e-8;
   if(obsolete)removed++;
   else{for(int j=0;j<3;j++)retained[3*nr+j]=h[j];nr++;}
  }
  int edges=reduce(retained,nr,domain,nd,0.,0.,R,result+6*side,true);
  if(edges<0||edges>2)return;
 }
 if(!removed)return;
 bool differs=false;for(int k=0;k<12;k++)if(result[k]!=oldclips[12*i+k])differs=true;
 if(!differs)return;
 for(int k=0;k<12;k++)output[12*i+k]=result[k];changed[i]=1;
}
'''


def refresh_planar_continuations(snapshot,nold,draw,material,retained=None):
    """Refresh only old circular-floor GS covered by a new same-ID prism face."""
    stats={'analytic_continuation_candidates':0,'analytic_continuation_refreshed':0}
    if not nold or nold==len(draw):return stats
    table=snapshot.analytic_patches;patchids=snapshot.analytic_patch_ids
    identity=snapshot.material_surface_ids;p=snapshot.endpoints['positions']
    radius=CLIP_SIGMA*snapshot.gaussian.attributes['scale'][:,0].astype(float)
    new=[]
    for key in np.unique(patchids[nold:]):
        patch=table[int(key)];e=patch.equation
        if patch.stock or np.any(e[3:12]) or abs(e[14])<1e-10 or np.any(e[12:14]):continue
        rec=patch.solid.packed[0][patch.part]
        if rec[13]:continue  # curvilinear/nonconvex domains need the full fitter
        h=patch.solid.packed[2][slice(*patch.solid.packed[1][patch.part]),:4]
        if len(h)>8:continue
        box=patch.solid.packed[3][patch.part]
        ids=np.unique(identity[nold:][patchids[nold:]==key])
        new.append((int(key),e,ids,h,box))
    if not new:return stats
    rows=[];new_ids=[];circles=[];planes=[];spans=[];offset=0
    for key in np.unique(patchids[:nold]):
        patch=table[int(key)];e=patch.equation;records=patch.solid.packed[0]
        if patch.stock or len(records)!=1 or np.any(e[3:12]) or abs(e[14])<1e-10 or np.any(e[12:14]):continue
        rec=records[0]
        if not rec[13] or not np.allclose(rec[3:12].reshape(3,3),np.diag([1.,1.,0.]),atol=1e-12,rtol=0):continue
        active=(patchids[:nold]==key)&np.any(np.isfinite(draw[:nold,[2,5]]),axis=1)
        if retained is not None:active &= retained[:nold]
        old=np.flatnonzero(active)
        for new_key,eq,ids,h,box in new:
            if not np.allclose(e,eq,atol=1e-9,rtol=0):continue
            selected=old[np.isin(identity[old],ids)&np.all((p[old]+radius[old,None]>=box[:3])&(p[old]-radius[old,None]<=box[3:]),axis=1)]
            if not len(selected):continue
            centre=rec[:3]-p[selected];u=snapshot.gaussian.u[selected];v=snapshot.gaussian.v[selected]
            rows.extend(selected);new_ids.extend([new_key]*len(selected));circles.extend(np.column_stack((np.sum(centre*u,axis=1),np.sum(centre*v,axis=1),np.full(len(selected),rec[12]))))
            planes.append(h);spans.extend([(offset,offset+len(h))]*len(selected));offset+=len(h)
    if not rows:return stats
    from cutsim.cuda.profile_dexel_cuda import runtime
    rt=runtime();cp=rt.cp;rows=np.asarray(rows,np.int64)
    g=np.column_stack((p[rows],snapshot.gaussian.u[rows],snapshot.gaussian.v[rows],np.minimum(radius[rows],np.minimum(draw[rows,6],material[rows,6]))))
    old=np.column_stack((draw[rows,:6],material[rows,:6]))
    with rt.context():
        if not hasattr(rt,'analytic_continuation_kernel'):rt.analytic_continuation_kernel=cp.RawKernel(CUDA,'continue_floor',options=('--fmad=false',))
        output=cp.empty_like(cp.asarray(old));changed=cp.empty(len(rows),cp.uint8)
        rt.analytic_continuation_kernel(((len(rows)+127)//128,),(128,),
            (cp.asarray(g),cp.asarray(old),cp.asarray(np.asarray(circles)),cp.asarray(np.concatenate(planes)),cp.asarray(np.asarray(spans,np.int64)),np.int32(len(rows)),output,changed))
        selected=cp.flatnonzero(changed);indices=cp.asnumpy(selected,stream=rt.stream);values=cp.asnumpy(output[selected],stream=rt.stream)
    # Deterministic if multiple equivalent new prism faces certify a row.
    _,first=np.unique(rows[indices],return_index=True);indices=indices[first];values=values[first]
    draw[rows[indices],:6]=values[:,:6];material[rows[indices],:6]=values[:,6:]
    if len(indices):
        # The expanded support is now certified by the new finite planar face.
        # Keeping its old circular domain would lose later peer intersections.
        from cutsim.rendering.analytic_surfaces import set_metadata
        updated=patchids.copy();updated[rows[indices]]=np.asarray(new_ids,np.int64)[indices]
        set_metadata(snapshot,updated,table)
    stats.update(analytic_continuation_candidates=len(rows),analytic_continuation_refreshed=len(indices))
    return stats
