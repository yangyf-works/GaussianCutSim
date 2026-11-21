"""Certify trimming of affected old GS, independently of residual samples.

Every removed polygon is certified inside the current cutter. Replacing an
occupied clipping slot additionally requires containment in the previous
domain, so an older cut can never be undone to make room for a new edge.
No deletion quota, larger cutter, resized GS or relaxed coverage tolerance.
"""
import numpy as np
from cutsim.rendering.finite_domain_retirement import CUDA as DOMAIN_CUDA

CUDA=DOMAIN_CUDA+r'''
__device__ bool removable_half(const double* a,double R,double nx,double ny,double offset,
 const long long* m,const long long* spans,const double* planes,const double* stock,
 const double* cr,const long long* cs,const double* cp,int parts){
 for(int side=0;side<2;side++){
  bool present=false;
  for(long long part=m[0];part<m[0]+m[1];part++){
   double polygon[2*DOMAIN_VERTICES];
   int n=finite_polygon(a,side,R,-1,spans+2*part,planes,stock,polygon);
   if(n<0)return false;if(!n)continue;
   n=domain_clip(polygon,n,-nx,-ny,offset);
   if(n<0)return false;if(!n)continue;
   present=true;
   if(!polygon_cut(a,polygon,n,cr,cs,cp,parts))return false;
  }
  if(!present)return false;
 }
 return true;
}
__device__ int compatible_slot(const double* a,int side,double R,double nx,double ny,double offset,
 const long long* m,const long long* spans,const double* planes,const double* stock){
 for(int slot=0;slot<6;slot+=3)
  if(isinf(a[13+7*side+slot+2])&&a[13+7*side+slot+2]>0.)return slot;
 for(int slot=0;slot<6;slot+=3){
  double trial[27];for(int k=0;k<27;k++)trial[k]=a[k];
  double* h=trial+13+7*side+slot;h[0]=nx;h[1]=ny;h[2]=-offset;
  const double* old=a+13+7*side+slot;bool contained=true;
  // Check the entire proposed footprint, not just its sampled witnesses.
  // The other old edge stays in place; the replaced edge must be redundant.
  for(long long part=m[0];part<m[0]+m[1]&&contained;part++){
   double polygon[2*DOMAIN_VERTICES];
   int n=finite_polygon(trial,side,R,-1,spans+2*part,planes,stock,polygon);
   if(n<0){contained=false;break;}
   double eps=1e-9*(1.+fabs(old[2])+R*hypot(old[0],old[1]));
   for(int j=0;j<n;j++)if(old[0]*polygon[2*j]+old[1]*polygon[2*j+1]+old[2]<-eps){contained=false;break;}
  }
  if(contained)return slot;
 }
 return -1;
}
extern "C" __global__ void trim_empty_residual(const double* g,const long long* rows,int count,
 const long long* patches,const long long* meta,const long long* spans,const double* planes,
 const double* stock,const double* cr,const long long* cs,const double* cp,int parts,
 double* output,unsigned char* changed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 changed[i]=0;const double* a=g+27*rows[i];double R=2*a[12];
 if(!(R>0.)||!isfinite(R)||!parts)return;
 int slots[2]={-1,-1};
 const long long* m=meta+4*patches[rows[i]];
 double best=0.,bx=0.,by=0.;
 for(int angle=0;angle<36;angle++){
  double t=angle*6.283185307179586/32.,nx=cos(t),ny=sin(t);
  if(angle>=32){
   int side=(angle-32)/2,slot=3*((angle-32)%2);const double* old=a+13+7*side+slot;
   double len=hypot(old[0],old[1]);if(!isfinite(old[2])||len<1e-12)continue;
   nx=old[0]/len;ny=old[1]/len;
  }
  if(!removable_half(a,R,nx,ny,best,m,spans,planes,stock,cr,cs,cp,parts))continue;
  double lo=best,hi=R;
  for(int step=0;step<12;step++){
   double mid=.5*(lo+hi);
   if(removable_half(a,R,nx,ny,mid,m,spans,planes,stock,cr,cs,cp,parts))lo=mid;else hi=mid;
  }
  if(lo>best){
   int first=compatible_slot(a,0,R,nx,ny,lo,m,spans,planes,stock);
   int second=compatible_slot(a,1,R,nx,ny,lo,m,spans,planes,stock);
   if(first>=0&&second>=0){best=lo;bx=nx;by=ny;slots[0]=first;slots[1]=second;}
  }
 }
 if(best<=1e-8*(1.+R))return;
 for(int side=0;side<2;side++){
  for(int k=0;k<6;k++)output[12*i+6*side+k]=a[13+7*side+k];
  double* h=output+12*i+6*side+slots[side];h[0]=bx;h[1]=by;h[2]=-best;
 }
 changed[i]=1;
}
'''


def trim_cut_interior(rt,snapshot,rows,draw,material,domain,*,cache=None):
    """Update draw/material copies only where a current-cut certificate succeeds."""
    if not len(rows) or domain is None:return 0
    cp=rt.cp;g=snapshot.gaussian
    geometry=np.ascontiguousarray(np.column_stack((snapshot.endpoints['positions'],g.u,g.v,
        snapshot.endpoints['normals'],g.attributes['scale'][:,0],draw,material)),dtype=float)
    # The caller owns this cache for ONE sweep/stock domain only. Match the
    # complete input certificate and immutable finite source patch, not row
    # numbers: publication can compact rows, and supplemental GS can change an
    # old row's fitted edges. Both positive and negative proofs are reusable.
    keys=None;reused=0;hits=0
    if cache is not None:
        keys=[(snapshot.analytic_patches[int(snapshot.analytic_patch_ids[row])],
               *geometry[row]) for row in rows]
        pending=[];pending_keys=[]
        for row,key in zip(rows,keys):
            if key not in cache:
                pending.append(row);pending_keys.append(key);continue
            hits+=1
            value=cache[key]
            if value is not None:
                draw[row,:6]=value[:6];material[row,:6]=value[6:];reused+=1
        rows=np.asarray(pending,np.int64);keys=pending_keys
        cache['reused']=cache.get('reused',0)+hits
        if not len(rows):return reused
    with rt.context():
        if not hasattr(rt,'trim_empty_residual_kernel'):
            rt.trim_empty_residual_kernel=cp.RawKernel(CUDA,'trim_empty_residual',options=('--fmad=false',))
        output=cp.empty((len(rows),12),cp.float64);changed=cp.zeros(len(rows),cp.uint8)
        rt.trim_empty_residual_kernel(((len(rows)+63)//64,),(64,),
            (cp.asarray(geometry),cp.asarray(rows,dtype=cp.int64),np.int32(len(rows)),*domain,output,changed))
        good=cp.asnumpy(changed,stream=rt.stream).astype(bool)
        values=cp.asnumpy(output[good],stream=rt.stream)
    if cache is not None:
        for key in keys:cache[key]=None
        for key,value in zip((key for key,yes in zip(keys,good) if yes),values):
            cache[key]=value.copy()
    draw[np.asarray(rows)[good],:6]=values[:,:6]
    material[np.asarray(rows)[good],:6]=values[:,6:]
    return reused+int(good.sum())


def affected_old_rows(rt,region):
    """Select all current-sweep candidates, not just empty residual grids."""
    if not hasattr(region,'sweep_data'):return np.empty(0,np.int64)
    cp=rt.cp
    with rt.context():
        rows=region.rows
        return cp.asnumpy(rows[(rows<region.preserve_nonempty)&(region.sweep_data[0][rows]!=0)],stream=rt.stream)
