"""Addition-only material evidence from finite analytic faces on GPU.

Evaluate each face polynomial along a Dexel, split at its extremum, and
refine only sign-changing intervals. No quadratic root formula or face/face
intersection is used. Live GS footprints bound the available face evidence;
missing coverage is UNKNOWN, never an extrapolated infinite material solid.
Witness rows returned here are associations, NOT finite-disc hits.
"""
from time import perf_counter
import numpy as np


CUDA = r'''
__device__ bool ar_inside_sweep(const double* p,const double* rec,const long long* spans,
 const double* planes,int parts){
 for(int part=0;part<parts;part++){
  bool inside=true;
  for(long long j=spans[2*part];j<spans[2*part+1];j++){
   const double* h=planes+5*j;
   if(dot(p,h)+h[3]<-1e-8*(1.+fabs(h[3]))){inside=false;break;}
  }
  const double* r=rec+14*part;
  if(inside&&r[13]){
   double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
   for(int j=0;j<3;j++)for(int k=0;k<3;k++)q+=d[j]*r[3+3*j+k]*d[k];
   if(q>r[12]*r[12]+1e-8*(1.+r[12]*r[12]))inside=false;
  }
  if(inside)return true;
 }return false;
}
__device__ bool ar_box(const double* o,const double* d,const double* b,double* lo,double* hi){
 if(b[0]>b[3])return false;
 for(int k=0;k<3;k++){
  if(fabs(d[k])<1e-15){if(o[k]<b[k]||o[k]>b[k+3])return false;}
  else{double a=(b[k]-o[k])/d[k],c=(b[k+3]-o[k])/d[k];
   *lo=fmax(*lo,fmin(a,c));*hi=fmin(*hi,fmax(a,c));if(*lo>*hi)return false;}
 }return true;
}
__device__ double ar_value(double a,double b,double c,double t){return (a*t+b)*t+c;}
__device__ bool ar_domain(const double* p,const double* g,const long long* m,
 const double* rec,const long long* spans,const double* planes){
 double d[3]={p[0]-g[0],p[1]-g[1],p[2]-g[2]},x=dot(d,g+3),y=dot(d,g+6),R=g[12];
 if(x*x+y*y>R*R+1e-12||fabs(dot(d,g+9))>R)return false;
 if(g[13]*x+g[14]*y+g[15]<-1e-10||g[16]*x+g[17]*y+g[18]<-1e-10)return false;
 // The generating finite face, not its infinite extension.
 long long part=m[2];
 for(long long j=spans[2*part];j<spans[2*part+1];j++)
  if(dot(p,planes+5*j)+planes[5*j+3]<-1e-8)return false;
 const double* r=rec+14*part;
 if(r[13]){double z[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
  for(int j=0;j<3;j++)for(int k=0;k<3;k++)q+=z[j]*r[3+3*j+k]*z[k];
  if(q>r[12]*r[12]+1e-8*(1.+r[12]*r[12]))return false;
 }return true;
}
// Returns the nearest oriented boundary, retaining entry/contact conflicts.
__device__ int ar_nearest(const double* o,const double* d,const double* g,const double* boxes,
 const long long* order,const long long* source,long long n,long long leaves,
 const long long* ids,const double* eq,const long long* meta,const double* rec,
 const long long* spans,const double* planes,double* distance,long long* witness,
 const double* cr=nullptr,const long long* cs=nullptr,const double* ch=nullptr,int parts=0){
 double best=INFINITY;int state=0,stack[64],top=0;stack[top++]=1;*witness=-1;
 while(top){int node=stack[--top];double lo=0.,hi=best+1e-8;
  if(!ar_box(o,d,boxes+6*node,&lo,&hi))continue;
  if(node<leaves){
   double la=0.,ha=hi,lb=0.,hb=hi;int a=2*node,b=a+1;
   bool va=ar_box(o,d,boxes+6*a,&la,&ha),vb=ar_box(o,d,boxes+6*b,&lb,&hb);
   if(va&&vb){if(la<lb){stack[top++]=b;stack[top++]=a;}else{stack[top++]=a;stack[top++]=b;}}
   else if(va)stack[top++]=a;else if(vb)stack[top++]=b;
   continue;
  }
  long long leaf=node-leaves;if(leaf>=n)continue;long long row=order[leaf];
  const double* a=g+19*row;long long patch=ids[row];const double* e=eq+16*patch;
  double z[3]={o[0]-e[0],o[1]-e[1],o[2]-e[2]},A=0.,B=dot(e+12,d),C=dot(e+12,z)+e[15];
  for(int j=0;j<3;j++)for(int k=0;k<3;k++){
   A+=d[j]*e[3+3*j+k]*d[k];B+=(d[j]*z[k]+z[j]*d[k])*e[3+3*j+k];C+=z[j]*e[3+3*j+k]*z[k];}
  double cuts[3]={lo,hi,hi};int pieces=1;
  if(fabs(A)>1e-20){double vertex=-B/(2*A);if(vertex>lo&&vertex<hi){cuts[1]=vertex;pieces=2;}}
  for(int k=0;k<pieces;k++){
   double l=cuts[k],h=cuts[k+1],fl=ar_value(A,B,C,l),fh=ar_value(A,B,C,h);
   if((fl>0.&&fh>0.)||(fl<0.&&fh<0.)||(fl==0.&&fh==0.))continue;
   // Tangency does not toggle material. Split-at-extremum also detects
   // two crossings with equal signs at the original interval endpoints.
   double t;
   if(fl==0.)t=l;else if(fh==0.)t=h;
   else if(fabs(A)<1e-20)t=l-fl*(h-l)/(fh-fl);
   else{for(int iteration=0;iteration<36;iteration++){
     double mid=.5*(l+h),fm=ar_value(A,B,C,mid);
     if((fm<0.)==(fl<0.)){l=mid;fl=fm;}else h=mid;
    }t=.5*(l+h);}
   double p[3],grad[3],len=0.;for(int j=0;j<3;j++)p[j]=o[j]+t*d[j];
   for(int j=0;j<3;j++){grad[j]=e[12+j];for(int v=0;v<3;v++)grad[j]+=(e[3+3*j+v]+e[3+3*v+j])*(p[v]-e[v]);len+=grad[j]*grad[j];}
   len=sqrt(len);double orientation=meta[4*patch+3]?1.:-1.;
   if(len<1e-12||orientation*dot(grad,a+9)<.5*len)continue;
   double slope=orientation*dot(grad,d)/len;if(fabs(slope)<1e-10)continue;
   if(!ar_domain(p,a,meta+4*patch,rec,spans,planes))continue;
   if(parts&&!ar_inside_sweep(p,cr,cs,ch,parts))continue;
   int next=(t>1e-8&&slope>0.)?1:-1;
   if(t<best-1e-8){best=t;state=next;*witness=source[row];}
   else if(t<=best+1e-8){best=fmin(best,t);
    if(next<state||(next==state&&source[row]<*witness)){state=next;*witness=source[row];}}
  }
 }*distance=best;return state;
}
// A local live face can prove air without following a ray to the stock back
// wall: points inside its finite generating cutter were already removed.
// This is rejection only, never an inference that unreferenced space is solid.
__device__ long long ar_local_air(const double* p,const double* g,const double* boxes,
 const long long* order,const long long* source,long long n,long long leaves,const long long* ids,
 const double* eq,const long long* meta,const double* rec,const long long* spans,const double* planes){
 int stack[64],top=0;stack[top++]=1;
 while(top){int node=stack[--top];const double* box=boxes+6*node;
  if(p[0]<box[0]||p[0]>box[3]||p[1]<box[1]||p[1]>box[4]||p[2]<box[2]||p[2]>box[5])continue;
  if(node<leaves){stack[top++]=2*node+1;stack[top++]=2*node;continue;}
  long long leaf=node-leaves;if(leaf>=n)continue;long long row=order[leaf],patch=ids[row];
  const double* e=eq+16*patch;double z[3]={p[0]-e[0],p[1]-e[1],p[2]-e[2]},f=dot(e+12,z)+e[15],grad[3],len=0.;
  for(int j=0;j<3;j++){grad[j]=e[12+j];for(int k=0;k<3;k++){
   f+=z[j]*e[3+3*j+k]*z[k];grad[j]+=(e[3+3*j+k]+e[3+3*k+j])*z[k];}len+=grad[j]*grad[j];}
  len=sqrt(len);if(len<1e-12)continue;
  bool contact=fabs(f)<=1e-10*len;
  bool cut_interior=!meta[4*patch+3]&&f<-1e-8*len;
  if((contact||cut_interior)&&ar_domain(p,g+19*row,meta+4*patch,rec,spans,planes))return source[row];
 }return -1;
}
extern "C" __global__ void analytic_evidence(const double* points,const double* directions,int count,
 const double* g,const double* boxes,const long long* order,const long long* source,long long n,long long leaves,
 const long long* ids,const double* eq,const long long* meta,const double* rec,const long long* spans,
 const double* planes,signed char* result,long long* witnesses,unsigned char* fallback,
 const double* cr,const long long* cs,const double* ch,int parts){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 result[i]=0;fallback[i]=0;witnesses[2*i]=witnesses[2*i+1]=-1;if(!n)return;
 const double* p=points+3*i;double solid=INFINITY,air=INFINITY;long long left=-1,right=-1,air_row=-1;
 if(parts&&!ar_inside_sweep(p,cr,cs,ch,parts))return;
 long long local=ar_local_air(p,g,boxes,order,source,n,leaves,ids,eq,meta,rec,spans,planes);
 if(local>=0){result[i]=-1;witnesses[2*i]=local;return;}
 // First evaluate the actual Dexel. Only an unresolved pair needs axis probes.
 for(int pair=0;pair<4;pair++){
  double d[3]={0.,0.,0.};
  if(pair==0){for(int k=0;k<3;k++)d[k]=directions[3*i+k];}
  else{d[(pair+1)%3]=1.;fallback[i]=1;}
  if(dot(d,d)<1e-20)continue;
  double da,db;long long wa,wb;
  int a=ar_nearest(p,d,g,boxes,order,source,n,leaves,ids,eq,meta,rec,spans,planes,&da,&wa,cr,cs,ch,parts);
  for(int k=0;k<3;k++)d[k]=-d[k];
  int b=ar_nearest(p,d,g,boxes,order,source,n,leaves,ids,eq,meta,rec,spans,planes,&db,&wb,cr,cs,ch,parts);
  if(a<0&&da<air){air=da;air_row=wa;}if(b<0&&db<air){air=db;air_row=wb;}
  if(a>0&&b>0&&fmin(da,db)<solid){solid=fmin(da,db);left=wa;right=wb;}
  // In sweep-local mode the nearest oriented crossing inside the sweep
  // labels the adjacent interval. No remote opposite boundary is required.
  if(parts&&a>0&&da<solid){solid=da;left=wa;right=-1;}
  if(parts&&b>0&&db<solid){solid=db;left=-1;right=wb;}
  if(pair==0&&a!=0&&b!=0)break;
 }
 result[i]=solid<air-1e-8?1:(isfinite(air)?-1:0);
 if(result[i]==1){witnesses[2*i]=left;witnesses[2*i+1]=right;}
 else if(result[i]<0)witnesses[2*i]=air_row;
}
'''


class AnalyticRayEvidence:
    """Shared equations and an expanded BVH of currently supported face pieces."""
    def __init__(self, surface):
        from cutsim.rendering.analytic_surfaces import pack_snapshot
        from cutsim.cuda.current_gs_surface import _CUDA
        started = perf_counter()
        self.surface = surface
        rt = surface.rt; cp = rt.cp
        with rt.context():
            selection=getattr(surface,'sweep_selection',None)
            self.local=selection is not None
            cut=selection.packed if self.local else (np.empty((0,14)),np.empty((0,2),np.int64),np.empty((0,5)))
            self.cut_data=(*(cp.asarray(a) for a in cut[:3]),np.int32(len(cut[0])))
            if not hasattr(rt, 'analytic_ray_kernel'):
                rt.analytic_ray_kernel = cp.RawKernel(_CUDA+CUDA, 'analytic_evidence', options=('--fmad=false',))
            if len(surface.rows) or (self.local and len(selection.rows)):
                ids, eq, meta, rec, spans, planes = pack_snapshot(surface.snapshot,
                    rows=selection.rows if self.local else None)
                self.all_patch_ids = cp.asarray(ids)
                self.data = (self.all_patch_ids[cp.asarray(surface.rows)], *(cp.asarray(a) for a in (eq, meta, rec, spans, planes)))
                # Finalization reuses the complete finite cutter domains for
                # air rejection. Deduplicate the small host table before upload;
                # CuPy's axis-wise unique performs many host/device syncs.
                self.coverage_cut_solids = cp.asarray(np.unique(meta[meta[:, 3] == 0, :2], axis=0))
            else:
                self.all_patch_ids = cp.full(len(surface.snapshot.endpoints['positions']),-1,cp.int64)
                self.data = (cp.empty(0, cp.int64), cp.empty((0,16)), cp.empty((0,4), cp.int64),
                             cp.empty((0,14)), cp.empty((0,2), cp.int64), cp.empty((0,5)))
                self.coverage_cut_solids = cp.empty((0, 2), cp.int64)
            self.boxes = cp.empty_like(surface.boxes)
            # Projected disk radius <= R and normal displacement <= R imply
            # this conservative box. It is a search bound, not larger support.
            broad = surface.device_geometry.copy(); broad[:,12] *= 2.
            surface._launch('leaves', surface.leaves, (broad, surface.order, np.int64(len(surface.rows)),
                np.int64(surface.leaves), self.boxes))
            level = surface.leaves//2
            while level:
                surface._launch('parents', level, (self.boxes, np.int64(level), np.int64(level)))
                level //= 2
            rt.stream.synchronize()
        self.prepare_ms = (perf_counter()-started)*1000

    def evaluate(self, points, directions):
        s = self.surface; rt = s.rt; cp = rt.cp
        with rt.context():
            points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float64))
            directions = cp.ascontiguousarray(cp.asarray(directions, dtype=cp.float64))
            count = len(points)
            if directions.shape != points.shape:raise ValueError('One direction required per analytic query')
            result = cp.zeros(count, cp.int8); witness = cp.full((count,2), -1, cp.int64)
            fallback = cp.zeros(count, cp.uint8)
            if count:rt.analytic_ray_kernel(((count+63)//64,), (64,), (points, directions, np.int32(count),
                s.device_geometry, self.boxes, s.order, s.device_rows, np.int64(len(s.rows)),
                np.int64(s.leaves), *self.data, result, witness, fallback,*self.cut_data))
            return result, witness, fallback


def evaluator(surface):
    if getattr(surface, '_analytic_ray_evidence', None) is None:
        surface._analytic_ray_evidence = AnalyticRayEvidence(surface)
    return surface._analytic_ray_evidence


def select_generation(model, candidate_mask=None):
    """Evaluate every in-domain tool sample, independent of old centres/hits."""
    rt = model.runtime; cp = rt.cp
    with rt.context():
        begin, end = cp.cuda.Event(), cp.cuda.Event(); begin.record()
        field = evaluator(model.surface)
        q = model._origins + model._distance[:,None]*model._directions
        bounds = cp.asarray(model.grid.ray_bounds)
        valid = (model._distance>=bounds[:,0]) & (model._distance<=bounds[:,1])
        if candidate_mask is not None:
            valid &= candidate_mask
        rows = cp.flatnonzero(valid)
        evidence = cp.zeros(len(q), cp.int8)
        # A missing local face must not let distant stock faces refill a known
        # cavity. Use the same finite-cutter veto as child-ray generation.
        from cutsim.cuda.material_evidence import finite_material_evidence_device
        evidence[rows], fallback, cut_rejected = finite_material_evidence_device(
            model.surface, q[rows], model._directions[rows])
        model.added[:] = evidence==1
        # Only classification survives generation; no old-GS associations.
        model.generation_evidence = evidence
        model.stats.update(generation_method='analytic_dexel_evaluation',
            analytic_generation_candidates=len(rows), analytic_generation_added=int(cp.count_nonzero(evidence==1).item()),
            analytic_generation_air=int(cp.count_nonzero(evidence<0).item()),
            analytic_generation_unknown=int(cp.count_nonzero(evidence[rows]==0).item()),
            analytic_generation_axis_fallback=int(cp.count_nonzero(fallback).item()),
            analytic_generation_cut_rejections=int(cp.count_nonzero(cut_rejected).item()),
            analytic_generation_hit_queries=0,analytic_generation_hit_links_retained=0,
            analytic_generation_prepare_ms=field.prepare_ms,
            hit_material_rejected=0, miss_material_candidates=len(rows),
            miss_material_added=int(cp.count_nonzero(evidence==1).item()),
            generation_policy='sweep_local_analytic_centres' if field.local else 'analytic_centres_only',
            analytic_generation_local=field.local, analytic_generation_patch_count=len(field.data[1]),
            analytic_generation_source_gs=len(model.surface.rows),
            boundary_support_enabled=False, boundary_support_candidates=0,
            boundary_support_probes=0, boundary_support_added=0,
            boundary_support_cut_rejections=0)
        # Rejected centres stay rejected. Unseeded cells may receive child
        # Dexels later; their own centres must pass this same material rule.
        end.record(); end.synchronize()
        model.stats['analytic_generation_ms'] = float(cp.cuda.get_elapsed_time(begin, end))
