"""Grouped analytic clipping using the same local opacity quadrature as R.

Broad phase still uses current GS. Before evaluating a quadric, CUDA groups
(receiver, peer analytic patch, material side). No historic cutter replay.
The transient sampled regions drive both best-two fitting and bounded cover;
they are released at publication, never used as persistent material state.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from cutsim.rendering.analytic_pair_clip import CUDA as ANALYTIC_CUDA
from cutsim.rendering.gs_sample_evaluation import CUDA as SAMPLE_CUDA, disc, device_disc
from cutsim.rendering.analytic_surfaces import pack_snapshot
from cutsim.rendering.clip_edge_batches import EdgeBatches
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip
from cutsim.rendering.planar_floor_domain import CUDA as FLOOR_CUDA
from cutsim.rendering.finite_surface_contact import CUDA as CONTACT_CUDA

# Protrusion is measured against the original evaluation disk. The new
# retirement policy measures missing opacity against the surviving OLD region,
# so a tiny remainder cannot be discarded just because its original mass was
# small. This is a fit error tolerance, never a remaining-mass deletion quota.
FIT_TOLERANCE = .02
MAX_GROUPS = 64
ALTERNATIVES = 8

CUDA = f'#define CLIP_EVAL_SIGMA {CLIP_SIGMA}\n' + ANALYTIC_CUDA[:ANALYTIC_CUDA.index('__device__ bool root(')] + SAMPLE_CUDA + ANALYTIC_CUDA[ANALYTIC_CUDA.index('__device__ bool primitive_inside'):ANALYTIC_CUDA.index('__device__ int analytic_edge')] + FLOOR_CUDA + CONTACT_CUDA + r'''
extern "C" __global__ void grouped_candidates(const double* g,const long long* ids,int n,int old,
 const double* boxes,const long long* order,int leaves,const long long* offsets,int* counts,
 long long* pairs,int write){
 int i=old+blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const double* a=g+25*i;int stack[64],top=0,found=0;stack[top++]=1;
 while(top){int node=stack[--top];if(!pair_overlap(a,boxes+6*node))continue;
  if(node<leaves){stack[top++]=node*2+1;stack[top++]=node*2;continue;}
  int leaf=node-leaves;if(leaf>=n)continue;long long j=order[leaf];
  // Continuity IDs are independent of analytic patch IDs. In particular,
  // a ball start cap and its straight sweep must never clip one another.
  if(j>=i||ids[i]==ids[j])continue;
  double d[3]={a[0]-g[25*j],a[1]-g[25*j+1],a[2]-g[25*j+2]};
  double r=a[12]+g[25*j+12];if(pdot(d,d)>r*r)continue;
  if(write){long long at=offsets[i-old]+found;pairs[2*at]=i;pairs[2*at+1]=j;}found++;
 }
 if(!write)counts[i-old]=found;
}
__device__ double poly(const double* q,double x,double y){
 return q[0]*x*x+q[1]*x*y+q[2]*y*y+q[3]*x+q[4]*y+q[5];
}
__device__ bool line_yes(const double* e,double x,double y){return e[0]*x+e[1]*y+e[2]>=-1e-10;}
// A former round-end wall can meet a newly extended planar floor INSIDE
// that floor's finite generating solid. Its equation is not a trim boundary
// there. Certify a two-sided neighbourhood along the proposed edge normal;
// a real floor/wall valley fails this test and remains a clipping edge.
__device__ bool planar_interior_edge(const long long* m,const double* p,const double* direction,const double* normal,
 const double* rec,const long long* spans,const double* planes){
 for(int i=m[0];i<m[0]+m[1];i++){
  bool yes=true,coplanar=false;
  for(long long j=spans[2*i];j<spans[2*i+1];j++){
   const double* h=planes+5*j;double value=pdot(h,p)+h[3],slope=pdot(h,direction);
   double tol=1e-9*(1.+fabs(h[3])+sqrt(pdot(h,h)));
   if(value < -tol || (value<=tol&&fabs(slope)>tol)){yes=false;break;}
   double cross[3]={h[1]*normal[2]-h[2]*normal[1],h[2]*normal[0]-h[0]*normal[2],h[0]*normal[1]-h[1]*normal[0]};
   if(h[4]&&fabs(value)<=tol&&pdot(normal,normal)>0.&&pdot(cross,cross)<1e-18*pdot(h,h)*pdot(normal,normal))coplanar=true;
  }
  const double* r=rec+14*i;
  if(yes&&r[13]){
   double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.,slope=0.,curve=0.;
   for(int j=0;j<3;j++)for(int k=0;k<3;k++){
    double Q=r[3+3*j+k];q+=d[j]*Q*d[k];slope+=2*direction[j]*Q*d[k];curve+=direction[j]*Q*direction[k];
   }
   double tol=1e-9*(1.+r[12]*r[12]+fabs(q)),margin=r[12]*r[12]-q;
   if(margin < -tol || (margin<=tol&&(fabs(slope)>tol||curve>tol)))yes=false;
  }
  if(yes&&coplanar)return true;
 }
 return false;
}
extern "C" __global__ void evaluate_groups(const double* g,const long long* owners,
 const long long* peers,const long long* starts,const long long* counts,const long long* witnesses,
 const double* equations,const long long* meta,const double* rec,const long long* spans,const double* planes,
 const double* signs,const double* radii,const long long* patch_ids,const double* samples,int ns,int ng,int finite_floor,
 double* coefficients,double* edges,unsigned char* enabled){
 int i=blockIdx.x,lane=threadIdx.x;if(i>=ng)return;
 long long row=owners[i],peer=peers[i];const double* a=g+25*row;const double* e=equations+16*peer;
 const long long* ownmeta=meta+4*patch_ids[row];const double* own=equations+16*patch_ids[row];
 bool planar_union=signs[i]<0.&&!meta[4*peer+3]&&!ownmeta[3];
 for(int j=3;j<12;j++)if(own[j]!=0.)planar_union=false;
 bool use_domain=finite_floor&&planar_union&&physical_planar_floor(own,ownmeta,a,spans,planes);
 bool finite_peer=finite_floor&&signs[i]>0.&&!meta[4*peer+3];
 if(finite_peer&&!ownmeta[3]&&surface_clear_of_solid(own,meta+4*peer,rec,spans,planes)){
  if(!lane)enabled[i]=0;return;
 }
 double q[6];double grad[3];double f=gs_quadric(e,a,grad),sign=signs[i];
 q[0]=q[1]=q[2]=0.;
 for(int j=0;j<3;j++)for(int k=0;k<3;k++){
  double Q=e[3+3*j+k];q[0]+=a[3+j]*Q*a[3+k];q[1]+=2*a[3+j]*Q*a[6+k];q[2]+=a[6+j]*Q*a[6+k];
 }
 q[3]=pdot(grad,a+3);q[4]=pdot(grad,a+6);q[5]=f;
 for(int k=0;k<6;k++){q[k]*=sign;if(!lane)coefficients[6*i+k]=q[k];}
 __shared__ double candidates[64*3],errors[64],original[64],original_errors[64],locations[128];double R=radii[row],sigma=R/CLIP_EVAL_SIGMA;
 __shared__ double sample_x[512],sample_y[512],sample_weight[512];__shared__ unsigned char sample_truth[512];
 for(int j=lane;j<ns;j+=64){
  double x=R*samples[2*j],y=R*samples[2*j+1];sample_x[j]=x;sample_y[j]=y;
  double point[3];gs_sample_point(a,x,y,point);
  sample_truth[j]=finite_peer?!finite_cut_inside(point,meta+4*peer,rec,spans,planes,ownmeta[3]):
   (use_domain?floor_domain_inside(ownmeta,point,rec,spans,planes):poly(q,x,y)>=-1e-10);
  sample_weight[j]=gs_sample_admitted(a+13,x,y)&&gs_sample_admitted(a+19,x,y)?gs_sample_weight(x,y,sigma):0.;
 }
 original[lane]=0.;errors[lane]=original_errors[lane]=1e100;__syncthreads();
 if(lane<32){int angle=lane;
  double t=angle*3.141592653589793/32.,ux=cos(t),uy=sin(t);
  double A=q[0]*ux*ux+q[1]*ux*uy+q[2]*uy*uy,B=q[3]*ux+q[4]*uy,C=q[5];
  double roots[2];int nr=0;
  if(fabs(A)<1e-12){if(fabs(B)>1e-12)roots[nr++]=-C/B;}
  else{double det=B*B-4*A*C;if(det>=0.){
   double z=-.5*(B+copysign(sqrt(det),B));
   if(fabs(z)>1e-20){roots[nr++]=z/A;roots[nr++]=C/z;}else roots[nr++]=-B/(2*A);
  }}
  for(int k=0;k<nr;k++){
   double x=roots[k]*ux,y=roots[k]*uy;if(hypot(x,y)>a[12]+1e-8)continue;
   double p[3];gs_sample_point(a,x,y,p);
   if(!gs_sample_admitted(a+13,x,y)||!gs_sample_admitted(a+19,x,y))continue;
   if(!domain(meta+4*peer,p,rec,spans,planes))continue;
   bool supported=false;
   for(long long j=starts[i];j<starts[i]+counts[i];j++)if(admitted(g+25*witnesses[j],p)){supported=true;break;}
   if(!supported)continue;
   double nx=2*q[0]*x+q[1]*y+q[3],ny=q[1]*x+2*q[2]*y+q[4],len=hypot(nx,ny);
   if(len<1e-10)continue;nx/=len;ny/=len;double d=-nx*x-ny*y;
   if(planar_union){double direction[3];for(int j=0;j<3;j++)direction[j]=nx*a[3+j]+ny*a[6+j];
    if(planar_interior_edge(ownmeta,p,direction,own+12,rec,spans,planes))continue;}
   double best=1e100,bestd=d,unshifted=1e100;
   for(int shift=-3;shift<=3;shift++){
    double dd=d+shift*.02*R,miss=0.,over=0.;
    for(int j=0;j<ns;j++){
     bool yes=sample_truth[j],approx=nx*sample_x[j]+ny*sample_y[j]+dd>=-1e-10;
     double w=sample_weight[j];if(yes&&!approx)miss+=w;if(!yes&&approx)over+=w;
    }
    double error=miss+over+.1*fabs(miss-over);if(!shift)unshifted=error;
    // Sampling plateaus must not move an exact boundary inward by accident.
    if(error<best-1e-12||(fabs(error-best)<=1e-12&&fabs(dd-d)<fabs(bestd-d))){best=error;bestd=dd;}
   }
   int slot=2*lane+k;candidates[3*slot]=nx;candidates[3*slot+1]=ny;candidates[3*slot+2]=bestd;errors[slot]=best;original[slot]=d;original_errors[slot]=unshifted;locations[2*slot]=x;locations[2*slot+1]=y;
  }
 }
 __syncthreads();
 if(lane)return;enabled[i]=0;int chosen[8],used=0;
 for(int k=0;k<8;k++){
  int best=-1;double score=k?-1.:1e100;
  for(int j=0;j<64;j++)if(original_errors[j]<1e100){
   double value=errors[j];
   if(k){value=1e100;for(int t=0;t<used;t++){
    double dx=locations[2*j]-locations[2*chosen[t]],dy=locations[2*j+1]-locations[2*chosen[t]+1];value=fmin(value,dx*dx+dy*dy);
   }}
   if((!k&&value<score)||(k&&value>score)){score=value;best=j;}
  }
  double* out=edges+24*i+3*k;out[0]=out[1]=0.;out[2]=INFINITY;
  if(best>=0){enabled[i]=finite_peer?4:(use_domain?3:1);chosen[used++]=best;
   out[0]=candidates[3*best];out[1]=candidates[3*best+1];out[2]=k?original[best]:candidates[3*best+2];
   for(int j=0;j<64;j++)if(original_errors[j]<1e100&&fabs(candidates[3*j]-out[0])+fabs(candidates[3*j+1]-out[1])+fabs(original[j]-out[2])<1e-6)original_errors[j]=1e100;
   original_errors[best]=1e100;
  }
 }
}
// Each block fits one disk. It can be the full parent or a subdivision
// candidate. Truth is sampled from ALL analytic groups, not chosen edges.
extern "C" __global__ void fit_disks(const double* g,const double* radii,const long long* rows,
 const long long* starts,const long long* counts,const double* coefficients,const double* group_edges,
 const unsigned char* enabled,const double* samples,int ns,const double* disks,int nd,
 int alternatives,double tolerance,int partial,int preserve_nonempty,int cached,const unsigned char* cached_truth,const double* cached_weights,const unsigned int* face_masks,double* result,double* errors,unsigned char* accepted){
 int id=blockIdx.x,lane=threadIdx.x;if(id>=nd)return;
 const double* disk=disks+5*id;int owner=(int)disk[0];long long row=rows[owner];
 const double* a=g+25*row;double R=radii[row],sigma=R/CLIP_EVAL_SIGMA,cx=disk[1],cy=disk[2],r=disk[3];
 __shared__ double xy[1024],weights[512],lines[3*720],loss[128],missing[128],extra[128];
 __shared__ unsigned char truth[512];__shared__ int choices[128],number;
 for(int j=lane;j<ns;j+=128){
  double x=cached?R*samples[2*j]:cx+r*samples[2*j],y=cached?R*samples[2*j+1]:cy+r*samples[2*j+1];xy[2*j]=x;xy[2*j+1]=y;
  bool yes=true;
  if(!cached){
   yes=x*x+y*y<=R*R+1e-9&&gs_sample_admitted(a+13,x,y)&&gs_sample_admitted(a+19,x,y);
   for(long long k=starts[owner];yes&&k<starts[owner]+counts[owner];k++)if(enabled[k]==1&&poly(coefficients+6*k,x,y)<-1e-10)yes=false;
  }
  truth[j]=cached?cached_truth[owner*ns+j]:yes;weights[j]=cached?cached_weights[j]:gs_sample_weight(x,y,sigma);
 }
 if(!lane){
  // Four inherited edges are candidates too. The zero edge permits one/no
  // edge solutions. A negative constant represents the empty region.
  number=0;lines[0]=lines[1]=0.;lines[2]=INFINITY;number++;
  lines[3]=lines[4]=0.;lines[5]=-1.;number++;
  for(int k=13;k<25;k+=3)if(isfinite(a[k+2])){for(int j=0;j<3;j++)lines[3*number+j]=a[k+j];number++;}
  for(long long k=starts[owner];k<starts[owner]+counts[owner];k++)if(enabled[k]){
   int first=number;
   for(int t=0;t<alternatives;t++)if(isfinite(group_edges[24*k+3*t+2])){
    for(int j=0;j<3;j++)lines[3*number+j]=group_edges[24*k+3*t+j];number++;
   }
   if(partial){
    const double* q=coefficients+6*k;double gx=2*q[0]*cx+q[1]*cy+q[3],gy=q[1]*cx+2*q[2]*cy+q[4],len=hypot(gx,gy);
    if(len>1e-12){double ux=gx/len,uy=gy/len,A=q[0]*ux*ux+q[1]*ux*uy+q[2]*uy*uy,B=len,C=poly(q,cx,cy),t=0.;bool found=true;
     if(fabs(A)<1e-12)t=-C/B;
     else{double det=B*B-4*A*C;if(det<0.)found=false;else{double z=-.5*(B+sqrt(det));double ta=z/A,tb=fabs(z)>1e-20?C/z:ta;t=fabs(ta)<fabs(tb)?ta:tb;}}
     double x=cx+t*ux,y=cy+t*uy;gx=2*q[0]*x+q[1]*y+q[3];gy=q[1]*x+2*q[2]*y+q[4];len=hypot(gx,gy);
     if(found&&len>1e-12){gx/=len;gy/=len;for(int shift=-1;shift<=1;shift++){
      lines[3*number]=gx;lines[3*number+1]=gy;lines[3*number+2]=-gx*x-gy*y+shift*.02*r;number++;
     }}
     if(partial>1&&found&&len>1e-12&&(fabs(q[0])+fabs(q[1])+fabs(q[2])>1e-14)){
      // A tangent can admit the wrong side of a curved boundary. Reuse the
      // fitting witnesses to add a conservative parallel chord for this
      // child, rather than relying on a fixed small tangent displacement.
      double limit=-INFINITY;
      for(int j=0;j<ns;j++){
       double px=xy[2*j],py=xy[2*j+1],dx=px-cx,dy=py-cy;
       if(dx*dx+dy*dy<=r*r+1e-9&&poly(q,px,py)<-1e-10)limit=fmax(limit,gx*px+gy*py);
      }
      if(isfinite(limit)){lines[3*number]=gx;lines[3*number+1]=gy;lines[3*number+2]=-limit-1e-9;number++;}
     }
    }
   }
   if(partial==3&&number>first){
    // Refit this face locally from exact crossings bracketed by the saved
    // R neighbours. Child placement and fitting use no new material queries.
    const double* q=coefficients+6*k;double mass=0.,mx=0.,my=0.,mxx=0.,mxy=0.,myy=0.;
    for(int j=0;j<ns;j++)for(int direction=0;direction<2;direction++){
     int other=direction?j+32:j/32*32+(j+1)%32;if(other>=ns)continue;
     double x=xy[2*j],y=xy[2*j+1],dx=xy[2*other]-x,dy=xy[2*other+1]-y;
     double f=poly(q,x,y),F=poly(q,x+dx,y+dy);if((f>=0.)==(F>=0.))continue;
     double A=q[0]*dx*dx+q[1]*dx*dy+q[2]*dy*dy,B=(2*q[0]*x+q[1]*y+q[3])*dx+(q[1]*x+2*q[2]*y+q[4])*dy,t;
     if(fabs(A)<1e-14){if(fabs(B)<1e-20)continue;t=-f/B;}
     else{double det=B*B-4*A*f;if(det<0.)continue;double z=-.5*(B+copysign(sqrt(det),B)),ta=z/A,tb=fabs(z)>1e-25?f/z:-B/(2*A);t=ta>=0.&&ta<=1.?ta:tb;}
     if(t<0.||t>1.||!isfinite(t))continue;x+=t*dx;y+=t*dy;
     if((x-cx)*(x-cx)+(y-cy)*(y-cy)>r*r+1e-9)continue;
     double w=exp(-2*(x*x+y*y)/(R*R));mass+=w;mx+=w*x;my+=w*y;mxx+=w*x*x;mxy+=w*x*y;myy+=w*y*y;
    }
    if(mass>0.){double x=mx/mass,y=my/mass,xx=mxx/mass-x*x,yy=myy/mass-y*y,xyc=mxy/mass-x*y;
     double angle=.5*atan2(2*xyc,xx-yy),nx=-sin(angle),ny=cos(angle);
     if(xx+yy>1e-16){lines[3*number]=nx;lines[3*number+1]=ny;lines[3*number+2]=-nx*x-ny*y;number++;}}
    // One analytic face contributes ONE line to this child. Compare local
    // alternatives against that face's saved mask on the SAME R witnesses;
    // never spend both child edges on two tangents from one analytic face.
    int best=first,orientation=1;double loss=INFINITY;
    for(int t=first;t<number;t++)for(int sign=1;sign>=-1;sign-=2){
     double wrong=0.;
     for(int j=0;j<ns;j++){double dx=xy[2*j]-cx,dy=xy[2*j+1]-cy;
      if(dx*dx+dy*dy>r*r+1e-9)continue;
      bool wanted=(face_masks[k*(ns/32)+j/32]>>(j%32))&1u;
      bool yes=sign*(lines[3*t]*xy[2*j]+lines[3*t+1]*xy[2*j+1]+lines[3*t+2])>=-1e-10;
      if(yes!=wanted)wrong+=weights[j];
     }
     // Sampling plateaus must not prefer a distant tangent over a boundary
     // fitted through the very same R-neighbour crossings.
     if(mass>0.){double nx=lines[3*t],ny=lines[3*t+1],d=lines[3*t+2];
      wrong+=1e-6*fmax(0.,(nx*nx*mxx+2*nx*ny*mxy+ny*ny*myy+2*d*(nx*mx+ny*my)+d*d*mass)/mass)/(R*R);}
     if(wrong<loss-1e-15){loss=wrong;best=t;orientation=sign;}
    }
    for(int j=0;j<3;j++)lines[3*first+j]=orientation*lines[3*best+j];number=first+1;
   }
  }
 }
 __syncthreads();double best=1e100,bm=1.,be=1.;int chosen=-1;
 int pairs=number*(number+1)/2;
 for(int candidate=lane;candidate<pairs;candidate+=128){
  int i=0,z=candidate;while(z>=number-i){z-=number-i;i++;}int k=i+z;
  double miss=0.,over=0.,total=0.,kept=0.,target_weight=0.;
  for(int j=0;j<ns;j++){
   double dx=xy[2*j]-cx,dy=xy[2*j+1]-cy;
   bool yes=(!cached||dx*dx+dy*dy<=r*r+1e-9)&&line_yes(lines+3*i,xy[2*j],xy[2*j+1])&&line_yes(lines+3*k,xy[2*j],xy[2*j+1]);
   total+=weights[j];if(truth[j])target_weight+=weights[j];if(yes&&truth[j]){double focus=disk[4];double w=focus>=0.?1.+.95*(cos(focus)*xy[2*j]+sin(focus)*xy[2*j+1])/R:1.;kept+=weights[j]*w;}
   if(yes&&!truth[j])over+=weights[j];if(!yes&&truth[j])miss+=weights[j];
  }
  miss/=(row<preserve_nonempty&&target_weight>0.?target_weight:total);over/=total;double score=partial?(over<=tolerance?-kept/total:1e90+over):miss+over;
  if(score<best){best=score;chosen=candidate;bm=miss;be=over;}
 }
 loss[lane]=best;choices[lane]=chosen;missing[lane]=bm;extra[lane]=be;__syncthreads();
 for(int stride=64;stride;stride/=2){if(lane<stride&&(loss[lane+stride]<loss[lane]||
  (loss[lane+stride]==loss[lane]&&choices[lane+stride]>=0&&(choices[lane]<0||choices[lane+stride]<choices[lane])))){
  loss[lane]=loss[lane+stride];choices[lane]=choices[lane+stride];missing[lane]=missing[lane+stride];extra[lane]=extra[lane+stride];}__syncthreads();}
 if(!lane){
  int i=0,z=choices[0];while(z>=number-i){z-=number-i;i++;}int k=i+z;
  double* out=result+15*id;out[0]=cx;out[1]=cy;out[2]=r;
  for(int t=0;t<2;t++)for(int j=0;j<3;j++){
   double v=lines[3*(t?k:i)+j];if(j==2)v+=lines[3*(t?k:i)]*cx+lines[3*(t?k:i)+1]*cy;
   out[3+3*t+j]=out[9+3*t+j]=v;
  }
  errors[2*id]=missing[0];errors[2*id+1]=extra[0];
  accepted[id]=extra[0]<=tolerance&&(partial||missing[0]<=tolerance);
 }
}
extern "C" __global__ void region_truth(const double* g,const double* radii,const long long* rows,
 const long long* starts,const long long* counts,const double* q,const unsigned char* enabled,
 const double* samples,int ns,int nr,unsigned char* truth,double* weights){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr*ns)return;int owner=i/ns,j=i%ns;
 long long row=rows[owner];const double* a=g+25*row;double R=radii[row],x=R*samples[2*j],y=R*samples[2*j+1];
 bool yes=gs_sample_admitted(a+13,x,y)&&gs_sample_admitted(a+19,x,y);
 for(long long k=starts[owner];yes&&k<starts[owner]+counts[owner];k++)if(enabled[k]&&poly(q+6*k,x,y)<-1e-10)yes=false;
 truth[i]=yes;weights[i]=gs_sample_weight(x,y,R/CLIP_EVAL_SIGMA);
}
'''


@dataclass
class SampledRegions:
    """CPU serialization boundary for debug capture, device arrays in normal use."""
    geometry: object
    radii: object
    rows: object
    starts: object
    counts: object
    coefficients: object
    edges: object
    enabled: object
    preview: object
    error: object
    accepted: object
    # Retained prefix length. Only old GS replace R's retirement decision;
    # new GS keep their existing clipping tolerances.
    preserve_nonempty: int=0


class SampledEdges(EdgeBatches):
    def __init__(self, rows, preview, regions):
        super().__init__();self.append(rows, preview[:, 3:9].reshape(-1, 2, 3));self.regions=regions


def kernels(rt):
    if not hasattr(rt, 'sampled_analytic_kernels'):
        module=rt.cp.RawModule(code=CUDA,options=('--fmad=false',))
        rt.sampled_analytic_kernels={s:module.get_function(s) for s in
            ('pair_codes','pair_leaves','pair_parents','grouped_candidates','evaluate_groups','fit_disks','region_truth')}
    return rt.sampled_analytic_kernels


def grouped_regions(geometry, identity, nold, packed, radius, *, preserve_nonempty=False,defer_fit=False,sample_first=False):
    from cutsim.cuda.profile_dexel_cuda import runtime
    rt=runtime();cp=rt.cp;n=len(geometry)
    with rt.context():
        k=kernels(rt);g=cp.asarray(geometry,dtype=cp.float64,order='C');ids=cp.asarray(identity,dtype=cp.int64,order='C');r=cp.asarray(radius,dtype=cp.float64)
        patch,equations,meta,rec,spans,planes=map(cp.asarray,packed)
        leaves=1<<(n-1).bit_length();codes=cp.empty(n,cp.uint32);boxes=cp.empty((2*leaves,6))
        k['pair_codes'](((n+127)//128,),(128,),(g,np.int32(n),cp.min(g[:,:3],axis=0),cp.max(g[:,:3],axis=0),codes))
        order=cp.argsort(codes).astype(cp.int64)
        k['pair_leaves'](((leaves+127)//128,),(128,),(g,order,np.int32(n),np.int32(leaves),boxes))
        level=leaves//2
        while level:k['pair_parents'](((level+127)//128,),(128,),(boxes,np.int32(level)));level//=2
        count=cp.zeros(n-nold,cp.int32);dummy=cp.empty(1,cp.int64)
        args=(g,ids,np.int32(n),np.int32(nold),boxes,order,np.int32(leaves))
        if n>nold:k['grouped_candidates'](((n-nold+127)//128,),(128,),(*args,dummy,count,dummy,np.int32(0)))
        offsets=cp.cumsum(count,dtype=cp.int64);total=int(offsets[-1].item()) if len(count) else 0;offsets-=count
        if total>2_000_000:raise RuntimeError('Grouped analytic clipping pair budget exceeded')
        pairs=cp.empty((total,2),cp.int64)
        if n>nold:k['grouped_candidates'](((n-nold+127)//128,),(128,),(*args,offsets,count,pairs,np.int32(1)))
        coplanar=cp.empty((0,2),cp.int64)
        if defer_fit and len(pairs):
            # Turning blocks have different surface IDs even when their floor
            # is one physical plane. Preserve these pairs for live-support
            # sharing after the new disks have their final clipping fitted.
            a,b=pairs[:,0],pairs[:,1]
            planar=cp.all(equations[patch,3:12]==0.,axis=1)&(meta[patch,3]==0)
            matching=(b<nold)&planar[a]&planar[b]
            matching &= cp.sum(g[a,9:12]*g[b,9:12],axis=1)>1.-1e-10
            matching &= cp.abs(cp.sum((g[a,:3]-g[b,:3])*g[b,9:12],axis=1))<1e-9
            coplanar=pairs[matching]
        owner=pairs.ravel();witness=pairs[:,::-1].ravel();peer=patch[witness]
        same=(ids[owner]>=7)&(ids[witness]>=7)&((ids[owner]-7)//3==(ids[witness]-7)//3)
        sign=cp.where((meta[peer,3]!=0)|same,-1.,1.)
        order=cp.lexsort(cp.stack((sign,peer,owner)));owner=owner[order];peer=peer[order];sign=sign[order];witness=witness[order]
        first=cp.ones(len(order),bool)
        if len(first)>1:first[1:]=(owner[1:]!=owner[:-1])|(peer[1:]!=peer[:-1])|(sign[1:]!=sign[:-1])
        begin=cp.flatnonzero(first);end=cp.r_[begin[1:],len(order)];group_count=end-begin
        owners=owner[begin];peers=peer[begin];signs=sign[begin];ng=len(begin)
        q=cp.zeros((ng,6));edges=cp.zeros((ng,8,3));edges[:,:,2]=cp.inf;enabled=cp.zeros(ng,cp.uint8)
        samples=device_disc(rt)
        args2=(g,owners,peers,begin,group_count,witness,equations,meta,rec,spans,planes,signs,r,patch,samples,np.int32(512),np.int32(ng),np.int32(defer_fit),q,edges,enabled)
        if ng:
            if sample_first:
                from cutsim.rendering.sample_priority_clip import kernels as priority_kernels
                priority_kernels(rt)['describe_groups'](((ng+127)//128,),(128,),args2)
            else:k['evaluate_groups']((ng,),(64,),args2)
        context=cp.column_stack((peers,begin,group_count,patch[owners])) if sample_first else None
        # Ignore groups with no admitted finite boundary before fitting. They
        # cannot turn an unsupported infinite extension into a clipping edge.
        active=cp.flatnonzero(enabled);owners=owners[active];q=q[active];edges=edges[active];enabled=enabled[active]
        active_peers=peers[active]
        if sample_first:context=context[active]
        if len(coplanar):
            # Tangential contact otherwise has no active clipping group. It
            # still needs cached residual samples to release an old cap seam.
            extra=cp.unique(coplanar[:,1]);blank=cp.zeros((len(extra),8,3));blank[:,:,2]=cp.inf
            owners=cp.r_[owners,extra];q=cp.concatenate((q,cp.zeros((len(extra),6))))
            edges=cp.concatenate((edges,blank));enabled=cp.r_[enabled,cp.zeros(len(extra),cp.uint8)]
            active_peers=cp.r_[active_peers,cp.full(len(extra),-1,cp.int64)]
            if sample_first:context=cp.concatenate((context,cp.column_stack((cp.full(len(extra),-1,cp.int64),cp.zeros((len(extra),2),cp.int64),patch[extra]))))
            order=cp.argsort(owners);owners=owners[order];q=q[order];edges=edges[order];enabled=enabled[order];active_peers=active_peers[order]
            if sample_first:context=context[order]
        rows,starts,counts=cp.unique(owners,return_index=True,return_counts=True)
        limit=128 if sample_first else MAX_GROUPS
        if len(counts) and int(counts.max().item())>limit:
            raise RuntimeError(f'More than {limit} analytic groups for one GS')
        disks=cp.zeros((len(rows),5));disks[:,0]=cp.arange(len(rows));disks[:,3]=r[rows];disks[:,4]=-1.
        result=cp.empty((len(rows),15));error=cp.empty((len(rows),2));accepted=cp.empty(len(rows),cp.uint8)
        region=SampledRegions(g,r,rows,starts,counts,q,edges,enabled,result,error,accepted,preserve_nonempty)
        # Kind 3 groups are emitted only for defer_fit (the standard shared
        # path); prepare() builds their cached truth before any fit is run.
        region.floor_domain_data=(patch,meta,rec,spans,planes)
        region.analytic_equations=equations
        region.peer_surfaces=active_peers
        region.coplanar_pairs=coplanar
        if sample_first:
            region.sample_first=True;region.group_context=context;region.witnesses=witness
        if not defer_fit and not sample_first:fit(rt,region,disks,result,error,accepted,alternatives=1)
        detail=dict(candidate_gs_pairs=total,analytic_group_candidates=ng,analytic_group_active=len(active),
                    analytic_group_duplicates_removed=2*total-ng,analytic_fit_rows=len(rows))
        return region,detail


def fit(rt,region,disks,result,error,accepted,*,alternatives=1,partial=False,validation=False,tolerance=FIT_TOLERANCE,curvature_adjust=False,one_line_per_face=False):
    if curvature_adjust and (not partial or alternatives>4):
        raise ValueError('Curvature adjustment requires partial fitting with at most four alternatives')
    cp=rt.cp;cached=hasattr(region,'sample_truth')
    if one_line_per_face and (not partial or not cached or validation):
        raise ValueError('Facewise child fitting requires the saved residual samples')
    truth=region.sample_truth if cached else cp.empty(0,cp.uint8)
    weights=region.sample_weights if cached else cp.empty(0)
    samples=region.sample_xy if cached and hasattr(region,'sample_xy') else device_disc(rt)
    ns=len(samples);kernel=kernels(rt)['fit_disks']
    if validation:
        # Only a failed cover pays for the larger shared-memory fitter.
        # Use exactly the witnesses/weights that rejected the coarse fit.
        samples=region.validation_samples;ns=len(samples)
        if ns>896:raise ValueError('Validation fitter supports at most 896 samples')
        truth=region.validation_truth;weights=region.validation_weights.copy();weights[-128:]*=.125
        cached=True
    if ns>896:raise ValueError('Disk fitter supports at most 896 samples')
    if validation or ns>512:
        if not hasattr(rt,'validation_cover_kernel'):
            code=CUDA.replace('void fit_disks(', 'void fit_validation_disks(').replace(
                'xy[1024],weights[512]', 'xy[1792],weights[896]').replace('truth[512];__shared__ int choices','truth[896];__shared__ int choices')
            module=cp.RawModule(code=code,options=('--fmad=false',))
            rt.validation_cover_kernel=module.get_function('fit_validation_disks')
        kernel=rt.validation_cover_kernel
    if len(disks):kernel((len(disks),),(128,),
        (region.geometry,region.radii,region.rows,region.starts,region.counts,region.coefficients,
         region.edges,region.enabled,samples,np.int32(ns),disks,np.int32(len(disks)),
         np.int32(alternatives),np.float64(tolerance),np.int32(3 if one_line_per_face else (2 if curvature_adjust else partial)),np.int32(region.preserve_nonempty),np.int32(cached),truth,weights,
         region.face_masks if one_line_per_face else cp.empty(0,cp.uint32),result,error,accepted))


def update_sampled_clipping(snapshot,previous,kept,*,collect_edges=True,gpu_prepare=True,preserve_nonempty=False,retained_sweep=None,current_surface_ids=None,stock_bounds=None):
    if not collect_edges:raise ValueError('Sampled analytic clipping requires 2edge subdivision')
    from cutsim.cuda.profile_dexel_cuda import runtime
    start=perf_counter();rt=runtime();cp=rt.cp;n=len(snapshot.endpoints['positions']);nold=int(np.count_nonzero(kept))
    draw=np.tile([0.,0.,np.inf,0.,0.,np.inf,np.inf],(n,1));material=draw.copy()
    draw[:nold]=previous.clipping.planes[kept];material[:nold]=(previous.material_clipping or previous.clipping).planes[kept]
    floor_extended=0
    if retained_sweep is not None:
        from cutsim.rendering.planar_floor_domain import extend_round_floors
        floor_extended=extend_round_floors(snapshot,nold,draw,material)
    sigma=snapshot.gaussian.attributes['scale'][:,0].astype(float);radius=CLIP_SIGMA*sigma
    # Publish the current clipping domain for every newly generated GS,
    # including those with no active edge. Sigma stays unchanged; ray queries
    # independently cap this certificate at the configured ray support.
    draw[nold:,6]=radius[nold:];material[nold:,6]=radius[nold:]
    g=np.ascontiguousarray(np.column_stack((snapshot.endpoints['positions'],snapshot.gaussian.u,snapshot.gaussian.v,
        np.cross(snapshot.gaussian.u,snapshot.gaussian.v),np.where((draw[:,6]>0)&(material[:,6]>0),2*sigma,0.),draw[:,:6],material[:,:6])))
    with rt.context():
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        region,detail=grouped_regions(g,snapshot.material_surface_ids,nold,pack_snapshot(snapshot),radius,preserve_nonempty=nold if preserve_nonempty else 0,defer_fit=retained_sweep is not None)
        if retained_sweep is not None:
            from cutsim.rendering.retained_region_samples import prepare
            detail.update(prepare(rt,region,snapshot,nold,retained_sweep,current_surface_ids,stock_bounds=stock_bounds))
            disks=cp.zeros((len(region.rows),5));disks[:,0]=cp.arange(len(region.rows));disks[:,3]=region.radii[region.rows];disks[:,4]=-1.
            fit(rt,region,disks,region.preview,region.error,region.accepted,alternatives=8)
            from cutsim.rendering.coplanar_coverage import extend_coverage
            detail.update(extend_coverage(rt,region,nold))
        end.record();end.synchronize();ms=float(cp.cuda.get_elapsed_time(begin,end))
        rows=cp.asnumpy(region.rows,stream=rt.stream);preview=cp.asnumpy(region.preview,stream=rt.stream)
    draw[rows,:6]=preview[:,3:9];material[rows,:6]=preview[:,9:15]
    raw=SampledEdges(rows,preview,region)
    stats=dict(backend='cuda',mode='analytic_sampled',clipping_source='analytic_sampled',material_source='current_gs',
        kernel_ms=ms,total_ms=(perf_counter()-start)*1000,analytic_patch_count=len(snapshot.analytic_patches),
        recomputed_gs=len(rows),reused_gs=n-len(rows),clip_edge_method='analytic-sampled',
        boundary_search_sigma=2.,edge_evaluation_sigma=CLIP_SIGMA,radius_limited_gs=0,preserve_nonempty=preserve_nonempty,
        planar_floor_domains_extended=floor_extended,
        clipped_gs=int(np.count_nonzero(np.minimum(draw[:,2],draw[:,5])<radius)),
        gs_pair_count=detail['candidate_gs_pairs'],gs_pair_ridges=0,gs_pair_valleys=0,
        rejected_extrapolated_boundaries=0,accepted_finite_boundaries=detail['analytic_group_active'],boundary_segment_queries=0,reselected_boundaries=0,
        accepted_boundary_planes=detail['analytic_group_active'],analytic_fit_tolerance=FIT_TOLERANCE,analytic_input_old=nold,**detail)
    return ProfileClip(draw,stats,raw),ProfileClip(material,dict(stats),raw)


def _publish_parent_fits(snapshot,*,trim_cache=None):
    """Inspection mode: retain parent geometry and publish its best two edges.

    Fit errors remain diagnostic; they do not trigger children or a failure.
    This approximation is used by both drawing and subsequent material rays.
    """
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.rendering.analytic_continuation import refresh_planar_continuations
    started=perf_counter();rt=runtime();cp=rt.cp
    region=snapshot.clipping.constraints.regions
    keep=np.ones(len(snapshot.endpoints['positions']),bool)
    stats=dict(snapshot.clipping.stats)
    draw=snapshot.clipping.planes.copy();material=snapshot.material_clipping.planes.copy()
    with rt.context():
        rows=cp.asnumpy(region.rows,stream=rt.stream)
        preview=cp.asnumpy(region.preview,stream=rt.stream)
        radii=cp.asnumpy(region.radii,stream=rt.stream)
        fallback=getattr(region,'twoedge_fallback',None)
        failed=(np.zeros(len(rows),bool) if fallback is None or not stats.get('twoedge_invalid_or_empty',0)
                else cp.asnumpy(fallback,stream=rt.stream).astype(bool))
        rollback_rows=rows[failed];radius_rows=rows[~failed]
        for target,sl in ((draw,slice(3,9)),(material,slice(9,15))):
            target[rows,:6]=preview[:,sl];target[radius_rows,6]=radii[radius_rows]
        deferred=np.zeros(len(rows),bool)
        if hasattr(region,'validation_truth') and len(rows):
            deferred=cp.asnumpy((region.rows<region.preserve_nonempty)&
                ~region.validation_truth.any(axis=1)&~region.sample_truth.any(axis=1),stream=rt.stream)
            deferred &= ~failed
            rr=rows[deferred]
            original=cp.asnumpy(region.geometry[cp.asarray(rr)],stream=rt.stream)
            draw[rr,:6]=original[:,13:19];material[rr,:6]=original[:,19:25]
        rejected=int(cp.count_nonzero(~region.accepted.astype(bool)).item())
    from cutsim.rendering.empty_residual_clip import trim_cut_interior,affected_old_rows
    candidates=affected_old_rows(rt,region)
    if len(rollback_rows):candidates=candidates[~np.isin(candidates,rollback_rows)]
    reused_before=0 if trim_cache is None else trim_cache.get('reused',0)
    trimmed=trim_cut_interior(rt,snapshot,candidates,draw,material,getattr(region,'retirement_domain',None),cache=trim_cache)
    continuation_keep=keep.copy();continuation_keep[rollback_rows]=False
    stats.update(refresh_planar_continuations(snapshot,stats.get('analytic_input_old',0),draw,material,retained=continuation_keep))
    stats.update(clip_subdivision=False,clip_parent_only=True,clip_gpu_prepare=True,
        clip_partition_checked=len(rows),clip_partition_kept=len(rows),
        clip_partition_parents=0,clip_partition_children=0,
        clip_parent_fit_outside_tolerance=rejected,retained_region_empty_deferred=int(deferred.sum()),
        retained_region_certified_trimmed=trimmed,retained_region_certified_checked=len(candidates),
        retained_region_certified_reused=0 if trim_cache is None else trim_cache.get('reused',0)-reused_before,
        clip_sampled_fit_tolerance=FIT_TOLERANCE,clip_sampled_validation_points=0,
        clip_partition_ms=(perf_counter()-started)*1000,radius_limited_gs=0)
    snapshot.clipping=ProfileClip(draw,stats);snapshot.material_clipping=ProfileClip(material,dict(stats))
    return keep,np.empty(0,np.int64),np.empty((0,15)),stats


def partition_sampled(snapshot,*,max_children=4096,per_parent=32,subdivide=True,trim_cache=None):
    """Publish an accepted best-two fit or a bounded sampled cover of children.

    Children are selected greedily only for failed parent fits. Full-radius
    clipped fragments cover boundary arcs; smaller real GS cover interior
    regions. No parent gets a reduced radius as a clipping fallback.
    """
    if not subdivide:
        return _publish_parent_fits(snapshot,trim_cache=trim_cache)
    if getattr(snapshot.clipping.constraints.regions,'sample_first',False):
        from cutsim.rendering.multi_edge_subdivision import partition_multi_edge
        return partition_multi_edge(snapshot,max_children=max_children,trim_cache=trim_cache)
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.rendering.clip_subdivision import ClippingSubdivisionError
    from cutsim.rendering.cover_selection import select_cover
    started=perf_counter();rt=runtime();cp=rt.cp;region=snapshot.clipping.constraints.regions
    n=len(snapshot.endpoints['positions']);stats=dict(snapshot.clipping.stats)
    with rt.context():
        rows=region.rows;nr=len(rows);accepted=region.accepted.astype(bool)
        # Verify on staggered quadrature as well; fitting samples alone could
        # hide a narrow missed region between polar spokes.
        failed=cp.flatnonzero(~accepted);children=[];parents=[];unresolved=[]
        cached=hasattr(region,'validation_truth')
        if cached:
            parent_samples=region.validation_samples
            ns=len(parent_samples);truth=region.validation_truth
            base_weights=region.validation_weights.copy();base_weights[-128:]*=.125
            weights=cp.broadcast_to(base_weights,(nr,ns))
        else:
            parent_samples=cp.asarray(np.vstack((disc(24,32),np.column_stack((np.cos(np.arange(128)*np.pi/64),np.sin(np.arange(128)*np.pi/64)))*.999)))
            ns=len(parent_samples);truth=cp.empty((nr,ns),cp.uint8);weights=cp.empty((nr,ns))
            if nr:kernels(rt)['region_truth'](((nr*ns+127)//128,),(128,),
                (region.geometry,region.radii,rows,region.starts,region.counts,region.coefficients,region.enabled,
                 parent_samples,np.int32(ns),np.int32(nr),truth,weights))
        if not cached:weights[:,-128:]*=.125  # boundary witnesses carry bounded area weight
        def coverage(c,xy):
            local=xy[None]-c[:,None,:2]
            inside=cp.sum(local*local,axis=2)<=c[:,None,2]**2+1e-9
            for sl in (3,6):inside&=cp.sum(c[:,None,sl:sl+2]*local,axis=2)+c[:,None,sl+2]>=-1e-10
            return inside
        # Bound validation scratch independently of the affected-GS count.
        # A full nr*896*2 position array would exceed the 512 MiB pool.
        for first in range(0,nr,256):
            sl=slice(first,min(first+256,nr));rr=rows[sl];ww=weights[sl]
            xy=region.radii[rr,None,None]*parent_samples[None]
            c=region.preview[sl];local=xy-c[:,None,:2]
            predicted=cp.sum(local*local,axis=2)<=c[:,None,2]**2+1e-9
            for slot in (3,6):predicted&=cp.sum(c[:,None,slot:slot+2]*local,axis=2)+c[:,None,slot+2]>=-1e-10
            wsum=ww.sum(axis=1);target=truth[sl].astype(bool)
            missing_denominator=cp.where(rr<region.preserve_nonempty,cp.maximum(cp.sum(ww*target,axis=1),1e-30),wsum)
            miss=cp.sum(ww*(target&~predicted),axis=1)/missing_denominator
            over=cp.sum(ww*(~target&predicted),axis=1)/wsum
            accepted[sl]&=(miss<=FIT_TOLERANCE)&(over<=FIT_TOLERANCE)
        failed=cp.flatnonzero(~accepted)
        # Group spans are dispatch metadata. Download them once, rather than
        # synchronizing twice for every failed parent below.
        group_spans=cp.asnumpy(cp.column_stack((region.starts[failed],region.counts[failed])),stream=rt.stream)
        # Small batches bound the fit scratch space independently of scene size.
        # CPU only dispatches bounded batches; all fitting/cover scoring is CUDA.
        for batch in range(0,len(failed),32):
            ids=failed[batch:batch+32];R=region.radii[rows[ids]]
            seeds=cp.asarray(disc(4,12));length=cp.linalg.norm(seeds,axis=1)
            local=seeds[None]*R[:,None,None]
            radii=cp.minimum((1.-length)[None],.5)*R[:,None]
            if cached:
                # One actual surviving witness per polar bin. No candidates
                # are placed in fully removed bins. All selection stays GPU.
                angles=cp.arctan2(parent_samples[:,1],parent_samples[:,0])
                bins=cp.minimum((cp.sum(parent_samples**2,axis=1)*4).astype(cp.int64),3)*12+cp.minimum(((angles+cp.pi)/(2*cp.pi)*12).astype(cp.int64),11)
                scores=cp.where(truth[ids].astype(bool),weights[ids],-1.)
                best=cp.full((len(ids),48),-1.)
                cp.maximum.at(best,(cp.arange(len(ids))[:,None],bins[None]),scores)
                indices=cp.full((len(ids),48),ns,cp.int64)
                choices=cp.where((scores>=0.)&(scores==best[:,bins]),cp.arange(ns)[None],ns)
                cp.minimum.at(indices,(cp.arange(len(ids))[:,None],bins[None]),choices)
                valid_seed=indices<ns
                points=parent_samples[cp.minimum(indices,ns-1)]
                local=points*R[:,None,None]
                radii=cp.where(valid_seed,cp.minimum(1.-cp.linalg.norm(points,axis=2),.5)*R[:,None],0.)
            # Eight focus directions allow different maximal two-edge pieces
            # of a nonconvex or multi-boundary domain without GS radius changes.
            nc=8+len(seeds)*3;disks=cp.zeros((len(ids),nc,5));disks[:,:,0]=ids[:,None];disks[:,:,4]=-1.
            disks[:,:8,3]=R[:,None];disks[:,:8,4]=cp.arange(8)[None]*np.pi/4
            for level in range(3):
                sl=slice(8+level*len(seeds),8+(level+1)*len(seeds));disks[:,sl,1:3]=local
                disks[:,sl,3]=radii*(.5**level)
            disks=disks.reshape(-1,5);cert=cp.empty((len(disks),15));err=cp.empty((len(disks),2));ok=cp.empty(len(disks),cp.uint8)
            fit(rt,region,disks,cert,err,ok,alternatives=8,partial=True)
            cert=cert.reshape(len(ids),nc,15);ok=ok.reshape(len(ids),nc).astype(bool)
            for j in range(len(ids)):
                owner=ids[j];qxy=region.radii[rows[owner]]*parent_samples
                wanted=truth[owner].astype(bool);w=weights[owner];total=w.sum()
                missing_total=cp.where(rows[owner]<region.preserve_nonempty,cp.maximum(cp.sum(w*wanted),1e-30),total)
                # Include each local tangent combination directly. A focused
                # best fit can otherwise choose the same large fragment eight
                # times and never propose the small complementary crescent.
                start,count=map(int,group_spans[batch+j])
                lines=region.edges[start:start+count].reshape(-1,3)
                prior=region.geometry[rows[owner],13:25].reshape(4,3)
                lines=cp.concatenate((cp.array([[0.,0.,cp.inf]]),prior,lines))
                lines=lines[(cp.isfinite(lines[:,2]))|(cp.arange(len(lines))==0)]
                aa,bb=cp.triu_indices(len(lines));direct=cp.zeros((len(aa),15));direct[:,2]=region.radii[rows[owner]]
                direct[:,3:6]=direct[:,9:12]=lines[aa];direct[:,6:9]=direct[:,12:15]=lines[bb]
                candidates=cp.concatenate((cert[j],direct))
                if len(candidates)>40000:raise ClippingSubdivisionError('Sampled clipping candidate budget exceeded')
                cover=coverage(candidates,qxy)
                false=cp.sum(cover*(~wanted)[None]*w[None],axis=1)/total
                valid=cp.r_[ok[j],cp.ones(len(direct),bool)]&(false<=FIT_TOLERANCE*.25)
                union=cp.zeros(ns,bool);selected=[]
                # Usually the precomputed candidates suffice. Only uncovered
                # regions request extra child candidates centred on witnesses.
                for refinement in range(3):
                    left,excess=select_cover(rt,cover,wanted,w,valid,union,selected,
                        per_parent=per_parent,missing_total=missing_total,total=total,tolerance=FIT_TOLERANCE)
                    if left<=FIT_TOLERANCE or len(selected)>=per_parent or refinement==2:break
                    missing=cp.flatnonzero(wanted&~union)
                    if len(missing)>128:missing=missing[cp.linspace(0,len(missing)-1,128,dtype=cp.int64)]
                    centres=qxy[missing];lengths=cp.linalg.norm(centres,axis=1);rad=region.radii[rows[owner]]
                    rr=cp.minimum(rad-lengths,rad*.5)
                    extra_disks=cp.zeros((len(missing)*3,5));extra_disks[:,0]=owner;extra_disks[:,4]=-1.
                    extra_disks[:,1:3]=cp.tile(centres,(3,1));extra_disks[:,3]=cp.concatenate((rr,rr*.5,rr*.25))
                    additional=cp.empty((len(extra_disks),15));ae=cp.empty((len(extra_disks),2));ao=cp.empty(len(extra_disks),cp.uint8)
                    fit(rt,region,extra_disks,additional,ae,ao,alternatives=8,partial=True)
                    ac=coverage(additional,qxy);af=cp.sum(ac*(~wanted)[None]*w[None],axis=1)/total
                    valid=cp.r_[valid,ao.astype(bool)&(af<=FIT_TOLERANCE*.25)]
                    cover=cp.concatenate((cover,ac));candidates=cp.concatenate((candidates,additional))
                if left>FIT_TOLERANCE or excess>FIT_TOLERANCE:
                    # The 512-point fit may not see a thin surviving region
                    # present in the 896-point validation grid. Refining that
                    # same coarse fit again cannot propose a useful fragment.
                    # Refit only the uncovered witnesses, keeping both caps
                    # and the final missing/protrusion tolerances unchanged.
                    if cached and len(selected)<per_parent:
                        stats['clip_validation_refit_parents']=stats.get('clip_validation_refit_parents',0)+1
                        for retry in range(2):
                            missing=cp.flatnonzero(wanted&~union)
                            if len(missing)>128:missing=missing[cp.linspace(0,len(missing)-1,128,dtype=cp.int64)]
                            centres=qxy[missing];rad=region.radii[rows[owner]]
                            rr=cp.minimum(rad-cp.linalg.norm(centres,axis=1),rad*.5)
                            # Thin survivors may be smaller than the coarse
                            # R/8 minimum candidate. Refine actual CHILD
                            # geometry down to R/64, never shrink the parent.
                            levels=6
                            disks=cp.zeros((8+len(missing)*levels,5));disks[:,0]=owner;disks[:,4]=-1.
                            disks[:8,3]=rad;disks[:8,4]=cp.arange(8)*np.pi/4
                            disks[8:,1:3]=cp.tile(centres,(levels,1))
                            disks[8:,3]=(rr[None]*cp.exp2(-cp.arange(levels))[:,None]).ravel()
                            extra=cp.empty((len(disks),15));validation_error=cp.empty((len(disks),2));validation_ok=cp.empty(len(disks),cp.uint8)
                            fit(rt,region,disks,extra,validation_error,validation_ok,alternatives=8,partial=True,
                                validation=True,tolerance=FIT_TOLERANCE*.25)
                            extra_cover=coverage(extra,qxy)
                            false=cp.sum(extra_cover*(~wanted)[None]*w[None],axis=1)/total
                            candidates=cp.concatenate((candidates,extra));cover=cp.concatenate((cover,extra_cover))
                            valid=cp.r_[valid,validation_ok.astype(bool)&(false<=FIT_TOLERANCE*.25)]
                            left,excess=select_cover(rt,cover,wanted,w,valid,union,selected,
                                per_parent=per_parent,missing_total=missing_total,total=total,tolerance=FIT_TOLERANCE)
                            if left<=FIT_TOLERANCE or len(selected)>=per_parent:break
                    if left>FIT_TOLERANCE or excess>FIT_TOLERANCE:
                        unresolved.append(dict(gs=int(rows[owner].item()),children=len(selected),
                                               missing=left,excess=excess));continue
                if selected:
                    chosen=candidates[cp.asarray(selected)];children.append(chosen);parents.append(cp.full(len(selected),rows[owner],cp.int64))
        if unresolved:raise ClippingSubdivisionError(f'Sampled analytic cover search stalled (child limit {per_parent}): {unresolved[:5]}')
        if sum(len(x) for x in children)>max_children:raise ClippingSubdivisionError(f'Sampled analytic child cap {max_children} exceeded')
        cc=cp.concatenate(children) if children else cp.empty((0,15));pp=cp.concatenate(parents) if parents else cp.empty(0,cp.int64)
        keep=np.ones(n,bool);hostrows=cp.asnumpy(rows,stream=rt.stream);good=cp.asnumpy(accepted,stream=rt.stream)
        keep[hostrows[~good]]=False
        deferred=np.zeros(nr,bool)
        if cached and nr:
            deferred=cp.asnumpy((rows<region.preserve_nonempty)&~truth.any(axis=1)&~region.sample_truth.any(axis=1),stream=rt.stream)
            keep[hostrows[deferred]]=True
            good[deferred]=False
            stats['retained_region_empty_deferred']=int(deferred.sum())
        draw=snapshot.clipping.planes.copy();material=snapshot.material_clipping.planes.copy()
        c=cp.asnumpy(region.preview,stream=rt.stream);radius=cp.asnumpy(region.radii,stream=rt.stream)
        for target,sl in ((draw,slice(3,9)),(material,slice(9,15))):
            target[hostrows[good],:6]=c[good,sl];target[hostrows[good],6]=radius[hostrows[good]]
        if deferred.any():
            rr=hostrows[deferred];original=cp.asnumpy(region.geometry[cp.asarray(rr)],stream=rt.stream)
            draw[rr,:6]=original[:,13:19];material[rr,:6]=original[:,19:25]
        pp=cp.asnumpy(pp,stream=rt.stream);cc=cp.asnumpy(cc,stream=rt.stream)
    from cutsim.rendering.empty_residual_clip import trim_cut_interior,affected_old_rows
    candidates=affected_old_rows(rt,region)
    candidates=candidates[keep[candidates]]
    trimmed=trim_cut_interior(rt,snapshot,candidates,draw,material,getattr(region,'retirement_domain',None))
    stats['retained_region_certified_trimmed']=trimmed
    stats['retained_region_certified_checked']=len(candidates)
    stats['retained_region_empty_deferred']=int(deferred.sum())
    # New wall boundaries must already be fitted before proving that a former
    # circular floor limit is obsolete. Updating the input alone would let the
    # inherited fit reintroduce that limit during publication.
    from cutsim.rendering.analytic_continuation import refresh_planar_continuations
    stats.update(refresh_planar_continuations(snapshot,stats.get('analytic_input_old',0),draw,material,retained=keep))
    stats.update(clip_subdivision=True,clip_gpu_prepare=True,clip_partition_checked=nr,clip_partition_kept=int(keep[hostrows].sum()),
        clip_partition_parents=int((~keep[hostrows]).sum()),clip_partition_children=len(cc),clip_partition_ms=(perf_counter()-started)*1000,
        clip_sampled_fit_tolerance=FIT_TOLERANCE,clip_sampled_validation_points=ns,radius_limited_gs=0)
    snapshot.clipping=ProfileClip(draw,stats);snapshot.material_clipping=ProfileClip(material,dict(stats))
    return keep,pp,cc,stats
