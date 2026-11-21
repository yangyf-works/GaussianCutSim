"""Sample-first analytic face selection, then at most two fitted lines on CUDA.

Published support remains 2 sigma. Four equal-area radial rings cover 1.5
sigma, with the same 128 points and Gaussian weights as deletion.
No angular root search, parent resizing, or subdivision is used here.
Peer eligibility is established by GS overlap and supported boundary points,
not by the receiving centre's membership in the peer generating solid.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip
from cutsim.rendering.analytic_surfaces import pack_snapshot
from cutsim.rendering.gs_sample_evaluation import disc

from cutsim.rendering.analytic_residual_retirement import RESIDUAL_SIGMA, RESIDUAL_RADIAL, RESIDUAL_ANGULAR


def sample_grid():
    # region.radii remains the published 2-sigma support. Only the quadrature
    # changes: these are exactly the deletion pass's physical sample points.
    xy=disc(RESIDUAL_RADIAL,RESIDUAL_ANGULAR)*(RESIDUAL_SIGMA/CLIP_SIGMA)
    weight=np.exp(-.5*np.sum((xy*CLIP_SIGMA)**2,axis=1))/len(xy)
    return xy,weight


CUDA = r'''
#define MAX_FACE 128
#define MAX_CHOICE 134
__device__ void sample_section(const double* e,const double* a,double* q){
 double grad[3];q[5]=gs_quadric(e,a,grad);q[0]=q[1]=q[2]=0.;
 for(int j=0;j<3;j++)for(int k=0;k<3;k++){
  q[0]+=a[3+j]*e[3+3*j+k]*a[3+k];q[1]+=2*a[3+j]*e[3+3*j+k]*a[6+k];q[2]+=a[6+j]*e[3+3*j+k]*a[6+k];}
 q[3]=pdot(grad,a+3);q[4]=pdot(grad,a+6);
}
extern "C" __global__ void describe_groups(const double* g,const long long* owners,
 const long long* peers,const long long* starts,const long long* counts,const long long* witnesses,
 const double* equations,const long long* meta,const double* rec,const long long* spans,const double* planes,
 const double* signs,const double* radii,const long long* patch_ids,const double* samples,int ns,int ng,int finite_floor,
 double* coefficients,double* edges,unsigned char* enabled){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ng)return;
 const double* a=g+25*owners[i];long long peer=peers[i],own=patch_ids[owners[i]];
 double* q=coefficients+6*i;sample_section(equations+16*peer,a,q);for(int k=0;k<6;k++)q[k]*=signs[i];
 bool planar=signs[i]<0.&&!meta[4*peer+3]&&!meta[4*own+3];for(int j=3;j<12;j++)if(equations[16*own+j]!=0.)planar=false;
 bool finite=finite_floor&&signs[i]>0.&&!meta[4*peer+3];enabled[i]=0;
 // A centre outside the peer solid must not reject a disk crossing its edge.
 // Boundary points must still be supported by an admitted peer GS below.
 if(finite&&!meta[4*own+3]&&surface_clear_of_solid(equations+16*own,meta+4*peer,rec,spans,planes))return;
 enabled[i]=finite?4:((finite_floor&&planar&&physical_planar_floor(equations+16*own,meta+4*own,a,spans,planes))?3:1);
}
extern "C" __global__ void describe_sweep(const double* g,const long long* rows,int nr,
 const double* eq,int ne,const double* rec,const long long* spans,const double* planes,int parts,
 double* coefficients,double* edges,unsigned char* active){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr*ne)return;
 sample_section(eq+16*(i%ne),g+25*rows[i/ne],coefficients+6*i);active[i]=2;
 for(int k=0;k<8;k++){edges[24*i+3*k]=edges[24*i+3*k+1]=0.;edges[24*i+3*k+2]=INFINITY;}
}
__device__ bool sample_cut(const double* p,const double* rec,const long long* spans,const double* planes,int parts){
 for(int i=0;i<parts;i++){
  bool yes=true;for(long long j=spans[2*i];j<spans[2*i+1];j++){
   const double* h=planes+5*j;if(pdot(h,p)+h[3]<(h[4]?1e-8:-1e-10)){yes=false;break;}}
  const double* r=rec+14*i;if(yes&&r[13]){double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]};yes=gs_quadratic_form(r+3,d)<r[12]*r[12]-1e-8;}
  if(yes)return true;
 }return false;
}
struct SampleContext {
 const double *g,*a,*q,*rec,*planes,*cut_rec,*cut_planes;
 const long long *meta,*spans,*cut_spans,*ctx,*witness;int parts,kind;
};
__device__ bool sample_boundary(const SampleContext& c,double x,double y,double X,double Y,double* result){
 double f=poly(c.q,x,y),F=poly(c.q,X,Y);if((f>=0.)==(F>=0.))return false;
 double dx=X-x,dy=Y-y,A=c.q[0]*dx*dx+c.q[1]*dx*dy+c.q[2]*dy*dy;
 double B=(2*c.q[0]*x+c.q[1]*y+c.q[3])*dx+(c.q[1]*x+2*c.q[2]*y+c.q[4])*dy,t;
 if(fabs(A)<1e-14){if(fabs(B)<1e-20)return false;t=-f/B;}
 else{double det=B*B-4*A*f;if(det<0.)return false;double z=-.5*(B+copysign(sqrt(det),B));
  double t1=z/A,t2=fabs(z)>1e-25?f/z:-B/(2*A);t=t1>=0.&&t1<=1.?t1:t2;}
 if(t<0.||t>1.||!isfinite(t))return false;x+=t*dx;y+=t*dy;
 // Existing halfplanes admit the point with a small arithmetic tolerance.
 for(int j=13;j<25;j+=3)if(c.a[j]*x+c.a[j+1]*y+c.a[j+2]<-1e-8)return false;
 double nx=2*c.q[0]*x+c.q[1]*y+c.q[3],ny=c.q[1]*x+2*c.q[2]*y+c.q[4],len=hypot(nx,ny);
 if(len<1e-12)return false;nx/=len;ny/=len;double p[3];gs_sample_point(c.a,x,y,p);
 if(c.ctx[0]>=0){
  // Require local GS support at the actual boundary, with inherited clips.
  bool supported=false;for(long long k=c.ctx[1];k<c.ctx[1]+c.ctx[2];k++)if(admitted(c.g+25*c.witness[k],p)){supported=true;break;}
  if(!supported)return false;
 }
 if(c.kind==2||c.kind==3||c.kind==4){
  double plus[3],minus[3];for(int j=0;j<3;j++){double d=1e-6*(1.+c.a[12])*(nx*c.a[3+j]+ny*c.a[6+j]);plus[j]=p[j]+d;minus[j]=p[j]-d;}
  bool kp,km;
  if(c.kind==2){kp=!sample_cut(plus,c.cut_rec,c.cut_spans,c.cut_planes,c.parts);km=!sample_cut(minus,c.cut_rec,c.cut_spans,c.cut_planes,c.parts);}
  else if(c.kind==3){kp=floor_domain_inside(c.meta+4*c.ctx[3],plus,c.rec,c.spans,c.planes);km=floor_domain_inside(c.meta+4*c.ctx[3],minus,c.rec,c.spans,c.planes);}
  else{bool stock=c.meta[4*c.ctx[3]+3];kp=!finite_cut_inside(plus,c.meta+4*c.ctx[0],c.rec,c.spans,c.planes,stock);km=!finite_cut_inside(minus,c.meta+4*c.ctx[0],c.rec,c.spans,c.planes,stock);}
  if(kp==km)return false;if(!kp){nx=-nx;ny=-ny;}
 }
 result[0]=x;result[1]=y;result[2]=nx;result[3]=ny;return true;
}
// Ring neighbours bracket boundary crossings. No centre-through ray search.
extern "C" __global__ void classify_faces(const double* g,const double* radii,const long long* owners,
 const double* q,const long long* context,const long long* witness,const long long* meta,const double* rec,
 const long long* spans,const double* planes,const double* cr,const long long* cs,const double* cp,int parts,
 const double* xy,int ng,unsigned char* enabled,unsigned int* masks,double* proximity){
 int group=blockIdx.x,lane=threadIdx.x;if(group>=ng)return;int kind=enabled[group];if(!kind)return;
 const double* a=g+25*owners[group];double R=radii[owners[group]];
 SampleContext c={g,a,q+6*group,rec,planes,cr,cp,meta,spans,cs,context+4*group,witness,parts,kind};
 __shared__ int found,flip;__shared__ double near[128];if(!lane){found=0;flip=0;}__syncthreads();double best=INFINITY;
 for(int j=lane;j<128;j+=128){
  double x=R*xy[2*j],y=R*xy[2*j+1];bool keep=poly(c.q,x,y)>=-1e-10;
  unsigned int bits=__ballot_sync(0xffffffff,keep);if((lane&31)==0)masks[group*4+j/32]=bits;
  int neighbours[2]={j/32*32+(j+1)%32,j+32};
  for(int k=0;k<2;k++)if(neighbours[k]<128){double p[4];int other=neighbours[k];
   if(sample_boundary(c,x,y,R*xy[2*other],R*xy[2*other+1],p)){
    atomicOr(&found,1);best=fmin(best,hypot(p[0],p[1])/R);
    double gx=2*c.q[0]*p[0]+c.q[1]*p[1]+c.q[3],gy=c.q[1]*p[0]+2*c.q[2]*p[1]+c.q[4];
    if(gx*p[2]+gy*p[3]<0.)atomicOr(&flip,1);
   }}
 }
 near[lane]=best;__syncthreads();for(int step=64;step;step/=2){if(lane<step)near[lane]=fmin(near[lane],near[lane+step]);__syncthreads();}
 // A stock halfspace remains a TRUTH constraint even when the whole disk
 // lies outside it: there is then no crossing from which to fit an edge.
 if(!lane){enabled[group]=(found||c.ctx[0]==-3)?kind:0;proximity[group]=near[0];}
 if(flip&&lane<4)masks[4*group+lane]=~masks[4*group+lane];
}
// Compare ideal analytic masks BEFORE fitting lines. Ring weights are constant,
// so weighted pair errors use popcounts instead of a loop over 128 points.
extern "C" __global__ void choose_faces(const double* g,const long long* rows,const long long* starts,
 const long long* counts,const unsigned char* enabled,const unsigned int* masks,const double* proximity,
 const double* radii,const double* xy,const double* weights,const unsigned char* truth,const long long* work,
 int nw,int preserve,long long* selected){
 int item=blockIdx.x,lane=threadIdx.x;if(item>=nw)return;long long i=work[item],row=rows[i];const double* a=g+25*row;
 __shared__ unsigned int bits[MAX_CHOICE*4],target[4];__shared__ long long source[MAX_CHOICE];
 __shared__ double distance[MAX_CHOICE],loss[128];__shared__ int choice[128],number;
 if(!lane){number=2;source[0]=-1;source[1]=-2;distance[0]=distance[1]=0.;
  for(int k=0;k<4;k++)if(isfinite(a[15+3*k])){source[number]=-3-k;distance[number]=fabs(a[15+3*k])/radii[row];number++;}
  for(long long k=starts[i];k<starts[i]+counts[i];k++)if(enabled[k]&&isfinite(proximity[k])){source[number]=k;distance[number]=proximity[k];number++;}}
 __syncthreads();
 for(int j=lane;j<128;j+=128){unsigned int m=__ballot_sync(0xffffffff,truth[128*i+j]);if((lane&31)==0)target[j/32]=m;}
 for(int z=lane;z<number*4;z+=128){int k=z/4,ring=z%4;long long src=source[k];unsigned int m=0;
  if(src==-1)m=0xffffffff;else if(src>=0)m=masks[4*src+ring];
  else if(src<=-3){const double* e=a+13+3*(-src-3);for(int j=0;j<32;j++){
   int p=32*ring+j;if(e[0]*radii[row]*xy[2*p]+e[1]*radii[row]*xy[2*p+1]+e[2]>=-1e-10)m|=1u<<j;}}
  bits[z]=m;
 }
 __syncthreads();double total=0.,kept=0.;for(int r=0;r<4;r++){total+=32*weights[32*r];kept+=__popc(target[r])*weights[32*r];}
 // New and old GS must both preserve surviving witnesses. Normalizing a
 // new GS's loss by its whole disk makes deleting a small residual win.
 double missing_denom=kept>0.?kept:total,best=1e100;int best_pair=-1,n=number,pairs=n*(n+1)/2;
 for(int z=lane;z<pairs;z+=128){int k=0,t=z;while(t>=n-k){t-=n-k;k++;}int h=k+t;
  double miss=0.,over=0.;for(int r=0;r<4;r++){unsigned int pred=bits[4*k+r]&bits[4*h+r];
   miss+=__popc(target[r]&~pred)*weights[32*r];over+=__popc(~target[r]&pred)*weights[32*r];}
  // Reject invalid empty fits BEFORE selecting the best pair; the later
  // native check would otherwise restore an entirely unclipped parent.
  if(kept>0.&&miss>=kept-1e-15)continue;
  int edges=(source[k]!=-1)+(h!=k&&source[h]!=-1);
  double score=miss/fmax(missing_denom,1e-30)+over/fmax(total,1e-30)+1e-10*edges+1e-12*(distance[k]+distance[h]);
  if(score<best){best=score;best_pair=z;}
 }
 loss[lane]=best;choice[lane]=best_pair;__syncthreads();
 for(int step=64;step;step/=2){if(lane<step&&(loss[lane+step]<loss[lane]||(loss[lane+step]==loss[lane]&&choice[lane+step]<choice[lane]))){loss[lane]=loss[lane+step];choice[lane]=choice[lane+step];}__syncthreads();}
 if(!lane){int k=0,t=choice[0];while(t>=n-k){t-=n-k;k++;}int h=k+t;selected[2*i]=source[k];selected[2*i+1]=h==k?-1:source[h];}
}
extern "C" __global__ void fit_faces(const double* g,const double* radii,const long long* rows,
 const long long* owners,const double* q,const long long* context,const long long* witness,
 const long long* meta,const double* rec,const long long* spans,const double* planes,
 const double* cr,const long long* cs,const double* cp,int parts,const unsigned char* enabled,
 const unsigned int* masks,const double* xy,const double* weights,const unsigned char* truth,
 int ng,double* edges,double* errors,unsigned char* exposed){
 int group=blockIdx.x,lane=threadIdx.x;if(group>=ng)return;long long owner=owners[group],row=rows[owner];
 const double* a=g+25*row;double R=radii[row];
 __shared__ double sums[128*8],line[3];__shared__ int visible;
 if(!lane){visible=0;line[0]=line[1]=0.;line[2]=INFINITY;}__syncthreads();
 if(!enabled[group]){if(!lane){errors[2*group]=errors[2*group+1]=0.;exposed[group]=0;
  for(int j=0;j<3;j++)edges[24*group+j]=line[j];}return;}
  SampleContext c={g,a,q+6*group,rec,planes,cr,cp,meta,spans,cs,context+4*group,witness,parts,enabled[group]};
  double m[8]={0.};for(int j=lane;j<128;j+=128){int neighbours[2]={j/32*32+(j+1)%32,j+32};
   for(int k=0;k<2;k++)if(neighbours[k]<128){double p[4];int other=neighbours[k];
    if(!sample_boundary(c,R*xy[2*j],R*xy[2*j+1],R*xy[2*other],R*xy[2*other+1],p))continue;
    double x=p[0],y=p[1],r2=(x*x+y*y)/(R*R),w=exp(-2*r2);
    if(truth[128*owner+j]!=truth[128*owner+other])atomicOr(&visible,1);
    m[0]+=w;m[1]+=w*x;m[2]+=w*y;m[3]+=w*x*x;m[4]+=w*x*y;m[5]+=w*y*y;m[6]+=w*p[2];m[7]+=w*p[3];
   }}
  for(int j=0;j<8;j++)sums[8*lane+j]=m[j];__syncthreads();
  for(int step=64;step;step/=2){if(lane<step)for(int j=0;j<8;j++)sums[8*lane+j]+=sums[8*(lane+step)+j];__syncthreads();}
  if(!lane){double* out=line;out[0]=out[1]=0.;out[2]=INFINITY;
   if(sums[0]>0.){double x=sums[1]/sums[0],y=sums[2]/sums[0],xx=sums[3]/sums[0]-x*x,yy=sums[5]/sums[0]-y*y,xyc=sums[4]/sums[0]-x*y;
    double angle=.5*atan2(2*xyc,xx-yy),nx=-sin(angle),ny=cos(angle);
    if(xx+yy<1e-16){double len=hypot(sums[6],sums[7]);nx=len>0.?sums[6]/len:1.;ny=len>0.?sums[7]/len:0.;}
    if(nx*sums[6]+ny*sums[7]<0.){nx=-nx;ny=-ny;}
    out[0]=nx;out[1]=ny;out[2]=-nx*x-ny*y;
   }}__syncthreads();
 double miss=0.,over=0.,target=0.,total=0.;
 for(int j=lane;j<128;j+=128){double x=R*xy[2*j],y=R*xy[2*j+1],w=weights[j];
  bool wanted=(masks[4*group+j/32]>>(j%32))&1u,yes=line_yes(line,x,y);
  total+=w;if(wanted)target+=w;if(wanted&&!yes)miss+=w;if(!wanted&&yes)over+=w;}
 sums[8*lane]=miss;sums[8*lane+1]=over;sums[8*lane+2]=target;sums[8*lane+3]=total;__syncthreads();
 for(int step=64;step;step/=2){if(lane<step)for(int j=0;j<4;j++)sums[8*lane+j]+=sums[8*(lane+step)+j];__syncthreads();}
 if(!lane){for(int j=0;j<3;j++)edges[24*group+j]=line[j];
  errors[2*group]=sums[0]/fmax(sums[2],1e-30);errors[2*group+1]=sums[1]/fmax(sums[3],1e-30);exposed[group]=visible;}
}
extern "C" __global__ void fit_selected(const double* g,const double* radii,const long long* rows,
 const double* edges,const double* xy,const double* weights,const unsigned char* truth,
 const long long* work,int nw,int preserve,const long long* selected,double* preview,double* errors,unsigned char* accepted){
 int item=blockIdx.x,lane=threadIdx.x;if(item>=nw)return;long long i=work[item],row=rows[i];const double* a=g+25*row;double R=radii[row];
 __shared__ double sums[128*8],line[6];
 if(!lane)for(int slot=0;slot<2;slot++){
  long long group=selected[2*i+slot];double* out=line+3*slot;out[0]=out[1]=0.;out[2]=group==-2?-1.:INFINITY;
  if(group<=-3)for(int j=0;j<3;j++)out[j]=a[13+3*(-group-3)+j];
  if(group>=0)for(int j=0;j<3;j++)out[j]=edges[24*group+j];
 }__syncthreads();
 double miss=0.,over=0.,total=0.,target=0.;for(int j=lane;j<128;j+=128){double x=R*xy[2*j],y=R*xy[2*j+1],w=weights[j];
  bool wanted=truth[128*i+j],yes=line_yes(line,x,y)&&line_yes(line+3,x,y);
  total+=w;if(wanted)target+=w;if(wanted&&!yes)miss+=w;if(!wanted&&yes)over+=w;}
 sums[8*lane]=miss;sums[8*lane+1]=over;sums[8*lane+2]=total;sums[8*lane+3]=target;__syncthreads();
 for(int step=64;step;step/=2){if(lane<step)for(int j=0;j<4;j++)sums[8*lane+j]+=sums[8*(lane+step)+j];__syncthreads();}
 if(!lane){double* out=preview+15*i;out[0]=out[1]=0.;out[2]=R;for(int j=0;j<6;j++)out[3+j]=out[9+j]=line[j];
  double m=sums[0]/fmax(sums[3]>0.?sums[3]:sums[2],1e-30),e=sums[1]/fmax(sums[2],1e-30);
  errors[2*i]=m;errors[2*i+1]=e;accepted[i]=m<=.02&&e<=.02;
 }
}
'''


def kernels(rt):
    if not hasattr(rt,'sample_priority_kernels'):
        from cutsim.rendering.sampled_analytic_clip import CUDA as shared
        prefix=shared[:shared.index('extern "C" __global__ void evaluate_groups')]
        module=rt.cp.RawModule(code=prefix+CUDA,options=('--fmad=false',))
        rt.sample_priority_kernels={name:module.get_function(name) for name in
            ('describe_groups','describe_sweep','classify_faces','fit_faces','choose_faces','fit_selected')}
    return rt.sample_priority_kernels


def refit(rt,region,work=None):
    cp=rt.cp;nr=len(region.rows)
    if work is None:work=cp.arange(nr,dtype=cp.int64)
    if not len(work):return
    _,meta,rec,spans,planes=region.floor_domain_data
    _,cr,cs,cplanes=region.sweep_data
    k=kernels(rt)
    # Keep each face's one-line approximation and error separately. Truth is
    # still the all-face residual mask reused from deletion, never a two-line mask.
    ng=len(region.enabled);owners=cp.repeat(cp.arange(nr,dtype=cp.int64),region.counts)
    region.face_line_error=cp.zeros((ng,2));region.face_line_exposed=cp.zeros(ng,cp.uint8)
    if ng:k['fit_faces']((ng,),(128,),(region.geometry,region.radii,region.rows,owners,region.coefficients,
        region.group_context,region.witnesses,meta,rec,spans,planes,cr,cs,cplanes,np.int32(len(cr)),region.enabled,
        region.face_masks,region.sample_xy,region.sample_weights,region.sample_truth,np.int32(ng),
        region.edges,region.face_line_error,region.face_line_exposed))
    bad=(region.face_line_exposed!=0)&(cp.max(region.face_line_error,axis=1)>.02)
    region.face_line_failed=cp.bincount(owners[bad],minlength=nr)>0
    k['choose_faces']((len(work),),(128,),(region.geometry,region.rows,region.starts,region.counts,
        region.enabled,region.face_masks,region.face_proximity,region.radii,region.sample_xy,region.sample_weights,
        region.sample_truth,work,np.int32(len(work)),np.int32(region.preserve_nonempty),region.selected_faces))
    k['fit_selected']((len(work),),(128,),(region.geometry,region.radii,region.rows,region.edges,
        region.sample_xy,region.sample_weights,region.sample_truth,work,np.int32(len(work)),
        np.int32(region.preserve_nonempty),region.selected_faces,region.preview,region.error,region.accepted))


def evaluate_and_fit(rt,region,*,cached_truth=None):
    from cutsim.rendering.retained_region_samples import CUDA as retained_cuda
    cp=rt.cp;nr=len(region.rows);ng=len(region.enabled);xy,weight=sample_grid()
    region.sample_xy=cp.asarray(xy);region.sample_weights=cp.asarray(weight)
    region.validation_samples=region.sample_xy
    if not hasattr(region,'sweep_data'):
        region.sweep_data=(cp.zeros(len(region.geometry),cp.uint8),cp.empty((0,14)),cp.empty((0,2),cp.int64),cp.empty((0,5)))
    affected,cr,cs,cplanes=region.sweep_data;_,meta,rec,spans,planes=region.floor_domain_data
    region.face_masks=cp.zeros((ng,4),cp.uint32);region.face_proximity=cp.full(ng,cp.inf)
    if ng:kernels(rt)['classify_faces']((ng,),(128,),
        (region.geometry,region.radii,cp.repeat(region.rows,region.counts),region.coefficients,region.group_context,
         region.witnesses,meta,rec,spans,planes,cr,cs,cplanes,np.int32(len(cr)),region.sample_xy,np.int32(ng),
         region.enabled,region.face_masks,region.face_proximity))
    if not hasattr(rt,'priority_truth_kernel'):
        rt.priority_truth_kernel=cp.RawKernel('#define SHARED_RESIDUAL_CLIP_UNION\n'+retained_cuda,
            'shared_truth',options=('--fmad=false',))
    def truth(samples,work=None):
        work=cp.arange(nr,dtype=cp.int64) if work is None else work
        ns=len(samples);count=len(work);result=cp.empty((count,ns),cp.uint8);weights=cp.empty(ns)
        if count:rt.priority_truth_kernel(((count*ns+127)//128,),(128,),
            (region.geometry,region.radii,region.rows[work],region.starts[work],region.counts[work],region.coefficients,region.enabled,
             region.peer_surfaces,samples,np.int32(ns),np.int32(count),affected,cr,cs,cplanes,np.int32(len(cr)),
             *region.floor_domain_data,result,weights))
        return result,weights
    region.sample_truth=truth(region.sample_xy)[0] if cached_truth is None else cached_truth
    if cached_truth is not None:
        missing=cp.flatnonzero(~region.cached_truth_available).astype(cp.int64)
        # Only newly admitted candidates without R data need new truth queries.
        if len(missing):region.sample_truth[missing]=truth(region.sample_xy,missing)[0]
    region.validation_truth=region.sample_truth
    region.validation_weights=region.sample_weights
    region.validation_is_holdout=False
    region.selected_faces=cp.full((nr,2),-1,cp.int64)
    work=None
    if cached_truth is not None:
        boundary=cp.zeros(nr,cp.bool_)
        boundary[cp.repeat(cp.arange(nr),region.counts)[(region.enabled!=0)&cp.isfinite(region.face_proximity)]]=True
        unchanged=region.cached_unchanged&~boundary
        # No newly observed boundary: preserve inherited edges, including
        # those outside the 1.5-sigma sample grid but inside rendered support.
        region.preview[unchanged,:2]=0.;region.preview[unchanged,2]=region.radii[region.rows[unchanged]]
        region.preview[unchanged,3:15]=region.geometry[region.rows[unchanged],13:25]
        region.error[unchanged]=0.;region.accepted[unchanged]=1
        work=cp.flatnonzero(~unchanged).astype(cp.int64)
    refit(rt,region,work)


def update_sample_priority_clipping(snapshot,previous,kept,*,collect_edges=True,gpu_prepare=True,
        preserve_nonempty=False,retained_sweep=None,current_surface_ids=None,stock_bounds=None,
        residual_samples=None,new_samples=None):
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.rendering.sampled_analytic_clip import grouped_regions,SampledEdges
    from cutsim.rendering.retained_region_samples import prepare
    from cutsim.rendering.planar_floor_domain import extend_round_floors
    from cutsim.rendering.coplanar_coverage import extend_coverage
    if not collect_edges:raise ValueError('Sample-first clipping requires edge publication')
    rt=runtime();cp=rt.cp;start=perf_counter();n=len(snapshot.endpoints['positions']);nold=int(np.count_nonzero(kept))
    draw=np.tile([0.,0.,np.inf,0.,0.,np.inf,np.inf],(n,1));material=draw.copy()
    draw[:nold]=previous.clipping.planes[kept];material[:nold]=(previous.material_clipping or previous.clipping).planes[kept]
    extended=extend_round_floors(snapshot,nold,draw,material) if retained_sweep is not None else 0
    radius=CLIP_SIGMA*snapshot.gaussian.attributes['scale'][:,0].astype(float)
    draw[nold:,6]=radius[nold:];material[nold:,6]=radius[nold:]
    g=np.ascontiguousarray(np.column_stack((snapshot.endpoints['positions'],snapshot.gaussian.u,snapshot.gaussian.v,
        np.cross(snapshot.gaussian.u,snapshot.gaussian.v),np.where((draw[:,6]>0)&(material[:,6]>0),radius,0.),draw[:,:6],material[:,:6])))
    with rt.context():
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        region,detail=grouped_regions(g,snapshot.material_surface_ids,nold,pack_snapshot(snapshot),radius,
            preserve_nonempty=nold if preserve_nonempty else 0,defer_fit=True,sample_first=True)
        if extended:
            # Floor continuation may release inherited tangents before fitting.
            # Rollback must retain the input clips, not that prepared state.
            region.rollback_geometry=region.geometry.copy()
            region.rollback_geometry[:nold,13:19]=cp.asarray(previous.clipping.planes[kept,:6])
            region.rollback_geometry[:nold,19:25]=cp.asarray((previous.material_clipping or previous.clipping).planes[kept,:6])
        if retained_sweep is not None:detail.update(prepare(rt,region,snapshot,nold,retained_sweep,current_surface_ids,
            stock_bounds=stock_bounds,sample_first=True))
        cached=None
        if residual_samples is not None:
            cached,reuse_stats=select_residual_rows(rt,region,kept,residual_samples,new_samples)
            detail.update(reuse_stats)
        evaluate_and_fit(rt,region,cached_truth=cached)
        if cached is None:detail.update(extend_coverage(rt,region,nold))
        from cutsim.rendering.twoedge_validation import validate_twoedge
        detail.update(validate_twoedge(rt,region))
        detail.update(analytic_face_line_failed=int(cp.count_nonzero(getattr(region,'face_line_failed',cp.zeros(0,cp.bool_))).item()),
                      analytic_face_line_policy='one_line_per_face')
        end.record();end.synchronize();ms=float(cp.cuda.get_elapsed_time(begin,end))
        rows=cp.asnumpy(region.rows,stream=rt.stream);preview=cp.asnumpy(region.preview,stream=rt.stream)
        selected=cp.asnumpy(region.selected_faces,stream=rt.stream)
        failed_rows=(cp.asnumpy(region.rows[region.twoedge_fallback.astype(bool)],stream=rt.stream)
                     if detail['twoedge_invalid_or_empty'] else np.empty(0,np.int64))
    draw[rows,:6]=preview[:,3:9];material[rows,:6]=preview[:,9:15]
    old_failed=failed_rows[failed_rows<nold]
    if len(old_failed):
        source_rows=np.flatnonzero(kept)[old_failed]
        draw[old_failed]=previous.clipping.planes[source_rows]
        material[old_failed]=(previous.material_clipping or previous.clipping).planes[source_rows]
    stats=dict(backend='cuda',mode='sample_priority',clipping_source='sample_priority',material_source='current_gs',
        kernel_ms=ms,total_ms=(perf_counter()-start)*1000,analytic_patch_count=len(snapshot.analytic_patches),
        recomputed_gs=len(rows),reused_gs=n-len(rows),clip_edge_method='sample-priority',
        boundary_search_sigma=RESIDUAL_SIGMA,edge_evaluation_sigma=RESIDUAL_SIGMA,
        sample_fit_points=128,sample_primary_points=128,sample_outer_points=0,
        sample_first=True,peer_domain_test='gs_supported_boundary',sample_selected_analytic_faces=int(np.count_nonzero(selected>=0)),
        sample_max_faces_per_gs=int(np.count_nonzero(selected>=0,axis=1).max(initial=0)),
        angular_root_search=False,radius_limited_gs=0,preserve_nonempty=preserve_nonempty,
        analytic_input_old=nold,planar_floor_domains_extended=extended,
        rejected_extrapolated_boundaries=0,accepted_finite_boundaries=detail['analytic_group_active'],
        boundary_segment_queries=0,reselected_boundaries=0,gs_pair_count=detail['candidate_gs_pairs'],
        clipped_gs=int(np.count_nonzero(np.minimum(draw[:,2],draw[:,5])<radius)),**detail)
    raw=SampledEdges(rows,preview,region)
    return ProfileClip(draw,stats,raw),ProfileClip(material,dict(stats),raw)


def select_residual_rows(rt,region,kept,old,new):
    """Route every surviving R candidate directly into clipping.

    R=1 is not a non-intersection certificate. Reuse all available R grids,
    including unchanged ones; only uncached candidates need new evaluation.
    Sweep/contact and peer filters still determine eligible boundaries, but
    must not discard an old GS already retained by the retirement stage.
    """
    cp=rt.cp;n=len(region.geometry)
    mapping=cp.full(len(kept),-1,cp.int64)
    mapping[cp.asarray(kept)]=cp.arange(int(np.count_nonzero(kept)),dtype=cp.int64)
    survived=mapping[old.rows]>=0
    retained_rows=mapping[old.rows[survived]]
    cached_old=survived&cp.any(old.before,axis=1)
    old_rows=mapping[old.rows[cached_old]]
    new_rows=cp.empty(0,cp.int64) if new is None else new.rows
    cached_rows=cp.r_[old_rows,new_rows]
    cached=old.after[cached_old] if new is None else cp.concatenate((old.after[cached_old],new.after))
    owner=cp.repeat(region.rows,region.counts)
    affected,cr,cs,planes=region.sweep_data
    eligible=affected.astype(cp.bool_)
    eligible[owner[region.enabled!=0]]=True
    eligible[retained_rows]=True
    eligible[new_rows]=True
    target=cp.flatnonzero(eligible).astype(cp.int64)
    patch,meta,*_=region.floor_domain_data
    region.untouched_stock=cp.asnumpy(~eligible&(meta[patch,3]!=0),stream=rt.stream)
    take=eligible[owner]
    owner=owner[take]
    for name in ('coefficients','edges','enabled','peer_surfaces','group_context'):
        setattr(region,name,getattr(region,name)[take])
    # Keep sampled parents even when no analytic proposal crosses their grid.
    missing=target[~cp.isin(target,owner)]
    if len(missing):
        owner=cp.r_[owner,missing]
        region.coefficients=cp.concatenate((region.coefficients,cp.zeros((len(missing),6))))
        blank=cp.zeros((len(missing),8,3));blank[:,:,2]=cp.inf
        region.edges=cp.concatenate((region.edges,blank))
        region.enabled=cp.r_[region.enabled,cp.zeros(len(missing),cp.uint8)]
        region.peer_surfaces=cp.r_[region.peer_surfaces,cp.full(len(missing),-1,cp.int64)]
        context=cp.zeros((len(missing),4),cp.int64);context[:,0]=-1
        context[:,3]=region.floor_domain_data[0][missing]
        region.group_context=cp.concatenate((region.group_context,context))
    order=cp.argsort(owner)
    for name in ('coefficients','edges','enabled','peer_surfaces','group_context'):
        setattr(region,name,getattr(region,name)[order])
    region.rows,region.starts,region.counts=cp.unique(owner[order],return_index=True,return_counts=True)
    nr=len(region.rows);region.preview=cp.empty((nr,15));region.error=cp.empty((nr,2));region.accepted=cp.empty(nr,cp.uint8)
    # Do not narrow the Sweep mask with a sample-based partial-cut decision.
    lookup=cp.full(n,-1,cp.int64);lookup[cached_rows]=cp.arange(len(cached_rows),dtype=cp.int64)
    indices=lookup[region.rows];available=indices>=0
    truth=cp.empty((nr,128),cp.uint8);truth[available]=cached[indices[available]]
    region.cached_truth_available=available
    nochange=cp.zeros(n,cp.bool_)
    nochange[old_rows]=~cp.any(old.before[cached_old]!=old.after[cached_old],axis=1)
    region.cached_unchanged=nochange[region.rows]
    nold=int(np.count_nonzero(kept))
    return truth,dict(clipping_residual_reused=int(cp.count_nonzero(available&(region.rows<nold)).item()),
                      clipping_residual_retained_candidates=len(retained_rows),
                      clipping_new_edge_sampled=len(new_rows),
                      clipping_geometric_candidates=nr,
                      clipping_unchanged_residual_reused=int(cp.count_nonzero(eligible[old_rows]&~cp.any(old.before[cached_old]!=old.after[cached_old],axis=1)).item()),
                      clipping_uncached_candidates=int(cp.count_nonzero(~available).item()),
                      clipping_truth_recomputed_old=int(cp.count_nonzero(~available&(region.rows<nold)).item()),clipping_sample_points=128,
                      clipping_sample_domain='draw_material_union')
