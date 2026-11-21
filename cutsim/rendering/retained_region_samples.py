"""Transient 2-sigma residual samples shared by clipping and child selection.

No R quota and no sample-based full-deletion certificate. Cutter membership,
inherited clips and analytic peer limits are evaluated once per parent grid.
Child fitting reuses those truth/opacity arrays, not repeated Sweep queries.
"""
import numpy as np
from cutsim.rendering.gs_area_retirement import KERNEL as AREA_KERNEL
from cutsim.rendering.gs_sample_evaluation import disc
from cutsim.rendering.profile_cut_clip import pack_cutters
from cutsim.rendering.clip_search import update_candidates
from cutsim.rendering.planar_floor_domain import CUDA as FLOOR_CUDA
from cutsim.rendering.finite_surface_contact import CUDA as CONTACT_CUDA

CUDA='#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))\n'+AREA_KERNEL[:AREA_KERNEL.index('extern "C"')]+FLOOR_CUDA+CONTACT_CUDA+r'''
extern "C" __global__ void contact_filter(const long long* patch,const double* equations,const long long* meta,int n,
 const double* rec,const long long* spans,const double* planes,int parts,unsigned char* affected){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n||!affected[row]||meta[4*patch[row]+3])return;
 long long m[4]={0,parts,0,0};
 if(surface_clear_of_solid(equations+16*patch[row],m,rec,spans,planes))affected[row]=0;
}
__device__ double polynomial(const double* q,double x,double y){
 return q[0]*x*x+q[1]*x*y+q[2]*y*y+q[3]*x+q[4]*y+q[5];
}
extern "C" __global__ void sweep_edges(const double* g,const long long* rows,int nr,
 const double* eq,int ne,const double* rec,const long long* spans,const double* planes,int parts,
 double* coefficients,double* edges,unsigned char* active){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=nr*ne)return;
 const double* a=g+25*rows[id/ne];const double* e=eq+16*(id%ne);double* q=coefficients+6*id;
 double grad[3];q[5]=gs_quadric(e,a,grad);q[0]=q[1]=q[2]=0.;
 for(int j=0;j<3;j++)for(int k=0;k<3;k++){
  q[0]+=a[3+j]*e[3+3*j+k]*a[3+k];q[1]+=2*a[3+j]*e[3+3*j+k]*a[6+k];q[2]+=a[6+j]*e[3+3*j+k]*a[6+k];
 }
 q[3]=q[4]=0.;for(int j=0;j<3;j++){q[3]+=grad[j]*a[3+j];q[4]+=grad[j]*a[6+j];}
 int found=0;double R=a[12];
 for(int j=0;j<8;j++){edges[24*id+3*j]=edges[24*id+3*j+1]=0.;edges[24*id+3*j+2]=INFINITY;}
 // Roots on radial lines are direct linear/quadratic solves. Test both
 // sides against the WHOLE finite Sweep to reject internal partition faces.
 for(int j=0;j<32&&found<8;j++){
  double t=(j*13%32)*3.141592653589793/32.,ux=cos(t),uy=sin(t);
  double A=q[0]*ux*ux+q[1]*ux*uy+q[2]*uy*uy,B=q[3]*ux+q[4]*uy,C=q[5],root[2];int n=0;
  if(fabs(A)<1e-12){if(fabs(B)>1e-12)root[n++]=-C/B;}
  else{double det=B*B-4*A*C;if(det>=0.){double z=-.5*(B+copysign(sqrt(det),B));
   if(fabs(z)>1e-20){root[n++]=z/A;root[n++]=C/z;}else root[n++]=-B/(2*A);}}
  for(int k=0;k<n&&found<8;k++){
   double x=root[k]*ux,y=root[k]*uy;if(hypot(x,y)>R+1e-9)continue;
   if(!gs_sample_admitted(a+13,x,y)||!gs_sample_admitted(a+19,x,y))continue;
   double nx=2*q[0]*x+q[1]*y+q[3],ny=q[1]*x+2*q[2]*y+q[4],length=hypot(nx,ny);
   if(length<1e-10)continue;nx/=length;ny/=length;double p[3],plus[3],minus[3];gs_sample_point(a,x,y,p);
   for(int h=0;h<3;h++){double d=1e-6*(1.+R)*(nx*a[3+h]+ny*a[6+h]);plus[h]=p[h]+d;minus[h]=p[h]-d;}
   if(cut_inside(plus,rec,spans,planes,parts)||!cut_inside(minus,rec,spans,planes,parts))continue;
   double d=-nx*x-ny*y;bool duplicate=false;
   for(int h=0;h<found;h++)if(fabs(edges[24*id+3*h]-nx)+fabs(edges[24*id+3*h+1]-ny)+fabs(edges[24*id+3*h+2]-d)<1e-7)duplicate=true;
   if(duplicate)continue;edges[24*id+3*found]=nx;edges[24*id+3*found+1]=ny;edges[24*id+3*found+2]=d;found++;
  }
 }
 active[id]=found?2:0; // 2 = proposal only; union-of-cutters truth is separate.
}
extern "C" __global__ void shared_truth(const double* g,const double* radii,const long long* rows,
 const long long* starts,const long long* counts,const double* q,const unsigned char* enabled,const long long* peers,
 const double* samples,int ns,int nr,const unsigned char* affected,
 const double* rec,const long long* spans,const double* planes,int parts,
 const long long* patch,const long long* meta,const double* floor_rec,const long long* floor_spans,const double* floor_planes,
 unsigned char* truth,double* weights){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr*ns)return;int owner=i/ns,j=i%ns;
 long long row=rows[owner];const double* a=g+25*row;double R=radii[row],x=R*samples[2*j],y=R*samples[2*j+1];
#ifdef SHARED_RESIDUAL_CLIP_UNION
 bool yes=gs_sample_admitted(a+13,x,y)||gs_sample_admitted(a+19,x,y);
#else
 bool yes=gs_sample_admitted(a+13,x,y)&&gs_sample_admitted(a+19,x,y);
#endif
 // Group kinds: 1 = peer equation, 2 = sweep edge proposal only,
 // 3 = owner's finite planar domain, 4 = complement of a finite peer cut.
 bool domain_checked=false;long long last_solid=-1;
 for(long long k=starts[owner];yes&&k<starts[owner]+counts[owner];k++){
  if(enabled[k]==1&&polynomial(q+6*k,x,y)<-1e-10)yes=false;
  if(enabled[k]==3&&!domain_checked){double p[3];gs_sample_point(a,x,y,p);
   yes=floor_domain_inside(meta+4*patch[row],p,floor_rec,floor_spans,floor_planes);domain_checked=true;}
  if(enabled[k]==4){const long long* m=meta+4*peers[k];
   if(m[0]!=last_solid){double p[3];gs_sample_point(a,x,y,p);
    yes=!finite_cut_inside(p,m,floor_rec,floor_spans,floor_planes,meta[4*patch[row]+3]);last_solid=m[0];}}
 }
 if(yes&&affected[row]){double p[3];gs_sample_point(a,x,y,p);yes=!cut_inside(p,rec,spans,planes,parts);}
 truth[i]=yes;if(!owner)weights[j]=gs_sample_weight(x,y,R/2.);
}
'''


def prepare(rt,region,snapshot,nold,sweep,current_surface_ids,*,stock_bounds=None,sample_first=False):
    """Merge finite Sweep proposals, then evaluate shared residual grids on CUDA."""
    cp=rt.cp
    if not hasattr(rt,'retained_region_kernels'):
        module=cp.RawModule(code=CUDA,options=('--fmad=false',))
        rt.retained_region_kernels={k:module.get_function(k) for k in ('sweep_edges','shared_truth','contact_filter')}
    selected=update_candidates(snapshot.endpoints['positions'][:nold],snapshot.gaussian.u[:nold],
        snapshot.gaussian.v[:nold],snapshot.gaussian.attributes['scale'][:nold,0],sweep)
    # An identity is continuity metadata, not an exemption from material
    # removal. Exact same-ID peer pairs still do not invent a seam edge, but
    # old cap support inside the current finite sweep must be evaluated.
    same_selected=int(np.count_nonzero(selected & np.isin(
        snapshot.material_surface_ids[:nold],np.atleast_1d(current_surface_ids))))
    affected=cp.zeros(len(region.geometry),cp.uint8);affected[:nold]=cp.asarray(selected)
    packed=pack_cutters((sweep,));rec,spans,planes=[cp.asarray(x) for x in packed[:3]]
    if stock_bounds is not None:
        source_ids,source_meta,_,source_spans,source_planes=region.floor_domain_data
        region.retirement_domain=(source_ids,source_meta,source_spans,source_planes,
            cp.asarray(np.asarray(stock_bounds,dtype=float).reshape(6)),rec,spans,planes,np.int32(len(rec)))
    if nold:rt.retained_region_kernels['contact_filter'](((nold+127)//128,),(128,),
        (region.floor_domain_data[0],region.analytic_equations,region.floor_domain_data[1],np.int32(nold),rec,spans,planes,np.int32(len(rec)),affected))
    rows=cp.flatnonzero(affected).astype(cp.int64)
    contacts=int(np.count_nonzero(selected))-len(rows)
    equations=[]
    for r,span in zip(packed[0],packed[1]):
        for h in packed[2][span[0]:span[1]]:
            e=np.zeros(16);e[12:15]=-h[:3];e[15]=-h[3];equations.append(e)
        if r[13]:
            e=np.zeros(16);e[:3]=r[:3];e[3:12]=r[3:12];e[15]=-r[12]**2;equations.append(e)
    ne=len(equations);ng=len(rows)*ne
    if ng>2_000_000:raise RuntimeError('Retained region proposal budget exceeded')
    q=cp.zeros((ng,6));edges=cp.empty((ng,8,3));enabled=cp.zeros(ng,cp.uint8)
    edge_kernel=rt.retained_region_kernels['sweep_edges']
    if sample_first:
        from cutsim.rendering.sample_priority_clip import kernels
        edge_kernel=kernels(rt)['describe_sweep']
    if ng:edge_kernel(((ng+63)//64,),(64,),
        (region.geometry,rows,np.int32(len(rows)),cp.asarray(np.array(equations)),np.int32(ne),rec,spans,planes,np.int32(len(rec)),q,edges,enabled))
    active=cp.flatnonzero(enabled);owners=cp.repeat(region.rows,region.counts)
    # An empty proposal keeps the row in the grid even when no sampled
    # boundary is found. Its all-zero truth must never certify full deletion.
    owner=cp.r_[owners,cp.repeat(rows,ne)[active],rows]
    q=cp.concatenate((region.coefficients,q[active],cp.zeros((len(rows),6))))
    blank=cp.zeros((len(rows),8,3));blank[:,:,2]=cp.inf
    edges=cp.concatenate((region.edges,edges[active],blank));enabled=cp.r_[region.enabled,enabled[active],cp.zeros(len(rows),cp.uint8)]
    peers=cp.r_[region.peer_surfaces,cp.full(len(active)+len(rows),-1,cp.int64)]
    if sample_first:
        newrows=cp.repeat(rows,ne)[active]
        context=cp.concatenate((region.group_context,
            cp.column_stack((cp.full(len(active),-2,cp.int64),cp.zeros((len(active),2),cp.int64),region.floor_domain_data[0][newrows])),
            cp.column_stack((cp.full(len(rows),-1,cp.int64),cp.zeros((len(rows),2),cp.int64),region.floor_domain_data[0][rows]))))
    stock_groups=0
    if stock_bounds is not None:
        # A partially admitted new disk can straddle a stock edge even when
        # its nearest stock GS was deleted. The known box planes are exact
        # constraints; they must not depend on finding a surviving peer GS.
        g=region.geometry[nold:];new=cp.arange(nold,len(region.geometry),dtype=cp.int64)
        for axis in range(3):
            for sign,bound in ((1.,stock_bounds[0][axis]),(-1.,stock_bounds[1][axis])):
                nx=sign*g[:,3+axis];ny=sign*g[:,6+axis];d=sign*(g[:,axis]-bound)
                length=cp.hypot(nx,ny)
                selected=cp.flatnonzero((length>1e-12)&(d<region.radii[new]*length))
                if not len(selected):continue
                line=cp.column_stack((nx[selected],ny[selected],d[selected]))/length[selected,None]
                sq=cp.zeros((len(selected),6));sq[:,3:]=line
                se=cp.zeros((len(selected),8,3));se[:,:,2]=cp.inf;se[:,0]=line
                if sample_first:context=cp.concatenate((context,cp.column_stack((cp.full(len(selected),-3,cp.int64),cp.zeros((len(selected),2),cp.int64),region.floor_domain_data[0][new[selected]]))))
                owner=cp.r_[owner,new[selected]];q=cp.concatenate((q,sq));edges=cp.concatenate((edges,se))
                enabled=cp.r_[enabled,cp.ones(len(selected),cp.uint8)]
                peers=cp.r_[peers,cp.full(len(selected),-1,cp.int64)];stock_groups+=len(selected)
    order=cp.lexsort(cp.stack((peers,owner)));owner=owner[order]
    region.rows,region.starts,region.counts=cp.unique(owner,return_index=True,return_counts=True)
    if len(region.counts) and int(region.counts.max().item())>(128 if sample_first else 64):raise RuntimeError('Retained region group cap exceeded')
    region.coefficients=q[order];region.edges=edges[order];region.enabled=enabled[order]
    region.peer_surfaces=peers[order]
    if sample_first:
        region.group_context=context[order];region.sweep_data=(affected,rec,spans,planes)
    nr=len(region.rows);region.preview=cp.empty((nr,15));region.error=cp.empty((nr,2));region.accepted=cp.empty(nr,cp.uint8)
    if sample_first:
        return dict(retained_region_candidates=len(rows),retained_region_contact_protected=contacts,
            retained_region_same_id_evaluated=same_selected,retained_region_sweep_groups=len(active),
            retained_region_rows=nr,stock_boundary_groups=stock_groups)
    region.validation_samples=cp.asarray(np.vstack((disc(24,32),np.column_stack((np.cos(np.arange(128)*np.pi/64),np.sin(np.arange(128)*np.pi/64)))*.999)))
    region.sample_xy=cp.asarray(disc(16,32))
    def evaluate(samples):
        ns=len(samples);truth=cp.empty((nr,ns),cp.uint8);weights=cp.empty(ns)
        if nr:rt.retained_region_kernels['shared_truth'](((nr*ns+127)//128,),(128,),
            (region.geometry,region.radii,region.rows,region.starts,region.counts,region.coefficients,region.enabled,region.peer_surfaces,
             samples,np.int32(ns),np.int32(nr),affected,rec,spans,planes,np.int32(len(rec)),*region.floor_domain_data,truth,weights))
        return truth,weights
    region.sample_truth,region.sample_weights=evaluate(region.sample_xy)
    region.validation_truth,region.validation_weights=evaluate(region.validation_samples)
    return dict(retained_region_candidates=len(rows),retained_region_contact_protected=contacts,retained_region_same_id_evaluated=same_selected,retained_region_sweep_groups=len(active),
        retained_region_samples=nr*(512+896),retained_region_rows=nr,stock_boundary_groups=stock_groups)
