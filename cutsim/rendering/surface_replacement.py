"""Certify old-GS replacement using published geometry, without ray lineage.

Candidate old GS never donate to one another. New GS and permanently retained
old GS can cover the uncut finite footprint. Failure conservatively keeps it.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.finite_domain_retirement import CUDA as DOMAIN_CUDA

CUDA = DOMAIN_CUDA + r'''
__device__ double dp(const double* a,const double* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ bool same_domains(const double* a,const double* g,const long long* donors,
 long long start,long long count){
 for(int k=0;k<7;k++)if(a[13+k]!=a[20+k])return false;
 for(long long j=start;j<start+count;j++){
  const double* b=g+27*donors[j];
  for(int k=0;k<7;k++)if(b[13+k]!=b[20+k])return false;
 }
 return true;
}
// Packed: centre,u,v,normal,sigma, draw(7), material(7).
__device__ bool transfer(const double* a,const double* b,double x,double y,double reach,
 double* uv,double* axes){
 double cosine=dp(a+9,b+9);if(cosine<0.9961946980917455)return false; // 5 degrees
 double d[3];for(int k=0;k<3;k++)d[k]=a[k]+x*a[3+k]+y*a[6+k]-b[k];
 double h=dp(d,b+9),hx=dp(a+3,b+9),hy=dp(a+6,b+9);
 // Bound displacement of the ENTIRE source tile, not just its centre.
 double tolerance=fmin(.005,.04*fmin(a[12],b[12]));
 if((fabs(h)+reach*(fabs(hx)+fabs(hy)))/cosine>tolerance)return false;
 double nu=dp(a+9,b+3)/cosine,nv=dp(a+9,b+6)/cosine;
 uv[0]=dp(d,b+3)-h*nu;uv[1]=dp(d,b+6)-h*nv;
 axes[0]=dp(a+3,b+3)-hx*nu;axes[1]=dp(a+6,b+3)-hy*nu;
 axes[2]=dp(a+3,b+6)-hx*nv;axes[3]=dp(a+6,b+6)-hy*nv;
 return true;
}
__device__ bool identical_cover(const double* a,const double* b,int side,double factor){
 const double* c=a+13+7*side;const double* e=b+13+7*side;
 double r=fmin(factor*a[12],c[6]),s=fmin(factor*b[12],e[6]),uv[2],axes[4];
 if(!(r>0.)||!transfer(a,b,0.,0.,r,uv,axes))return false;
 double x=uv[0],y=uv[1];
 if(hypot(x,y)+r/dp(a+9,b+9)>s+1e-10)return false;
 for(int k=0;k<6;k+=3){
  if(isinf(e[k+2])&&e[k+2]>0.)continue;
  double nx=e[k]*axes[0]+e[k+1]*axes[2];
  double ny=e[k]*axes[1]+e[k+1]*axes[3];
  double off=e[k]*x+e[k+1]*y+e[k+2];
  if(off-r*hypot(nx,ny)>=0.)continue;
  bool implied=false;
  for(int j=0;j<6;j+=3)
   if(fabs(nx-c[j])+fabs(ny-c[j+1])<1e-12 && off>=c[j+2])implied=true;
  if(!implied)return false;
 }
 return true;
}
extern "C" __global__ void replacement_cover(const double* g,const long long* rows,
 const long long* donors,const long long* starts,const long long* counts,int nr,
 const bool* eligible,double factor,unsigned char* result){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr)return;
 result[i]=0;if(!eligible[i])return;
 const double* a=g+27*rows[i];if(!counts[i])return;
 for(int side=0;side<2;side++){
  if(side&&same_domains(a,g,donors,starts[i],counts[i]))continue;
  const double* c=a+13+side*7;double R=fmin(factor*a[12],c[6]);
  if(!(R>0.)||!isfinite(R))return;
  bool direct=false;
  for(long long j=starts[i];j<starts[i]+counts[i];j++)
   if(identical_cover(a,g+27*donors[j],side,factor)){direct=true;break;}
  if(direct)continue;
  // Uniform tiles enclose the entire old disk. A tile that MAY intersect
  // the clipped target must be wholly covered by one permanent donor.
  // Ambiguous boundary tiles conservatively retain the parent.
  const int N=32;double h=R/N;bool nonempty=false;
  for(int iy=0;iy<N;iy++)for(int ix=0;ix<N;ix++){
   double x=-R+(2*ix+1)*h,y=-R+(2*iy+1)*h;
   double xx=fmax(0.,fabs(x)-h),yy=fmax(0.,fabs(y)-h);
   if(xx*xx+yy*yy>R*R)continue;
   bool outside=false;
   for(int k=0;k<6;k+=3)
    if(c[k]*x+c[k+1]*y+c[k+2]+h*(fabs(c[k])+fabs(c[k+1]))<0.)outside=true;
   if(outside)continue;nonempty=true;
   bool covered=false;
   for(long long j=starts[i];j<starts[i]+counts[i];j++){
    const double* b=g+27*donors[j];double uv[2],axes[4];
    if(!transfer(a,b,x,y,h,uv,axes))continue;
    const double* e=b+13+side*7;double r=fmin(factor*b[12],e[6]);if(!(r>0.))continue;
    double u=uv[0],v=uv[1],ux=axes[0],uy=axes[1],vx=axes[2],vy=axes[3];
    bool yes=true;
    for(int corner=0;corner<4;corner++){
     double sx=(corner&1?1.:-1.)*h,sy=(corner&2?1.:-1.)*h;
     double U=u+sx*ux+sy*uy,V=v+sx*vx+sy*vy;
     if(U*U+V*V>r*r)yes=false;
    }
    for(int k=0;k<6;k+=3){
     double reach=h*(fabs(e[k]*ux+e[k+1]*vx)+fabs(e[k]*uy+e[k+1]*vy));
     if(e[k]*u+e[k+1]*v+e[k+2]-reach<0.)yes=false;
    }
    if(yes){covered=true;break;}
   }
   if(!covered)return;
  }
  if(!nonempty)return; // An empty sample set never authorizes retirement.
 }
 result[i]=1;
}
extern "C" __global__ void stable_donors(const double* targets,const long long* rows,int count,
 const double* boxes,const long long* order,const long long* sources,long long active,long long leaves,
 const long long* mapping,const unsigned int* candidates,const long long* offsets,
 long long* counts,long long* pairs,int write){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* p=targets+4*i;int stack[64],top=0;stack[top++]=1;long long found=0;
 while(top){
  int node=stack[--top];const double* b=boxes+6*node;bool outside=false;
  for(int k=0;k<3;k++)if(p[k]+p[3]<b[k]||p[k]-p[3]>b[k+3])outside=true;
  if(outside)continue;
  if(node<leaves){stack[top++]=2*node+1;stack[top++]=2*node;continue;}
  long long leaf=node-leaves;if(leaf>=active)continue;
  long long source=sources[order[leaf]],donor=mapping[source];
  // All retirement candidates are excluded, even if another check keeps one.
  // Thus old sources cannot mutually certify their own retirement.
  if(donor<0||candidates[source])continue;
  if(write){long long at=offsets[i]+found;pairs[2*at]=rows[i];pairs[2*at+1]=donor;}
  found++;
 }
 if(!write)counts[i]=found;
}
__device__ bool polygon_covered(const double* g,const double* a,const double* p,int n,
 int side,double factor,const long long* donors,long long start,long long count){
 if(n<3)return false;
 double low[2]={1e300,1e300},high[2]={-1e300,-1e300};
 for(int j=0;j<n;j++)for(int k=0;k<2;k++){low[k]=fmin(low[k],p[2*j+k]);high[k]=fmax(high[k],p[2*j+k]);}
 double x=.5*(low[0]+high[0]),y=.5*(low[1]+high[1]);
 double reach=.5*fmax(high[0]-low[0],high[1]-low[1]);
 for(long long j=start;j<start+count;j++){
  const double* b=g+27*donors[j];double uv[2],axes[4];
  if(!transfer(a,b,x,y,reach,uv,axes))continue;
  const double* c=b+13+side*7;double R=fmin(factor*b[12],c[6]);if(!(R>0.))continue;
  bool yes=true;
  for(int k=0;k<n&&yes;k++){
   double dx=p[2*k]-x,dy=p[2*k+1]-y;
   double u=uv[0]+axes[0]*dx+axes[1]*dy,v=uv[1]+axes[2]*dx+axes[3]*dy;
   if(u*u+v*v>R*R)yes=false;
   for(int h=0;h<6;h+=3)if(c[h]*u+c[h+1]*v+c[h+2]<0.)yes=false;
  }
  if(yes)return true;
 }
 return false;
}
extern "C" __global__ void finite_replacement_cover(const double* g,const long long* rows,
 const long long* donors,const long long* starts,const long long* counts,int nr,
 const bool* eligible,double factor,unsigned char* result,const long long* patches,const long long* meta,
 const long long* source_spans,const double* source_planes,const double* stock,
 const double* cut_records,const long long* cut_spans,const double* cut_planes,int cut_parts){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr)return;
 result[i]=0;if(!eligible[i])return;
 long long row=rows[i];const double* a=g+27*row;
 const long long* m=meta+4*patches[row];
 for(int side=0;side<2;side++){
  if(side&&same_domains(a,g,donors,starts[i],counts[i]))continue;
  double R=fmin(factor*a[12],a[19+7*side]);if(!(R>0.)||!isfinite(R))return;
  bool present=false;
  // Every part in the source union must be certified; choosing only the
  // generating primitive could lose a continuation across an internal seam.
  for(long long part=m[0];part<m[0]+m[1];part++){
   double polygon[2*DOMAIN_VERTICES];const long long* span=source_spans+2*part;
   int n=finite_polygon(a,side,R,-1,span,source_planes,stock,polygon);
   if(n<0)return;if(!n)continue;present=true;
   if(polygon_cut(a,polygon,n,cut_records,cut_spans,cut_planes,cut_parts))continue;
   if(!counts[i])return;
   // Only the part outside the current cut needs a permanent replacement.
   // No sampled empty region can authorize deletion.
   for(int tile=0;tile<32*32;tile++){
    n=finite_polygon(a,side,R,tile,span,source_planes,stock,polygon);
    if(n<0)return;if(!n)continue;
    if(polygon_cut(a,polygon,n,cut_records,cut_spans,cut_planes,cut_parts))continue;
    if(!polygon_covered(g,a,polygon,n,side,factor,donors,starts[i],counts[i]))return;
   }
  }
  if(!present)return;
 }
 result[i]=1;
}

extern "C" __global__ void centre_inside(const double* positions,int count,const double* records,
 const long long* spans,const double* planes,int parts,unsigned int* inside){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;inside[i]=0;const double* p=positions+3*i;
 for(int part=0;part<parts;part++){
  bool yes=true;const double* r=records+14*part;
  for(long long j=spans[2*part];j<spans[2*part+1];j++){
   const double* h=planes+5*j;double value=h[0]*p[0]+h[1]*p[1]+h[2]*p[2]+h[3];
   if(value<(h[4]?1e-8:-1e-10)){yes=false;break;}
  }
  if(yes&&r[13]){double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
   for(int a=0;a<3;a++)for(int b=0;b<3;b++)q+=d[a]*r[3+3*a+b]*d[b];
   yes=q<r[12]*r[12]-1e-8;
  }
  if(yes){inside[i]=1;return;}
 }
}

extern "C" __global__ void new_donors(const double* targets,const long long* rows,int count,
 const double* donors,const long long* donor_rows,int nd,const long long* offsets,
 long long* counts,long long* pairs,int write){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* p=targets+7*i;long long found=0;
 for(int j=0;j<nd;j++){
  const double* d=donors+7*j;double r=p[6]+d[6]+.005;
  if(fabs(p[0]-d[0])>r||fabs(p[1]-d[1])>r||fabs(p[2]-d[2])>r)continue;
  if(p[3]*d[3]+p[4]*d[4]+p[5]*d[5]<.9961946980917455)continue;
  if(write){long long at=offsets[i]+found;pairs[2*at]=rows[i];pairs[2*at+1]=donor_rows[j];}
  found++;
 }
 if(!write)counts[i]=found;
}

'''


def _unique_pairs(cp,pairs):
    """Lexicographic dedup without CuPy unique(axis=0)'s per-row syncs."""
    if len(pairs)<2:return pairs
    order=cp.lexsort(cp.stack((pairs[:,1],pairs[:,0])))
    pairs=pairs[order];first=cp.ones(len(pairs),cp.bool_)
    first[1:]=cp.any(pairs[1:]!=pairs[:-1],axis=1)
    return pairs[first]


def covered_rows(rt,snapshot,rows,pairs,*,sweep=None,stock_bounds=None):
    """Return targets covered in BOTH draw and material certificates.

    Pairs contain final snapshot row indices. Projection along the old normal
    must move the entire covered tile by <= min(.005 mm, .04*min(sigma)), with
    normals within 5 degrees. These bounds also allow small curved replacements
    while rejecting distant sheets. This is a bounded geometric approximation.
    With a sweep and analytic metadata, exclude proven cut polygons and parts
    outside the finite source domain before checking replacement coverage.
"""
    cp=rt.cp;rows=np.asarray(rows,np.int64)
    if not len(rows):return np.empty(0,bool)
    with rt.context():
        pairs=cp.asarray(pairs,dtype=cp.int64).reshape(-1,2)
        finite=sweep is not None and stock_bounds is not None and hasattr(snapshot,'analytic_patch_ids')
        if not len(pairs) and not finite:return np.zeros(len(rows),bool)
        pairs=_unique_pairs(cp,pairs)
        starts=cp.searchsorted(pairs[:,0],cp.asarray(rows),side='left').astype(cp.int64)
        counts=cp.searchsorted(pairs[:,0],cp.asarray(rows),side='right').astype(cp.int64)-starts
        g=snapshot.gaussian
        geometry=cp.asarray(np.ascontiguousarray(np.column_stack((snapshot.endpoints['positions'],g.u,g.v,
            snapshot.endpoints['normals'],g.attributes['scale'][:,0],snapshot.clipping.planes,
            snapshot.material_clipping.planes)),dtype=float))
        if not hasattr(rt,'hit_replacement_cover_kernel'):
            rt.hit_replacement_cover_kernel=cp.RawKernel(CUDA,'replacement_cover',options=('--fmad=false',))
        kernel=rt.hit_replacement_cover_kernel;domain=()
        if finite:
            from cutsim.rendering.finite_domain_retirement import device_domain
            if not hasattr(rt,'finite_replacement_cover_kernel'):
                rt.finite_replacement_cover_kernel=cp.RawKernel(CUDA,'finite_replacement_cover',options=('--fmad=false',))
            kernel=rt.finite_replacement_cover_kernel
            domain=device_domain(rt,snapshot,sweep,stock_bounds,rows=rows)
        result=cp.empty(len(rows),cp.uint8);good=cp.ones(len(rows),cp.bool_)
        from cutsim.rendering.gs_extent import support_factor
        # Publication uses 2 sigma, material rays may use 1.5 sigma. A larger
        # draw disk must never certify coverage missing from material queries.
        for factor in sorted({2.,support_factor(snapshot)}):
            # The certificate is a conjunction. A failed smaller support can
            # never be rescued by the next factor; skip its expensive polygons
            # on device, without another host readback/compaction.
            kernel(((len(rows)+127)//128,),(128,),
                (geometry,cp.asarray(rows),cp.ascontiguousarray(pairs[:,1]),starts,counts,
                 np.int32(len(rows)),good,np.float64(factor),result,*domain))
            good &= result!=0
        return cp.asnumpy(good,stream=rt.stream)


def replacement_candidates(snapshot, sweep, rt, *, selection=None):
    """Old centres inside the finite sweep are candidates, never deletion proof."""
    from cutsim.rendering.profile_cut_clip import pack_cutters
    cp = rt.cp
    with rt.context():
        if not hasattr(rt, 'replacement_centres_kernel'):
            rt.replacement_centres_kernel = cp.RawKernel(CUDA, 'centre_inside', options=('--fmad=false',))
        packed = selection.packed if selection is not None else pack_cutters((sweep,))
        p=snapshot.endpoints['positions']
        rows=np.arange(len(p)) if selection is None else selection.rows
        positions = cp.asarray(p[rows])
        local = cp.zeros(len(positions), cp.uint32)
        candidates = cp.zeros(len(p), cp.uint32)
        if len(positions):
            rt.replacement_centres_kernel(((len(positions)+127)//128,), (128,),
                (positions, np.int32(len(positions)), *[cp.asarray(a) for a in packed[:3]],
                 np.int32(len(packed[0])), local))
            candidates[cp.asarray(rows)]=local
        return candidates


def commit_replacements(surface, snapshot, added, removed, candidates, *, sweep=None, stock_bounds=None):
    """Replace only where final new geometry exists; no generating-ray IDs."""
    started = perf_counter(); rt = surface.rt; cp = rt.cp
    old_sources = np.flatnonzero(~removed); old_rows = np.flatnonzero(~added)
    if len(old_sources) != len(old_rows):
        raise ValueError('Old publication mapping mismatch')
    with rt.context():
        target = cp.asnumpy(candidates[cp.asarray(old_sources)] != 0, stream=rt.stream)
        rows = old_rows[target]; original_count = len(rows)
        pairs = cp.empty((0, 2), cp.int64)
        stable_count = new_count = skipped = 0
        donors = np.flatnonzero(added)
        if len(rows) and len(donors):
            if not hasattr(rt, 'surface_donor_kernels'):
                rt.surface_donor_kernels = {
                    name: cp.RawKernel(CUDA, name, options=('--fmad=false',))
                    for name in ('stable_donors', 'new_donors')}
            positions = snapshot.endpoints['positions']; normals = snapshot.endpoints['normals']
            radii = 2*snapshot.gaussian.attributes['scale'][:, 0]
            targets = cp.asarray(np.column_stack((positions[rows], normals[rows], radii[rows])))
            geometry = cp.asarray(np.column_stack((positions[donors], normals[donors], radii[donors])))
            counts = cp.empty(len(rows), cp.int64); dummy = cp.empty(1, cp.int64)
            args = (targets, cp.asarray(rows), np.int32(len(rows)), geometry,
                    cp.asarray(donors), np.int32(len(donors)))
            kernel = rt.surface_donor_kernels['new_donors']
            kernel(((len(rows)+127)//128,), (128,), (*args, dummy, counts, dummy, np.int32(0)))
            # Full-cut retirement already ran in stage 3. Without compatible
            # new geometry there is no replacement to verify; keep the source.
            eligible = cp.asnumpy(counts > 0, stream=rt.stream)
            rows = rows[eligible]; targets = targets[cp.asarray(eligible)]; counts = counts[cp.asarray(eligible)]
            if len(rows):
                launch = (((len(rows)+127)//128,), (128,))
                args = (targets, cp.asarray(rows), np.int32(len(rows)), geometry,
                        cp.asarray(donors), np.int32(len(donors)))
                offsets = cp.cumsum(counts); total = int(offsets[-1].item()); offsets -= counts
                if total <= 2_000_000:
                    pairs = cp.empty((total, 2), cp.int64)
                    kernel(*launch, (*args, offsets, counts, pairs, np.int32(1)))
                    new_count = total
                else:
                    skipped += total
                mapping = cp.full(len(removed), -1, cp.int64)
                mapping[cp.asarray(old_sources)] = cp.asarray(old_rows)
                targets = cp.asarray(np.column_stack((positions[rows], radii[rows]+.005)))
                args = (targets, cp.asarray(rows), np.int32(len(rows)), surface.boxes, surface.order,
                        surface.device_rows, np.int64(len(surface.rows)), np.int64(surface.leaves),
                        mapping, candidates)
                kernel = rt.surface_donor_kernels['stable_donors']
                kernel(*launch, (*args, dummy, counts, dummy, np.int32(0)))
                offsets = cp.cumsum(counts); total = int(offsets[-1].item()); offsets -= counts
                if total+len(pairs) <= 2_000_000:
                    extra = cp.empty((total, 2), cp.int64)
                    kernel(*launch, (*args, offsets, counts, extra, np.int32(1)))
                    pairs = cp.concatenate((pairs, extra)); stable_count = total
                else:
                    skipped += total
        else:
            rows = rows[:0]
    gone = rows[covered_rows(rt, snapshot, rows, pairs, sweep=sweep, stock_bounds=stock_bounds)]
    stats = dict(replacement_method='final_gs_geometry', replacement_checked=len(rows),
                 replacement_without_new_support=original_count-len(rows),
                 replacement_removed=len(gone), replacement_kept=len(rows)-len(gone),
                 replacement_stable_pairs=stable_count, replacement_new_pairs=new_count,
                 replacement_pair_budget_skipped=skipped, replacement_ms=(perf_counter()-started)*1000)
    return gone, stats
