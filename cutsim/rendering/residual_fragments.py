"""Bounded subdivision of deleted GS supports, using current geometry only.

Children inherit the exact parent disc/frame and split its clipped support
into exterior halfplane fragments. When two lines cannot encode a fragment,
its centre is moved to a residual sample and its disc is contained in the parent.
No enlarged disc or cutter history is used.
One fragment suffices for a straight boundary; extra fragments are selected
greedily only when residual samples are not covered. All selection is CUDA.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.gs_area_retirement import KERNEL as AREA_KERNEL, _disc
from cutsim.rendering.profile_cut_clip import _KERNEL as CLIP_KERNEL, pack_cutters, ProfileClip

KERNEL = r'''
__device__ bool fragment_visible(const double* c,double x,double y){
 return x*x+y*y<=c[6]*c[6] && c[6]>0. &&
        c[0]*x+c[1]*y+c[2]>=0. && c[3]*x+c[4]*y+c[5]>=0.;
}
__device__ void fragment_plane(const double* n,const double* s,const double* g,
 const double* p,double* best,double* plane){
 double neg[3]={-n[0],-n[1],-n[2]},maximum=-support_clip(s,neg);
 double x=dot_clip(n,g+3),y=dot_clip(n,g+6),length=hypot(x,y);
 if(length<1e-10)return;
 double gap=(dot_clip(n,p)-maximum)/length;
 if(gap>*best){*best=gap;plane[0]=x/length;plane[1]=y/length;
  plane[2]=(dot_clip(n,g)-maximum)/length-2e-7;}
}
extern "C" __global__ void fragment_candidates(const double* geom,const double* clips,
 const double* samples,int parents,int ns,const double* records,const long long* spans,
 const double* planes,int parts,const double* support,const double* original,int preserve,double* xy,unsigned char* valid,double* candidates){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=parents*ns)return;
 int parent=i/ns,j=i%ns;const double* g=geom+10*parent;const double* c=clips+7*parent;
 double x=g[9]*samples[2*j],y=g[9]*samples[2*j+1];xy[2*i]=x;xy[2*i+1]=y;valid[i]=0;
 const double* prior=original+14*parent;
 if(preserve){if(!fragment_visible(prior,x,y)||!fragment_visible(prior+7,x,y))return;}
 else if(!fragment_visible(c,x,y))return;
 double p[3];for(int k=0;k<3;k++)p[k]=g[k]+x*g[3+k]+y*g[6+k];
 if(cut_inside(p,records,spans,planes,parts))return;
 double best=0.,plane[3]={0,0,INFINITY};
 for(int part=0;part<parts;part++){
  for(long long h=spans[2*part];h<spans[2*part+1];h++){
   double n[3]={-planes[5*h],-planes[5*h+1],-planes[5*h+2]};
   fragment_plane(n,support,g,p,&best,plane);
  }
  const double* r=records+14*part;
  if(r[13]){double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},n[3];
   for(int k=0;k<3;k++)n[k]=dot_clip(r+3+3*k,d);
   fragment_plane(n,support,g,p,&best,plane);
  }
 }
 valid[i]=1;double* out=candidates+10*i;for(int k=0;k<7;k++)out[k]=c[k];
 out[7]=out[8]=0.;out[9]=g[9];
 if(best>0.)insert_clip(out,plane[0],plane[1],plane[2]);
 else out[6]=0.; // no exterior certificate: never invent a fragment
 if(best>0. && !fragment_visible(out,x,y)){
  // Only exceptional three-boundary corners need a smaller, recentered disc.
  double radius=fmin(.5*g[9],c[6]-hypot(x,y));
  for(int k=0;k<7;k++)out[k]=c[k];
  for(int k=0;k<6;k+=3)out[k+2]+=out[k]*x+out[k+1]*y;
  if(preserve){
   // A compressed two-line parent may already have lost a third/fourth
   // edge. Recenter inside the ORIGINAL support, not that compressed disc.
   radius=fmin(.5*g[9],fmin(g[9],fmin(prior[6],prior[13]))-hypot(x,y));
   for(int j=0;j<4;j++){const double* edge=prior+7*(j/2)+3*(j%2);double n=hypot(edge[0],edge[1]);
    if(n>1e-14)radius=fmin(radius,(edge[0]*x+edge[1]*y+edge[2])/n);}
   out[0]=out[1]=out[3]=out[4]=0.;out[2]=out[5]=INFINITY;
  }
  out[6]=radius;out[7]=x;out[8]=y;out[9]=radius;
  insert_clip(out,plane[0],plane[1],plane[2]+plane[0]*x+plane[1]*y);
 }
}
extern "C" __global__ void fragment_select(const double* xy,const unsigned char* valid,
 const double* candidates,int ns,int limit,double* output,int* counts,int* left){
 int parent=blockIdx.x,lane=threadIdx.x;__shared__ unsigned char uncovered[1024];
 __shared__ int scores[128],indices[128],chosen,remaining;
 for(int j=lane;j<ns;j+=128)uncovered[j]=valid[parent*ns+j];
 if(!lane){counts[parent]=0;remaining=0;}__syncthreads();
 for(int step=0;step<limit;step++){
  int score=0,index=-1;
  for(int j=lane;j<ns;j+=128){
   if(!valid[parent*ns+j])continue;const double* c=candidates+10*(parent*ns+j);int value=0;
   if(c[6]>0.)for(int k=0;k<ns;k++)if(uncovered[k]&&fragment_visible(c,xy[2*(parent*ns+k)]-c[7],xy[2*(parent*ns+k)+1]-c[8]))value++;
   if(value>score){score=value;index=j;}
  }
  scores[lane]=score;indices[lane]=index;__syncthreads();
  for(int stride=64;stride;stride/=2){
   if(lane<stride && (scores[lane+stride]>scores[lane] ||
      (scores[lane+stride]==scores[lane]&&indices[lane+stride]>=0&&indices[lane+stride]<indices[lane]))){
    scores[lane]=scores[lane+stride];indices[lane]=indices[lane+stride];}
   __syncthreads();
  }
  if(!lane)chosen=scores[0]?indices[0]:-1;__syncthreads();if(chosen<0)break;
  const double* c=candidates+10*(parent*ns+chosen);
  if(!lane){for(int k=0;k<10;k++)output[10*(parent*limit+step)+k]=c[k];counts[parent]++;}
  for(int k=lane;k<ns;k+=128)if(fragment_visible(c,xy[2*(parent*ns+k)]-c[7],xy[2*(parent*ns+k)+1]-c[8]))uncovered[k]=0;
  __syncthreads();
 }
 int n=0;for(int k=lane;k<ns;k+=128)n+=uncovered[k];if(n)atomicAdd(&remaining,n);__syncthreads();
 if(!lane)left[parent]=remaining;
}
extern "C" __global__ void fragment_donors(const double* points,const double* normals,long long count,
 const double* boxes,const long long* order,const long long* sources,long long active,long long leaves,
 const long long* owners,const double* geometry,const double* clips,const double* material,unsigned int* covered){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* p=points+3*i;covered[i]=0;if(!active)return;
 int stack[64],top=0;stack[top++]=1;
 while(top){
  int node=stack[--top];const double* box=boxes+6*node;bool outside=false;
  for(int k=0;k<3;k++)if(p[k]<box[k]-1e-7||p[k]>box[k+3]+1e-7)outside=true;
  if(outside)continue;
  if(node<leaves){stack[top++]=2*node+1;stack[top++]=2*node;continue;}
  long long leaf=node-leaves;if(leaf>=active)continue;
  long long row=owners[sources[order[leaf]]];if(row<0)continue;
  const double* g=geometry+13*row;
  if(dot_clip(g+9,normals+3*i)<1.-1e-8)continue;
  double d[3]={p[0]-g[0],p[1]-g[1],p[2]-g[2]};
  if(fabs(dot_clip(d,g+9))>1e-7)continue;
  double x=dot_clip(d,g+3),y=dot_clip(d,g+6),r=g[12];
  if(x*x+y*y>r*r)continue;
  if(fragment_visible(clips+7*row,x,y)&&fragment_visible(material+7*row,x,y)){covered[i]=1;return;}
 }
}
'''


def residual_fragments(before,sweep,removed,block,*,max_fragments=1024,per_parent=8,exempt_surface_ids=None,preserve_constraints=False,priority_mask=None,return_sources=False,no_radius_fallback=False):
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.cutting.tool_profile_simulator import _select
    from cutsim.rendering.profile_cut_clip import _insert
    started=perf_counter();rt=runtime();cp=rt.cp
    stats=dict(residual_parents=0,residual_fragments=0,residual_uncovered_samples=0,
               residual_budget_skipped=0,residual_ms=0.,residual_limit=max_fragments,residual_per_parent_limit=per_parent)
    eligible=np.asarray(removed,dtype=bool).copy()
    same=np.zeros(len(eligible),bool) if exempt_surface_ids is None else np.isin(before.material_surface_ids,np.atleast_1d(exempt_surface_ids))
    stats['residual_same_surface_skipped']=int(np.count_nonzero(eligible&same))
    eligible &= ~same
    ids=np.flatnonzero(eligible)
    if priority_mask is not None:
        priority=np.asarray(priority_mask,bool)[ids]
        ids=np.r_[ids[priority],ids[~priority]]
    if not len(ids) or max_fragments<=0:return None,stats
    # Both frozen certificates belong to the parent. Intersect before splitting.
    clips=before.clipping.planes[ids].copy();material=before.material_clipping.planes[ids]
    original=np.column_stack((clips,material))
    for slot in (0,3):_insert(clips,material[:,slot],material[:,slot+1],material[:,slot+2])
    radii=2.*before.gaussian.attributes['scale'][ids,0].astype(float)
    clips[:,6]=np.minimum(np.minimum(clips[:,6],material[:,6]),radii)
    geom=np.column_stack((before.endpoints['positions'][ids],before.gaussian.u[ids],before.gaussian.v[ids],radii))
    packed=pack_cutters((sweep,));sample_points=_disc(16,32)
    if preserve_constraints:
        angle=(np.arange(128)+.5)*2*np.pi/128
        sample_points=np.vstack((sample_points,np.column_stack((np.cos(angle),np.sin(angle)))))
    ns=len(sample_points)
    with rt.context():
        key='fragment_strict_kernels' if no_radius_fallback else 'fragment_kernels'
        if not hasattr(rt,key):
            code=KERNEL
            helpers=''
            if no_radius_fallback:
                from cutsim.rendering.clip_subdivision import CUDA
                helpers=CUDA.split('extern "C" __global__ void roots')[0]
                strict=r'''
 valid[i]=1;double* out=candidates+10*i;
 out[6]=0.;out[9]=0.;out[7]=out[8]=0.;
 if(best<=0.)return;
 double edges[15],cert[15];
 for(int j=0;j<4;j++)for(int k=0;k<3;k++)edges[3*j+k]=prior[7*(j/2)+3*(j%2)+k];
 for(int k=0;k<3;k++)edges[12+k]=plane[k];
 double radius=fmin(g[9],fmin(prior[6],prior[13]));
 bool fitted=fit(edges,edges,5,0.,0.,radius,nullptr,0,cert);
 if(!fitted){
  double cx=x,cy=y,len=hypot(x,y);
  if(len>=radius*(1.-1e-8)){cx=.75*x;cy=.75*y;}
  double limit=fmax(0.,radius-hypot(cx,cy)),r=limit;
  for(int it=0;it<20 && r>1e-7;it++,r*=.5){
   if(fit(edges,edges,5,cx,cy,r,nullptr,0,cert)){
    double lo=r,hi=fmin(limit,2*r);
    for(int b=0;b<8;b++){double mid=(lo+hi)*.5,tmp[15];
     if(fit(edges,edges,5,cx,cy,mid,nullptr,0,tmp)){lo=mid;for(int k=0;k<15;k++)cert[k]=tmp[k];}else hi=mid;}
    fitted=true;break;
   }
  }
 }
 if(fitted){for(int k=0;k<6;k++)out[k]=cert[3+k];out[6]=out[9]=cert[2];out[7]=cert[0];out[8]=cert[1];}
 return;
'''
                start=code.index(' valid[i]=1;double* out=candidates+10*i;')
                end=code.index('\n}\nextern "C" __global__ void fragment_select',start)
                code=code[:start]+strict+code[end:]
            module=cp.RawModule(code=AREA_KERNEL+CLIP_KERNEL+helpers+code,options=('--fmad=false',))
            setattr(rt,key,[module.get_function(n) for n in ('fragment_candidates','fragment_select','fragment_donors')])
        kernels=getattr(rt,key)
        parts=[cp.asarray(a) for a in packed[:3]];support=cp.asarray(packed[4][0])
        samples=cp.asarray(sample_points);collected=[];parents=[];left_total=0;eligible=0
        # Fixed-size batches bound temporary VRAM independently of scene size.
        for start in range(0,len(ids),64):
            n=min(64,len(ids)-start);xy=cp.empty((n,ns,2),cp.float64);valid=cp.empty((n,ns),cp.uint8)
            candidates=cp.empty((n,ns,10),cp.float64);output=cp.empty((n,per_parent,10),cp.float64)
            count=cp.empty(n,cp.int32);left=cp.empty(n,cp.int32)
            kernels[0](((n*ns+127)//128,),(128,),
                (cp.asarray(geom[start:start+n]),cp.asarray(clips[start:start+n]),samples,np.int32(n),np.int32(ns),
                 *parts,np.int32(len(packed[0])),support,cp.asarray(original[start:start+n]),np.int32(preserve_constraints),xy,valid,candidates))
            kernels[1]((n,),(128,),(xy,valid,candidates,np.int32(ns),np.int32(per_parent),output,count,left))
            selected=cp.nonzero(cp.arange(per_parent)[None,:]<count[:,None])
            pp=cp.asnumpy(selected[0],stream=rt.stream);rows=cp.asnumpy(output[selected],stream=rt.stream)
            eligible+=int(cp.asnumpy(cp.count_nonzero(count),stream=rt.stream));left_total+=int(cp.asnumpy(left.sum(),stream=rt.stream))
            parents.extend(ids[start+pp]);collected.extend(rows)
        stats.update(residual_parents=eligible,residual_uncovered_samples=left_total)
    if not parents:
        stats['residual_ms']=(perf_counter()-started)*1000;return None,stats
    parents=np.asarray(parents);certificates=np.asarray(collected)
    # Keep whole parent families when reaching the bound.
    if len(parents)>max_fragments:
        boundary=parents[max_fragments];take=np.flatnonzero(parents!=boundary);take=take[take<max_fragments]
        stats['residual_budget_skipped']=len(parents)-len(take)
        stats['_deferred_ids']=np.setdiff1d(parents,parents[take])
        parents=parents[take];certificates=certificates[take]
    fragment=_select(before,parents)
    offset=certificates[:,7:9]
    positions=fragment.endpoints['positions']+offset[:,0,None]*fragment.gaussian.u+offset[:,1,None]*fragment.gaussian.v
    fragment.endpoints['positions']=positions
    fragment.gaussian.attributes['positions']=positions.astype(np.float32)
    sigma=(certificates[:,9]/2.).astype(np.float32)
    if no_radius_fallback:
        sigma=np.where(2.*sigma.astype(float)>certificates[:,9],np.nextafter(sigma,np.float32(0)),sigma)
        certificates[:,6]=2.*sigma.astype(float)
    fragment.gaussian.attributes['scale']=sigma[:,None]
    certificates=certificates[:,:7].copy()
    fragment.clipping=ProfileClip(certificates,{})
    fragment.material_clipping=ProfileClip(certificates.copy(),{})
    fragment.endpoints['ray_ids']=-(block+1)*1_000_000-np.arange(1,len(parents)+1,dtype=np.int64)
    fragment.endpoints['owners']=np.full(len(parents),block,np.int64)
    _,families=np.unique(parents,return_counts=True)
    stats.update(residual_single_parents=int((families==1).sum()),residual_split_parents=int((families>1).sum()),
                 residual_split_children=int(families[families>1].sum()))
    stats.update(residual_fragments=len(parents),residual_ms=(perf_counter()-started)*1000)
    if return_sources:stats['_source_ids']=parents
    return fragment,stats


def redundant_fragments(snapshot,surface,kept,nkept,nfragments,*,candidate_rows=None,exclude_sources=None,boundary_samples=False,display_scale=1.,detail=None):
    """Prune explicit published children; retained donors may include other children.

    The final snapshot and ordered deletion determine donor availability,
    independently of the generating ID.
    """
    from cutsim.rendering.fragment_pruning import prune
    rows=np.arange(nkept,nkept+nfragments) if candidate_rows is None else np.asarray(candidate_rows,np.int64)
    if len(rows)!=nfragments:raise ValueError('Fragment candidate count mismatch')
    return prune(snapshot,surface.rt,rows,display_scale=display_scale,detail=detail)
