"""Bounded GPU child cover for a failed two-edge approximation.

Subdivision follows a failed two-edge fit. Failed attempts keep the published parent;
only a validated child union may replace it. No elapsed-time dependent results.
"""
from copy import copy
from time import perf_counter
import numpy as np

MAX_PARENTS = 32
MAX_CHILDREN = 256
CHILDREN_PER_PARENT = 8
TOLERANCE = .02
# Parent refinement is an independent policy. Child coverage/opacity checks
# keep TOLERANCE independently of this refinement trigger.
PARENT_SPLIT_TOLERANCE = .10


def parent_split_flags(cp, region, tolerance):
    """Reclassify saved fit errors without changing fitted edges or R truth."""
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError('Parent split tolerance must be finite and in [0,1)')
    nr = len(region.rows)
    face_failed = getattr(region, 'face_line_failed', cp.zeros(nr, cp.bool_))
    if hasattr(region, 'face_line_error'):
        owners = cp.repeat(cp.arange(nr, dtype=cp.int64), region.counts)
        bad = (region.face_line_exposed != 0) & (cp.max(region.face_line_error, axis=1) > tolerance)
        face_failed = cp.bincount(owners[bad], minlength=nr) > 0
    fit_failed = ~region.accepted.astype(bool)
    if hasattr(region, 'twoedge_validation_error'):
        fit_failed = cp.any(region.twoedge_validation_error > tolerance, axis=1)
    # Structural invalidity/empty output with surviving material never becomes
    # acceptable merely because a geometric error threshold is relaxed.
    invalid = getattr(region, 'twoedge_fallback', cp.zeros(nr, cp.uint8)).astype(bool)
    nonconvex = getattr(region, 'nonconvex_residual', cp.zeros(nr, cp.bool_))
    fit_failed |= invalid | nonconvex
    return face_failed, (face_failed | fit_failed) & region.sample_truth.any(axis=1)

NONCONVEX_CUDA = r'''
__device__ double hull_cross(const double* xy,int a,int b,int c){
 return (xy[2*b]-xy[2*a])*(xy[2*c+1]-xy[2*a+1])
       -(xy[2*b+1]-xy[2*a+1])*(xy[2*c]-xy[2*a]);
}
extern "C" __global__ void nonconvex_residual(const double* xy,
 const long long* order,const unsigned char* truth,int ns,int nr,unsigned char* result){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr)return;
 result[i]=0;const unsigned char* mask=truth+i*ns;
 int kept=0;for(int j=0;j<ns;j++)kept+=mask[j]!=0;
 if(kept<3||kept==ns)return;
 // Monotone-chain hull of surviving samples, in normalized GS coordinates.
 // Any removed point strictly inside this hull proves that no intersection
 // of halfplanes can reproduce the saved mask, regardless of area tolerance.
 int hull[256],n=0;
 for(int j=0;j<ns;j++){int p=order[j];if(!mask[p])continue;
  while(n>=2&&hull_cross(xy,hull[n-2],hull[n-1],p)<=0.)n--;
  hull[n++]=p;
 }
 int lower=n+1;
 for(int j=ns-2;j>=0;j--){int p=order[j];if(!mask[p])continue;
  // The largest retained point is already the end of the lower chain.
  if(p==hull[n-1])continue;
  while(n>=lower&&hull_cross(xy,hull[n-2],hull[n-1],p)<=0.)n--;
  hull[n++]=p;
 }
 if(n>1&&hull[n-1]==hull[0])n--;
 if(n<3)return;
 for(int j=0;j<ns;j++)if(!mask[j]){
  bool inside=true;
  for(int k=0;k<n;k++)if(hull_cross(xy,hull[k],hull[(k+1)%n],j)<=1e-10){inside=false;break;}
  if(inside){result[i]=1;return;}
 }
}
'''


def sampled_nonconvex_flags(rt, region):
    """GPU convexity witness using only the saved all-face residual samples."""
    cp=rt.cp;nr=len(region.rows);ns=len(region.sample_xy)
    if ns>128:
        raise ValueError('Nonconvex residual detection supports at most 128 shared samples')
    if not hasattr(rt,'nonconvex_residual_kernel'):
        rt.nonconvex_residual_kernel=cp.RawKernel(NONCONVEX_CUDA,'nonconvex_residual')
    xy=cp.ascontiguousarray(region.sample_xy,dtype=cp.float64)
    order=cp.lexsort(cp.stack((xy[:,1],xy[:,0]))).astype(cp.int64)
    truth=cp.ascontiguousarray(region.sample_truth,dtype=cp.uint8)
    result=cp.zeros(nr,cp.uint8)
    if nr:rt.nonconvex_residual_kernel(((nr+63)//64,),(64,),
        (xy,order,truth,np.int32(ns),np.int32(nr),result))
    return result.astype(cp.bool_)

COUNT_CUDA = r'''
extern "C" __global__ void count_boundaries(const long long* rows,
 const long long* starts,const long long* counts,const unsigned char* enabled,
 const double* proximity,const unsigned int* masks,const double* g,
 const double* radii,const double* xy,int ns,int nr,int* result){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr)return;
 // Stop at three distinct masks: only the low-boundary seed policy needs this count.
 unsigned int unique[3*16];int number=0,nrgs=ns/32;
 for(int item=-4;item<counts[i];item++){
  unsigned int bits[16];
  if(item<0){const double* h=g+25*rows[i]+13+3*(item+4);
   if(!isfinite(h[2]))continue;
   for(int ring=0;ring<nrgs;ring++){unsigned int value=0;
    for(int k=0;k<32;k++){int j=32*ring+k;
     if(h[0]*radii[rows[i]]*xy[2*j]+h[1]*radii[rows[i]]*xy[2*j+1]+h[2]>=-1e-10)value|=1u<<k;
    }bits[ring]=value;}
  }else{long long k=starts[i]+item;if(!enabled[k]||!isfinite(proximity[k]))continue;
   for(int ring=0;ring<nrgs;ring++)bits[ring]=masks[nrgs*k+ring];}
  bool some=false,all=true;for(int ring=0;ring<nrgs;ring++){some|=bits[ring]!=0;all&=bits[ring]==0xffffffff;}
  if(!some||all)continue;
  bool duplicate=false;for(int k=0;k<number;k++){bool same=true;
   for(int ring=0;ring<nrgs;ring++)same&=bits[ring]==unique[16*k+ring];if(same){duplicate=true;break;}}
  if(!duplicate){for(int ring=0;ring<nrgs;ring++)unique[16*number+ring]=bits[ring];number++;
   if(number==3){result[i]=number;return;}}
 }
 result[i]=number;
}
'''


def _native_certificates(cp, cert):
    """Validate the same FP32 scale/angle/split representation that is drawn."""
    cert=cert.copy()
    sigma=(cert[:,2]/2).astype(cp.float32)
    sigma=cp.where(2*sigma.astype(cp.float64)>cert[:,2],cp.nextafter(sigma,cp.float32(0)),sigma)
    cert[:,2]=2*sigma
    for slot in (3,6):
        h=cert[:,slot:slot+3];length=cp.linalg.norm(h[:,:2],axis=1)
        active=cp.isfinite(h[:,2])&(length>1e-12)
        angle=cp.arctan2(h[:,1],h[:,0]).astype(cp.float32)
        angle=cp.where(angle>=cp.float32(np.pi),-cp.float32(np.pi),angle)
        distance=h[:,2]/cp.maximum(length,1e-30)
        split=(-distance/cp.maximum(sigma,1e-30)).astype(cp.float32)
        h[:,0]=cp.where(active,cp.cos(angle),0.)
        h[:,1]=cp.where(active,cp.sin(angle),0.)
        h[:,2]=cp.where(active,-split*sigma,cp.inf)
    cert[:,9:15]=cert[:,3:9]
    return cert


def _coverage(cp,cert,xy):
    local=xy[None]-cert[:,None,:2]
    yes=cp.sum(local*local,axis=2)<=cert[:,None,2]**2+1e-10
    for slot in (3,6):yes&=cp.sum(local*cert[:,None,slot:slot+2],axis=2)+cert[:,None,slot+2]>=-1e-10
    return yes


def partition_multi_edge(snapshot,*,max_children,trim_cache=None):
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.rendering.sampled_analytic_clip import _publish_parent_fits,fit
    from cutsim.rendering.gs_sample_evaluation import disc
    from cutsim.rendering.cover_selection import select_cover, refine_opacity
    from cutsim.rendering.profile_cut_clip import ProfileClip
    from cutsim.rendering.opaque_coverage import display_factor, DEFAULT_DISPLAY_SCALE
    started=perf_counter();rt=runtime();cp=rt.cp
    region=snapshot.clipping.constraints.regions
    # Existing publication safeguards run even when all split attempts fail.
    keep,_,_,stats=_publish_parent_fits(snapshot,trim_cache=trim_cache)
    stats=dict(stats);parent_ms=stats.get('clip_partition_ms',0.)
    budget=max(0,min(max_children,MAX_CHILDREN))
    stats.update(clip_subdivision=True,clip_parent_only=False,clip_multi_edge=True,
        clip_split_parent_limit=MAX_PARENTS,clip_split_child_limit=budget,
        clip_split_per_parent=CHILDREN_PER_PARENT,clip_split_attempted=0,
        clip_split_candidates=0,clip_split_fallback=0,clip_split_budget_deferred=0,
        clip_split_cover_rejected=0,clip_split_opacity_rejected=0,
        clip_split_shared_samples_rejected=0,
        clip_split_holdout_rejected=0,clip_split_validation_points=512,
        clip_split_holdout_points=512,clip_split_unresolved_stops_calculation=False)
    parents=[];children=[]
    with rt.context():
        nr=len(region.rows);ns=len(region.sample_xy)
        if not hasattr(rt,'count_clip_boundaries'):
            rt.count_clip_boundaries=cp.RawKernel(COUNT_CUDA,'count_boundaries')
        counts=cp.zeros(nr,cp.int32)
        if nr:rt.count_clip_boundaries(((nr+63)//64,),(64,),
            (region.rows,region.starts,region.counts,region.enabled,region.face_proximity,
             region.face_masks,region.geometry,region.radii,region.sample_xy,np.int32(ns),np.int32(nr),counts))
        # Boundary count alone cannot establish representability: even one
        # curved boundary can defeat a straight-edge fit on the parent disk.
        region.nonconvex_residual=sampled_nonconvex_flags(rt,region)
        face_failed,needs_cover=parent_split_flags(cp,region,PARENT_SPLIT_TOLERANCE)
        eligible=cp.flatnonzero(needs_cover)
        stats['clip_split_candidates']=len(eligible)
        invalid=getattr(region,'twoedge_fallback',cp.zeros(nr,cp.uint8)).astype(bool)
        stats.update(clip_split_trigger='nonconvex_or_face_line_or_constraint_overflow',
            clip_split_nonconvex_candidates=int(cp.count_nonzero(needs_cover&region.nonconvex_residual).item()),
            clip_split_parent_tolerance=PARENT_SPLIT_TOLERANCE,
            clip_split_child_tolerance=TOLERANCE,
            clip_split_face_line_candidates=int(cp.count_nonzero(needs_cover&face_failed).item()),
            clip_split_low_boundary_candidates=int(cp.count_nonzero(needs_cover&(counts<3)).item()),
            clip_split_invalid_candidates=int(cp.count_nonzero(needs_cover&invalid).item()))
        # Prioritize invalid fits and nonconvex residuals over ordinary small
        # fit errors. Stable ties and existing caps keep work bounded.
        error=getattr(region,'twoedge_validation_error',region.error)
        priority=3.*invalid[eligible]+2.*region.nonconvex_residual[eligible]+cp.sum(cp.nan_to_num(error[eligible],nan=1.,posinf=1.,neginf=0.),axis=1)
        eligible=eligible[cp.argsort(-priority,kind='stable')]
        limit=min(MAX_PARENTS,budget//2)
        stats['clip_split_budget_deferred']=max(0,len(eligible)-limit)
        eligible=eligible[:limit]
        if len(eligible):
            # Compact dispatch metadata; all geometry and analytic tables stay resident.
            r=copy(region)
            for field in ('rows','starts','counts','sample_truth'):
                setattr(r,field,getattr(region,field)[eligible])
            # The deletion pass's all-face mask is the source of subdivision:
            # do not create a new truth/grid for seed placement or child fitting.
            dense_count=len(region.sample_xy)
            r.sample_xy=region.sample_xy
            truth=r.sample_truth;weights=region.sample_weights
            stats.update(clip_split_validation_points=dense_count,
                         clip_split_seed_source='shared_all_face_residual',
                         clip_split_reused_samples=len(eligible)*dense_count,
                         clip_split_one_line_per_face=True)
            _,cr,cs,cplanes=region.sweep_data
            affected=region.sweep_data[0]
            # Outer-support checks constrain the cover, but do not move seeds
            # or refit face lines from a replacement for the saved R samples.
            check_xy=cp.asarray(disc(16,32))
            check_truth=cp.empty((len(eligible),512),cp.uint8);check_weight=cp.empty(512)
            rt.priority_truth_kernel(((len(eligible)*512+127)//128,),(128,),
                (region.geometry,region.radii,r.rows,r.starts,r.counts,region.coefficients,region.enabled,
                 region.peer_surfaces,check_xy,np.int32(512),np.int32(len(eligible)),affected,
                 cr,cs,cplanes,np.int32(len(cr)),*region.floor_domain_data,check_truth,check_weight))
            stats['clip_split_support_check_points']=512
            angle=np.pi/32
            hold_xy=cp.asarray(disc(16,32))@cp.asarray([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            hold_truth=cp.empty((len(eligible),512),cp.uint8);hold_weight=cp.empty(512)
            rt.priority_truth_kernel(((len(eligible)*512+127)//128,),(128,),
                (region.geometry,region.radii,r.rows,r.starts,r.counts,region.coefficients,region.enabled,
                 region.peer_surfaces,hold_xy,np.int32(512),np.int32(len(eligible)),affected,
                 cr,cs,cplanes,np.int32(len(cr)),*region.floor_domain_data,hold_truth,hold_weight))
            r.sample_truth=truth;r.sample_weights=weights
            # Candidate centres follow surviving witnesses in 24 polar bins.
            ang=cp.arctan2(r.sample_xy[:,1],r.sample_xy[:,0])
            bins=cp.minimum((cp.sum(r.sample_xy**2,axis=1)*3).astype(cp.int32),2)*8+cp.minimum(((ang+cp.pi)/(2*cp.pi)*8).astype(cp.int32),7)
            # Reduction avoids float64 atomic-max rounding and exact equality
            # comparisons which can otherwise lose every seed in a bin.
            scores=cp.where(truth[:,:,None].astype(bool)&(bins[None,:,None]==cp.arange(24)[None,None,:]),weights[None,:,None],-1.)
            indices=cp.argmax(scores,axis=1)
            indices=cp.where(cp.max(scores,axis=1)>=0,indices,dense_count)
            points=r.sample_xy[cp.minimum(indices,dense_count-1)]
            R=region.radii[r.rows]
            radii=cp.where(indices<dense_count,cp.minimum(1.-cp.linalg.norm(points,axis=2),.7)*R[:,None],0.)
            # A central child is useful for three-sided residual islands;
            # polar witnesses alone bias all centres away from their core.
            points=cp.concatenate((points,cp.zeros((len(eligible),1,2))),axis=1)
            radii=cp.concatenate((radii,.7*R[:,None]),axis=1)
            # Retain a clipped, full-size central piece when possible. A wide
            # curved residual can reach the parent rim; smaller inscribed
            # disks alone cannot cover that rim within a small child budget.
            # A weighted bin centre complements the innermost witness: the
            # latter alone biases coverage inward on a long curved boundary.
            # Changing only the parent trigger must not remove the richer
            # curved-face seed set from a parent that is still selected.
            seed_face_failed=getattr(region,'face_line_failed',face_failed)
            low_boundary=(counts[eligible]<3)|invalid[eligible]|seed_face_failed[eligible]|region.nonconvex_residual[eligible]
            curved_candidates=bool(cp.any(low_boundary).item())
            nc=100 if curved_candidates else 75
            disks=cp.zeros((len(eligible),nc,5));disks[:,:,0]=cp.arange(len(eligible))[:,None];disks[:,:,4]=-1
            for level,scale in enumerate((1.,.85,.65)):
                sl=slice(level*25,(level+1)*25)
                disks[:,sl,1:3]=points*R[:,None,None];disks[:,sl,3]=radii*scale
            if curved_candidates:
                positive=cp.maximum(scores,0.);mass=positive.sum(axis=1)
                centres=cp.einsum('nsa,sd->nad',positive,r.sample_xy)/cp.maximum(mass[:,:,None],1e-30)
                centre_radii=cp.where((mass>0)&low_boundary[:,None],cp.minimum(1.-cp.linalg.norm(centres,axis=2),.7)*R[:,None],0.)
                disks[:,75:99,1:3]=centres*R[:,None,None];disks[:,75:99,3]=centre_radii
                disks[:,-1,3]=cp.where(low_boundary,R,0.)
            disks=disks.reshape(-1,5)
            cert=cp.empty((len(disks),15));err=cp.empty((len(disks),2));ok=cp.empty(len(disks),cp.uint8)
            fit(rt,r,disks,cert,err,ok,alternatives=1,partial=True,tolerance=TOLERANCE*.25,
                curvature_adjust=True,one_line_per_face=True)
            # Reject empty/invalid lines before native normalization.
            good=(ok!=0)&(cert[:,2]>1e-9)&cp.isfinite(cert[:,:3]).all(axis=1)
            for slot in (3,6):
                h=cert[:,slot:slot+3]
                good&=(cp.isfinite(h[:,:2]).all(axis=1)&
                    ((cp.isfinite(h[:,2])&(cp.linalg.norm(h[:,:2],axis=1)>1e-12))|cp.isposinf(h[:,2])))
            cert=_native_certificates(cp,cert).reshape(len(eligible),nc,15)
            good=good.reshape(len(eligible),nc)
            # These grids and normalized weights are identical for all parents.
            selection_xy=cp.concatenate((r.sample_xy,check_xy))
            selection_weights=cp.concatenate((weights/weights.sum(),check_weight/check_weight.sum()))
            total=selection_weights.sum()
            factor=display_factor(DEFAULT_DISPLAY_SCALE)
            parent_opaque=.99*cp.exp(-2*cp.sum(selection_xy**2,axis=1)/factor**2)>=.5
            hostrows=cp.asnumpy(r.rows)
            for i,row in enumerate(hostrows):
                if budget<2:
                    stats['clip_split_budget_deferred']+=len(hostrows)-i;break
                stats['clip_split_attempted']+=1
                xy=R[i]*selection_xy
                wanted=cp.concatenate((truth[i],check_truth[i])).astype(bool)
                coverage=_coverage(cp,cert[i],xy)
                target=cp.sum(selection_weights*wanted)
                valid=good[i]&(cp.sum(coverage*(~wanted)[None]*selection_weights,axis=1)/total<=TOLERANCE*.25)
                selected=[];union=cp.zeros(len(xy),cp.uint8)
                # Leave a margin for the independent staggered validation;
                # stopping exactly at its tolerance overfits the check grid.
                missing,excess=select_cover(rt,coverage.astype(cp.uint8),wanted.astype(cp.uint8),selection_weights,valid.astype(cp.uint8),union,selected,
                    per_parent=min(CHILDREN_PER_PARENT,budget),missing_total=cp.maximum(target,1e-30),total=total,tolerance=TOLERANCE*.5)
                if not selected or missing>TOLERANCE or excess>TOLERANCE:
                    stats['clip_split_cover_rejected']+=1
                    stats['clip_split_last_cover_missing']=missing
                    stats['clip_split_last_cover_excess']=excess
                    stats['clip_split_last_cover_children']=len(selected)
                    stats['clip_split_fallback']+=1;continue
                # Geometric cover can still be translucent. Add useful legal
                # children up to the SAME cap, scoring opacity gain on GPU.
                local=xy[None]-cert[i,:,None,:2]
                alpha=.99*cp.exp(-2*cp.sum(local*local,axis=2)/cp.maximum((cert[i,:,None,2]*factor)**2,1e-30))*coverage
                target_opaque=selection_weights*wanted*parent_opaque
                union=cp.any(coverage[cp.asarray(selected)],axis=0)
                current=1-cp.prod(1-alpha[cp.asarray(selected)],axis=0)
                lost=refine_opacity(rt,alpha,coverage,wanted,selection_weights,target_opaque,valid,
                    current,union,selected,dense_count=dense_count,per_parent=min(CHILDREN_PER_PARENT,budget),
                    total=total,tolerance=TOLERANCE)
                if lost>TOLERANCE:
                    stats['clip_split_opacity_rejected']+=1
                    stats['clip_split_last_opacity_missing']=lost
                    stats['clip_split_fallback']+=1;continue
                if len(selected)<2:
                    # A single smaller disk is not a subdivision. Let opacity
                    # repair add companions first, then require a real family.
                    stats['clip_split_single_child_deferred']=stats.get('clip_split_single_child_deferred',0)+1
                    stats['clip_split_fallback']+=1;continue
                chosen=cert[i,cp.asarray(selected)]
                # The normal fit may reuse R's draw/material UNION truth.
                # A denser analytic re-evaluation must not silently replace
                # that truth with the intersection of inherited clips.
                # Validate against the original shared samples as well.
                original_xy=R[i]*region.sample_xy
                original_cover=_coverage(cp,chosen,original_xy)
                original_union=cp.any(original_cover,axis=0)
                original_truth=region.sample_truth[eligible[i]].astype(bool)
                original_weights=region.sample_weights
                original_missing=cp.sum(original_weights*original_truth*~original_union)/cp.maximum(cp.sum(original_weights*original_truth),1e-30)
                original_excess=cp.sum(original_weights*~original_truth*original_union)/original_weights.sum()
                delta=original_xy[None]-chosen[:,None,:2]
                original_alpha=.99*cp.exp(-2*cp.sum(delta*delta,axis=2)/(chosen[:,None,2]*factor)**2)*original_cover
                original_opaque=1-cp.prod(1-original_alpha,axis=0)>=.5
                original_target=original_weights*original_truth*(.99*cp.exp(-2*cp.sum(region.sample_xy**2,axis=1)/factor**2)>=.5)
                original_lost=cp.sum(original_target*~original_opaque)/cp.maximum(original_target.sum(),1e-30)
                if bool(cp.any(cp.stack((original_missing,original_excess,original_lost))>TOLERANCE).item()):
                    stats['clip_split_shared_samples_rejected']+=1
                    stats['clip_split_last_shared_error']=cp.asnumpy(cp.stack((original_missing,original_excess,original_lost))).tolist()
                    stats['clip_split_fallback']+=1;continue
                # Independent staggered points must also pass before replacing
                # a parent; fitting quadrature alone can miss narrow slivers.
                hxy=R[i]*hold_xy;hcover=_coverage(cp,chosen,hxy)
                predicted=cp.any(hcover,axis=0);desired=hold_truth[i].astype(bool)
                hm=cp.sum(hold_weight*desired*~predicted)/cp.maximum(cp.sum(hold_weight*desired),1e-30)
                he=cp.sum(hold_weight*~desired*predicted)/hold_weight.sum()
                hd=hxy[None]-chosen[:,None,:2]
                ha=.99*cp.exp(-2*cp.sum(hd*hd,axis=2)/(chosen[:,None,2]*factor)**2)*hcover
                hopaque=1-cp.prod(1-ha,axis=0)>=.5
                htarget=hold_weight*desired*(.99*cp.exp(-2*cp.sum(hold_xy**2,axis=1)/factor**2)>=.5)
                ho=cp.sum(htarget*~hopaque)/cp.maximum(htarget.sum(),1e-30)
                if bool(cp.any(cp.stack((hm,he,ho))>TOLERANCE).item()):
                    stats['clip_split_holdout_rejected']+=1
                    stats['clip_split_last_holdout_error']=cp.asnumpy(cp.stack((hm,he,ho))).tolist()
                    stats['clip_split_last_holdout_children']=len(selected)
                    stats['clip_split_fallback']+=1;continue
                children.append(cp.asnumpy(chosen));parents.append(np.full(len(selected),row,np.int64))
                keep[row]=False;budget-=len(selected)
    pp=np.concatenate(parents) if parents else np.empty(0,np.int64)
    cc=np.concatenate(children) if children else np.empty((0,15))
    elapsed_ms=(perf_counter()-started)*1000
    stats.update(clip_partition_parents=int(np.count_nonzero(~keep)),clip_partition_children=len(pp),
        clip_partition_kept=len(region.rows)-len(parents),clip_partition_ms=elapsed_ms,
        clip_split_ms=max(0.,elapsed_ms-parent_ms))
    snapshot.clipping=ProfileClip(snapshot.clipping.planes,stats)
    snapshot.material_clipping=ProfileClip(snapshot.material_clipping.planes,dict(stats))
    return keep,pp,cc,stats
