"""One bounded GPU refinement pass on finite foreign-surface crossings.

No historical cutters, GS splitting, or CPU intersection lists. Reuses the
BEFORE surface's BVH, and runs subtraction only on newly generated rays.
CPU ray metadata is read back after endpoint selection, solely for GS fitting,
replay and picking. The production material state remains the current GS.
"""
from copy import copy
from types import SimpleNamespace
from time import perf_counter
import numpy as np

CHARTS=np.array(['ball','cylinder','lower_cap','upper_cap','strip_ball','strip_side','strip_cap'])
FIELDS=('origins','directions','charts','cell_bounds','radii','pitches','ray_bounds','surface_distance','feeds')


def append_grid(base,extra):
    result=copy(base)
    for key in FIELDS:
        value=np.concatenate((getattr(base,key),getattr(extra,key)));value.flags.writeable=False
        setattr(result,key,value)
    if hasattr(base,'tool_normals'):
        result.tool_normals=np.concatenate((base.tool_normals,extra.tool_normals))
    result.profile_count=len(result.origins)
    result.poses=[*base.poses,dict(base.poses[0],part='adaptive',rays=len(extra.origins))] if len(extra.origins) else list(base.poses)
    return result


def remove_parent_rays(atlas,parents):
    """Compact the ray atlas and pose spans; removed rays have no public ID."""
    keep=np.ones(len(atlas.origins),bool);keep[np.asarray(parents,dtype=np.int64)]=False
    result=copy(atlas)
    for key in (*FIELDS,'tool_normals'):
        if hasattr(atlas,key):
            value=getattr(atlas,key)[keep];value.flags.writeable=False;setattr(result,key,value)
    result.profile_count=int(keep.sum());result.poses=[];offset=0
    for pose in atlas.poses:
        end=offset+pose['rays'];count=int(keep[offset:end].sum());offset=end
        if count:result.poses.append(dict(pose,rays=count))
    return result


def _generate(surface,model,atlas,endpoint_ids,planes,sigma,command,block,budget,*,risk_mask=None,priority=None,divisions=2):
    """Select risky parents and evaluate child cells in tool coordinates."""
    cp=surface.rt.cp
    if np.any(endpoint_ids<0) or np.any(endpoint_ids>=len(atlas.origins)):
        raise ValueError('Adaptive parents require local ray IDs, not publication IDs')
    chart_codes=np.full(len(atlas.origins),-1,np.int32)
    for i,name in enumerate(CHARTS):chart_codes[atlas.charts==name]=i
    ids=cp.asarray(endpoint_ids)
    codes=cp.asarray(chart_codes)
    if risk_mask is None:
        # Compatibility for offline probes; production does not calculate
        # clipping to select additional Dexels.
        clips=cp.asarray(planes);radius=2.*cp.asarray(sigma)
        distance=cp.minimum(cp.minimum(clips[:,2],clips[:,5]),clips[:,6])
        risk=(distance<radius)&(distance>-radius)&(clips[:,6]>0)
    else:risk=cp.asarray(risk_mask,dtype=cp.bool_)
    risk=risk&(codes[ids]>=0)
    candidates=cp.flatnonzero(risk)
    if len(candidates):
        # Curved groove intersections lie predominantly on the translated
        # middle. Do not spend the entire cap on end fans stored first in the
        # atlas; retain deterministic order within each chart category.
        keys=[ids[candidates],(codes[ids[candidates]]<4).astype(cp.int64)]
        if priority is not None:keys.append(~cp.asarray(priority)[candidates])
        order=cp.lexsort(cp.stack(keys))
        candidates=candidates[order]
    # Stable row order interleaves both sides of a profile. Keep whole 2x2
    # families under the cap so the last side is not selected arbitrarily.
    if divisions not in (2,4,8):raise ValueError('Chart subdivisions must be 2, 4 or 8')
    family=divisions*divisions
    parents=ids[candidates[:budget//family]]
    candidate_count=len(candidates)
    if not len(parents):return None,None,candidate_count
    replaced_parents=parents.copy()
    parents=cp.repeat(parents,family)
    # Pack existing CPU chart descriptors once. All child evaluation and
    # rejection runs in one CUDA kernel, avoiding per-chart boolean syncs.
    tips=np.empty((len(atlas.origins),3));signs=np.zeros(len(atlas.origins));margins=np.empty(len(atlas.origins));profile=np.empty((len(atlas.origins),3));offset=0
    for pose in atlas.poses:
        end=offset+pose['rays'];tips[offset:end]=pose['tip'];margins[offset:end]=pose['ray_margin_mm']
        profile[offset:end,:2]=pose.get('axial_bounds',(command.cutting_range.start,command.cutting_range.end))
        profile[offset:end,2]=pose['kind']=='ball'
        signs[offset:end]={'start':-1.,'end':1.}.get(pose['part'],0.);offset=end
    from cutsim.core.tool_pose import DEFAULT_TOOL_AXIS, tool_basis, box_in_frame
    indexed = command.tool_axis != DEFAULT_TOOL_AXIS
    origins, directions, feeds = model._origins, model._directions, atlas.feeds
    box = atlas.local_boxes[block]
    if indexed:
        basis = tool_basis(command.tool_axis)
        origin = np.asarray(command.a)
        rotation = cp.asarray(basis)
        origins = cp.ascontiguousarray((origins-cp.asarray(origin))@rotation)
        directions = cp.ascontiguousarray(directions@rotation)
        feeds = feeds@basis
        tips = (tips-origin)@basis
        box = box_in_frame(*box, basis, origin)
    packed=cp.asarray(np.column_stack((atlas.cell_bounds,atlas.radii,atlas.pitches,atlas.surface_distance,
                                      feeds,tips,margins,signs,profile)))
    from cutsim.cuda.adaptive_profile_kernel import KERNEL
    kernel=cp.RawKernel(KERNEL,'adaptive_children',options=('--fmad=false',))
    output=cp.empty((len(parents),18),cp.float64);valid=cp.empty(len(parents),cp.uint8)
    kernel(((len(parents)+127)//128,),(128,),(parents,np.int64(len(parents)),origins,directions,
        packed,codes,cp.asarray(np.r_[box[0],box[1]]),np.int32(divisions),output,valid))
    if indexed:
        output[:,:3] = output[:,:3]@rotation.T+cp.asarray(origin)
        output[:,3:6] = output[:,3:6]@rotation.T
        output[:,15:18] = output[:,15:18]@rotation.T
        # The rotated local box encloses the work-coordinate window. Apply
        # the original window before accepting child rays.
        for axis in range(3):
            o, d = output[:,axis], output[:,3+axis]
            parallel = cp.abs(d)<1.e-15
            denominator = cp.where(parallel, 1., d)
            lo = (atlas.local_boxes[block][0][axis]-o)/denominator
            hi = (atlas.local_boxes[block][1][axis]-o)/denominator
            output[:,13] = cp.maximum(output[:,13], cp.where(parallel,-cp.inf,cp.minimum(lo,hi)))
            output[:,14] = cp.minimum(output[:,14], cp.where(parallel,cp.inf,cp.maximum(lo,hi)))
            valid &= (~parallel|((o>=atlas.local_boxes[block][0][axis])&(o<=atlas.local_boxes[block][1][axis]))).astype(cp.uint8)
        valid &= (output[:,13]<output[:,14]).astype(cp.uint8)
    selected=cp.flatnonzero(valid);output=output[selected];parents=parents[selected]
    values=dict(origins=output[:,:3],directions=output[:,3:6],charts=codes[parents],cell_bounds=output[:,6:10],
        radii=output[:,10],pitches=output[:,11],surface_distance=output[:,12],ray_bounds=output[:,13:15],feeds=output[:,15:18])
    return SimpleNamespace(**{k:cp.ascontiguousarray(v) for k,v in values.items()},
                           replaced_parents=replaced_parents),parents,candidate_count



def refine(surface,model,atlas,ep,generated,command,sweep,block,budget):
    started=perf_counter();rt=surface.rt;extra=None
    stats=dict(adaptive_budget=budget,adaptive_candidates=0,adaptive_rays=0,adaptive_accepted_endpoints=0,
        adaptive_seed_ms=0.,adaptive_cut_ms=0.,adaptive_seed_payload_readback_bytes=0,adaptive_total_ms=0.,
        adaptive_seed_control_readback_bytes=0,adaptive_endpoint_readback_bytes=0)
    if budget<4 or not len(ep['positions']):return None,None,stats
    try:
        with rt.context():
            centered=getattr(model,'center_ray_update',False)
            parent_ep=ep;proposed=generated;priority=None
            if centered and getattr(model,'complete_unselected',False):
                # A zero-thickness contact parent may generate no GS, while
                # its two neighbouring child samples lie on real material.
                # Include bounded, rejected tool cells in adaptive selection.
                from cutsim.cuda.profile_program import _snapshot
                nominal=atlas.origins+atlas.surface_distance[:,None]*atlas.directions
                bounds=surface.boxes[1]
                bounded=rt.cp.all((rt.cp.asarray(nominal)>=bounds[:3])&(rt.cp.asarray(nominal)<=bounds[3:]),axis=1)
                valid=bounded & (rt.cp.asarray(atlas.surface_distance)>=rt.cp.asarray(atlas.ray_bounds[:,0])) & (rt.cp.asarray(atlas.surface_distance)<=rt.cp.asarray(atlas.ray_bounds[:,1]))
                candidates=surface._download(rt.cp.flatnonzero(valid))
                parent_ep=dict(positions=nominal[candidates],normals=atlas.tool_normals[candidates],ray_ids=candidates,
                    distances=atlas.surface_distance[candidates],owners=np.zeros(len(candidates),np.int64),tool_ids=np.full(len(candidates),command.tool_id,np.int32))
                group=int((generated.material_surface_ids.min()-7)//3) if len(generated.material_surface_ids) else 0
                proposed=_snapshot(parent_ep,atlas,[command],surface_groups=np.array([group]))
                proposed.material_surface_ids=proposed.gaussian.attributes['surface_id'][:,0]
                priority=model.added[rt.cp.asarray(candidates)]==0
            risk=surface.boundary_risk_device(proposed)
            if priority is not None and len(parent_ep['positions']):
                # Opposing nearly tangent walls can have no transverse GS
                # crossing at all. Refine their curved tool cells as well.
                signed,owners=surface.query_device(rt.cp.asarray(parent_ep['positions']))
                safe=rt.cp.maximum(owners,0)
                old_normals=rt.cp.asarray(surface.snapshot.endpoints['normals'])[safe]
                old_ids=rt.cp.asarray(surface.snapshot.material_surface_ids)[safe]
                opposite=rt.cp.sum(old_normals*rt.cp.asarray(parent_ep['normals']),axis=1)<-.5
                foreign=old_ids!=rt.cp.asarray(proposed.material_surface_ids)
                proximity=(owners>=0)&opposite&foreign&(rt.cp.abs(signed)<.1*rt.cp.asarray(atlas.pitches[parent_ep['ray_ids']]))
                risk |= proximity;priority |= proximity
            template=getattr(atlas,'profile_template',atlas)
            generation_model=(SimpleNamespace(_origins=rt.cp.asarray(template.origins),_directions=rt.cp.asarray(template.directions)) if centered else model)
            grid,parents,candidates=_generate(surface,generation_model,template,parent_ep['ray_ids'],None,
                proposed.gaussian.attributes['scale'][:,0],command,block,budget,risk_mask=risk,priority=priority)
            stats['adaptive_candidates']=candidates
            if grid is None:
                stats['adaptive_total_ms']=(perf_counter()-started)*1000;return None,None,stats
            replaced=surface._download(grid.replaced_parents)
            if centered:model.replace_parent_rays(replaced)
            stats['adaptive_rays']=len(grid.origins)
            if centered:
                from cutsim.cuda.center_ray_update import CenterRayModel,center_grid
                grid=center_grid(grid,sweep,atlas.local_boxes[block],xp=rt.cp,margin=getattr(model,'ray_margin',None))
                extra=CenterRayModel(surface,grid,parent=model,device_input=True)
            else:extra=surface.seed_device(grid,device_input=True,source_model=model)
            stats['adaptive_seed_ms']=extra.stats['initialization_s']*1000
            stats['adaptive_seed_payload_readback_bytes']=extra.stats['seed_payload_readback_bytes']
            stats['adaptive_seed_control_readback_bytes']=extra.stats['seed_control_readback_bytes']
            extra.step_device(sweep);stats['adaptive_cut_ms']=extra.stats['last_device_step_s']*1000
            if centered:
                stats['adaptive_miss_material_queries']=extra.stats['miss_material_candidates']
                stats['adaptive_miss_material_added']=extra.stats['miss_material_added']
                stats['adaptive_hit_material_rejected']=extra.stats['hit_material_rejected']
            arrays=extra.endpoints_device(first_current=True)
            accepted=(rt.cp.ones(len(arrays['positions']),rt.cp.bool_) if centered else
                surface.endpoint_mask_device(arrays['positions'],extra.initial_hit_counts[arrays['ray_ids']]>0))
            chosen=rt.cp.flatnonzero(accepted)
            if getattr(extra,'shared_hits',None) is not None:
                extra.shared_hits.consume(extra,arrays['ray_ids'][chosen],ray_base=len(atlas.origins))
            result={k:extra._download(v[chosen]) for k,v in arrays.items()}
            stats['adaptive_endpoint_readback_bytes']=sum(v.nbytes for v in result.values())
            stats['adaptive_accepted_endpoints']=len(chosen)
            # Only final endpoints and bounded debug/grid descriptors leave
            # CUDA. Intersection lists and intervals never leave the GPU.
            cpu={k:surface._download(getattr(grid,k)) for k in FIELDS}
            if centered:cpu['tool_normals']=surface._download(grid.tool_normals)
            cpu['charts']=CHARTS[cpu['charts']]
            parent_ids=surface._download(parents)
            for a in [*cpu.values(),parent_ids]:a.flags.writeable=False
            replaced.flags.writeable=False
            descriptor=dict(grid=SimpleNamespace(**cpu),parents=parent_ids,base_rays=len(atlas.origins),
                            source_base_rays=len(atlas.origins),replaced_parents=replaced)
            stats['adaptive_metadata_bytes']=sum(a.nbytes for a in cpu.values())+parent_ids.nbytes+replaced.nbytes
            stats['adaptive_total_ms']=(perf_counter()-started)*1000
            return descriptor,result,stats
    finally:
        if extra is not None:extra.close()
