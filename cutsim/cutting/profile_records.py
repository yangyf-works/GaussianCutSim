"""Immutable, per-GS calculation records shared with existing playback."""
import numpy as np
from cutsim.core.splat_cloud import SplatCloud, _rt_snapshot_cloud
from cutsim.core.local_state import _local_initial_state


def cloud_from_snapshot(snapshot):
    ep=snapshot.endpoints; n=len(ep['positions'])
    cloud=SplatCloud(ep['positions'],ep['normals'],
        snapshot.gaussian.attributes['scale'][:,0].astype(float)/.75,
        ep['tool_ids'],np.zeros(n,np.int16))
    if snapshot.clipping is None:
        attributes={k:v.copy() for k,v in snapshot.gaussian.attributes.items()}
        for slot in (1,2):
            attributes[f'clip{slot}_angle']=np.full((n,1),4.,np.float32)
            attributes[f'clip{slot}_split']=np.zeros((n,1),np.float32)
    else:
        from cutsim.rendering.profile_cut_clip import clipped_draw
        attributes=clipped_draw(snapshot,np.ones(n,bool))
    # CPU-only provenance: smoothing surface IDs may join multiple blocks.
    attributes['machining_owner']=np.asarray(ep['owners'],dtype=np.int32).copy()
    for value in attributes.values():value.flags.writeable=False
    cloud._profile_attributes=attributes
    _local_initial_state(cloud)
    return _rt_snapshot_cloud(cloud)


def render_records(cloud,settings):
    """Appearance only: never fit or subdivide a completed clipping result."""
    from cutsim.rendering.gaussian_attributes import _gaussian_shading
    attributes={k:v.copy() for k,v in cloud._profile_attributes.items()}
    from cutsim.rendering.opaque_coverage import apply_display_scale
    apply_display_scale(attributes, settings.sigma_scale)
    for slot in (1,2):
        if not settings.half_clip:
            attributes[f'clip{slot}_angle'].fill(4.)
            attributes[f'clip{slot}_split'].fill(0.)
    attributes['source_ids']=np.arange(len(cloud.positions),dtype=np.int64)
    attributes['surface_color']=_gaussian_shading(cloud,attributes['source_ids'],cloud.normals,settings)[0].astype(np.float32)
    # Existing workspace diagnostics distinguish native two-edge records by
    # these fields. They describe the completed representation, never refit it.
    from cutsim.rendering.clipping import NativeClipArrays
    attributes=NativeClipArrays(attributes)
    n=len(cloud.positions)
    count=(attributes['clip1_angle'][:,0]!=4.).astype(np.int32)+(attributes['clip2_angle'][:,0]!=4.)
    for name in ('clip_edge_count','clip_candidate_count','clip_ridge_count','clip_eligible_count'):
        attributes[name]=count.copy()
    for name in ('clip_pending','clip_empty','clip_sweep_top2','clip_occlusion_fallback',
                 'clip_concave','clip_join_uncertain','clip_approximated'):
        attributes[name]=np.zeros(n,bool)
    attributes['clip_reason']=np.zeros(n,np.int8)
    attributes['edge_case']=np.zeros(n,np.int16)
    for name in ('clip_partition_guard','clip_omitted_mass'):
        attributes[name]=np.zeros(n,np.float32)
    attributes['clip_second_values']=np.tile(np.array([4.,0.,4.,0.],np.float32),(n,1))
    attributes['clip_selected_indices']=np.full((n,2),-1,np.int32)
    attributes['clip_selection_scores']=np.zeros((n,2),np.float32)
    for value in attributes.values():value.flags.writeable=False
    return attributes
