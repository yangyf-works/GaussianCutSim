"""One bounded child-cell pass on unseeded neighbours of generated base cells.

The base sample's air/unknown classification does not certify the whole cell.
Only edge-adjacent supported chart cells are split, and every emitted child needs positive
BEFORE finite material evidence. Children are never recursively subdivided.
"""
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from cutsim.cuda.adaptive_profile import CHARTS, FIELDS, _generate


def _selected_template(template, parents):
    """Upload only selected parents; retain pose spans and original row order."""
    # parents are increasing, including the deterministic budget subsample.
    # Thus chart sorting in _generate has the same tie order after compaction.
    values = {name:np.asarray(getattr(template,name))[parents] for name in FIELDS}
    poses=[];offset=0
    for pose in template.poses:
        end=offset+pose['rays']
        count=int(np.searchsorted(parents,end)-np.searchsorted(parents,offset))
        if count:poses.append(dict(pose,rays=count))
        offset=end
    return SimpleNamespace(**values,poses=poses,local_boxes=template.local_boxes)


def refine_unseeded(surface, atlas, accepted_rays, command, sweep, block, group,
                    material_group, stock_bounds, *, max_rays=1024, ray_margin=None):
    """Append 2x2 child samples; preserve base ray IDs and accepted base GS."""
    from cutsim.cutting.tool_profile_state import RAY_STRIDE
    from cutsim.cuda.adaptive_device import generate_tool_gs_device
    if isinstance(max_rays, bool) or not isinstance(max_rays, int) or max_rays < 0:
        raise ValueError('Addition ray budget must be a nonnegative integer')
    started = perf_counter()
    template = getattr(atlas, 'profile_template', atlas)
    count = len(template.origins)
    accepted = np.asarray(accepted_rays, np.int64)
    if np.any(accepted < 0) or np.any(accepted >= count):
        raise ValueError('Accepted base rays must reference the base tool grid')
    missing = np.setdiff1d(np.arange(count), accepted)
    supported = missing[np.isin(template.charts[missing], CHARTS)]
    from cutsim.cuda.dexel_edge_cells import adjacent_unseeded_cells
    parents = adjacent_unseeded_cells(template, accepted, supported)
    selection_ms = (perf_counter()-started)*1000
    capacity = min(max_rays, max(0, RAY_STRIDE-len(atlas.origins)))//4
    stats = dict(addition_refinement_enabled=True, addition_refinement_divisions=2,
        addition_refinement_passes=0, addition_refinement_unseeded_cells=len(missing),
        addition_refinement_selection='generated_edge_neighbours',
        addition_refinement_adjacent_cells=len(parents),
        addition_refinement_nonadjacent_cells=len(supported)-len(parents),
        addition_refinement_selection_ms=selection_ms,
        addition_refinement_unsupported_cells=len(missing)-len(supported),
        addition_refinement_selected_cells=min(len(parents), capacity),
        addition_refinement_budget_skipped_cells=max(0,len(parents)-capacity),
        addition_refinement_budget=max_rays, addition_refinement_rays=0,
        addition_refinement_added=0, addition_refinement_ms=0.,
        addition_refinement_material_points=0, addition_refinement_air_points=0,
        addition_refinement_unknown_points=0, addition_refinement_intermediate_readback_bytes=0,
        addition_refinement_publication_readback_bytes=0, addition_refinement_uploaded_parent_cells=0)
    # Deterministic, spatially spread selection instead of consuming the budget
    # on the first end fan. Keep complete child families within the ray cap.
    if len(parents)>capacity:
        parents=parents[np.linspace(0,len(parents)-1,capacity,dtype=np.int64)]
    if not len(parents):
        stats['addition_refinement_ms']=(perf_counter()-started)*1000
        return None,None,stats
    rt=surface.rt;cp=rt.cp
    compact=_selected_template(template,parents)
    with rt.context():
        model=SimpleNamespace(_origins=cp.asarray(compact.origins),
                              _directions=cp.asarray(compact.directions))
        grid,source,_=_generate(surface,model,compact,np.arange(len(parents),dtype=np.int64),None,None,command,block,
            len(parents)*4,risk_mask=np.ones(len(parents),bool),divisions=2)
        source=cp.asarray(parents)[source]
        new,grid,source,generation=generate_tool_gs_device(surface,grid,source,command,sweep,block,
            group,material_group,atlas.local_boxes[block],len(atlas.origins),ray_margin,
            stock_bounds=stock_bounds)
    for value in (*vars(grid).values(),source):
        if isinstance(value,np.ndarray):value.flags.writeable=False
    descriptor=dict(grid=grid,parents=source,base_rays=len(atlas.origins),
                    replaced_parents=np.empty(0,np.int64))
    stats.update(addition_refinement_passes=1,addition_refinement_rays=len(grid.origins),
        addition_refinement_uploaded_parent_cells=len(parents),
        addition_refinement_added=len(new.endpoints['positions']),
        addition_refinement_material_points=len(new.endpoints['positions']),
        addition_refinement_air_points=generation['adaptive_analytic_air'],
        addition_refinement_unknown_points=generation['adaptive_analytic_unknown'],
        addition_refinement_publication_readback_bytes=generation['adaptive_publication_readback_bytes'],
        addition_refinement_ms=(perf_counter()-started)*1000)
    return new,descriptor,stats


def merge_descriptors(first,second):
    """Keep addition and later repair rays in their original publication order."""
    if first is None:return second
    if second is None:return first
    if first['base_rays']+len(first['grid'].origins)!=second['base_rays']:
        raise ValueError('Additional ray descriptors must be contiguous')
    values={name:np.concatenate((getattr(first['grid'],name),getattr(second['grid'],name)))
            for name in (*FIELDS,'tool_normals')}
    parents=np.concatenate((first['parents'],second['parents']))
    for value in (*values.values(),parents):value.flags.writeable=False
    return dict(grid=SimpleNamespace(**values),parents=parents,base_rays=first['base_rays'],
                replaced_parents=np.empty(0,np.int64))
