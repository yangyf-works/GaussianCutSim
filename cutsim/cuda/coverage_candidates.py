"""Select boundary and ungenerated tool cells for the single final audit.

Selection is CPU metadata only: no material query, coverage kernel or GS
creation occurs here. Boundary and ordinary coverage-grid selections have
independent budgets, and their union is deduplicated later.
"""
import numpy as np
from cutsim.cutting.tool_profile_state import RAY_STRIDE


BOUNDARY_MAX_PROBES = 16384


def boundary_coverage_cells(after, atlas, block, *, adaptive=None,
                            unseeded_parents=None, max_probes=BOUNDARY_MAX_PROBES):
    stats = dict(boundary_selected_cells=0)
    base=len(atlas.origins);ep=after.endpoints
    local=ep['ray_ids']-(block+1)*RAY_STRIDE
    current=(ep['owners']==block)&(local>=0)
    clips=after.clipping.planes;scale=after.gaussian.attributes['scale'][:,0]
    mapped=local.copy()
    if adaptive is not None:
        extra=current&(mapped>=base)&(mapped<base+len(adaptive['parents']))
        mapped[extra]=adaptive['parents'][mapped[extra]-base]
    eligible=current&(mapped<base)
    # Clipped edges and curvature-capped footprints narrower than their cells
    # can both leave gaps. Coverage testing below rejects already covered cells.
    narrow=np.zeros(len(local),bool)
    narrow[eligible]=2.*scale[eligible]<.8*atlas.pitches[mapped[eligible]]
    risk=eligible & ((np.minimum(np.minimum(clips[:,2],clips[:,5]),clips[:,6])<2.*scale)|narrow)
    parents=np.unique(mapped[risk])
    # Cells without a generated centre may still contain boundary material.
    # Probe them directly, independently of Dexel / old-GS intersections.
    extra=np.asarray((),dtype=np.int64) if unseeded_parents is None else np.asarray(unseeded_parents,dtype=np.int64)
    extra=np.unique(extra[(extra>=0)&(extra<base)])
    stats['boundary_unseeded_cells']=len(extra)
    stats['boundary_unseeded_extra_cells']=len(np.setdiff1d(extra,parents))
    parents=np.union1d(parents,extra)
    if adaptive is not None:
        # Include boundary cells whose adaptive samples were all rejected:
        # the unsampled part of such a cell can still contain real material.
        parents=np.union1d(parents,adaptive['parents'])
        parents=parents[(parents>=0)&(parents<base)]
    stats['boundary_candidate_cells']=len(parents)
    stats['boundary_probe_budget_skipped']=max(0,len(parents)*16-max_probes)
    if not len(parents):return parents,stats
    if len(parents)>max_probes//16:
        parents=parents[np.linspace(0,len(parents)-1,max_probes//16,dtype=np.int64)]
    stats['boundary_selected_cells']=len(parents)
    return parents,stats
