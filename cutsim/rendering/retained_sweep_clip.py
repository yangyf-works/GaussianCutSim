"""Current-sweep edges for surviving old GS; transient R-workspace reuse."""
from types import SimpleNamespace
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA, update_candidates
from cutsim.rendering.profile_cut_clip import prepare_cut_clipping


def retained_sweep_edges(previous, kept, sweep, backend, *, stock_bounds=None,
                         current_surface_id=None, workspace=None):
    """Return retained-prefix rows and uncompressed analytic constraints.

    R's broad candidates are reused, NOT its sampled hit/non-hit result:
    quadrature can miss a narrow boundary. No prior command is consulted.
    The cache is owned by compute_next and released at the end of that block.
    """
    source = np.flatnonzero(kept)
    identities = np.atleast_1d(current_surface_id) if current_surface_id is not None else np.empty(0)
    reusable = (workspace is not None and workspace.get('snapshot') is previous
                and workspace.get('sweep') is sweep
                and workspace.get('radius_sigma') == CLIP_SIGMA)
    cached = workspace.get('clip_cache') if reusable else None
    if (cached is not None and np.array_equal(cached[0], source)
            and np.array_equal(cached[1], identities)):
        rows, clip, stats = cached[2:]
        return rows, clip, dict(stats, sweep_clip_cache_reused=True, sweep_clip_kernel_ms=0.)
    if reusable:
        selected = np.zeros(len(kept), bool)
        selected[workspace['ids']] = True
    else:
        selected = update_candidates(previous.endpoints['positions'], previous.gaussian.u,
            previous.gaussian.v, previous.gaussian.attributes['scale'][:, 0], sweep)
    selected &= kept
    ids = getattr(previous, 'material_surface_ids', None)
    if ids is not None:
        selected &= ~np.isin(ids, identities)
    rows = np.flatnonzero(selected[source])
    selected_ids = source[rows]
    clip = None
    if len(rows):
        subset = SimpleNamespace(endpoints={'positions': previous.endpoints['positions'][selected_ids]},
            gaussian=SimpleNamespace(u=previous.gaussian.u[selected_ids], v=previous.gaussian.v[selected_ids],
                attributes={'scale': previous.gaussian.attributes['scale'][selected_ids]}))
        low, high = stock_bounds if stock_bounds is not None else (np.full(3, -np.inf), np.full(3, np.inf))
        clip = prepare_cut_clipping(subset, (sweep,), low, high, backend, collect_edges=True,
            packed_cutters=workspace.get('packed_cutters') if reusable else None,
            device_cutters=workspace.get('device_cutters') if reusable else None)
    stats = dict(sweep_clip_candidates=len(rows), sweep_clip_r_candidates_reused=reusable,
        sweep_clip_cache_reused=False, sweep_clip_kernel_ms=0. if clip is None else clip.stats.get('kernel_ms') or 0.)
    if reusable:
        workspace['clip_cache'] = (source, identities.copy(), rows, clip, stats)
    return rows, clip, stats
