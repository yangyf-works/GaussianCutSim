"""Shared conservative finite-GS/sweep overlap candidates, never deletion proof."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from cutsim.rendering.gs_extent import support_factor
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import pack_cutters
from cutsim.cutting.local_dexel import bounds


@dataclass(frozen=True)
class SweepGSSelection:
    rows: np.ndarray
    mask: np.ndarray
    packed: tuple
    stats: dict


def select_sweep_gs(snapshot, sweep):
    """Reject only certified separation; include centres outside the cutter.

    Enclose both the drawing disk and the finite analytic face footprint
    (tangent radius R, normal displacement R). Test the swept AABB, then each
    convex part's planes and PSD quadric lower bound. Uncertain overlap stays
    a candidate. Existing clipping can only shrink these enclosing supports.
    """
    started = perf_counter()
    p = snapshot.endpoints['positions']
    sigma = snapshot.gaussian.attributes['scale'][:,0].astype(float)
    material = getattr(snapshot, 'material_clipping', None) or snapshot.clipping
    mr = support_factor(snapshot)*sigma
    if material is not None:
        mr = np.maximum(0., np.minimum(mr, material.planes[:,6]))
    radius = np.maximum(CLIP_SIGMA*sigma, np.sqrt(2.)*mr)
    packed = pack_cutters((sweep,))
    lo, hi = bounds(sweep)
    eps = 1e-8*(1.+np.abs(p).sum(axis=1)+radius)
    broad = np.flatnonzero((radius>0)&np.all(
        (p+radius[:,None]+eps[:,None]>=lo)&(p-radius[:,None]-eps[:,None]<=hi), axis=1))
    hit = np.zeros(len(broad), bool)
    centers, r, tolerance = p[broad], radius[broad], eps[broad]
    for record, span in zip(packed[0], packed[1]):
        possible = ~hit
        for plane in packed[2][span[0]:span[1]]:
            possible &= centers@plane[:3]+plane[3]+r*np.linalg.norm(plane[:3]) >= -tolerance
        if record[13]:
            delta = centers-record[:3]
            qd = delta@record[3:12].reshape(3,3)
            lower = np.sum(delta*qd,axis=1)-2*r*np.linalg.norm(qd,axis=1)
            possible &= lower <= record[12]**2+tolerance*(1.+abs(record[12]))
        hit |= possible
    rows = broad[hit]
    mask = np.zeros(len(p), bool); mask[rows] = True
    rows.flags.writeable = mask.flags.writeable = False
    return SweepGSSelection(rows, mask, packed, dict(
        sweep_gs_source_count=len(p), sweep_gs_broad_candidates=len(broad),
        sweep_gs_candidates=len(rows), sweep_gs_selection_ms=(perf_counter()-started)*1000,
        sweep_gs_selection='conservative_finite_support_overlap'))
