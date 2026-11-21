"""GaussianCutSim rendering / gaussian_attributes implementation."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
import hashlib
import math
import numpy as np
import time
from cutsim.core.material import AnalyticMaterial
from cutsim.core.splat_cloud import SplatCloud
from cutsim.geometry.frames import rotation_matrices_to_quaternions, tangent_frames
from cutsim.rendering.settings import RenderSettings


MACHINING_TOOL_PALETTE = (
    (.32, .58, .96),                  # Blue
    (1.00, .76, .30),                 # Gold
    (.76, .45, .96),                  # Purple
    (.28, .85, .85),                  # Cyan
    (232 / 255, 91 / 255, 112 / 255),  # Crimson
    (.32, .81, .47),                  # Green
    (233 / 255, 143 / 255, 200 / 255), # Pink
    (143 / 255, 158 / 255, 78 / 255),  # Olive
)


def machining_tool_colors(tool_ids: np.ndarray) -> np.ndarray:
    """Cycle positive tool IDs through eight colors; stock stays gray."""
    ids = np.asarray(tool_ids).reshape(-1)
    colors = np.tile([.56, .62, .70], (len(ids), 1))
    cut = ids > 0
    palette = np.asarray(MACHINING_TOOL_PALETTE)
    colors[cut] = palette[((ids[cut] - 1) % len(palette)).astype(np.intp)]
    return colors


def balanced_point_groups(points: np.ndarray, max_points: int = 512):
    """Same bounded spatial grouping idea as the reference implementation."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pending = [np.arange(len(points), dtype=np.int64)]
    while pending:
        ids = pending.pop()
        if not len(ids):
            continue
        if len(ids) <= max_points:
            yield ids
            continue
        axis = int(np.argmax(np.ptp(points[ids], axis=0)))
        order = ids[np.argsort(points[ids, axis], kind="stable")]
        middle = len(order) // 2
        pending.append(order[middle:])
        pending.append(order[:middle])


def _reference_half_plane_support_group(points: np.ndarray, normals: np.ndarray,
                                        sizes: np.ndarray, material: AnalyticMaterial,
                                        sigma_scale: float,
                                        min_intersection_angle_deg: float = 5.0):
    """Reference-style geometric Half-GS decision, without a global grid.

    This follows the original _half_plane_support_group algorithm:
      1. Treat the six stock planes and nearby cutter-sweep boundaries as
         candidate clipping surfaces.
      2. Project each candidate gradient into the current splat tangent plane.
      3. Estimate the nearest tangent-plane intersection analytically.
      4. Probe just beyond that candidate on both sides of the current surface.
         If the normal material boundary still continues there, reject it.
      5. Use the closest distinct clipping boundary for the Half-GS direction
         and the second distinct boundary to shrink sigma near corners.

    Group candidate discovery uses a persistent cutter index, group bounds and
    per-point filters. Ties follow cutter history order.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    h = np.asarray(sizes, dtype=np.float64).reshape(-1)
    count = len(p)
    sigma = np.maximum(h * sigma_scale, 1e-8)
    nearest = np.full(count, np.inf)
    second = np.full(count, np.inf)
    inward = np.zeros((count, 3), dtype=np.float64)
    second_inward = np.zeros((count, 3), dtype=np.float64)
    if not count:
        u, v = tangent_frames(n)
        return (sigma, u, v, np.zeros(0), np.zeros(0, dtype=bool), 0, 0,
                nearest, second, inward, second_inward)

    reach = 2.5 * sigma
    all_ids = np.arange(count, dtype=np.int64)
    items = []

    # The six stock half-spaces are identical to the reference implementation.
    for axis in range(3):
        g = np.eye(3, dtype=np.float64)[axis]
        items.append((all_ids, p[:, axis] - material.stock_hi[axis],
                      np.broadcast_to(g, p.shape)))
        items.append((all_ids, material.stock_lo[axis] - p[:, axis],
                      np.broadcast_to(-g, p.shape)))

    # Reference CSGField.query_primitives_aabb() + per-point expanded AABB test,
    # using the persistent analytic cutter index.
    group_lo = p.min(axis=0) - reach.max()
    group_hi = p.max(axis=0) + reach.max()
    candidate_sweeps = 0
    for sweep_id in material.query_sweep_ids(group_lo,group_hi):
        sweep=material.sweeps[sweep_id]
        box_lo, box_hi = sweep.world_aabb()
        ids = np.flatnonzero(np.all(p + reach[:, None] >= box_lo, axis=1) &
                             np.all(p - reach[:, None] <= box_hi, axis=1))
        if not len(ids):
            continue
        candidate_sweeps += 1
        d, g = sweep.sdf_gradient(p[ids])
        # Material is stock MINUS cutter: candidate field/gradient are -d/-g.
        items.append((ids, -d, -g))

    lengths = [len(item[0]) for item in items]
    offsets = np.r_[0, np.cumsum(lengths)]
    ids = np.concatenate([item[0] for item in items])
    values = np.concatenate([item[1] for item in items])
    gradients = np.concatenate([item[2] for item in items])

    nn = n[ids]
    tangent = gradients - np.sum(gradients * nn, axis=1)[:, None] * nn
    length = np.linalg.norm(tangent, axis=1)
    direction = -tangent / np.maximum(length[:, None], 1e-30)
    distance = np.maximum(-values, 0.0) / np.maximum(length, 1e-30)

    # The reference implementation rejected candidate intersections whose
    # tangent component was below sin(45 deg).  That suppresses real acute
    # CSG edges: e.g. the demo's blue/green crossing has surface normals about
    # 139.9 deg apart, so the complementary tangent angle is only ~40.1 deg.
    # Keep only truly near-parallel/ill-conditioned candidates instead.
    # The continuation probe below still rejects boundaries that do not really
    # terminate the current surface.
    # ``length`` is sin(theta), where theta is the angle between the rendered
    # splat normal and candidate-boundary normal.  Five degrees is retained only
    # as a numerical conditioning floor, not as a seam/hole workaround.
    min_intersection_angle_deg = float(np.clip(min_intersection_angle_deg, 0.0, 89.0))
    distance[length < np.sin(np.deg2rad(min_intersection_angle_deg))] = np.inf
    probe = np.flatnonzero(np.isfinite(distance) & (distance < 2.5 * sigma[ids]))

    material_queries = 0
    if len(probe):
        pid = ids[probe]
        travel = np.maximum(h[pid] * 0.1, 1e-5)
        eps = np.maximum(h[pid] * 0.01, 1e-6)
        q = p[pid] - direction[probe] * (distance[probe] + travel)[:, None]
        q_in = q - n[pid] * eps[:, None]
        q_out = q + n[pid] * eps[:, None]
        both = np.vstack([q_in, q_out])
        vv = material.value(both)
        vi, vo = vv[:len(q)], vv[len(q):]
        # If the regular material boundary continues beyond the candidate, it is
        # not a clipping edge.  This is the reference continuation test.
        distance[probe[(vi < -eps * 0.1) & (vo > eps * 0.1)]] = np.inf
        material_queries = 2 * len(probe)

    # Keep two geometrically distinct clipping directions, matching the 25 deg
    # separation test from the reference code.
    cos25 = np.cos(np.deg2rad(25.0))
    for a, z in zip(offsets[:-1], offsets[1:]):
        ii = ids[a:z]
        dd = distance[a:z]
        dr = direction[a:z]
        old = nearest[ii].copy()
        old_dir = inward[ii].copy()
        distinct = np.sum(dr * old_dir, axis=1) < cos25
        better = dd < old

        # A new, closer DISTINCT edge promotes the previous nearest edge to
        # second place. Keep its direction as well as its distance to
        # build one child Half2DGS for each physical intersection.
        move = better & distinct & np.isfinite(old)
        if np.any(move):
            jj = ii[move]
            replace_second = old[move] < second[jj]
            if np.any(replace_second):
                kk = jj[replace_second]
                second[kk] = old[move][replace_second]
                second_inward[kk] = old_dir[move][replace_second]

        # A farther distinct edge may itself become the second edge.
        nxt = (~better) & distinct & np.isfinite(dd)
        if np.any(nxt):
            jj = ii[nxt]
            replace_second = dd[nxt] < second[jj]
            if np.any(replace_second):
                kk = jj[replace_second]
                second[kk] = dd[nxt][replace_second]
                second_inward[kk] = dr[nxt][replace_second]

        nearest[ii[better]] = dd[better]
        inward[ii[better]] = dr[better]

    sigma = np.minimum(sigma, np.maximum(second / 2.5, h * 0.025))
    active = np.isfinite(nearest) & (nearest < 2.5 * sigma)
    u, v = tangent_frames(n)
    u[active] = inward[active]
    # Numerical safeguard only; the reference direction is already tangent.
    u[active] -= np.sum(u[active] * n[active], axis=1)[:, None] * n[active]
    u[active] /= np.maximum(np.linalg.norm(u[active], axis=1)[:, None], 1e-30)
    v[active] = np.cross(n[active], u[active])
    split = np.zeros(count, dtype=np.float64)
    split[active] = -nearest[active] / sigma[active]
    return (sigma, u, v, split, active, candidate_sweeps, material_queries,
            nearest, second, inward, second_inward)


def _force_coincident_sharp_edge_masks(points: np.ndarray, normals: np.ndarray,
                                       sizes: np.ndarray, sigma: np.ndarray,
                                       u: np.ndarray, v: np.ndarray,
                                       split: np.ndarray, active: np.ndarray,
                                       min_angle_deg: float = 5.0):
    """Force Half-GS at exact face-pair samples emitted for a sharp CSG edge.

    ``split_face_samples()`` deliberately emits coincident samples with different
    face normals at a true edge.  A primitive SDF has only one subgradient at a
    non-differentiable rim, so a later primitive-level Half-GS query can miss the
    *other* incident face even though the face pair is already known exactly.

    Group coincident splats by position.  For every member, project a peer face
    normal into its tangent plane and use the same orientation convention as the
    reference Half-GS code (u = -projected peer normal).  The edge passes through
    the splat centre, therefore split=0 exactly.  This is not a heuristic edge
    detector: only the coincident multi-normal samples explicitly created by the
    sharp-face extractor are affected.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    h = np.asarray(sizes, dtype=np.float64).reshape(-1)
    if len(p) < 2:
        return 0

    quant = max(float(np.min(h, initial=1.0)) * 2e-4, 1e-8)
    keys = np.rint(p / quant).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    cuts = np.r_[0, np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1,
                 len(order)]
    # Skip singleton groups before the Python loop; only shared positions
    # require intersection processing.
    sizes_group = np.diff(cuts)
    multi = np.flatnonzero(sizes_group >= 2)
    min_sine = math.sin(math.radians(float(min_angle_deg)))
    forced = 0

    for gi in multi:
        left, right = int(cuts[gi]), int(cuts[gi+1])
        ids = order[left:right]
        gn = n[ids]
        for local_i, idx in enumerate(ids):
            ni = gn[local_i]
            peers = np.delete(gn, local_i, axis=0)
            if not len(peers):
                continue
            tangent = peers - (peers @ ni)[:, None] * ni[None, :]
            length = np.linalg.norm(tangent, axis=1)
            valid = length >= min_sine
            if not np.any(valid):
                continue
            # Prefer the best-conditioned incident face when more than two
            # surfaces meet at the same point.
            score = np.where(valid, length, -1.0)
            j = int(np.argmax(score))
            direction = -tangent[j] / max(float(length[j]), 1e-30)
            u[idx] = direction
            v[idx] = np.cross(ni, direction)
            v[idx] /= max(float(np.linalg.norm(v[idx])), 1e-30)
            split[idx] = 0.0
            if not active[idx]:
                forced += 1
            active[idx] = True

            # A third distinct clipping direction cannot be represented by one
            # half-plane.  Shrink the disk conservatively near such corners.
            dirs = -tangent[valid] / np.maximum(length[valid, None], 1e-30)
            if len(dirs) >= 2:
                d0 = dirs[j if j < len(dirs) else 0]
                distinct = np.sum(dirs * d0[None, :], axis=1) < math.cos(math.radians(25.0))
                if np.count_nonzero(distinct) >= 1:
                    sigma[idx] = min(float(sigma[idx]), max(float(h[idx]) * 0.10, 1e-8))

    return forced


def reference_half_plane_support(cloud: SplatCloud, material: AnalyticMaterial,
                                 sigma_scale: float,
                                 min_angle_deg: float = 5.0, *, _prepared=None):
    """Group wrapper matching reference half_plane_support() behavior."""
    points = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(cloud.normals, dtype=np.float64).reshape(-1, 3)
    sizes = np.asarray(cloud.support, dtype=np.float64).reshape(-1)
    count = len(points)
    sigma = np.empty(count, dtype=np.float64)
    u = np.empty((count, 3), dtype=np.float64)
    v = np.empty((count, 3), dtype=np.float64)
    split = np.empty(count, dtype=np.float64)
    active = np.zeros(count, dtype=bool)
    candidates = queries = groups = 0
    preparation=_prepared if _prepared is not None else _RenderPreparation(None)
    for ids,row in preparation.groups(cloud,material,sigma_scale,min_angle_deg):
        ss, uu, vv, sp, ac, cc, qq, _near, _second, _dir1, _dir2 = row
        sigma[ids], u[ids], v[ids], split[ids], active[ids] = ss, uu, vv, sp, ac
        candidates += cc
        queries += qq
        groups += 1
    # Primitive SDFs can expose only one subgradient at an exact rim.  The
    # surface extractor already knows coincident incident-face normals there,
    # so use that exact information as a final sharp-edge override.
    # Use the same numerical minimum-angle guard for exact coincident face-pair masks.
    if hasattr(cloud,'_local_groups'):
        forced=_local_force_masks(cloud,sigma,u,v,split,active,float(min_angle_deg))
    else:
        forced = _force_coincident_sharp_edge_masks(
            points, normals, sizes, sigma, u, v, split, active,
            min_angle_deg=float(min_angle_deg))

    # Display/debug counters; they do not participate in material state.
    material.last_half_candidate_sweeps = candidates
    material.last_half_material_queries = queries
    material.last_half_groups = groups
    material.last_half_sharp_forced = int(forced)
    return sigma, u, v, split, active


def reference_half_feature_metrics(cloud: SplatCloud, material: AnalyticMaterial,
                                   sigma_scale: float):
    """Return Half2DGS feature distances and the detected clipping-edge tangent.

    ``nearest`` and ``second`` reuse the exact geometric candidate/continuation
    logic used by Half2DGS.  ``edge_tangent`` is the local v-axis produced from
    the nearest valid clipping edge and is used only to merge nearby micro-grid
    candidate islands along the same physical edge.
    """
    points = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(cloud.normals, dtype=np.float64).reshape(-1, 3)
    sizes = np.asarray(cloud.support, dtype=np.float64).reshape(-1)
    nearest = np.full(len(points), np.inf, dtype=np.float64)
    second = np.full(len(points), np.inf, dtype=np.float64)
    edge_tangent = np.zeros((len(points), 3), dtype=np.float64)
    for ids in balanced_point_groups(points, 512):
        _ss, _u, vv, _sp, _ac, _cc, _qq, nn, ss, _dir1, _dir2 = _reference_half_plane_support_group(
            points[ids], normals[ids], sizes[ids], material, sigma_scale,
            min_intersection_angle_deg=5.0)
        nearest[ids] = nn
        second[ids] = ss
        edge_tangent[ids] = vv
    return nearest, second, edge_tangent


def reference_half_double_intersections(cloud: SplatCloud, material: AnalyticMaterial,
                                        sigma_scale: float,
                                        min_angle_deg: float = 5.0, *, _prepared=None):
    """Return two verified Half2DGS clipping intersections per splat when present.

    The same candidate discovery, tangent projection and continuation probe as
    normal Half2DGS are reused.  A parent is considered a double-intersection
    splat only when BOTH geometrically distinct edges lie inside the parent's
    nominal 2.5*sigma footprint.  Directions point along the Half2DGS U axis;
    an edge point is ``p - direction * distance``.
    """
    points=np.asarray(cloud.positions,dtype=np.float64).reshape(-1,3)
    normals=np.asarray(cloud.normals,dtype=np.float64).reshape(-1,3)
    sizes=np.asarray(cloud.support,dtype=np.float64).reshape(-1)
    count=len(points)
    nearest=np.full(count,np.inf,dtype=np.float64)
    second=np.full(count,np.inf,dtype=np.float64)
    dir1=np.zeros((count,3),dtype=np.float64)
    dir2=np.zeros((count,3),dtype=np.float64)
    preparation=_prepared if _prepared is not None else _RenderPreparation(None)
    for ids,row in preparation.groups(cloud,material,sigma_scale,min_angle_deg):
        (_ss,_u,_v,_sp,_ac,_cc,_qq,nn,ss,d1,d2)=row
        nearest[ids]=nn; second[ids]=ss
        dir1[ids]=d1; dir2[ids]=d2
    nominal_sigma=np.maximum(sizes*float(sigma_scale),1e-8)
    reach=2.5*nominal_sigma
    valid=(np.isfinite(nearest)&np.isfinite(second)&
           (nearest<reach)&(second<reach))
    # Safety: both stored directions must be non-degenerate and distinct.
    l1=np.linalg.norm(dir1,axis=1); l2=np.linalg.norm(dir2,axis=1)
    cos25=math.cos(math.radians(25.0))
    d12=np.sum(dir1*dir2,axis=1)
    valid &= (l1>0.5)&(l2>0.5)&(d12<cos25)
    return valid,nearest,second,dir1,dir2


def _split_double_intersection_render_arrays(cloud: SplatCloud, material: AnalyticMaterial,
                                             sigma_scale: float, min_angle_deg: float,
                                             offset_scale: float = 0.34,
                                             min_sigma_scale: float = 0.78, *, _prepared=None):
    """Replace a two-edge parent by one centre and two boundary render children.

    Child A is shifted toward edge A and clipped by edge A. Child B is shifted
    toward edge B and clipped by edge B.  The split axis is ``dir1-dir2``: for
    a 90-degree corner this moves the two centers in opposite diagonal
    directions so each child becomes preferentially associated with one edge.

    The children are render-only. They inherit tool/resolution/seam metadata from
    the parent and do not alter material truth or cutting ownership. This is the
    official fix for the confirmed two-intersection Half2DGS representation limit.
    """
    p=np.asarray(cloud.positions,dtype=np.float64).reshape(-1,3)
    n=np.asarray(cloud.normals,dtype=np.float64).reshape(-1,3)
    count=len(p)
    if count==0:
        empty=np.empty(0,dtype=np.int64)
        return cloud,empty,np.zeros(0,dtype=bool),np.empty((0,3)),np.empty((0,3)),np.empty(0),np.zeros(0,dtype=bool)

    two,d1dist,d2dist,d1,d2=reference_half_double_intersections(
        cloud,material,sigma_scale,min_angle_deg=min_angle_deg,_prepared=_prepared)
    split_ids=np.flatnonzero(two)
    keep_ids=np.flatnonzero(~two)
    if not len(split_ids):
        source=np.arange(count,dtype=np.int64)
        return cloud,source,np.zeros(count,dtype=bool),np.empty((0,3)),np.empty((0,3)),np.empty(0),two

    # Keep boundary pairs adjacent; append the original-position centre children.
    source=np.concatenate([keep_ids,np.repeat(split_ids,2),split_ids])
    child=np.zeros(len(source),dtype=bool)
    child[len(keep_ids):len(keep_ids)+2*len(split_ids)]=True
    pp=p[source].copy(); nn=n[source].copy()
    support=np.asarray(cloud.support,dtype=np.float64)[source].copy()
    tool_ids=np.asarray(cloud.tool_ids,dtype=np.int32)[source].copy()
    levels=np.asarray(cloud.refine_levels,dtype=np.int16)[source].copy()
    seam=np.asarray(cloud.seam_weight,dtype=np.float64)[source].copy()

    # Per-child prescribed clipping edge.  NaN for retained parents; they are
    # evaluated by the ordinary Half2DGS routine after expansion.
    forced_u=np.full((len(source),3),np.nan,dtype=np.float64)
    forced_dist=np.full(len(source),np.nan,dtype=np.float64)

    base_sigma=np.maximum(np.asarray(cloud.support,dtype=np.float64)*float(sigma_scale),1e-8)
    pos=len(keep_ids)
    for parent in split_ids:
        a=np.asarray(d1[parent],dtype=np.float64)
        b=np.asarray(d2[parent],dtype=np.float64)
        sep=a-b
        sep-=n[parent]*float(sep@n[parent])
        sl=float(np.linalg.norm(sep))
        if sl<1e-12:
            # Distinct-edge test should make this rare. Fall back to an edge
            # tangent of the first intersection.
            sep=np.cross(n[parent],a); sl=float(np.linalg.norm(sep))
        sep/=max(sl,1e-30)
        off=float(offset_scale)*float(base_sigma[parent])
        shift_a=-off*sep   # closer to edge 1
        shift_b=+off*sep   # closer to edge 2
        pp[pos]=p[parent]+shift_a
        pp[pos+1]=p[parent]+shift_b
        # Preserve geometry support metadata; render sigma is assigned later.
        forced_u[pos]=a; forced_u[pos+1]=b
        # Keep the signed distance: a shifted centre may cross its own edge.
        # Clamping to zero would move the clipping line along with that centre.
        forced_dist[pos]=float(d1dist[parent]+shift_a@a)
        forced_dist[pos+1]=float(d2dist[parent]+shift_b@b)
        pos+=2

    expanded=SplatCloud(pp,nn,support,tool_ids,levels,seam)
    if hasattr(cloud,'_local_groups'): expanded._local_groups=cloud._local_groups[source]
    return expanded,source,child,forced_u,pp,forced_dist,two


def resolution_debug_colors(refine_levels: np.ndarray) -> np.ndarray:
    """Color splats by *effective spatial resolution* used by the realtime demo.

    Realtime Direct SweepSurface samples can be intentionally coarser than the
    block base pitch (for example 3h on a smooth flat bottom or 2h on a smooth
    side wall). Resolution labels distinguish these support sizes; ``-2`` marks
    Direct samples whose resolution has not been classified.

    Realtime convention:
      -4 : Direct coarse ~3h
      -3 : Direct coarse ~2h
      -2 : Direct sample with unclassified resolution
      -1 : initial analytic stock
       0 : base h
       1 : h/2
       2 : h/4
      >=3: still finer
    """
    level = np.asarray(refine_levels, dtype=np.int16).reshape(-1)
    # Stock / unknown default: neutral gray.
    colors = np.tile([0.48, 0.50, 0.54], (len(level), 1))
    colors[level == -4] = [0.18, 0.38, 1.00]  # ~3h: blue
    colors[level == -3] = [0.10, 0.78, 1.00]  # ~2h: cyan
    colors[level == -2] = [0.42, 0.56, 0.76]  # Direct, unclassified resolution
    colors[level ==  0] = [0.18, 0.92, 0.32]  # h: green
    colors[level ==  1] = [1.00, 0.82, 0.12]  # h/2: yellow
    colors[level ==  2] = [1.00, 0.35, 0.10]  # h/4: orange-red
    colors[level >=  3] = [1.00, 0.08, 0.55]  # finer: magenta-red
    return colors


def _child_crosses_other_boundary(sigma, own_distance, u, v, other_inward,
                                  other_distance, guard_sigma: float = 3.5):
    """Does the retained circular footprint cross the other half-plane?

    Minimize B.r over the guard-radius disk after its own Half2DGS clip,
    r.U >= -own_distance. Ignore overlap already removed by that clip.
    Overlap between children on valid surface is intentionally allowed.
    """
    if not np.isfinite(guard_sigma) or guard_sigma <= 0:
        raise ValueError('guard_sigma must be finite and positive')
    radius=np.maximum(np.asarray(sigma,dtype=np.float64),0.0)*guard_sigma
    bu=np.sum(other_inward*u,axis=1)
    bv=np.sum(other_inward*v,axis=1)
    length=np.sqrt(bu*bu+bv*bv)
    xmin=-np.asarray(own_distance,dtype=np.float64)
    worst_x=-radius*bu/np.maximum(length,1e-30)
    minimum=-radius*length
    clipped=worst_x<xmin
    edge_x=np.clip(xmin,-radius,radius)
    edge_y=np.sqrt(np.maximum(radius*radius-edge_x*edge_x,0.0))
    minimum[clipped]=(bu*edge_x-np.abs(bv)*edge_y)[clipped]
    visible=xmin<=radius
    tolerance=1e-10*np.maximum(1.0,radius)
    return visible & (length>1e-10) & (other_distance+minimum < -tolerance)


def _limit_child_ellipse(sigma_u, sigma_v, u, v, other_inward, other_distance,
                         guard_sigma: float = 3.5):
    """Fit a Gaussian ellipse inside the other edge's local half-plane.

    For inward normal B the projected standard deviation is
    sqrt((B.U * sigma_u)**2 + (B.V * sigma_v)**2). Bound its guard_sigma
    contour by the signed edge distance. This limits Gaussian tails, not an
    exact second clipping mask. At 90 degrees only V needs to shrink.
    """
    if not np.isfinite(guard_sigma) or guard_sigma <= 0:
        raise ValueError('guard_sigma must be finite and positive')
    su=np.asarray(sigma_u,dtype=np.float64).copy()
    sv=np.asarray(sigma_v,dtype=np.float64).copy()
    bu=np.abs(np.sum(other_inward*u,axis=1))
    bv=np.abs(np.sum(other_inward*v,axis=1))
    budget=np.maximum(np.asarray(other_distance,dtype=np.float64),0.0)/guard_sigma
    along_u=bu>1e-10
    along_v=bv>1e-10
    # At oblique corners reserve half the squared projection budget for V.
    # Parallel edges need only the U cap; orthogonal edges leave U unchanged.
    allocation=np.where(along_v,1.0/math.sqrt(2.0),1.0)
    su[along_u]=np.minimum(su[along_u],
                          budget[along_u]*allocation[along_u]/bu[along_u])
    remaining=np.sqrt(np.maximum(budget*budget-(bu*su)**2,0.0))
    sv[along_v]=np.minimum(sv[along_v],remaining[along_v]/bv[along_v])
    # Positive scales are required by the renderer, including zero-clearance cases.
    return np.maximum(su,1e-8),np.maximum(sv,1e-8)


def _gaussian_shading(original_cloud,source_ids,n,settings):
    """Unlit palette only; shared lighting is applied by Open3D at draw time."""
    count=len(n)
    if settings.resolution_color:
        base = resolution_debug_colors(original_cloud.refine_levels[source_ids])
    elif settings.tool_id_color:
        base = machining_tool_colors(original_cloud.tool_ids[source_ids])
    else:
        base = np.tile([0.56, 0.62, 0.70], (count, 1))

    return (base,)



class _RenderPreparation:
    """Transient shared work for two packs; never caches changing material state."""
    def __init__(self,cloud):
        self.cloud=cloud
        self.queries={}
        self.shading={}
        self.stats=dict(edge_shared_half_points=0,edge_shared_shading_points=0)

    def groups(self,cloud,material,sigma_scale,min_angle):
        key=(float(sigma_scale),float(min_angle))
        cached=self.queries.get(key) if cloud is self.cloud else None
        if cached is not None:
            self.stats['edge_shared_half_points']+=len(cloud.positions)
            yield from cached
            return
        result=[]
        groups=(range(start,min(start+512,len(cloud.positions))) for start in range(0,len(cloud.positions),512)) if hasattr(cloud,'_local_groups') else balanced_point_groups(cloud.positions,512)
        for ids in groups:
            ids=np.asarray(ids,dtype=np.int64)
            row=_reference_half_plane_support_group(
                cloud.positions[ids],cloud.normals[ids],cloud.support[ids],
                material,sigma_scale,min_intersection_angle_deg=float(min_angle))
            if cloud is self.cloud: result.append((ids,row))
            yield ids,row
        if cloud is self.cloud: self.queries[key]=result

    def colors(self,source_ids,n,settings):
        key=tuple(getattr(settings,name) for name in _EDGE_APPEARANCE_FIELDS)
        cached=self.shading.get(key)
        if cached is None:
            cached=_gaussian_shading(self.cloud,np.arange(len(self.cloud.positions)),
                                     self.cloud.normals,settings)
            self.shading[key]=cached
        else:
            self.stats['edge_shared_shading_points']+=len(source_ids)
        out=[value[source_ids].copy() for value in cached]
        # SplatCloud may normalize expanded child normals again. Respect the
        # actual normals, including rounding, instead of assuming equality.
        changed=np.flatnonzero(np.any(n!=self.cloud.normals[source_ids],axis=1))
        if len(changed):
            fixed=_gaussian_shading(self.cloud,source_ids[changed],n[changed],settings)
            for value,patch in zip(out,fixed): value[changed]=patch
        return out


def _pack_gaussian_attributes_legacy(cloud: SplatCloud, material: AnalyticMaterial,
                             settings: RenderSettings = RenderSettings(), *, _prepared=None):
    """Pack 2DGS render attributes with the official two-edge Half2DGS rule.

    A two-edge splat produces one reduced circular centre and two boundary children.
    Each boundary child owns one Half2DGS clip and becomes elliptical only when
    its retained circular footprint would cross the other physical boundary.
    """
    original_cloud = cloud
    original_count = len(cloud.positions)
    if settings.sigma_scale <= 0 or not (0 < settings.alpha < 1):
        raise ValueError("Require sigma scale > 0 and 0 < alpha < 1")
    if min(settings.ambient, settings.diffuse, settings.specular) < 0:
        raise ValueError("Lighting strengths must be non-negative")

    # One parent with two distinct clipping edges becomes three render children.
    if settings.half_clip and settings.half_double_edge_split and original_count:
        (render_cloud, source_ids, double_child, forced_u, _child_pos,
         forced_dist, double_parent_mask) = _split_double_intersection_render_arrays(
            original_cloud, material, settings.sigma_scale,
            settings.half_min_angle_deg,
            offset_scale=settings.half_double_edge_offset_scale,
            min_sigma_scale=settings.half_double_edge_min_sigma_scale,_prepared=_prepared)
    else:
        render_cloud = original_cloud
        source_ids = np.arange(original_count, dtype=np.int64)
        double_child = np.zeros(original_count, dtype=bool)
        forced_u = np.full((original_count, 3), np.nan, dtype=np.float64)
        forced_dist = np.full(original_count, np.nan, dtype=np.float64)
        double_parent_mask = np.zeros(original_count, dtype=bool)

    cloud = render_cloud
    p = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(cloud.normals, dtype=np.float64).reshape(-1, 3)
    count = len(p)
    center_child=double_parent_mask[source_ids] & ~double_child

    if settings.half_clip and count:
        sigma, u, v, split, half = reference_half_plane_support(
            cloud, material, settings.sigma_scale,
            min_angle_deg=settings.half_min_angle_deg,_prepared=_prepared)

        # The two children must remain one-edge representations.  Re-apply the
        # edge detected on the original parent after moving the child center.
        ids = np.flatnonzero(double_child)
        if len(ids):
            base_sigma = np.maximum(
                np.asarray(cloud.support, dtype=np.float64) * settings.sigma_scale, 1e-8)
            sigma_child = np.maximum(
                base_sigma[ids] * float(settings.half_double_edge_min_sigma_scale),
                forced_dist[ids] / 2.40)
            sigma[ids] = sigma_child
            uu = forced_u[ids].copy()
            uu -= np.sum(uu * n[ids], axis=1)[:, None] * n[ids]
            uu /= np.maximum(np.linalg.norm(uu, axis=1)[:, None], 1e-30)
            u[ids] = uu
            v[ids] = np.cross(n[ids], u[ids])
            half[ids] = True
            split[ids] = -forced_dist[ids] / np.maximum(sigma[ids], 1e-30)
    else:
        u, v = tangent_frames(n) if count else (np.empty((0,3)), np.empty((0,3)))
        sigma = cloud.support * settings.sigma_scale
        split = np.zeros(count)
        half = np.zeros(count, dtype=bool)

    # Existing zero-overlap resolution transition treatment is render-only.
    if count and cloud.seam_weight is not None:
        seam_scale = max(float(settings.resolution_seam_scale), 1.0)
        seam_factor = 1.0 + np.clip(cloud.seam_weight, 0.0, 1.0) * (seam_scale - 1.0)
        sigma *= seam_factor
        split /= seam_factor

    sigma_u=sigma.copy()
    sigma_v=sigma.copy()
    interference=np.zeros(count,dtype=bool)
    elliptical_child=np.zeros(count,dtype=bool)
    center_valid=np.zeros(count,dtype=bool)
    if np.any(double_child):
        ids=np.flatnonzero(double_child)
        # Split children are emitted in adjacent A/B pairs. Each peer carries
        # the other clipping plane. Transfer its signed distance to this centre.
        peer=ids.reshape(-1,2)[:,::-1].reshape(-1)
        other_u=forced_u[peer]
        other_distance=(forced_dist[peer]
                        + np.sum((p[ids]-p[peer])*other_u,axis=1))
        guard=float(settings.half_double_edge_guard_sigma)
        interference[ids]=_child_crosses_other_boundary(
            sigma_u[ids],forced_dist[ids],u[ids],v[ids],other_u,other_distance,guard)
        if settings.half_double_edge_elliptical:
            hit=interference[ids]
            selected=ids[hit]
            sigma_u[selected],sigma_v[selected]=_limit_child_ellipse(
                sigma_u[selected],sigma_v[selected],u[selected],v[selected],
                other_u[hit],other_distance[hit],guard_sigma=guard)
            elliptical_child[selected]=True
        # Half2DGS splits are expressed in standard deviations along U, not V.
        split[ids]=-forced_dist[ids]/sigma_u[ids]

        centers=np.flatnonzero(center_child)
        pairs=ids.reshape(-1,2)
        delta=p[centers,None,:]-p[pairs]
        distances=forced_dist[pairs]+np.sum(delta*forced_u[pairs],axis=2)
        clearance=np.min(distances,axis=1)
        factor=float(settings.half_double_edge_center_sigma_scale)
        if not np.isfinite(factor) or not 0.0<factor<1.0:
            raise ValueError('centre sigma scale must be between zero and one')
        center_sigma=np.minimum(cloud.support[centers]*settings.sigma_scale*factor,
                                np.maximum(clearance,0.0)/guard)
        center_valid[centers]=center_sigma>1e-8
        sigma_u[centers]=np.maximum(center_sigma,1e-8)
        sigma_v[centers]=sigma_u[centers]
        u[centers],v[centers]=tangent_frames(n[centers])
        half[centers]=False
        split[centers]=0.0

    frames = np.stack([u, v, n], axis=2) if count else np.empty((0,3,3))
    rot = rotation_matrices_to_quaternions(frames) if count else np.empty((0,4))
    if _prepared is None:
        (base,)=_gaussian_shading(original_cloud,source_ids,n,settings)
    else:
        (base,)=_prepared.colors(source_ids,n,settings)

    opacity = np.full(
        (count, 1), math.log(settings.alpha/(1.0-settings.alpha)), dtype=np.float64)
    if np.any(double_child):
        child_alpha = float(np.clip(
            settings.alpha * settings.half_double_edge_opacity_scale, 1e-4, 0.9999))
        opacity[double_child] = math.log(child_alpha/(1.0-child_alpha))
        # Fill the original centre without simply adding another full-opacity
        # layer. Match parent alpha at that point under alpha compositing.
        centers=np.flatnonzero(center_child)
        pairs=np.flatnonzero(double_child).reshape(-1,2)
        delta=p[centers,None,:]-p[pairs]
        x=np.sum(delta*u[pairs],axis=2)/sigma_u[pairs]
        y=np.sum(delta*v[pairs],axis=2)/sigma_v[pairs]
        visible=x>=split[pairs]
        a=child_alpha*np.exp(-0.5*(x*x+y*y))*visible
        transmittance=np.prod(1.0-a,axis=1)
        center_alpha=np.clip(1.0-(1.0-settings.alpha)/np.maximum(transmittance,1e-12),
                             1e-6,settings.alpha)
        center_alpha[~center_valid[centers]]=1e-6
        opacity[centers,0]=np.log(center_alpha/(1.0-center_alpha))

    result = {
        "positions": p,
        "scale": np.column_stack([sigma_u, sigma_v, np.full(count, 0.01)]),
        "rot": rot,
        "coverage_logit": opacity,
        "boundary_split": split[:, None],
        "surface_color": base,
        "colors": base,
        "half_active": half,
        "half_double_child": double_child,
        "split_center_child": center_child,
        "split_center_valid": center_valid,
        "split_boundary_interference": interference,
        "split_elliptical_child": elliptical_child,
        # Parent mask belongs to the original geometric cloud and is debug/count metadata.
        "half_double_parent": double_parent_mask,
        "source_ids": source_ids,
    }
    for key, value in list(result.items()):
        if key in ("source_ids", "half_active", "half_double_child", "half_double_parent",
                   "split_center_child", "split_center_valid",
                   "split_boundary_interference", "split_elliptical_child"):
            continue
        result[key] = np.ascontiguousarray(value, dtype=np.float32)
    return result


EDGE_CASE_NAMES = ('none', 'single', 'convex', 'concave', 'band', 'independent', 'ambiguous')


def _edge_split_status(material):
    s=getattr(material,'last_edge_split_stats',{})
    if s.get('clip_native'):
        return (f"Native clip: 0 edges {s.get('clip_zero',0)} | 1 edge {s.get('clip_one',0)} | 2 edges {s.get('clip_two',0)}\n"
                f"Ridge selection: {s.get('clip_approximated',0)} | candidates >2 {s.get('clip_candidates_over_two',0)} | omitted edges {s.get('clip_omitted_edges',0)} | max omitted mass {100*s.get('clip_omitted_mass_max',0):.2f}%")
    return (f"convex {s.get('convex',0)} | concave {s.get('concave',0)} | band {s.get('band',0)} | independent {s.get('independent',0)}\n"
            f"Unclassified {s.get('ambiguous',0)} | coverage warnings {s.get('coverage_unresolved',0)} | prior kept {s.get('regression_rejected',0)}\n"
            f"Edge plans: reused {s.get('edge_reused',0)} | recomputed {s.get('edge_recomputed',0)}")


def _nearest_distinct_edge_pairs(admissible,dist,direction,support,missing):
    """Batch the original first/second-boundary selection, including tie rules."""
    possible=missing[np.count_nonzero(admissible[missing],axis=1)>=2]
    if not len(possible): return possible,np.empty((0,2),dtype=np.int64)
    rank=np.argsort(np.where(admissible[possible],np.abs(dist[possible]),np.inf),axis=1)
    first=rank[:,0]
    ordered_dir=direction[possible[:,None],rank]
    first_dir=direction[possible,first]
    dot=np.sum(ordered_dir*first_dir[:,None,:],axis=2)
    cos2=math.cos(math.radians(2.0))
    # Preserve the old scalar np.dot rounding at the strict threshold.
    for row,col in np.argwhere(np.abs(dot-cos2)<1e-12):
        dot[row,col]=np.dot(ordered_dir[row,col],first_dir[row])
    ordered_dist=dist[possible[:,None],rank]
    duplicate=(dot>cos2)&(np.abs(ordered_dist-dist[possible,first,None])<
                         np.maximum(.02*support[possible],1e-7)[:,None])
    distinct=admissible[possible[:,None],rank]&~duplicate
    distinct[:,0]=False
    found=np.any(distinct,axis=1)
    second=rank[np.arange(len(possible)),np.argmax(distinct,axis=1)]
    return possible[found],np.column_stack([first[found],second[found]])


class EdgeCandidateCache:
    """One bounded snapshot of per-parent candidate results, including negatives.

    The key includes the EXACT ordered list of sweeps admitted by the current
    balanced group. A distant primitive can extrapolate a plane into a group;
    a small per-parent AABB alone cannot capture all candidate dependencies.
    """
    def __init__(self,max_bytes=32*1024*1024):
        self.max_bytes=max_bytes
        self.clear()

    def clear(self):
        self.configuration=None
        self.keys=np.empty(0,dtype='V80')
        self.values=None
        self.byte_count=0

    def begin(self,cloud,material,sigma_scale):
        if (type(material) is not AnalyticMaterial or
                (len(material.sweeps) and not material._sweep_index.supported[-1])):
            self.clear()
            return False
        configuration=(material.stock_lo.tobytes(),material.stock_hi.tobytes(),float(sigma_scale))
        if configuration!=self.configuration:
            self.clear(); self.configuration=configuration
        count=len(cloud.positions)
        self.current=np.empty(count,dtype=[('point','V56'),('sweeps','V24')])
        raw=np.ascontiguousarray(np.column_stack([cloud.positions,cloud.normals,cloud.support]),dtype=np.float64)
        self.current['point']=raw.view('V56').ravel()
        self.tokens=material._sweep_index.tokens
        self.stats=dict(edge_candidate_points=count,edge_candidate_reused=0,
                        edge_candidate_recomputed=0,edge_candidate_groups_reused=0)
        return True

    def restore(self,ids,sweep_ids,valid,distances,directions):
        signature=hashlib.blake2b(b''.join(self.tokens[i] for i in sweep_ids),digest_size=24).digest()
        self.current['sweeps'][ids]=np.void(signature)
        query=self.current.view('V80').ravel()[ids]
        hit=np.zeros(len(ids),dtype=bool)
        if len(self.keys):
            at=np.searchsorted(self.keys,query)
            at=np.minimum(at,len(self.keys)-1)
            hit=self.keys[at]==query
            selected=ids[hit]; previous=at[hit]
            valid[selected]=self.values[0][previous]
            distances[selected]=self.values[1][previous]
            directions[selected]=self.values[2][previous]
        self.stats['edge_candidate_reused']+=int(np.count_nonzero(hit))
        self.stats['edge_candidate_recomputed']+=int(np.count_nonzero(~hit))
        self.stats['edge_candidate_groups_reused']+=int(np.all(hit))
        return np.flatnonzero(~hit)

    def finish(self,valid,distances,directions):
        # Memory bound includes owned keys and the three result arrays.
        size=self.current.nbytes+valid.nbytes+distances.nbytes+directions.nbytes
        if size<=self.max_bytes:
            keys=self.current.view('V80').ravel()
            order=np.argsort(keys,kind='stable')
            self.keys=keys[order].copy()
            self.values=tuple(x[order].copy() for x in (valid,distances,directions))
            self.byte_count=size
        self.stats['edge_candidate_cache_bytes']=self.byte_count
        del self.current


def _edge_pair_candidates(cloud, material, sigma_scale, *, candidate_cache=None):
    """Discover signed tangent constraints without discarding concave edges.

    A primitive plane remains a candidate until the sampled local face patch
    confirms or rejects it.
    """
    p=cloud.positions; n=cloud.normals; count=len(p)
    distances=np.full((count,2),np.inf); directions=np.zeros((count,2,3))
    valid=np.zeros(count,dtype=bool)
    enabled=candidate_cache is not None and candidate_cache.begin(cloud,material,sigma_scale)
    material.last_edge_candidate_stats={}
    local=hasattr(cloud,'_local_groups')
    groups=(range(start,min(start+256,count)) for start in range(0,count,256)) if local else balanced_point_groups(p,256)
    for ids in groups:
        ids=np.asarray(ids,dtype=np.int64)
        pp=p[ids]; nn=n[ids]; reach=2.5*cloud.support[ids]*sigma_scale
        group_lo=pp.min(axis=0)-reach.max(); group_hi=pp.max(axis=0)+reach.max()
        sweep_ids=material.query_sweep_ids(group_lo,group_hi)
        missing=(candidate_cache.restore(ids,sweep_ids,valid,distances,directions)
                 if enabled else np.arange(len(ids)))
        if not len(missing): continue
        # Keep the original group's primitive list, but query only changed rows.
        # This avoids repeating gradient and four-direction probes for cache hits.
        ids=ids[missing]; pp=pp[missing]; nn=nn[missing]; reach=reach[missing]
        missing=np.arange(len(ids))
        values=[]; gradients=[]
        for axis in range(3):
            g=np.eye(3)[axis]
            values.extend([pp[:,axis]-material.stock_hi[axis],material.stock_lo[axis]-pp[:,axis]])
            gradients.extend([np.broadcast_to(g,pp.shape),np.broadcast_to(-g,pp.shape)])
        for index in sweep_ids:
            sw=material.sweeps[index]
            d,g=sw.sdf_gradient(pp)
            lo,hi=sw.world_aabb()
            member=(np.all(pp+reach[:,None]>=lo,axis=1)&np.all(pp-reach[:,None]<=hi,axis=1)) if local else np.ones(len(pp),bool)
            values.append(np.where(member,-d,np.inf)); gradients.append(-g)
            # A single cutter can contribute both sides of a concave notch.
            # Probe across this footprint to discover sharp alternate faces;
            # one SDF subgradient at the centre cannot enumerate those faces.
            tu,tv=tangent_frames(nn)
            for axis in (tu,-tu,tv,-tv):
                qq=pp+reach[:,None]*axis
                dd,gg=sw.sdf_gradient(qq)
                sharp=np.sum(g*gg,axis=1)<math.cos(math.radians(25.0))
                plane=-dd+np.sum((-gg)*(pp-qq),axis=1)
                values.append(np.where(sharp&member,plane,np.inf)); gradients.append(-gg)
        val=np.stack(values,axis=1); g=np.stack(gradients,axis=1)
        tangent=g-np.sum(g*nn[:,None,:],axis=2)[:,:,None]*nn[:,None,:]
        length=np.linalg.norm(tangent,axis=2)
        direction=-tangent/np.maximum(length[:,:,None],1e-30)
        dist=-val/np.maximum(length,1e-30)
        admissible=(length>=math.sin(math.radians(5.0))) & (np.abs(dist)<reach[:,None])
        selected,chosen=_nearest_distinct_edge_pairs(admissible,dist,direction,cloud.support[ids],missing)
        targets=ids[selected]
        distances[targets]=dist[selected[:,None],chosen]
        directions[targets]=direction[selected[:,None],chosen]
        valid[targets]=True
    if enabled:
        candidate_cache.finish(valid,distances,directions)
        material.last_edge_candidate_stats=candidate_cache.stats.copy()
    else:
        material.last_edge_candidate_stats=dict(edge_candidate_points=count,
            edge_candidate_reused=0,edge_candidate_recomputed=count,edge_candidate_cache_bytes=0)
    return valid,distances,directions


def _sample_edge_patch(p,n,h,sigma,dist,directions,material,grid_size=25):
    """A temporary tangent grid with extra probes in the four corner sectors."""
    tu,tv=tangent_frames(n[None,:]); tu=tu[0]; tv=tv[0]
    axis=np.linspace(-2.5,2.5,grid_size)*sigma
    xx,yy=np.meshgrid(axis,axis,indexing='ij')
    uv=np.column_stack([xx.ravel(),yy.ravel()])
    uv=uv[np.linalg.norm(uv,axis=1)<=2.5*sigma+1e-12]
    A=np.column_stack([directions@tu,directions@tv])
    intersection=None
    if abs(np.linalg.det(A))>.02:
        intersection=np.linalg.solve(A,-dist)
        if np.linalg.norm(intersection)<3.0*sigma:
            # Geometric spacing resolves the rapidly shrinking admissible
            # footprint near a vertex; a few isolated sector probes miss gaps.
            levels=np.array([-.58,-.34,-.20,-.12,-.07,-.04,.04,.07,.12,.20,.34,.58])
            ex,ey=np.meshgrid(levels,levels,indexing='ij')
            rhs=np.column_stack([ex.ravel(),ey.ravel()])*sigma-dist
            # Solve all 144 two-by-two systems together, preserving probe order.
            extras=np.linalg.solve(A,rhs.T).T
            extras=extras[np.linalg.norm(extras,axis=1)<3.0*sigma]
            uv=np.vstack([uv,extras])
    uv=np.unique(np.round(uv/max(sigma,1e-12),10),axis=0)*sigma
    q=p+uv[:,0,None]*tu+uv[:,1,None]*tv
    eps=max(.04*h,1e-6)
    values=material.value(np.vstack([q-n*eps,q+n*eps]))
    fi,fo=np.split(values,2)
    face=(fi<-.08*eps)&(fo>.08*eps)
    known=face | (fi>.08*eps) | (fo<-.08*eps)
    signed=(q-p)@directions.T+dist
    # Samples on the actual clipping lines have ambiguous finite-difference signs.
    known &= np.all(np.abs(signed)>.1*eps,axis=1)
    return q,face,known,signed,intersection


def _classify_edge_patch(p,n,h,sigma,dist,directions,material):
    q,face,known,signed,intersection=_sample_edge_patch(p,n,h,sigma,dist,directions,material)
    if np.count_nonzero(known)<12 or not np.any(face&known):
        return 6,'unknown',q,face,known
    masks=signed>=0
    models=[masks[:,0],masks[:,1],masks.all(axis=1),masks.any(axis=1)]
    errors=np.array([np.mean(pred[known]!=face[known]) for pred in models])
    best=int(np.argmin(errors))
    if errors[best]>.035: return 6,'unknown',q,face,known
    if best<2: return 1,str(best),q,face,known
    operator='and' if best==2 else 'or'
    dot=float(np.dot(directions[0],directions[1]))
    if operator=='and' and dot < -math.cos(math.radians(10.0)):
        case=4
    elif intersection is None or np.linalg.norm(intersection)>2.5*sigma:
        case=5
    else:
        case=2 if operator=='and' else 3
    return case,operator,q,face,known


def _render_axes(rot):
    w,x,y,z=np.asarray(rot,dtype=float).T
    u=np.column_stack([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)])
    v=np.column_stack([2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)])
    return u,v


def _patch_kernels(q,patches):
    if not patches: return np.empty((len(q),0))
    centers=np.array([c['p'] for c in patches]); u=np.array([c['u'] for c in patches])
    v=np.array([c['v'] for c in patches]); size=np.array([c['size'] for c in patches])
    r=q[:,None,:]-centers[None,:,:]
    x=np.sum(r*u[None,:,:],axis=2)/size[None,:,0]
    y=np.sum(r*v[None,:,:],axis=2)/size[None,:,1]
    kernel=np.exp(-.5*(x*x+y*y))
    for j,c in enumerate(patches):
        if c['half']: kernel[:,j]*=x[:,j]>=c['split']
    return kernel


def _fit_patch_opacity(kernel,target,weight,alpha=None,passes=6):
    """Coordinate descent on alpha compositing, not a sum of Gaussian weights."""
    count=kernel.shape[1]
    a=np.zeros(count) if alpha is None else np.asarray(alpha,dtype=float).copy()
    # Coordinate descent visits columns; contiguous columns avoid strided reads.
    columns=np.ascontiguousarray(kernel.T)
    for _ in range(passes):
        trans=np.prod(1-kernel*a[None,:],axis=1)
        for j in range(count):
            column=columns[j]
            others=trans/np.maximum(1-column*a[j],1e-9)
            basis=others*column
            residual=target-(1-others)
            weighted_basis=weight*basis
            value=np.sum(weighted_basis*residual)/max(np.sum(weighted_basis*basis),1e-20)
            a[j]=min(max(value,0.),.999)
            trans=others*(1-column*a[j])
    return a,1-np.prod(1-kernel*a[None,:],axis=1)


def _fit_edge_children(p,n,h,sigma,dist,directions,case,operator,q,face,known,
                       settings,neighbor_transmittance=None):
    """Fit the surviving parent footprint, adding children only at uncovered samples."""
    guard=float(settings.half_double_edge_guard_sigma)
    tu,tv=tangent_frames(n[None,:]); tu=tu[0]; tv=tv[0]
    relative=q-p
    target=settings.alpha*np.exp(-.5*np.sum(relative*relative,axis=1)/(sigma*sigma))*face
    weight=np.where(face,1.,5.)*known
    if neighbor_transmittance is not None:
        weight[face]*=np.maximum(neighbor_transmittance[face]**2,.05)
    patches=[]

    def add(center,preferred=None,role='fill'):
        d=dist+(center-p)@directions.T
        inside=d>=-1e-9
        if operator=='and' and not np.all(inside): return False
        if operator=='or' and not np.any(inside): return False
        if preferred is None:
            available=np.flatnonzero(inside)
            preferred=int(available[np.argmax(d[available])]) if operator=='or' else int(np.argmin(d))
        own=int(preferred); other=1-own
        u=directions[own]; v=np.cross(n,u); v/=max(np.linalg.norm(v),1e-30)
        size=np.array([sigma*.78,sigma*.78])
        if role=='fill':
            # A broad extra child can over-darken already covered samples while
            # its own peak stays too weak. Fit a more local correction instead.
            size[:]=sigma*.50
        half=True; split=-float(d[own])/size[0]
        interference=False; elliptical=False
        if role=='centre' and np.all(d>0):
            radius=min(.5*sigma,float(np.min(d))/guard)
            if radius < .015*sigma: return False
            size[:]=radius; u=tu; v=tv; half=False; split=0.
        elif operator=='and' and case!=5:
            interference=bool(_child_crosses_other_boundary(size[:1],d[own:own+1],u[None,:],v[None,:],
                                 directions[other:other+1],d[other:other+1],guard)[0])
            if interference and settings.half_double_edge_elliptical:
                if role=='fill': size[0]=min(size[0],max(.02*sigma,.9*max(d[own],0.)))
                su,sv=_limit_child_ellipse(size[:1],size[1:],u[None,:],v[None,:],
                                           directions[other:other+1],d[other:other+1],guard)
                size=np.array([su[0],sv[0]]); elliptical=True
            split=-float(d[own])/size[0]
        # Independent edges keep circular children and are checked for leakage
        # by the sampled fit/validation rather than forced corner narrowing.
        # A concave OR branch is already wholly valid after its own half clip;
        # the other boundary is not an exclusion constraint for that branch.
        if np.min(size)<.0002*sigma: return False
        patches.append(dict(p=np.asarray(center).copy(),u=u.copy(),v=v.copy(),size=size,
                            half=half,split=split,role=role,interference=interference,ellipse=elliptical))
        return True

    if operator=='or':
        for edge in range(2):
            center=p+directions[edge]*max(-dist[edge]+.1*sigma,0.)
            add(center,edge,'boundary')
    else:
        add(p,role='centre')
        for edge in range(2):
            center=p+.34*sigma*(directions[1-edge]-directions[edge])
            for _ in range(8):
                d=dist+(center-p)@directions.T
                for k in range(2):
                    if d[k]<.06*sigma: center+=(.06*sigma-d[k])*directions[k]
            add(center,edge,'boundary')
    if not patches: return [],dict(gap=1.,leak=1.,children=0)
    max_children = max(3, int(settings.half_double_edge_max_children))
    alpha = None
    kernel_buffer = np.empty(
        (len(q), max(max_children, len(patches))), dtype=float)
    kernel_count = 0
    for _ in range(max_children):
        if len(patches)>kernel_count:
            kernel_buffer[:,kernel_count:len(patches)]=_patch_kernels(q,patches[kernel_count:])
            kernel_count=len(patches)
        # Preserve the original reduction order, including symmetric ties in
        # the greedy placement step. Only copy values, not their exponentials.
        kernel=np.ascontiguousarray(kernel_buffer[:,:kernel_count])
        alpha,coverage=_fit_patch_opacity(kernel,target,weight,alpha)
        residual=target-coverage
        eligible=face&known&(target>.03)
        gap=float(np.max(residual[eligible],initial=0.))
        if gap<=.065 or len(patches)>= max_children: break
        order=np.argsort(-np.where(eligible,residual,-np.inf))
        added=False
        for index in order[:16]:
            if residual[index]<=.065: break
            coincident=[c for c in patches if np.linalg.norm(q[index]-c['p'])<.015*sigma]
            if add(q[index]):
                if coincident:
                    patches[-1]['size']=np.minimum(patches[-1]['size'],.5*np.min([c['size'] for c in coincident],axis=0))
                    if np.min(patches[-1]['size'])<.0002*sigma:
                        patches.pop(); continue
                    # Recompute clip location after changing the corrective width.
                    own_distance=float(np.dot(q[index]-p,patches[-1]['u'])+dist[np.argmax(directions@patches[-1]['u'])])
                    patches[-1]['split']=-own_distance/patches[-1]['size'][0]
                added=True; break
        if not added: break
        alpha=np.r_[alpha,0.]
    for c,a in zip(patches,alpha): c['alpha']=float(a)
    keep=[c for c in patches if c['alpha']>1e-4]
    leak=float(np.max(coverage[known&~face],initial=0.))
    return keep,dict(gap=gap,leak=leak,children=len(keep))


_EDGE_APPEARANCE_FIELDS = ('ambient', 'diffuse', 'specular', 'tool_id_color',
                           'resolution_color', 'resolution_base_h')


def _edge_stats_empty():
    stats = {name: 0 for name in EDGE_CASE_NAMES}
    stats.update(max_sample_gap=0., max_sample_leak=0., max_validation_gap=0.,
                 max_validation_leak=0., coverage_unresolved=0, extra_children=0,
                 regression_rejected=0, off_surface_rejected=0)
    return stats


def _edge_stats_merge(total, part):
    for key, value in part.items():
        if key.startswith('max_'):
            total[key] = max(total.get(key, 0.), value)
        else:
            total[key] = total.get(key, 0) + value


def _segment_rectangle_distance_sq_scalar(ax,ay,bx,by,x0,y0,x1,y1):
    """Small-query equivalent without allocating per-sweep NumPy arrays."""
    dx=bx-ax; dy=by-ay
    enter=0.; leave=1.
    for start,move,low,high in ((ax,dx,x0,x1),(ay,dy,y0,y1)):
        if move==0.:
            if start<low or start>high:
                enter=2.; leave=1.; break
        else:
            t0=(low-start)/move; t1=(high-start)/move
            enter=max(enter,min(t0,t1)); leave=min(leave,max(t0,t1))
    if enter<=leave: return 0.
    ga=max(x0-ax,ax-x1,0.); gb=max(y0-ay,ay-y1,0.)
    best=ga*ga+gb*gb
    ga=max(x0-bx,bx-x1,0.); gb=max(y0-by,by-y1,0.)
    best=min(best,ga*ga+gb*gb)
    den=dx*dx+dy*dy
    if den>0.:
        for x,y in ((x0,y0),(x0,y1),(x1,y0),(x1,y1)):
            ox=x-ax; oy=y-ay
            t=min(max((ox*dx+oy*dy)/den,0.),1.)
            rx=ox-t*dx; ry=oy-t*dy
            best=min(best,rx*rx+ry*ry)
    return best


def _segment_rectangle_distance_sq(a, b, lo, hi):
    """Exact XY segment/rectangle distance, vectorized over segments.

    Intersecting segments have distance zero. Otherwise the closest pair
    includes either a segment endpoint or a rectangle corner.
    """
    delta = b-a
    gap_a = np.maximum(np.maximum(lo-a, a-hi), 0.)
    gap_b = np.maximum(np.maximum(lo-b, b-hi), 0.)
    result = np.minimum(np.sum(gap_a*gap_a, axis=1), np.sum(gap_b*gap_b, axis=1))
    corners = np.array([[lo[0],lo[1]], [lo[0],hi[1]],
                        [hi[0],lo[1]], [hi[0],hi[1]]])
    den = np.sum(delta*delta, axis=1)
    offset = corners[None,:,:]-a[:,None,:]
    t = np.clip(np.sum(offset*delta[:,None,:], axis=2)/np.maximum(den[:,None],1e-300),0.,1.)
    residual = offset-t[:,:,None]*delta[:,None,:]
    result = np.minimum(result,np.min(np.sum(residual*residual,axis=2),axis=1))
    moving = delta != 0.
    safe_delta = np.where(moving,delta,1.)
    t0 = (lo-a)/safe_delta; t1 = (hi-a)/safe_delta
    low = np.where(moving,np.minimum(t0,t1),-np.inf)
    high = np.where(moving,np.maximum(t0,t1),np.inf)
    inside_static = np.all(moving | ((a>=lo)&(a<=hi)),axis=1)
    cross = inside_static & (np.maximum(np.max(low,axis=1),0.) <= np.minimum(np.min(high,axis=1),1.))
    return np.where(cross,0.,result)


def _edge_probe_bounds(p,n,h,sigma,attempted):
    """Enclose ALL probe grids and attempted children, with threshold margin.

    Material values are used here only for face/known booleans and the child
    surface tolerance. Keeping cutters outside this padded box preserves those
    decisions, not necessarily the exact numerical distance to distant cutters.
    """
    u,v = tangent_frames(n[None,:])
    eps = max(.04*h,1e-6)
    padding = max(2*eps,.02*h,1e-6)
    # All tangent probes lie in the radius-3*sigma disk. Account for normal
    # offsets, coordinate rounding and the acceptance tolerance of child centres.
    extent = (3*abs(sigma)*(1+1e-12)+1e-12)*np.sqrt(u[0]**2+v[0]**2)
    extent += np.abs(n)*eps+padding
    lo=p-extent; hi=p+extent
    if attempted is not None and len(attempted):
        lo=np.minimum(lo,attempted.min(axis=0)-padding)
        hi=np.maximum(hi,attempted.max(axis=0)+padding)
    return lo,hi


class EdgeParentCache:
    """Reuse expensive per-parent plans only when all local dependencies agree.

    Global Half-GS clipping and candidate discovery run: their grouping and
    extrapolated planes can change even for distant parents. Their *actual*
    outputs, the selected neighbours, and nearby cutter geometry are checked.
    """
    def __init__(self, max_bytes=64*1024*1024, max_entries=32768):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.byte_count = 0
        self.configuration = None

    def prepare(self, material, settings):
        if (type(material) is not AnalyticMaterial or
                (len(material.sweeps) and not material._sweep_index.supported[-1])):
            self.clear()
            return False
        configuration = (material.stock_lo.tobytes(), material.stock_hi.tobytes(),
                         tuple((f.name, getattr(settings, f.name)) for f in fields(settings)
                               if f.name not in _EDGE_APPEARANCE_FIELDS))
        if configuration != self.configuration:
            self.clear()
            self.configuration = configuration
        self.filter_stats = dict(edge_material_aabb_pairs=0, edge_material_shape_excluded=0,
                                 edge_reused_shape_filter=0)
        self.sweep_index=material._sweep_index
        self.sweep_shapes=self.sweep_index.shapes
        self.sweep_coordinate_scale=self.sweep_index.scales[-1] if len(material.sweeps) else 1.
        self.sweep_tokens=self.sweep_index.tokens
        return True

    def _material_candidates(self,lo,hi):
        magnitude=max(float(np.max(np.abs(lo))),float(np.max(np.abs(hi))),1.)
        tol=64*np.finfo(float).eps*magnitude
        ids=self.sweep_index.query_aabb(lo-tol,hi+tol,original_boxes=False)
        broad=hashlib.blake2b(b''.join(self.sweep_tokens[i] for i in ids),digest_size=24).digest()
        return ids,broad

    def _shape_material_key(self,lo,hi,ids):
        self.filter_stats['edge_material_aabb_pairs']+=len(ids)
        if not len(ids): return b''
        # Expand rather than shrink at tangency, including large coordinates.
        x0,y0,z0=map(float,lo); x1,y1,z1=map(float,hi)
        magnitude=max(abs(x0),abs(y0),abs(z0),abs(x1),abs(y1),abs(z1),
                      self.sweep_coordinate_scale,1.)
        tol=1.4210854715202004e-14*magnitude  # 64 * float64 epsilon
        x0-=tol; y0-=tol; z0-=tol; x1+=tol; y1+=tol; z1+=tol
        kept=[]
        for i in ids:
            ax,ay,bx,by,radius,bottom,top,ball=self.sweep_shapes[i]
            radial2=_segment_rectangle_distance_sq_scalar(ax,ay,bx,by,x0,y0,x1,y1)
            radius2=(radius+tol)**2
            hit=z1>=bottom and z0<=top and radial2<=radius2
            if not hit and ball:
                dz=max(z0-bottom,bottom-z1,0.)
                hit=radial2+dz*dz<=radius2
            if hit: kept.append(self.sweep_tokens[i])
        self.filter_stats['edge_material_shape_excluded']+=len(ids)-len(kept)
        return b''.join(kept)

    def material_key(self,lo,hi):
        ids,_=self._material_candidates(lo,hi)
        return self._shape_material_key(lo,hi,ids)

    @staticmethod
    def parent_key(cloud, parent):
        return b''.join(np.asarray(getattr(cloud, name)[parent]).tobytes()
                        for name in ('positions', 'normals', 'support',
                                     'tool_ids', 'refine_levels', 'seam_weight'))

    @staticmethod
    def dependency_key(cloud, parent, dist, directions, legacy, old_ids, base, neighbours):
        digest = hashlib.blake2b(digest_size=24)
        def add(array):
            array = np.ascontiguousarray(array)
            digest.update(repr((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        add(dist); add(directions)
        # These are exactly the attributes used by old-footprint validation.
        for name in ('positions', 'scale', 'rot', 'coverage_logit', 'half_active', 'boundary_split'):
            add(legacy[name][old_ids])
        # Order matters to compositing and to tie-breaking; retain it in the key.
        add(cloud.positions[neighbours])
        for name in ('scale', 'rot', 'coverage_logit', 'half_active', 'boundary_split'):
            add(base[name][neighbours])
        return digest.digest()

    def lookup(self, key, dependency):
        entry = self.entries.get(key)
        if entry is None:
            return None, 'new'
        if entry['dependency'] != dependency:
            return None, 'dependencies'
        ids,broad=self._material_candidates(entry['lo'],entry['hi'])
        if entry['broad'] != broad:
            if entry['material'] != self._shape_material_key(entry['lo'],entry['hi'],ids):
                return None,'material'
            self.filter_stats['edge_reused_shape_filter']+=1
            # Same precise dependencies; remember the current coarse snapshot.
            entry['broad']=broad
        self.entries.move_to_end(key)
        return entry, None

    def put(self, key, dependency, lo, hi, children, stats):
        ids,broad=self._material_candidates(lo,hi)
        material_key = self._shape_material_key(lo,hi,ids)
        # Conservative allowance for Python dictionaries, arrays and scalar objects.
        size = 4096 + len(key) + len(dependency) + len(material_key) + len(broad) + 2048*len(children or ())
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.byte_count -= previous['bytes']
        if size > self.max_bytes or self.max_entries <= 0:
            return
        while self.entries and (len(self.entries) >= self.max_entries or
                                self.byte_count + size > self.max_bytes):
            _, old = self.entries.popitem(last=False)
            self.byte_count -= old['bytes']
        if children is not None:
            # Detach views (notably single-edge directions) from whole-cloud arrays.
            children = [{name: value.copy() if isinstance(value, np.ndarray) else value
                         for name, value in child.items()} for child in children]
            for child in children:
                for value in child.values():
                    if isinstance(value, np.ndarray): value.flags.writeable = False
        self.entries[key] = dict(dependency=dependency, material=material_key, broad=broad,
                                 lo=lo.copy(), hi=hi.copy(), children=children,
                                 stats=stats.copy(), bytes=size)
        self.byte_count += size


class _EdgeNeighborIndex:
    """Uniform spatial buckets; no SciPy requirement, exact final filtering."""
    def __init__(self, cloud, candidate, sigma_scale):
        self.cloud = cloud
        self.sigma_scale = sigma_scale
        self.max_support = float(np.max(cloud.support, initial=0.))
        self.cell = max(5.0*self.max_support*sigma_scale, 1e-8)
        self.buckets = {}
        ids = np.flatnonzero(~candidate)
        coords = np.floor(cloud.positions[ids]/self.cell).astype(np.int64)
        for index, coord in zip(ids, coords):
            self.buckets.setdefault(tuple(coord), []).append(int(index))
        self.buckets = {key: np.asarray(value, dtype=np.int64)
                        for key, value in self.buckets.items()}

    def query(self, parent):
        cloud = self.cloud
        p = cloud.positions[parent]; n = cloud.normals[parent]
        h = float(cloud.support[parent]); sigma = h*self.sigma_scale
        radius = 2.5*(h+self.max_support)*self.sigma_scale
        radius += 1e-12*max(1.,radius)
        # Include boundary buckets; the strict original test is applied below.
        lo = np.floor((p-radius)/self.cell).astype(np.int64)
        hi = np.floor((p+radius)/self.cell).astype(np.int64)
        chunks = []
        for x in range(int(lo[0]), int(hi[0])+1):
            for y in range(int(lo[1]), int(hi[1])+1):
                for z in range(int(lo[2]), int(hi[2])+1):
                    ids = self.buckets.get((x,y,z))
                    if ids is not None: chunks.append(ids)
        if not chunks: return np.empty(0, dtype=np.int64)
        ids = np.sort(np.concatenate(chunks))
        delta = cloud.positions[ids]-p
        d2 = np.sum(delta*delta, axis=1)
        same = ((cloud.normals[ids]@n > math.cos(math.radians(10.))) &
                (np.abs(delta@n) < .025*np.minimum(h, cloud.support[ids])))
        near = same & (d2 < (2.5*sigma+2.5*cloud.support[ids]*self.sigma_scale)**2)
        ids = ids[near]; d2 = d2[near]
        if len(ids) > 8: ids = ids[np.argsort(d2)[:8]]
        return ids


def _evaluate_edge_parent(cloud,material,settings,parent,distances,directions,
                          legacy,base,bu,bv,old_ids,ids,phases=None):
    phases={} if phases is None else phases
    def mark(name,started):
        phases[name]=phases.get(name,0.)+(time.perf_counter()-started)*1000.
    started=time.perf_counter()
    stats=_edge_stats_empty()
    p=cloud.positions[parent]; n=cloud.normals[parent]; h=float(cloud.support[parent])
    sigma=h*settings.sigma_scale
    case,operator,q,face,known=_classify_edge_patch(p,n,h,sigma,distances[parent],directions[parent],material)
    mark('classify_ms',started)
    stats[EDGE_CASE_NAMES[case]]+=1
    if case==6: return None,stats,None
    if case==1:
        edge=int(operator); u=directions[parent,edge]; v=np.cross(n,u)
        children=[dict(p=p.copy(),u=u,v=v,size=np.array([sigma,sigma]),half=True,
            split=-distances[parent,edge]/sigma,alpha=settings.alpha,role='boundary',interference=False,ellipse=False)]
        return children,stats,None
    started=time.perf_counter()
    fq,fface,fknown,_,_=_sample_edge_patch(p,n,h,sigma,distances[parent],directions[parent],material,grid_size=40)
    q=np.vstack([q,fq]); face=np.r_[face,fface]; known=np.r_[known,fknown]
    mark('fit_sampling_ms',started)
    started=time.perf_counter()
    # Only geometrically compatible, unchanged neighbours influence seam fitting.
    trans=np.ones(len(q))
    if len(ids):
        r=q[:,None,:]-cloud.positions[ids]
        x=np.sum(r*bu[ids],axis=2)/base['scale'][ids,0]
        y=np.sum(r*bv[ids],axis=2)/base['scale'][ids,1]
        visible=(~base['half_active'][ids])[None,:] | (x>=base['boundary_split'][ids,0])
        alpha=1/(1+np.exp(-base['coverage_logit'][ids,0]))
        trans=np.prod(1-alpha*np.exp(-.5*(x*x+y*y))*visible,axis=1)
    mark('neighbor_coverage_ms',started)
    started=time.perf_counter()
    children,quality=_fit_edge_children(p,n,h,sigma,distances[parent],directions[parent],case,operator,
                                        q,face,known,settings,trans)
    mark('fit_ms',started)
    started=time.perf_counter()
    # A different grid phase/density checks between fitting samples. Report
    # failures rather than treating a finite fitting-grid pass as a proof.
    vq,vface,vknown,_,_=_sample_edge_patch(p,n,h,sigma,distances[parent],directions[parent],material,grid_size=47)
    kernel=_patch_kernels(vq,children)
    va=np.array([c['alpha'] for c in children])
    coverage=1-np.prod(1-kernel*va[None,:],axis=1)
    target=settings.alpha*np.exp(-.5*np.sum((vq-p)**2,axis=1)/(sigma*sigma))*vface
    validation_gap=float(np.max((target-coverage)[vface&vknown],initial=0.))
    validation_leak=float(np.max(coverage[~vface&vknown],initial=0.))
    old_u,old_v=_render_axes(legacy['rot'][old_ids])
    r=vq[:,None,:]-legacy['positions'][old_ids]
    x=np.sum(r*old_u,axis=2)/legacy['scale'][old_ids,0]
    y=np.sum(r*old_v,axis=2)/legacy['scale'][old_ids,1]
    visible=(~legacy['half_active'][old_ids])[None,:] | (x>=legacy['boundary_split'][old_ids,0])
    old_alpha=1/(1+np.exp(-legacy['coverage_logit'][old_ids,0]))
    old_coverage=1-np.prod(1-old_alpha*np.exp(-.5*(x*x+y*y))*visible,axis=1)
    old_gap=float(np.max((target-old_coverage)[vface&vknown],initial=0.))
    validation_weight=np.where(vface,1.,5.)*vknown
    old_error=np.sum(validation_weight*(old_coverage-target)**2)
    new_error=np.sum(validation_weight*(coverage-target)**2)
    accepted=(bool(children) and new_error<=old_error+1e-8
              and validation_gap<=max(old_gap+.03,.10))
    if children:
        residual=np.abs(material.value(np.array([c['p'] for c in children])))
        on_surface=np.all(np.isfinite(residual)) and np.all(residual<=max(.02*h,1e-6))
        if not on_surface:
            accepted=False
            stats['off_surface_rejected']+=1
    if not accepted: stats['regression_rejected']+=1
    stats['max_sample_gap']=max(stats['max_sample_gap'],quality['gap'])
    stats['max_sample_leak']=max(stats['max_sample_leak'],quality['leak'])
    stats['max_validation_gap']=max(stats['max_validation_gap'],validation_gap)
    stats['max_validation_leak']=max(stats['max_validation_leak'],validation_leak)
    stats['coverage_unresolved']+=int(validation_gap>.10 or validation_leak>.04)
    stats['extra_children']+=max(0,len(children)-3)

    mark('validation_ms',started)
    attempted=np.array([c['p'] for c in children]).reshape(-1,3)
    return children if accepted else None,stats,attempted


def pack_gaussian_attributes(cloud,material,settings=RenderSettings(),*,edge_cache=None):
    """Pack native two-plane clipping or Half-GS split attributes, per settings."""
    from cutsim.rendering.clipping import _native_clip_stats, _pack_native_clip, _resolve_pending_clip_draws
    if settings.native_two_clip:
        started=time.perf_counter(); out=_pack_native_clip(cloud,material,settings)
        _resolve_pending_clip_draws(out)
        _native_clip_stats(out,material)
        material.last_edge_phase_ms={"total_ms":1000*(time.perf_counter()-started)}
        return out
    phase_start=time.perf_counter()
    phases={name:0. for name in ('legacy_ms','base_ms','candidate_ms','setup_ms',
        'cache_check_ms','classify_ms','fit_sampling_ms','neighbor_coverage_ms',
        'fit_ms','validation_ms','cache_store_ms','assemble_ms')}
    material.last_edge_phase_ms=phases
    if not (settings.half_clip and settings.half_double_edge_split and len(cloud.positions)):
        out=_pack_gaussian_attributes_legacy(cloud,material,replace(settings,half_double_edge_split=False))
        material.last_edge_split_stats={}
        phases['legacy_ms']=(time.perf_counter()-phase_start)*1000.
        phases['total_ms']=phases['legacy_ms']
        phases['other_ms']=0.
        return out
    preparation=_RenderPreparation(cloud)
    started=time.perf_counter()
    legacy=_pack_gaussian_attributes_legacy(cloud,material,settings,_prepared=preparation)
    phases['legacy_ms']=(time.perf_counter()-started)*1000.
    started=time.perf_counter()
    base=_pack_gaussian_attributes_legacy(cloud,material,replace(settings,half_double_edge_split=False),_prepared=preparation)
    phases['base_ms']=(time.perf_counter()-started)*1000.
    started=time.perf_counter()
    if edge_cache is not None and not hasattr(edge_cache,'candidate_cache'):
        edge_cache.candidate_cache=EdgeCandidateCache()
    candidate,distances,directions=_edge_pair_candidates(cloud,material,settings.sigma_scale,
        candidate_cache=edge_cache.candidate_cache if edge_cache is not None else None)
    phases['candidate_ms']=(time.perf_counter()-started)*1000.
    started=time.perf_counter()
    count=len(cloud.positions); cases=np.zeros(count,dtype=np.int16)
    replacements={}; stats=_edge_stats_empty()
    stats.update(edge_candidates=int(np.count_nonzero(candidate)),edge_reused=0,edge_recomputed=0,
                 edge_miss_new=0,edge_miss_dependencies=0,edge_miss_material=0,edge_miss_disabled=0)
    enabled=edge_cache is not None and edge_cache.prepare(material,settings)
    neighbours=_EdgeNeighborIndex(cloud,candidate,settings.sigma_scale)
    order=np.argsort(legacy['source_ids'],kind='stable')
    starts=np.r_[0,np.cumsum(np.bincount(legacy['source_ids'],minlength=count))]
    bu,bv=_render_axes(base['rot'])
    phases['setup_ms']=(time.perf_counter()-started)*1000.
    for parent in np.flatnonzero(candidate):
        started=time.perf_counter()
        old_ids=order[starts[parent]:starts[parent+1]]
        ids=neighbours.query(parent)
        entry=None; reason='disabled'
        if enabled:
            key=edge_cache.parent_key(cloud,parent)
            dependency=edge_cache.dependency_key(cloud,parent,distances[parent],directions[parent],
                                                 legacy,old_ids,base,ids)
            entry,reason=edge_cache.lookup(key,dependency)
        phases['cache_check_ms']+=(time.perf_counter()-started)*1000.
        if entry is not None:
            children,part=entry['children'],entry['stats']
            stats['edge_reused']+=1
        else:
            children,part,attempted=_evaluate_edge_parent(cloud,material,settings,parent,
                distances,directions,legacy,base,bu,bv,old_ids,ids,phases)
            stats['edge_recomputed']+=1
            stats['edge_miss_'+reason]+=1
            if enabled:
                started=time.perf_counter()
                p=cloud.positions[parent]; n=cloud.normals[parent]; h=float(cloud.support[parent])
                lo,hi=_edge_probe_bounds(p,n,h,h*settings.sigma_scale,attempted)
                edge_cache.put(key,dependency,lo,hi,children,part)
                phases['cache_store_ms']+=(time.perf_counter()-started)*1000.
        _edge_stats_merge(stats,part)
        cases[parent]=next(i for i,name in enumerate(EDGE_CASE_NAMES) if part[name])
        if children is not None: replacements[parent]=children
    started=time.perf_counter()
    if enabled: stats.update(edge_cache.filter_stats)
    stats.update(material.last_edge_candidate_stats)
    stats.update(preparation.stats)
    stats['edge_cache_entries']=len(edge_cache.entries) if enabled else 0
    stats['edge_cache_bytes']=edge_cache.byte_count if enabled else 0
    material.last_edge_split_stats=stats
    keep=np.flatnonzero(~np.isin(legacy['source_ids'],list(replacements)))
    source=list(legacy['source_ids'][keep]); added_source=[]; plans=[]
    for parent,children in replacements.items():
        source.extend([parent]*len(children)); added_source.extend([parent]*len(children)); plans.extend(children)
    source=np.asarray(source,dtype=np.int64)
    added_source=np.asarray(added_source,dtype=np.int64)
    out={key:np.concatenate([legacy[key][keep],value[added_source]],axis=0)
         for key,value in base.items() if key!='half_double_parent'}
    out['source_ids']=source
    out['half_double_parent']=legacy['half_double_parent'].copy()
    out['half_double_parent'][list(replacements)]=True
    out['edge_case']=cases[source]
    start=len(keep)
    for offset,c in enumerate(plans):
        i=start+offset; parent=source[i]
        out['positions'][i]=c['p']; out['scale'][i,:2]=c['size']
        frame=np.column_stack([c['u'],c['v'],cloud.normals[parent]])
        out['rot'][i]=rotation_matrices_to_quaternions(frame[None,:,:])[0]
        a=float(np.clip(c['alpha'],1e-6,.999999)); opacity=math.log(a/(1-a))
        out['coverage_logit'][i]=opacity
        out['boundary_split'][i]=c['split']; out['half_active'][i]=c['half']
        out['half_double_child'][i]=c['role']!='centre'
        out['split_center_child'][i]=c['role']=='centre'; out['split_center_valid'][i]=c['role']=='centre'
        out['split_boundary_interference'][i]=c['interference']; out['split_elliptical_child'][i]=c['ellipse']
    phases['assemble_ms']=(time.perf_counter()-started)*1000.
    total=(time.perf_counter()-phase_start)*1000.
    phases['other_ms']=max(0.,total-sum(phases.values()))
    phases['total_ms']=total
    return out



def _local_force_masks(cloud,sigma,u,v,split,active,min_angle):
    groups=cloud._local_groups
    if len(groups)<2: return 0
    order=np.lexsort((groups[:,2],groups[:,1],groups[:,0]))
    sk=groups[order]
    cuts=np.r_[0,np.flatnonzero(np.any(sk[1:]!=sk[:-1],axis=1))+1,len(order)]
    forced=0
    for j in np.flatnonzero(np.diff(cuts)>1):
        ids=order[cuts[j]:cuts[j+1]]
        arrays=[x[ids].copy() for x in (sigma,u,v,split,active)]
        forced+=_force_coincident_sharp_edge_masks(cloud.positions[ids],cloud.normals[ids],
            cloud.support[ids],*arrays,min_angle_deg=min_angle)
        for dst,src in zip((sigma,u,v,split,active),arrays): dst[ids]=src
    return forced
