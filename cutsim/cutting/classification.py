"""GaussianCutSim cutting / classification implementation."""
from __future__ import annotations

import math
import numpy as np
import time
from cutsim.core.material import AnalyticMaterial
from cutsim.core.scipy_support import _HAVE_SCIPY_GRAPH, cKDTree
from cutsim.core.splat_cloud import SplatCloud
from cutsim.cutting.spatial_index import _RealtimeSplatIndex
from cutsim.geometry.frames import tangent_frames
from cutsim.geometry.sweep_strips import strip_material_value, strip_material_range
from cutsim.geometry.local_grid import LocalFrame, points_in_local_box, replacement_local_bounds


def local_patch_delete_mask(frame: LocalFrame, h: float, splats: SplatCloud) -> np.ndarray:
    """Replace every existing splat CENTER in the owned local patch.

    This deliberately includes uncut stock splats.  Therefore the unmachined
    top surface inside the local patch is refined to ``grid_cell`` together with
    the newly exposed cutter surface.  Splats outside the patch remain cached.
    """
    if not len(splats.positions):
        return np.zeros(0, dtype=bool)
    lo, hi = replacement_local_bounds(frame, h)
    return points_in_local_box(frame, splats.positions, lo, hi, eps=max(h*1e-6, 1e-9))


def sweep_intersects_splats(sweep, splats: SplatCloud, support_scale: float) -> np.ndarray:
    """Conservative delete test: Gaussian support sphere intersects newest sweep."""
    if not len(splats.positions):
        return np.zeros(0, dtype=bool)
    reach = splats.support * support_scale
    # SDF <= support radius means support can touch cutter volume.
    return sweep.sdf(splats.positions) <= reach


def classify_hybrid_sweep_candidates(old_material: AnalyticMaterial, cloud: SplatCloud,
                                     sigma_scale: float = 0.75,
                                     gaussian_reach_sigma: float = 2.5,
                                     margin_cells: float = 0.15,
                                     ring_samples: int = 8,
                                     early_exit: bool = True):
    """Prove easy cases using center distance, then evaluate the sampled ring.

    Box/flat/ball fields are 1-Lipschitz. Min/max Boolean composition preserves
    that bound even where the composed field is not an exact Euclidean SDF.
    Thus f(center)+reach < -margin proves DIRECT, and f(center)-reach > margin
    proves DISCARD. A roundoff guard sends borderline cases to the full ring test.
    Unknown material/primitive types also use the full ring test.

    For early exits min_old_sdf/max_old_sdf are NaN (the ring was not evaluated).
    sdf_extrema_exact identifies measured entries; old_sdf_lower/upper_bound
    expose conservative bounds instead of pretending they are sampled extrema.
    """
    count=len(cloud.positions)
    ring_samples=max(4,int(ring_samples))
    baseline=count*(1+ring_samples)
    certified=(type(old_material) is AnalyticMaterial and
               (not len(old_material.sweeps) or old_material._sweep_index.supported[-1]) and
               np.isfinite(margin_cells) and margin_cells>=0 and
               np.all(np.asarray(cloud.support)>=0))
    if not early_exit or not certified or not count:
        out=_classify_hybrid_sweep_candidates_full(
            old_material,cloud,sigma_scale,gaussian_reach_sigma,margin_cells,ring_samples)
        out.update(baseline_queries=baseline,early_direct=0,early_discard=0,
                   ring_candidates=count,center_queries=0,center_ms=0.0,
                   sdf_extrema_exact=np.ones(count,dtype=bool),
                   old_sdf_lower_bound=np.full(count,-np.inf),
                   old_sdf_upper_bound=np.full(count,np.inf))
        return out

    p=np.asarray(cloud.positions,dtype=np.float64)
    n=np.asarray(cloud.normals,dtype=np.float64)
    h=np.asarray(cloud.support,dtype=np.float64)
    start=time.perf_counter()
    center=strip_material_value(old_material, p)
    center_ms=(time.perf_counter()-start)*1000.0
    reach=np.abs(h*float(sigma_scale)*float(gaussian_reach_sigma))
    # tangent_frames normalizes U, while |V|=|normal|. Accommodate non-unit
    # input normals too; for normal unit inputs this is simply reach.
    radius=reach*np.maximum(1.0,np.linalg.norm(n,axis=1))
    margin=h*float(margin_cells)
    scale=1.0+np.max(np.abs(p),axis=1)+np.abs(center)+radius+np.abs(margin)
    guard=64.0*np.finfo(np.float64).eps*scale
    lower=center-radius-guard
    upper=center+radius+guard
    early_direct=upper < -margin
    early_discard=lower > margin
    uncertain=~(early_direct|early_discard)
    ids=np.flatnonzero(uncertain)
    direct=early_direct.copy(); discard=early_discard.copy()
    grid=np.zeros(count,dtype=bool)
    minimum=np.full(count,np.nan); maximum=np.full(count,np.nan)
    queries=count
    if len(ids):
        sub=SplatCloud(cloud.positions[ids],cloud.normals[ids],cloud.support[ids],
                       cloud.tool_ids[ids],cloud.refine_levels[ids])
        # Keep the original center+ring comparisons for ambiguous points, while
        # reusing the centers already evaluated above.
        checked=_classify_hybrid_sweep_candidates_full(
            old_material,sub,sigma_scale,gaussian_reach_sigma,margin_cells,ring_samples,
            center_values=center[ids])
        direct[ids]=checked['direct_mask']; grid[ids]=checked['grid_mask']
        discard[ids]=checked['discard_mask']
        minimum[ids]=checked['min_old_sdf']; maximum[ids]=checked['max_old_sdf']
        queries+=checked['footprint_queries']
    return dict(cloud=cloud,direct_mask=direct,grid_mask=grid,discard_mask=discard,
                center_old_sdf=center.copy(),min_old_sdf=minimum,max_old_sdf=maximum,
                footprint_queries=queries,
                baseline_queries=baseline,early_direct=int(np.count_nonzero(early_direct)),
                early_discard=int(np.count_nonzero(early_discard)),ring_candidates=len(ids),
                center_queries=count,center_ms=center_ms,sdf_extrema_exact=uncertain,
                old_sdf_lower_bound=lower,old_sdf_upper_bound=upper)


def _classify_hybrid_sweep_candidates_full(old_material: AnalyticMaterial, cloud: SplatCloud,
                                     sigma_scale: float = 0.75,
                                     gaussian_reach_sigma: float = 2.5,
                                     margin_cells: float = 0.15,
                                     ring_samples: int = 8,
                                     center_values=None):
    """Conservatively classify analytic sweep candidates.

    SAFE_DIRECT: every sampled point across the nominal Gaussian footprint lies
    comfortably inside the *old* material.  GRID_BAND: not provably safe, but at
    least part of the footprint touches/approaches old material.  DISCARD: the
    footprint is already outside old material, so this sweep face is not exposed.

    Importantly, the classifier does not try to solve the exact Boolean
    intersection curve.  It proves only the easy/safe region and sends ambiguity
    to the grid path.
    """
    count = len(cloud.positions)
    if not count:
        empty = np.zeros(0, dtype=bool)
        return {
            'cloud': cloud, 'direct_mask': empty, 'grid_mask': empty, 'discard_mask': empty,
            'center_old_sdf': np.empty(0), 'min_old_sdf': np.empty(0), 'max_old_sdf': np.empty(0),
            'footprint_queries': 0,
        }

    p = np.asarray(cloud.positions, dtype=np.float64)
    n = np.asarray(cloud.normals, dtype=np.float64)
    tu, tv = tangent_frames(n)
    ring_samples = max(4, int(ring_samples))
    angles = (np.arange(ring_samples, dtype=np.float64) *
              (2.0 * math.pi / ring_samples))
    reach = np.asarray(cloud.support, dtype=np.float64) * float(sigma_scale) * float(gaussian_reach_sigma)

    # Center + one outer ring.  This deliberately errs toward GRID_BAND when the
    # support approaches an old surface.  A later revision can add adaptive rings
    # only for ambiguous candidates if needed.
    samples = [p] if center_values is None else []
    for ang in angles:
        direction = math.cos(float(ang)) * tu + math.sin(float(ang)) * tv
        samples.append(p + reach[:, None] * direction)
    probe = np.stack(samples, axis=1)  # [N, 1+ring, 3]
    values = strip_material_value(old_material, probe.reshape(-1, 3)).reshape(count, -1)
    query_count=int(values.size)
    if center_values is not None:
        values=np.column_stack([np.asarray(center_values,dtype=np.float64),values])

    margin = np.asarray(cloud.support, dtype=np.float64) * float(margin_cells)
    safe_direct = np.all(values <= -margin[:, None], axis=1)
    # Anything with some footprint support at/inside an expanded old-material
    # boundary remains ambiguous and is delegated to the local grid.
    touches_old = np.any(values <= margin[:, None], axis=1)
    grid_band = (~safe_direct) & touches_old
    discard = (~safe_direct) & (~grid_band)
    return {
        'cloud': cloud,
        'direct_mask': safe_direct,
        'grid_mask': grid_band,
        'discard_mask': discard,
        'center_old_sdf': (np.asarray(center_values,dtype=np.float64).copy()
                           if center_values is not None else values[:,0].copy()),
        'min_old_sdf': values.min(axis=1),
        'max_old_sdf': values.max(axis=1),
        'footprint_queries': query_count,
    }


def _cyclic_boolean_transitions(mask: np.ndarray) -> np.ndarray:
    """Count inside/outside switches while walking once around each sampled ring."""
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=2 or m.shape[1]<2:
        return np.zeros(m.shape[0] if m.ndim else 0,dtype=np.int16)
    return np.count_nonzero(m != np.roll(m,-1,axis=1),axis=1).astype(np.int16)


def classify_representation_complexity_candidates(old_material: AnalyticMaterial,
                                                  cloud: SplatCloud,
                                                  sigma_scale: float = 0.75,
                                                  gaussian_reach_sigma: float = 2.5,
                                                  margin_cells: float = 0.15,
                                                  ring_samples: int = 8,
                                                  inner_ring_scale: float = 0.50):
    """Classify SweepSurface samples by *representational* complexity.

    The centre is used only as a certified 1-Lipschitz broad phase.  Borderline
    splats get an inner and outer tangent ring.  A mixed footprint is kept Direct
    when its sampled topology looks like one/two simple clipping edges (<=4 cyclic
    transitions and no hidden radial double crossing).  Local Grid is reserved for
    topology that a Full/Half/double-Half 2DGS cannot safely represent: closed
    islands, + - + / - + - radial patterns, or >4 outer-ring transitions.

    A later three-normal-family test may additionally promote a candidate to Grid.
    """
    count=len(cloud.positions)
    empty=np.zeros(count,dtype=bool)
    if not count:
        return dict(cloud=cloud,direct_mask=empty,grid_mask=empty,discard_mask=empty,
                    simple_clip_mask=empty,complex_mask=empty,hidden_crossing_mask=empty,
                    closed_island_mask=empty,ring_transitions=np.zeros(0,dtype=np.int16),
                    center_old_sdf=np.empty(0),min_old_sdf=np.empty(0),max_old_sdf=np.empty(0),
                    footprint_queries=0,baseline_queries=0,early_direct=0,early_discard=0,
                    ring_candidates=0,center_queries=0,center_ms=0.0,
                    grid_level_hint=np.zeros(0,dtype=np.int16))

    ring_samples=max(4,int(ring_samples))
    inner_ring_scale=float(np.clip(inner_ring_scale,0.10,0.90))
    p=np.asarray(cloud.positions,dtype=np.float64)
    n=np.asarray(cloud.normals,dtype=np.float64)
    support=np.asarray(cloud.support,dtype=np.float64)
    tu,tv=tangent_frames(n)
    reach=np.abs(support*float(sigma_scale)*float(gaussian_reach_sigma))
    margin=np.maximum(support*float(margin_cells),1e-8)

    t0=time.perf_counter()
    center=np.asarray(strip_material_value(old_material, p),dtype=np.float64)
    center_ms=(time.perf_counter()-t0)*1000.0

    # Analytic stock / flat / ball CSG is 1-Lipschitz under min/max composition.
    certified=(type(old_material) is AnalyticMaterial and
               (not len(old_material.sweeps) or old_material._sweep_index.supported[-1]))
    if certified:
        radius=reach*np.maximum(1.0,np.linalg.norm(n,axis=1))
        scale=1.0+np.max(np.abs(p),axis=1)+np.abs(center)+radius+np.abs(margin)
        guard=64.0*np.finfo(np.float64).eps*scale
        lower=center-radius-guard; upper=center+radius+guard
        early_direct=upper < -margin
        early_discard=lower > margin
    else:
        early_direct=np.zeros(count,dtype=bool)
        early_discard=np.zeros(count,dtype=bool)

    uncertain=~(early_direct|early_discard)
    ids=np.flatnonzero(uncertain)
    direct=early_direct.copy(); discard=early_discard.copy(); grid=np.zeros(count,dtype=bool)
    simple=np.zeros(count,dtype=bool); hidden=np.zeros(count,dtype=bool)
    closed=np.zeros(count,dtype=bool); complex_mask=np.zeros(count,dtype=bool)
    transitions=np.zeros(count,dtype=np.int16)
    minv=np.full(count,np.nan,dtype=np.float64); maxv=np.full(count,np.nan,dtype=np.float64)
    level_hint=np.zeros(count,dtype=np.int16)
    queries=count

    if len(ids):
        angles=np.arange(ring_samples,dtype=np.float64)*(2.0*math.pi/ring_samples)
        dirs=(np.cos(angles)[None,:,None]*tu[ids,None,:] +
              np.sin(angles)[None,:,None]*tv[ids,None,:])
        outer=p[ids,None,:] + reach[ids,None,None]*dirs
        inner=p[ids,None,:] + (reach[ids]*inner_ring_scale)[:,None,None]*dirs
        vv=np.asarray(strip_material_value(old_material, np.concatenate([inner,outer],axis=1).reshape(-1,3)),
                      dtype=np.float64).reshape(len(ids),2*ring_samples)
        queries+=vv.size
        iv=vv[:,:ring_samples]; ov=vv[:,ring_samples:]
        cv=center[ids]
        vals=np.concatenate([cv[:,None],iv,ov],axis=1)
        minv[ids]=vals.min(axis=1); maxv[ids]=vals.max(axis=1)

        ci=cv <= 0.0; ii=iv <= 0.0; oi=ov <= 0.0
        all_in=np.all(vals <= 0.0,axis=1)
        all_out=np.all(vals > 0.0,axis=1)
        mixed=~(all_in|all_out)

        tr=_cyclic_boolean_transitions(oi)
        transitions[ids]=tr
        # Two sign changes along one radial ray means the footprint contains a
        # hidden thin feature / double crossing that a single clip cannot encode.
        radial_double=np.any((ci[:,None] == oi) & (ii != ci[:,None]),axis=1)
        # If the outer ring is one sign but the centre is the other, the visible
        # region is a closed island/hole rather than one or two open clipping edges.
        outer_uniform=(tr==0)
        closed_local=outer_uniform & np.any(oi != ci[:,None],axis=1)
        # >4 cyclic transitions implies at least three boundary crossings / islands
        # at the sampled footprint scale.  <=4 is within the current Full/Half/
        # double-Half representation budget, provided no hidden radial event exists.
        complex_local=mixed & (radial_double | closed_local | (tr>4))
        simple_local=mixed & ~complex_local

        direct_local=all_in | simple_local
        discard_local=all_out
        grid_local=complex_local
        direct[ids]=direct_local; discard[ids]=discard_local; grid[ids]=grid_local
        simple[ids]=simple_local; hidden[ids]=radial_double
        closed[ids]=closed_local; complex_mask[ids]=complex_local
        # Hidden double crossings deserve h/4; other complex cases default h/2.
        hint=np.zeros(len(ids),dtype=np.int16)
        hint[complex_local]=1
        hint[radial_double | closed_local | (tr>6)]=2
        level_hint[ids]=hint

    return dict(cloud=cloud,direct_mask=direct,grid_mask=grid,discard_mask=discard,
                simple_clip_mask=simple,complex_mask=complex_mask,
                hidden_crossing_mask=hidden,closed_island_mask=closed,
                ring_transitions=transitions,center_old_sdf=center,
                min_old_sdf=minv,max_old_sdf=maxv,footprint_queries=int(queries),
                baseline_queries=int(count*(1+2*ring_samples)),
                early_direct=int(np.count_nonzero(early_direct)),
                early_discard=int(np.count_nonzero(early_discard)),
                ring_candidates=int(len(ids)),center_queries=int(count),center_ms=float(center_ms),
                grid_level_hint=level_hint)


def classify_sweep_exposure_novelty(old_material: AnalyticMaterial, cloud: SplatCloud,
                                    base_h: float, ring_samples: int = 4,
                                    footprint_scale: float = 0.60):
    """Decide whether a SweepSurface candidate exposes genuinely new surface.

    Grid necessity and surface admission are deliberately separate questions.  A
    candidate may be easy to represent but still redundant because the same material/
    air interface already existed before the current sweep.  We sample the candidate
    centre plus a small tangent ring and probe both sides along the material normal.

    NEW sample      : material existed on both sides before this sweep.
    EXISTING sample : material on -N, air on +N already before this sweep.
    AIR sample      : both sides were already outside material.

    Only candidates with no NEW/ambiguous samples and consisting entirely of EXISTING
    or AIR samples are confidently redundant.  Mixed NEW/EXISTING footprints are kept
    so Half2DGS/geometric splitting can represent the newly exposed portion.
    No NEW samples plus any ambiguous sample is UNRESOLVED, not permission to emit
    a whole Direct splat. Such candidates require local-grid reconstruction.
    """
    count=len(cloud.positions)
    empty=np.zeros(count,dtype=bool)
    if not count:
        return dict(redundant_mask=empty,fully_novel_mask=empty,partial_novel_mask=empty,
                    unresolved_mask=empty,
                    novel_samples=0,existing_samples=0,air_samples=0,ambiguous_samples=0,
                    query_points=0)

    p=np.asarray(cloud.positions,dtype=np.float64)
    n=np.asarray(cloud.normals,dtype=np.float64)
    n=n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    support=np.asarray(cloud.support,dtype=np.float64)
    tu,tv=tangent_frames(n)
    ring_samples=max(4,int(ring_samples))
    aa=np.arange(ring_samples,dtype=np.float64)*(2.0*math.pi/ring_samples)
    dirs=(np.cos(aa)[None,:,None]*tu[:,None,:] +
          np.sin(aa)[None,:,None]*tv[:,None,:])
    radius=(support*float(footprint_scale))[:,None,None]
    surf=np.concatenate([p[:,None,:],p[:,None,:]+radius*dirs],axis=1)
    ns=np.repeat(n[:,None,:],1+ring_samples,axis=1)

    eps=np.clip(0.16*support,0.035*float(base_h),0.14*float(base_h))
    tol=np.maximum(0.025*support,1e-7)
    minus=surf-ns*eps[:,None,None]
    plus =surf+ns*eps[:,None,None]
    m=1+ring_samples
    probes=np.concatenate([minus,plus],axis=1).reshape(-1,3)
    tt=np.broadcast_to(tol[:,None],(count,2*m)).reshape(-1)
    inside,outside=strip_material_range(old_material,probes,-tt,tt)
    inside=inside.reshape(count,2*m); outside=outside.reshape(count,2*m)
    novel=inside[:,:m] & inside[:,m:]
    existing=inside[:,:m] & outside[:,m:]
    air=outside[:,:m] & outside[:,m:]
    known=novel|existing|air
    ambiguous=~known

    any_novel=np.any(novel,axis=1)
    any_amb=np.any(ambiguous,axis=1)
    redundant=(~any_novel) & (~any_amb) & np.all(existing|air,axis=1)
    fully_novel=np.all(novel,axis=1)
    partial_novel=any_novel & ~fully_novel
    unresolved=(~any_novel) & any_amb
    return dict(redundant_mask=redundant,fully_novel_mask=fully_novel,
                partial_novel_mask=partial_novel,
                unresolved_mask=unresolved,
                novel_samples=int(np.count_nonzero(novel)),
                existing_samples=int(np.count_nonzero(existing)),
                air_samples=int(np.count_nonzero(air)),
                ambiguous_samples=int(np.count_nonzero(ambiguous)),
                query_points=len(probes))


def _rt_three_face_grid_mask(candidates: SplatCloud, before: SplatCloud,
                             candidate_mask: np.ndarray, base_h: float,
                             angle_deg: float = 38.0, *, spatial_index=None):
    """Find boundary candidates whose local neighbourhood contains >=3 normal families.

    The mask selects candidates for reconstruction by an h/4 SDF local grid.
    Normals are collected from both the current SweepSurface
    candidate cloud and the pre-cut geometric SplatCloud.
    """
    t0=time.perf_counter()
    count=len(candidates.positions)
    target=np.asarray(candidate_mask,dtype=bool).reshape(-1)
    out=np.zeros(count,dtype=bool)
    if count==0 or not np.any(target):
        return out,dict(vertex_grid_triggers=0,vertex_grid_ms=0.0)

    cp=np.asarray(candidates.positions,dtype=np.float64)
    cn=np.asarray(candidates.normals,dtype=np.float64)
    cn=cn/np.maximum(np.linalg.norm(cn,axis=1)[:,None],1e-30)
    cs=np.asarray(candidates.support,dtype=np.float64)
    bp=np.asarray(before.positions,dtype=np.float64).reshape(-1,3)
    bn=np.asarray(before.normals,dtype=np.float64).reshape(-1,3)
    if len(bn) and spatial_index is None: bn=bn/np.maximum(np.linalg.norm(bn,axis=1)[:,None],1e-30)
    face_cos=math.cos(math.radians(float(angle_deg)))

    cindex=_RealtimeSplatIndex(candidates,max(float(base_h)*4,1e-6)) if spatial_index is not None else None
    if _HAVE_SCIPY_GRAPH and spatial_index is None:
        ctree=cKDTree(cp); btree=cKDTree(bp) if len(bp) else None
    else:
        ctree=btree=None

    for i in np.flatnonzero(target):
        # Keep the detector local enough not to interpret smooth cutter curvature
        # as a trihedral corner, while still seeing incident h-scale face samples.
        radius=max(0.90*float(cs[i]),0.70*float(base_h),1e-8)
        if spatial_index is not None:
            ci=cindex.query(cp[i]-radius,cp[i]+radius,cp)
            ci=ci[np.sum((cp[ci]-cp[i])**2,axis=1)<=radius*radius].tolist()
            bi=spatial_index.query(cp[i]-radius,cp[i]+radius,bp)
            bi=bi[np.sum((bp[bi]-cp[i])**2,axis=1)<=radius*radius].tolist()
        elif ctree is not None:
            ci=ctree.query_ball_point(cp[i],radius)
            bi=btree.query_ball_point(cp[i],radius) if btree is not None else []
        else:
            ci=np.flatnonzero(np.sum((cp-cp[i])**2,axis=1)<=radius*radius).tolist()
            bi=(np.flatnonzero(np.sum((bp-cp[i])**2,axis=1)<=radius*radius).tolist()
                if len(bp) else [])
        reps=[]
        # Candidate normal first so the new cutter face is always represented.
        peers=bn[bi]
        if spatial_index is not None and len(peers): peers=peers/np.maximum(np.linalg.norm(peers,axis=1)[:,None],1e-30)
        for nn in [cn[i], *[cn[j] for j in ci], *peers]:
            if not reps or max(float(np.dot(nn,r)) for r in reps) < face_cos:
                reps.append(nn)
                if len(reps)>=3:
                    out[i]=True
                    break
    return out,dict(vertex_grid_triggers=int(np.count_nonzero(out)),
                    vertex_grid_ms=float((time.perf_counter()-t0)*1000.0))


def _rt_tag_direct_resolution_levels(cloud: SplatCloud, base_h: float) -> None:
    """Retag Direct SweepSurface splats by their actual support / base-h ratio.

    ToolProfileCache deliberately emits several Direct resolutions (3h, 2h, h,
    h/2), initially labeled ``refine_level=-2``.  The
    realtime renderer uses ``refine_levels`` as compact persistent debug metadata,
    so convert only those unresolved Direct labels here.  Stock (-1) and grid-generated
    levels (0,1,2,...) are left untouched.
    """
    if cloud is None or not len(cloud.positions) or not (float(base_h) > 0.0):
        return
    direct = np.asarray(cloud.refine_levels) == -2
    if not np.any(direct):
        return
    ratio = np.asarray(cloud.support, dtype=np.float64)[direct] / float(base_h)
    tagged = np.zeros(len(ratio), dtype=np.int16)
    # Coarser-than-base Direct families.
    tagged[ratio >= 2.50] = -4            # about 3h
    tagged[(ratio >= 1.50) & (ratio < 2.50)] = -3  # about 2h
    # Base h stays 0.  Finer Direct features are encoded in the same dyadic
    # convention as adaptive-grid results, so h/2 and h/4 get identical colors.
    fine = ratio < 0.75
    if np.any(fine):
        lev = np.rint(np.log2(1.0 / np.maximum(ratio[fine], 1e-12))).astype(np.int16)
        tagged[fine] = np.maximum(1, lev)
    cloud.refine_levels[direct] = tagged


def _rt_motion_u(sweep, points):
    if hasattr(sweep, 'a_xyz'):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        d = sweep.b_xyz-sweep.a_xyz
        return np.clip((p-sweep.a_xyz)@d/max(float(d@d), 1e-30), 0., 1.)
    p=np.asarray(points,dtype=np.float64).reshape(-1,3)
    if not len(p):
        return np.empty(0,dtype=np.float64)
    d=np.asarray(sweep.b_xy-sweep.a_xy,dtype=np.float64)
    den=float(d@d)
    if den < 1e-20:
        return np.ones(len(p),dtype=np.float64)
    u=((p[:,:2]-np.asarray(sweep.a_xy,dtype=np.float64))@d)/den
    return np.clip(u,0.0,1.0)
