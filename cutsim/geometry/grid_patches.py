"""GaussianCutSim geometry / grid_patches implementation."""
from __future__ import annotations

import math
import numpy as np
import time
from cutsim.core.material import AnalyticMaterial
from cutsim.core.scipy_support import _HAVE_SCIPY_GRAPH, cKDTree, coo_matrix, sparse_connected_components
from cutsim.core.sweep import FlatSweep
from cutsim.geometry.local_grid import LocalFrame, local_grid_axes, points_in_local_box, replacement_local_bounds, tool_shape_mask_points
from cutsim.geometry.tool_profile import _centered_samples, _sweep_uv_frame


def _connected_probe_components_python(frame: LocalFrame, points: np.ndarray, h: float,
                                       link_scale: float = 2.15):
    """Find connected probe components with a spatial hash and Python BFS."""
    if not len(points):
        return []
    q = frame.local_from_world(points)
    cell = max(float(h)*1.6, 1e-9)
    bins = np.floor(q/cell).astype(np.int64)
    table = {}
    for i, b in enumerate(map(tuple, bins)):
        table.setdefault(b, []).append(i)
    seen = np.zeros(len(q), dtype=bool)
    max_d2 = (float(h)*float(link_scale))**2
    comps=[]
    neigh=[(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)]
    for seed in range(len(q)):
        if seen[seed]:
            continue
        seen[seed]=True
        stack=[seed]; comp=[]
        while stack:
            i=stack.pop(); comp.append(i)
            b=bins[i]
            for di,dj,dk in neigh:
                for j in table.get((int(b[0]+di),int(b[1]+dj),int(b[2]+dk)),()):
                    if seen[j]:
                        continue
                    d=q[j]-q[i]
                    if float(d@d) <= max_d2:
                        seen[j]=True; stack.append(j)
        comps.append(np.asarray(comp,dtype=np.int64))
    return comps


def _connected_probe_components(frame: LocalFrame, points: np.ndarray, h: float,
                                link_scale: float = 2.15):
    """Find connected components using distance and adjacent-bin connectivity.

    cKDTree finds all geometric neighbors in compiled code. The spatial-hash rule
    rejects pairs whose integer bins differ by >1 on any local axis.
    Small inputs and environments without SciPy use Python BFS.
    """
    if not len(points):
        return []
    if not _HAVE_SCIPY_GRAPH or len(points) < 32:
        return _connected_probe_components_python(frame, points, h, link_scale)

    q = frame.local_from_world(points)
    cell = max(float(h)*1.6, 1e-9)
    bins = np.floor(q/cell).astype(np.int64)
    radius = float(h)*float(link_scale)
    pairs = cKDTree(q).query_pairs(radius, output_type='ndarray')
    if len(pairs):
        near_bin = np.all(np.abs(bins[pairs[:, 0]] - bins[pairs[:, 1]]) <= 1, axis=1)
        pairs = pairs[near_bin]

    n = len(q)
    if not len(pairs):
        return [np.array([i], dtype=np.int64) for i in range(n)]

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n)).tocsr()
    _ncomp, labels = sparse_connected_components(graph, directed=False, return_labels=True)

    order = np.argsort(labels, kind='stable')
    sorted_labels = labels[order]
    cuts = np.r_[0, np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1, n]
    comps = [np.sort(order[a:b]).astype(np.int64, copy=False) for a, b in zip(cuts[:-1], cuts[1:])]
    comps.sort(key=lambda ids: int(ids[0]) if len(ids) else n)
    return comps


def _translate_bounds_within_clip(lo: np.ndarray, hi: np.ndarray,
                                  clip_lo: np.ndarray, clip_hi: np.ndarray,
                                  delta: np.ndarray):
    """Translate one local BBOX without changing its size, clamped to a clip box."""
    lo=np.asarray(lo,dtype=np.float64).copy(); hi=np.asarray(hi,dtype=np.float64).copy()
    clo=np.asarray(clip_lo,dtype=np.float64); chi=np.asarray(clip_hi,dtype=np.float64)
    d=np.asarray(delta,dtype=np.float64)
    lo += d; hi += d
    for ax in range(3):
        if lo[ax] < clo[ax]:
            sh=clo[ax]-lo[ax]; lo[ax]+=sh; hi[ax]+=sh
        if hi[ax] > chi[ax]:
            sh=hi[ax]-chi[ax]; lo[ax]-=sh; hi[ax]-=sh
        lo[ax]=max(lo[ax],clo[ax]); hi[ax]=min(hi[ax],chi[ax])
    return lo,hi


def _offset_patch_bounds(lo: np.ndarray, hi: np.ndarray,
                         clip_lo: np.ndarray, clip_hi: np.ndarray,
                         h: float, bbox_offset_cells: float):
    """Quarter-cell BBOX translation, independent from the 0.5-cell grid phase.

    With grid_phase=0.5 the lattice nodes already avoid the BBOX faces.  Moving the
    BBOX itself by another quarter cell prevents a detected surface/feature from
    repeatedly landing at the same relative cell or owned-box boundary.
    """
    off=float(bbox_offset_cells)*float(h)
    # Different signs avoid preserving symmetry on all three axes.
    delta=np.array([+off,-off,+off],dtype=np.float64)
    return _translate_bounds_within_clip(lo,hi,clip_lo,clip_hi,delta)


def _split_contiguous_true(mask: np.ndarray):
    mask=np.asarray(mask,dtype=bool).reshape(-1)
    ids=np.flatnonzero(mask)
    if not len(ids):
        return []
    cuts=np.flatnonzero(np.diff(ids)>1)+1
    return [g for g in np.split(ids,cuts) if len(g)]


def _flat_rim_feature_segments(sweep: FlatSweep, spacing: float):
    """Known intrinsic 90-degree bottom/side rim, returned as ordered curve pieces."""
    h=max(float(spacing),1e-9)
    u,v,_w,length=_sweep_uv_frame(sweep)
    u2,v2=u[:2],v[:2]
    a=np.asarray(sweep.a_xy,dtype=np.float64); b=np.asarray(sweep.b_xy,dtype=np.float64)
    r=float(sweep.radius); pieces=[]
    if length < 1e-12:
        n=max(16,int(math.ceil(2.0*math.pi*r/h)))
        ang=(np.arange(n,dtype=np.float64)+0.5)*(2.0*math.pi/n)
        d=np.cos(ang)[:,None]*u2+np.sin(ang)[:,None]*v2
        pieces.append(a[None,:]+r*d)
    else:
        t=_centered_samples(0.0,length,h)
        line=a[None,:]+t[:,None]*u2
        pieces.append(line+r*v2)
        pieces.append(line-r*v2)
        ncap=max(8,int(math.ceil(math.pi*r/h)))
        da=math.pi/ncap
        ang0=math.pi*0.5+(np.arange(ncap,dtype=np.float64)+0.5)*da
        d0=np.cos(ang0)[:,None]*u2+np.sin(ang0)[:,None]*v2
        pieces.append(a[None,:]+r*d0)
        ang1=-math.pi*0.5+(np.arange(ncap,dtype=np.float64)+0.5)*da
        d1=np.cos(ang1)[:,None]*u2+np.sin(ang1)[:,None]*v2
        pieces.append(b[None,:]+r*d1)
    return pieces


def build_flat_feature_microgrid_frames(sweep: FlatSweep, old_material: AnalyticMaterial,
                                        base_frame: LocalFrame, h: float,
                                        bbox_offset_cells: float=0.25,
                                        feature_strip_cells: float=8.0,
                                        support_halo: float=0.0):
    """Seed h/2 grids on exposed Flat-End 90-degree rim segments.

    Analytically determined exposed rim pieces seed a local SDF grid at h/2.
    Adaptive refinement can then promote difficult pieces to h/4.
    """
    fine_h=0.5*float(h)
    if fine_h <= 0.0:
        return []
    clip_lo,clip_hi=replacement_local_bounds(base_frame,float(h))
    frames=[]; pad=max(1.5*float(h),2.5*fine_h,float(support_halo))
    max_len=max(float(feature_strip_cells)*float(h),4.0*float(h))
    pts_per=max(2,int(math.ceil(max_len/max(float(h),1e-9))))
    for xy in _flat_rim_feature_segments(sweep,float(h)):
        p3=np.column_stack([xy,np.full(len(xy),float(sweep.z_lo))])
        # A rim only becomes exposed where the old material still exists.
        v=old_material.value(p3)
        exposed=v <= max(0.20*float(h),1e-6)
        for run in _split_contiguous_true(exposed):
            for k in range(0,len(run),pts_per):
                ids=run[k:k+pts_per]
                if not len(ids):
                    continue
                q=base_frame.local_from_world(p3[ids])
                lo=np.maximum(q.min(axis=0)-pad,clip_lo)
                hi=np.minimum(q.max(axis=0)+pad,clip_hi)
                if np.any(hi<=lo+1e-9):
                    continue
                lo,hi=_offset_patch_bounds(lo,hi,clip_lo,clip_hi,
                                           fine_h,bbox_offset_cells)
                # Fine extraction needs its own one-cell halo beyond the owned box.
                fr=LocalFrame(base_frame.origin.copy(),base_frame.u.copy(),base_frame.v.copy(),base_frame.w.copy(),
                              lo-fine_h,hi+fine_h,base_frame.grid_phase)
                frames.append(fr)
    return frames


def _owned_delete_mask_for_frames(points: np.ndarray, frame_h_pairs,
                                  active_sweep=None,
                                  tool_mask_halo_cells: float = 0.0):
    """Delete centers only where the corresponding reconstruction can replace them.

    Intersect the owner BBOX with the *same swept-tool compute mask* used by
    extraction. Therefore:
      * actual cutter-interior centers are still deleted separately by center_removed;
      * grid replacement deletes only owner centers inside the tool-shaped mask;
      * untouched stock outside the masked reconstruction footprint survives.

    Candidate lookup uses x-sorted coordinates.
    """
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    mask=np.zeros(len(points),dtype=bool)
    if not len(points) or not frame_h_pairs:
        return mask

    families={}
    for fr,hh in frame_h_pairs:
        key=tuple(np.asarray(a,dtype=np.float64).tobytes()
                  for a in (fr.origin,fr.u,fr.v,fr.w))
        families.setdefault(key, [fr, []])[1].append((fr,float(hh)))

    for _key,(base_fr,pairs) in families.items():
        q=base_fr.local_from_world(points)
        order=np.argsort(q[:,0],kind='stable')
        sx=q[order,0]
        for fr,hh in pairs:
            lo,hi=replacement_local_bounds(fr,hh)
            eps=max(hh*1e-6,1e-9)
            left=int(np.searchsorted(sx,lo[0]-eps,side='left'))
            right=int(np.searchsorted(sx,hi[0]+eps,side='right'))
            if right<=left:
                continue
            ids=order[left:right]
            ids=ids[~mask[ids]]
            if not len(ids):
                continue
            qq=q[ids]
            yz=((qq[:,1]>=lo[1]-eps)&(qq[:,1]<=hi[1]+eps)&
                (qq[:,2]>=lo[2]-eps)&(qq[:,2]<=hi[2]+eps))
            owned_ids=ids[yz]
            if not len(owned_ids):
                continue

            if active_sweep is not None:
                # Match the physical halo of extract_surface_from_grid_union():
                # halo = tool_mask_halo_cells * this grid resolution.
                halo=max(float(tool_mask_halo_cells),0.0)*float(hh)
                inside_tool_mask=tool_shape_mask_points(
                    active_sweep, q[owned_ids], halo)
                owned_ids=owned_ids[inside_tool_mask]

            mask[owned_ids]=True
    return mask


def _output_keep_mask_for_frames(points: np.ndarray, frame_h_pairs, overlap_cells: float):
    """Keep reconstructed splats inside owner boxes plus a controlled collar.

    ``replacement_local_bounds()`` is the central owner.  The temporary extraction
    grid extends one full cell beyond it.  This helper trims only the *output*
    splats to owner +/- overlap_cells*h, leaving grid evaluation/root finding
    unchanged.  Direct/cached splats are still deleted only inside the owner, so
    overlap_cells directly controls the visible Direct/Grid transition width.
    """
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    keep=np.zeros(len(points),dtype=bool)
    if not len(points) or not frame_h_pairs:
        return keep
    overlap_cells=float(np.clip(overlap_cells,0.0,1.0))
    families={}
    for fr,hh in frame_h_pairs:
        key=tuple(np.asarray(a,dtype=np.float64).tobytes()
                  for a in (fr.origin,fr.u,fr.v,fr.w))
        families.setdefault(key,[fr,[]])[1].append((fr,float(hh)))
    for _key,(base_fr,pairs) in families.items():
        q=base_fr.local_from_world(points)
        order=np.argsort(q[:,0],kind='stable')
        sx=q[order,0]
        for fr,hh in pairs:
            lo,hi=replacement_local_bounds(fr,hh)
            collar=overlap_cells*hh
            lo=lo-collar; hi=hi+collar
            eps=max(hh*1e-6,1e-9)
            left=int(np.searchsorted(sx,lo[0]-eps,side='left'))
            right=int(np.searchsorted(sx,hi[0]+eps,side='right'))
            if right<=left:
                continue
            ids=order[left:right]
            ids=ids[~keep[ids]]
            if not len(ids):
                continue
            qq=q[ids]
            yz=((qq[:,1]>=lo[1]-eps)&(qq[:,1]<=hi[1]+eps)&
                (qq[:,2]>=lo[2]-eps)&(qq[:,2]<=hi[2]+eps))
            keep[ids[yz]]=True
    return keep


def build_intersection_grid_frames(base_frame: LocalFrame, red_points: np.ndarray, h: float,
                                   halo_cells: float = 1.5, max_strip_cells: float = 10.0,
                                   transition_cells: float = 0.0,
                                   bbox_offset_cells: float = 0.25,
                                   support_halo: float = 0.0, timings=None):
    """Convert red Boolean-intersection probe points into short oriented grid strips.

    Long contours are intentionally split instead of being wrapped by one huge AABB.
    """
    started=time.perf_counter()
    if timings is not None:
        timings.update(connect_ms=0.0,bounds_ms=0.0)
    red_points=np.asarray(red_points,dtype=np.float64).reshape(-1,3)
    if not len(red_points):
        return []
    connect_start=time.perf_counter()
    components=_connected_probe_components(base_frame,red_points,h)
    connect_end=time.perf_counter()
    q=base_frame.local_from_world(red_points)
    frames=[]
    for comp in components:
        qc=q[comp]
        # Split primarily along the longest local extent; this prevents a long contour
        # from owning the entire swept interior.
        extent=np.ptp(qc,axis=0)
        axis=int(np.argmax(extent))
        strip_len=max(float(h)*float(max_strip_cells), 2.0*float(h))
        amin=float(qc[:,axis].min()); amax=float(qc[:,axis].max())
        nseg=max(1,int(math.ceil(max(amax-amin,0.0)/strip_len)))
        edges=np.linspace(amin,amax,nseg+1) if nseg>1 else np.array([amin,amax+1e-12])
        for s in range(nseg):
            if s==nseg-1:
                take=(qc[:,axis]>=edges[s]-1e-12)&(qc[:,axis]<=edges[s+1]+1e-12)
            else:
                take=(qc[:,axis]>=edges[s]-1e-12)&(qc[:,axis]<edges[s+1])
            if not np.any(take):
                continue
            qs=qc[take]
            pad=max(float(halo_cells)*float(h), 1.50*float(h), float(support_halo))
            lo=qs.min(axis=0)-pad
            hi=qs.max(axis=0)+pad
            # Extra stock-transition band so the rebuilt patch overlaps the untouched
            # cached stock better even when the neighboring support sizes are different.
            grow=max(float(transition_cells)*float(h), 0.0)
            lo -= grow
            hi += grow
            # Keep each patch compact and inside the base sweep extraction envelope.
            lo=np.maximum(lo, base_frame.lo)
            hi=np.minimum(hi, base_frame.hi)
            for ax in range(3):
                if hi[ax]-lo[ax] < 3.0*h:
                    mid=0.5*(hi[ax]+lo[ax])
                    lo[ax]=max(base_frame.lo[ax], mid-1.50*h)
                    hi[ax]=min(base_frame.hi[ax], mid+1.50*h)
            if np.all(hi>lo+1e-9):
                lo,hi=_offset_patch_bounds(lo,hi,base_frame.lo,base_frame.hi,
                                           h,bbox_offset_cells)
                frames.append(LocalFrame(base_frame.origin.copy(), base_frame.u.copy(),
                                         base_frame.v.copy(), base_frame.w.copy(),
                                         lo.astype(np.float64), hi.astype(np.float64),
                                         base_frame.grid_phase))
    if timings is not None:
        timings.update(connect_ms=(connect_end-connect_start)*1000.0,
                       bounds_ms=((time.perf_counter()-started)-(connect_end-connect_start))*1000.0)
    return frames


def _mask_points_in_any_owned_frame(points: np.ndarray, frames, h: float, expand: float = 0.0):
    mask=np.zeros(len(points),dtype=bool)
    if not len(points):
        return mask
    for fr in frames:
        lo,hi=replacement_local_bounds(fr,h)
        lo=lo-float(expand); hi=hi+float(expand)
        mask |= points_in_local_box(fr, points, lo, hi, eps=max(h*1e-6,1e-9))
    return mask


def _aggregate_grid_stats(stats_list, h: float):
    sums={k:0.0 for k in ('grid_build_ms','field_ms','crossing_ms','root_ms','normal_ms','filter_ms')}
    out={
        'grid_nodes':0, 'grid_shape':f'{len(stats_list)} intersection patches',
        'grid_spacing':(float(h),float(h),float(h)), 'crossing_edges':0,
        'sharp_edge_roots':0, 'sharp_edge_extra_splats':0, 'fallback_normal_roots':0,
        'debug_crossing_cells_local':np.empty((0,3),dtype=np.float64),
    }
    for st in stats_list:
        out['grid_nodes'] += int(st.get('grid_nodes',0))
        out['crossing_edges'] += int(st.get('crossing_edges',0))
        out['sharp_edge_roots'] += int(st.get('sharp_edge_roots',0))
        out['sharp_edge_extra_splats'] += int(st.get('sharp_edge_extra_splats',0))
        out['fallback_normal_roots'] += int(st.get('fallback_normal_roots',0))
        for k in sums: sums[k]+=float(st.get(k,0.0))
    out.update(sums)
    return out


def align_patch_frames(frames, anchor):
    """Keep compact patch bounds; share a lattice only within this cut step."""
    return [LocalFrame(fr.origin,fr.u,fr.v,fr.w,fr.lo,fr.hi,
                       fr.grid_phase,np.asarray(anchor,dtype=np.float64).copy())
            for fr in frames]


def _inside_extraction_union(frame, points, frames, h):
    q=frame.local_from_world(points)
    mask=np.zeros(len(points),dtype=bool)
    for fr in frames:
        axes=local_grid_axes(fr,h)
        lo=np.array([a[0] for a in axes]); hi=np.array([a[-1] for a in axes])
        mask|=np.all(q>=lo-1e-9,axis=1)&np.all(q<=hi+1e-9,axis=1)
    return mask
