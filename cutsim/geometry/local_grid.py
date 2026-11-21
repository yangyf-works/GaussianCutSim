"""GaussianCutSim geometry / local_grid implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import List, Tuple
import math
import numpy as np
from cutsim.core.sweep import BallSweep, FlatSweep


class StepGridFieldCache:
    """Step-local exact SDF cache, including dyadic h/h2/h4 lattice reuse.

    Two cache paths coexist:
      * axis-array reuse for arbitrary temporary micro grids,
      * integer dyadic lattice keys for aligned union grids at h, h/2, h/4...

    The dyadic path never rounds world coordinates. A node is identified by exact
    integer coordinates on the temporary step-local lattice, normalized to a common
    power-of-two denominator. Cache lifetime is one CSG snapshot / cut step only.
    """
    _MAX_DYADIC_LEVEL = 10
    _PACK_BITS = 21
    _PACK_BIAS = 1 << 20

    def __init__(self, max_bytes=48*1024*1024, max_entries=48):
        self.max_bytes=int(max_bytes)
        self.max_entries=int(max_entries)
        self.entries=[]
        self.lattice_entries=[]  # Compatibility field; lattice data uses one sorted store.
        self.lattice_store_key=None
        self.lattice_store_values=None
        self.lattice_store_sig=None
        self.bytes=0
        self.peak_bytes=0
        self.requested=0
        self.reused=0
        self.base_h=None
        self.lattice_signature=None

    def configure_lattice(self, frame, anchor, base_h):
        self.base_h=float(base_h)
        self.lattice_signature=tuple(np.asarray(a,dtype=np.float64).tobytes()
            for a in (frame.origin,frame.u,frame.v,frame.w,np.asarray(anchor,dtype=np.float64)))

    def invalidate(self):
        self.entries.clear()
        self.lattice_entries.clear()
        self.lattice_store_key=None
        self.lattice_store_values=None
        self.lattice_store_sig=None
        self.bytes=0
        if hasattr(self,'_adaptive_state'):del self._adaptive_state

    def _evict_for(self, size):
        if size>self.max_bytes or self.max_entries<=0:
            return False
        while (self.entries or self.lattice_entries) and (
                self.bytes+size>self.max_bytes or
                len(self.entries)+len(self.lattice_entries)>=self.max_entries):
            # Prefer evicting the oldest arbitrary-grid entry, then dyadic entry.
            if self.entries:
                self.bytes-=self.entries.pop(0)[-1]
            else:
                self.bytes-=self.lattice_entries.pop(0)[-1]
        return True

    def evaluate(self, material, frame, axes, world_nodes):
        key=tuple(np.asarray(a,dtype=np.float64).tobytes()
                  for a in (frame.origin,frame.u,frame.v,frame.w))
        shape=tuple(len(a) for a in axes)
        values=np.empty(shape,dtype=np.float64)
        filled=np.zeros(shape,dtype=bool)
        lo=np.array([a[0] for a in axes]); hi=np.array([a[-1] for a in axes])
        for old_key,old_axes,old_values,old_lo,old_hi,_size in reversed(self.entries):
            if key!=old_key or np.any(hi<old_lo) or np.any(lo>old_hi):
                continue
            current_ids=[]; previous_ids=[]
            for axis,old_axis in zip(axes,old_axes):
                _,i,j=np.intersect1d(axis,old_axis,assume_unique=True,return_indices=True)
                if not len(i):
                    break
                current_ids.append(i); previous_ids.append(j)
            if len(current_ids)!=3:
                continue
            dst=np.ix_(*current_ids)
            values[dst]=old_values[np.ix_(*previous_ids)]
            filled[dst]=True
            if np.all(filled):
                break
        missing=~filled.ravel()
        hits=int(len(world_nodes)-np.count_nonzero(missing))
        if hits==0:
            values[:]=material.value(world_nodes).reshape(shape)
        elif np.any(missing):
            values.ravel()[missing]=material.value(world_nodes[missing])
        self.requested+=len(world_nodes)
        self.reused+=hits
        size=values.nbytes+sum(a.nbytes for a in axes)+lo.nbytes+hi.nbytes
        if self._evict_for(size):
            self.entries.append((key,axes,values,lo,hi,size))
            self.bytes+=size
            self.peak_bytes=max(self.peak_bytes,self.bytes)
        return values.ravel()

    def _dyadic_level(self, h):
        if self.base_h is None or self.base_h<=0 or h<=0:
            return None
        ratio=self.base_h/float(h)
        if ratio<=0:
            return None
        level=int(round(math.log2(ratio)))
        if level<0 or level>self._MAX_DYADIC_LEVEL:
            return None
        if not math.isclose(ratio,2.0**level,rel_tol=2e-12,abs_tol=2e-12):
            return None
        return level

    def _pack_dyadic(self, ijk, h):
        level=self._dyadic_level(h)
        if level is None:
            return None
        factor=1 << (self._MAX_DYADIC_LEVEL-level)
        c=np.asarray(ijk,dtype=np.int64)*factor
        bias=self._PACK_BIAS
        if np.any(c<=-bias) or np.any(c>=bias):
            return None
        u=(c+bias).astype(np.uint64)
        b=self._PACK_BITS
        return (u[:,0] << np.uint64(2*b)) | (u[:,1] << np.uint64(b)) | u[:,2]

    def evaluate_lattice(self, material, frame, anchor, ijk, h, world_nodes, raw_requested=None):
        """Reuse exact base-h nodes in aligned h/2, h/4... union grids.

        The base union is unique and lexicographically ordered, so it is stored
        directly without sorting/merging. Finer unions only search that compact
        base table and evaluate missing nodes; they are not merged back into the table.
        This keeps cross-resolution reuse cheaper than recomputation.
        """
        n=len(world_nodes)
        raw=n if raw_requested is None else int(raw_requested)
        sig=tuple(np.asarray(a,dtype=np.float64).tobytes()
            for a in (frame.origin,frame.u,frame.v,frame.w,np.asarray(anchor,dtype=np.float64)))
        level=self._dyadic_level(h) if sig==self.lattice_signature else None
        packed=self._pack_dyadic(ijk,h) if level is not None else None
        if packed is None:
            values=material.value(world_nodes)
            self.requested+=raw; self.reused+=max(0,raw-n)
            return values

        # First/base-level call: no lookup overhead. Keep its exact keys for finer grids.
        if level==0 or self.lattice_store_key is None or self.lattice_store_sig!=sig:
            values=material.value(world_nodes)
            self.requested+=raw; self.reused+=max(0,raw-n)
            if level==0:
                if len(packed)>1 and np.any(packed[1:]<packed[:-1]):
                    order=np.argsort(packed)
                    self.lattice_store_key=packed[order].copy()
                    self.lattice_store_values=values[order].copy()
                else:
                    self.lattice_store_key=packed.copy()
                    self.lattice_store_values=values.copy()
                self.lattice_store_sig=sig
                self.peak_bytes=max(self.peak_bytes,int(self.lattice_store_key.nbytes+self.lattice_store_values.nbytes+self.bytes))
            return values

        old_keys=self.lattice_store_key; old_vals=self.lattice_store_values
        pos=np.searchsorted(old_keys,packed)
        valid=pos<len(old_keys)
        hit=np.zeros(n,dtype=bool)
        hit[valid]=old_keys[pos[valid]]==packed[valid]
        values=np.empty(n,dtype=np.float64)
        if np.any(hit):
            values[hit]=old_vals[pos[hit]]
        missing=~hit
        if np.any(missing):
            values[missing]=material.value(world_nodes[missing])
        hits=int(np.count_nonzero(hit))
        self.requested+=raw
        self.reused+=max(0,raw-n)+hits
        return values


@dataclass(frozen=True)
class LocalFrame:
    origin: np.ndarray
    u: np.ndarray   # long axis
    v: np.ndarray   # short axis
    w: np.ndarray   # Z
    lo: np.ndarray  # local coordinate min
    hi: np.ndarray  # local coordinate max
    grid_phase: float = 0.5  # fractional cell offset; 0.5 puts bounds between nodes
    lattice_origin: np.ndarray | None = None  # common step-local lattice, never global

    def world_from_local(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
        return (self.origin + q[:, 0, None] * self.u + q[:, 1, None] * self.v
                + q[:, 2, None] * self.w)

    def local_from_world(self, p: np.ndarray) -> np.ndarray:
        d = np.asarray(p, dtype=np.float64).reshape(-1, 3) - self.origin
        return np.column_stack([d @ self.u, d @ self.v, d @ self.w])


def make_step_frame(sweep, stock_lo: np.ndarray, stock_hi: np.ndarray,
                    halo: float, grid_phase: float = 0.5) -> LocalFrame:
    dxy = sweep.b_xy - sweep.a_xy
    length = float(np.linalg.norm(dxy))
    if length < 1e-12:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([dxy[0] / length, dxy[1] / length, 0.0])
    v = np.array([-u[1], u[0], 0.0])
    w = np.array([0.0, 0.0, 1.0])
    origin = np.array([sweep.a_xy[0], sweep.a_xy[1], 0.0], dtype=np.float64)
    lo = np.array([-sweep.radius - halo,
                   -sweep.radius - halo,
                   max(stock_lo[2], sweep.z_lo - halo)], dtype=np.float64)
    hi = np.array([length + sweep.radius + halo,
                   sweep.radius + halo,
                   min(stock_hi[2] + halo, sweep.z_hi + halo)], dtype=np.float64)
    return LocalFrame(origin, u, v, w, lo, hi, float(grid_phase))


def tool_shape_mask_local_grid(sweep, xu: np.ndarray, yv: np.ndarray, zw: np.ndarray,
                               halo: float) -> np.ndarray:
    """Return active nodes of a regular local lattice inside the swept tool shape.

    The lattice remains Cartesian/oriented, but nodes outside the cutter sweep plus a
    small compute halo are skipped entirely.  For a translating flat end mill the XY
    footprint is an exact capsule (rectangle + semicircular end caps).  For a ball end
    mill the mask follows the lower swept sphere/capsule plus the upper cylindrical
    sweep.  This is a compute mask only; it does not create an artificial material
    boundary.
    """
    x=np.asarray(xu,dtype=np.float64).reshape(-1,1,1)
    y=np.asarray(yv,dtype=np.float64).reshape(1,-1,1)
    z=np.asarray(zw,dtype=np.float64).reshape(1,1,-1)
    length=float(sweep.move_length)
    # Local U runs from cutter start to end. Distance to that segment in the UV plane.
    du=np.where(x < 0.0, -x, np.where(x > length, x-length, 0.0))
    radial2=du*du + y*y
    r=float(sweep.radius)+max(float(halo),0.0)
    r2=r*r
    if isinstance(sweep, FlatSweep):
        z_ok=(z >= float(sweep.z_lo)-halo) & (z <= float(sweep.z_hi)+halo)
        return (radial2 <= r2) & z_ok
    if isinstance(sweep, BallSweep):
        # Conservative expanded union of the swept lower ball and upper cylinder.
        lower=(radial2 + (z-float(sweep.center_z))**2) <= r2
        upper=((radial2 <= r2) &
               (z >= float(sweep.center_z)-halo) &
               (z <= float(sweep.z_hi)+halo))
        return lower | upper
    return np.ones((len(xu),len(yv),len(zw)),dtype=bool)


def tool_shape_mask_points(sweep, local_points: np.ndarray, halo: float) -> np.ndarray:
    """Point form of :func:`tool_shape_mask_local_grid` for temporary micro grids."""
    q=np.asarray(local_points,dtype=np.float64).reshape(-1,3)
    if not len(q):
        return np.zeros(0,dtype=bool)
    length=float(sweep.move_length)
    x=q[:,0]; y=q[:,1]; z=q[:,2]
    du=np.where(x < 0.0, -x, np.where(x > length, x-length, 0.0))
    radial2=du*du+y*y
    r=float(sweep.radius)+max(float(halo),0.0)
    r2=r*r
    if isinstance(sweep, FlatSweep):
        return ((radial2 <= r2) &
                (z >= float(sweep.z_lo)-halo) &
                (z <= float(sweep.z_hi)+halo))
    if isinstance(sweep, BallSweep):
        lower=(radial2+(z-float(sweep.center_z))**2) <= r2
        upper=((radial2 <= r2) &
               (z >= float(sweep.center_z)-halo) &
               (z <= float(sweep.z_hi)+halo))
        return lower | upper
    return np.ones(len(q),dtype=bool)


def tool_shape_cell_safe_node_mask(sweep, xu: np.ndarray, yv: np.ndarray, zw: np.ndarray,
                                   halo: float) -> np.ndarray:
    """Conservative cell-based tool mask converted back to active lattice nodes.

    A cell is active when its center is inside the swept shape expanded by the
    cell half-diagonal. This covers crossing edges near curved boundaries.
    All 8 corners of an active cell are evaluated. The extra nodes are
    compute-only safety nodes:
    emitted roots are trimmed back to the original ``halo`` envelope.
    """
    xu=np.asarray(xu,dtype=np.float64); yv=np.asarray(yv,dtype=np.float64); zw=np.asarray(zw,dtype=np.float64)
    if min(len(xu),len(yv),len(zw)) < 2:
        return tool_shape_mask_local_grid(sweep,xu,yv,zw,halo)
    cx=(xu[:-1]+xu[1:])*0.5
    cy=(yv[:-1]+yv[1:])*0.5
    cz=(zw[:-1]+zw[1:])*0.5
    dx=float(np.max(np.abs(np.diff(xu)))) if len(xu)>1 else 0.0
    dy=float(np.max(np.abs(np.diff(yv)))) if len(yv)>1 else 0.0
    dz=float(np.max(np.abs(np.diff(zw)))) if len(zw)>1 else 0.0
    half_diag=0.5*math.sqrt(dx*dx+dy*dy+dz*dz)
    cells=tool_shape_mask_local_grid(sweep,cx,cy,cz,float(halo)+half_diag)
    nodes=np.zeros((len(xu),len(yv),len(zw)),dtype=bool)
    # Convert active cells to their 8 corner nodes.
    for ox in (0,1):
        for oy in (0,1):
            for oz in (0,1):
                nodes[ox:ox+cells.shape[0],oy:oy+cells.shape[1],oz:oz+cells.shape[2]] |= cells
    return nodes


def _adjacent_cells_for_edges(edge_bases: np.ndarray, axis: int,
                               lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return valid coarse-cell lower indices touching selected lattice edges."""
    e=np.asarray(edge_bases,dtype=np.int64).reshape(-1,3)
    if not len(e):
        return np.empty((0,3),dtype=np.int64)
    others=[a for a in range(3) if a!=int(axis)]
    parts=[]
    for o0 in (-1,0):
        for o1 in (-1,0):
            off=np.zeros(3,dtype=np.int64)
            off[others[0]]=o0; off[others[1]]=o1
            c=e+off
            # Cell lower node must be in [lo, hi-1] on all axes.
            good=np.all(c>=lo,axis=1)&np.all(c<hi,axis=1)
            if np.any(good):
                parts.append(c[good])
    return np.concatenate(parts,axis=0) if parts else np.empty((0,3),dtype=np.int64)


def _thin_cell_frames(frame: LocalFrame, anchor: np.ndarray, cells: np.ndarray,
                      coarse_h: float, fine_h: float):
    """Merge neighboring suspect cells and build h/2 extraction frames.

    Each returned frame owns exactly the selected coarse-cell component, with one
    fine-cell extraction halo on every side.  The fine lattice shares the coarse
    anchor, so h, h/2 and h/4 nodes are exactly nested.
    """
    from cutsim.geometry.grid_patches import _connected_probe_components
    cells=np.asarray(cells,dtype=np.int64).reshape(-1,3)
    if not len(cells):
        return []
    cells=np.unique(cells,axis=0)
    centers_local=np.asarray(anchor,dtype=np.float64)+(
        cells.astype(np.float64)+0.5)*float(coarse_h)
    centers_world=frame.world_from_local(centers_local)
    comps=_connected_probe_components(frame,centers_world,float(coarse_h),link_scale=1.80)
    out=[]
    for ids in comps:
        cc=cells[np.asarray(ids,dtype=np.int64)]
        cmin=cc.min(axis=0); cmax=cc.max(axis=0)+1
        own_lo=np.asarray(anchor,dtype=np.float64)+cmin.astype(np.float64)*float(coarse_h)
        own_hi=np.asarray(anchor,dtype=np.float64)+cmax.astype(np.float64)*float(coarse_h)
        # One fine cell outside the owner is extraction halo; replacement_local_bounds()
        # will remove it again when defining ownership.
        fr=LocalFrame(frame.origin.copy(),frame.u.copy(),frame.v.copy(),frame.w.copy(),
                      own_lo-float(fine_h),own_hi+float(fine_h),frame.grid_phase,
                      np.asarray(anchor,dtype=np.float64).copy())
        out.append(fr)
    return out


def aligned_axis(lo: float, hi: float, h: float, phase: float = 0.5) -> np.ndarray:
    """Temporary local axis with an EXACT cell pitch ``h``.

    The requested interval [lo, hi] is covered with exact pitch ``h``.
    ``phase`` is a fraction of one cell measured from ``lo`` toward the previous
    grid node.  The default 0.5 therefore places ``lo`` halfway between two
    nodes instead of exactly on a node.  With cut-scaled grids this is important:
    tool radii and move lengths are often integer multiples of ``h``, so an
    unshifted lattice would repeatedly put cutter endpoints / sharp boundaries
    exactly on grid nodes.

    The phase shift does not change cell size or orientation; it only changes
    lattice origin.  Extra nodes are added as needed so [lo, hi] is still fully
    covered.
    """
    if not (np.isfinite(h) and h > 0.0):
        raise ValueError("Local grid cell size must be finite and > 0")
    if not np.isfinite(phase):
        raise ValueError("Local grid phase must be finite")
    phase = float(phase) % 1.0
    span = float(hi - lo)
    if not (np.isfinite(span) and span > 0.0):
        raise ValueError("Local grid axis requires hi > lo")

    start = float(lo) - phase * float(h)
    n = max(1, int(math.ceil((float(hi) - start) / float(h) - 1e-12)))
    return start + np.arange(n + 1, dtype=np.float64) * float(h)


def local_grid_axes(frame: LocalFrame, h: float):
    """Return the EXACT temporary grid coordinates used by computation and display.

    All three axes use the same pitch ``h``.  Keeping this in one helper is
    important: the renderer visualizes the same grid that the surface
    extractor evaluates.
    """
    if frame.lattice_origin is not None:
        anchor=np.asarray(frame.lattice_origin,dtype=np.float64)
        first=np.floor((frame.lo-anchor)/h+1e-10).astype(np.int64)
        last=np.ceil((frame.hi-anchor)/h-1e-10).astype(np.int64)
        return tuple(anchor[k]+np.arange(first[k],last[k]+1,dtype=np.float64)*h
                     for k in range(3))
    phase = float(frame.grid_phase)
    return (aligned_axis(frame.lo[0], frame.hi[0], h, phase),
            aligned_axis(frame.lo[1], frame.hi[1], h, phase),
            aligned_axis(frame.lo[2], frame.hi[2], h, phase))


def make_local_grid(frame: LocalFrame, h: float):
    xu, yv, zw = local_grid_axes(frame, h)
    U, V, W = np.meshgrid(xu, yv, zw, indexing="ij")
    local = np.column_stack([U.ravel(), V.ravel(), W.ravel()])
    world = frame.world_from_local(local)
    return (xu, yv, zw), local, world


def replacement_local_bounds(frame: LocalFrame, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Owned/replaced portion of the temporary grid.

    The full temporary grid is an extraction halo.  One exact grid cell is kept
    around the outside so old splats just outside the patch overlap the rebuilt
    patch instead of leaving a seam.  Everything inside this box is regenerated
    from the *current* analytic CSG, including uncut stock faces.
    """
    axes = local_grid_axes(frame, h)
    lo = np.array([a[0] for a in axes], dtype=np.float64)
    hi = np.array([a[-1] for a in axes], dtype=np.float64)
    margin = float(h)
    own_lo = lo + margin
    own_hi = hi - margin
    # Degenerate safety for very small demo boxes.
    bad = own_hi <= own_lo
    own_lo[bad], own_hi[bad] = lo[bad], hi[bad]
    return own_lo, own_hi


def points_in_local_box(frame: LocalFrame, points: np.ndarray,
                        lo: np.ndarray, hi: np.ndarray,
                        eps: float = 1e-9) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    q = frame.local_from_world(points)
    return np.all(q >= np.asarray(lo)-eps, axis=1) & np.all(q <= np.asarray(hi)+eps, axis=1)
