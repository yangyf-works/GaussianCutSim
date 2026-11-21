"""GaussianCutSim ui / helpers implementation."""
from __future__ import annotations

import numpy as np
import time
from cutsim.geometry.local_grid import LocalFrame, local_grid_axes


from cutsim.rendering.opaque_coverage import DEFAULT_DISPLAY_SCALE
DEFAULT_GS_DISPLAY_SIZE = DEFAULT_DISPLAY_SCALE
GS_DISPLAY_SIZES = (0.25, 0.50, DEFAULT_GS_DISPLAY_SIZE)
PLAYBACK_SPEEDS = (.25, .5, 1., 1.5, 2., 4., 8., 16.)


def display_size_index(value):
    return min(range(len(GS_DISPLAY_SIZES)),key=lambda i:abs(GS_DISPLAY_SIZES[i]-float(value)))


def actual_grid_local_bounds(frame: LocalFrame, h: float):
    """Bounds of the actual aligned temporary grid, not the requested OBB."""
    axes = local_grid_axes(frame, h)
    lo = np.array([axis[0] for axis in axes], dtype=np.float64)
    hi = np.array([axis[-1] for axis in axes], dtype=np.float64)
    return axes, lo, hi


def obb_lines(frame: LocalFrame, h: float):
    """Wire box around the ACTUAL isotropic local grid."""
    _, lo, hi = actual_grid_local_bounds(frame, h)
    corners_local = np.array([[x, y, z] for x in [lo[0], hi[0]]
                                        for y in [lo[1], hi[1]]
                                        for z in [lo[2], hi[2]]], dtype=np.float64)
    p = frame.world_from_local(corners_local)
    bits = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(bits[i] != bits[j]) == 1:
                edges.append([i, j])
    return p, np.asarray(edges, np.int32)


def local_grid_lines(frame: LocalFrame, h: float):
    """Visualize the actual temporary 3-D grid used for SDF evaluation.

    Previous revisions only drew one XY slice at ``zmid`` and therefore could
    not show that the W/Z pitch was equal to U/V.  This version draws the full
    three families of lattice lines.  Every line position comes directly from
    ``local_grid_axes()``, the same arrays consumed by ``make_local_grid()``.
    """
    xu, yv, zw = local_grid_axes(frame, h)
    points = []
    lines = []

    def add_local(a, b):
        world = frame.world_from_local(np.asarray([a, b], dtype=np.float64))
        i = len(points)
        points.extend([world[0], world[1]])
        lines.append([i, i + 1])

    # U-directed lines: every (V, W) lattice intersection.
    for y in yv:
        for z in zw:
            add_local([xu[0], y, z], [xu[-1], y, z])

    # V-directed lines: every (U, W) lattice intersection.
    for x in xu:
        for z in zw:
            add_local([x, yv[0], z], [x, yv[-1], z])

    # W/Z-directed lines: every (U, V) lattice intersection.
    for x in xu:
        for y in yv:
            add_local([x, y, zw[0]], [x, y, zw[-1]])

    return np.asarray(points, dtype=np.float64), np.asarray(lines, dtype=np.int32)


DEBUG_LEVEL_COLORS = np.array([
    [0.12, 0.82, 1.00],   # base h: cyan
    [0.25, 1.00, 0.45],   # h/2: green
    [1.00, 0.78, 0.15],   # h/4: yellow
    [0.85, 0.40, 1.00],   # h/8+: purple
], dtype=np.float64)


def cell_boxes_lines(frame: LocalFrame, lowers: np.ndarray, h: float, color) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wireframe boxes for selected local cells, matching the grid-debug style."""
    lower = np.asarray(lowers, dtype=np.float64).reshape(-1, 3)
    if not len(lower):
        return (np.empty((0, 3), np.float64), np.empty((0, 2), np.int32),
                np.empty((0, 3), np.float64))
    bits = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                     [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64)
    cube_edges = np.array([[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],
                           [4,6],[5,7],[0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
    local = (lower[:, None, :] + bits[None, :, :] * float(h)).reshape(-1, 3)
    points = frame.world_from_local(local)
    lines = (cube_edges[None, :, :] + np.arange(len(lower))[:, None, None] * 8).reshape(-1, 2)
    colors = np.repeat(np.asarray(color, dtype=np.float64).reshape(1, 3), len(lines), axis=0)
    return points, lines.astype(np.int32, copy=False), colors


def world_aabb_lines(lo: np.ndarray, hi: np.ndarray):
    """Axis-aligned world-space debug box, used for the newest sweep AABB."""
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    bits = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                     [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64)
    edges = np.array([[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],
                      [4,6],[5,7],[0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
    return lo + bits * (hi-lo), edges


def slice_debug_cells(frame: LocalFrame, lowers: np.ndarray, h: float, axis: int | None, fraction: float):
    lower = np.asarray(lowers, dtype=np.float64).reshape(-1, 3)
    if axis is None or not len(lower):
        return lower
    _, lo, hi = actual_grid_local_bounds(frame, h)
    cut = lo[axis] + float(fraction) * (hi[axis] - lo[axis])
    keep = (lower[:, axis] <= cut + 1e-10) & (lower[:, axis] + float(h) >= cut - 1e-10)
    return lower[keep]


class _SliderUpdate:
    """Keep only the latest edit; commit after 300 ms without input."""
    def __init__(self,initial,minimum,maximum):
        self.minimum=float(minimum); self.maximum=float(maximum)
        self.reset(initial)

    def reset(self,value):
        self.applied=float(np.clip(value,self.minimum,self.maximum))
        self.pending=None; self.deadline=0.0
        return self.applied

    def request(self,value):
        value=float(np.clip(value,self.minimum,self.maximum))
        self.pending=value if value!=self.applied else None
        self.deadline=time.perf_counter()+0.30
        return value

    def take(self,now=None):
        if now is None: now=time.perf_counter()
        if self.pending is None or now<self.deadline: return None
        self.applied=self.pending; self.pending=None
        return self.applied


class _DisplaySizeUpdate(_SliderUpdate):
    def __init__(self,initial=0.75):
        super().__init__(initial,0.15,1.00)


def _take_slider_updates(*updates):
    """Wait for all edited sliders to settle, then consume one redraw batch."""
    now=time.perf_counter()
    if any(u.pending is not None and now<u.deadline for u in updates):
        return (None,)*len(updates)
    return tuple(u.take(now) for u in updates)
