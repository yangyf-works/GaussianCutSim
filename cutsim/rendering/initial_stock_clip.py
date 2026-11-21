"""Clipping certificates assigned directly to generated axis-aligned box faces."""
from time import perf_counter
import numpy as np

from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip


def box_face_clipping(positions, u, v, sigma, low, high):
    """Encode known rectangle edges; no surface search or cutter intersection.

    The box generator supplies signed coordinate axes as its tangent frame.
    Each row therefore has exactly four known edge distances. Keep the two
    nearest edges, with a radius limit for unusually narrow/coarsely sampled
    faces where a third edge also crosses the support disk.
    """
    started = perf_counter()
    p, low, high = np.asarray(positions), np.asarray(low), np.asarray(high)
    n = len(p)
    rows = np.arange(n)
    axes = np.column_stack((np.argmax(abs(u), axis=1), np.argmax(abs(v), axis=1)))
    signs = np.column_stack((u[rows, axes[:, 0]], v[rows, axes[:, 1]]))
    coordinate = np.take_along_axis(p, axes, axis=1)
    lower, upper = coordinate-low[axes], high[axes]-coordinate
    positive = np.where(signs > 0., lower, upper)
    negative = np.where(signs > 0., upper, lower)
    distances = np.column_stack((positive[:, 0], negative[:, 0], positive[:, 1], negative[:, 1]))
    order = np.argsort(distances, axis=1, kind='stable')
    ordered = np.take_along_axis(distances, order, axis=1)
    radius = CLIP_SIGMA*np.asarray(sigma, float)
    # Preserve the existing allowance for FP32 position/quaternion uploads.
    allowance = 8*np.finfo(np.float32).eps*(1+np.max(abs(p), axis=1)+radius)
    directions = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    planes = np.zeros((n, 7))
    planes[:, [2, 5, 6]] = np.inf
    for slot in (0, 1):
        distance = ordered[:, slot]-allowance
        active = distance <= radius
        planes[active, 3*slot:3*slot+2] = directions[order[active, slot]]
        planes[active, 3*slot+2] = distance[active]
    third = ordered[:, 2]
    limited = third < radius-1e-12
    planes[limited, 6] = np.maximum(0., third[limited]-allowance[limited])
    planes.flags.writeable = False
    stats = dict(backend='cpu', mode='stock-direct', primitives=0, history_count=0,
                 input_gs=n, recomputed_gs=n, reused_gs=0, selection_ms=0.,
                 clipped_gs=int(np.count_nonzero(planes[:, 2]<radius)),
                 radius_limited_gs=int(np.count_nonzero(limited)),
                 stock_bounds=tuple(np.r_[low, high].tolist()),
                 pack_ms=0., kernel_ms=None, total_ms=(perf_counter()-started)*1000.)
    return ProfileClip(planes, stats)
