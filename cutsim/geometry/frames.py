"""GaussianCutSim geometry / frames implementation."""
from __future__ import annotations

import numpy as np


def tangent_frames(normals: np.ndarray):
    """Same tangent-frame convention as the original uploaded file."""
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    helper = np.zeros_like(n)
    helper[:, 2] = 1.0
    vertical = np.abs(n[:, 2]) > 0.90
    helper[vertical] = [1.0, 0.0, 0.0]
    u = np.cross(n, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1e-30)
    v = np.cross(n, u)
    return u, v


def rotation_matrices_to_quaternions(m: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to quaternions with vectorized trace/diagonal branches.

    Array operations avoid a Python loop over splats during GUI redraws.
    """
    m = np.asarray(m, dtype=np.float64).reshape(-1, 3, 3)
    count = len(m)
    q = np.empty((count, 4), dtype=np.float64)
    if not count:
        return q

    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    pos = tr > 0.0
    if np.any(pos):
        r = m[pos]
        ss = np.sqrt(np.maximum(tr[pos] + 1.0, 0.0)) * 2.0
        ss = np.maximum(ss, 1e-30)
        q[pos, 0] = ss * 0.25
        q[pos, 1] = (r[:, 2, 1] - r[:, 1, 2]) / ss
        q[pos, 2] = (r[:, 0, 2] - r[:, 2, 0]) / ss
        q[pos, 3] = (r[:, 1, 0] - r[:, 0, 1]) / ss

    neg_ids = np.flatnonzero(~pos)
    if len(neg_ids):
        r = m[neg_ids]
        diag = np.stack([r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], axis=1)
        axes = np.argmax(diag, axis=1)
        for axis in range(3):
            take_local = np.flatnonzero(axes == axis)
            if not len(take_local):
                continue
            ids = neg_ids[take_local]
            rr = m[ids]
            j, k = (axis + 1) % 3, (axis + 2) % 3
            ss = np.sqrt(np.maximum(1.0 + rr[:, axis, axis] -
                                    rr[:, j, j] - rr[:, k, k], 0.0)) * 2.0
            ss = np.maximum(ss, 1e-30)
            q[ids] = 0.0
            q[ids, 0] = (rr[:, k, j] - rr[:, j, k]) / ss
            q[ids, axis + 1] = ss * 0.25
            q[ids, j + 1] = (rr[:, j, axis] + rr[:, axis, j]) / ss
            q[ids, k + 1] = (rr[:, k, axis] + rr[:, axis, k]) / ss

    q /= np.maximum(np.linalg.norm(q, axis=1)[:, None], 1e-30)
    return q
