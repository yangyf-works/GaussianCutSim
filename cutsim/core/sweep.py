"""GaussianCutSim core / sweep implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import List, Tuple
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("zero-length vector")
    return v / n


def box_sdf(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact SDF of axis-aligned stock box. Negative = material."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = (lo + hi) * 0.5
    h = (hi - lo) * 0.5
    q = np.abs(p - c) - h
    outside = np.maximum(q, 0.0)
    return np.linalg.norm(outside, axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def capsule2d_parameter(xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ab = b - a
    den = max(float(ab @ ab), 1e-30)
    d = xy - a
    t = np.clip((d @ ab) / den, 0.0, 1.0)
    nearest = a + t[:, None] * ab
    return t, nearest


def segment3d_parameter(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest parameter/point on a 3-D line segment."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    ab = b - a
    den = max(float(ab @ ab), 1e-30)
    t = np.clip(((p - a) @ ab) / den, 0.0, 1.0)
    nearest = a + t[:, None] * ab
    return t, nearest


@dataclass(frozen=True)
class FlatSweep:
    """Vertical flat-end-mill sweep.

    The cutter volume is a 2-D capsule in XY extruded between z_lo and z_hi.
    That exactly models a vertical cylindrical cutter translating from A to B.
    """
    a_xy: np.ndarray
    b_xy: np.ndarray
    radius: float
    z_lo: float
    z_hi: float
    tool_id: int = 1

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, nearest = capsule2d_parameter(p[:, :2], self.a_xy, self.b_xy)
        radial = np.linalg.norm(p[:, :2] - nearest, axis=1) - self.radius
        zc = (self.z_lo + self.z_hi) * 0.5
        zh = (self.z_hi - self.z_lo) * 0.5
        axial = np.abs(p[:, 2] - zc) - zh
        x = np.maximum(radial, 0.0)
        y = np.maximum(axial, 0.0)
        return np.hypot(x, y) + np.minimum(np.maximum(radial, axial), 0.0)

    def sdf_gradient(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return cutter SDF and its outward gradient.

        This is the analytic counterpart of cylinder_sweep_field(..., with_normals=True)
        used by the reference implementation.  The sweep here is a vertical
        capsule prism, so its radial and axial gradients can be evaluated directly.
        """
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, nearest = capsule2d_parameter(p[:, :2], self.a_xy, self.b_xy)
        radial_vec = p[:, :2] - nearest
        radial_len = np.linalg.norm(radial_vec, axis=1)
        radial = radial_len - self.radius

        # Valid subgradient on the capsule centerline.  Choose a perpendicular
        # to the sweep direction; for a zero-length move, use +X.
        move = np.asarray(self.b_xy - self.a_xy, dtype=np.float64)
        ml = float(np.linalg.norm(move))
        fallback = np.array([1.0, 0.0]) if ml < 1e-15 else np.array([-move[1], move[0]]) / ml
        gr_xy = np.empty_like(radial_vec)
        good = radial_len > 1e-14
        gr_xy[good] = radial_vec[good] / radial_len[good, None]
        gr_xy[~good] = fallback
        gr = np.column_stack([gr_xy, np.zeros(len(p))])

        zc = (self.z_lo + self.z_hi) * 0.5
        zh = (self.z_hi - self.z_lo) * 0.5
        dz = p[:, 2] - zc
        axial = np.abs(dz) - zh
        gz = np.zeros_like(p)
        gz[:, 2] = np.where(dz >= 0.0, 1.0, -1.0)

        xr = np.maximum(radial, 0.0)
        xz = np.maximum(axial, 0.0)
        outside_len = np.hypot(xr, xz)
        value = outside_len + np.minimum(np.maximum(radial, axial), 0.0)

        grad = np.empty_like(p)
        outside = outside_len > 1e-12
        if np.any(outside):
            grad[outside] = (xr[outside, None] * gr[outside] +
                             xz[outside, None] * gz[outside]) / outside_len[outside, None]
        inside = ~outside
        if np.any(inside):
            choose_radial = radial[inside] >= axial[inside]
            ids = np.flatnonzero(inside)
            grad[ids[choose_radial]] = gr[ids[choose_radial]]
            grad[ids[~choose_radial]] = gz[ids[~choose_radial]]
        grad /= np.maximum(np.linalg.norm(grad, axis=1)[:, None], 1e-30)
        return value, grad

    @property
    def kind(self) -> str:
        return "flat"

    @property
    def move_length(self) -> float:
        return float(np.linalg.norm(self.b_xy - self.a_xy))

    @property
    def diameter(self) -> float:
        return 2.0 * float(self.radius)

    def face_candidates(self, points: np.ndarray, tol: float):
        """Incident cutter faces as material-outward normals."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        sd = self.sdf(p)
        active = np.abs(sd) <= tol * 1.5
        _, nearest = capsule2d_parameter(p[:, :2], self.a_xy, self.b_xy)
        rv = p[:, :2] - nearest
        rl = np.linalg.norm(rv, axis=1)
        radial = rl - self.radius
        in_z = (p[:, 2] >= self.z_lo-tol) & (p[:, 2] <= self.z_hi+tol)
        out = []

        side = active & (np.abs(radial) <= tol) & in_z & (rl > 1e-14)
        ids = np.flatnonzero(side)
        if len(ids):
            nn = np.zeros((len(ids), 3), dtype=np.float64)
            nn[:, :2] = -rv[ids] / rl[ids, None]
            out.append((ids, nn))

        bottom = active & (radial <= tol) & (np.abs(p[:, 2]-self.z_lo) <= tol)
        top = active & (radial <= tol) & (np.abs(p[:, 2]-self.z_hi) <= tol)
        out.append((np.flatnonzero(bottom), np.array([0.0, 0.0, 1.0])))
        out.append((np.flatnonzero(top), np.array([0.0, 0.0, -1.0])))
        return out

    def world_aabb(self, pad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([
            min(self.a_xy[0], self.b_xy[0]) - self.radius - pad,
            min(self.a_xy[1], self.b_xy[1]) - self.radius - pad,
            self.z_lo - pad,
        ], dtype=np.float64)
        hi = np.array([
            max(self.a_xy[0], self.b_xy[0]) + self.radius + pad,
            max(self.a_xy[1], self.b_xy[1]) + self.radius + pad,
            self.z_hi + pad,
        ], dtype=np.float64)
        return lo, hi


@dataclass(frozen=True)
class BallSweep:
    """Vertical ball-end-mill sweep along one XY line segment.

    ``tip_z`` is the lowest point of the ball.  The swept cutter volume is the
    union of a 3-D capsule centered at ``tip_z + radius`` and the vertical
    cylindrical sweep above the ball equator.  This is an exact representation
    for a vertically oriented ball-end mill translating along a straight XY move.
    """
    a_xy: np.ndarray
    b_xy: np.ndarray
    radius: float
    tip_z: float
    z_hi: float
    tool_id: int = 1

    @property
    def kind(self) -> str:
        return "ball"

    @property
    def z_lo(self) -> float:
        return float(self.tip_z)

    @property
    def center_z(self) -> float:
        return float(self.tip_z + self.radius)

    @property
    def move_length(self) -> float:
        return float(np.linalg.norm(self.b_xy - self.a_xy))

    @property
    def diameter(self) -> float:
        return 2.0 * float(self.radius)

    def _ball_capsule_sdf_gradient(self, points: np.ndarray):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        a = np.array([self.a_xy[0], self.a_xy[1], self.center_z], dtype=np.float64)
        b = np.array([self.b_xy[0], self.b_xy[1], self.center_z], dtype=np.float64)
        _, nearest = segment3d_parameter(p, a, b)
        rv = p - nearest
        rl = np.linalg.norm(rv, axis=1)
        d = rl - self.radius
        g = np.empty_like(p)
        good = rl > 1e-14
        g[good] = rv[good] / rl[good, None]
        # Degenerate centerline points are interior; a stable downward fallback
        # is sufficient until another component becomes the active union surface.
        g[~good] = np.array([0.0, 0.0, -1.0])
        return d, g

    def _upper_sweep(self) -> FlatSweep:
        return FlatSweep(self.a_xy, self.b_xy, self.radius, self.center_z, self.z_hi, self.tool_id)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        # Root finding and grid classification only need distance. Do not build
        # and normalize a full gradient array that would immediately be discarded.
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        a = np.array([self.a_xy[0], self.a_xy[1], self.center_z], dtype=np.float64)
        b = np.array([self.b_xy[0], self.b_xy[1], self.center_z], dtype=np.float64)
        _, nearest = segment3d_parameter(p, a, b)
        d_ball = np.linalg.norm(p-nearest, axis=1)-self.radius
        d_upper = self._upper_sweep().sdf(points)
        return np.minimum(d_ball, d_upper)

    def sdf_gradient(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        d_ball, g_ball = self._ball_capsule_sdf_gradient(p)
        d_upper, g_upper = self._upper_sweep().sdf_gradient(p)
        use_ball = d_ball <= d_upper
        d = np.where(use_ball, d_ball, d_upper)
        g = np.where(use_ball[:, None], g_ball, g_upper)
        g /= np.maximum(np.linalg.norm(g, axis=1)[:, None], 1e-30)
        return d, g

    def face_candidates(self, points: np.ndarray, tol: float):
        """Exposed ball/cylinder faces as material-outward normals."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        active = np.abs(self.sdf(p)) <= tol * 1.5
        out = []

        # Rounded lower ball sweep (3-D capsule).  Above the equator this
        # capsule surface is buried inside the vertical cylinder and is omitted.
        a = np.array([self.a_xy[0], self.a_xy[1], self.center_z], dtype=np.float64)
        b = np.array([self.b_xy[0], self.b_xy[1], self.center_z], dtype=np.float64)
        _, nearest3 = segment3d_parameter(p, a, b)
        rv3 = p - nearest3
        rl3 = np.linalg.norm(rv3, axis=1)
        rounded = (active & (np.abs(rl3-self.radius) <= tol) &
                   (p[:, 2] <= self.center_z + tol) & (rl3 > 1e-14))
        ids = np.flatnonzero(rounded)
        if len(ids):
            nn = -rv3[ids] / rl3[ids, None]
            out.append((ids, nn))

        # Vertical side above the ball equator.
        _, nearest2 = capsule2d_parameter(p[:, :2], self.a_xy, self.b_xy)
        rv2 = p[:, :2] - nearest2
        rl2 = np.linalg.norm(rv2, axis=1)
        radial = rl2 - self.radius
        side = (active & (np.abs(radial) <= tol) &
                (p[:, 2] >= self.center_z-tol) & (p[:, 2] <= self.z_hi+tol) &
                (rl2 > 1e-14))
        ids = np.flatnonzero(side)
        if len(ids):
            nn = np.zeros((len(ids), 3), dtype=np.float64)
            nn[:, :2] = -rv2[ids] / rl2[ids, None]
            out.append((ids, nn))

        top = active & (radial <= tol) & (np.abs(p[:, 2]-self.z_hi) <= tol)
        out.append((np.flatnonzero(top), np.array([0.0, 0.0, -1.0])))
        return out

    def world_aabb(self, pad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([
            min(self.a_xy[0], self.b_xy[0]) - self.radius - pad,
            min(self.a_xy[1], self.b_xy[1]) - self.radius - pad,
            self.tip_z - pad,
        ], dtype=np.float64)
        hi = np.array([
            max(self.a_xy[0], self.b_xy[0]) + self.radius + pad,
            max(self.a_xy[1], self.b_xy[1]) + self.radius + pad,
            self.z_hi + pad,
        ], dtype=np.float64)
        return lo, hi
