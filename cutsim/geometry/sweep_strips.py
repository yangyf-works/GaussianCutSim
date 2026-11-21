"""Sweep-aligned region work without moving or resampling any GS candidate.

Cross-section bins form longitudinal strips, subdivided along travel for local
bounds. Bounds only prune work; every admitted query keeps its original point
and analytic predicate. No interval-endpoint interpolation replaces a query.
"""
from collections import OrderedDict
import numpy as np

from cutsim.core.sweep import BallSweep, FlatSweep, box_sdf
from cutsim.geometry.sweep_lines import AdaptiveSweepLines


def sweep_key(sweep):
    def array_key(value):
        a = np.asarray(value)
        return a.dtype.str, a.shape, a.tobytes()
    if hasattr(sweep, 'a_xyz'):
        return (type(sweep).__name__, sweep.kind, array_key(sweep.a_xyz), array_key(sweep.b_xyz),
                float(sweep.radius), float(sweep.z_hi))
    return (type(sweep).__name__, array_key(sweep.a_xy), array_key(sweep.b_xy),
            float(sweep.radius), float(sweep.z_lo), float(sweep.z_hi),
            float(sweep.center_z) if isinstance(sweep, BallSweep) else None)


class SweepStripWork:
    """One block's strip partitions and shared, exact curve boundary results."""
    def __init__(self, sweep, pitch, max_curve_entries=32768):
        move = sweep.b_xy-sweep.a_xy
        length = np.linalg.norm(move)
        u = move/length if length > 1e-12 else np.array([1., 0.])
        self.frame = np.array([[u[0], u[1], 0.], [-u[1], u[0], 0.], [0., 0., 1.]])
        self.origin = np.r_[sweep.a_xy, sweep.z_lo]
        self.width = max(4.*float(pitch), 1e-5)
        self.spacing = np.array([8.*self.width, self.width, self.width])
        self.curves = OrderedDict()
        self.max_curve_entries = int(max_curve_entries)
        self.curve_bytes = 0
        self.max_curve_bytes = 32*1024*1024
        self.stats = dict(region_queries=0, region_points=0, region_strips=0,
                          region_possible_pairs=0, region_evaluated_pairs=0,
                          domain_point_queries=0, domain_strip_queries=0,
                          curve_hits=0, curve_misses=0)
        self.lines = AdaptiveSweepLines(self.frame, self.origin, pitch, self.stats)

    def partition(self, points):
        p = np.asarray(points, float).reshape(-1, 3)
        # Quantization is ONLY a work partition; geometry and candidate order
        # remain the caller's original floating-point arrays.
        local = (p-self.origin) @ self.frame.T
        keys = np.floor(local/self.spacing).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(inverse[order]))+1]
        return inverse, order, starts

    def value(self, material, points, upto=None):
        """Exact CSG values, with one conservative region test per strip.

        A 1-Lipschitz bound excludes a sweep only if it cannot exceed the stock
        field anywhere in a strip's actual point bounding box. Near-zero values
        use the original evaluator, retaining touching/signed-zero behaviour.
        """
        p = np.asarray(points, float).reshape(-1, 3)
        limit = len(material.sweeps) if upto is None else max(0, min(int(upto), len(material.sweeps)))
        if not len(p) or not limit or not material._sweep_index.supported[limit-1]:
            return material.value(p, upto=upto)
        inverse, order, starts = self.partition(p)
        lo = np.minimum.reduceat(p[order], starts, axis=0)
        hi = np.maximum.reduceat(p[order], starts, axis=0)
        centre = (lo+hi)*.5
        radius = np.linalg.norm((hi-lo)*.5, axis=1)
        stock_lower = box_sdf(centre, material.stock_lo, material.stock_hi)-radius
        guard = 256.*np.finfo(float).eps*(1.+np.max(np.abs(centre), axis=1)+radius+np.abs(stock_lower))
        stock = box_sdf(p, material.stock_lo, material.stock_hi)
        cut = -stock.copy()
        evaluated = 0
        for sid in range(limit):
            sweep = material.sweeps[sid]
            lower = sweep.sdf(centre)-radius
            admitted = (-lower >= stock_lower-guard)[inverse]
            rows = np.flatnonzero(admitted)
            if len(rows):
                cut[rows] = np.minimum(cut[rows], sweep.sdf(p[rows]))
                evaluated += len(rows)
        result = np.maximum(stock, -cut)
        near = np.abs(result) <= 1e-12*(1.+np.max(np.abs(p), axis=1))
        if np.any(near):
            result[near] = material.value(p[near], upto=upto)
        self.stats['region_queries'] += 1
        self.stats['region_points'] += len(p)
        self.stats['region_strips'] += len(starts)
        self.stats['region_possible_pairs'] += len(p)*limit
        self.stats['region_evaluated_pairs'] += evaluated + len(centre)*limit
        return result

    def domain_groups(self, material, positions, extent):
        """Share sweep AABB queries per band, then retain exact per-GS boxes."""
        p = np.asarray(positions)
        if not len(p):
            return {}
        _, order, starts = self.partition(p)
        ends = np.r_[starts[1:], len(p)]
        lower, upper = p-extent, p+extent
        groups = {}
        for start, end in zip(starts, ends):
            rows = order[start:end]
            ids = material.query_sweep_ids(lower[rows].min(axis=0), upper[rows].max(axis=0), original_boxes=False)
            for sid in ids:
                lo = material._sweep_index.lo[sid]
                hi = np.array(material._sweep_index.hi[sid], copy=True)
                sw = material.sweeps[sid]
                if type(sw) is BallSweep:
                    hi[2] = max(hi[2], sw.center_z+sw.radius)
                hit = np.all(upper[rows] >= lo, axis=1) & np.all(lower[rows] <= hi, axis=1)
                if np.any(hit):
                    groups.setdefault(sid, []).append(rows[hit])
        self.stats['domain_point_queries'] += len(p)
        self.stats['domain_strip_queries'] += len(starts)
        return {sid: np.sort(np.concatenate(parts)).tolist() for sid, parts in groups.items()}

    def curve_boundaries(self, source, target, centres, normals, sigma, calculate):
        """Share exact roots/planes for identical GS geometry and source pairs.

        Store unsuccessful projections too. Never transfer a plane from a nearby
        candidate or interpolate between strip endpoints; that changes clipping.
        """
        # BLAS may round a three-component dot product differently when its
        # batch size/order changes. Per-row memoization is therefore unsafe for
        # bit-exact persistent domains, even with identical geometric rows.
        geometry = np.column_stack((centres, normals, sigma))
        key = (sweep_key(source), sweep_key(target), geometry.shape, geometry.tobytes())
        record = self.curves.get(key)
        if record is not None:
            self.curves.move_to_end(key)
            self.stats['curve_hits'] += len(geometry)
            return tuple(a.copy() for a in record[0])
        result = calculate(source, target, centres, normals, sigma)
        self.stats['curve_misses'] += len(geometry)
        size = len(key[-1]) + sum(a.nbytes for a in result) + 1024
        if self.max_curve_entries > 0 and size <= self.max_curve_bytes:
            while self.curves and (len(self.curves) >= self.max_curve_entries or
                                  self.curve_bytes+size > self.max_curve_bytes):
                _, (_, old_size) = self.curves.popitem(last=False)
                self.curve_bytes -= old_size
            self.curves[key] = (tuple(a.copy() for a in result), size)
            self.curve_bytes += size
        return result


def strip_material_value(material, points, upto=None):
    work = getattr(material, '_strip_work', None)
    if work is None:
        return material.value(points) if upto is None else material.value(points, upto=upto)
    return work.value(material, points, upto=upto)


def strip_material_range(material, points, low, high, closed=True):
    """Predicate-only fast path. Numeric SDF consumers still use value()."""
    work = getattr(material, '_strip_work', None)
    if work is not None and hasattr(work, 'lines'):
        return work.lines.classify(material, points, low, high,
                                  lambda p: work.value(material, p), closed=closed)
    values = strip_material_value(material, points)
    return ((values < low, values > high) if closed else
            (values <= low, values >= high))


def strip_material_surface_mask(material, points, tolerance=1e-7, closed=True):
    """Retain abs(SDF) <= tolerance (or <), including NaN rejection."""
    work = getattr(material, '_strip_work', None)
    if work is not None and hasattr(work, 'lines'):
        below, above, valid = work.lines.classify(
            material, points, -tolerance, tolerance,
            lambda p: work.value(material, p), closed=closed, return_validity=True)
        return valid & ~(below | above)
    value = np.abs(strip_material_value(material, points))
    return value <= tolerance if closed else value < tolerance
