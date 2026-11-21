"""Bounded, adaptive horizontal dexel queries; never a replacement SDF.

Lines run along the current straight sweep. Only occupied cross-section bins
are visited (4h, 2h, h). Analytic expanded/eroded cutter intervals certify an
off-line predicate using the 1-Lipschitz bound. Uncertainty, small batches and
budget exhaustion go to the original evaluator, not to approximate geometry.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import numpy as np

from cutsim.core.material import AnalyticMaterial
from cutsim.core.sweep import BallSweep, FlatSweep, box_sdf


EMPTY = (math.inf, -math.inf)


def _slab(offset, direction, low, high):
    if low > high:
        return EMPTY
    if direction == 0.:
        return (-math.inf, math.inf) if low <= offset <= high else EMPTY
    a, b = (low-offset)/direction, (high-offset)/direction
    return min(a, b), max(a, b)


def _capsule_line(origin, direction, a, b, radius):
    """Intersection with a 2-D capsule, including parallel/stationary cases."""
    if radius < 0. or not math.isfinite(radius):
        return EMPTY
    dx, dy = direction[:2]
    ox, oy = origin[:2]
    dd = dx*dx+dy*dy
    intervals = []
    for c in (a, b):
        rx, ry = ox-c[0], oy-c[1]
        cross = rx*dy-ry*dx
        distance2 = cross*cross/dd
        discriminant = radius*radius-distance2
        if discriminant >= 0.:
            mid = -(rx*dx+ry*dy)/dd
            half = math.sqrt(discriminant/dd)
            intervals.append((mid-half, mid+half))
    vx, vy = b-a
    length = math.hypot(vx, vy)
    if length > 0.:
        ux, uy = vx/length, vy/length
        rx, ry = ox-a[0], oy-a[1]
        along = _slab(rx*ux+ry*uy, dx*ux+dy*uy, 0., length)
        across = _slab(-rx*uy+ry*ux, -dx*uy+dy*ux, -radius, radius)
        lo, hi = max(along[0], across[0]), min(along[1], across[1])
        if lo <= hi:
            intervals.append((lo, hi))
    if not intervals:
        return EMPTY
    # A capsule is convex; the primitive pieces form one interval on a line.
    return min(x[0] for x in intervals), max(x[1] for x in intervals)


def sweep_line_interval(sweep, origin, direction, level=0.):
    """All t with sweep.sdf(origin+t*direction) <= level (horizontal line).

    FlatSweep's rounded positive offsets and BallSweep's complete capsule plus
    upper cylinder follow the actual SDF, including short shanks and zero moves.
    """
    if hasattr(sweep, 'line_interval'):
        return sweep.line_interval(origin, direction, level)
    if type(sweep) not in (FlatSweep, BallSweep):
        raise TypeError('unsupported analytic sweep')
    if direction[2] != 0. or np.dot(direction[:2], direction[:2]) <= 0.:
        raise ValueError('a nonzero horizontal line is required')
    zlo = sweep.center_z if type(sweep) is BallSweep else sweep.z_lo
    axial = abs(origin[2]-(zlo+sweep.z_hi)*.5)-(sweep.z_hi-zlo)*.5
    radius = -1.
    if axial <= level:
        radial_level = (math.sqrt(max(0., level*level-max(axial, 0.)**2))
                        if level >= 0. else level)
        radius = sweep.radius+radial_level
    if type(sweep) is BallSweep:
        r = sweep.radius+level
        dz = abs(origin[2]-sweep.center_z)
        if r >= dz:
            radius = max(radius, math.sqrt(max(0., (r-dz)*(r+dz))))
    return _capsule_line(origin, direction, sweep.a_xy, sweep.b_xy, radius)


def stock_line_interval(material, origin, direction):
    lo, hi = -math.inf, math.inf
    for j in range(3):
        a, b = _slab(origin[j], direction[j], material.stock_lo[j], material.stock_hi[j])
        lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if lo <= hi else EMPTY


@dataclass(frozen=True)
class LineSpan:
    start: float
    end: float
    start_source: int   # -1: stock, >=0: historical sweep index
    end_source: int


@dataclass
class LineRecord:
    origin: np.ndarray
    direction: np.ndarray
    ids: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    nominal: np.ndarray
    stock: tuple

    def remaining(self):
        """Nominal material intervals with boundary provenance, in line order."""
        lo, hi = self.stock
        spans = [LineSpan(lo, hi, -1, -1)] if lo < hi else []
        for sid, (a, b) in zip(self.ids, self.nominal):
            if a >= b:  # isolated tangency removes no volume
                continue
            result = []
            for s in spans:
                if b <= s.start or a >= s.end:
                    result.append(s)
                else:
                    if a > s.start:
                        result.append(LineSpan(s.start, a, s.start_source, int(sid)))
                    if b < s.end:
                        result.append(LineSpan(b, s.end, int(sid), s.end_source))
            spans = result
        return spans

    def boundary(self, material, t, source):
        """Return the shared geometric endpoint and material-outward normal.

        This is a search hint, NOT a clipping plane for a neighbouring GS.
        At a multi-face seam it selects one incident normal; GS clipping keeps
        its existing two-edge construction.
        """
        p = self.origin+t*self.direction
        if source >= 0:
            normal = -material.sweeps[source].sdf_gradient(p[None])[1][0]
        else:
            distances = np.r_[np.abs(p-material.stock_lo), np.abs(p-material.stock_hi)]
            face = int(np.argmin(distances))
            normal = np.zeros(3); normal[face % 3] = -1. if face < 3 else 1.
        return p, normal

    @property
    def bytes(self):
        return sum(a.nbytes for a in (self.origin, self.direction, self.ids,
                                      self.inner, self.outer, self.nominal))+1024


class AdaptiveSweepLines:
    """A block-local bounded cache and a per-query subdivision budget."""
    def __init__(self, frame, origin, pitch, stats):
        self.frame = frame
        self.origin = origin
        self.pitch = max(float(pitch), 2.5e-6)
        self.stats = stats
        self.min_points = 48
        self.max_lines = 4096
        self.max_cache_bytes = 16*1024*1024
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.history = None
        self.last = {}
        for name in ('queries', 'points', 'lines', 'cache_hits', 'splits', 'cost_stops',
                     'budget_stops', 'certified', 'fallback', 'level_0', 'level_1', 'level_2',
                     'max_per_query', 'max_budget', 'max_depth', 'peak_cache_bytes'):
            stats['line_'+name] = 0

    def _history(self, material):
        token = (material.stock_lo.tobytes(), material.stock_hi.tobytes(),
                 material._sweep_index.digests[-1] if len(material.sweeps) else b'')
        if token != self.history:
            self.cache.clear(); self.cache_bytes = 0; self.history = token
        while self.cache and (self.cache_bytes > self.max_cache_bytes or
                              len(self.cache) > max(0, self.max_lines)):
            _, old = self.cache.popitem(last=False); self.cache_bytes -= old.bytes

    def record(self, material, key, width, low, high):
        """Share exact ordered line/cutter intersections within one history."""
        self._history(material)
        cache_key = (tuple(key), float(width), float(low), float(high))
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            self.stats['line_cache_hits'] += 1
            return self.cache[cache_key]
        local = np.r_[0., (np.asarray(key)+.5)*width]
        origin = self.origin+local @ self.frame
        direction = self.frame[0]
        scale = 1.+max(np.max(np.abs(material.stock_lo)), np.max(np.abs(material.stock_hi)),
                       np.max(np.abs(origin)), abs(low), abs(high),
                       material._sweep_index.scales[-1] if len(material.sweeps) else 0.)
        guard = 1e-10*scale
        rho = math.sqrt(2.)*width*.5+guard
        # Query a conservative tube across the entire stock, never just its
        # endpoint signs. The tree bounds include short-shank ball capsules.
        centre = (material.stock_lo+material.stock_hi)*.5
        extent = (material.stock_hi-material.stock_lo)*.5
        xm = float((centre-origin) @ direction)
        xr = float(np.abs(direction) @ extent)+max(abs(low), abs(high))+rho
        ends = origin+np.array([xm-xr, xm+xr])[:, None]*direction
        pad = rho+max(abs(low), abs(high))
        ids = material.query_sweep_ids(ends.min(axis=0)-pad, ends.max(axis=0)+pad,
                                       original_boxes=False)
        inner, outer, nominal = [], [], []
        for sid in ids:
            sw = material.sweeps[sid]
            inner.append(sweep_line_interval(sw, origin, direction, -high-rho))
            outer.append(sweep_line_interval(sw, origin, direction, -low+rho))
            nominal.append(sweep_line_interval(sw, origin, direction))
        result = LineRecord(origin, direction.copy(), np.asarray(ids, dtype=np.int64),
                            np.asarray(inner).reshape(-1, 2), np.asarray(outer).reshape(-1, 2),
                            np.asarray(nominal).reshape(-1, 2),
                            stock_line_interval(material, origin, direction))
        size = result.bytes
        if size <= self.max_cache_bytes and self.max_lines > 0:
            while self.cache and (self.cache_bytes+size > self.max_cache_bytes or
                                  len(self.cache) >= self.max_lines):
                _, old = self.cache.popitem(last=False); self.cache_bytes -= old.bytes
            self.cache[cache_key] = result; self.cache_bytes += size
            self.stats['line_peak_cache_bytes'] = max(self.stats['line_peak_cache_bytes'], self.cache_bytes)
        return result

    @staticmethod
    def _groups(local, rows, width):
        keys, inv, counts = np.unique(np.floor(local[rows, 1:]/width).astype(np.int64),
                                      axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(counts)]
        # LIFO processing visits the most populated bands first.
        return [(keys[i], rows[order[starts[i]:starts[i+1]]]) for i in np.argsort(counts)]

    def classify(self, material, points, low, high, fallback, closed=True, return_validity=False):
        """Return below/above flags, identical to comparisons with the old SDF.

        closed=True: below f<low, above f>high. False uses <= and >=.
        Strict certificates apply to both; boundary equalities use fallback.
        """
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        low = np.broadcast_to(np.asarray(low, dtype=float), (len(p),))
        high = np.broadcast_to(np.asarray(high, dtype=float), (len(p),))
        if np.any(low > high):
            raise ValueError('low must not exceed high')
        self.stats['line_queries'] += 1; self.stats['line_points'] += len(p)
        supported = (type(material) is AnalyticMaterial and
                     (not len(material.sweeps) or material._sweep_index.supported[-1]))
        good = (supported and np.all(np.isfinite(p)) and np.all(np.isfinite(low)) and
                np.all(np.isfinite(high)))
        below = np.zeros(len(p), dtype=bool); above = below.copy(); done = below.copy()
        valid = np.ones(len(p), dtype=bool)
        tested = 0; budget = 0; deepest = -1
        if good and len(p):
            stock = box_sdf(p, material.stock_lo, material.stock_hi)
            # Stock alone can certify outside. Equalities remain conservative.
            above = stock > high; done |= above
            rows = np.flatnonzero(~done)
            local = (p-self.origin) @ self.frame.T
            fine_count = len(np.unique(np.floor(local[rows, 1:]/self.pitch).astype(np.int64), axis=0))
            budget = min(fine_count, max(0, self.max_lines))
            stack = [(key, ids, 0) for key, ids in self._groups(local, rows, 4.*self.pitch)]
            while stack:
                key, rows, level = stack.pop()
                if tested >= budget:
                    self.stats['line_budget_stops'] += len(stack)+1
                    break
                # Three interval solves per cutter plus Python/tree overhead
                # need enough original predicates to amortize a new line.
                if len(rows) < self.min_points:
                    self.stats['line_cost_stops'] += 1
                    continue
                width = self.pitch*(4 >> level)
                record = self.record(material, key, width, low[rows].min(), high[rows].max())
                tested += 1; deepest = max(deepest, level)
                self.stats['line_level_'+str(level)] += 1
                x = local[rows, 0]
                # Origin differs only in the two transverse coordinates.
                possible = np.zeros(len(rows), dtype=bool)
                cut = possible.copy()
                for a, b in record.outer:
                    possible |= (x >= a) & (x <= b)
                for a, b in record.inner:
                    cut |= (x > a) & (x < b)
                above[rows[cut]] = True
                safe = ~possible
                below[rows[safe]] = stock[rows[safe]] < low[rows[safe]]
                resolved = cut | safe
                # For open bands, exact stock equality must use original <=/>=.
                if not closed:
                    resolved &= (stock[rows] != low[rows]) & (stock[rows] != high[rows])
                done[rows[resolved]] = True
                unknown = rows[~resolved]
                if len(unknown) and level < 2:
                    children = self._groups(local, unknown, width*.5)
                    eligible = [(k, ids, level+1) for k, ids in children if len(ids) >= self.min_points]
                    self.stats['line_cost_stops'] += len(children)-len(eligible)
                    if eligible:
                        self.stats['line_splits'] += 1
                        stack.extend(eligible)
            self.stats['line_certified'] += int(done.sum())
        unresolved = np.flatnonzero(~done)
        if len(unresolved):
            values = fallback(p[unresolved])
            valid[unresolved] = ~np.isnan(values)
            below[unresolved] = values < low[unresolved] if closed else values <= low[unresolved]
            above[unresolved] = values > high[unresolved] if closed else values >= high[unresolved]
        self.stats['line_fallback'] += len(unresolved)
        self.stats['line_lines'] += tested
        self.stats['line_max_per_query'] = max(self.stats['line_max_per_query'], tested)
        self.stats['line_max_budget'] = max(self.stats['line_max_budget'], budget)
        self.stats['line_max_depth'] = max(self.stats['line_max_depth'], deepest)
        self.last = dict(points=len(p), lines=tested, budget=budget, deepest=deepest,
                         fallback=len(unresolved), certified=len(p)-len(unresolved))
        return (below, above, valid) if return_validity else (below, above)
