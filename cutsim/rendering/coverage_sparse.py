"""Finite-support coverage graph: child -> only probes inside its 3-sigma disk.

Probe sets, error thresholds and the ordered greedy rule match dense mode. Zero
entries are implicit; deleting/restoring a child touches only its neighbours.
The x-sorted broad phase is bounded by the same evaluation budget as dense mode.
"""
import os
from dataclasses import dataclass
import numpy as np

from cutsim.geometry.strip_kernels import njit
from cutsim.rendering.coverage_cuda import packed_inputs

MAX_SUPPORT_PAIRS = 2_000_000


class GraphBudgetExceeded(Exception):
    """Use the bounded dense reference instead of growing the sparse graph."""


def algorithm():
    from cutsim.geometry.strip_kernels import numba
    default = ('resident' if os.environ.get('CUTSIM_COVERAGE_BACKEND', 'cpu').lower() == 'cuda'
               else 'sparse' if numba is not None else 'dense')
    value = os.environ.get('CUTSIM_COVERAGE_ALGORITHM', default).lower()
    if value not in ('dense', 'sparse', 'resident'):
        raise ValueError('CUTSIM_COVERAGE_ALGORITHM must be dense, sparse or resident')
    if value == 'resident' and os.environ.get('CUTSIM_COVERAGE_BACKEND', 'cpu').lower() != 'cuda':
        raise ValueError('Resident coverage requires CUTSIM_COVERAGE_BACKEND=cuda')
    return value


@njit(cache=True, nogil=True)
def disk_graph(projected, centres, scale, budget=MAX_SUPPORT_PAIRS):
    """Exact finite-disk filter after conservative x intervals, in child order."""
    order = np.argsort(projected[:, 0])
    xs = projected[order, 0]
    pointers = np.zeros(len(centres)+1, np.int64)
    limits = np.empty((len(centres), 2), np.int64)
    scanned = 0
    for j in range(len(centres)):
        radius = np.nextafter(3.*float(scale[j]), np.inf)
        lower = np.nextafter(centres[j, 0]-radius, -np.inf)
        upper = np.nextafter(centres[j, 0]+radius, np.inf)
        first = np.searchsorted(xs, lower)
        last = np.searchsorted(xs, upper, side='right')
        limits[j, 0], limits[j, 1] = first, last
        scanned += last-first
        count = 0
        for at in range(first, last):
            i = order[at]
            x = (projected[i, 0]-centres[j, 0])/scale[j]
            y = (projected[i, 1]-centres[j, 1])/scale[j]
            if x*x+y*y <= 9.:
                count += 1
        pointers[j+1] = pointers[j]+count
    if pointers[-1] > budget:
        return pointers, np.empty(0, np.int32), scanned
    indices = np.empty(pointers[-1], np.int32)
    for j in range(len(centres)):
        offset = pointers[j]
        for at in range(limits[j, 0], limits[j, 1]):
            i = order[at]
            x = (projected[i, 0]-centres[j, 0])/scale[j]
            y = (projected[i, 1]-centres[j, 1])/scale[j]
            if x*x+y*y <= 9.:
                indices[offset] = i
                offset += 1
        indices[pointers[j]:pointers[j+1]].sort()
    return pointers, indices, scanned


@njit(cache=True, nogil=True)
def pair_geometry(pointers, indices, projected, centres, scale, positions, world32,
                  u32, v32, angles, splits, cosine, sine, cosine32, sine32, dual):
    rho = np.empty(len(indices), np.float64)
    visible = np.empty(len(indices), np.bool_)
    visible32 = np.empty(len(indices), np.bool_)
    for j in range(len(pointers)-1):
        for at in range(pointers[j], pointers[j+1]):
            i = indices[at]
            x = (projected[i, 0]-centres[j, 0])/scale[j]
            y = (projected[i, 1]-centres[j, 1])/scale[j]
            rho[at] = x*x+y*y
            keep = True
            for edge in range(2):
                keep = keep and (angles[j, edge] == 4. or
                    x*cosine[j, edge]+y*sine[j, edge] >= splits[j, edge])
            visible[at] = keep
            keep32 = True
            if dual:
                dx = np.float32(world32[i, 0]-positions[j, 0])
                dy = np.float32(world32[i, 1]-positions[j, 1])
                dz = np.float32(world32[i, 2]-positions[j, 2])
                cx = np.float32(np.float32(np.float32(dx*u32[j, 0])+np.float32(dy*u32[j, 1]))+np.float32(dz*u32[j, 2]))/scale[j]
                cy = np.float32(np.float32(np.float32(dx*v32[j, 0])+np.float32(dy*v32[j, 1]))+np.float32(dz*v32[j, 2]))/scale[j]
                for edge in range(2):
                    value = np.float32(np.float32(cx*cosine32[j, edge])+np.float32(cy*sine32[j, edge]))
                    keep32 = keep32 and (angles[j, edge] == 4. or value >= splits[j, edge])
            visible32[at] = keep32
    return rho, visible, visible32


@njit(cache=True, nogil=True)
def compact(pointers, indices, values, values32, probes, dual):
    lengths = np.zeros(len(pointers), np.int64)
    for j in range(len(pointers)-1):
        count = 0
        for at in range(pointers[j], pointers[j+1]):
            count += int(values[at] > 0.)
            if dual:
                count += int(values32[at] > 0.)
        lengths[j+1] = lengths[j]+count
    points = np.empty(lengths[-1], np.int32)
    weights = np.empty(lengths[-1], np.float32)
    for j in range(len(pointers)-1):
        offset = lengths[j]
        for mode in range(2 if dual else 1):
            data = values32 if mode else values
            for at in range(pointers[j], pointers[j+1]):
                if data[at] > 0.:
                    points[offset] = indices[at]+mode*probes
                    weights[offset] = data[at]
                    offset += 1
    return lengths, points, weights


@dataclass
class CoverageGraph:
    pointers: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    probes: int
    candidates: int
    scanned: int

    @property
    def nbytes(self):
        return self.pointers.nbytes+self.indices.nbytes+self.values.nbytes


def evaluate(draw, points, origin, basis, *, dual=False):
    from cutsim.rendering.coverage_cuda import evaluator
    probes, children = len(points), len(draw['positions'])
    modes = 2 if dual else 1
    if probes*children*modes > 16_000_000:
        raise GraphBudgetExceeded('Dense reference evaluation budget exceeded')
    if not probes or not children:
        return CoverageGraph(np.zeros(children+1, np.int64), np.empty(0, np.int32),
                             np.empty(0, np.float32), probes*modes, 0, 0)
    arrays = packed_inputs(draw, points, origin, basis)
    pointers, indices, scanned = disk_graph(*arrays[:3], MAX_SUPPORT_PAIRS//modes)
    if pointers[-1]*modes > MAX_SUPPORT_PAIRS:
        raise GraphBudgetExceeded('Sparse graph storage budget exceeded')
    gpu = evaluator()
    if gpu is None:
        rho, visible, visible32 = pair_geometry(pointers, indices, *arrays, dual)
        alpha = .99*np.exp(-.5*rho)
        values = (-np.log1p(-alpha*visible)).astype(np.float32)
        values32 = (-np.log1p(-alpha*visible32)).astype(np.float32) if dual else values
    else:
        values, values32 = gpu.evaluate_pairs(arrays, pointers, indices, dual)
    ptr, rows, data = compact(pointers, indices, values, values32, len(points), dual)
    return CoverageGraph(ptr, rows, data, len(points)*(2 if dual else 1),
                         len(indices)*(2 if dual else 1), scanned)


@njit(cache=True, nogil=True)
def row_sums(pointers, indices, values, probes, keep):
    total = np.zeros(probes, np.float64)
    for j in range(len(keep)):
        if keep[j]:
            for at in range(pointers[j], pointers[j+1]):
                total[indices[at]] += values[at]
    return total


@njit(cache=True, nogil=True)
def impacts(pointers, indices, contributions, transmission):
    result = np.zeros(len(pointers)-1)
    for j in range(len(result)):
        for at in range(pointers[j], pointers[j+1]):
            result[j] += transmission[indices[at]]*contributions[at]
    return result


def stable_impacts(graph, total):
    """Refine near-ties with the original NumPy reduction and zero positions.

    Summing only nonzeros changes Float64 rounding, which can swap symmetric
    children in the stable greedy order. The fast positive sum supplies an error
    envelope; only neighbouring scores within that envelope need dense scratch.
    This allocates one probe vector, never a probe x child matrix.
    """
    transmission = np.exp(-total)
    scores = impacts(graph.pointers, graph.indices, np.expm1(graph.values.astype(float)), transmission)
    order = np.argsort(scores, kind='stable')
    # Conservative bound for positive summation and NumPy's different reduction
    # tree. At most 16M input entries are admitted by the caller's budget.
    error = 8.*np.finfo(float).eps*(graph.probes+1)*np.abs(scores)
    adjacent = np.diff(scores[order]) <= error[order[:-1]]+error[order[1:]]
    refine = np.unique(np.r_[order[:-1][adjacent], order[1:][adjacent]])
    if len(refine):
        scratch = np.zeros(graph.probes)
        for j in refine:
            section = slice(graph.pointers[j], graph.pointers[j+1])
            points = graph.indices[section]
            scratch[points] = graph.values[section]
            scores[j] = np.sum(transmission*np.expm1(scratch))
            scratch[points] = 0.
    return scores, len(refine)


@njit(cache=True, nogil=True)
def remove_ordered(pointers, indices, values, total, required, order):
    keep = np.ones(len(order), np.bool_)
    for j in order:
        accepted = True
        for at in range(pointers[j], pointers[j+1]):
            i = indices[at]
            if total[i]-values[at] < required[i]-1e-10:
                accepted = False
                break
        if accepted:
            keep[j] = False
            for at in range(pointers[j], pointers[j+1]):
                total[indices[at]] -= values[at]
    return keep


@njit(cache=True, nogil=True)
def restore_ordered(pointers, indices, values, keep, current, required):
    restored = 0
    while True:
        deficit = required-current
        worst = np.argmax(deficit)
        if deficit[worst] <= 1e-10:
            return restored
        best, score = -1, -1.
        for j in range(len(keep)):
            if keep[j]:
                continue
            start, end = pointers[j], pointers[j+1]
            at = start+np.searchsorted(indices[start:end], worst)
            value = values[at] if at < end and indices[at] == worst else 0.
            if value > score:
                best, score = j, value
        if best == -1:
            return restored
        keep[best] = True
        restored += 1
        for at in range(pointers[best], pointers[best+1]):
            current[indices[at]] += values[at]


def prune(draw, points, boundary, checks, check_boundary, origin, basis, tolerance, stats):
    from cutsim.rendering.clip_fallback import _coverage_target
    count = len(draw['positions'])
    decision = evaluate(draw, points, origin, basis)
    stats.update(coverage_algorithm='sparse', dense_pairs=count*len(points),
                 support_pairs=decision.candidates, nonzero_pairs=len(decision.values),
                 broadphase_pairs=decision.scanned, graph_bytes=decision.nbytes)
    d = decision
    total = row_sums(d.pointers, d.indices, d.values, d.probes, np.ones(count, bool))
    reference = -np.expm1(-total)
    required = -np.log1p(-_coverage_target(reference, boundary, tolerance))
    impact, refined = stable_impacts(d, total)
    stats['order_refined_children'] = refined
    keep = remove_ordered(d.pointers, d.indices, d.values, total, required, np.argsort(impact, kind='stable'))
    if np.all(keep):
        return np.arange(count)
    v = evaluate(draw, checks, origin, basis, dual=True)
    stats['validation_probes'] = 2*len(checks)
    stats['dense_pairs'] += count*v.probes
    stats['support_pairs'] += v.candidates
    stats['nonzero_pairs'] += len(v.values)
    stats['broadphase_pairs'] += v.scanned
    stats['graph_bytes'] += v.nbytes
    check_boundary = np.tile(check_boundary, 2)
    full = row_sums(v.pointers, v.indices, v.values, v.probes, np.ones(count, bool))
    check_reference = -np.expm1(-full)
    check_required = -np.log1p(-_coverage_target(check_reference, check_boundary, tolerance))
    current = row_sums(v.pointers, v.indices, v.values, v.probes, keep)
    stats['restored'] = restore_ordered(v.pointers, v.indices, v.values, keep, current, check_required)
    decision_after = -np.expm1(-row_sums(d.pointers, d.indices, d.values, d.probes, keep))
    check_after = -np.expm1(-current)
    drops = np.concatenate((reference-decision_after, check_reference-check_after))
    edge = np.concatenate((boundary, check_boundary))
    stats.update(max_drop=float(np.max(drops, initial=0.)),
                 boundary_max_drop=float(np.max(drops[edge], initial=0.)),
                 new_holes=int(np.count_nonzero((reference >= .5)&(decision_after < .5)))+
                           int(np.count_nonzero((check_reference >= .5)&(check_after < .5))))
    return np.flatnonzero(keep)
