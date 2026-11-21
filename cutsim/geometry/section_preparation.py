"""Exact-order geometry preparation; no sampling or tolerance changes."""
import math
import os
import numpy as np
from cutsim.geometry import strip_kernels as kernels


def enabled():
    return kernels.enabled() and os.environ.get('CUTSIM_SECTION_PREP_ACCEL', '1') != '0'


@kernels.njit(cache=True, nogil=True)
def _edge_roots(local):
    roots = np.empty((2*len(local), 2))
    angles = np.empty(2*len(local)); count = 0
    for i in range(len(local)):
        a = local[i]; b = local[(i+1) % len(local)]
        d = b-a; aa = float(d@d)
        if aa == 0.: continue
        bb = float(a@d); cc = float(a@a)-1.
        disc = bb*bb-aa*cc
        if disc < 0.: continue
        root = math.sqrt(disc)
        for t in ((-bb-root)/aa, (-bb+root)/aa):
            if 0. <= t <= 1.:
                z = a+t*d
                roots[count] = z
                angles[count] = math.atan2(z[1], z[0]) % (2.*math.pi)
                count += 1
    return roots[:count].copy(), angles[:count].copy()


@kernels.njit(cache=True, nogil=True, parallel=True)
def _edges_batch(polygons):
    roots = [np.empty((0, 2)) for _ in range(len(polygons))]
    angles = [np.empty(0) for _ in range(len(polygons))]
    for task in kernels.prange(len(polygons)):
        i = np.int64(task)
        roots[i], angles[i] = _edge_roots(polygons[i])
    return roots, angles


def conics_batch(tasks):
    """Preserve NumPy eigensolve/transforms/trig; batch independent edge roots.

    Degenerate, near-parallel and non-Float64 sections use the original solver.
    Each task is (polygon, matrix, linear, constant, chord_error).
    """
    from cutsim.geometry.strip_intersections import conic_section, halfplanes, ordered
    result = [None]*len(tasks); active = []; locals_ = []; prepared = []
    for i, (poly, matrix, linear, constant, error) in enumerate(tasks):
        if len(poly) < 3:
            result[i] = np.empty((0, 2)); continue
        if any(x.dtype != np.float64 for x in (poly, matrix, linear)):
            result[i] = conic_section(poly, matrix, linear, constant, error); continue
        ev, axes = np.linalg.eigh(matrix)
        if np.count_nonzero(ev > 64.*np.finfo(float).eps) != 2:
            result[i] = conic_section(poly, matrix, linear, constant, error); continue
        q = axes.T@linear
        center = -axes@(q/ev)
        rhs = float(q@(q/ev)-constant)
        if rhs <= 0.:
            result[i] = np.empty((0, 2)); continue
        radii = np.sqrt(rhs/ev); transform = axes*radii
        local = (poly-center)@axes/radii
        norms = np.sum(local*local, axis=1)
        if np.all(norms <= 1.):
            result[i] = poly; continue
        active.append(i); locals_.append(local)
        prepared.append((center, radii, transform, norms))
    if active:
        kernels.configure_threads()
        roots, root_angles = _edges_batch(kernels.numba.typed.List(locals_))
    for slot, i in enumerate(active):
        poly, _, _, _, error = tasks[i]
        center, radii, transform, norms = prepared[slot]
        points = list(poly[norms <= 1.])
        points.extend(center+transform@z for z in roots[slot])
        angles = sorted(set(list(np.arange(5)*math.pi*.5)+list(root_angles[slot])))
        planes = halfplanes(poly)
        step = 2.*math.acos(max(-1., min(1., 1.-error/float(max(radii)))))
        step = max(step, 1e-8)
        for a, b in zip(angles[:-1], angles[1:]):
            mid = .5*(a+b)
            p = center+transform@np.array([math.cos(mid), math.sin(mid)])
            if np.all(planes[:, :2]@p+planes[:, 2] >= -1e-10):
                count = max(1, int(math.ceil((b-a)/step)))
                theta = np.linspace(a, b, count+1)
                points.extend(center+np.column_stack((np.cos(theta), np.sin(theta)))@transform.T)
        result[i] = ordered(points)
    return result


@kernels.njit(cache=True, nogil=True, parallel=True)
def _worlds(planes, offsets, bases, origins):
    output = np.empty((len(planes), 4))
    for i in kernels.prange(len(bases)):
        for j in range(offsets[i], offsets[i+1]):
            normal = planes[j, :2]@bases[i]
            output[j, :3] = normal
            output[j, 3] = planes[j, 2]-normal@origins[i]
    return output


def world_planes(planes, bases, origins):
    bases = np.asarray(bases); origins = np.asarray(origins)
    if bases.dtype != np.float64 or any(p.dtype != np.float64 for p in planes):
        result = []
        for ps, basis, p in zip(planes, bases, origins):
            rows = []
            for plane in ps:
                normal = plane[:2]@basis
                rows.append(np.r_[normal, plane[2]-normal@p])
            result.append(np.asarray(rows).reshape(-1, 4))
        return result
    offsets = np.r_[0, np.cumsum([len(p) for p in planes])].astype(np.int64)
    # NumPy also promotes a Float32 origin when dotted with a Float64 normal.
    packed = _worlds(np.concatenate(planes), offsets, bases, np.asarray(origins, dtype=np.float64))
    return [packed[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


class RegionReplay:
    """Chunk-local prefix cache of exact ordered clipping operations.

    Never reuse a previous residual polygon as a new footprint. Replay the
    original full constraint sequence, reusing only identical local prefixes.
    The instance is bound to one immutable GS footprint and frame.
    """
    def __init__(self, polygon):
        self.root = (polygon, {})

    def intersect(self, planes):
        poly, children = self.root
        for plane in planes:
            key = plane.tobytes()
            node = children.get(key)
            if node is None:
                node = (kernels.clip_polygon(poly, plane), {})
                children[key] = node
            poly, children = node
            if len(poly) < 3: break
        return poly
