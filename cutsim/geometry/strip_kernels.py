"""Optional deterministic native kernels. Float64, no fastmath, bounded threads.

CUTSIM_STRIP_ACCEL=0 selects the original scalar implementation for validation.
CUTSIM_CPU_THREADS limits native parallelism (default at most eight).
"""
import math
import os
# Independent native batches alternate with ordered Python geometry updates.
# Sleeping workers avoid competing with the coordinator between those batches.
# Respect explicit application/user runtime settings.
if os.environ.get('CUTSIM_STRIP_ACCEL', '1') != '0':
    os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    os.environ.setdefault('KMP_BLOCKTIME', '0')
import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError:
    numba = None
    def njit(*args, **kwargs):
        return lambda function: function
    prange = range


def enabled():
    return numba is not None and os.environ.get('CUTSIM_STRIP_ACCEL', '1') != '0'


def configure_threads():
    if numba is None: return 1
    requested = int(os.environ.get('CUTSIM_CPU_THREADS', min(8, max(1, (os.cpu_count() or 2)-1))))
    if requested < 1: raise ValueError('CUTSIM_CPU_THREADS must be positive')
    count = min(requested, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(count)
    return count


@njit(cache=True, nogil=True)
def clip_polygon(poly, plane):
    if len(poly) == 0: return poly
    f = poly @ plane[:2] + plane[2]
    error = 64.*np.finfo(np.float64).eps*(1.+np.max(np.abs(poly))*np.sum(np.abs(plane[:2]))+abs(plane[2]))
    for i in range(len(f)):
        if abs(f[i]) <= error: f[i] = 0.
    inside = f >= 0.
    if np.all(inside): return poly
    if not np.any(inside): return np.empty((0, 2))
    out = np.empty((2*len(poly), 2)); count = 0
    for i in range(len(poly)):
        j = (i+1) % len(poly)
        if inside[i]:
            out[count] = poly[i]; count += 1
        if inside[i] != inside[j]:
            out[count] = poly[i]+f[i]/(f[i]-f[j])*(poly[j]-poly[i]); count += 1
    return out[:count].copy()


@njit(cache=True, nogil=True)
def intersect_polygon(poly, planes):
    for i in range(len(planes)):
        poly = clip_polygon(poly, planes[i])
        if len(poly) < 3: break
    return poly


@njit(cache=True, nogil=True)
def polygon_planes(poly):
    out = np.empty((len(poly), 3)); count = 0
    for i in range(len(poly)):
        j = (i+1) % len(poly)
        x = -(poly[j, 1]-poly[i, 1]); y = poly[j, 0]-poly[i, 0]
        length = math.sqrt(x*x+y*y)
        if length == 0.: continue
        x /= length; y /= length
        out[count, 0] = x; out[count, 1] = y
        out[count, 2] = -(x*poly[i, 0]+y*poly[i, 1]); count += 1
    return out[:count].copy()


@njit(cache=True, nogil=True)
def hull(points):
    if len(points) < 3: return np.empty((0, 2))
    # Lexicographic insertion order, identical to sorted(set(map(tuple,...))).
    p = points.copy()
    for i in range(1, len(p)):
        x, y = p[i, 0], p[i, 1]; j = i-1
        while j >= 0 and (p[j, 0] > x or (p[j, 0] == x and p[j, 1] > y)):
            p[j+1] = p[j]; j -= 1
        p[j+1, 0] = x; p[j+1, 1] = y
    n = 1
    for i in range(1, len(p)):
        if p[i, 0] != p[n-1, 0] or p[i, 1] != p[n-1, 1]:
            p[n] = p[i]; n += 1
    out = np.empty((2*n+1, 2)); count = 0
    for i in range(n):
        while count > 1:
            a = out[count-1]-out[count-2]; b = p[i]-out[count-2]
            if a[0]*b[1]-a[1]*b[0] > 0.: break
            count -= 1
        out[count] = p[i]; count += 1
    lower = count
    for i in range(n-2, -1, -1):
        while count > lower:
            a = out[count-1]-out[count-2]; b = p[i]-out[count-2]
            if a[0]*b[1]-a[1]*b[0] > 0.: break
            count -= 1
        out[count] = p[i]; count += 1
    return out[:max(0, count-1)].copy()


@njit(cache=True, nogil=True, parallel=True)
def intersect_batches(polygons, planes):
    output = [np.empty((0, 2)) for _ in range(len(polygons))]
    for task in prange(len(polygons)):
        i = np.int64(task)
        output[i] = intersect_polygon(polygons[i], planes[i])
    return output


def sections_batch(positions, bases, polygons, errors, planes, centers, matrices, radii, quadrics, *, stats=None):
    # Transform with the original NumPy operations. LAPACK/BLAS rounding in a
    # JIT rewrite of conics can change near-coincident hull vertices. Preserve
    # those operations and parallelise only independent closed-plane clipping.
    from cutsim.geometry.strip_intersections import conic_section
    local_planes = []; inputs = []
    # A repeated tangent frame has identical coefficients for this batch's
    # fixed primitives. Translation terms still use each individual GS.
    frames = {}
    row_frames = []
    for p, basis, polygon in zip(positions, bases, polygons):
        key=basis.tobytes()
        if key not in frames:
            frames[key]=([world[:, :3]@basis.T for world in planes], {})
        projected, conics = frames[key]
        row_frames.append(conics)
        for xy,world in zip(projected,planes):
            local_planes.append(np.column_stack((xy, world[:, :3]@p+world[:, 3])))
            inputs.append(polygon)
    if stats is not None:
        stats['section_frame_rows']=stats.get('section_frame_rows',0)+len(positions)
        stats['section_frame_reuses']=stats.get('section_frame_reuses',0)+len(positions)-len(frames)
    output = intersect_batches(numba.typed.List(inputs), numba.typed.List(local_planes))
    from cutsim.geometry.section_preparation import enabled as preparation_enabled, conics_batch
    batched = preparation_enabled()
    tasks = []; indices = []
    for i, poly in enumerate(output):
        row, primitive = divmod(i, len(planes))
        if quadrics[primitive] and len(poly) >= 3:
            p = positions[row]; basis = bases[row]; matrix = matrices[primitive]
            d = p-centers[primitive]
            conics=row_frames[row]
            if primitive not in conics:
                product=basis@matrix
                conics[primitive]=(product@basis.T,product)
            quadratic,product=conics[primitive]
            args = (poly, quadratic, product@d,
                    float(d@matrix@d-radii[primitive]**2), errors[row])
            if batched:
                indices.append(i); tasks.append(args)
            else: output[i] = conic_section(*args)
    if tasks:
        for i, poly in zip(indices, conics_batch(tasks)): output[i] = poly
    return output, np.zeros(len(output), np.bool_)


@njit(cache=True, nogil=True, parallel=True)
def ray_region_hits(rows, origins, ray_indices, direction, positions, normals, bases,
                    region_offsets, plane_offsets, planes, plane_xy, bounds, eps, slots):
    """Each candidate owns disjoint output slots; reductions stay in row order."""
    starts = np.empty(slots[-1]); ends = np.empty(slots[-1]); counts = np.zeros(len(rows), np.int64)
    coplanar = np.zeros(len(rows), np.bool_)
    for i in prange(len(rows)):
        row = rows[i]; ray = ray_indices[i]; origin = origins[ray]; p = positions[row]
        den = float(normals[row]@direction); distance = float(normals[row]@(p-origin))
        first = region_offsets[row, 0]; last = region_offsets[row, 1]
        low_stock, high_stock = bounds[ray, 0], bounds[ray, 1]
        if abs(den) > 1e-12:
            t = distance/den
            if t < low_stock-eps or t > high_stock+eps: continue
            xy = (origin+t*direction-p)@bases[row].T
            for region in range(first, last):
                a = plane_offsets[region]; b = plane_offsets[region+1]
                if np.all(plane_xy[a:b]@xy+planes[a:b, 2] >= -eps):
                    starts[slots[i]] = t; ends[slots[i]] = t; counts[i] = 1; break
        elif abs(distance) <= eps:
            coplanar[i] = True
            xy = (origin-p)@bases[row].T; d = direction@bases[row].T
            for region in range(first, last):
                a = plane_offsets[region]; b = plane_offsets[region+1]
                slopes = plane_xy[a:b]@d; offsets = plane_xy[a:b]@xy+planes[a:b, 2]
                lo = low_stock; hi = high_stock; rejected = False
                for k in range(len(slopes)):
                    slope = slopes[k]; offset = offsets[k]
                    if abs(slope) < 1e-14:
                        if offset < -eps: rejected = True; break
                    elif slope >= 1e-14: lo = max(lo, -offset/slope)
                    else: hi = min(hi, -offset/slope)
                if not rejected and lo <= hi:
                    slot = slots[i]+counts[i]; starts[slot] = lo; ends[slot] = hi; counts[i] += 1
    return starts, ends, counts, coplanar
