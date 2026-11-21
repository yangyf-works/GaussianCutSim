"""Exact-order fallback geometry kernels; Float64, no fastmath.

Transcendentals used for opacity and encoded clipping remain NumPy operations.
This keeps the existing Float32 boundary ties and pruning decisions intact.
"""
import os
from threading import local
import numpy as np
from cutsim.geometry.strip_kernels import njit, prange, enabled as acceleration_enabled, configure_threads
from cutsim.rendering.clip_polygon_kernel import render_clip_polygon

# Only parent-task workers set this flag. Numba's thread count is thread-local;
# the main cutting/pruning thread retains its configured parallel team.
parent_worker = local()


def enabled():
    return acceleration_enabled() and os.environ.get('CUTSIM_FALLBACK_ACCEL', '1') != '0'


def subdivision_policy():
    """Experimental, default OFF: low-density timing/visual acceptance failed.

    Both switches are also part of the exact draw-cache key.
    """
    return (os.environ.get('CUTSIM_DRAW_LOCAL_BOUNDARIES', '0') != '0',
            os.environ.get('CUTSIM_DRAW_COLLECTIVE_COVERAGE', '0') != '0')


@njit(cache=True, nogil=True)
def collectively_covered(polygon, center, ratio, certified):
    """Conservative coverage certificate, using only already retained siblings.

    Convex vertex maxima bound distance over the whole cell. A sibling whose
    finite disk does not cover the whole cell contributes zero. Every sibling
    belongs to this same convex region, so its selected planes retain the cell.
    Only retained siblings from the same convex region may be passed here.
    """
    if ratio <= 0.: return False
    core = polygon
    target = .99
    transmission = 1.
    for j in range(-1, len(certified)):
        i = -1 if j < 0 else len(certified)-1-j
        c, r = (center, ratio) if i < 0 else certified[i]
        rho = 0.
        for p in core:
            d = p-c
            rho = max(rho, np.dot(d, d)/(r*r))
        if rho <= 9.: transmission *= 1.-.99*np.exp(-.5*rho)
        # Margin avoids accepting a floating-point tie at the coverage target.
        if 1.-transmission >= target+2e-6: return True
    return False


@njit(cache=True, nogil=True)
def retained_disk_radius(planes, clearance, order, limit, guard):
    # Distance to each omitted boundary *inside the two retained halfplanes*.
    # A line hidden entirely behind the two encoded clips does not limit the disk.
    radius=limit
    if clearance[order[0]] < 0. or clearance[order[1]] < 0.:
        return min(limit,clearance[order[2]]-guard)
    for index in order[2:]:
        normal=planes[index,:2];d=clearance[index]
        if d<=guard:return min(radius,d-guard)
        length2=np.dot(normal,normal)
        point=-(d-guard)*normal/length2
        tangent=np.array([-normal[1],normal[0]])/np.sqrt(length2)
        low=-np.inf;high=np.inf
        for selected in order[:2]:
            slope=np.dot(planes[selected,:2],tangent)
            value=np.dot(planes[selected,:2],point)+clearance[selected]+guard
            if slope==0.:
                if value<0.:low=np.inf;high=-np.inf;break
            elif slope>0.:low=max(low,-value/slope)
            else:high=min(high,-value/slope)
        if low>high:continue
        along=min(max(0.,low),high)
        distance=np.sqrt(np.dot(point,point)+along*along)
        radius=min(radius,distance-guard)
    return radius


@njit(cache=True, nogil=True)
def coarse_region(polygon, planes, origin, basis, inverse, sigma, guard, minimum,
                  base_depth, max_depth, budget):
    """Try a bounded coarse cover before allocating the fine subdivision tree.

    Each finite disk is entirely on the retained side of every omitted plane.
    Coverage is certified over convex subcells, not sampled at their centres.
    Without a certificate, use fine subdivision with its size stopping rule.
    """
    empty = [(np.empty(3, np.float32), np.float32(0.), np.empty((0, 3)), False, 0)]
    empty.pop()
    for point in polygon:
        if np.dot(point,point) >= (3.-guard)**2:return empty,-1
    lo, hi = bounds(polygon)
    side = np.max(hi-lo)
    corner = .5*(lo+hi)-side/2.
    for depth in range(base_depth, min(3, max_depth)+1):
        divisions = 2**depth
        if divisions*divisions > budget: break
        width = side/divisions
        leaves = empty.copy()
        certificates = [(np.zeros(2), 1.)]; certificates.pop()
        pieces = [np.empty((0, 2))]; pieces.pop()
        safe = True
        for iy in range(divisions):
            for ix in range(divisions):
                cell = corner+width*np.array([ix, iy])
                poly = polygon
                for a,b,c in ((1.,0.,-cell[0]),(-1.,0.,cell[0]+width),
                              (0.,1.,-cell[1]),(0.,-1.,cell[1]+width)):
                    poly=render_clip_polygon(poly,np.array([a,b,c]))
                if polygon_area(poly)<=guard*guard: continue
                center=np.zeros(2)
                for row in poly: center+=row
                center/=len(poly)
                position=(origin+basis@center).astype(np.float32)
                center=inverse@(position.astype(np.float64)-origin)
                clearance=planes[:,:2]@center+planes[:,2]
                order=np.argsort(clearance,kind='mergesort')
                radius=retained_disk_radius(planes,clearance,order,
                    min(3.5*width,3.-np.sqrt(np.dot(center,center))-guard),guard)
                child_sigma=np.float32(sigma*radius/3.*(1.-1e-6))
                if radius<=3.*guard or float(child_sigma)<minimum or np.any(clearance < -guard):
                    safe=False;break
                ratio=float(child_sigma)/sigma
                selected=order[:2]
                local=planes[selected].copy()
                local[:,:2]*=ratio;local[:,2]=clearance[selected]
                leaves.append((position,child_sigma,local,False,0))
                certificates.append((center.copy(),ratio))
                pieces.append(poly)
            if not safe: break
        if not safe: continue
        # Corners are exactly where centre-only cell disks have weak tails.
        # Add boundary anchors, still bounded by the parent and third plane.
        # These are draw-only children; no parent rotation or region expansion.
        for edge in range(len(polygon)):
            a=polygon[edge];b=polygon[(edge+1)%len(polygon)]
            segments=max(1,int(np.ceil(np.sqrt(np.dot(b-a,b-a))/width)))
            for index in range(segments):
                center=a+(b-a)*(index/segments)
                position=(origin+basis@center).astype(np.float32)
                center=inverse@(position.astype(np.float64)-origin)
                clearance=planes[:,:2]@center+planes[:,2]
                order=np.argsort(clearance,kind='mergesort')
                radius=retained_disk_radius(planes,clearance,order,
                    min(3.5*width,3.-np.sqrt(np.dot(center,center))-guard),guard)
                child_sigma=np.float32(sigma*radius/3.*(1.-1e-6))
                if radius<=3.*guard or float(child_sigma)<minimum or np.any(clearance < -guard):continue
                if len(leaves)>=min(budget,128):safe=False;break
                ratio=float(child_sigma)/sigma;selected=order[:2]
                local=planes[selected].copy();local[:,:2]*=ratio;local[:,2]=clearance[selected]
                leaves.append((position,child_sigma,local,False,0))
                certificates.append((center.copy(),ratio))
            if not safe:break
        if not safe:continue
        # Tighter conservative bounds in 4x4 subcells; no point-only sampling.
        for poly in pieces:
            pl,ph=bounds(poly);step=(ph-pl)/4.
            for iy in range(4):
                for ix in range(4):
                    cell=pl+step*np.array([ix,iy]);part=poly
                    for a,b,c in ((1.,0.,-cell[0]),(-1.,0.,cell[0]+step[0]),
                                  (0.,1.,-cell[1]),(0.,-1.,cell[1]+step[1])):
                        part=render_clip_polygon(part,np.array([a,b,c]))
                    if polygon_area(part)<=guard*guard:continue
                    # A vanishing dummy candidate contributes no attenuation.
                    if not collectively_covered(part,np.array([1e6,1e6]),1e-9,
                                                certificates):
                        safe=False;break
                if not safe:break
            if not safe:break
        if safe:return leaves,depth
    return empty,-1


@njit(cache=True, nogil=True)
def early_cell_cover(poly, corner, width, depth, planes, origin, basis, inverse,
                     sigma, guard, minimum, base_depth, max_depth, budget):
    empty=[(np.empty(3,np.float32),np.float32(0.),np.empty((0,3)),False,0)]
    empty.pop()
    if depth not in (0,2) or depth<base_depth or budget<4:return empty
    for point in poly:
        if np.dot(point,point)>4.:return empty
    # A bounded three-level lookahead estimates work without encoding leaves.
    # This is only a cost heuristic; the independent full-area certificate below
    # is mandatory for every accepted replacement.
    theta=np.arange(8)*np.pi/4.
    octagon=np.column_stack((-np.cos(theta),-np.sin(theta),np.full(8,2.)))
    pending=[(poly,corner,width,depth)]
    estimated=0
    while len(pending):
        piece,cell,w,d=pending.pop()
        if polygon_area(piece)<=guard*guard:continue
        if d==depth+3:
            estimated+=1
        else:
            center=np.zeros(2)
            for row in piece:center+=row
            center/=len(piece)
            pos=(origin+basis@center).astype(np.float32)
            center=inverse@(pos.astype(np.float64)-origin)
            clearance=planes[:,:2]@center+planes[:,2]
            radius=min(3.5*w,3.-np.sqrt(np.dot(center,center))-guard)
            order=np.argsort(clearance,kind='mergesort')
            exact_radius=min(radius,clearance[order[2]]-guard)
            split=(d<base_depth or ((np.count_nonzero(clearance<radius+guard)>2 or
                coverage_split(piece,center,max(0.,exact_radius)/3.*(1.-1e-6),octagon,
                               2./np.cos(np.pi/8),-2.*np.log(.75/.99))) and d<max_depth))
            if split and sigma*.375*w*(1.-1e-6)>=minimum:
                half=w/2.
                for dx,dy in ((1,1),(0,1),(1,0),(0,0)):
                    c=cell+half*np.array([dx,dy]);part=piece
                    for aa,bb,cc in ((1.,0.,-c[0]),(-1.,0.,c[0]+half),
                                     (0.,1.,-c[1]),(0.,-1.,c[1]+half)):
                        part=render_clip_polygon(part,np.array([aa,bb,cc]))
                    if len(part):pending.append((part,c,half,d+1))
            else:estimated+=1
        if estimated>=64:break
    if estimated<8:return empty
    coarse,level=coarse_region(poly,planes,origin,basis,inverse,sigma,guard,minimum,
                              0,2,min(estimated-1,63,budget))
    return coarse if level>=0 else empty


def probe_attenuation(draw, points, origin, basis, float32_clips):
    from cutsim.rendering.gaussian_attributes import _render_axes
    u, v = _render_axes(draw['rot'])
    world = origin+points@basis.T
    projected = np.column_stack((world@u[0], world@v[0]))
    centres = np.column_stack((np.sum(draw['positions']*u, axis=1),
                               np.sum(draw['positions']*v, axis=1)))
    angles = np.column_stack((draw['clip1_angle'][:, 0], draw['clip2_angle'][:, 0])).astype(float)
    splits = np.column_stack((draw['clip1_split'][:, 0], draw['clip2_split'][:, 0]))
    cosine, sine = np.cos(angles), np.sin(angles)
    cosine32, sine32 = np.cos(angles.astype(np.float32)), np.sin(angles.astype(np.float32))
    result = np.empty((len(points), len(centres)), np.float32)
    if getattr(parent_worker,'single_thread',False):
        from cutsim.geometry.strip_kernels import numba
        numba.set_num_threads(1)
    else:
        configure_threads()
    # Bound temporary arrays independently of the number of test points.
    block = max(1, min(4096, 262144//max(1, len(centres))))
    u32, v32 = u.astype(np.float32), v.astype(np.float32)
    for start in range(0, len(points), block):
        end = min(start+block, len(points))
        squared, visible = probe_geometry(projected[start:end], centres, draw['scale'][:, 0],
            draw['positions'], world[start:end].astype(np.float32), u32, v32,
            angles, splits, cosine, sine, cosine32, sine32, float32_clips)
        alpha = .99*np.exp(-.5*squared)*visible
        result[start:end] = -np.log1p(-alpha)
    return result


@njit(cache=True, nogil=True, parallel=True)
def probe_geometry(projected, centres, scale, positions, world32, u32, v32,
                   angles, splits, cosine, sine, cosine32, sine32, float32_clips):
    squared = np.empty((len(projected), len(centres)))
    visible = np.empty(squared.shape, np.bool_)
    for i in prange(len(projected)):
        for j in range(len(centres)):
            x = (projected[i, 0]-centres[j, 0])/scale[j]
            y = (projected[i, 1]-centres[j, 1])/scale[j]
            rho = x*x+y*y
            keep = rho <= 9.
            if float32_clips:
                dx = np.float32(world32[i, 0]-positions[j, 0])
                dy = np.float32(world32[i, 1]-positions[j, 1])
                dz = np.float32(world32[i, 2]-positions[j, 2])
                cx = np.float32(np.float32(np.float32(dx*u32[j, 0])+np.float32(dy*u32[j, 1]))+np.float32(dz*u32[j, 2]))/scale[j]
                cy = np.float32(np.float32(np.float32(dx*v32[j, 0])+np.float32(dy*v32[j, 1]))+np.float32(dz*v32[j, 2]))/scale[j]
                for k in range(2):
                    value = np.float32(np.float32(cx*cosine32[j, k])+np.float32(cy*sine32[j, k]))
                    keep = keep and (angles[j, k] == 4. or value >= splits[j, k])
            else:
                for k in range(2):
                    keep = keep and (angles[j, k] == 4. or x*cosine[j, k]+y*sine[j, k] >= splits[j, k])
            squared[i, j] = rho
            visible[i, j] = keep
    return squared, visible


@njit(cache=True, nogil=True)
def pairwise_sum(values):
    """NumPy's contiguous Float64 sum order, including the eight accumulators."""
    n = len(values)
    if n < 8:
        result = -0.
        for value in values: result += value
        return result
    if n <= 128:
        partial = values[:8].copy()
        i = 8
        while i <= n-8:
            for j in range(8): partial[j] += values[i+j]
            i += 8
        result = ((partial[0]+partial[1])+(partial[2]+partial[3]))+((partial[4]+partial[5])+(partial[6]+partial[7]))
        while i < n:
            result += values[i]; i += 1
        return result
    middle = n//2
    middle -= middle % 8
    return pairwise_sum(values[:middle])+pairwise_sum(values[middle:])


@njit(cache=True, nogil=True)
def polygon_area(polygon):
    if len(polygon) < 3: return 0.
    terms = np.empty(len(polygon))
    for i in range(len(polygon)):
        j = (i+1) % len(polygon)
        terms[i] = polygon[i, 0]*polygon[j, 1]-polygon[i, 1]*polygon[j, 0]
    return .5*abs(pairwise_sum(terms))


@njit(cache=True, nogil=True)
def bounds(polygon):
    lower = polygon[0].copy(); upper = lower.copy()
    for i in range(1, len(polygon)):
        for j in range(2):
            lower[j] = min(lower[j], polygon[i, j])
            upper[j] = max(upper[j], polygon[i, j])
    return lower, upper


@njit(cache=True, nogil=True)
def coverage_split(polygon, center, ratio, octagon, nearest_limit, strong_rho):
    if ratio <= 0.: return True
    if ratio >= 1. and np.sqrt(np.dot(center, center)) < 1e-12: return False
    lower, upper = bounds(polygon)
    nearest = np.maximum(lower, np.minimum(upper, 0.))
    if np.sqrt(np.dot(nearest, nearest)) > nearest_limit: return False
    core = polygon
    if np.max(core[:, 0]*core[:, 0]+core[:, 1]*core[:, 1]) > 4.:
        for plane in octagon:
            core = render_clip_polygon(core, plane)
            if len(core) < 3: return False
    child_rho = ((core[:, 0]-center[0])**2+(core[:, 1]-center[1])**2)/(ratio*ratio)
    if np.max(child_rho) <= strong_rho+2e-6: return False
    return np.max(child_rho-(core[:, 0]*core[:, 0]+core[:, 1]*core[:, 1])) > 2e-6


@njit(cache=True, nogil=True)
def subdivide(packed, offsets, lengths, disk, origin, basis, inverse, sigma, guard, floor_sigma,
              max_children, base_depth, max_depth, min_child_sigma, preserve_simple_regions,
              octagon, nearest_limit, strong_rho, local_boundaries, collective):
    # Leaf payload: position, sigma, unencoded planes, needs Top-2, region index.
    children = [(np.empty(3, np.float32), np.float32(0.), np.empty((0, 3)), False, 0)]
    children.pop()
    domains = [(np.empty((0, 2)), np.empty((0, 3)))]
    domains.pop()
    # polygons, simple, size stops, coverage splits, Top-2, max depth, reason bits
    stats = np.zeros(9, np.int64)
    for region_index in range(len(offsets)-1):
        region = packed[offsets[region_index]:offsets[region_index+1]]
        planes_buffer = np.empty((len(region), 3)); plane_count = 0
        polygon = disk.copy()
        for plane_index in range(len(region)):
            plane = region[plane_index]
            length = lengths[offsets[region_index]+plane_index]
            if length < 1e-14:
                if plane[2] < 0.:
                    polygon = np.empty((0, 2)); break
                continue
            normalized = plane/length
            planes_buffer[plane_count] = normalized; plane_count += 1
            polygon = render_clip_polygon(polygon, normalized)
            if not len(polygon): break
        if polygon_area(polygon) <= guard*guard: continue
        stats[0] += 1
        planes = planes_buffer[:plane_count].copy()
        domains.append((polygon.copy(), planes.copy()))
        if preserve_simple_regions and len(planes) <= 2:
            if len(children) >= max_children:
                stats[6] |= 1; break
            children.append((origin.astype(np.float32), np.float32(sigma), planes.copy(), False, region_index))
            stats[1] += 1
            continue
        lo, hi = bounds(polygon)
        side = np.max(hi-lo); middle = .5*(lo+hi)
        stack = [(polygon, middle-side/2., side, 0)]
        while len(stack):
            poly, corner, width, depth = stack.pop()
            if polygon_area(poly) <= guard*guard: continue
            if len(children) >= max_children:
                stats[6] |= 1; break
            stats[5] = max(stats[5], depth)
            center = np.zeros(2)
            for row in poly: center += row
            center /= len(poly)
            position = (origin+basis@center).astype(np.float32)
            center = inverse@(position.astype(np.float64)-origin)
            clearance = planes[:, :2]@center+planes[:, 2]
            radius = min((3.5 if preserve_simple_regions else 2.25)*width,
                         3.-np.sqrt(np.dot(center, center))-guard)
            crossing = np.count_nonzero(clearance < radius+guard)
            order = np.argsort(clearance, kind='mergesort')
            exact_radius = min(radius, clearance[order[2]]-guard) if len(order) > 2 else radius
            split_coverage = (preserve_simple_regions and depth < max_depth and
                coverage_split(poly, center, max(0., exact_radius)/3.*(1.-1e-6), octagon, nearest_limit, strong_rho))
            wants_split = depth < base_depth or ((crossing > 2 or split_coverage) and depth < max_depth)
            can_split = sigma*(.375*width)*(1.-1e-6) >= min_child_sigma
            size_stopped = wants_split and not can_split
            if size_stopped:
                stats[2] += 1; stats[6] |= 2
            if wants_split and can_split:
                if preserve_simple_regions and local_boundaries and collective:
                    coarse=early_cell_cover(poly,corner,width,depth,planes,origin,basis,inverse,
                        sigma,guard,min_child_sigma,base_depth,max_depth,max_children-len(children))
                    if len(coarse):
                        for p,s,q,top,_ in coarse:children.append((p,s,q,top,region_index))
                        stats[7]+=1;stats[8]+=len(coarse)
                        continue
                if split_coverage: stats[3] += 1
                half = width/2.
                for dx, dy in ((1, 1), (0, 1), (1, 0), (0, 0)):
                    cell = corner+half*np.array([dx, dy])
                    piece = poly
                    for a, b, c in ((1., 0., -cell[0]), (-1., 0., cell[0]+half),
                                    (0., 1., -cell[1]), (0., -1., cell[1]+half)):
                        piece = render_clip_polygon(piece, np.array([a, b, c]))
                    if len(piece): stack.append((piece, cell, half, depth+1))
                continue
            if radius <= 3.*guard or np.any(clearance < -guard):
                stats[6] |= 4; continue
            child_sigma = np.float32(sigma*exact_radius/3.*(1.-1e-6))
            top_two = min_child_sigma > 0. and (size_stopped or float(child_sigma) < min_child_sigma)
            if top_two:
                if not size_stopped: stats[2] += 1
                stats[4] += 1; stats[6] |= 2
                child_sigma = np.float32(max(float(floor_sigma), sigma*radius/3.*(1.-1e-6)))
            elif exact_radius <= 3.*guard:
                stats[6] |= 4; continue
            ratio = float(child_sigma)/sigma
            if top_two:
                local = planes.copy(); local[:, 2] = clearance/ratio
            else:
                selected = order[:2]
                local = planes[selected].copy()
                local[:, :2] *= ratio
                local[:, 2] = clearance[selected] if preserve_simple_regions else clearance[selected]-guard
            children.append((position, child_sigma, local, top_two, region_index))
        if stats[6] & 1: break
    return children, domains, stats


def build_subdivision(regions, disk, origin, basis, inverse, sigma, guard, floor_sigma,
                      max_children, base_depth, max_depth, min_child_sigma, preserve_simple_regions):
    regions = [np.asarray(region, float).reshape(-1, 3) for region in regions]
    offsets = np.r_[0, np.cumsum([len(region) for region in regions])].astype(np.int64)
    packed = np.concatenate(regions) if regions else np.empty((0, 3))
    # NumPy's norm uses a dot/sqrt reduction; BLAS nrm2 changes rounding.
    # These few boundary normals are computed once, outside the cell traversal.
    lengths = np.array([np.linalg.norm(plane[:2]) for plane in packed])
    theta = np.arange(8)*np.pi/4
    octagon = np.column_stack((-np.cos(theta), -np.sin(theta), np.full(8, 2.)))
    return subdivide(packed, offsets, lengths, disk, origin, basis, inverse, sigma, guard, floor_sigma,
        max_children, base_depth, max_depth, min_child_sigma, preserve_simple_regions,
        octagon, 2./np.cos(np.pi/8), -2.*np.log(.75/.99), *subdivision_policy())
