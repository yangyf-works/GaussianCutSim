"""Bounded, render-only sampling of otherwise unsupported clip domains.

The original geometric parent and all its constraints stay intact. Small circular
GS cover its feasible tangent polygons. Every emitted 3-sigma disk obeys all
constraints until the minimum-size limit. At that limit, the two most important
analytic boundaries approximate the remaining detail without further splitting.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from cutsim.rendering.gaussian_attributes import _render_axes
from cutsim.rendering.gs_schema import CLIP_FIELDS, GS_FIELDS, validate_gaussian_attributes

_PRUNE_MAX_EVALUATIONS=16_000_000  # At most 64 MiB of Float32 attenuation samples.
_MIN_CHILD_SIGMA=.005  # World units (mm in the demo); 3-sigma diameter is .03 mm.


def _encode_leaf_planes(rows):
    """Vectorize the existing Float32 encoding, preserving plane/child order.

    Degenerate constraints still use the public scalar validation/encoding.
    NumPy's same hypot/atan2 ufuncs preserve the scalar boundary rounding.
    """
    from cutsim.rendering.clipping import local_planes_to_clip_values
    values=np.tile(np.array([4.,0.,4.,0.],np.float32),(len(rows),1))
    for count in (1,2):
        indices=np.array([i for i,q in enumerate(rows) if q.shape==(count,3)],np.int64)
        if not len(indices):continue
        planes=np.stack([rows[i] for i in indices]).astype(np.float64,copy=False)
        lengths=np.hypot(planes[:,:,0],planes[:,:,1])
        regular=np.all(np.isfinite(planes),axis=(1,2))&np.all(lengths>0.,axis=1)
        for i in indices[~regular]:values[i]=local_planes_to_clip_values(rows[i])
        p=planes[regular];ids=indices[regular]
        angles=np.arctan2(p[:,:,1],p[:,:,0]).astype(np.float32)
        angles[angles>=np.float32(np.pi)]=-np.float32(np.pi)
        with np.errstate(over='ignore',invalid='ignore'):
            splits=(-p[:,:,2]/lengths[regular]).astype(np.float32)
        if not np.isfinite(splits).all():raise ValueError('Boundary split must fit finite Float32')
        values[ids,:2*count:2]=angles;values[ids,1:2*count:2]=splits
    for i,q in enumerate(rows):
        if q.shape not in ((0,3),(1,3),(2,3)):values[i]=local_planes_to_clip_values(q)
    return values


def _clip_polygon(polygon, plane):
    from cutsim.geometry.strip_kernels import enabled
    if (enabled() and getattr(polygon,'dtype',None)==np.dtype('float64')
            and getattr(plane,'dtype',None)==np.dtype('float64')):
        from cutsim.rendering.clip_polygon_kernel import render_clip_polygon
        return render_clip_polygon(polygon,plane)
    if not len(polygon):
        return polygon
    signed = polygon @ plane[:2] + plane[2]
    result = []
    for i, end in enumerate(polygon):
        start = polygon[i - 1]
        a, b = signed[i - 1], signed[i]
        if (a >= 0.) != (b >= 0.):
            result.append(start + (end - start) * (a / (a - b)))
        if b >= 0.:
            result.append(end)
    return np.asarray(result, dtype=float).reshape(-1, 2)


def _area(polygon):
    if len(polygon) < 3:
        return 0.
    return .5 * abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) -
                             polygon[:, 1] * np.roll(polygon[:, 0], -1)))


def _central_coverage_needs_split(polygon, center, ratio):
    """Conservative single-child Gaussian bound in the parent's 2-sigma core.

    Clip only a TEMPORARY test polygon to a circumscribed octagon containing
    that disk. For ratio<=1, child rho minus parent rho is convex; its maximum
    on this convex polygon is attained at a vertex. No Gaussian opacity, clip
    boundary, geometric footprint or neighbour search range changes here.
    """
    if ratio<=0.:return True
    if ratio>=1. and np.linalg.norm(center)<1e-12:return False
    lower=polygon.min(axis=0);upper=polygon.max(axis=0)
    nearest=np.maximum(lower,np.minimum(upper,0.))
    if np.linalg.norm(nearest)>2./np.cos(np.pi/8):return False
    core=polygon
    if np.max(np.sum(core*core,axis=1))>4.:
        for theta in np.arange(8)*np.pi/4:
            core=_clip_polygon(core,np.array([-np.cos(theta),-np.sin(theta),2.]))
            if len(core)<3:return False
    child_rho=np.sum((core-center)**2,axis=1)/(ratio*ratio)
    # A .75 single-child floor stays clear of the .5 depth threshold. Requiring
    # one displaced child to reproduce the parent's .99 peak is impossible at
    # cell corners and would refine the whole central area to the size floor.
    # Neighbouring child disks then provide the remaining overlapping coverage.
    strong_rho=-2.*np.log(.75/.99)
    if np.max(child_rho)<=strong_rho+2e-6:return False
    difference=child_rho-np.sum(core*core,axis=1)
    return bool(np.max(difference)>2e-6)


def _prune_probes(draw, origin, basis, inverse, domains, phase):
    """Child-scale rings plus explicit edge/corner insets, in parent UV units.

    The second phase uses different radii, angles and edge positions. These are
    coverage probes, not points from the subdivision's centre classification.
    """
    centres=(draw['positions'].astype(float)-origin)@inverse.T
    sigma=float(np.linalg.norm(basis[:,0]))
    ratios=draw['scale'][:,0].astype(float)/sigma
    count=20 if phase else 16
    angles=2.*np.pi*(np.arange(count)+(.37 if phase else 0.))/count
    radii=(.22,.69,1.17,1.64,2.11,2.57,2.98) if phase else (.45,.95,1.45,1.95,2.45,2.9)
    ring=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    samples=np.vstack((np.zeros((1,2)),(np.asarray(radii)[:,None,None]*ring).reshape(-1,2)))
    accepted=np.ones((len(centres),len(samples)),bool)
    for angle_name,split_name in zip(CLIP_FIELDS[::2],CLIP_FIELDS[1::2]):
        angle=draw[angle_name][:,0].astype(float);split=draw[split_name][:,0].astype(float)
        accepted&=(angle[:,None]==4.)|(
            np.cos(angle[:,None])*samples[None,:,0]+np.sin(angle[:,None])*samples[None,:,1]>=split[:,None])
    points=(centres[:,None,:]+ratios[:,None,None]*samples)[accepted]
    probe_limit=_PRUNE_MAX_EVALUATIONS//max(1,len(centres))
    if len(points)>probe_limit:return None,None
    pieces=[points]
    point_count=len(points)
    # Uploaded Float32 split values move nominally identical boundaries by tiny
    # amounts. Probe EACH actual child clip line, including its uncertainty
    # collar: a geometric polygon-edge sample alone misses these raster ties.
    from cutsim.rendering.coverage_sparse import algorithm
    if algorithm() in ('sparse','resident'):
        from cutsim.rendering.coverage_probes import child_clip_probes
        collars=child_clip_probes(draw,centres,ratios,phase,probe_limit-point_count)
        if collars is None:return None,None
        point_count+=len(collars)
        pieces.append(collars)
    else:
        for row,center in enumerate(centres):
            for angle_name,split_name in zip(CLIP_FIELDS[::2],CLIP_FIELDS[1::2]):
                angle=float(draw[angle_name][row,0]);split=float(draw[split_name][row,0])
                if angle==4. or abs(split)>=3.:continue
                normal=np.array([np.cos(angle),np.sin(angle)])
                tangent=np.array([-normal[1],normal[0]])
                half=np.sqrt(9.-split*split)
                n=12 if phase else 8
                t=-half+2.*half*(np.arange(n)+(.61 if phase else .17))/n
                local=split*normal+t[:,None]*tangent
                valid=np.ones(n,bool)
                for other_angle,other_split in zip(CLIP_FIELDS[::2],CLIP_FIELDS[1::2]):
                    a=float(draw[other_angle][row,0]);b=float(draw[other_split][row,0])
                    valid&=(a==4.)|(local@np.array([np.cos(a),np.sin(a)])>=b-2e-6)
                local=local[valid]
                for offset in (-2e-7,0.,2e-7):
                    piece=center+ratios[row]*local+offset*normal
                    point_count+=len(piece)
                    if point_count>probe_limit:return None,None
                    pieces.append(piece)
    # Boundary samples are independent of the leaf-cell grid, including tiny
    # branches. Each edge contributes samples even if it is very short.
    step=max(float(np.median(ratios))*.4,1e-4)
    inset=(.00003,.003,.018,.09) if phase else (.00002,.005,.03,.12)
    for polygon,_ in domains:
        middle=polygon.mean(axis=0)
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            n=max(3,int(np.ceil(np.linalg.norm(b-a)/step)))
            if n>4096:return None,None
            t=(np.arange(n)+(.41 if phase else .0))/n
            edge=a+t[:,None]*(b-a)
            for amount in inset:
                point_count+=len(edge)
                if point_count>probe_limit:return None,None
                pieces.append(edge+amount*(middle-edge))
        point_count+=len(polygon)+1
        if point_count>probe_limit:return None,None
        pieces.append(polygon.copy())
        pieces.append(middle[None,:])
    points=np.concatenate(pieces)
    # A point close to a boundary of ANY containing convex branch gets the
    # stricter error bound. This is conservative for overlapping OR branches.
    band=max(.15,.5*float(np.median(ratios)))
    boundary=3.-np.linalg.norm(points,axis=1)<band
    for _,planes in domains:
        if not len(planes):continue
        distance=points@planes[:,:2].T+planes[:,2]
        boundary|=np.all(distance>=-1e-7,axis=1)&(np.min(distance,axis=1)<band)
    return points,boundary


def _probe_attenuation(draw, points, origin, basis, *, float32_clips=False):
    """-log(1-alpha) per child/probe; stable even under very dense overlap."""
    from cutsim.rendering.coverage_cuda import evaluator
    gpu = evaluator()
    if gpu is not None:
        return gpu.evaluate(draw, points, origin, basis, float32_clips)
    from cutsim.rendering.fallback_kernels import enabled, probe_attenuation
    if enabled():
        return probe_attenuation(draw, points, origin, basis, float32_clips)
    u,v=_render_axes(draw['rot'])
    world=origin+points@basis.T
    # Every fallback child inherits the same frame. Avoid an N x M x 3 array.
    projected=np.column_stack((world@u[0],world@v[0]))
    centres=np.column_stack((np.sum(draw['positions']*u,axis=1),
                             np.sum(draw['positions']*v,axis=1)))
    result=np.empty((len(points),len(centres)),np.float32)
    for start in range(0,len(centres),32):
        end=min(start+32,len(centres));scale=draw['scale'][start:end,0]
        x=(projected[:,0,None]-centres[None,start:end,0])/scale
        y=(projected[:,1,None]-centres[None,start:end,1])/scale
        squared=x*x+y*y;visible=squared<=9.
        if float32_clips:
            delta=world.astype(np.float32)[:,None,:]-draw['positions'][None,start:end,:]
            cx=np.sum(delta*u[None,start:end,:].astype(np.float32),axis=-1)/scale
            cy=np.sum(delta*v[None,start:end,:].astype(np.float32),axis=-1)/scale
        for angle_name,split_name in zip(CLIP_FIELDS[::2],CLIP_FIELDS[1::2]):
            a=draw[angle_name][start:end,0].astype(float)
            b=draw[split_name][start:end,0]
            if float32_clips:
                a32=a.astype(np.float32)
                visible&=(a[None,:]==4.)|(cx*np.cos(a32)+cy*np.sin(a32)>=b)
            else:visible&=(a[None,:]==4.)|(x*np.cos(a)+y*np.sin(a)>=b)
        alpha=.99*np.exp(-.5*squared)*visible
        result[:,start:end]=-np.log1p(-alpha)
    return result


def _coverage_target(reference,boundary,tolerance):
    # Preserve weak support relatively, use a stricter absolute bound near
    # boundaries, and never deliberately cross the renderer's .5 depth threshold.
    allowed=np.minimum(np.where(boundary,min(.01,tolerance),tolerance),.1*reference)
    target=reference-allowed
    strong=reference>=.5
    # The .5 contour falls between finite probes. Preserve its supporting GS
    # over a wider transition band, not merely the sampled threshold crossing.
    target[strong]=np.maximum(target[strong],np.minimum(reference[strong],.75))
    transition=boundary&(reference>=.25)&(reference<.75)
    target[transition]=reference[transition]
    return target


def _prune_pending_children(draw, origin, basis, inverse, domains, tolerance):
    """Remove only redundant rows, then restore rows failing held-out probes.

    No positions/scales/clips are fitted or moved. Exceeding the evaluation
    budget keeps every child rather than reducing the checking resolution.
    """
    started=perf_counter();count=len(draw['positions'])
    stats=dict(before=count,after=count,removed=0,restored=0,decision_probes=0,
               validation_probes=0,max_drop=0.,boundary_max_drop=0.,new_holes=0,
               tolerance=float(tolerance),boundary_tolerance=min(.01,float(tolerance)),
               skip_reason='',milliseconds=0.)
    def done(indices):
        stats.update(after=len(indices),removed=count-len(indices),milliseconds=1000*(perf_counter()-started))
        return {name:value[indices].copy() for name,value in draw.items()},stats
    all_rows=np.arange(count)
    if count<2 or tolerance==0.:
        stats['skip_reason']='small_or_disabled';return done(all_rows)
    if count>512:
        stats['skip_reason']='probe_budget';return done(all_rows)
    points,boundary=_prune_probes(draw,origin,basis,inverse,domains,0)
    checks,check_boundary=_prune_probes(draw,origin,basis,inverse,domains,1)
    if points is None or checks is None or count*(len(points)+2*len(checks))>_PRUNE_MAX_EVALUATIONS:
        stats['skip_reason']='probe_budget';return done(all_rows)
    stats.update(decision_probes=len(points),validation_probes=len(checks))
    from cutsim.rendering.coverage_sparse import algorithm, prune, GraphBudgetExceeded
    if algorithm() == 'resident':
        from cutsim.rendering.coverage_resident import prune as resident_prune
        return done(resident_prune(draw,points,boundary,checks,check_boundary,origin,basis,tolerance,stats))
    if algorithm() == 'sparse':
        try:
            return done(prune(draw,points,boundary,checks,check_boundary,origin,basis,tolerance,stats))
        except GraphBudgetExceeded:
            # Dense overlap can make an index graph larger than the matrix.
            # Preserve every check and use the bounded reference in that case.
            stats.update(coverage_algorithm='dense',sparse_fallback='graph_budget')
    attenuation=_probe_attenuation(draw,points,origin,basis)
    total=np.sum(attenuation,axis=1,dtype=float)
    reference=-np.expm1(-total)
    target=_coverage_target(reference,boundary,tolerance)
    required=-np.log1p(-target)
    transmission=np.exp(-total)
    impact=np.array([np.sum(transmission*np.expm1(attenuation[:,i].astype(float))) for i in all_rows])
    keep=np.ones(count,bool)
    for row in np.argsort(impact,kind='stable'):
        trial=total-attenuation[:,row]
        if np.all(trial>=required-1e-10):
            keep[row]=False;total=trial
    if np.all(keep):return done(all_rows)
    validation=np.concatenate((_probe_attenuation(draw,checks,origin,basis),
        _probe_attenuation(draw,checks,origin,basis,float32_clips=True)),axis=0)
    check_boundary=np.tile(check_boundary,2)
    stats['validation_probes']=2*len(checks)
    full=np.sum(validation,axis=1,dtype=float)
    check_reference=-np.expm1(-full)
    check_target=_coverage_target(check_reference,check_boundary,tolerance)
    check_required=-np.log1p(-check_target)
    current=np.sum(validation[:,keep],axis=1,dtype=float)
    while True:
        deficit=check_required-current
        worst=int(np.argmax(deficit))
        if deficit[worst]<=1e-10:break
        removed=np.flatnonzero(~keep)
        if not len(removed):break
        row=int(removed[np.argmax(validation[worst,removed])])
        keep[row]=True;current+=validation[:,row];stats['restored']+=1
    # Restoration can only improve the decision-point checks. Report both sets,
    # not just the points used to choose deletions.
    decision_after=-np.expm1(-np.sum(attenuation[:,keep],axis=1,dtype=float))
    check_after=-np.expm1(-current)
    drops=np.concatenate((reference-decision_after,check_reference-check_after))
    edge=np.concatenate((boundary,check_boundary))
    stats.update(max_drop=float(np.max(drops,initial=0.)),
        boundary_max_drop=float(np.max(drops[edge],initial=0.)),
        new_holes=int(np.count_nonzero((reference>=.5)&(decision_after<.5)))+
                  int(np.count_nonzero((check_reference>=.5)&(check_after<.5))))
    return done(np.flatnonzero(keep))


def build_pending_fallback(record, regions, *, max_children=2048, base_depth=3, max_depth=7,
                           min_child_sigma=_MIN_CHILD_SIGMA, prune=True, prune_tolerance=.02,
                           preserve_simple_regions=False, stock_planes=None):
    """Return ordinary public GS attributes plus an honest completion flag.

    Resolution, child count and minimum child sigma are bounded per parent.
    Empty domains resolve to no draws. Exhaustion retains safe partial output
    and an honest incomplete flag. Size-only stops use a Top-2 approximation
    without a marker; other failures retain the unresolved marker.
    min_child_sigma is in world units; zero disables only the size floor.
    The parent Gaussian's alpha profile is deliberately not fitted: each child
    uses the existing fixed .99 alpha and inherits its color, frame and surface ID.
    """
    from cutsim.rendering.clipping import local_planes_to_clip_values, _sweep_top_two
    from cutsim.rendering.stock_clipping import terminal_stock_selection

    if max_children < 1 or base_depth < 0 or max_depth < base_depth:
        raise ValueError('Invalid pending-clip subdivision limits')
    if not np.isfinite(min_child_sigma) or min_child_sigma < 0.:
        raise ValueError('Minimum child sigma must be finite and nonnegative')
    if not np.isfinite(prune_tolerance) or not 0.<=prune_tolerance<.5:
        raise ValueError('Pruning tolerance must be finite and in [0,.5)')
    sigma = float(record['scale'][0])
    if not np.isfinite(sigma) or sigma <= 0.:
        raise ValueError('Pending clip fallback requires positive finite sigma')
    if len(record['scale']) > 1 and record['scale'][1] != record['scale'][0]:
        raise ValueError('Pending clip fallback requires circular GS')
    if 'coverage_logit' in record and np.any(record['coverage_logit'] != np.float32(np.log(.99 / .01))):
        raise ValueError('Variable alpha is incompatible with fixed GS alpha 0.99')
    origin = np.asarray(record['positions'], float)
    u, v = _render_axes(np.asarray(record['rot']).reshape(1, 4))
    basis = sigma * np.column_stack((u[0], v[0]))
    inverse = np.linalg.pinv(basis)
    # Cover Float32 world-position and boundary encoding roundoff conservatively.
    guard = max(2e-6, 8 * np.finfo(np.float32).eps * (np.max(np.abs(origin)) / sigma + 3.))
    theta = np.arange(64) * (2. * np.pi / 64)
    disk = 3. * np.column_stack((np.cos(theta), np.sin(theta)))
    children = []; simple_children=0
    complete = True
    incomplete_reasons = set()
    polygons = 0
    domains = []
    approximations = []
    subdivision=dict(min_child_sigma=float(min_child_sigma),size_stopped=0,coverage_splits=0,
                     size_discarded=0,top_two_children=0,max_depth_used=0,
                     local_boundary_stops=0,collective_coverage_stops=0)
    # Keep an already smaller source at its original size. Never enlarge it to
    # the floor or subdivide it further. Round a new child's floor UP to Float32.
    floor_sigma=np.float32(min(float(min_child_sigma),sigma))
    if float(floor_sigma)<min(float(min_child_sigma),sigma):
        floor_sigma=np.nextafter(floor_sigma,np.float32(np.inf))
    from cutsim.rendering.fallback_kernels import enabled, build_subdivision, subdivision_policy, early_cell_cover
    local_boundaries, collective = subdivision_policy()
    if enabled():
        leaves, domains, counts = build_subdivision(regions, disk, origin, basis, inverse,
            sigma, guard, floor_sigma, max_children, base_depth, max_depth,
            min_child_sigma, preserve_simple_regions)
        polygons, simple_children = int(counts[0]), int(counts[1])
        subdivision.update(size_stopped=int(counts[2]), coverage_splits=int(counts[3]),
                           top_two_children=int(counts[4]), max_depth_used=int(counts[5]),
                           local_boundary_stops=int(counts[7]),collective_coverage_stops=int(counts[8]))
        incomplete_reasons = {name for bit,name in ((1,'budget'),(2,'min_size'),(4,'numerical')) if counts[6]&bit}
        complete = not incomplete_reasons
        import os
        terminal = None
        if os.environ.get('CUTSIM_RENDER_REUSE','1')!='0':
            from cutsim.rendering.boundary_batch import terminal_many
            terminal=iter(terminal_many([(p,s,q) for p,s,q,top,_ in leaves if top],
                                        stock_planes,origin,inverse,sigma))
        # Keep canonical Top-2 tie-breaking and Float32 angle encoding unchanged.
        encoded_planes=[]
        for position, child_sigma, local, top_two, region_index in leaves:
            if top_two:
                candidates = local
                if terminal is None:
                    local,selected,candidates=terminal_stock_selection(candidates,stock_planes,origin,inverse,
                        position,float(child_sigma)/sigma)
                else:local,selected,candidates=next(terminal)
                approximations.append(dict(child_row=len(children), region=region_index,
                    candidate_planes=candidates.copy(), selected=tuple(selected),
                    omitted=tuple(i for i in range(len(candidates)) if i not in selected)))
            encoded_planes.append(local)
            children.append((position,child_sigma,None))
        encoded=_encode_leaf_planes(encoded_planes)
        children=[(p,s,encoded[i]) for i,(p,s,_) in enumerate(children)]
    else:
        for region_index,region in enumerate(regions):
            planes = []
            polygon = disk.copy()
            for plane in np.asarray(region, float).reshape(-1, 3):
                length = np.linalg.norm(plane[:2])
                if length < 1e-14:
                    if plane[2] < 0.:
                        polygon = np.empty((0, 2))
                        break
                    continue
                normalized = plane / length
                planes.append(normalized)
                polygon = _clip_polygon(polygon, normalized)
                if not len(polygon):
                    break
            if _area(polygon) <= guard * guard:
                continue
            polygons += 1
            planes = np.asarray(planes).reshape(-1, 3)
            domains.append((polygon.copy(),planes.copy()))
            if preserve_simple_regions and len(planes)<=2:
                # An exact OR branch already fits the native schema. Retain the
                # parent's Gaussian, including its central coverage, instead of
                # replacing it with displaced disks whose tails can leave holes.
                # This is the same overlap convention as the existing two-branch
                # concave renderer, extended to more branches after Boolean
                # partition-plane elimination. No material region is enlarged.
                if len(children)>=max_children:
                    complete=False;incomplete_reasons.add('budget');break
                children.append((origin.astype(np.float32),np.float32(sigma),
                                 local_planes_to_clip_values(planes)))
                simple_children+=1
                continue
            lo, hi = polygon.min(axis=0), polygon.max(axis=0)
            side = float(np.max(hi - lo))
            middle = .5 * (lo + hi)
            stack = [(polygon, middle - side / 2., side, 0)]
            while stack:
                poly, corner, width, depth = stack.pop()
                if _area(poly) <= guard * guard:
                    continue
                if len(children) >= max_children:
                    complete = False
                    incomplete_reasons.add('budget')
                    break
                subdivision['max_depth_used']=max(subdivision['max_depth_used'],depth)
                center = np.mean(poly, axis=0)
                position = (origin + basis @ center).astype(np.float32)
                center = inverse @ (position.astype(float) - origin)
                clearance = planes[:, :2] @ center + planes[:, 2]
                # Finite-domain children may use a wider CELL covering disk only
                # where the original parent disk and every omitted plane permit it.
                # Neither the parent Gaussian nor the user's size setting changes.
                radius = min((3.5 if preserve_simple_regions else 2.25) * width,
                             3. - np.linalg.norm(center) - guard)
                crossing = np.count_nonzero(clearance < radius + guard)
                order = np.argsort(clearance, kind='stable')
                exact_radius=min(radius,clearance[order[2]]-guard) if len(order)>2 else radius
                coverage_split=(preserve_simple_regions and depth<max_depth and
                    _central_coverage_needs_split(poly,center,max(0.,exact_radius)/3.*(1.-1e-6)))
                wants_split = depth < base_depth or ((crossing > 2 or coverage_split) and depth < max_depth)
                # A half-width cell can produce at most this sigma before clipping
                # by the parent disk/third plane. Stop BEFORE allocating four cells.
                # Actual emitted sizes are checked separately after those limits.
                can_split = sigma * (.375 * width) * (1. - 1e-6) >= min_child_sigma
                size_stopped = wants_split and not can_split
                if size_stopped:
                    subdivision['size_stopped']+=1
                    complete=False
                    incomplete_reasons.add('min_size')
                if wants_split and can_split:
                    if preserve_simple_regions and local_boundaries and collective:
                        coarse=early_cell_cover(poly,corner,width,depth,planes,origin,basis,inverse,
                            sigma,guard,min_child_sigma,base_depth,max_depth,max_children-len(children))
                        if len(coarse):
                            for p,s,q,top,_ in coarse:children.append((p,s,local_planes_to_clip_values(q)))
                            subdivision['local_boundary_stops']+=1
                            subdivision['collective_coverage_stops']+=len(coarse)
                            continue
                    if coverage_split:subdivision['coverage_splits']+=1
                    half = width / 2.
                    for dx, dy in ((1, 1), (0, 1), (1, 0), (0, 0)):
                        cell = corner + half * np.array([dx, dy])
                        piece = poly
                        for plane in ((1., 0., -cell[0]), (-1., 0., cell[0] + half),
                                      (0., 1., -cell[1]), (0., -1., cell[1] + half)):
                            piece = _clip_polygon(piece, np.asarray(plane))
                        if len(piece):
                            stack.append((piece, cell, half, depth + 1))
                    continue
                if radius <= 3. * guard or np.any(clearance < -guard):
                    complete = False
                    incomplete_reasons.add('numerical')
                    continue
                child_sigma = np.float32(sigma * exact_radius / 3. * (1. - 1e-6))
                top_two = min_child_sigma>0. and (size_stopped or float(child_sigma)<float(min_child_sigma))
                if top_two:
                    if not size_stopped:subdivision['size_stopped']+=1
                    subdivision['top_two_children']+=1
                    complete=False
                    incomplete_reasons.add('min_size')
                    child_sigma=np.float32(max(float(floor_sigma),sigma*radius/3.*(1.-1e-6)))
                elif exact_radius <= 3.*guard:
                    complete=False
                    incomplete_reasons.add('numerical')
                    continue
                ratio = float(child_sigma) / sigma
                if top_two:
                    # Reuse analytic Sweep/stock scoring: finite-edge Gaussian peak
                    # first, then peak x incremental rejection for a diverse second.
                    # No peer-Gaussian density is fabricated for these constraints.
                    candidates=np.column_stack((planes[:,:2],clearance/ratio))
                    local,selected,candidates=terminal_stock_selection(candidates,stock_planes,origin,inverse,
                        position,float(child_sigma)/sigma)
                    approximations.append(dict(child_row=len(children),region=region_index,
                        candidate_planes=candidates.copy(),selected=tuple(selected),
                        omitted=tuple(i for i in range(len(candidates)) if i not in selected)))
                else:
                    selected = order[:2]
                    # Exact strip domains already own their complete boundaries.
                    # Preserve their position, as in ordinary native clipping;
                    # eroding every selected edge by the numerical disk guard
                    # creates cracks between adjacent pieces. The guard remains
                    # on the omitted-third-plane and parent-disk radius bounds.
                    offset=clearance[selected] if preserve_simple_regions else clearance[selected]-guard
                    local = np.column_stack((planes[selected, :2] * ratio, offset))
                values = local_planes_to_clip_values(local)
                children.append((position, child_sigma, values))
            if 'budget' in incomplete_reasons:
                break
    count = len(children)
    names = [name for name in GS_FIELDS if name in record]
    out = {name: np.repeat(np.asarray(record[name])[None, :], count, axis=0) for name in names}
    out['positions'] = np.asarray([c[0] for c in children], np.float32).reshape(-1, 3)
    out['scale'] = np.asarray([c[1] for c in children], np.float32).reshape(-1, 1)
    for column, name in enumerate(CLIP_FIELDS):
        out[name] = np.asarray([c[2][column] for c in children], np.float32).reshape(-1, 1)
    validate_gaussian_attributes(out)
    pruning=dict(before=count,after=count,removed=0,restored=0,decision_probes=0,validation_probes=0,
                 max_drop=0.,boundary_max_drop=0.,new_holes=0,skip_reason='disabled' if not prune else 'incomplete',milliseconds=0.)
    if simple_children==count and count:
        # Every remaining analytic branch owns some region not covered by a
        # single other branch (exact absorption was already done). Keep this
        # representation; sampling-based pruning is for generated child disks.
        pruning['skip_reason']='analytic_union_branches'
    elif prune and complete:
        out,pruning=_prune_pending_children(out,origin,basis,inverse,domains,float(prune_tolerance))
    return dict(attributes=out, complete=complete, polygons=polygons, children=len(out['positions']),
                children_before_prune=count,pruning=pruning,subdivision=subdivision,
                incomplete_reasons=tuple(sorted(incomplete_reasons)),
                min_size_approximations=approximations,simple_region_children=simple_children,
                size_approximation=not complete and incomplete_reasons=={'min_size'},abandoned=False)


def pending_fallbacks(arr):
    """Sparse per-parent records survive native caching and playback row selection."""
    return {int(row): data['pending_fallback']
            for row, data in (getattr(arr, 'approximated', None) or {}).items()
            if 'pending_fallback' in data}


def unresolved_clip_mask(arr):
    """Geometrically incomplete parents, including accepted Top-2 approximations."""
    mask = np.asarray(arr.get('clip_pending', np.zeros(len(arr['positions']), bool)), bool).copy()
    for row, fallback in pending_fallbacks(arr).items():
        if fallback['complete']:
            mask[row] = False
    return mask


def abandoned_clip_mask(arr):
    """Unresolved rows whose cached fallback is marked abandoned."""
    mask=np.zeros(len(arr['positions']),bool)
    for row,fallback in pending_fallbacks(arr).items():
        mask[row]=fallback.get('abandoned',False)
    return mask & unresolved_clip_mask(arr)


def size_approximation_mask(arr):
    """Parents accepted with terminal Top-2, without budget/numerical failures."""
    mask=np.zeros(len(arr['positions']),bool)
    for row,fallback in pending_fallbacks(arr).items():
        mask[row]=fallback.get('size_approximation',False)
    return mask & unresolved_clip_mask(arr)


def clip_marker_mask(arr):
    """Actionable unresolved parents, excluding accepted size-limit results."""
    return unresolved_clip_mask(arr) & ~(abandoned_clip_mask(arr)|size_approximation_mask(arr))


def append_pending_fallbacks(draw, arr):
    """Append cached draws without modifying one-record-per-geometric-parent data."""
    extras = [entry['attributes'] for row, entry in sorted(pending_fallbacks(arr).items())
              if arr['clip_pending'][row] and not arr.get('clip_empty', np.zeros(len(arr['positions']), bool))[row]
              and entry['children']]
    if not extras:
        return draw
    return {name: np.concatenate([draw[name]] + [part[name] for part in extras], axis=0)
            for name in draw}
