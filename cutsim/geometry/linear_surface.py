"""Surface-only samples of the analytic XYZ sweep; no volume grid or pose steps."""
import itertools
import math
import numpy as np

from cutsim.core.splat_cloud import SplatCloud
from cutsim.core.regions import ExactSurfaceDomains
from cutsim.geometry.frames import tangent_frames
from cutsim.geometry.strip_intersections import halfplanes


def unsupported_ball_cap(sweep, stock_top, z, material_normals):
    """An axial lid facing down at/above the stock top has no material above it."""
    eps=64.*np.finfo(float).eps*max(1.,abs(float(stock_top)),abs(float(sweep.z_hi)))
    normals=np.asarray(material_normals)
    axial=(normals[...,2]==-1.) & np.all(normals[...,:2]==0.,axis=-1)
    return ((sweep.kind=='ball') & axial &
            (np.asarray(z)>=stock_top-eps))


def linear_surface_candidates(sweep, h, stock_lo, stock_hi, cache=None, *, ball_angle_deg=None,
                              include_barrel_junctions=False):
    h = max(float(h), 1e-6)
    lo, hi = sweep.world_aabb()
    lo, hi = np.maximum(lo, np.asarray(stock_lo)-2.*h), np.minimum(hi, np.asarray(stock_hi)+2.*h)
    if np.any(lo > hi): return SplatCloud.empty()
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    points, normals, domains, supports = [], [], [], []

    def template(key, build):
        return build() if cache is None else cache._get(key, build)

    def emit(p, outward, constraints, support=h):
        p = np.asarray(p, float).reshape(-1, 3)
        if not len(p): return
        outward = np.broadcast_to(outward, p.shape).copy()
        outward /= np.maximum(np.linalg.norm(outward, axis=1)[:, None], 1e-30)
        tolerance = 1e-8*max(1., float(np.max(np.abs(corners))))
        valid = np.all((p >= lo-tolerance) & (p <= hi+tolerance), axis=1)
        if len(constraints): valid &= np.all(p@constraints[:, :3].T+constraints[:, 3] >= -tolerance, axis=1)
        # This sweep is convex. A visible normal must support the whole solid,
        # including the other primitive pieces, not only the sampled piece.
        valid &= np.abs(sweep.support(outward)-np.sum(p*outward, axis=1)) <= tolerance
        p, outward = p[valid], outward[valid]
        points.extend(p); normals.extend(-outward)
        supports.extend([support]*len(p))
        domains.extend([ExactSurfaceDomains((constraints.copy(),)) for _ in range(len(p))])

    for part in sweep.parts:
        # Planar faces are clipped in their own plane by the same finite
        # primitive used for old-GS intersections. This is a surface lattice.
        for plane in part.planes:
            length = np.linalg.norm(plane[:3]); outward = -plane[:3]/length
            if (outward[2]==1. and np.all(plane[:2]==0.) and
                    unsupported_ball_cap(sweep,stock_hi[2],-plane[3]/plane[2],-outward)):
                continue
            signed = corners@plane[:3]+plane[3]
            if signed.min() > 1e-9 or signed.max() < -1e-9: continue
            center = (lo+hi)*.5
            center -= ((center@plane[:3]+plane[3])/(length*length))*plane[:3]
            u, v = tangent_frames(outward[None]); basis = np.stack((u[0], v[0]))
            xy = (corners-center)@basis.T
            low, high = xy.min(axis=0), xy.max(axis=0)
            rectangle = np.array([[low[0],low[1]],[high[0],low[1]],
                                  [high[0],high[1]],[low[0],high[1]]])
            polygon = part.section(center, basis, rectangle, max(1e-8, .005*h))
            if len(polygon) < 3: continue
            ps = halfplanes(polygon)
            axes = [np.linspace(low[i], high[i], max(2, int(math.ceil((high[i]-low[i])/h))+1)) for i in range(2)]
            x, y = np.meshgrid(*axes, indexing='ij')
            samples = np.column_stack((x.ravel(), y.ravel()))
            samples = samples[np.all(samples@ps[:, :2].T+ps[:, 2]>=-1e-10, axis=1)]
            # Explicit boundary rows prevent the surface lattice phase from
            # omitting a narrow planar face or its rim.
            edge = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge.extend(a+np.linspace(0., 1., max(1, int(np.ceil(np.linalg.norm(b-a)/(.75*h))))+1)[:, None]*(b-a))
            samples = np.vstack((samples, np.asarray(edge).reshape(-1, 2)))
            world_n = ps[:, :2]@basis
            constraints = np.vstack((part.planes, np.column_stack((world_n, ps[:, 2]-world_n@center))))
            emit(center+samples@basis, outward, constraints)
        if part.matrix is None: continue
        if part.name == 'ramp_rim' or part.name.startswith('cut_disk_'):
            # Parameterise the actual translated horizontal rim directly.
            # An eigendecomposition of its world quadric loses the null axis
            # for shallow ramps (condition number grows as 1/dz^2).
            if part.name.startswith('cut_disk_'):
                a = part.center
                b = a+sweep.b_xyz-sweep.a_xyz
                radius = part.radius
            else:
                offset = np.array([0.,0.,sweep.radius if sweep.kind=='ball' else 0.])
                a,b = sweep.a_xyz+offset,sweep.b_xyz+offset
                radius = sweep.radius
            if a[2]>b[2]: a,b=b,a
            d=b-a;start,stop=0.,1.
            margin=np.array([radius,radius,0.])
            for k in range(3):
                lower,upper=lo[k]-margin[k],hi[k]+margin[k]
                if d[k]==0.:
                    if not lower<=a[k]<=upper:start,stop=1.,0.;break
                else:
                    t0,t1=(lower-a[k])/d[k],(upper-a[k])/d[k]
                    start,stop=max(start,min(t0,t1)),min(stop,max(t0,t1))
            if start>stop:continue
            count=max(16,int(math.ceil(2.*math.pi*radius/h)))
            circle=template(('xyz_circle',count),lambda:np.column_stack((
                np.cos(np.arange(count)*2.*math.pi/count),np.sin(np.arange(count)*2.*math.pi/count))))
            times=np.linspace(start,stop,max(2,int(np.ceil((stop-start)*np.linalg.norm(d)/h))+1))
            rim=np.column_stack((radius*circle,np.zeros(count)))
            p=(a+rim[:,None,:]+times[None,:,None]*d).reshape(-1,3)
            outward=np.column_stack((circle,-(circle@d[:2])/d[2]))
            emit(p,np.repeat(outward,len(times),axis=0),part.planes)
            continue
        values, axes = np.linalg.eigh(part.matrix)
        positive = values > 1e-11
        rank = int(positive.sum())
        if rank == 2:
            radial = axes[:, positive]*(part.radius/np.sqrt(values[positive]))
            axis = axes[:, ~positive][:, 0]
            q = (corners-part.center)@axis
            along = np.linspace(q.min(), q.max(), max(2, int(np.ceil(np.ptp(q)/h))+1))
            ball_barrel = sweep.kind == 'ball' and part.name in ('ball_straight', 'cut_ball_barrel')
            if include_barrel_junctions and ball_barrel:
                # Share the spherical end fan's tangent ring. A lattice phased
                # from the stock window can stop short of either junction.
                ends = np.array([0., (sweep.b_xyz-sweep.a_xyz)@axis])
                along = np.unique(np.r_[along, ends[(ends >= q.min()) & (ends <= q.max())]])
            sphere_barrel = ball_barrel and ball_angle_deg is not None
            count = (max(4, int(math.ceil(360./ball_angle_deg))) if sphere_barrel else
                     max(16, int(math.ceil(2.*math.pi*part.radius/h))))
            circle = template(('xyz_circle', count), lambda: np.column_stack((
                np.cos(np.arange(count)*2.*math.pi/count), np.sin(np.arange(count)*2.*math.pi/count))))
            rim = circle@radial.T
            p = part.center+rim[:, None, :]+along[None, :, None]*axis
            p = p.reshape(-1, 3)
            support = max(h, part.radius*math.radians(ball_angle_deg)) if sphere_barrel else h
            emit(p, (p-part.center)@part.matrix, part.planes, support)
        elif rank == 3:
            angular = sweep.kind == 'ball' and ball_angle_deg is not None
            curved_h = part.radius*math.radians(ball_angle_deg) if angular else h
            count = max(2, int(math.ceil(180./ball_angle_deg))) if angular else max(8, int(math.ceil(math.pi*part.radius/h)))
            def sphere():
                rows = [[0.,0.,-1.],[0.,0.,1.]]
                if angular:
                    quarter = np.deg2rad(np.r_[np.arange(0.,90.,ball_angle_deg),90.])
                    latitudes = np.unique(np.r_[quarter, np.pi-quarter])[1:-1]
                else:
                    latitudes = np.arange(1,count)*math.pi/count
                for angle in latitudes:
                    n = max(8, int(math.ceil(2.*math.pi*part.radius*math.sin(angle)/curved_h)))
                    theta = np.arange(n)*2.*math.pi/n
                    rows.extend(np.column_stack((np.sin(angle)*np.cos(theta),
                                np.sin(angle)*np.sin(theta), np.full(n,np.cos(angle)))))
                return np.asarray(rows)
            unit = template(('xyz_sphere', sweep.radius, curved_h, count), sphere)
            p = part.center+unit@(axes*(part.radius/np.sqrt(values))).T
            emit(p, (p-part.center)@part.matrix, part.planes, curved_h)
        else:
            raise ArithmeticError('Degenerate XYZ sweep quadric')
    if not points: return SplatCloud.empty()
    p, n = np.asarray(points), np.asarray(normals)
    key = np.column_stack((np.rint(p/(h*1e-7)), np.rint(n*1e8))).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True); first.sort()
    out = SplatCloud(p[first], n[first], np.asarray(supports)[first],
                     np.full(len(first), sweep.tool_id, np.int32), np.zeros(len(first), np.int16))
    for i, old in enumerate(first): out.sweep_domains[i] = domains[old]
    return out
