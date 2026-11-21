"""Finite GS/analytic swept-strip intersections; no volume lattice or SDF probes.

Lengths are mm.
Conic sections are evaluated on each actual GS plane, not a representative ray.
"""
from dataclasses import dataclass, field
import math
import numpy as np

from cutsim.core.sweep import BallSweep, FlatSweep
from cutsim.geometry.frames import tangent_frames
from cutsim.geometry.tool_profile import _sweep_uv_frame
from cutsim.core.regions import ExactSurfaceDomains
from cutsim.geometry import strip_kernels


def area(poly):
    if len(poly) < 3:
        return 0.
    q = np.roll(poly, -1, axis=0)
    return abs(float(np.sum(poly[:, 0]*q[:, 1]-poly[:, 1]*q[:, 0])))*.5


def halfplanes(poly):
    edge = np.roll(poly, -1, axis=0)-poly
    n = np.column_stack((-edge[:, 1], edge[:, 0]))
    length = np.linalg.norm(n, axis=1)
    good = length > 0.
    n = n[good]/length[good, None]
    return np.column_stack((n, -np.sum(n*poly[good], axis=1)))


def clip(poly, plane):
    """Closed halfplane clipping; no minimum-length/area rejection."""
    if strip_kernels.enabled():
        return strip_kernels.clip_polygon(poly, np.asarray(plane, dtype=np.float64))
    if not len(poly):
        return poly
    f = poly @ plane[:2]+plane[2]
    roundoff = 64.*np.finfo(float).eps*(1.+np.max(np.abs(poly))*np.sum(np.abs(plane[:2]))+abs(plane[2]))
    f[np.abs(f) <= roundoff] = 0.
    inside = f >= 0.
    if inside.all(): return poly
    if not inside.any(): return np.empty((0, 2))
    out = []
    for i in range(len(poly)):
        j = (i+1) % len(poly)
        if inside[i]: out.append(poly[i])
        if inside[i] != inside[j]:
            out.append(poly[i]+f[i]/(f[i]-f[j])*(poly[j]-poly[i]))
    return np.asarray(out).reshape(-1, 2)


def intersect(poly, planes):
    if strip_kernels.enabled():
        return strip_kernels.intersect_polygon(poly, np.asarray(planes, dtype=np.float64).reshape(-1, 3))
    for plane in planes:
        poly = clip(poly, plane)
        if len(poly) < 3: break
    return poly


def ordered(points):
    if len(points) < 3: return np.empty((0, 2))
    # A conic and a convex polygon have a convex intersection. Monotone hull
    # removes duplicate quadratic endpoints and collinear arc endpoints.
    if strip_kernels.enabled():
        return strip_kernels.hull(np.asarray(points, dtype=np.float64))
    points = sorted(set(map(tuple, np.asarray(points))))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 0.: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 0.: upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1]).reshape(-1, 2)


def conic_section(poly, matrix, linear, constant, chord_error):
    """Clip a convex polygon by x'Mx+2q'x+c<=0, M positive semidefinite.

    Circle/ellipse edge roots are analytic. Only arc representation is refined;
    no ambiguous point is converted to an unknown or silently skipped.
    """
    if len(poly) < 3: return np.empty((0, 2))
    ev, axes = np.linalg.eigh(matrix)
    eps = 64.*np.finfo(float).eps
    rank = ev > eps
    if not rank.any():
        return clip(poly, np.r_[-2.*linear, -constant])
    q = axes.T @ linear
    if rank.sum() == 1:
        k = int(np.flatnonzero(rank)[0]); other = 1-k
        if q[other] != 0. or ev[other] > 0.:
            return _near_parallel_section(poly, axes[:, [k,other]],
                                          ev[[k,other]], q[[k,other]], constant, chord_error)
        mid = -q[k]/ev[k]; rad2 = mid*mid-constant/ev[k]
        if rad2 <= 0.: return np.empty((0, 2))
        radius = math.sqrt(rad2); n = axes[:, k]
        return intersect(poly, [np.r_[n, radius-mid], np.r_[-n, radius+mid]])
    center = -axes @ (q/ev)
    rhs = float(q @ (q/ev)-constant)
    if rhs <= 0.: return np.empty((0, 2))
    radii = np.sqrt(rhs/ev)
    transform = axes * radii
    local = (poly-center) @ axes / radii
    norms = np.sum(local*local, axis=1)
    if np.all(norms <= 1.): return poly
    points = list(poly[norms <= 1.]); angles = list(np.arange(5)*math.pi*.5)
    for a, b in zip(local, np.roll(local, -1, axis=0)):
        d = b-a; aa = float(d@d)
        if aa == 0.: continue
        bb = float(a@d); cc = float(a@a)-1.
        disc = bb*bb-aa*cc
        if disc < 0.: continue
        root = math.sqrt(disc)
        for t in ((-bb-root)/aa, (-bb+root)/aa):
            if 0. <= t <= 1.:
                z = a+t*d
                points.append(center+transform@z)
                angles.append(float(math.atan2(z[1], z[0]) % (2.*math.pi)))
    angles = sorted(set(angles)); planes = halfplanes(poly)
    # The maximum ellipse semi-axis bounds chord error in every direction.
    angle_step = 2.*math.acos(max(-1., min(1., 1.-chord_error/float(max(radii)))))
    angle_step = max(angle_step, 1e-8)
    for a, b in zip(angles[:-1], angles[1:]):
        mid = .5*(a+b)
        p = center+transform@np.array([math.cos(mid), math.sin(mid)])
        if np.all(planes[:, :2]@p+planes[:, 2] >= -1e-10):
            count = max(1, int(math.ceil((b-a)/angle_step)))
            theta = np.linspace(a, b, count+1)
            points.extend(center+np.column_stack((np.cos(theta), np.sin(theta)))@transform.T)
    return ordered(points)


def _real_roots(a,b,c):
    if a==0.:return [] if b==0. else [-c/b]
    discriminant=b*b-4.*a*c
    if discriminant<0.:return []
    root=math.sqrt(discriminant)
    v=-.5*(b+math.copysign(root,b))
    return [-b/(2.*a)] if v==0. else [v/a,c/v]


def _near_parallel_section(poly, axes, eigenvalues, linear, constant, error):
    """Bounded-coordinate conic solve without dividing by a tiny eigenvalue.

    Parameterise the two branches by their finite null-axis coordinate y.
    Exact polygon-edge roots delimit visible arcs. On each retained arc the
    range of x bounds its vertical chord deviation, including a parabolic tip.
    No small interval is discarded or certified using a representative line.
    """
    polygon=poly@axes; a,b=map(float,np.maximum(eigenvalues,0.)); q,r=map(float,linear)
    value=a*polygon[:,0]**2+b*polygon[:,1]**2+2*q*polygon[:,0]+2*r*polygon[:,1]+constant
    points=list(polygon[value<=0.]); ys=list(polygon[:,1])
    for p,end in zip(polygon,np.roll(polygon,-1,axis=0)):
        d=end-p
        for t in _real_roots(a*d[0]**2+b*d[1]**2,
                            2*(a*p[0]*d[0]+b*p[1]*d[1]+q*d[0]+r*d[1]),
                            a*p[0]**2+b*p[1]**2+2*q*p[0]+2*r*p[1]+constant):
            if 0.<=t<=1.:
                hit=p+t*d;points.append(hit);ys.append(float(hit[1]))
    low,high=float(polygon[:,1].min()),float(polygon[:,1].max())
    ys.extend(y for y in _real_roots(b,2*r,constant-q*q/a) if low<=y<=high)
    ys=sorted(set(ys)); planes=halfplanes(polygon)
    # The eigenbasis may reverse orientation; use the CCW hull for its planes.
    planes=halfplanes(ordered(polygon))
    def branch(y,sign):
        discriminant=q*q-a*(b*y*y+2*r*y+constant)
        return np.array([(-q+sign*math.sqrt(max(0.,discriminant)))/a,y])
    for lo,hi in zip(ys[:-1],ys[1:]):
        middle=.5*(lo+hi)
        if q*q-a*(b*middle*middle+2*r*middle+constant)<0.:continue
        for sign in (-1.,1.):
            probe=branch(middle,sign)
            if np.any(planes[:,:2]@probe+planes[:,2]<-1e-10):continue
            stack=[(lo,hi)]
            while stack:
                left,right=stack.pop(); pa,pb=branch(left,sign),branch(right,sign)
                extrema=[pa[0],pb[0]]
                if b>0. and left < -r/b < right:extrema.append(branch(-r/b,sign)[0])
                mid=.5*(left+right)
                if max(extrema)-min(extrema)<=error or mid==left or mid==right:
                    points.extend((pa,pb))
                else:stack.extend(((left,mid),(mid,right)))
    return ordered(np.asarray(points).reshape(-1,2)@axes.T)


@dataclass
class Primitive:
    name: str
    planes: np.ndarray
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    matrix: np.ndarray | None = None
    radius: float = 0.

    def section(self, p, basis, polygon, error):
        planes = np.column_stack((self.planes[:, :3]@basis.T,
                                  self.planes[:, :3]@p+self.planes[:, 3]))
        poly = intersect(polygon, planes)
        if self.matrix is None or len(poly) < 3: return poly
        d = p-self.center
        return conic_section(poly, basis@self.matrix@basis.T,
                             basis@self.matrix@d, float(d@self.matrix@d-self.radius**2), error)


def slab(axis, low, high):
    return [np.r_[axis, -low], np.r_[-axis, high]]


def swept_primitives(sw):
    """Exact finite quadric decomposition, including stationary/short shanks."""
    if hasattr(sw, 'parts'):
        return sw.parts
    if type(sw) not in (FlatSweep, BallSweep): raise TypeError(type(sw))
    u, v, w, length = _sweep_uv_frame(sw)
    a = np.r_[sw.a_xy, 0.]; b = np.r_[sw.b_xy, 0.]
    axial = slab(u, float(a@u), float(b@u))
    zlo = sw.center_z if type(sw) is BallSweep else sw.z_lo
    vertical = slab(w, zlo, sw.z_hi)
    result = []
    if sw.z_hi >= zlo:
        if length > 0.:
            result.append(Primitive('straight_prism', np.array(axial+slab(v, float(a@v)-sw.radius,
                                      float(a@v)+sw.radius)+vertical)))
        for i, c in enumerate((a, b) if length else (a,)):
            result.append(Primitive(f'vertical_cap_{i}', np.array(vertical), c,
                                    np.diag([1., 1., 0.]), sw.radius))
    if type(sw) is BallSweep:
        a = a+sw.center_z*w; b = b+sw.center_z*w
        if length > 0.:
            result.append(Primitive('rounded_straight', np.array(axial), a,
                                    np.eye(3)-np.outer(u, u), sw.radius))
        for i, c in enumerate((a, b) if length else (a,)):
            result.append(Primitive(f'ball_cap_{i}', np.empty((0, 4)), c, np.eye(3), sw.radius))
    return result


@dataclass
class Region:
    polygon: np.ndarray
    constraints: np.ndarray


def connected_region_groups(regions, tolerance=1e-10):
    """Connected residual components, not the convex pieces of a Boolean split.

    Closed edge/point contacts belong to the same component. Only the adjacency
    test uses the coordinate tolerance; no polygon or material plane expands.
    """
    count=len(regions)
    if not count:return []
    lower=np.array([r.polygon.min(axis=0) for r in regions])
    upper=np.array([r.polygon.max(axis=0) for r in regions])
    planes=[halfplanes(r.polygon) for r in regions]
    roots=list(range(count))
    def root(i):
        while roots[i]!=i:
            roots[i]=roots[roots[i]];i=roots[i]
        return i
    for i in range(count):
        neighbours=np.flatnonzero(np.all(lower<=upper[i]+tolerance,axis=1)&
                                  np.all(upper>=lower[i]-tolerance,axis=1))
        for j in neighbours:
            if j<=i or root(i)==root(int(j)):continue
            intersection=regions[i].polygon
            for plane in planes[j]:
                q=plane.copy();q[2]+=tolerance
                intersection=clip(intersection,q)
                if not len(intersection):break
            if len(intersection):roots[root(int(j))]=root(i)
    groups={}
    for i in range(count):groups.setdefault(root(i),[]).append(i)
    return list(groups.values())


def subtract(regions, cutter, p, basis, *, section_planes=None):
    """Disjoint convex difference; same cut polygons feed boundary records."""
    if len(cutter) < 3 or area(cutter) == 0.: return regions, []
    planes = halfplanes(cutter) if section_planes is None else section_planes
    result = []; cut = []
    worlds = [None]*len(planes)
    for region in regions:
        roundoff = 128.*np.finfo(float).eps*max(1., np.max(np.abs(region.polygon)))
        signed=region.polygon@planes[:, :2].T+planes[:, 2]
        # A separating supporting edge proves disjointness or zero-width
        # contact. A clipped line can otherwise acquire a spurious 1e-17 mm²
        # shoelace area and retire an unchanged GS on every repeated pass.
        # Use the existing Float64 distance roundoff (mm), not a minimum
        # feature area or an expansion of the cutter/remaining region.
        if np.any(np.max(signed,axis=0)<=roundoff):
            result.append(region);continue
        if np.all(signed >= -roundoff):
            cut.append(region.polygon)
            continue
        common = intersect(region.polygon, planes)
        if len(common) < 3 or area(common) == 0.:
            result.append(region); continue
        cut.append(common)
        rest = region.polygon; constraints = region.constraints; suffix = []
        for plane_index, plane in enumerate(planes):
            signed = rest@plane[:2]+plane[2]
            if np.all(signed >= -roundoff):
                # Already implied within the fixed finite footprint. In
                # particular, cutter-section clipping adds no artificial
                # footprint edges to the exported CPU domain.
                continue
            if np.all(signed <= roundoff):
                result.append(Region(rest, _constraint_suffix(constraints, suffix)))
                break
            outside = clip(rest, -plane)
            world = worlds[plane_index]
            if world is None:
                world_n = plane[:2]@basis
                world = np.r_[world_n, plane[2]-world_n@p]
                worlds[plane_index] = world
            if len(outside) >= 3 and area(outside) > 0.:
                result.append(Region(outside, _constraint_suffix(constraints, suffix, -world)))
            rest = clip(rest, plane)
            if len(rest) < 3 or area(rest) == 0.: break
            suffix.append(world)
    return result, cut


def _constraint_suffix(base, suffix, last=None):
    """Materialize each published region once, retaining every plane in order."""
    if not suffix and last is None: return base
    tail = suffix if last is None else [*suffix, last]
    return np.vstack((base, np.asarray(tail)))


def gs_frame(cloud, row):
    u, v = tangent_frames(cloud.normals[row:row+1])
    return np.stack((u[0], v[0]))


def footprint(cloud, row):
    radius = 3.*float(cloud.support[row])*(1.+.2*float(np.clip(cloud.seam_weight[row], 0., 1.)))
    theta = np.arange(32)*2.*math.pi/32.
    return radius*np.column_stack((np.cos(theta), np.sin(theta)))


def stock_planes(material):
    return np.array([plane for j in range(3) for plane in slab(np.eye(3)[j], material.stock_lo[j], material.stock_hi[j])])


def effective_regions(cloud, row, material, basis, *, footprint_polygon=None, stock=None, replay=None):
    p = cloud.positions[row]; poly = footprint(cloud, row) if footprint_polygon is None else footprint_polygon
    if getattr(cloud,'_strip_round_geometry',False):
        return [Region(poly,np.empty((0,4)))]
    domains = cloud.sweep_domains[row]
    if domains is None: domains = cloud.inherited_clip[row]
    result = []
    for domain in domains:
        world = np.vstack((stock_planes(material) if stock is None else stock, domain))
        local = np.column_stack((world[:, :3]@basis.T, world[:, :3]@p+world[:, 3]))
        clipped = intersect(poly, local) if replay is None else replay.intersect(local)
        if len(clipped) >= 3 and area(clipped) > 0.: result.append(Region(clipped, world))
    return result


@dataclass
class IntersectionRecord:
    stable_id: int
    row: int
    block: int
    tool_id: int
    normal: np.ndarray
    origin: np.ndarray
    basis: np.ndarray
    before: list
    remaining: list
    cuts: list
    primitive_ids: list
    state: str
    chord_error: float
    residual_components: list = field(default_factory=list)
    source_index: int = -1
    cutter_sections: list = field(default_factory=list)
    surface_source_index: int = -1
    surface_curve_point: object = None
    surface_curve_plane: object = None
    material_cap_contact: bool = False

    @property
    def intersection_lines(self):
        return [self.origin+poly@self.basis for poly in self.cuts]

    @property
    def domains(self):
        return ExactSurfaceDomains(region.constraints for region in self.remaining)
