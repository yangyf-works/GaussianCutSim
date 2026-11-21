"""Continuous XYZ line sweeps of a Z-aligned flat/ball mill.

The shaft extends to a fixed plane above the work.
Geometry is an analytic union of clipped planes and positive semidefinite
quadrics. ``sdf`` is a 1-Lipschitz signed field with the exact solid/zero set,
not generally a Euclidean signed distance. Spatial distance pruning must not
be applied to this field. There is no sampling of intermediate tool poses.
"""
from dataclasses import dataclass
from functools import cached_property
import math
import numpy as np


def _flat_parts(a, b, r, top, tool_id):
    from cutsim.geometry.strip_intersections import Primitive, slab, swept_primitives
    from cutsim.core.sweep import FlatSweep
    if a[2] > b[2]: a, b = b, a
    d = b-a; length = float(np.linalg.norm(d[:2])); z = np.array([0.,0.,1.])
    if length == 0. or d[2] == 0.:
        return list(swept_primitives(FlatSweep(a[:2],b[:2],r,float(a[2]),top,tool_id)))
    u = np.r_[d[:2]/length,0.]; v = np.array([-u[1],u[0],0.])
    bottom = z-(d[2]/length)*u
    result = [Primitive('ramp_prism',np.array(
        slab(u,float(a@u),float(b@u))+slab(v,float(a@v)-r,float(a@v)+r)+
        [np.r_[bottom,-bottom@a],np.r_[-z,top]]))]
    for i,p in enumerate((a,b)):
        result.append(Primitive(f'vertical_cap_{i}',np.array(slab(z,float(p[2]),top)),
                                np.r_[p[:2],0.],np.diag([1.,1.,0.]),r))
    shear = np.column_stack((np.eye(2),-d[:2]/d[2]))
    result.append(Primitive('ramp_rim',np.array(slab(z,float(a[2]),float(b[2]))),a.copy(),shear.T@shear,r))
    return result


def _ball_parts(a, b, radius):
    from cutsim.geometry.strip_intersections import Primitive, slab
    result=[];d=b-a;length=float(np.linalg.norm(d))
    if length:
        u=d/length
        result.append(Primitive('ball_straight',np.array(slab(u,float(a@u),float(b@u))),
                                a.copy(),np.eye(3)-np.outer(u,u),radius))
    for i,p in enumerate((a,b) if length else (a,)):
        result.append(Primitive(f'ball_cap_{i}',np.empty((0,4)),p.copy(),np.eye(3),radius))
    return result


def _flat_field(points, a, b, radius, top):
    """Analytic min_t max(radial(t)-r, tip_z(t)-z, z-top).

    The radial term is convex in t; the other moving term is affine. Its
    minimum is at an endpoint, the radial minimum, or an intersection of
    those two terms. The intersections are quadratic roots, not pose samples.
    Unlike a minimum of clipped primitive fields this has no internal zero
    sheets at the artificial decomposition caps.
    """
    p=np.asarray(points,float).reshape(-1,3);d=b-a;q=p[:,:2]-a[:2]
    den=float(d[:2]@d[:2]);along=q@d[:2]
    middle=np.clip(along/max(den,1e-30),0.,1.)
    offset=radius+a[2]-p[:,2]
    aa=den-d[2]*d[2];bb=-2.*(along+offset*d[2]);cc=np.sum(q*q,axis=1)-offset*offset
    if abs(aa)>1e-14*max(den+d[2]*d[2],1e-30):
        disc=bb*bb-4.*aa*cc
        root=np.sqrt(np.maximum(disc,0.))
        stable=-.5*(bb+np.copysign(root,bb))
        first=stable/aa
        second=np.divide(cc,stable,out=-bb/(2.*aa),where=stable!=0.)
        first=np.where(disc>=0.,first,0.);second=np.where(disc>=0.,second,0.)
    else:
        first=np.divide(-cc,bb,out=np.zeros(len(p)),where=bb!=0.);second=first
    times=np.column_stack((np.zeros(len(p)),np.ones(len(p)),middle,np.clip(first,0.,1.),np.clip(second,0.,1.)))
    delta=q[:,None,:]-times[:,:,None]*d[:2]
    radial_length=np.linalg.norm(delta,axis=2)
    radial=radial_length-radius
    axial=a[2]+times*d[2]-p[:,2,None]
    values=np.maximum(radial,axial)
    index=np.argmin(values,axis=1);rows=np.arange(len(p))
    t=times[rows,index];r=radial[rows,index];z=axial[rows,index]
    unit=delta[rows,index]/np.maximum(radial_length[rows,index,None],1e-30)
    gradient=np.zeros_like(p)
    active=r>=z;gradient[active,:2]=unit[active];gradient[~active,2]=-1.
    drdt=-unit@d[:2]
    weight=np.divide(d[2],d[2]-drdt,out=np.zeros(len(p)),where=np.abs(d[2]-drdt)>1e-14)
    tied=(np.abs(r-z)<1e-8*(1.+np.abs(r)+np.abs(z)))&(t>1e-12)&(t<1.-1e-12)&(weight>=0.)&(weight<=1.)
    gradient[tied,:2]=weight[tied,None]*unit[tied];gradient[tied,2]=-(1.-weight[tied])
    value=values[rows,index];upper=p[:,2]-top;active=upper>value
    value=np.maximum(value,upper);gradient[active]=[0.,0.,1.]
    return value,gradient


@dataclass(frozen=True)
class LinearSweep3D:
    euclidean_distance = False
    a_xyz: np.ndarray
    b_xyz: np.ndarray
    radius: float
    z_hi: float
    tool_id: int = 1
    kind: str = 'flat'

    def __post_init__(self):
        for name in ('a_xyz', 'b_xyz'):
            p = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError('XYZ sweep endpoints must be finite 3-vectors')
            p.flags.writeable = False
            object.__setattr__(self, name, p)
        if self.kind not in ('flat', 'ball') or not np.isfinite(self.radius) or self.radius <= 0.:
            raise ValueError('A positive-radius flat or ball tool is required')
        if not np.isfinite(self.z_hi) or self.z_hi <= max(self.a_xyz[2], self.b_xyz[2])+2.*self.radius:
            raise ValueError('Tool top must enclose both endpoint tools')

    @property
    def a_xy(self): return self.a_xyz[:2]
    @property
    def b_xy(self): return self.b_xyz[:2]
    @property
    def z_lo(self): return float(min(self.a_xyz[2], self.b_xyz[2]))
    @property
    def tip_z(self): return self.z_lo
    @property
    def center_z(self): return float(max(self.a_xyz[2], self.b_xyz[2])+self.radius)
    @property
    def diameter(self): return 2.*self.radius
    @property
    def move_length(self): return float(np.linalg.norm(self.b_xyz-self.a_xyz))

    def tip_at(self, progress):
        return self.a_xyz+np.clip(progress, 0., 1.)*(self.b_xyz-self.a_xyz)

    def vertical_volume_already_cut(self, history, stock_top):
        """Certify a pure Z sweep from a previously visited endpoint pose.

        Within the stock, a Z move has the volume of its deepest pose. A
        same-kind, no-smaller tool at the same XY and no-higher tip contains
        that pose. Both shafts must extend above the stock. No centre-only
        material test is used as a whole-tool certificate.
        """
        if not np.array_equal(self.a_xy,self.b_xy): return False
        for source in history:
            if source.kind!=self.kind or source.radius<self.radius or source.z_hi<stock_top: continue
            endpoints=((source.a_xyz,source.b_xyz) if hasattr(source,'a_xyz') else
                       (np.r_[source.a_xy,source.z_lo],np.r_[source.b_xy,source.z_lo]))
            if any(np.array_equal(p[:2],self.a_xy) and p[2]<=self.z_lo for p in endpoints): return True
        return False

    @cached_property
    def parts(self):
        r = self.radius
        z = np.array([0., 0., 1.])
        offset = r*z if self.kind == 'ball' else np.zeros(3)
        a, b = self.a_xyz+offset, self.b_xyz+offset
        result = _flat_parts(a,b,r,self.z_hi,self.tool_id)
        if self.kind == 'ball':
            result.extend(_ball_parts(a,b,r))
        for part in result:
            for arr in (part.planes, part.center, part.matrix):
                if arr is not None: arr.flags.writeable = False
        return tuple(result)

    @cached_property
    def _offset_parts(self): return {0.:self.parts}

    def sdf_gradient(self, points):
        p = np.asarray(points, float).reshape(-1, 3)
        offset=self.radius if self.kind=='ball' else 0.
        a,b=self.a_xyz+[0.,0.,offset],self.b_xyz+[0.,0.,offset]
        value,gradient=_flat_field(p,a,b,self.radius,self.z_hi)
        if self.kind=='ball':
            d=b-a;t=np.clip((p-a)@d/max(float(d@d),1e-30),0.,1.)
            delta=p-a-t[:,None]*d;length=np.linalg.norm(delta,axis=1)
            sphere=length-self.radius;active=sphere<value
            value[active]=sphere[active]
            gradient[active]=delta[active]/np.maximum(length[active,None],1e-30)
        return value,gradient

    def sdf(self, points): return self.sdf_gradient(points)[0]

    def project_surface(self, points):
        p = np.array(points, dtype=float, copy=True).reshape(-1, 3)
        for _ in range(12):
            f, g = self.sdf_gradient(p)
            p -= (f/np.maximum(np.sum(g*g, axis=1), 1e-30))[:, None]*g
        _, g = self.sdf_gradient(p)
        return p, g/np.maximum(np.linalg.norm(g, axis=1)[:, None], 1e-30)

    def face_candidates(self, points, tol):
        f, g = self.sdf_gradient(points)
        rows = np.flatnonzero(np.abs(f) <= 1.5*tol)
        return [(rows, -g[rows]/np.maximum(np.linalg.norm(g[rows], axis=1)[:, None], 1e-30))]

    def world_aabb(self, pad=0.):
        low = np.minimum(self.a_xyz, self.b_xyz)-[self.radius+pad, self.radius+pad, pad]
        high = np.maximum(self.a_xyz, self.b_xyz)+[self.radius+pad, self.radius+pad, pad]
        high[2] = self.z_hi+pad
        return low, high

    def support(self, normals):
        n = np.asarray(normals, float)
        offset = self.radius if self.kind == 'ball' else 0.
        a, b = self.a_xyz+[0., 0., offset], self.b_xyz+[0., 0., offset]
        nz = np.minimum(n[..., 2], 0.)
        base = np.maximum(n[..., :2]@a[:2]+nz*a[2], n[..., :2]@b[:2]+nz*b[2])
        shaft = base+self.radius*np.linalg.norm(n[..., :2], axis=-1)+np.maximum(n[..., 2], 0.)*self.z_hi
        if self.kind == 'flat': return shaft
        sphere = np.maximum(n@a, n@b)+self.radius*np.linalg.norm(n, axis=-1)
        return np.maximum(shaft, sphere)

    def line_interval(self, origin, direction, level=0.):
        """Exact zero-level section; outer hull / inner component certificates.

        Offset component unions can have multiple intervals. Positive levels
        return an enclosing interval; negative levels return one contained
        component. Both are conservative for whole-footprint certificates.
        """
        origin, direction = np.asarray(origin, float), np.asarray(direction, float)
        radius=self.radius+level
        if radius<=0.: return math.inf,-math.inf
        if level not in self._offset_parts:
            offset=self.radius if self.kind=='ball' else 0.
            a,b=self.a_xyz+[0.,0.,offset],self.b_xyz+[0.,0.,offset]
            shift=np.array([0.,0.,level])
            parts=_flat_parts(a-shift,b-shift,radius,self.z_hi+level,self.tool_id)
            if self.kind=='ball':parts.extend(_ball_parts(a,b,radius))
            if len(self._offset_parts)>=32:self._offset_parts.clear()
            self._offset_parts[level]=tuple(parts)
        intervals = []
        for part in self._offset_parts[level]:
            lo, hi = -np.inf, np.inf
            for plane in part.planes:
                value = float(plane[:3]@origin+plane[3])
                slope = float(plane[:3]@direction)
                if slope == 0.:
                    if value < 0.: lo, hi = np.inf, -np.inf; break
                elif slope > 0.: lo = max(lo, -value/slope)
                else: hi = min(hi, -value/slope)
            if lo > hi: continue
            if part.matrix is not None:
                r = part.radius
                if r < 0.: continue
                delta = origin-part.center
                aa = float(direction@part.matrix@direction)
                bb = float(direction@part.matrix@delta)
                cc = float(delta@part.matrix@delta-r*r)
                if aa <= 1e-14:
                    if abs(bb) <= 1e-14:
                        if cc > 0.: continue
                    elif bb > 0.: hi = min(hi, -cc/(2.*bb))
                    else: lo = max(lo, -cc/(2.*bb))
                else:
                    disc = bb*bb-aa*cc
                    if disc < 0.: continue
                    root = math.sqrt(disc)
                    lo, hi = max(lo, (-bb-root)/aa), min(hi, (-bb+root)/aa)
            if lo <= hi: intervals.append((lo, hi))
        if not intervals: return math.inf, -math.inf
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]+1e-12:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else: merged.append((lo, hi))
        if level < 0.: return max(merged, key=lambda x: x[1]-x[0])
        return merged[0][0], merged[-1][1]
