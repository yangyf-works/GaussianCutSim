"""Continuous sweep of a cutter intersected with a tip-local cutting interval.

The cutting solid is separate from the full tool. No shank or future collision
geometry contributes to material removal. Geometry uses analytic quadrics and
planes, including moving lower/upper caps; no intermediate poses are sampled.
"""
from dataclasses import dataclass
from functools import cached_property
import os
import numpy as np

from cutsim.core.linear_sweep import LinearSweep3D


def _disk_parts(a, b, radius, name):
    from cutsim.geometry.strip_intersections import Primitive, slab, swept_primitives
    from cutsim.core.sweep import FlatSweep
    if radius <= 0.:
        return []
    d = b-a
    if d[2] == 0.:
        return list(swept_primitives(FlatSweep(a[:2], b[:2], radius, a[2], a[2])))
    shear = np.column_stack((np.eye(2), -d[:2]/d[2]))
    return [Primitive(name, np.array(slab(np.array([0., 0., 1.]), min(a[2], b[2]), max(a[2], b[2]))),
                      a.copy(), shear.T@shear, radius)]


def _cylinder_parts(a, b, radius, low, high):
    from cutsim.geometry.strip_intersections import Primitive, slab, swept_primitives
    from cutsim.core.sweep import FlatSweep
    z = np.array([0., 0., 1.]); d = b-a
    length = float(np.linalg.norm(d[:2]))
    if length == 0. or d[2] == 0.:
        return list(swept_primitives(FlatSweep(a[:2], b[:2], radius,
            min(a[2], b[2])+low, max(a[2], b[2])+high)))
    u = np.r_[d[:2]/length, 0.]; v = np.array([-u[1], u[0], 0.])
    bottom = z-(d[2]/length)*u
    aa, bb = sorted((float(a@u), float(b@u)))
    result = [Primitive('cut_prism', np.array(slab(u, aa, bb)+
        slab(v, float(a@v)-radius, float(a@v)+radius)+
        [np.r_[bottom, -bottom@a-low], np.r_[-bottom, bottom@a+high]]))]
    for i, p in enumerate((a, b)):
        result.append(Primitive(f'cut_endpoint_{i}', np.array(slab(z, p[2]+low, p[2]+high)),
                                np.r_[p[:2], 0.], np.diag([1., 1., 0.]), radius))
    result.extend(_disk_parts(a+low*z, b+low*z, radius, 'cut_disk_lower'))
    result.extend(_disk_parts(a+high*z, b+high*z, radius, 'cut_disk_upper'))
    return result


def _sphere_parts(a, b, radius, equator, low, high):
    from cutsim.geometry.strip_intersections import Primitive, slab
    z = np.array([0., 0., 1.]); d = b-a
    center_a, center_b = a+equator*z, b+equator*z
    length = float(np.linalg.norm(d)); result = []
    if length:
        u = d/length
        normal = z-u[2]*u
        # The nearest sphere parameter must also satisfy the moving axial
        # limits. If it does not, the optimum is on one of the cap disks.
        planes = slab(u, float(center_a@u), float(center_b@u))
        planes += slab(normal, float(normal@center_a)+low-equator,
                       float(normal@center_a)+high-equator)
        result.append(Primitive('cut_ball_barrel', np.array(planes), center_a,
                                np.eye(3)-np.outer(u, u), radius))
    for i, (tip, center) in enumerate(((a, center_a), (b, center_b)) if length else ((a, center_a),)):
        result.append(Primitive(f'cut_ball_endpoint_{i}', np.array(slab(z, tip[2]+low, tip[2]+high)),
                                center.copy(), np.eye(3), radius))
    for name, height in (('cut_disk_ball_lower', low), ('cut_disk_ball_upper', high)):
        cap_radius = np.sqrt(max(0., radius*radius-(height-equator)**2))
        result.extend(_disk_parts(a+height*z, b+height*z, cap_radius, name))
    return result


def _quadratic_times(q, d, radius, slope, offset):
    """Roots of ||q-t*d||-radius = slope*t+offset, per query point."""
    aa = float(d@d)-slope*slope
    bb = -2.*(q@d+slope*(radius+offset))
    cc = np.sum(q*q, axis=1)-(radius+offset)**2
    if abs(aa) <= 1e-14*max(float(d@d)+slope*slope, 1e-30):
        root = np.divide(-cc, bb, out=np.zeros_like(bb), where=bb != 0.)
        return root, root
    discriminant = bb*bb-4.*aa*cc
    stable = -.5*(bb+np.copysign(np.sqrt(np.maximum(discriminant, 0.)), bb))
    first = stable/aa
    second = np.divide(cc, stable, out=-bb/(2.*aa), where=stable != 0.)
    return np.where(discriminant >= 0., first, 0.), np.where(discriminant >= 0., second, 0.)


@dataclass(frozen=True)
class CuttingSweep(LinearSweep3D):
    cutting_start: float = 0.
    cutting_end: float = 20.

    def __post_init__(self):
        from cutsim.core.commands import CuttingRange
        CuttingRange(self.cutting_start, self.cutting_end)
        if self.kind not in ('flat', 'ball') or not np.isfinite(self.radius) or self.radius <= 0.:
            raise ValueError('A positive-radius flat or ball tool is required')
        for name in ('a_xyz', 'b_xyz'):
            p = np.array(getattr(self, name), dtype=float, copy=True)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError('Cutting sweep endpoints must be finite XYZ vectors')
            p.flags.writeable = False
            object.__setattr__(self, name, p)
        object.__setattr__(self, 'z_hi', float(max(self.a_xyz[2], self.b_xyz[2])+self.cutting_end))

    @property
    def z_lo(self):
        return float(min(self.a_xyz[2], self.b_xyz[2])+self.cutting_start)

    @property
    def tip_z(self):
        return float(min(self.a_xyz[2], self.b_xyz[2]))

    def vertical_volume_already_cut(self, history, stock_top):
        # The unbounded-shaft/deepest-pose certificate is invalid for finite
        # cutters: withdrawing can cut material above the previous upper cap.
        return False

    def _parts_at_level(self, level):
        radius = self.radius+level
        low, high = self.cutting_start-level, self.cutting_end+level
        if radius <= 0. or high < low:
            return ()
        if self.kind == 'flat':
            result = _cylinder_parts(self.a_xyz, self.b_xyz, radius, low, high)
        else:
            result = []
            sphere_low, sphere_high = max(low, self.radius-radius), min(high, self.radius)
            if sphere_low < sphere_high:
                result.extend(_sphere_parts(self.a_xyz, self.b_xyz, radius, self.radius, sphere_low, sphere_high))
            if high > max(low, self.radius):
                result.extend(_cylinder_parts(self.a_xyz, self.b_xyz, radius, max(low, self.radius), high))
        valid_parts = []
        for part in result:
            zero = np.all(part.planes[:, :3] == 0., axis=1)
            if np.any(part.planes[zero, 3] < 0.):
                continue
            part.planes = part.planes[~zero]
            for array in (part.planes, part.center, part.matrix):
                if array is not None:
                    array.flags.writeable = False
            valid_parts.append(part)
        return tuple(valid_parts)

    @cached_property
    def parts(self):
        return self._parts_at_level(0.)

    def sdf(self, points):
        # Classification needs the field only. Keep exactly the same candidate
        # times and reductions, without constructing unused envelope normals.
        if os.environ.get('CUTSIM_LATE_BLOCK_ACCEL', '1') == '0':
            return self.sdf_gradient(points)[0]
        return self._evaluate_field(points, gradient_required=False)

    def sdf_gradient(self, points):
        return self._evaluate_field(points, gradient_required=True)

    def _evaluate_field(self, points, *, gradient_required):
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        d = self.b_xyz-self.a_xyz
        q = p-self.a_xyz
        low = self.a_xyz[2]+self.cutting_start-p[:, 2]
        high = p[:, 2]-self.a_xyz[2]-self.cutting_end
        times = [np.zeros(len(p)), np.ones(len(p))]
        if d[2]:
            times.extend(((p[:, 2]-self.a_xyz[2]-.5*(self.cutting_start+self.cutting_end))/d[2],))
        forms = [(q[:, :2], d[:2])]
        if self.kind == 'ball':
            sphere_q = q-np.array([0., 0., self.radius])
            forms.append((sphere_q, d))
            if d[2]:
                times.append((q[:, 2]-self.radius)/d[2])
        for query, delta in forms:
            den = float(delta@delta)
            if den:
                times.append((query@delta)/den)
            for slope, offset in ((d[2], low), (-d[2], high)):
                times.extend(_quadratic_times(query, delta, self.radius, slope, offset))
        t = np.clip(np.column_stack(times), 0., 1.)
        relative = q[:, None, :]-t[:, :, None]*d
        radial_vector = relative.copy()
        if self.kind == 'ball':
            radial_vector[:, :, 2] = np.minimum(relative[:, :, 2]-self.radius, 0.)
        else:
            radial_vector[:, :, 2] = 0.
        length = np.linalg.norm(radial_vector, axis=2)
        values = np.stack((length-self.radius,
            self.cutting_start-relative[:, :, 2], relative[:, :, 2]-self.cutting_end), axis=2)
        field = np.max(values, axis=2)
        choice = np.argmin(field, axis=1); rows = np.arange(len(p))
        value = field[rows, choice]
        if not gradient_required:
            return value
        gradients = np.zeros((len(p), 3, 3))
        gradients[:, 0] = radial_vector[rows, choice]/np.maximum(length[rows, choice, None], 1e-30)
        gradients[:, 1, 2], gradients[:, 2, 2] = -1., 1.
        active = values[rows, choice] >= value[:, None]-1e-10*(1.+np.abs(value[:, None]))
        projection = gradients@d
        projection[np.abs(projection) <= 64.*np.finfo(float).eps*max(1., np.linalg.norm(d))] = 0.
        first = np.argmin(np.where(active, projection, np.inf), axis=1)
        second = np.argmax(np.where(active, projection, -np.inf), axis=1)
        lower, upper = projection[rows, first], projection[rows, second]
        g0, g1 = gradients[rows, first], gradients[rows, second]
        weight = np.divide(upper, upper-lower, out=np.ones_like(upper), where=upper != lower)
        gradient = weight[:, None]*g0+(1.-weight[:, None])*g1
        # Interior envelope normals cancel motion. At segment endpoints choose
        # the active normal satisfying the one-sided minimisation condition.
        interior = (t[rows, choice] > 1e-12) & (t[rows, choice] < 1.-1e-12)
        mix = interior & (lower <= 0.) & (upper >= 0.)
        gradient[~mix] = np.where((t[rows, choice][~mix] <= .5)[:, None], g0[~mix], g1[~mix])
        return value, gradient

    def support(self, normals):
        n = np.asarray(normals, dtype=float)
        radial = np.linalg.norm(n[..., :2], axis=-1)
        nz = n[..., 2]
        low, high, r = self.cutting_start, self.cutting_end, self.radius
        if self.kind == 'flat' or low >= r:
            local = radial*r+np.maximum(nz*low, nz*high)
        else:
            length = np.maximum(np.linalg.norm(n, axis=-1), 1e-30)
            h = np.clip(r+r*nz/length, low, min(high, r))
            local = radial*np.sqrt(np.maximum(0., r*r-(h-r)**2))+nz*h
            if high > r:
                local = np.maximum(local, radial*r+nz*np.where(nz >= 0., high, r))
        return np.maximum(n@self.a_xyz, n@self.b_xyz)+local

    def world_aabb(self, pad=0.):
        r = self.radius
        if self.kind == 'ball' and self.cutting_end < r:
            r = np.sqrt(max(0., r*r-(self.cutting_end-r)**2))
        low = np.minimum(self.a_xyz, self.b_xyz)-[r+pad, r+pad, pad]
        high = np.maximum(self.a_xyz, self.b_xyz)+[r+pad, r+pad, pad]
        low[2] += self.cutting_start
        high[2] += self.cutting_end
        return low, high

    def line_interval(self, origin, direction, level=0.):
        if level not in self._offset_parts:
            if len(self._offset_parts) >= 32:
                self._offset_parts.clear()
            self._offset_parts[level] = self._parts_at_level(level)
        return super().line_interval(origin, direction, level)
