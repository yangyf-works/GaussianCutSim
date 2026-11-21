"""Rigidly transformed finite sweeps, with equations retained in work coordinates."""
from dataclasses import dataclass
from functools import cached_property
import numpy as np

from cutsim.geometry.strip_intersections import Primitive


@dataclass(frozen=True)
class IndexedSweep:
    local: object
    basis: np.ndarray
    origin: np.ndarray
    euclidean_distance = False

    def __post_init__(self):
        basis, origin = np.array(self.basis,copy=True), np.array(self.origin,copy=True)
        if (basis.shape!=(3,3) or origin.shape!=(3,) or
                not np.isfinite(basis).all() or not np.isfinite(origin).all() or
                not np.allclose(basis.T@basis,np.eye(3),atol=1.e-12,rtol=0.) or
                not np.isclose(np.linalg.det(basis),1.,atol=1.e-12,rtol=0.)):
            raise ValueError('Indexed sweep requires a rigid right-handed frame')
        basis.flags.writeable = origin.flags.writeable = False
        object.__setattr__(self,'basis',basis)
        object.__setattr__(self,'origin',origin)

    @property
    def a_xyz(self):
        return self.origin+self.local.a_xyz@self.basis.T

    @property
    def b_xyz(self):
        return self.origin+self.local.b_xyz@self.basis.T

    @property
    def radius(self):
        return self.local.radius

    @property
    def kind(self):
        return self.local.kind

    @property
    def tool_id(self):
        return self.local.tool_id

    @property
    def diameter(self):
        return 2.*self.radius

    @property
    def move_length(self):
        return self.local.move_length

    @property
    def cutting_start(self):
        return self.local.cutting_start

    @property
    def cutting_end(self):
        return self.local.cutting_end

    @cached_property
    def parts(self):
        result = []
        for part in self.local.parts:
            normals = part.planes[:, :3]@self.basis.T
            planes = np.column_stack((normals, part.planes[:, 3]-normals@self.origin))
            matrix = None if part.matrix is None else self.basis@part.matrix@self.basis.T
            center = self.origin+part.center@self.basis.T
            for value in (planes,matrix,center):
                if value is not None:
                    value.flags.writeable = False
            result.append(Primitive(part.name, planes, center, matrix, part.radius))
        return tuple(result)

    def sdf(self, points):
        return self.local.sdf((np.asarray(points)-self.origin)@self.basis)

    def sdf_gradient(self, points):
        value, gradient = self.local.sdf_gradient((np.asarray(points)-self.origin)@self.basis)
        return value, gradient@self.basis.T

    def support(self, normals):
        normals = np.asarray(normals)
        return normals@self.origin+self.local.support(normals@self.basis)

    def world_aabb(self, pad=0.):
        axes = np.eye(3)
        return -self.support(-axes)-pad, self.support(axes)+pad

    def line_interval(self, origin, direction, level=0.):
        return self.local.line_interval((np.asarray(origin)-self.origin)@self.basis,
                                        np.asarray(direction)@self.basis, level)

    def vertical_volume_already_cut(self, history, stock_top):
        return False
