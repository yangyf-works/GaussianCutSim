"""Fixed tool-axis frames for indexed (3+2-axis) milling."""
from functools import lru_cache
import numpy as np

DEFAULT_TOOL_AXIS = (0., 0., 1.)


def normalized_axis(value):
    axis = np.asarray(value, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.any(axis):
        raise ValueError('tool_axis must be a finite nonzero XYZ vector')
    axis = axis / np.max(np.abs(axis))
    axis = axis / np.linalg.norm(axis)
    axis[np.abs(axis) < 1.e-14] = 0.
    return tuple(float(x) for x in axis)


def axis_from_ab(a_deg, b_deg):
    """Active R_y(B) R_x(A), degrees; +tool Z points from tip to shank."""
    a, b = np.radians([a_deg, b_deg])
    return normalized_axis((np.sin(b)*np.cos(a), -np.sin(a), np.cos(b)*np.cos(a)))


@lru_cache(maxsize=128)
def tool_basis(axis=DEFAULT_TOOL_AXIS):
    z = np.asarray(normalized_axis(axis))
    helper = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
    x = helper-z*np.dot(helper, z)
    x /= np.linalg.norm(x)
    basis = np.column_stack((x, np.cross(z, x), z))
    basis.flags.writeable = False
    return basis


def box_in_frame(low, high, basis, origin):
    corners = np.array(np.meshgrid(*zip(low, high), indexing='ij')).reshape(3, -1).T
    points = (corners-origin)@basis
    return points.min(axis=0), points.max(axis=0)
