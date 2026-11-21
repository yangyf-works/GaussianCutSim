"""Perspective camera framing for the current workpiece bounds."""
import numpy as np


def stock_camera_clip_range(stock_lo, stock_hi, eye, backward):
    """Depth range for the current pose, independent of the original fit.

    Keep near behind the closest stock corner to preserve useful depth
    precision. A tiny fixed near makes the custom renderer's 16-bit merged
    depth inaccurate when the workpiece is still far from the camera.
    """
    lo, hi = np.asarray(stock_lo, float), np.asarray(stock_hi, float)
    backward = np.asarray(backward, float)
    center_depth = np.dot((lo+hi)*.5-np.asarray(eye, float), -backward)
    half_depth = np.dot((hi-lo)*.5, np.abs(backward))
    diagonal = np.linalg.norm(hi-lo)
    near = max(diagonal*1e-4, (center_depth-half_depth)*.5)
    # Tool geometry and an orbit must not be cropped by a tightly fitted far.
    far = max(diagonal, center_depth+half_depth+2.*diagonal)
    return float(near), float(far)


def update_stock_camera_clipping(camera, stock_lo, stock_hi, width, height):
    """Follow native dolly/orbit/pan without changing its pose, FOV or GS."""
    model = np.asarray(camera.get_model_matrix())
    near, far = stock_camera_clip_range(stock_lo, stock_hi, model[:3,3], model[:3,2])
    if (np.isclose(camera.get_near(), near, rtol=1e-6, atol=0.) and
            np.isclose(camera.get_far(), far, rtol=1e-6, atol=0.)):
        return False
    camera.set_projection(camera.get_field_of_view(), max(1,width)/max(1,height),
                          near, far, camera.get_field_of_view_type())
    return True


def fitted_stock_camera(stock_lo, stock_hi, direction, up, aspect, fov=35., padding=1.12):
    """Fit all eight stock corners, with a margin in both viewport dimensions."""
    lo, hi = np.asarray(stock_lo, float), np.asarray(stock_hi, float)
    center = (lo+hi)*.5
    backward = np.array(direction, dtype=float)
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    vertical = np.cross(backward, right)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])-center
    tan_vertical = np.tan(np.deg2rad(fov)*.5)
    tan_horizontal = tan_vertical*aspect
    along = corners@backward
    distance = np.max(along+padding*np.maximum(
        np.abs(corners@right)/tan_horizontal, np.abs(corners@vertical)/tan_vertical))
    eye = center+backward*distance
    near, far = stock_camera_clip_range(lo, hi, eye, backward)
    return center, eye, vertical, near, far
