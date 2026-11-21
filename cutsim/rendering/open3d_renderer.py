"""Clipped surface GS uploads for the required Open3D-2Cut2DGS renderer."""
from __future__ import annotations

import numpy as np
from cutsim.rendering.gs_schema import (
    GS_FIELDS, draw_from_calculation, validate_gaussian_attributes, validate_lighting, validate_shininess,
)


SURFACE_MATERIAL_FIELDS = ('gaussian_splat_surface_lighting',
                           'gaussian_splat_surface_shininess', 'gaussian_splat_opaque_surface')
REQUIRED_OPEN3D_MESSAGE = (
    'Open3D-2Cut2DGS with clipping and surface materials is required; standard Open3D is not supported. '
    'Run setup.bat, or use setup.bat --open3d-wheel with an extended wheel.')


def has_surface_material(material):
    return all(hasattr(material, name) for name in SURFACE_MATERIAL_FIELDS)


def require_gaussian_api(o3d):
    """Reject builds without the required clipping and surface material APIs."""
    material = o3d.visualization.rendering.MaterialRecord()
    point_cloud = getattr(getattr(getattr(o3d, 't', None), 'geometry', None), 'PointCloud', None)
    if not (has_surface_material(material) and
            hasattr(point_cloud, 'normalize_gaussian_splat_clipping')):
        raise RuntimeError(REQUIRED_OPEN3D_MESSAGE)


def configure_gaussian_material(material, lighting, opaque=False, shininess=64.):
    if not has_surface_material(material):
        raise RuntimeError(REQUIRED_OPEN3D_MESSAGE)
    lighting, shininess = validate_lighting(lighting), validate_shininess(shininess)
    material.shader = 'gaussianSplat'
    material.gaussian_splat_surface_lighting = lighting
    material.gaussian_splat_surface_shininess = shininess
    material.gaussian_splat_opaque_surface = bool(opaque)
    return material


def surface_draw_attributes(attributes, solid_color=None):
    out = dict(attributes)
    if solid_color is not None:
        out['surface_color'] = np.broadcast_to(np.asarray(solid_color, np.float32),
                                               (len(out['positions']), 3)).copy()
    validate_gaussian_attributes(out)
    return out


def split_draw_attributes(arr):
    """Preserve x >= boundary_split as cos(0)*x + sin(0)*y >= split."""
    count = len(arr['positions'])
    active = np.asarray(arr['half_active'], dtype=bool)
    if active.shape != (count,):
        raise ValueError('Split active mask must have shape (N,)')
    out = dict(arr)
    out['clip1_angle'] = np.where(active, 0., 4.).astype(np.float32)[:, None]
    out['clip1_split'] = np.where(active[:, None], arr['boundary_split'], 0.).astype(np.float32)
    out['clip2_angle'] = np.full((count, 1), 4., np.float32)
    out['clip2_split'] = np.zeros((count, 1), np.float32)
    return draw_from_calculation(out)


def gaussian_tensor_cloud(o3d, attributes):
    require_gaussian_api(o3d)
    validate_gaussian_attributes(attributes)
    cpu = o3d.core.Device('CPU:0')
    pc = o3d.t.geometry.PointCloud(cpu)
    for name in GS_FIELDS:
        if name in attributes:
            dtype = o3d.core.Dtype.Int32 if name == 'surface_id' else o3d.core.Dtype.Float32
            pc.point[name] = o3d.core.Tensor(np.ascontiguousarray(attributes[name]), dtype=dtype, device=cpu)
    # Input has already been validated. Do not rewrite calculated boundaries.
    return pc


class GaussianScene:
    """Own every GS registration, including hidden and debug objects.

    Style changes retain the exact tensors, visibility and camera. No callback
    to a simulator, attribute packer, or clip calculator exists here.
    """
    def __init__(self, scene, lighting=(.5, .5, .5), opaque=False, shininess=64.):
        self.scene = scene
        self.lighting = validate_lighting(lighting)
        self.opaque = bool(opaque)
        self.shininess = validate_shininess(shininess)
        self.entries = {}
        self.visible = True

    def add(self, name, cloud, material, visible=True):
        configure_gaussian_material(material, self.lighting, self.opaque, self.shininess)
        self.remove(name)
        self.entries[name] = (cloud, material, bool(visible))
        self.scene.add_geometry(name, cloud, material)
        self.scene.show_geometry(name, bool(visible) and self.visible)

    def remove(self, name):
        self.entries.pop(name, None)
        if self.scene.has_geometry(name):
            self.scene.remove_geometry(name)

    def set_visible(self, visible):
        self.visible = bool(visible)
        for name, (_, _, enabled) in self.entries.items():
            self.scene.show_geometry(name, enabled and self.visible)

    def set_style(self, lighting=None, opaque=None, shininess=None):
        values = self.lighting if lighting is None else validate_lighting(lighting)
        mode = self.opaque if opaque is None else bool(opaque)
        exponent = self.shininess if shininess is None else validate_shininess(shininess)
        if np.array_equal(values, self.lighting) and mode == self.opaque and exponent == self.shininess:
            return
        # Remove ALL first: the compute renderer uses common scene lighting.
        entries = list(self.entries.items())
        for name, _ in entries:
            self.scene.remove_geometry(name)
        self.lighting, self.opaque = values.copy(), mode
        self.shininess = exponent
        for name, (cloud, material, visible) in entries:
            configure_gaussian_material(material, self.lighting, self.opaque, self.shininess)
            self.scene.add_geometry(name, cloud, material)
            self.scene.show_geometry(name, visible and self.visible)


def _gs_zoom_capacity(width,height,count,entry_target=8*1024*1024):
    """Cover the viewport per GS; reserve a bounded-target tile-entry pool.

    The existing renderer enforces a minimum average of 32 entries. Dense
    overlap beyond the pool is still reported by Open3D, never hidden here.
    """
    tiles=max(1,(int(width)+15)//16)*max(1,(int(height)+15)//16)
    return max(256,tiles),max(32,min(tiles,entry_target//max(1,int(count))))
