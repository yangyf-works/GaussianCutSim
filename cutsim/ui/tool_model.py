"""Rigid display tools, independent of the analytic sweep's world-space cap."""
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
import numpy as np


TOOL_ENVIRONMENT_INTENSITY = 60000.
TOOL_CUTTING_COLOR = (.92, .72, .32)
TOOL_SHANK_COLOR = (.68, .74, .81)
TOOL_ENVIRONMENT = Path(__file__).resolve().parent/'resources/tool_uniform'


@dataclass(frozen=True)
class ToolModelSpec:
    kind: str
    radius: float
    height: float
    cutting_height: float
    cutting_start: float = 0.


def tool_model_spec(kind, radius, stock_lo, stock_hi, cutting_range=None):
    """Use a consistent illustrative length until tool lengths are input data."""
    kind, radius = str(kind).lower(), float(radius)
    if kind not in ('flat', 'ball') or not np.isfinite(radius) or radius <= 0.:
        raise ValueError('Display tools require a flat/ball kind and positive radius')
    thickness = max(0., float(stock_hi[2])-float(stock_lo[2]))
    height = max(4.*radius, thickness+2.*radius)
    if cutting_range is not None:
        height = max(height, cutting_range.end+2.*radius)
        return ToolModelSpec(kind, radius, height, cutting_range.end, cutting_range.start)
    return ToolModelSpec(kind, radius, height, min(.7*height, max(2.*radius, .45*height)))


def tool_model_meshes(o3d, spec, segments=64):
    """Tip-local exterior only, with smooth sides and separate sharp cap normals.

    The ball is a lower hemisphere meeting a cylinder at its equator. Neither
    the equator nor the cutting/shank color seam has an internal closing disk.
    """
    angle = 2.*np.pi*np.arange(segments)/segments
    circle = np.column_stack((np.cos(angle), np.sin(angle)))

    def mesh(rings, *, pole=False, bottom=False, top=False):
        vertices, normals, triangles = [], [], []
        if pole:
            vertices.append([0., 0., 0.]); normals.append([0., 0., -1.])
        offsets = []
        for z, radius, nr, nz in rings:
            offsets.append(len(vertices))
            vertices.extend(np.column_stack((radius*circle, np.full(segments, z))))
            normals.extend(np.column_stack((nr*circle, np.full(segments, nz))))
        for j in range(segments):
            k = (j+1) % segments
            if pole:
                triangles.append([0, offsets[0]+k, offsets[0]+j])
            for lower, upper in zip(offsets[:-1], offsets[1:]):
                triangles.extend(([lower+j, lower+k, upper+k], [lower+j, upper+k, upper+j]))
        for enabled, ring, sign in ((bottom, rings[0], -1.), (top, rings[-1], 1.)):
            if not enabled:
                continue
            z, radius = ring[:2]; center = len(vertices)
            vertices.append([0., 0., z]); normals.append([0., 0., sign])
            vertices.extend(np.column_stack((radius*circle, np.full(segments, z))))
            normals.extend([[0., 0., sign]]*segments)
            for j in range(segments):
                a, b = center+1+j, center+1+(j+1) % segments
                triangles.append([center, a, b] if sign > 0. else [center, b, a])
        result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                         o3d.utility.Vector3iVector(triangles))
        result.compute_triangle_normals()
        result.vertex_normals = o3d.utility.Vector3dVector(normals)
        return result

    r, cut, height = spec.radius, spec.cutting_height, spec.height
    if spec.cutting_start > 0. or (spec.kind == 'ball' and cut <= r):
        def section(low, high):
            if high <= low:
                return o3d.geometry.TriangleMesh()
            if spec.kind == 'flat':
                return mesh([(low, r, 1., 0.), (high, r, 1., 0.)], bottom=low == 0., top=high == height)
            bottom = low == 0.
            values = [low, high]
            values.extend(r*(1.-np.cos(t)) for t in np.linspace(0., np.pi/2., 25)[1:-1] if low < r*(1.-np.cos(t)) < high)
            if low < r < high:
                values.append(r)
            rings = []
            for z in sorted(set(values)):
                if z == 0.:
                    continue
                if z < r:
                    radius = np.sqrt(max(0., r*r-(z-r)**2))
                    rings.append((z, radius, radius/r, (z-r)/r))
                else:
                    rings.append((z, r, 1., 0.))
            return mesh(rings, pole=bottom, top=high == height)
        cutting = section(spec.cutting_start, cut)
        shank = section(0., spec.cutting_start)+section(cut, height)
        cutting.paint_uniform_color(TOOL_CUTTING_COLOR)
        shank.paint_uniform_color(TOOL_SHANK_COLOR)
        return {'tool_cutting': cutting, 'tool_shank': shank}
    if spec.kind == 'ball':
        phi = np.linspace(0., np.pi/2., 25)[1:]
        rings = [(r*(1.-np.cos(t)), r*np.sin(t), np.sin(t), -np.cos(t)) for t in phi]
        rings[-1] = (r, r, 1., 0.)
        cutting = mesh(rings+[(cut, r, 1., 0.)], pole=True)
    else:
        cutting = mesh([(0., r, 1., 0.), (cut, r, 1., 0.)], bottom=True)
    shank = mesh([(cut, r, 1., 0.), (height, r, 1., 0.)], top=True)
    cutting.paint_uniform_color(TOOL_CUTTING_COLOR)
    shank.paint_uniform_color(TOOL_SHANK_COLOR)
    return {'tool_cutting': cutting, 'tool_shank': shank}


def tool_model_material(rendering, alpha=1.):
    material = rendering.MaterialRecord()
    material.shader = 'defaultLit' if alpha == 1. else 'defaultLitTransparency'
    material.base_color = [1., 1., 1., alpha]
    material.base_metallic = .05
    material.base_roughness = .55
    return material


def configure_tool_lighting(scene, direction=(0., 0., -1.), *, intensity=45000.):
    """Camera headlight plus neutral ambient fill, without fixed IBL hotspots."""
    if not scene.scene.set_indirect_light(str(TOOL_ENVIRONMENT)):
        raise RuntimeError(f'Could not load the uniform tool environment: {TOOL_ENVIRONMENT}')
    scene.scene.set_indirect_light_intensity(TOOL_ENVIRONMENT_INTENSITY)
    scene.scene.enable_indirect_light(True)
    scene.view.set_shadowing(False)
    scene.scene.set_sun_light(np.asarray(direction,dtype=np.float32),
                             np.ones(3,dtype=np.float32),float(intensity))
    scene.scene.enable_sun_light(True)


class CameraHeadlight:
    """Keep parallel rays aligned with the active camera's forward axis."""
    def __init__(self, scene, *, intensity=45000.):
        self.scene = scene
        self.intensity = float(intensity)
        self.direction = None
        configure_tool_lighting(scene, intensity=intensity)

    def update(self):
        model = np.asarray(self.scene.camera.get_model_matrix(), dtype=float)
        direction = -model[:3,2].copy()
        length = np.linalg.norm(direction)
        if not np.isfinite(direction).all() or not np.isfinite(length) or length <= 0.:
            raise ValueError('Camera headlight requires a finite camera forward direction')
        direction /= length
        if self.direction is not None and np.allclose(direction, self.direction, rtol=0., atol=1e-6):
            return False
        self.scene.scene.set_sun_light(direction.astype(np.float32),
                                       np.ones(3,dtype=np.float32),self.intensity)
        self.direction = direction.copy()
        return True


class ToolModelScene:
    """Upload on tool changes only; replay updates the rigid tool pose."""
    def __init__(self, o3d, scene):
        self.o3d, self.scene = o3d, scene
        self.light = CameraHeadlight(scene)
        self.spec = None
        self.visible = False
        self._cache = OrderedDict()

    def set_visible(self, visible):
        visible = bool(visible)
        if self.spec is not None and visible != self.visible:
            for name in ('tool_cutting', 'tool_shank'):
                self.scene.show_geometry(name, visible)
        self.visible = visible

    def set_pose(self, spec, tip, tool_axis=(0.,0.,1.)):
        if spec != self.spec:
            if self.spec is not None:
                for name in ('tool_cutting', 'tool_shank'):
                    self.scene.remove_geometry(name)
            if spec not in self._cache:
                self._cache[spec] = tool_model_meshes(self.o3d, spec)
                if len(self._cache) > 16:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(spec)
            material = tool_model_material(self.o3d.visualization.rendering)
            for name, geometry in self._cache[spec].items():
                self.scene.add_geometry(name, geometry, material)
            self.spec, self.visible = spec, False
        transform = np.eye(4)
        from cutsim.core.tool_pose import tool_basis
        transform[:3,:3] = tool_basis(tuple(tool_axis))
        transform[:3, 3] = np.asarray(tip, dtype=np.float64)
        for name in ('tool_cutting', 'tool_shank'):
            self.scene.set_geometry_transform(name, transform)
        self.set_visible(True)
