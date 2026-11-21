"""Read-only tool inspection using the production profile sampler."""
from dataclasses import dataclass
import numpy as np

from cutsim.core.commands import command_sweep
from cutsim.ui.tool_model import tool_model_spec, tool_model_meshes, tool_model_material, CameraHeadlight
from cutsim.rendering.opaque_coverage import DEFAULT_DISPLAY_SCALE


@dataclass
class ToolPreviewData:
    cloud: object
    radius: float
    height: float
    kind: str
    pitch: float
    ball_angle_deg: float | None = None
    dexel_grid: object = None


def stationary_section(atlas):
    """Rotate representative production cells onto the X-Z meridian.

    The sampler places azimuthal cells at midpoints, so filtering Y==0 would
    omit most rings. Rigid rotations preserve ray lengths and intersections;
    this is a section of the rotationally symmetric cutter, not a projection.
    """
    from copy import copy
    surface = atlas.points(np.arange(len(atlas.origins)), atlas.surface_distance)
    physical = ((atlas.charts != 'upper_cap') &
                ((atlas.charts != 'lower_cap') | (surface[:, 2] <= 1e-12)))
    rows = np.flatnonzero(physical)
    keys = np.column_stack((atlas.charts[rows], atlas.cell_bounds[rows, :2].astype(str)))
    _, first = np.unique(keys, axis=0, return_index=True)
    rows = np.repeat(rows[np.sort(first)], 2)
    angle = np.arctan2(surface[rows, 1], surface[rows, 0])
    turn = np.tile([np.pi, 0.], len(rows)//2)-angle
    c, s = np.cos(turn), np.sin(turn)

    def rotate(values):
        source = values[rows]
        result = source.copy()
        result[:, 0] = c*source[:, 0]-s*source[:, 1]
        result[:, 1] = s*source[:, 0]+c*source[:, 1]
        result[:, 1] = 0.  # Remove roundoff on this rotational meridian.
        return result

    section = copy(atlas)
    for name in ('origins', 'directions', 'tool_normals', 'feeds'):
        setattr(section, name, rotate(getattr(atlas, name)))
    for name in ('charts', 'radii', 'pitches', 'ray_bounds', 'surface_distance'):
        setattr(section, name, getattr(atlas, name)[rows].copy())
    section.cell_bounds = atlas.cell_bounds[rows].copy()
    section.cell_bounds[:, 2:] += turn[:, None]
    section.profile_count = len(rows)
    return section


def tool_preview_data(command, stock_lo, stock_hi, grid_sizing, gs_density=1., *, cache=None):
    """Stationary production rays and candidate GS in the tool-local section.

    No stock is assumed: the bounds contain the entire cutter plus its ray
    margin. Material admission and adaptive refinement need a cutting scene.
    """
    from dataclasses import replace
    from cutsim.core.commands import CuttingRange
    from cutsim.cutting.profile_records import cloud_from_snapshot
    from cutsim.cuda.profile_program import build_atlas, _snapshot
    from cutsim.cuda.center_ray_update import center_grid
    model = tool_model_spec(command.tool_type, command.radius, stock_lo, stock_hi, command.cutting_range)
    stationary = replace(command, a=(0., 0., 0.), b=(0., 0., 0.), depth=None,
        tool_axis=(0., 0., 1.), cutting_range=command.cutting_range or CuttingRange(0., model.cutting_height))
    trajectory = command_sweep(stationary, float('inf'))
    # build_atlas applies density itself; pass the undensified pitch.
    pitch = stationary.gs_sampling.resolve_sweep_pitch(trajectory, grid_sizing)
    from cutsim.geometry.sampling_density import sampling_pitch
    h = sampling_pitch(pitch, gs_density)
    r = stationary.radius
    low = np.array([-r-h, -r-h, -h])
    high = np.array([r+h, r+h, stationary.cutting_range.end+h])
    atlas, _ = build_atlas([stationary], low, high, gs_density, pitch,
                          include_stock=False, sweep_envelope=True)
    atlas = center_grid(atlas, command_sweep(stationary, high[2]), atlas.local_boxes[0])
    section = stationary_section(atlas)
    count = len(section.origins)
    positions = section.points(np.arange(count), section.surface_distance)
    normals = section.tool_normals
    endpoints = dict(positions=positions, normals=normals, distances=section.surface_distance,
        ray_ids=np.arange(count), owners=np.zeros(count, np.int32),
        tool_ids=np.full(count, command.tool_id, np.int32))
    snapshot = _snapshot(endpoints, section, [stationary])
    cloud = cloud_from_snapshot(snapshot)
    angle = stationary.gs_sampling.resolve_ball_angle(gs_density) if command.tool_type.lower() == 'ball' else None
    return ToolPreviewData(cloud, r, model.height, command.tool_type.lower(), h, angle, section)


def section_camera_fit(lower, upper, aspect):
    """Fit the cutting section in a 40-degree vertical field of view."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    half = .5*(upper-lower)
    tangent = np.tan(np.deg2rad(20.))
    distance = half[1] + 1.12*max(half[2]/tangent, half[0]/(tangent*max(aspect, 1e-6)))
    return .5*(lower+upper), max(float(distance), 1e-6)


class ToolPreviewScene:
    def __init__(self, backend, *, window=None, request_redraw=None):
        self.backend = backend
        self.window = backend.window if window is None else window
        self.request_redraw = backend._request_redraw if request_redraw is None else request_redraw
        self._dexels_visible = True
        self.display_size = DEFAULT_DISPLAY_SCALE
        self.opaque = True
        self.o3d, self.rendering = backend.o3d, backend.rendering
        self.widget = backend.gui.SceneWidget()
        self.widget.scene = self.rendering.Open3DScene(self.window.renderer)
        from cutsim.ui.mouse_controls import camera_mouse_handler
        self.widget.set_on_mouse(camera_mouse_handler(backend.gui))
        self.widget.enable_scene_caching(True)
        self.widget.scene.set_background(np.array([.055, .065, .08, 1.]))
        # Filament mesh lighting only. GaussianScene uses its own explicit
        # surface-lighting coefficients; do not change exposure or GS settings.
        self.widget.scene.set_lighting(
            self.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
            np.array([-.45, -.55, -.70], dtype=np.float32))
        # Face the open X-Z section squarely so its Dexel layout is not foreshortened.
        self._eye_direction = np.array([0., -1., 0.])
        self.light = CameraHeadlight(self.widget.scene, intensity=250000.)
        self.widget.visible = False
        self.window.add_child(self.widget)

    def show(self, command, centers=True, *, fit_camera=True):
        from cutsim.cutting.profile_records import render_records
        from cutsim.rendering.playback import playback_draw_attributes
        from cutsim.rendering.open3d_renderer import GaussianScene, gaussian_tensor_cloud, _gs_zoom_capacity
        b = self.backend
        data = tool_preview_data(command, b.sim.stock_lo, b.sim.stock_hi, b.sim.grid_sizing,
                                 getattr(b.sim, 'gs_density', 1.))
        scene = self.widget.scene
        scene.clear_geometry()
        r, height = data.radius, max(data.height, data.radius)
        # Model and section share tool-local coordinates: tip Z=0, axis X=Y=0.
        spec = tool_model_spec(data.kind, r, b.sim.stock_lo, b.sim.stock_hi, command.cutting_range)
        mesh = self.o3d.geometry.TriangleMesh()
        for part in tool_model_meshes(self.o3d, spec).values():
            mesh += part
        # Open the viewer-facing half at the X-Z section. Transparent meshes
        # still occlude Gaussian depth; moving GS off the surface would lie
        # about their intersections. The 64-segment mesh has vertices at Y=0,
        # so no triangle crosses this section and no new cap is introduced.
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        mesh.remove_triangles_by_mask(np.any(vertices[triangles, 1] < -1e-10*r, axis=1))
        mesh.remove_unreferenced_vertices()
        material = tool_model_material(self.rendering, alpha=.65)
        scene.add_geometry('tool_mesh', mesh, material)
        cloud = data.cloud
        # Appearance-only production upload: never refit/subdivide the section.
        settings = self.render_settings()
        draw = playback_draw_attributes(render_records(cloud, settings))
        self.gs = GaussianScene(scene, lighting=b._gs_scene.lighting, opaque=self.opaque,
                                shininess=b._gs_scene.shininess)
        gs_material = self.rendering.MaterialRecord()
        cap, average = _gs_zoom_capacity(self.window.content_rect.width, self.window.content_rect.height, len(cloud.positions))
        gs_material.gaussian_splat_max_tiles_per_splat = cap
        gs_material.gaussian_splat_avg_tiles_per_splat = average
        gs_material.gaussian_splat_max_tile_entries_total = 32*1024*1024
        self.gs.add('tool_gs', gaussian_tensor_cloud(self.o3d, draw), gs_material)
        self.gs.set_visible(b._gs_scene.visible)
        points = self.o3d.geometry.PointCloud()
        points.points = self.o3d.utility.Vector3dVector(draw['positions'])
        points.paint_uniform_color([1., .35, .12])
        point_mat = self.rendering.MaterialRecord()
        point_mat.shader, point_mat.point_size = 'defaultUnlit', 4.
        scene.add_geometry('tool_centres', points, point_mat)
        scene.show_geometry('tool_centres', centers)
        # Full finite calculation intervals, including the interior origin.
        grid = data.dexel_grid
        segments = grid.origins[:, None, :] + grid.ray_bounds[:, :, None]*grid.directions[:, None, :]
        ray_points = draw['positions']
        dexel_points = self.o3d.geometry.PointCloud()
        dexel_points.points = self.o3d.utility.Vector3dVector(ray_points)
        dexel_points.paint_uniform_color([.1, .9, 1.])
        dexel_material = self.rendering.MaterialRecord()
        dexel_material.shader, dexel_material.point_size = 'defaultUnlit', 4.
        scene.add_geometry('tool_dexel_points', dexel_points, dexel_material)
        rays = self.o3d.geometry.LineSet()
        rays.points = self.o3d.utility.Vector3dVector(segments.reshape(-1, 3))
        rays.lines = self.o3d.utility.Vector2iVector(np.arange(2*len(segments)).reshape(-1, 2))
        rays.paint_uniform_color([.1, .9, 1.])
        ray_material = self.rendering.MaterialRecord()
        ray_material.shader, ray_material.line_width = 'unlitLine', 1.
        scene.add_geometry('tool_dexel_rays', rays, ray_material)
        for name in ('tool_dexel_points', 'tool_dexel_rays'):
            scene.show_geometry(name, self._dexels_visible)
        cutting = command.cutting_range
        low, high = (cutting.start, cutting.end) if cutting else (0., spec.cutting_height)
        lower, upper = np.array([-r, -r, low]), np.array([r, r, high])
        if len(segments):
            lower = np.minimum(lower, segments.min(axis=(0, 1)))
            upper = np.maximum(upper, segments.max(axis=(0, 1)))
        self._focus_bounds = (lower, upper)
        if fit_camera:
            self.fit_camera()
        self.light.update()
        self.widget.force_redraw()
        cutting = command.cutting_range
        legend = (f'Gold: cutting {cutting.start:g}–{cutting.end:g} mm · Gray: non-cutting · ' if cutting else '')
        angle_text = f' · ball Δθ={data.ball_angle_deg:.3f}°' if data.ball_angle_deg is not None else ''
        divisor_text = (f' D/{command.gs_sampling.diameter_divisions}'
                        if command.gs_sampling.mode == 'auto' and command.gs_sampling.diameter_divisions is not None else '')
        return (legend + f'Stationary tool section (X-Z, cutaway): {len(cloud.positions):,} samples · Tool-profile Dexels: {len(segments):,} · '
                f'GS {command.gs_sampling.mode}{divisor_text} · h={data.pitch:.4f} mm{angle_text} · display L={height:.2f} mm')

    def render_settings(self):
        from dataclasses import replace
        b = self.backend
        return replace(b._render_settings(half_clip=bool(b.half_gs.checked)), sigma_scale=self.display_size)

    def fit_camera(self, width=None, height=None):
        if not hasattr(self, '_focus_bounds'):
            return
        rect = self.window.content_rect
        width = max(1, rect.width if width is None else width)
        height = max(1, rect.height if height is None else height)
        lower, upper = self._focus_bounds
        target, distance = section_camera_fit(lower, upper, width/height)
        bounds = self.o3d.geometry.AxisAlignedBoundingBox(lower, upper)
        self.widget.setup_camera(40., bounds, target)
        self.widget.look_at(target, target+self._eye_direction*distance, [0., 0., 1.])
        self._fit_size = (width, height)

    def set_centers(self, visible):
        self.widget.scene.show_geometry('tool_centres', visible)
        self.request_redraw()

    def set_dexels(self, visible):
        self._dexels_visible = bool(visible)
        for name in ('tool_dexel_points', 'tool_dexel_rays'):
            self.widget.scene.show_geometry(name, self._dexels_visible)
        self.request_redraw()


class ToolPreviewViewport:
    """Independent, lazily created tool view; never replaces the machining view."""
    def __init__(self, backend):
        import os
        from uuid import uuid4
        from cutsim.ui.native_window import create_embedded_window
        self.backend = backend
        self.native_title = f'GaussianCutSim tool preview {os.getpid()} {uuid4().hex}'
        self.window = create_embedded_window(backend.gui, self.native_title, 360, 420)
        self.visible = self.closed = False
        self._queued = self._followup = False
        self._followup_at = None
        self._command = self._settings = None
        self._centers = True
        self.status = ''
        try:
            self.scene = ToolPreviewScene(backend, window=self.window, request_redraw=self.request_redraw)
        except Exception:
            self.close()
            raise
        self.scene.widget.visible = True
        self.window.set_on_layout(self._layout)
        self.window.set_on_tick_event(self._tick)
        self.window.set_on_close(lambda: False)

    def _layout(self, context):
        r = self.window.content_rect
        self.scene.widget.frame = self.backend.gui.Rect(r.x, r.y, max(1, r.width), max(1, r.height))
        if getattr(self.scene, '_fit_size', None) != (max(1, r.width), max(1, r.height)):
            self.scene.fit_camera(max(1, r.width), max(1, r.height))

    def _render_settings(self):
        from dataclasses import replace
        b = self.backend
        settings = self.scene.render_settings()
        # Lighting has a shader-only update path. Density/stock changes also
        # invalidate the profile when Apply & reset is used in the Tools tab.
        return (replace(settings, ambient=0., diffuse=0., specular=0.),
                tuple(b.sim.stock_lo), tuple(b.sim.stock_hi), b.sim.grid_sizing,
                getattr(b.sim, 'gs_density', 1.))

    def show(self, command, centers=True):
        self._command, self._centers = command, centers
        self.status = self.scene.show(command, centers)
        self._settings = self._render_settings()
        self.visible = True
        self.window.set_needs_layout()
        self.window.show(True)
        self.request_redraw()
        return self.status

    def hide(self):
        self.visible = False
        self._followup_at = None
        self.window.show(False)

    def set_centers(self, value):
        self._centers = value
        if self._command is not None:
            self.scene.set_centers(value)

    def set_dexels(self, value):
        self.scene.set_dexels(value)

    def set_style(self, *, opaque, display_size):
        from cutsim.rendering.opaque_coverage import display_factor
        display_factor(display_size)
        self.scene.opaque = bool(opaque)
        self.scene.display_size = float(display_size)
        self.sync_settings()

    def sync_settings(self):
        if not self.visible or self.closed:
            return
        settings = self._render_settings()
        if settings != self._settings:
            self.status = self.scene.show(self._command, self._centers, fit_camera=False)
            self._settings = settings
            self.request_redraw()
        gs, source = self.scene.gs, self.backend._gs_scene
        if (not np.array_equal(gs.lighting, source.lighting) or
                gs.shininess != source.shininess or gs.opaque != self.scene.opaque):
            gs.set_style(lighting=source.lighting, shininess=source.shininess, opaque=self.scene.opaque)
            self.request_redraw()
        if gs.visible != source.visible:
            gs.set_visible(source.visible)
            self.request_redraw()

    def request_redraw(self):
        self._followup = True
        self._followup_at = None
        self._queue_redraw()

    def _queue_redraw(self):
        if self.closed or self._queued or not self.visible:
            return
        self._queued = True
        self.backend.gui.Application.instance.post_to_main_thread(self.window, self._flush_redraw)

    def _flush_redraw(self):
        import time
        self._queued = False
        followup, self._followup = self._followup, False
        if self.closed or not self.visible:
            self._followup_at = None
            return
        self.scene.widget.force_redraw()
        self.window.post_redraw()
        self._followup_at = time.perf_counter() + .04 if followup else None

    def _tick(self):
        import time
        if self.closed or not self.visible:
            return False
        changed = self.scene.light.update()
        if changed:
            self.request_redraw()
        if self._followup_at is not None and time.perf_counter() >= self._followup_at:
            self._followup_at = None
            self._queue_redraw()
            changed = True
        return changed

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.window.set_on_close(lambda: True)
        self.window.close()
