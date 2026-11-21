"""Ctrl+double-click machining overlays for the operator viewport."""
import time
import numpy as np

from cutsim.core.commands import command_endpoints
from cutsim.ui.machining_pick import draw_owners, gs_outline, inspection_tip, pick_gs
from cutsim.ui.tool_model import tool_model_spec, tool_model_meshes, tool_model_material


class MachiningInspectionMixin:
    _inspection_names = ('inspection_tool_cutting', 'inspection_tool_shank', 'inspection_path',
                         'inspection_gs_outline')

    def _machining_mouse(self, event):
        gui = self.gui
        ignored = gui.Widget.EventCallbackResult.IGNORED
        consumed = gui.Widget.EventCallbackResult.CONSUMED
        if self._closing:
            return ignored
        previous = self._machining_click
        if event.type == gui.MouseEvent.Type.DRAG:
            if previous and np.hypot(event.x-previous[1], event.y-previous[2]) > 5:
                self._machining_click = None
            return consumed if getattr(self, '_machining_press', False) else ignored
        if event.type == gui.MouseEvent.Type.BUTTON_UP and getattr(self, '_machining_press', False):
            self._machining_press = False
            return consumed
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
            return ignored
        if (not event.is_button_down(gui.MouseButton.LEFT)
                or not event.is_modifier_down(gui.KeyModifier.CTRL)):
            self._machining_click = None
            return ignored
        now = time.perf_counter()
        self._machining_press = True
        self._machining_click = (now, event.x, event.y)
        if not previous or now-previous[0] > .4 or np.hypot(event.x-previous[1], event.y-previous[2]) > 5:
            return consumed
        self._machining_click = None
        self._inspect_machining_at(event.x, event.y)
        return consumed

    def _inspect_machining_at(self, x, y):
        frame = self.scene.frame
        x, y = x-frame.x, y-frame.y
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            return
        self._stop()
        camera = self.scene.scene.camera
        near = np.asarray(camera.unproject(x, y, 0., frame.width, frame.height)).reshape(3)
        far = np.asarray(camera.unproject(x, y, .999999, frame.width, frame.height)).reshape(3)
        hit = (pick_gs(getattr(self, '_last_work_draw', None), near, far-near, np.linalg.norm(far-near))
               if self._gs_scene.visible else None)
        self._clear_machining_inspection()
        owners = self._last_work_owners
        if hit is not None and owners is not None:
            row, point = hit
            block = int(owners[row])
            if 0 <= block < len(self.sim.program):
                self._show_machining_inspection(block, point)
                self._machining_gs_row = row
                self._update_machining_outline()
        self._request_redraw()

    def _show_machining_inspection(self, block, point):
        command = self.sim.program[block]
        start, end = command_endpoints(command)
        tip = inspection_tip(command, point)
        spec = tool_model_spec(command.tool_type, command.radius, self.sim.stock_lo,
                               self.sim.stock_hi, command.cutting_range)
        material = tool_model_material(self.rendering, alpha=.25)
        # Compensate the transparent lit material's premultiplied darkening.
        # Keep the supported PBR shader (unlit transparency is incompatible
        # with the bundled renderer's srgbColor parameter).
        material.base_color = [16., 16., 16., .25]
        material.base_metallic = 0.
        scene = self.scene.scene
        from cutsim.core.tool_pose import tool_basis
        for name, mesh in tool_model_meshes(self.o3d, spec).items():
            mesh.rotate(tool_basis(command.tool_axis), center=np.zeros(3))
            mesh.translate(tip)
            scene.add_geometry('inspection_' + name, mesh, material)
        path = self.o3d.geometry.LineSet()
        path.points = self.o3d.utility.Vector3dVector([start, end])
        path.lines = self.o3d.utility.Vector2iVector([[0, 1]])
        path.colors = self.o3d.utility.Vector3dVector([[1., .72, .18]])
        line = self.rendering.MaterialRecord()
        line.shader = 'unlitLine'
        line.line_width = 4.
        scene.add_geometry('inspection_path', path, line)
        self._machining_selection = block
        self._tool_scene.set_visible(False)

    def _update_machining_outline(self):
        row = getattr(self, '_machining_gs_row', None)
        draw = getattr(self, '_last_work_draw', None)
        if self._machining_selection is None or row is None or draw is None:
            return
        self._remove('inspection_gs_outline')
        if row >= len(draw['positions']):
            return
        camera = np.asarray(self.scene.scene.camera.get_model_matrix())[:3, 3]
        points = gs_outline(draw, row, camera-draw['positions'][row])
        if not len(points):
            return
        outline = self.o3d.geometry.LineSet()
        outline.points = self.o3d.utility.Vector3dVector(points)
        indices = np.arange(len(points))
        outline.lines = self.o3d.utility.Vector2iVector(np.column_stack((indices, np.roll(indices, -1))))
        outline.paint_uniform_color([0., 1., 1.])
        material = self.rendering.MaterialRecord()
        material.shader = 'unlitLine'
        material.line_width = 3.
        self.scene.scene.add_geometry('inspection_gs_outline', outline, material)

    def _clear_machining_inspection(self):
        if self._machining_selection is None:
            return
        for name in self._inspection_names:
            self._remove(name)
        self._machining_selection = None
        self._machining_gs_row = None
        self._add_tool_visual(self._current_delta())

    def _add_tool_visual(self, delta):
        if self._machining_selection is not None:
            self._tool_scene.set_visible(False)
        else:
            super()._add_tool_visual(delta)

    def _add_splat_geometry(self, name, *args, **kwargs):
        records = super()._add_splat_geometry(name, *args, **kwargs)
        if name == 'work':
            draw = getattr(self, '_last_work_draw', None)
            self._last_work_owners = (draw_owners(records, len(draw['positions']))
                                      if records is not None and draw is not None else None)
        return records

    def _redraw(self):
        key = (id(self.sim), self.render_delta_index, self.progress)
        if key != self._machining_view_key:
            self._clear_machining_inspection()
            self._machining_view_key = key
        super()._redraw()
        self._update_machining_outline()

    def _reset(self):
        if not self._busy:
            self._clear_machining_inspection()
        super()._reset()
