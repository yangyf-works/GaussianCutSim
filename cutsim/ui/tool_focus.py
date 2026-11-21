"""Camera-only tool tracking; machining geometry stays in world coordinates."""
import numpy as np

from cutsim.core.commands import command_endpoints
from cutsim.core.tool_pose import tool_basis
from cutsim.ui.camera import fitted_stock_camera, update_stock_camera_clipping
from cutsim.ui.tool_model import tool_model_spec


def tool_focus_bounds(command, progress, stock_lo, stock_hi):
    start, end = command_endpoints(command)
    tip = start + (end - start) * np.clip(progress, 0., 1.)
    spec = tool_model_spec(command.tool_type, command.radius, stock_lo, stock_hi, command.cutting_range)
    basis = tool_basis(command.tool_axis)
    center = tip + basis[:, 2] * (.5 * (spec.cutting_start + spec.cutting_height))
    # Include nearby stock around the cutter, not the full illustrative shank.
    half_size = np.abs(basis) @ np.array([
        3. * spec.radius, 3. * spec.radius,
        .5 * (spec.cutting_height - spec.cutting_start) + spec.radius])
    return center - half_size, center + half_size


class ToolFocusMixin:
    def set_tool_focus(self, enabled):
        self._tool_focus = bool(enabled)
        self._tool_focus_target = None
        if self._tool_focus:
            self._update_tool_focus(fit=True)

    def _focus_command(self):
        # Follow the displayed motion, including scrubbing and loop rewind.
        # A pending seek keeps the last computed image until it is available.
        delta = self._current_delta()
        if delta is not None:
            return delta.command, self.progress
        index = self._selected_block
        if 0 <= index < len(self.sim.program):
            return self.sim.program[index], 0.
        return None, 0.

    def _update_tool_focus(self, *, fit=False, direction=None, up=None):
        if not getattr(self, '_tool_focus', False):
            return False
        command, progress = self._focus_command()
        if command is None:
            self._tool_focus_target = None
            return False
        low, high = tool_focus_bounds(command, progress, self.sim.stock_lo, self.sim.stock_hi)
        target = (low + high) * .5
        camera = self.scene.scene.camera
        model = np.asarray(camera.get_model_matrix(), dtype=float)
        backward = model[:3, 2]
        vertical = model[:3, 1]
        frame = self.scene.frame
        if fit or self._tool_focus_target is None:
            target, eye, vertical, _, _ = fitted_stock_camera(
                low, high, backward if direction is None else direction,
                vertical if up is None else up,
                max(1, frame.width) / max(1, frame.height))
            camera.set_projection(35., max(1, frame.width) / max(1, frame.height),
                                  camera.get_near(), camera.get_far(),
                                  self.rendering.Camera.FovType.Vertical)
        else:
            # Read the current camera so user orbit/zoom is retained. Lateral
            # panning is recentered in focus mode; the cutter stays stationary.
            distance = max(float(np.dot(model[:3, 3] - self._tool_focus_target, backward)),
                           float(command.radius) * .01)
            eye = target + backward * distance
            if (np.allclose(target, self._tool_focus_target, rtol=0., atol=1.e-7)
                    and np.allclose(eye, model[:3, 3], rtol=1.e-6, atol=1.e-7)):
                return False
        self.scene.look_at(target, eye, vertical)
        self.scene.center_of_rotation = target
        self._tool_focus_target = target.copy()
        update_stock_camera_clipping(camera, self.sim.stock_lo, self.sim.stock_hi,
                                     frame.width, frame.height)
        self._tool_scene.light.update()
        self._request_redraw()
        return True

    def _camera_view(self, view):
        if getattr(self, '_tool_focus', False):
            direction, up = {
                'Isometric': ([90., -110., 75.], [0., 0., 1.]),
                'Top': ([0., 0., 1.], [0., 1., 0.]),
                'Front': ([0., -1., 0.], [0., 0., 1.]),
            }[view]
            command, _ = self._focus_command()
            if command is not None:
                basis = tool_basis(command.tool_axis)
                if self._update_tool_focus(fit=True, direction=basis @ direction, up=basis @ up):
                    return
        super()._camera_view(view)
