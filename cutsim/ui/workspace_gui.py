"""Operator workspace layered on the existing, preserved debug renderer."""
from __future__ import annotations
from cutsim.core.history_store import history_enabled
from cutsim.geometry.sampling_density import GS_DENSITY_PERCENTAGES

import time
import numpy as np

from cutsim.ui.debug_gui import SweepDrivenRealtimeGUI
from cutsim.ui.camera import fitted_stock_camera
from cutsim.ui.helpers import GS_DISPLAY_SIZES, PLAYBACK_SPEEDS
from cutsim.ui.workspace_model import (
    ComputeJob, CompletedBlock, DisplayState, block_details, block_label, path_length, tools_in_program,
)
from cutsim.rendering.playback import PlaybackAttributeCache
from cutsim.rendering.render_cache import RenderAttributeCache


class MachiningWorkspaceGUI(SweepDrivenRealtimeGUI):
    """Reuse the renderer/playback for the operator workspace."""

    def __init__(self, sim, *, compute_job=None):
        self._workspace_ready = False
        self._busy = False
        self._failed = False
        self._closing = False
        self._selection_sync = False
        self._selected_block = 0 if sim.program else -1
        self._play_block_only = False
        self._loop_block = False
        self._space_resume_repeat = False
        self._playback_scrubbing = False
        self._playback_seek_target = None
        from cutsim.application.process_job import ProcessComputeJob
        self._job = compute_job if compute_job is not None else ProcessComputeJob()
        self._job.start_background_warmup()
        self._compute_sim = sim
        self.machining_project = getattr(sim, 'machining_project', None)
        self._compute_attributes = PlaybackAttributeCache(disk_history=history_enabled())
        self._compute_render_cache = RenderAttributeCache()
        self._message = 'Compute All runs ahead. Play uses completed blocks and waits for the next result.'
        self._list_signature = None
        self._path_signature = None
        self._disabled_widgets = []
        super().__init__(DisplayState(sim, disk_history=history_enabled()))
        gui = self.gui
        self.window.title = 'GaussianCutSim'
        self.cut_debug.checked = False
        self.panel.visible = False

        self.header = gui.Horiz(8, gui.Margins(12, 6, 12, 6))
        self.header.add_child(gui.Label('GaussianCutSim'))
        self.compute_all_button = self._button(self.header, 'Compute All', self._compute_all)
        self.cancel_button = self._button(self.header, 'Stop Compute', self._cancel_compute)
        self.cancel_button.enabled = False
        self.reset_button = self._button(self.header, 'Reset', self._confirm_reset)

        self.sidebar = gui.TabControl()
        program = gui.ScrollableVert(8, gui.Margins(12, 12, 12, 12))
        self.sidebar.add_tab('Program', program)
        dims = sim.stock_hi - sim.stock_lo
        program.add_child(gui.Label('MACHINING PROGRAM'))
        self.stock_summary = gui.Label(f'Stock  {dims[0]:g} x {dims[1]:g} x {dims[2]:g} mm')
        program.add_child(self.stock_summary)
        program.add_child(gui.Label('GS density (%) · 100 = baseline'))
        self.density_input = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.density_input.set_limits(GS_DENSITY_PERCENTAGES[0], GS_DENSITY_PERCENTAGES[-1])
        self.density_input.double_value = 100.*getattr(sim, 'gs_density', 1.)
        program.add_child(self.density_input)
        self.density_apply = self._button(program, 'Apply',
            lambda: self._apply_gs_density(self.density_input.double_value/100.))
        program.add_child(gui.Label('Changing density discards computed results.'))
        self.program_summary = gui.Label('')
        program.add_child(self.program_summary)
        self.block_list = gui.ListView()
        self.block_list.set_max_visible_items(12)
        self.block_list.set_on_selection_changed(self._select_block)
        program.add_child(self.block_list)
        self.block_info = gui.Label('')
        program.add_child(self.block_info)
        program.add_child(gui.Label('Select a ready block to view its start.\nPending blocks show path data only.'))
        row = gui.Horiz(6)
        self._button(row, 'Block start', lambda: self._seek_selected(0.0))
        self._button(row, 'Block result', lambda: self._seek_selected(1.0))
        self.play_block_button = self._button(row, 'Play block', self._play_selected_block)
        program.add_child(row)
        program.add_child(gui.Label('Gold: selected  /  Green: computed\nGray: pending. Paths use actual tool-tip Z.'))

        tools = gui.ScrollableVert(8, gui.Margins(12, 12, 12, 12))
        self.sidebar.add_tab('Tools', tools)
        tools.add_child(gui.Label('TOOLS USED IN THIS PROGRAM'))
        self._tools = tools_in_program(sim.program)
        self.tool_list = gui.ListView()
        self.tool_list.set_items([tool.label for tool in self._tools])
        self.tool_list.set_max_visible_items(10)
        self.tool_list.set_on_selection_changed(self._select_tool)
        tools.add_child(self.tool_list)
        self.tool_info = gui.Label('Select a tool to inspect its usage.')
        tools.add_child(self.tool_info)
        self._button(tools, 'Go to first use', self._go_to_tool)
        tools.add_child(gui.Label('Tools are grouped by ID, type and diameter.\nRepeated IDs with different diameters stay separate.'))

        display = gui.ScrollableVert(8, gui.Margins(12, 12, 12, 12))
        self.sidebar.add_tab('View', display)
        display.add_child(gui.Label('CAMERA'))
        row = gui.Horiz(6)
        for name in ('Isometric', 'Top', 'Front'):
            self._button(row, name, lambda view=name: self._camera_view(view))
        display.add_child(row)
        self.path_toggle = gui.Checkbox('Show tool paths')
        self.path_toggle.checked = True
        self.path_toggle.set_on_checked(self._set_paths_visible)
        display.add_child(self.path_toggle)
        self.workspace_tool = gui.Checkbox('Show tool model')
        self.workspace_tool.checked = self.show_tool.checked
        self.workspace_tool.set_on_checked(self._set_tool_visible)
        display.add_child(self.workspace_tool)
        display.add_child(gui.Label('SURFACE'))
        self.workspace_smoothing = gui.Checkbox('Opaque surface smoothing')
        self.workspace_smoothing.checked = self.opaque_surface.checked
        self.workspace_smoothing.set_on_checked(self._set_smoothing)
        display.add_child(self.workspace_smoothing)
        self.workspace_colors = gui.Checkbox('Color by tool')
        self.workspace_colors.checked = self.tool_colors.checked
        self.workspace_colors.set_on_checked(self._set_tool_colors)
        display.add_child(self.workspace_colors)
        display.add_child(gui.Label('Display size'))
        self.workspace_size = gui.Combobox()
        for value in GS_DISPLAY_SIZES:self.workspace_size.add_item(f'{value:.2f}x')
        self.workspace_size.selected_index = self.sigma.selected_index
        self.workspace_size.set_on_selection_changed(lambda _label,index:self._set_display_size(GS_DISPLAY_SIZES[index]))
        display.add_child(self.workspace_size)

        self.transport = gui.Vert(6, gui.Margins(12, 8, 12, 8))
        row = gui.Horiz(8)
        self.prev_button = self._button(row, 'Previous', self._prev_buffer)
        self.play_button = self._button(row, 'Play', self._play)
        self.pause_button = self._button(row, 'Pause', self._stop)
        self.next_button = self._button(row, 'Next', self._next_buffer)
        row.add_stretch()
        row.add_child(gui.Label('Speed'))
        self.workspace_speed = gui.Slider(gui.Slider.DOUBLE)
        self.workspace_speed.set_limits(PLAYBACK_SPEEDS[0], PLAYBACK_SPEEDS[-1])
        self.workspace_speed.double_value = 1.
        self.workspace_speed.set_on_value_changed(self._set_speed)
        self.workspace_speed_label = gui.Label('1.00x')
        row.add_child(self.workspace_speed)
        row.add_child(self.workspace_speed_label)
        self.transport.add_child(row)
        row = gui.Horiz(8)
        row.add_child(gui.Label('Block position'))
        self.workspace_progress = gui.Slider(gui.Slider.DOUBLE)
        self.workspace_progress.set_limits(0., 100.)
        self.workspace_progress.set_on_value_changed(self._progress_changed)
        self.workspace_position = gui.Label('0%')
        row.add_child(self.workspace_progress)
        row.add_child(self.workspace_position)
        self.transport.add_child(row)
        self.playback_info = gui.Label('')
        self.transport.add_child(self.playback_info)
        self.notice = gui.Label(self._message)
        self.transport.add_child(self.notice)

        self.header.add_stretch()
        self.compute_info = gui.Label('Ready')
        self.compute_bar = gui.ProgressBar()
        self.header.add_child(self.compute_info)
        self.header.add_child(self.compute_bar)
        for widget in (self.header, self.sidebar, self.transport):
            self.window.add_child(widget)
        self._workspace_ready = True
        self.window.set_on_close(self._on_close)
        self._refresh_workspace()
        self._redraw()
        self.window.set_needs_layout()

    def _button(self, parent, text, callback):
        button = self.gui.Button(text)
        button.set_on_clicked(callback)
        parent.add_child(button)
        return button

    def _layout(self, context):
        if not self._workspace_ready:
            return super()._layout(context)
        r = self.window.content_rect
        em = self.window.theme.font_size
        header_h = int(3.2 * em)
        body_y, body_h = r.y + header_h, max(1, r.height - header_h)
        self.header.frame = self.gui.Rect(r.x, r.y, r.width, header_h)
        width = min(int(27 * em), max(280, int(r.width * .34)))
        controls_h = min(int(10 * em), body_h // 2)
        self.sidebar.frame = self.gui.Rect(r.x, body_y, width, body_h)
        self.scene.frame = self.gui.Rect(r.x + width, body_y, max(1, r.width - width), max(1, body_h - controls_h))
        self.transport.frame = self.gui.Rect(r.x + width, body_y + body_h - controls_h,
                                              max(1, r.width - width), controls_h)
        size = (self.scene.frame.width, self.scene.frame.height)
        if size != self._viewport_size:
            self._viewport_size = size
            self._viewport_pending_at = time.perf_counter()

    def _select_block(self, _value, _double=False):
        if self._selection_sync:
            return
        index = self.block_list.selected_index
        if not 0 <= index < len(self.sim.program):
            return
        self._selected_block = index
        self._stop()
        self._message = '' if index < len(self.sim.deltas) else 'Path preview only: this block has not been computed.'
        if index < len(self.sim.deltas):
            self._set_render_delta(index, 0.)
        else:
            self._refresh_workspace()
            self._update_paths()

    def _seek_selected(self, progress):
        if 0 <= self._selected_block < len(self.sim.deltas):
            self._set_render_delta(self._selected_block, progress)

    def set_playback_scrubbing(self, active):
        # Hold the motion clock during a drag without changing the play mode.
        self._playback_scrubbing = bool(active)
        self._playback_time = self._playback_next_frame = None

    def seek_playback(self, index, progress=0.):
        """Move the transport while retaining Play All / Repeat state."""
        if not 0 <= index < len(self.sim.program):
            return
        self._selected_block = index
        if not self._playing:
            self._playback_seek_target = None
            if index < len(self.sim.deltas):
                self._set_render_delta(index, progress)
            else:
                self._update_paths()
            self._refresh_workspace()
            return
        # Commit on the next render tick, after newly calculated blocks arrive.
        # A pending target must not resume motion on the previously shown block.
        self._playback_seek_target = (index, float(np.clip(progress, 0., 1.)))
        self._playback_prepare_index = None
        self._playback_time = self._playback_next_frame = None
        self._refresh_workspace()
        self._request_redraw()

    def _apply_playback_seek(self):
        target = getattr(self, '_playback_seek_target', None)
        if target is None:
            return False
        index, progress = target
        if index >= len(self.sim.deltas):
            return True  # Keep the requested mode armed until this block exists.
        self._playback_seek_target = None
        self.render_delta_index = self._selected_block = index
        self._sync_progress(progress)
        self._playback_prepare_index = index
        return False

    def _step_block(self, offset):
        if self._closing:
            return
        target = self._selected_block + offset
        if 0 <= target < len(self.sim.program):
            self.seek_playback(target)

    def _prev_buffer(self):
        self._step_block(-1)

    def _next_buffer(self):
        self._step_block(1)

    def _set_render_delta(self, index, u=0.0):
        if self.sim.deltas:
            self._selected_block = max(0, min(index, len(self.sim.deltas) - 1))
        super()._set_render_delta(index, u)

    def _select_tool(self, _value, _double=False):
        index = self.tool_list.selected_index
        if 0 <= index < len(self._tools):
            tool = self._tools[index]
            blocks = ', '.join(str(i + 1) for i in tool.blocks)
            length = sum(path_length(self.sim.program[i]) for i in tool.blocks)
            self.tool_info.text = f'{tool.label}\nUsed in blocks: {blocks}\nTotal cut length: {length:.2f} mm'

    def _go_to_tool(self):
        index = self.tool_list.selected_index
        if not 0 <= index < len(self._tools):
            return
        self.sidebar.selected_tab_index = 0
        self.block_list.selected_index = self._tools[index].blocks[0]
        self._select_block('')

    def _set_smoothing(self, value):
        self.opaque_surface.checked = value
        self._opaque_surface_changed(value)

    def _set_tool_colors(self, value):
        if self._busy:return
        self.tool_colors.checked = value
        self._redraw()

    def _set_display_size(self, value):
        if self._busy:return
        self._display_size_changed(value)

    def _set_paths_visible(self,value):
        self.path_toggle.checked=bool(value)
        self._update_paths()

    def _set_speed(self, value):
        self.speed_slider.double_value = value
        self._speed_changed(value)

    def _camera_view(self, view):
        direction, up = {
            'Isometric': ([90., -110., 75.], [0., 0., 1.]),
            'Top': ([0., 0., 150.], [0., 1., 0.]),
            'Front': ([0., -150., 0.], [0., 0., 1.]),
        }[view]
        frame = self.scene.frame
        aspect = max(1, frame.width)/max(1, frame.height)
        fov = 35.
        center, eye, up, near, far = fitted_stock_camera(
            self.sim.stock_lo, self.sim.stock_hi, direction, up, aspect, fov=fov)
        bounds = self.o3d.geometry.AxisAlignedBoundingBox(self.sim.stock_lo, self.sim.stock_hi)
        self.scene.setup_camera(fov, bounds, center)
        self.scene.look_at(center, eye, up)
        self.scene.scene.camera.set_projection(fov, aspect, near, far, self.rendering.Camera.FovType.Vertical)
        self._tool_scene.light.update()
        self._request_redraw()

    def _update_paths(self):
        from cutsim.core.commands import command_endpoints
        from cutsim.ui.workspace_model import playback_path_indices
        if not self._workspace_ready:
            return
        current = self.render_delta_index if self._playing else self._selected_block
        signature = (self.path_toggle.checked, current, len(self.sim.deltas), self._playing)
        if signature == self._path_signature:
            return
        self._path_signature = signature
        self._remove('workspace_paths')
        if self.path_toggle.checked and self.sim.program:
            points, lines, colors = [], [], []
            indices = (playback_path_indices(self.sim.program, current, self.machining_project)
                       if self._playing else range(len(self.sim.program)))
            for index in indices:
                cmd = self.sim.program[index]
                offset = len(points)
                points.extend(command_endpoints(cmd))
                lines.append((offset, offset + 1))
                colors.append([1., .72, .18] if index == current else
                              ([.25, .8, .6] if index < len(self.sim.deltas) else [.5, .55, .65]))
            if not lines:
                self._request_redraw()
                return
            geometry = self.o3d.geometry.LineSet()
            geometry.points = self.o3d.utility.Vector3dVector(points)
            geometry.lines = self.o3d.utility.Vector2iVector(lines)
            geometry.colors = self.o3d.utility.Vector3dVector(colors)
            self.scene.scene.add_geometry('workspace_paths', geometry, self.line_mat)
        self._request_redraw()

    def _redraw(self):
        super()._redraw()
        if self._workspace_ready:
            self._refresh_workspace()
            self._update_paths()

    def _refresh_workspace(self):
        if not self._workspace_ready:
            return
        computed, total = len(self.sim.deltas), len(self.sim.program)
        if (self._playing and self.render_delta_index >= 0
                and getattr(self, '_playback_seek_target', None) is None):
            self._selected_block = self.render_delta_index
        self.program_summary.text = f'{total} blocks  /  {len(self._tools)} tools\n{computed} computed  /  {total - computed} remaining'
        signature = (computed, total)
        if signature != self._list_signature:
            self._selection_sync = True
            try:
                self.block_list.set_items([block_label(i, cmd, computed) for i, cmd in enumerate(self.sim.program)])
                self.block_list.selected_index = self._selected_block
            finally:
                self._selection_sync = False
            self._list_signature = signature
        if self.block_list.selected_index != self._selected_block:
            self._selection_sync = True
            try:
                self.block_list.selected_index = self._selected_block
            finally:
                self._selection_sync = False
        if 0 <= self._selected_block < total:
            from cutsim.ui.workspace_model import block_timing_text
            self.block_info.text = block_details(self._selected_block, self.sim.program[self._selected_block], computed)+block_timing_text(self.sim,self._selected_block)
        self._syncing_progress = True
        try:
            self.workspace_progress.double_value = float(np.float32(self.progress * 100))
        finally:
            self._syncing_progress = False
        self.workspace_position.text = f'{self.progress * 100:.1f}%'
        self.workspace_speed_label.text = self.speed_label.text
        state = 'Preparing playback' if self._playback_prepare_index is not None else ('Playing' if self._playing else 'Paused')
        if self._playing and getattr(self, '_loop_block', False):
            state = 'Looping block'
        elif self._playing and self._busy and (not computed or
                (self.render_delta_index == computed - 1 and self.progress >= 1.)):
            state = 'Waiting for next computed block'
        viewing = f'Block {self.render_delta_index + 1}/{total}' if self.render_delta_index >= 0 else 'Initial stock'
        self.playback_info.text = f'{viewing}  |  {state}  |  {computed}/{total} ready'
        self.notice.text = self.status.text.split('\n')[0] if self._playback_prepare_index is not None else self._message
        if getattr(self, '_native_upload_error', ''):
            self.notice.text = self._native_upload_error
        available = computed > 0
        self.play_button.enabled = (available or self._busy) and not self._playing
        self.pause_button.enabled = self._playing
        self.prev_button.enabled = not self._closing and 0 < self._selected_block < total
        self.next_button.enabled = not self._closing and 0 <= self._selected_block < total - 1
        self.workspace_progress.enabled = available
        self.play_block_button.enabled = 0 <= self._selected_block < computed
        self.compute_all_button.enabled = computed < total and not self._failed and not self._busy
        self.cancel_button.text = 'Stop Compute'
        density_enabled = not self._busy and not self._closing
        self.density_input.enabled = self.density_apply.enabled = density_enabled
        if not self._busy:
            self.compute_bar.value = computed / max(1, total)
            self.compute_info.text = f'{computed}/{total} blocks ready' if not self._failed else 'Calculation failed: Reset to compute again; completed blocks can be replayed'

    def _set_busy_controls(self, busy):
        if busy:
            self._disabled_widgets = []
            def disable(widget):
                self._disabled_widgets.append((widget, widget.enabled))
                widget.enabled = False
                for child in widget.get_children():
                    disable(child)
            # Debug settings stay fixed for worker endpoint preparation. Program
            # selection, camera and transport operate on GUI-owned snapshots.
            for root in (self.panel,):
                disable(root)
            for button in (self.compute_all_button, self.reset_button, self.density_input,
                           self.density_apply,self.workspace_size,self.workspace_colors):
                self._disabled_widgets.append((button, button.enabled))
                button.enabled = False
        else:
            for widget, enabled in self._disabled_widgets:
                widget.enabled = enabled
            self._disabled_widgets = []
        self.cancel_button.enabled = busy and not self._closing

    def _start_compute(self, all_blocks):
        if self._busy or self._failed:
            return
        self._compute_start = len(self.sim.deltas)
        count = len(self.sim.program) - self._compute_start if all_blocks else min(1, len(self.sim.program) - self._compute_start)
        if count <= 0:
            return
        self._busy = True
        self._set_busy_controls(True)
        self.compute_info.text = f'Computing block {self._compute_start + 1}/{len(self.sim.program)}...'
        self.window.post_redraw()
        # Commit a last slider edit before freezing the worker configuration.
        # Cutting and CPU endpoint preparation own private caches. Completed
        # buffers cross the queue; Open3D calls remain on the GUI thread.
        if self._sigma_updates.pending is not None:
            self._sigma_updates.reset(self._sigma_updates.pending)
        settings = self._render_settings(half_clip=bool(self.half_gs.checked))
        attributes, cache, sim = self._compute_attributes, self._compute_render_cache, self._compute_sim
        def prepare(delta):
            attributes.prepare_computed_block(delta, sim.material, settings, cache)
            return CompletedBlock.capture(sim, delta, attributes.export())
        from cutsim.application.process_job import ProcessComputeJob
        if isinstance(self._job,ProcessComputeJob):
            self._job.start_prepared(sim,count,settings,self._compute_start)
        else:
            self._job.start(sim, count, prepare_block=prepare)
        self._refresh_workspace()

    def _compute_next(self):
        self._start_compute(False)

    def _compute_all(self):
        self._start_compute(True)

    def _cancel_compute(self):
        if self._busy:
            self._job.cancel()
            self.cancel_button.enabled = False
            self.compute_info.text = 'Stopping after the current block completes...'
            self.window.post_redraw()

    def _apply_slider_updates(self):
        if not self._workspace_ready:
            return super()._apply_slider_updates()
        published = False
        view_changed = False
        events = []
        if self._closing and not self._busy and not self._job.startup_running:
            self.window.close()
            return True
        if self._busy or not self._closing:
            events = self._job.drain()
            for event in events:
                self.compute_bar.value = event.completed / max(1, len(self.sim.program))
                if event.kind == 'progress' and event.result is not None:
                    block = event.result
                    self.sim.publish(block)
                    if event.metrics is not None and hasattr(self.sim,'record_metrics'):
                        self.sim.record_metrics(block.delta.block_index,event.metrics)
                    self._playback_attributes.install(block.delta, block.attributes,
                        self._render_settings(half_clip=bool(self.half_gs.checked)))
                    if self.render_delta_index < 0:
                        self.render_delta_index = 0
                        self._sync_progress(0.)
                        view_changed = True
                    published = True
                if event.kind in ('done', 'error'):
                    self._busy = False
                    self._failed = event.kind == 'error'
                    self._set_busy_controls(False)
                    self._message = event.message
                    if self._closing:
                        self.window.close()
                        return True
                    if self._failed:
                        self.window.show_message_box('Calculation failed', event.message +
                            '\nCompleted blocks remain available. Reset before computing again.')
                elif self.cancel_button.enabled:
                    if event.kind == 'warming':
                        continue  # Internal compilation does not change the operator UI.
                    elif event.kind == 'preparing':
                        self.compute_info.text = f'Preparing playback: block {event.completed}/{len(self.sim.program)}...'
                    elif event.kind == 'working':
                        self.compute_info.text = f'Computing block {event.completed + 1}/{len(self.sim.program)}...'
                    else:
                        self.compute_info.text = f'{event.completed}/{len(self.sim.program)} ready'
        # Play can be armed before the first result. No intermediate material
        # or geometry is evaluated while waiting at the publication frontier.
        if self._playing and self._apply_playback_seek():
            changed = False
        elif self._playing and not self.sim.deltas:
            if not self._busy:
                self._stop()
            changed = False
        else:
            changed = super()._apply_slider_updates()
        # A newly completed future block does not change a paused viewport.
        # Keep its current GPU buffers and camera interaction uninterrupted.
        if view_changed and not changed:
            self._redraw()
        self._refresh_workspace()
        self._update_paths()
        if events:
            self.window.post_redraw()
        return changed or published or bool(events)

    def _play_selected_block(self):
        if 0 <= self._selected_block < len(self.sim.deltas):
            self._set_render_delta(self._selected_block, 0.)
            self._play()
            self._play_block_only = True

    def _advance_playback(self):
        if getattr(self, '_playback_scrubbing', False):
            self._playback_time = self._playback_next_frame = None
            return False
        if self._playing and getattr(self, '_loop_block', False) and self.progress >= 1.:
            now = time.perf_counter()
            if self._playback_next_frame is not None and now + 1e-9 < self._playback_next_frame:
                return False
            # Show the completed endpoint for one frame, then rewind only motion.
            self._sync_progress(0.)
            self._playback_time = now
            self._playback_next_frame = now + 1./30.
            return True
        changed = super()._advance_playback()
        if getattr(self, '_play_block_only', False) and self.progress >= 1.:
            self._stop()
        return changed

    def _stop(self):
        self._playback_seek_target = None
        self._play_block_only = False
        self._loop_block = False
        super()._stop()

    def _toggle_playback(self):
        if self._playing:
            self._space_resume_repeat = bool(self._loop_block)
            self._stop()
        elif getattr(self, '_space_resume_repeat', False):
            self._loop_selected_block(resume=True)
        else:
            self._play()

    def _loop_selected_block(self, *, resume=False):
        if 0 <= self._selected_block < len(self.sim.deltas):
            if self._playing and self.render_delta_index == self._selected_block:
                self._play_block_only = False
                self._loop_block = True
                self._refresh_workspace()
                return
            if resume and self.render_delta_index == self._selected_block:
                # At the endpoint, rewind this block rather than letting Play
                # advance to the next block. Otherwise retain the paused pose.
                if self.progress >= 1.:
                    self._sync_progress(0.)
            else:
                self._set_render_delta(self._selected_block, 0.)
            self._play()
            self._loop_block = self._playing
            self._refresh_workspace()

    def _play(self):
        self._space_resume_repeat = False
        self._loop_block = False
        self._play_block_only = False
        self._message = ''
        if self._busy and (not self.sim.deltas or
                (self.render_delta_index == len(self.sim.deltas) - 1 and self.progress >= 1.)):
            self.cut_debug.checked = False
            self._playing = True
            self._playback_settings = self._render_settings(half_clip=bool(self.half_gs.checked))
            self._playback_time = self._playback_next_frame = None
        else:
            super()._play()
        self._refresh_workspace()

    def _playback_has_pending_blocks(self):
        return getattr(self, '_loop_block', False) or (self._busy and not self._closing)

    def _confirm_reset(self):
        if self._busy:
            return
        dialog = self.gui.Dialog('Reset calculation')
        content = self.gui.Vert(10, self.gui.Margins(16, 16, 16, 16))
        content.add_child(self.gui.Label('Discard computed results and return to the initial stock?\nThe tool list and program are kept.'))
        row = self.gui.Horiz(8)
        self._button(row, 'Cancel', self.window.close_dialog)
        def confirm():
            self.window.close_dialog()
            self._reset()
        self._button(row, 'Reset', confirm)
        content.add_child(row)
        dialog.add_child(content)
        self.window.show_dialog(dialog)

    def _apply_gs_density(self, value):
        from cutsim.geometry.sampling_density import validate_gs_density
        if self._busy or self._closing:
            return
        value = validate_gs_density(value)
        if value == getattr(self._compute_sim, 'gs_density', 1.):
            return
        self._compute_sim.gs_density = value
        self._reset()
        self.density_input.double_value = 100.*value
        self._message = f'GS density {100.*value:g}% applied. Compute All to recalculate.'
        self._refresh_workspace()

    def load_machining_project(self, project):
        """Replace stock and program only while the compute worker is idle."""
        if self._busy or self._closing:
            raise RuntimeError('Stop calculation and wait for the current block before changing machining data')
        from cutsim.cutting.realtime_simulator import SweepDrivenRealtimeSimulator
        old = self._compute_sim
        options = {key: getattr(old, key) for key in (
            'initial_splat', 'grid_phase', 'max_fallback_patches', 'root_iterations',
            'use_splat_surface_grid', 'splat_sdf_k', 'adaptive_grid_refine',
            'adaptive_grid_max_depth', 'sweep_surface_repair', 'sweep_repair_max_new',
            'shape_update_mode', 'gs_density')}
        options.update(tool_width_cells=old.grid_sizing.tool_width_cells,
                       min_grid_cell=old.grid_sizing.min_cell, max_grid_cell=old.grid_sizing.max_cell,
                       ambiguity_band_cells=old.ambiguity_band_cells)
        # Validate and construct completely before changing the current workspace.
        replacement = SweepDrivenRealtimeSimulator(program=project.program,
            stock_lo=project.stock_lower, stock_hi=project.stock_upper, **options)
        replacement.grid_sizing = old.grid_sizing
        replacement.machining_project = project
        self._stop()
        if hasattr(self, 'hide_tool_preview'):
            self.hide_tool_preview()
        self._compute_sim = replacement
        self.machining_project = project
        self._selected_block = 0
        self._play_block_only = False
        self._path_signature = self._list_signature = None
        self._tools = tools_in_program(project.program)
        self.tool_list.set_items([tool.label for tool in self._tools])
        self.tool_info.text = 'Select a tool to inspect its usage.'
        dims = replacement.stock_hi - replacement.stock_lo
        self.stock_summary.text = f'Stock  {dims[0]:g} x {dims[1]:g} x {dims[2]:g} mm'
        self._reset()
        self._camera_view('Isometric')
        self._message = f'Loaded {project.name}. Compute All to calculate.'
        self._refresh_workspace()

    def _reset(self):
        if self._busy:
            return
        self._space_resume_repeat = False
        self._failed = False
        self._list_signature = None
        self._message = 'Calculation reset. The program is ready to compute.'
        super()._reset()

    def _reset_simulation(self):
        from cutsim.application.process_job import ProcessComputeJob
        if isinstance(self._job, ProcessComputeJob):self._job.reset_session()
        self._compute_sim.reset()
        self.sim = DisplayState(self._compute_sim, disk_history=history_enabled())
        self._compute_attributes.clear()
        self._compute_render_cache.clear()

    def _on_close(self):
        self._stop()
        if self._busy or self._job.startup_running:
            self._closing = True
            self._job.cancel()
            self._cancel_compute()
            self.compute_info.text = 'Closing...'
            return False
        self._close_compute_worker()
        return True

    def _close_compute_worker(self):
        from cutsim.application.process_job import ProcessComputeJob
        if isinstance(self._job,ProcessComputeJob):self._job.close()
