"""Existing renderer/playback hosted by the Qt workspace.

The operator widgets are hidden; their state remains the adapter to the proven
workspace controller.
No GS, camera, clipping, or GPU readback conversion is introduced here.
"""
from __future__ import annotations

import os
import time
from uuid import uuid4

from cutsim.ui.workspace_gui import MachiningWorkspaceGUI
from cutsim.ui.machining_inspection import MachiningInspectionMixin
from cutsim.ui.tool_focus import ToolFocusMixin


class EmbeddedViewport(ToolFocusMixin, MachiningInspectionMixin, MachiningWorkspaceGUI):
    def _create_window(self,title,width,height):
        from cutsim.ui.native_window import create_embedded_window
        return create_embedded_window(self.gui,title,width,height)

    def __init__(self, sim, *, compute_job=None):
        self._tool_focus = False
        self._tool_focus_target = None
        self._machining_selection = None
        self._machining_click = None
        self._machining_view_key = None
        self._last_work_owners = None
        self.request_host_close = None
        self.request_host_playback_toggle = None
        self._tool_preview = None
        self._tool_dexels_visible = True
        self.tool_preview_status = ''
        self._redraw_queued = False
        self._redraw_needs_followup = False
        self._redraw_followup_at = None
        super().__init__(sim, compute_job=compute_job)
        from cutsim.ui.mouse_controls import camera_mouse_handler
        self.scene.set_on_mouse(camera_mouse_handler(self.gui, self._machining_mouse))
        self.window.set_on_key(self._playback_key)
        # Never locate a foreign process's viewer, even with another instance open.
        self.native_title = f'GaussianCutSim viewport {os.getpid()} {uuid4().hex}'
        self.window.title = self.native_title
        self._hide_operator_widgets()
        self.window.set_needs_layout()

    def _request_redraw(self):
        """Commit Qt-originated changes with Open3D's draw context current.

        Coalesce requests until the native pump runs. One delayed refresh also
        presents deferred GPU output when the pointer stays outside the view;
        stable idle windows do not continuously invalidate their scene cache.
        """
        self._redraw_needs_followup = True
        self._redraw_followup_at = None
        self._queue_redraw()

    def _queue_redraw(self):
        if self._redraw_queued or self._closing:
            return
        self._redraw_queued = True
        self.gui.Application.instance.post_to_main_thread(self.window, self._flush_redraw)

    def _flush_redraw(self):
        self._redraw_queued = False
        followup = self._redraw_needs_followup
        self._redraw_needs_followup = False
        if self._closing:
            self._redraw_followup_at = None
            return
        super()._request_redraw()
        self._redraw_followup_at = time.perf_counter() + .04 if followup else None

    def _on_view_tick(self):
        changed = super()._on_view_tick()
        changed = self._update_tool_focus() or changed
        if self._redraw_followup_at is not None and time.perf_counter() >= self._redraw_followup_at:
            self._redraw_followup_at = None
            self._queue_redraw()
            return True
        return changed

    def _hide_operator_widgets(self):
        for widget in (self.header, self.sidebar, self.transport, self.panel):
            widget.visible = False

    def _layout(self, context):
        if not self._workspace_ready:
            return super()._layout(context)
        self._hide_operator_widgets()
        r = self.window.content_rect
        self.scene.frame = self.gui.Rect(r.x, r.y, max(1, r.width), r.height)
        size = (self.scene.frame.width, self.scene.frame.height)
        if size != self._viewport_size:
            self._viewport_size = size
            self._viewport_pending_at = time.perf_counter()

    def _on_close(self):
        # Qt owns the lifetime. In particular detach its foreign-window wrapper
        # BEFORE Open3D destroys the HWND or the renderer singleton.
        if self.request_host_close is not None:
            self.request_host_close()
        return False

    def can_close(self):
        return MachiningWorkspaceGUI._on_close(self)

    def _playback_key(self, event):
        if event.key != self.gui.KeyName.SPACE:
            return False
        if (event.type == self.gui.KeyEvent.Type.DOWN and not event.is_repeat
                and self.request_host_playback_toggle is not None):
            self.request_host_playback_toggle()
        return True

    def select_block(self, row):
        self.block_list.selected_index = row
        self._select_block('')

    def set_paths_visible(self, value):
        self._set_paths_visible(value)

    def create_tool_preview(self):
        """A separate native surface hosted inside the Tools tab."""
        if self._tool_preview is None:
            from cutsim.ui.tool_preview import ToolPreviewViewport
            self._tool_preview = ToolPreviewViewport(self)
            self._tool_preview.window.set_on_key(self._playback_key)
        return self._tool_preview

    def show_tool_preview(self, tool, centers=True):
        if self._closing:
            return
        preview = self.create_tool_preview()
        self.tool_preview_status = preview.show(self.sim.program[tool.blocks[0]], centers)
        preview.set_dexels(self._tool_dexels_visible)

    def hide_tool_preview(self):
        if self._tool_preview is not None:
            self._tool_preview.hide()

    def set_tool_centers(self, value):
        if self._tool_preview is not None:
            self._tool_preview.set_centers(value)

    def set_tool_dexels(self, value):
        self._tool_dexels_visible = bool(value)
        if self._tool_preview is not None:
            self._tool_preview.set_dexels(value)
