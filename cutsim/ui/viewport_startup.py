"""Do not expose a native swapchain before its first scene has rendered."""
from threading import Event


class ViewportStartup:
    def __init__(self, window, backend):
        self.window, self.backend = window, backend
        self._rendered = Event()
        self._requested = False
        self._present_ticks = 0
        self.complete = False
        # Unlike hiding/minimizing, transparency keeps the native window
        # renderable and gives Qt/Open3D their final layout and swapchain size.
        window.setWindowOpacity(0.)

    def advance(self):
        if self.complete:
            return
        b = self.backend
        if not self._requested:
            frame = b.scene.frame
            if frame.width <= 0 or frame.height <= 0 or b._viewport_pending_at is not None:
                return
            self._requested = True
            # This completion comes from the GPU, not from a fixed startup
            # delay or a mouse event. No image is saved or used for display.
            b.scene.scene.scene.render_to_image(lambda _image: self._rendered.set())
            b.scene.force_redraw()
            b.window.post_redraw()
            return
        if not self._rendered.is_set():
            return
        # ImGui/GS composite textures are bound on subsequent window draws.
        # Keep pumping while transparent, then expose the composed window.
        self._present_ticks += 1
        b.scene.force_redraw()
        b.window.post_redraw()
        if self._present_ticks >= 3:
            self.window.setWindowOpacity(1.)
            self.complete = True
