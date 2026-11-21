"""Keep Open3D camera controls without its temporary sun-light editor."""


def camera_mouse_handler(gui, on_mouse=None):
    """Filter lighting shortcuts before Open3D or an application callback."""
    def handle(event):
        # ROTATE_CAMERA temporarily switches to ROTATE_SUN on middle-button
        # or Alt press. Block its start and drag, but allow releases through
        # so an existing camera gesture can finish. Wheel zoom is unaffected.
        if event.type in (gui.MouseEvent.Type.BUTTON_DOWN, gui.MouseEvent.Type.DRAG):
            if (event.is_button_down(gui.MouseButton.MIDDLE)
                    or event.is_modifier_down(gui.KeyModifier.ALT)):
                return gui.Widget.EventCallbackResult.CONSUMED
        if on_mouse is not None:
            return on_mouse(event)
        return gui.Widget.EventCallbackResult.IGNORED

    return handle
