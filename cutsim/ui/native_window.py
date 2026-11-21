"""Windows HWND hosting; neither copies pixels nor creates another renderer."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys


VIEWPORT_BACKGROUND = (.055, .065, .08, 1.)


def child_style(style):
    # Strip top-level decorations, keep the rendering/input styles.
    return (style & ~(0x80000000 | 0x00C00000 | 0x00040000 |
                      0x00080000 | 0x00020000 | 0x00010000)) | 0x40000000 | 0x06000000


def create_embedded_window(gui,title,width,height):
    """Keep the temporary top-level viewport off the desktop, then hide it.

    Open3D Window::FLAG_HIDDEN is 1, but Application::AddWindow calls Show()
    unconditionally. Explicit off-desktop coordinates prevent that brief show
    from exposing the old Debug UI before Qt reparents the HWND. Qt controls
    its final position/size; the camera and render target are unchanged.
    """
    api=ctypes.WinDLL('user32',use_last_error=True)
    api.GetSystemMetrics.argtypes=[ctypes.c_int]
    api.GetSystemMetrics.restype=ctypes.c_int
    right=api.GetSystemMetrics(76)+api.GetSystemMetrics(78)  # Virtual desktop x + width.
    window=gui.Application.instance.create_window(title,width,height,max(0,right)+64,0,1)
    # The window renderer defaults to white independently of SceneWidget's
    # background. Set it before embedding/showing, including tool previews.
    window.renderer.set_clear_color(VIEWPORT_BACKGROUND)
    window.show(False)
    return window


class WindowsHandle:
    def __init__(self, title):
        if sys.platform != 'win32':
            raise RuntimeError('The embedded viewport currently requires Windows.')
        api = ctypes.WinDLL('user32', use_last_error=True)
        api.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        api.FindWindowW.restype = wintypes.HWND
        api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        api.GetWindowThreadProcessId.restype = wintypes.DWORD
        # LONG_PTR is pointer-sized: ctypes' default int would truncate on x64.
        api.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        api.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        api.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        api.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        api.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        api.SetWindowPos.restype = wintypes.BOOL
        self.api = api
        self.hwnd = api.FindWindowW(None, title)
        if not self.hwnd:
            raise RuntimeError('The newly created Open3D viewport could not be located.')
        pid = wintypes.DWORD()
        api.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid))
        if pid.value != os.getpid():
            raise RuntimeError('Refusing to embed a window owned by another process.')

    def make_child(self):
        style = self.api.GetWindowLongPtrW(self.hwnd, -16)
        ctypes.set_last_error(0)
        old = self.api.SetWindowLongPtrW(self.hwnd, -16, child_style(style))
        if old == 0 and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        # Frame changed, without activation, movement, resizing or z-order change.
        if not self.api.SetWindowPos(self.hwnd, None, 0, 0, 0, 0, 0x0037):
            raise ctypes.WinError(ctypes.get_last_error())


class NativeViewportHost:
    """Qt owns only the wrapper; Open3D remains owner of the native window.

    close() must run on the GUI thread before Open3D Application cleanup.
    """
    def __init__(self, title, parent=None):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPalette, QWindow
        from PySide6.QtWidgets import QWidget
        self.handle = WindowsHandle(title)
        self.foreign = QWindow.fromWinId(int(self.handle.hwnd))
        if self.foreign is None:
            raise RuntimeError('Qt could not wrap the native Open3D viewport.')
        self.widget = None
        try:
            self.widget = QWidget.createWindowContainer(self.foreign, parent)
            palette = self.widget.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor.fromRgbF(*VIEWPORT_BACKGROUND))
            self.widget.setPalette(palette)
            self.widget.setAutoFillBackground(True)
            self.widget.setStyleSheet('background-color: #0e1114;')
            self.handle.make_child()
            self.widget.setMinimumSize(320, 240)
            self.widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            self.close()
            raise

    def close(self):
        import shiboken6
        if self.foreign is None:
            return
        if self.widget is not None:
            self.widget.hide()
        # Release ownership from createWindowContainer, then destroy wrappers
        # while the actual GLFW HWND is still valid. No DestroyWindow call here.
        self.foreign.setParent(None)
        self.foreign.hide()
        shiboken6.delete(self.foreign)
        self.foreign = None
        if self.widget is not None:
            shiboken6.delete(self.widget)
            self.widget = None
