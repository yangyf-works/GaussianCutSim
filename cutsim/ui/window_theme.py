"""Shared dock and native Windows title-bar accent for the dark workspace."""
import ctypes
from ctypes import wintypes
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDockWidget, QHBoxLayout, QLabel, QStyle, QToolButton, QWidget


TITLE_BACKGROUND = '#1b7199'
TITLE_FOREGROUND = '#f2f8ff'


def main_caption_height(widget):
    """Visible native caption height converted from physical to Qt pixels."""
    if sys.platform == 'win32' and QApplication.platformName() == 'windows' and widget.isVisible():
        try:
            api = ctypes.WinDLL('user32', use_last_error=True)
            dwm = ctypes.WinDLL('dwmapi', use_last_error=True)
            api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
            api.ClientToScreen.restype = wintypes.BOOL
            dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            dwm.DwmGetWindowAttribute.restype = ctypes.c_long
            frame, client = wintypes.RECT(), wintypes.POINT(0, 0)
            hwnd = int(widget.winId())
            # Extended frame bounds exclude Windows' invisible resize border.
            if (dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(frame), ctypes.sizeof(frame)) == 0
                    and api.ClientToScreen(hwnd, ctypes.byref(client))):
                height = (client.y-frame.top)/widget.devicePixelRatioF()
                if 16 <= height <= 80:
                    return round(height)
        except (OSError, AttributeError):
            pass
    return max(24, widget.style().pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, widget))


class DockTitleBar(QWidget):
    """Consistent, readable controls on docked and floating panels."""
    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self._main_window = dock.parentWidget()
        self.setObjectName('dockTitleBar')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 2, 6, 2)
        row.setSpacing(6)
        self.label = QLabel(dock.windowTitle())
        self.label.setObjectName('dockTitle')
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(self.label, 1)
        self.detach = QToolButton(self)
        self.close_button = QToolButton(self)
        for button, name, icon, action_name in (
                (self.detach, 'dockFloat', 'dock_float.svg', 'Detach panel'),
                (self.close_button, 'dockClose', 'dock_close.svg', 'Close panel')):
            button.setObjectName(name)
            button.setIcon(QIcon(str(Path(__file__).with_name('resources') / icon)))
            button.setIconSize(QSize(20, 20))
            button.setFixedSize(34, 28)
            button.setAccessibleName(f'{action_name}: {dock.windowTitle()}')
            row.addWidget(button)
        self.detach.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self.close_button.clicked.connect(dock.close)
        dock.windowTitleChanged.connect(self.label.setText)
        dock.topLevelChanged.connect(self._floating_changed)
        dock.featuresChanged.connect(self._features_changed)
        self._features_changed(dock.features())
        self.sync_height()

    def sync_height(self):
        height = main_caption_height(self._main_window or self.dock)
        self.setFixedHeight(height)
        button_height = max(12, height-4)
        icon_size = min(20, button_height-4)
        for button in (self.detach, self.close_button):
            button.setFixedSize(max(30, button_height+6), button_height)
            button.setIconSize(QSize(icon_size, icon_size))

    def _features_changed(self, features):
        self.detach.setVisible(bool(features & QDockWidget.DockWidgetFeature.DockWidgetFloatable))
        self.close_button.setVisible(bool(features & QDockWidget.DockWidgetFeature.DockWidgetClosable))

    def _floating_changed(self, floating):
        icon = 'dock_restore.svg' if floating else 'dock_float.svg'
        action_name = 'Dock panel' if floating else 'Detach panel'
        self.detach.setIcon(QIcon(str(Path(__file__).with_name('resources') / icon)))
        self.detach.setAccessibleName(f'{action_name}: {self.dock.windowTitle()}')


def _colorref(color):
    """Windows COLORREF packs RGB in the opposite byte order to CSS hex."""
    return int(color[1:3],16) | (int(color[3:5],16)<<8) | (int(color[5:7],16)<<16)


def set_native_title_colors(hwnd):
    """Style this app's native caption when the Windows DWM supports it."""
    if sys.platform != 'win32':
        return False
    try:
        dwm = ctypes.WinDLL('dwmapi',use_last_error=True)
    except OSError:
        return False
    set_attribute = dwm.DwmSetWindowAttribute
    set_attribute.argtypes = [wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
    set_attribute.restype = ctypes.c_long
    values = ((20,1),  # DWMWA_USE_IMMERSIVE_DARK_MODE: light window controls.
              (34,_colorref(TITLE_BACKGROUND)),  # DWMWA_BORDER_COLOR
              (35,_colorref(TITLE_BACKGROUND)),  # DWMWA_CAPTION_COLOR
              (36,_colorref(TITLE_FOREGROUND)))  # DWMWA_TEXT_COLOR
    applied = []
    for attribute, color in values:
        value = wintypes.DWORD(color)
        applied.append(set_attribute(int(hwnd),attribute,ctypes.byref(value),ctypes.sizeof(value)) == 0)
    # Caption/text colors are Windows 11 attributes. Older systems retain
    # their native dark decoration without preventing the workspace opening.
    return applied[2] and applied[3]


class WindowTitleTheme(QObject):
    """Reapply the native accent after showing or recreating a floating HWND."""
    def watch(self, widget):
        widget.installEventFilter(self)

    def eventFilter(self, widget, event):
        if event.type() in (QEvent.Type.Show,QEvent.Type.WinIdChange,
                            QEvent.Type.DevicePixelRatioChange,QEvent.Type.WindowStateChange):
            # The dock may still be changing its native window flags here.
            # The QObject context also cancels this callback on destruction.
            QTimer.singleShot(0,widget,lambda:self.apply(widget))
        return False

    def apply(self, widget):
        for title in widget.findChildren(DockTitleBar):
            title.sync_height()
        if (QApplication.platformName() == 'windows' and
                widget.isWindow() and widget.isVisible()):
            set_native_title_colors(int(widget.winId()))
