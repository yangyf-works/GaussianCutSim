"""PySide6 operator shell with the native Open3D viewport in one window."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QSignalBlocker, QTimer, Slot, QPoint
from PySide6.QtGui import QKeySequence, QPainter, QColor, QPolygon
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QTabWidget,
    QScrollArea, QSizePolicy, QStyle, QStyleOptionSpinBox, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
)

from cutsim.ui.workspace_model import block_details_html, block_timing_html, tool_details_html, tools_in_program
from cutsim.ui.helpers import GS_DISPLAY_SIZES, PLAYBACK_SPEEDS
from cutsim.ui.block_slider import BlockSlider, SeekSlider
from cutsim.ui.program_table import ProgramTable
from cutsim.ui.tool_table import ToolTable
from cutsim.ui.playback_shortcuts import PlaybackSpaceFilter
from cutsim.ui.program_commands import source_details_html
from cutsim.core.machining_data import DEFAULT_DATA_ROOT, discover_datasets, load_project
from cutsim.geometry.sampling_density import GS_DENSITY_PERCENTAGES
from cutsim.ui.window_theme import TITLE_BACKGROUND, TITLE_FOREGROUND, WindowTitleTheme, DockTitleBar


SEPARATOR_COLOR = '#657486'

STYLE = """
QMainWindow, QWidget { background: #20252d; color: #e5eaf2; }
QToolBar { background: #292f39; border: 0; spacing: 8px; padding: 8px; }
QWidget#calculationStatus, QWidget#controlSeparatorContainer,
QToolBar#calculationToolbar QLabel { background: transparent; }
QDockWidget::title { background: @titleBackground; color: @titleForeground;
                     padding: 8px; font-weight: bold; }
QWidget#dockTitleBar { background: @titleBackground; }
QLabel#dockTitle { background: transparent; color: @titleForeground; }
QToolButton#dockFloat, QToolButton#dockClose {
    background: #155a7c; border: 1px solid #77bbd6; border-radius: 3px;
    padding: 0;
}
QToolButton#dockFloat:hover { background: #348bb2; border-color: #d7f2ff; }
QToolButton#dockFloat:pressed { background: #10425d; }
QToolButton#dockClose:hover { background: #c74248; border-color: #ffd6d8; }
QToolButton#dockClose:pressed { background: #a52e35; }
QToolButton#dockFloat:focus, QToolButton#dockClose:focus { border: 2px solid #f2f8ff; }
QPushButton { background: #343e4c; border: 1px solid #4b596c;
              border-radius: 4px; padding: 7px 12px; }
QPushButton:hover { background: #46556a; }
QPushButton:pressed, QPushButton:checked { background: #236f88; }
QPushButton:disabled { color: #788493; background: #292f38; border-color: #353d48; }
QPushButton#primary { background: #177e99; border-color: #3099b3; font-weight: bold; }
QPushButton#primary:disabled { background: #2d424b; color: #788493; }
QPushButton#stopCompute:enabled { background: #9b3a43; border-color: #cc6871; color: #ffffff; }
QPushButton#stopCompute:enabled:hover { background: #b14751; }
QPushButton#stopCompute:enabled:pressed { background: #7d2c35; }
QPushButton#pausePlayback:enabled { background: #d2ad43; border-color: #e8c96a; color: #20252d; }
QPushButton#pausePlayback:enabled:hover { background: #e2bf55; }
QPushButton#pausePlayback:enabled:pressed { background: #b79432; }
QPushButton#blockNavigation { background: #2b4663; border-color: #587da2; padding: 7px 6px; }
QPushButton#blockNavigation:hover { background: #395e84; }
QPushButton#blockNavigation:pressed { background: #223b55; }
QPushButton#blockNavigation:disabled { background: #293440; color: #7e91a6; border-color: #3d5063; }
QPushButton#playbackMode { background: #294b40; border-color: #548b73; }
QPushButton#playbackMode:hover { background: #356653; }
QPushButton#playbackMode:pressed { background: #214438; }
QPushButton#playbackMode:disabled { background: #293b33; color: #80988b; border-color: #3e584b; }
QPushButton#playbackMode:checked, QPushButton#playbackMode:checked:disabled {
    background: #21815c; border: 2px solid #79d9aa; color: #ffffff; font-weight: bold;
}
QWidget#uiSeparator { background: @separatorColor; }
QMenu::separator { background: @separatorColor; height: 1px; margin: 4px 6px; }
QToolBar::separator { background: @separatorColor; width: 1px; margin: 4px; }
QPushButton#valueStep { padding: 3px; min-height: 24px; }
QPushButton#cameraPreset { padding: 5px 6px; }
QPushButton#toolFocus:checked:enabled { background: #22576c; border-color: #50c3dc; color: white; }
QTableWidget { background: #191f27; alternate-background-color: #242d38;
               gridline-color: #353e4c; border: 1px solid #3d4858; }
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background: #275a72; }
QHeaderView::section { background: #303a48; padding: 6px; border: 0;
                       border-bottom: 1px solid #4b596c; }
QTabBar::tab { background: #2a333f; padding: 9px 18px; }
QTabBar::tab:selected { background: #3a4c5e; border-bottom: 2px solid #50c3dc; }
QTabBar#workspaceTabBar::tab {
    background: #29333f; color: #c5d1df; font-weight: bold;
    border: 1px solid #4b596c; border-bottom: 3px solid #4b596c;
    padding: 6px 12px;
}
QTabBar#workspaceTabBar::tab:hover { background: #354858; }
QTabBar#workspaceTabBar::tab:selected {
    background: #22576c; color: #f2f8ff; border-color: #527b8b; border-bottom-color: #50c3dc;
}
QTabWidget::pane { border: 0; }
QLabel#machiningDatasetSummary {
    background: #29333f; border: 1px solid #465263; border-radius: 4px;
}
QTabWidget#workspaceTabs::pane { background: #1b222b; border: 1px solid #4b596c; }
QGroupBox { border: 1px solid #465263; border-radius: 4px; margin-top: 14px; padding: 12px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QDoubleSpinBox, QComboBox { background: #17212b; padding: 5px; border: 1px solid #48576a; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; width: 22px; background: #34495e;
    border-left: 1px solid #5b7188;
}
QDoubleSpinBox::up-button { subcontrol-position: top right; border-bottom: 1px solid #5b7188; }
QDoubleSpinBox::down-button { subcontrol-position: bottom right; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #47627e; }
QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed { background: #25415d; }
QDoubleSpinBox::up-button:disabled, QDoubleSpinBox::down-button:disabled { background: #293541; }
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow { image: none; width: 0; height: 0; }
QDoubleSpinBox QLineEdit { background: transparent; border: 0; padding: 0; }
QDoubleSpinBox#materialValue { padding-right: 5px; }
QDoubleSpinBox#materialValue::up-button, QDoubleSpinBox#materialValue::down-button { width: 0px; height: 0px; }
QSlider { background: #20252d; }
QSlider::groove:horizontal { height: 6px; background: #414f60; border-radius: 3px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #5bbed6; border-radius: 5px;
                            border: 2px solid #9aafc2; }
QSlider[pending="true"]::handle:horizontal, QSlider::handle:horizontal:disabled { background: #87909c; }
QProgressBar { border: 1px solid #465263; border-radius: 3px; text-align: center; min-height: 18px; }
QProgressBar#calculationProgress { background: #17212b; }
QProgressBar::chunk { background: #248aa2; }
QLabel#muted { color: #a6b5c8; }
QStatusBar { background: #171e27; }
QToolTip { background: #2a333f; color: #e5eaf2; border: 1px solid #48576a; padding: 4px 7px; }
""".replace('@titleBackground',TITLE_BACKGROUND).replace('@titleForeground',TITLE_FOREGROUND).replace('@separatorColor', SEPARATOR_COLOR)


def set_silently(widget, setter, value):
    with QSignalBlocker(widget):
        getattr(widget, setter)(value)


class StepSpinBox(QDoubleSpinBox):
    """Display-only number, changed by native arrows or explicit step buttons."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setAcceptDrops(False)
        self._editor = editor = self.lineEdit()
        # Read-only on the editor, not the spin box: native arrows stay enabled.
        editor.setReadOnly(True)
        editor.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        editor.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        editor.setAcceptDrops(False)
        editor.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        editor.selectionChanged.connect(self._clear_selection)

    @Slot()
    def _clear_selection(self):
        self._editor.deselect()

    def stepBy(self, steps):
        super().stepBy(steps)
        self.lineEdit().deselect()

    def mousePressEvent(self, event):
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        control = self.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_SpinBox, option, event.position().toPoint(), self)
        if control in (QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown):
            super().mousePressEvent(event)
            self.lineEdit().deselect()
        else:
            event.accept()

    def mouseDoubleClickEvent(self, event):
        # Qt delivers the second rapid press as a double-click event. Arrows
        # must still step; presses on the number retain the read-only behavior.
        self.mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        for control, flag, direction in (
                (QStyle.SubControl.SC_SpinBoxUp, self.StepEnabledFlag.StepUpEnabled, -1),
                (QStyle.SubControl.SC_SpinBoxDown, self.StepEnabledFlag.StepDownEnabled, 1)):
            rect = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, self)
            if rect.width() < 10 or rect.height() < 6:
                continue  # Material fields use their separate +/- buttons.
            enabled = self.isEnabled() and bool(option.stepEnabled & flag)
            painter.setBrush(QColor('#dce8f3' if enabled else '#718297'))
            x, y = rect.center().x(), rect.center().y()
            painter.drawPolygon(QPolygon([QPoint(x - 4, y - direction * 2),
                QPoint(x + 4, y - direction * 2), QPoint(x, y + direction * 3)]))

    def keyPressEvent(self, event):
        event.ignore()

    def wheelEvent(self, event):
        event.ignore()


class PresetSpinBox(StepSpinBox):
    """Step through discrete values while retaining numeric value signals."""
    def __init__(self, values, suffix, decimals, parent=None):
        super().__init__(parent)
        self._presets = tuple(values)
        self.setRange(self._presets[0], self._presets[-1])
        self.setDecimals(decimals)
        self.setSuffix(suffix)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def textFromValue(self, value):
        return f'{value:g}'

    def stepBy(self, steps):
        from bisect import bisect_left, bisect_right
        if steps > 0:
            index = bisect_right(self._presets, self.value()) + steps - 1
        elif steps < 0:
            index = bisect_left(self._presets, self.value()) + steps
        else:
            return
        self.setValue(self._presets[max(0, min(index, len(self._presets) - 1))])
        self._clear_selection()


class PlaybackSpeedSpinBox(PresetSpinBox):
    def __init__(self, parent=None):
        super().__init__(PLAYBACK_SPEEDS, ' x', 2, parent)
        self.setValue(1.)


class DensitySpinBox(PresetSpinBox):
    def __init__(self, parent=None):
        super().__init__(GS_DENSITY_PERCENTAGES, '%', 0, parent)
        self.setValue(100.)


class GSSizeSpinBox(PresetSpinBox):
    def __init__(self, parent=None):
        super().__init__(GS_DISPLAY_SIZES, ' x', 2, parent)
        self.setValue(GS_DISPLAY_SIZES[-1])
        self.setAccessibleName('GS size')
        self.setFixedWidth(max(self.fontMetrics().horizontalAdvance(f'{value:g} x')
                               for value in GS_DISPLAY_SIZES) + 36)


class SteppedValue(QWidget):
    """Display-only number with large, explicit minus/plus buttons."""
    def __init__(self, label, minimum, maximum, step, decimals):
        super().__init__()
        self.limits = minimum, maximum
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.spin = StepSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        # Fusion + the workspace stylesheet collapses the editor with NoButtons.
        # Hide the native arrows in the stylesheet and keep its edit field intact.
        self.spin.setObjectName('materialValue')
        self.spin.setKeyboardTracking(False)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin.setAccessibleName(label)
        self.minus, self.plus = QPushButton('−'), QPushButton('+')
        for button, name, action in ((self.minus, 'Decrease', self.spin.stepDown),
                                     (self.plus, 'Increase', self.spin.stepUp)):
            button.setObjectName('valueStep')
            button.setFixedWidth(32)
            button.setAutoRepeat(True)
            button.setAutoRepeatInterval(150)
            button.setAccessibleName(f'{name} {label}')
            button.setToolTip(f'{name} {label} by {step:g}')
            button.clicked.connect(lambda _=False, action=action: action())
        row.addWidget(self.minus)
        row.addWidget(self.spin, 1)
        row.addWidget(self.plus)
        self.spin.valueChanged.connect(self._update_buttons)
        self._update_buttons()

    def _update_buttons(self, _value=None):
        self.minus.setEnabled(self.spin.value() > self.spin.minimum())
        self.plus.setEnabled(self.spin.value() < self.spin.maximum())

    def sync_value(self, value):
        # Debug accepts any nonnegative coefficient; keep those values visible.
        with QSignalBlocker(self.spin):
            self.spin.setRange(min(self.limits[0], value), max(self.limits[1], value))
            self.spin.setValue(value)
        self._update_buttons()


class WorkspaceTabs(QTabWidget):
    """Keep the two workspace modes prominent across the entire dock width."""
    def __init__(self):
        super().__init__()
        self.setObjectName('workspaceTabs')
        self.tabBar().setObjectName('workspaceTabBar')
        self.tabBar().setExpanding(True)
        self.tabBar().setUsesScrollButtons(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Match the 9px page margins instead of extending to the dock edges.
        self.tabBar().setFixedWidth(self.contentsRect().width())


class QtMachiningWindow(QMainWindow):
    """All operator controls are Qt. The backend retains algorithm ownership.

    backend and viewport_host are injectable for offscreen tests without creating
    an Open3D window or touching a user's screen.
    """
    def __init__(self, backend, viewport_host, *, persist=True, data_root=DEFAULT_DATA_ROOT,
                 preview_host_factory=None):
        super().__init__()
        self.backend, self.viewport_host = backend, viewport_host
        self.tool_preview_host = None
        self._preview_host_factory = preview_host_factory
        self._disposed = False
        self._in_tick = False
        self._last_refresh = 0.
        self._startup = None
        self._table_signature = None
        self._persist = persist
        self._settings = QSettings('GaussianCutSim', 'QtWorkspace') if persist else None
        self.data_root = Path(data_root).resolve()
        # Presentation data is replaced only at an idle dataset switch. Never
        # read mutable simulator state while the worker runs.
        self.program = tuple(backend.sim.program)
        self.tools = tools_in_program(self.program)
        self.setWindowTitle('GaussianCutSim')
        self.resize(1500, 940)
        self.setStyleSheet(STYLE)
        self._title_theme = WindowTitleTheme(self)
        self._title_theme.watch(self)
        self._build_toolbar()
        self._build_central(viewport_host.widget)
        self._build_program()
        self._build_view()
        self.resizeDocks([self.program_dock, self.view_dock], [370, 195], Qt.Orientation.Horizontal)
        self._default_layout_state = self.saveState()
        self._build_file_menu()
        view_menu = self.menuBar().addMenu('View')
        for dock in (self.program_dock, self.view_dock):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addAction('Restore panels', self._restore_panels)
        view_menu.addSeparator()
        view_menu.addAction('Reset window sizes', self._reset_window_sizes)
        self.backend.request_host_close = lambda: QTimer.singleShot(0, self.close)
        self.backend.request_host_playback_toggle = self._queue_playback_toggle
        self._playback_space_filter = PlaybackSpaceFilter(self, self._queue_playback_toggle)
        if self._settings is not None:
            geometry, state = self._settings.value('geometry'), self._settings.value('state')
            if geometry is not None:
                self.restoreGeometry(geometry)
            if state is not None:
                self.restoreState(state)
        self.refresh()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(8)
        self.timer.timeout.connect(self._tick)

    def _button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(lambda _=False: self._call(callback))
        return button

    def _size_button_group(self, buttons, alternate_labels=()):
        """Reserve one width for all labels and active styles in a control group."""
        width = 0
        for button in buttons:
            probe = QPushButton(self)
            probe.hide()
            probe.setObjectName(button.objectName())
            probe.setCheckable(button.isCheckable())
            font = button.font()
            font.setBold(True)  # Active playback/primary buttons use bold text.
            probe.setFont(font)
            for checked in (False, True) if button.isCheckable() else (False,):
                probe.setChecked(checked)
                probe.style().unpolish(probe)
                probe.style().polish(probe)
                for label in (button.text(), *alternate_labels):
                    probe.setText(label)
                    width = max(width, probe.sizeHint().width())
            probe.deleteLater()
        for button in buttons:
            button.setFixedWidth(width)

    def _call(self, callback):
        if self._disposed or self.backend._closing:
            return
        try:
            callback()
        except Exception as exc:
            self.backend._stop()
            QMessageBox.critical(self, 'Operation failed', f'{type(exc).__name__}: {exc}')
        self.refresh()

    def _queue_playback_toggle(self):
        # Native input may arrive inside Open3D's message pump. Dispatch the
        # transport action through Qt after that pump returns.
        if (QApplication.activeModalWidget() is None and QApplication.activePopupWidget() is None
                and QApplication.queryKeyboardModifiers() == Qt.KeyboardModifier.NoModifier):
            QTimer.singleShot(0, self._toggle_playback)

    def _toggle_playback(self):
        if (self._disposed or self.backend._closing or QApplication.activeModalWidget() is not None
                or QApplication.activePopupWidget() is not None):
            return
        if self.backend._playing or self.play.isEnabled():
            self._call(self.backend._toggle_playback)

    @staticmethod
    def _control_separator():
        container = QWidget()
        container.setObjectName('controlSeparatorContainer')
        container.setFixedWidth(5)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 0, 2, 0)
        line = QWidget()
        line.setObjectName('uiSeparator')
        line.setFixedSize(1, 28)
        layout.addWidget(line, 0, Qt.AlignmentFlag.AlignVCenter)
        return container

    @staticmethod
    def _horizontal_separator(layout):
        layout.addSpacing(4)
        line = QWidget()
        line.setObjectName('uiSeparator')
        line.setFixedHeight(1)
        layout.addWidget(line)
        layout.addSpacing(4)

    def _build_toolbar(self):
        bar = QToolBar('Calculation', self)
        bar.setObjectName('calculationToolbar')
        bar.setMovable(False)
        self.addToolBar(bar)
        self.compute_all = self._button('Compute All', self.backend._compute_all)
        self.compute_all.setObjectName('primary')
        self.cancel = self._button('Stop Compute', self.backend._cancel_compute)
        self.cancel.setObjectName('stopCompute')
        self.reset = self._button('Reset', self._reset)
        for button in (self.compute_all, self.cancel, self.reset):
            bar.addWidget(button)
        self._size_button_group((self.compute_all, self.cancel))
        self._size_button_group((self.reset,))
        status = QWidget()
        status.setObjectName('calculationStatus')
        self.compute_status = status
        status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        status.setMaximumWidth(190)
        row = QHBoxLayout(status)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.notice = QLabel()
        self.notice.hide()
        self.notice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.notice, 1)
        self.compute_bar = QProgressBar()
        self.compute_bar.setObjectName('calculationProgress')
        self.compute_bar.setFormat('%p%')
        self.compute_bar.setFixedWidth(190)
        row.addWidget(self.compute_bar)
        bar.addWidget(status)
        bar.addWidget(self._control_separator())
        bar.addWidget(QLabel('GS density'))
        self.density = DensitySpinBox()
        self._applied_density = 100.*getattr(self.backend.sim, 'gs_density', 1.)
        self.density.setValue(self._applied_density)
        self.density.setToolTip('100% = current baseline. Controls surface and intersection sampling, curvature approximation and split size.\nAnalytic boundaries and critical clipping remain protected.\nApply discards computed results; then use Compute All.')
        bar.addWidget(self.density)
        self.density_apply = self._button('Apply',
            lambda: self.backend._apply_gs_density(self.density.value()/100.))
        bar.addWidget(self.density_apply)
        self.density.valueChanged.connect(lambda _: self.refresh())

    def _build_central(self, viewport):
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(viewport, 1)
        self.transport = QWidget()
        transport = QVBoxLayout(self.transport)
        transport.setContentsMargins(4, 8, 4, 4)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.previous_ten = self._button('<<', lambda: self._step_block(-10))
        self.previous = self._button('<', lambda: self._step_block(-1))
        self.play = self._button('Play All', self.backend._play)
        self.loop_block = self._button('Repeat', self.backend._loop_selected_block)
        for button in (self.play, self.loop_block):
            button.setObjectName('playbackMode')
            button.setCheckable(True)
        self.pause = self._button('Pause', self.backend._stop)
        self.pause.setObjectName('pausePlayback')
        self.next = self._button('>', lambda: self._step_block(1))
        self.next_ten = self._button('>>', lambda: self._step_block(10))
        navigation = (self.previous_ten, self.previous, self.next, self.next_ten)
        for button, name, hint in zip(navigation,
                ('Back 10 blocks', 'Previous block', 'Next block', 'Forward 10 blocks'),
                ('−10', '−1', '＋1', '＋10')):
            button.setObjectName('blockNavigation')
            button.setAccessibleName(name)
            button.setToolTip(hint)
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(400)
            button.setAutoRepeatInterval(120)
        for button in (self.play, self.loop_block, self.pause):
            row.addWidget(button)
        self._size_button_group(navigation)
        self._size_button_group((self.play, self.loop_block))
        self._size_button_group((self.pause,))
        row.addWidget(QLabel('Speed'))
        self.speed = PlaybackSpeedSpinBox()
        # Reserve the widest text in the ranges, with identical arrow controls.
        for spin in (self.density, self.speed):
            spin.ensurePolished()
        # QDoubleSpinBox's generic size hint reserves unused decimal digits.
        # Fit actual labels plus the 22px arrow column and text/border padding.
        number_width = max(
            *(self.density.fontMetrics().horizontalAdvance(f'{value}%') for value in GS_DENSITY_PERCENTAGES),
            *(self.speed.fontMetrics().horizontalAdvance(f'{value:g} x') for value in PLAYBACK_SPEEDS)) + 36
        for spin in (self.density, self.speed):
            spin.setFixedWidth(number_width)
        self.speed.setToolTip('Playback multiplier; not NC feed time.')
        self.speed.valueChanged.connect(lambda value: self._call(lambda: self.backend._set_speed(value)))
        row.addWidget(self.speed)
        row.addWidget(self._control_separator())
        row.addWidget(QLabel('Block'))
        navigation_row = QHBoxLayout()
        navigation_row.setContentsMargins(0, 0, 0, 0)
        navigation_row.setSpacing(4)
        for button in navigation:
            navigation_row.addWidget(button)
        row.addLayout(navigation_row)
        self.block_seek = BlockSlider()
        self.block_seek.set_program(self.program)
        self.block_seek.setSingleStep(1)
        self.block_seek.setTracking(True)
        self.block_seek.sliderPressed.connect(lambda: self.backend.set_playback_scrubbing(True))
        self.block_seek.sliderReleased.connect(lambda: self.backend.set_playback_scrubbing(False))
        self.block_seek.sliderMoved.connect(lambda _: self.refresh())
        self.block_seek.valueChanged.connect(lambda value: self._call(lambda: self.backend.seek_playback(value-1)))
        row.addWidget(self.block_seek, 3)
        self.block_number = QLabel()
        self.block_number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_block_number_width()
        row.addWidget(self.block_number)
        row.addSpacing(16)
        row.addWidget(QLabel('Within block'))
        self.progress = SeekSlider()
        self.progress.setRange(0, 1000)
        self.progress.sliderPressed.connect(lambda: self.backend.set_playback_scrubbing(True))
        self.progress.sliderReleased.connect(lambda: self.backend.set_playback_scrubbing(False))
        self.progress.valueChanged.connect(lambda value: self._call(lambda: self._seek_within_block(value)))
        row.addWidget(self.progress, 1)
        self.position = QLabel('0.0%')
        self.position.setFixedWidth(self.position.fontMetrics().horizontalAdvance('100.0%'))
        self.position.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.position)
        transport.addLayout(row)
        self.setCentralWidget(body)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.transport, 1)

    def _step_block(self, offset):
        # Fine adjustment follows the thumb, even when the viewport still
        # shows an earlier completed block or a drag has not committed yet.
        current = self.block_seek.sliderPosition()
        target = max(1, min(current + offset, len(self.program)))
        if self.program and target != current:
            self.backend.seek_playback(target - 1)

    def _update_block_number_width(self):
        # Reserve every possible numerator up front, including proportional
        # digits. Recalculate only when the program's total changes.
        label = self.block_number
        label.ensurePolished()
        metrics = label.fontMetrics()
        total = str(len(self.program))
        digit_width = max(metrics.horizontalAdvance(str(digit)) for digit in range(10))
        margins = label.contentsMargins()
        width = digit_width * len(total) + metrics.horizontalAdvance('/' + total)
        label.setFixedWidth(width + margins.left() + margins.right() + 2 * label.margin() + 2)

    def _seek_within_block(self, value):
        b = self.backend
        if not 0 <= b._selected_block < len(b.sim.deltas):
            return
        b.seek_playback(b._selected_block, value / 1000.)

    def _table(self, headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    @staticmethod
    def _fill_table(table, rows):
        # Replacing every item in a visible ResizeToContents table triggers
        # expensive layout/paint work on the GUI thread (627 rows for mold).
        # Keep existing items, selection and scroll position; publish only the
        # changed text and repaint once at the end of the batch.
        updates_enabled = table.updatesEnabled()
        table.setUpdatesEnabled(False)
        try:
            with QSignalBlocker(table):
                if table.rowCount() != len(rows):
                    table.setRowCount(len(rows))
                for row, values in enumerate(rows):
                    for column, value in enumerate(values):
                        text = str(value)
                        item = table.item(row, column)
                        if item is None:
                            table.setItem(row, column, QTableWidgetItem(text))
                        elif item.text() != text:
                            item.setText(text)
        finally:
            table.setUpdatesEnabled(updates_enabled)

    def _dock(self, name, title, contents, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.title_bar = DockTitleBar(dock)
        dock.setTitleBarWidget(dock.title_bar)
        self._title_theme.watch(dock)
        dock.setWidget(contents)
        dock.setMinimumWidth(270)
        self.addDockWidget(area, dock)
        return dock

    def _build_program(self):
        contents = QWidget()
        contents_layout = QVBoxLayout(contents)
        contents_layout.setContentsMargins(9, 8, 9, 8)
        contents_layout.setSpacing(8)
        self.tabs = WorkspaceTabs()
        self.dataset_info = QLabel()
        self.dataset_info.setObjectName('machiningDatasetSummary')
        self.dataset_info.setContentsMargins(8, 6, 8, 6)
        self.dataset_info.setWordWrap(True)
        self.dataset_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        contents_layout.addWidget(self.dataset_info)
        contents_layout.addWidget(self.tabs, 1)
        self._refresh_datasets()
        program = QWidget()
        layout = QVBoxLayout(program)
        layout.setContentsMargins(9, 9, 9, 9)
        self.blocks = ProgramTable()
        self.blocks.set_program(self.program, getattr(self.backend, 'machining_project', None))
        self.blocks.itemSelectionChanged.connect(self._select_block)
        layout.addWidget(self.blocks, 1)
        self._horizontal_separator(layout)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.RichText)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.timings = QLabel()
        self.timings.setWordWrap(True)
        self.timings.setTextFormat(Qt.TextFormat.RichText)
        self.timings.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.timings.setText(block_timing_html(None, None))
        layout.addWidget(self.timings)
        self.tabs.addTab(program, 'Program')
        tools = QWidget()
        layout = QVBoxLayout(tools)
        layout.setContentsMargins(9, 9, 9, 9)
        self.tool_table = ToolTable()
        self.tool_table.set_tools(self.tools, self.program)
        self.tool_table.itemSelectionChanged.connect(self._select_tool)
        self.tool_table.setMinimumHeight(180)
        self.tool_table.setMaximumHeight(240)
        layout.addWidget(self.tool_table)
        layout.addWidget(self._button('Go to first use', self._go_to_tool))
        self._horizontal_separator(layout)
        self.tool_preview_panel = QWidget()
        self.tool_preview_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.tool_preview_panel.setToolTip('Drag to rotate · Wheel to zoom\nGold: cutting · Gray: body · Cyan: Dexels')
        self.tool_preview_layout = QVBoxLayout(self.tool_preview_panel)
        self.tool_preview_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_preview_hint = QLabel('Select a tool to view its 3D model.')
        self.tool_preview_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tool_preview_hint.setWordWrap(True)
        self.tool_preview_layout.addWidget(self.tool_preview_hint)
        self.tool_preview_panel.setMinimumHeight(260)
        layout.addWidget(self.tool_preview_panel, 1)
        self.tabs.addTab(tools, 'Tools')
        options = QHBoxLayout()
        options.setSpacing(6)
        self.tool_centers = QCheckBox('GS centres')
        self.tool_centers.setChecked(True)
        self.tool_centers.toggled.connect(lambda value: self._call(lambda: self.backend.set_tool_centers(value)))
        options.addWidget(self.tool_centers)
        self.tool_dexels = QCheckBox('Dexels')
        self.tool_dexels.setChecked(True)
        self.tool_dexels.toggled.connect(lambda value: self._call(lambda: self.backend.set_tool_dexels(value)))
        options.addWidget(self.tool_dexels)
        self.tool_opaque = QCheckBox('Opaque Surface')
        self.tool_opaque.setChecked(True)
        self.tool_opaque.toggled.connect(lambda _: self._call(self._apply_tool_preview_style))
        options.addWidget(self.tool_opaque)
        options.addStretch()
        layout.addLayout(options)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel('GS size'))
        size_row.addStretch()
        self.tool_gs_size = GSSizeSpinBox()
        self.tool_gs_size.valueChanged.connect(lambda _: self._call(self._apply_tool_preview_style))
        size_row.addWidget(self.tool_gs_size)
        layout.addLayout(size_row)
        self.tool_details = QLabel()
        self.tool_details.setTextFormat(Qt.TextFormat.RichText)
        self.tool_details.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # A fixed information area prevents long values or ball-only fields
        # from stealing viewport height when the selected tool changes.
        self.tool_details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.tool_details.setFixedHeight(2*(self.tool_details.fontMetrics().height()+2)+8)
        layout.addWidget(self.tool_details)
        self.tool_preview_status = QLabel(tools)
        self.tool_preview_status.hide()
        self.tabs.currentChanged.connect(self._tool_tab_changed)
        self.program_dock = self._dock('programDock', 'Machining program', contents, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.program_dock.visibilityChanged.connect(self._tools_visibility_changed)

    def _refresh_datasets(self, *, update_info=True):
        active = getattr(self.backend, 'machining_project', None)
        previous = getattr(self, '_datasets', None)
        self._datasets = discover_datasets(self.data_root, validate=False)
        # A --project input outside the current library remains selectable.
        if active is not None and not any(entry.path == active.path for entry in self._datasets):
            from cutsim.core.machining_data import DatasetEntry
            self._datasets.append(DatasetEntry(active.path, active.name))
        if hasattr(self, 'dataset_menu') and self._datasets != previous:
            self._rebuild_dataset_menu()
        if update_info:
            self._update_dataset_info()
        else:
            self._sync_dataset_menu()

    def _build_file_menu(self):
        self.dataset_menu = self.menuBar().addMenu('File')
        self.dataset_menu.setToolTipsVisible(True)
        self.dataset_open_action = self.dataset_menu.addAction(
            'Open…', lambda: self._call(self._open_dataset))
        self.dataset_open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.dataset_select_menu = self.dataset_menu.addMenu('Open Dataset')
        self.dataset_select_menu.setToolTipsVisible(True)
        self.dataset_menu.addSeparator()
        self.dataset_folder_action = self.dataset_menu.addAction(
            'Data Folder…', lambda: self._call(self._choose_data_folder))
        self.dataset_menu.addSeparator()
        self.dataset_exit_action = self.dataset_menu.addAction('Exit', self.close)
        self.dataset_exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        # Read project names on every open, but leave expensive tool/path
        # validation to loading. Avoid refreshing the rest of the workspace.
        self.dataset_menu.aboutToShow.connect(lambda: self._refresh_datasets(update_info=False))
        self._rebuild_dataset_menu()
        self._sync_dataset_menu()

    def _rebuild_dataset_menu(self):
        self.dataset_select_menu.clear()
        self._dataset_actions = []
        for entry in self._datasets:
            path = str(entry.path)
            action = self.dataset_select_menu.addAction(
                entry.name + (' [unavailable]' if entry.error else ''))
            action.setCheckable(True)
            action.setData(path)
            action.setToolTip(path + (f'\n{entry.error}' if entry.error else ''))
            action.triggered.connect(
                lambda _checked=False, path=path: self._call(lambda: self._load_dataset(path)))
            self._dataset_actions.append((action, bool(entry.error)))

    def _sync_dataset_menu(self):
        if not hasattr(self, 'dataset_menu'):
            return
        enabled = not self.backend._busy and not self.backend._closing
        self.dataset_select_menu.setEnabled(enabled and bool(self._datasets))
        self.dataset_open_action.setEnabled(enabled)
        self.dataset_folder_action.setEnabled(enabled)
        active = getattr(self.backend, 'machining_project', None)
        selected = str(active.path) if active is not None else None
        for action, unavailable in self._dataset_actions:
            action.setChecked(action.data() == selected)
            action.setEnabled(enabled and not unavailable)

    def _update_dataset_info(self):
        """Describe only the loaded project, never a prospective menu selection."""
        from html import escape
        active = getattr(self.backend, 'machining_project', None)
        source = active if active is not None else self.backend.sim
        lo = source.stock_lower if active is not None else source.stock_lo
        hi = source.stock_upper if active is not None else source.stock_hi
        size = [upper - lower for lower, upper in zip(lo, hi)]
        name = active.name if active is not None else 'Command-line / reference program'
        info = f'<div><span style="font-size:12pt; color:#e5eaf2"><b>{escape(name)}</b></span></div>'
        if active is not None and active.description:
            info += f'<p style="margin-top:5px; margin-bottom:9px; color:#a6b5c8">{escape(active.description)}</p>'
        fields = [('Stock size', f'{size[0]:g} × {size[1]:g} × {size[2]:g} mm'),
                  ('Tools', str(len(self.tools))), ('Cutting blocks', str(len(self.program)))]
        info += '<table width="100%" border="0" cellspacing="0" cellpadding="1"><tr>'
        info += ''.join(f'<td style="color:#a6b5c8">{label}</td>' for label, _ in fields)
        info += '</tr><tr>' + ''.join(f'<td><b>{value}</b></td>' for _, value in fields) + '</tr></table>'
        if active is not None and len(active.blocks) != len(self.program):
            info += (f'<p style="margin-top:5px; margin-bottom:0; color:#a6b5c8">'
                     f'Source instructions: {len(active.blocks)} · including non-cutting moves</p>')
        self.dataset_info.setTextFormat(Qt.TextFormat.RichText)
        self.dataset_info.setText(info)
        self.dataset_info.setToolTip(str(active.path) if active is not None else '')
        self._sync_dataset_menu()

    def _open_dataset(self):
        if self.backend._busy or self.backend._closing:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open machining data', str(self.data_root),
            'Machining project (project.json);;JSON files (*.json)')
        if path:
            self._load_dataset(path)

    def _choose_data_folder(self):
        if self.backend._busy or self.backend._closing:
            return
        folder = QFileDialog.getExistingDirectory(self, 'Select machining data library', str(self.data_root))
        if folder:
            self.data_root = Path(folder).resolve()
            self._refresh_datasets()

    def _load_dataset(self, path):
        if self.backend._busy or self.backend._closing:
            return
        if not path:
            return
        project = load_project(path)
        self.backend.load_machining_project(project)
        self.program = tuple(self.backend.sim.program)
        self.tools = tools_in_program(self.program)
        self.block_seek.set_program(self.program)
        self._update_block_number_width()
        self._table_signature = None
        self.tool_table.set_tools(self.tools, self.program)
        self.tool_table.clearSelection()
        self.tool_table.setCurrentCell(-1, -1)
        self.tool_details.setText('Select a tool to inspect its usage.')
        self.tool_preview_status.clear()
        self.tabs.setCurrentIndex(0)
        self._refresh_datasets()

    def _build_view(self):
        self.view_controls = QWidget()
        layout = QVBoxLayout(self.view_controls)
        layout.setContentsMargins(6, 9, 6, 9)
        layout.setSpacing(6)

        def section(title, layout_type):
            if layout.count():
                self._horizontal_separator(layout)
            heading = QLabel(title)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            heading.setObjectName('muted')
            layout.addWidget(heading)
            body = QWidget()
            form = layout_type(body)
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(6)
            if isinstance(form, QFormLayout):
                form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            layout.addWidget(body)
            return form

        camera_layout = section('Camera', QVBoxLayout)
        self.tool_focus = QPushButton('Tool focus OFF')
        self.tool_focus.setObjectName('toolFocus')
        self.tool_focus.setCheckable(True)
        self.tool_focus.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tool_focus.toggled.connect(lambda value: self.tool_focus.setText('Tool focus ON' if value else 'Tool focus OFF'))
        self.tool_focus.toggled.connect(lambda value: self._call(lambda: self.backend.set_tool_focus(value)))
        camera_layout.addWidget(self.tool_focus)
        presets = QHBoxLayout()
        camera_layout.addLayout(presets)
        presets.setSpacing(4)
        for name in ('Isometric', 'Top', 'Front'):
            label = 'Iso' if name == 'Isometric' else name
            button = self._button(label, lambda view=name: self.backend._camera_view(view))
            button.setAccessibleName(name)
            button.setObjectName('cameraPreset')
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(label) + 14)
            presets.addWidget(button, 1)
        form = section('Visibility', QVBoxLayout)
        self.paths = QCheckBox('Show tool paths')
        self.paths.toggled.connect(lambda value:self._call(lambda:self.backend.set_paths_visible(value)))
        form.addWidget(self.paths)
        self.tool_model = QCheckBox('Show tool model')
        self.tool_model.toggled.connect(lambda value:self._call(lambda:self.backend._set_tool_visible(value)))
        form.addWidget(self.tool_model)
        form = section('Surface', QFormLayout)
        self.smoothing = QCheckBox('Opaque Surface')
        self.smoothing.toggled.connect(lambda value: self._call(lambda: self.backend._set_smoothing(value)))
        form.addRow(self.smoothing)
        self.colors = QCheckBox('Color by tool')
        self.colors.toggled.connect(lambda value: self._call(lambda: self.backend._set_tool_colors(value)))
        form.addRow(self.colors)
        self.size = GSSizeSpinBox()
        self.size.valueChanged.connect(lambda value:self._call(lambda:self.backend._set_display_size(value)))
        form.addRow('GS size', self.size)
        form = section('Material', QFormLayout)
        self.ambient = SteppedValue('Ambient', 0., 1., .1, 2)
        self.diffuse = SteppedValue('Diffuse', 0., 1., .1, 2)
        for index, (label, control) in enumerate((('Ambient', self.ambient), ('Diffuse', self.diffuse))):
            control.spin.valueChanged.connect(lambda value, index=index: self._call(lambda: self.backend._lighting_changed(index, value)))
            form.addRow(label, control)
        self.specular = SteppedValue('Specular', 0., 1., .1, 2)
        self.specular.setToolTip('Highlight strength. Buttons change the value by 0.1.')
        self.specular.spin.valueChanged.connect(lambda value: self._call(lambda: self.backend._lighting_changed(2, value)))
        form.addRow('Specular', self.specular)
        self.shininess = SteppedValue('Shininess', 1., 256., 8., 0)
        self.shininess.setToolTip('Higher values make the highlight narrower. Buttons change the value by 8.')
        self.shininess.spin.valueChanged.connect(lambda value: self._call(lambda: self.backend._shininess_changed(value)))
        form.addRow('Shininess', self.shininess)
        material_controls = (self.ambient, self.diffuse, self.specular, self.shininess)
        value_width = max(control.spin.fontMetrics().horizontalAdvance(sample)
                          for control, sample in zip(material_controls, ('1.00', '1.00', '1.00', '256'))) + 14
        for control in material_controls:
            control.spin.setFixedWidth(value_width)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self.view_controls)
        self.view_dock = self._dock('viewDock', 'View', scroll, Qt.DockWidgetArea.RightDockWidgetArea)
        self.view_dock.setMinimumWidth(180)

    def _select_block(self):
        entry = self.blocks.current_instruction()
        if entry is not None:
            if entry.block_index is not None:
                self._call(lambda: self.backend.seek_playback(entry.block_index))
            else:
                self.details.setText(source_details_html(entry))

    def _select_tool(self):
        row = self.tool_table.currentRow()
        if 0 <= row < len(self.tools):
            tool = self.tools[row]
            self.tool_details.setText(tool_details_html(tool, self.program))
            self.tool_details.setToolTip('Used in blocks: ' + ', '.join(str(i + 1) for i in tool.blocks))
            if self.tabs.currentIndex() == 1:
                def show():
                    self._ensure_tool_preview()
                    self._apply_tool_preview_style()
                    self.backend.show_tool_preview(tool, self.tool_centers.isChecked())
                    self.tool_preview_status.setText('Stationary X-Z section · GS at Dexel/surface intersections')
                    self.tool_preview_status.setToolTip(self.backend.tool_preview_status)
                self._call(show)

    def _apply_tool_preview_style(self):
        preview = getattr(self.backend, '_tool_preview', None)
        if preview is not None:
            preview.set_style(opaque=self.tool_opaque.isChecked(), display_size=self.tool_gs_size.value())

    def _ensure_tool_preview(self):
        if self.tool_preview_host is not None:
            return
        from cutsim.ui.native_window import NativeViewportHost
        preview = self.backend.create_tool_preview()
        factory = self._preview_host_factory or NativeViewportHost
        try:
            host = factory(preview.native_title)
        except Exception:
            preview.close()
            self.backend._tool_preview = None
            raise
        self.tool_preview_host = host
        host.widget.setMinimumSize(1, 1)
        host.widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.tool_preview_hint.hide()
        self.tool_preview_layout.addWidget(host.widget)

    def _tool_tab_changed(self, index):
        if index == 0:
            self._call(self.backend.hide_tool_preview)
        elif self.tools:
            if self.tool_table.currentRow() < 0:
                self.tool_table.selectRow(0)
            else:
                self._select_tool()

    def _tools_visibility_changed(self, visible):
        if visible and self.tabs.currentIndex() == 1:
            self._select_tool()
        elif not visible:
            self._call(self.backend.hide_tool_preview)

    def _go_to_tool(self):
        row = self.tool_table.currentRow()
        if 0 <= row < len(self.tools):
            self.tabs.setCurrentIndex(0)
            self.backend.select_block(self.tools[row].blocks[0])

    def _reset(self):
        if QMessageBox.question(self, 'Reset calculation', 'Discard computed results and return to the initial stock?',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.backend._reset()

    def _restore_panels(self):
        for dock, area in ((self.program_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
                           (self.view_dock, Qt.DockWidgetArea.RightDockWidgetArea)):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()

    def _reset_window_sizes(self):
        self.showNormal()
        available = self.screen().availableGeometry()
        self.resize(min(1500, available.width()), min(940, available.height()))
        self._restore_panels()
        self.restoreState(self._default_layout_state)
        self.resizeDocks([self.program_dock, self.view_dock], [370, 195], Qt.Orientation.Horizontal)
        self.blocks.reset_column_widths()
        self.tool_table.resizeColumnsToContents()
        if self._settings is not None:
            self._settings.remove('geometry')
            self._settings.remove('state')
            self._settings.sync()

    def refresh(self):
        b = self.backend
        busy, failed = b._busy, b._failed
        locked = b._closing
        computed = len(b.sim.deltas)
        if not locked:
            project = getattr(b, 'machining_project', None)
            signature = (id(self.program), id(project))
            if signature != self._table_signature:
                self.blocks.set_program(self.program, project)
                self._table_signature = signature
            self.blocks.set_completed(computed)
            self.tool_table.set_completed(computed)
        self._sync_dataset_menu()
        self.tabs.setEnabled(not locked)
        self.view_controls.setEnabled(not locked)
        self.tool_focus.setEnabled(not locked and bool(self.program))
        focusing = bool(getattr(b, '_tool_focus', False))
        set_silently(self.tool_focus, 'setChecked', focusing)
        self.tool_focus.setText('Tool focus ON' if focusing else 'Tool focus OFF')
        for control in (self.smoothing, self.ambient, self.diffuse, self.specular, self.shininess):
            control.setEnabled(not locked)
        # These rebuild CPU endpoints; keep the worker's frozen configuration.
        # Camera, lighting and opacity remain interactive during computation.
        self.size.setEnabled(not busy and not locked)
        self.colors.setEnabled(not busy and not locked)
        self.transport.setEnabled(not locked)
        self.reset.setEnabled(not busy and not b._closing)
        self.cancel.setEnabled(b.cancel_button.enabled)
        self.cancel.setText(b.cancel_button.text)
        applied_density = 100.*getattr(b.sim, 'gs_density', 1.)
        if applied_density != self._applied_density:
            self._applied_density = applied_density
            set_silently(self.density, 'setValue', applied_density)
        density_enabled = not busy and not locked and b.density_input.enabled
        self.density.setEnabled(density_enabled)
        self.density_apply.setEnabled(density_enabled and self.density.value() != applied_density)
        for control, source in ((self.compute_all, b.compute_all_button),
                                (self.pause, b.pause_button),
                                (self.progress, b.workspace_progress)):
            control.setEnabled(source.enabled and not locked)
        self.compute_bar.setValue(round(b.compute_bar.value * 100))
        self.compute_bar.setFormat(f'%p% ({len(b.sim.deltas)}/{len(self.program)})')
        looping = b._playing and getattr(b, '_loop_block', False)
        # Set the final state once: briefly disabling an enabled button cancels
        # a mouse press if a periodic refresh lands before its release.
        self.play.setEnabled(not locked and (b.play_button.enabled or looping))
        for button, active in ((self.play, b._playing and not looping), (self.loop_block, looping)):
            with QSignalBlocker(button):
                button.setChecked(active)
        self.loop_block.setEnabled(not locked and 0 <= b._selected_block < len(b.sim.deltas)
                                   and not looping)
        error = getattr(b, '_native_upload_error', '') or (b.compute_info.text if failed else '')
        self.notice.setText(error)
        self.notice.setVisible(bool(error))
        self.compute_bar.setVisible(not bool(error))
        self.compute_status.setMaximumWidth(400 if error else 190)
        if b._tool_preview is not None and not locked:
            b._tool_preview.sync_settings()
            self.tool_preview_status.setToolTip(b._tool_preview.status)
        self.notice.setToolTip(self.notice.text())
        if locked:
            # b.sim is GUI-owned published history, never the compute engine.
            return
        selected = b._selected_block
        playing = b.render_delta_index if b._playing and 0 <= b.render_delta_index < computed else None
        inspected = getattr(b, '_machining_selection', None)
        self.blocks.follow_playback(selected, playing, inspected_block=inspected)
        entry = self.blocks.current_instruction()
        detail_block = inspected if inspected in self.blocks.block_rows else selected
        self.details.setText(source_details_html(entry) if entry is not None and entry.block_index is None else block_details_html(
            detail_block, self.program[detail_block],
            project=project, display=b.sim)
            if 0 <= detail_block < len(self.program) else 'No program blocks')
        timing_index = entry.block_index if entry is not None else None
        self.timings.setText(block_timing_html(b.sim, timing_index))
        self.progress.setEnabled(0 <= selected < computed)
        total = len(self.program)
        self.block_seek.setEnabled(total > 1)
        if not self.block_seek.isSliderDown():
            with QSignalBlocker(self.block_seek):
                self.block_seek.setRange(1 if total else 0, total)
                self.block_seek.setValue(selected+1 if selected >= 0 else 0)
        target = self.block_seek.sliderPosition() if self.block_seek.isSliderDown() else (selected+1 if selected >= 0 else 0)
        self.previous.setEnabled(not locked and 1 < target <= total)
        self.next.setEnabled(not locked and 1 <= target < total)
        self.previous_ten.setEnabled(self.previous.isEnabled())
        self.next_ten.setEnabled(self.next.isEnabled())
        self.block_number.setText(f'{target}/{total}')
        self.block_seek.set_completed(computed)
        self.progress.set_rail_color(self.block_seek.color_for_block(target))
        pending = bool(total and target > computed)
        for slider in (self.block_seek, self.progress):
            if slider.property('pending') != pending:
                slider.setProperty('pending', pending)
                slider.style().unpolish(slider)
                slider.style().polish(slider)
                slider.update()
        block_progress = b.progress if selected == b.render_delta_index else 0.
        seek_target = getattr(b, '_playback_seek_target', None)
        if seek_target is not None and seek_target[0] == selected:
            block_progress = seek_target[1]
        self.position.setText(f'{block_progress*100:.1f}%')
        for control, setter, value in (
                (self.progress, 'setValue', round(block_progress * 1000)),
                (self.speed, 'setValue', b.speed_slider.double_value),
                (self.size, 'setValue', GS_DISPLAY_SIZES[b.sigma.selected_index]),
                (self.smoothing, 'setChecked', b.opaque_surface.checked),
                (self.colors, 'setChecked', b.tool_colors.checked),
                (self.paths, 'setChecked', b.path_toggle.checked),
                (self.tool_model, 'setChecked', b.show_tool.checked)):
            # Do not snap an actively dragged seek back to its previous frame.
            if control is self.progress and (self.progress.isSliderDown() or b._progress_updates.pending is not None):
                continue
            set_silently(control, setter, value)
        for control, value in ((self.ambient, b._gs_scene.lighting[0]),
                               (self.diffuse, b._gs_scene.lighting[1]),
                               (self.specular, b._gs_scene.lighting[2]),
                               (self.shininess, b._gs_scene.shininess)):
            control.sync_value(float(value))

    def _tick(self):
        if self._disposed or self._in_tick:
            return
        # Qt and Open3D run exclusively on the same main thread. Timer ticks do
        # not depend on mouse focus; the existing clock still targets 30 fps.
        self._in_tick = True
        try:
            alive = self.backend.gui.Application.instance.run_one_tick()
            if alive:
                if self._startup is not None:
                    self._startup.advance()
                    if self._startup.complete:
                        self._startup = None
                now = time.perf_counter()
                if now - self._last_refresh >= .10:
                    self.refresh()
                    self._last_refresh = now
        finally:
            self._in_tick = False
        if not alive:
            self.timer.stop()
            self._disposed = True
            self.close()
            return

    def closeEvent(self, event):
        if self._disposed:
            event.accept()
            return
        if self._in_tick:
            # GLFW's Windows message pump can dispatch Qt close events while
            # run_one_tick is still on the stack. Never tear its engine down
            # recursively from inside that pump.
            QTimer.singleShot(0, self.close)
            event.ignore()
            return
        if not self.backend.can_close():
            self.refresh()
            event.ignore()
            return
        self.timer.stop()
        if self._settings is not None:
            self._settings.setValue('geometry', self.saveGeometry())
            self._settings.setValue('state', self.saveState())
        self._disposed = True
        if self.tool_preview_host is not None:
            self.tool_preview_host.close()
        if self.backend._tool_preview is not None:
            self.backend._tool_preview.close()
        self.viewport_host.close()
        self.backend.request_host_close = None
        self.backend.request_host_playback_toggle = None
        self.backend.window.set_on_close(lambda: True)
        self.backend.window.close()
        # Complete Open3D's queued destruction before QApplication disappears.
        self.backend.gui.Application.instance.run_one_tick()
        event.accept()


class QtMachiningApplication:
    def __init__(self, sim, *, data_root=DEFAULT_DATA_ROOT, persist=True, compute_job=None):
        if sys.platform != 'win32':
            raise RuntimeError('The embedded PySide workspace currently requires Windows. Use --open3d-ui on other platforms.')
        from cutsim.ui.embedded_viewport import EmbeddedViewport
        from cutsim.ui.native_window import NativeViewportHost
        self.app = QApplication.instance() or QApplication(sys.argv[:1])
        self.app.setStyle('Fusion')
        self.backend = EmbeddedViewport(sim, compute_job=compute_job)
        host = None
        try:
            host = NativeViewportHost(self.backend.native_title)
            self.window = QtMachiningWindow(self.backend, host, data_root=data_root, persist=persist)
        except Exception:
            if host is not None:
                host.close()
            if self.backend._tool_preview is not None:
                self.backend._tool_preview.close()
            self.backend.window.set_on_close(lambda: True)
            self.backend.window.close()
            self.backend.gui.Application.instance.run_one_tick()
            raise

    def run(self):
        from cutsim.ui.viewport_startup import ViewportStartup
        self.window._startup = ViewportStartup(self.window, self.backend)
        self.window.show()
        # The Open3D window stays hidden until its HWND is a child of the
        # visible Qt workspace; never flash the standalone Debug window.
        self.backend.window.show(True)
        self.window.timer.start()
        return self.app.exec()
