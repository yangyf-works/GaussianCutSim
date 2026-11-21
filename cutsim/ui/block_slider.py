"""Block timeline with tool-run bands and boundary lines."""
from PySide6.QtCore import QEvent, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QToolTip

from cutsim.ui.workspace_model import next_tool_block


class SeekSlider(QSlider):
    """Horizontal transport slider with absolute groove-click positioning."""
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._jump_drag = False
        self._rail_color = None
        self.setMinimumHeight(30)

    def set_rail_color(self, color):
        if color != self._rail_color:
            self._rail_color = QColor(color) if color is not None else None
            self.update()

    def paintEvent(self, event):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        painter = QPainter(self)
        option.subControls = QStyle.SubControl.SC_SliderGroove | QStyle.SubControl.SC_SliderTickmarks
        self.style().drawComplexControl(QStyle.ComplexControl.CC_Slider, option, painter, self)
        self._paint_rail(painter, option)
        # Paint the rounded handle last, over both colors and tool boundaries.
        painter.setClipping(False)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                            QStyle.SubControl.SC_SliderHandle, self)
        handle_outline = QPainterPath()
        handle_outline.addRoundedRect(QRectF(handle), 5., 5.)
        # The stylesheet also paints the widget background for a handle-only
        # draw. Limit that draw to the handle's silhouette to retain the rail.
        painter.setClipPath(handle_outline)
        option.subControls = QStyle.SubControl.SC_SliderHandle
        self.style().drawComplexControl(QStyle.ComplexControl.CC_Slider, option, painter, self)

    def _paint_rail(self, painter, option):
        if self._rail_color is None:
            return
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderGroove, self)
        outline = QPainterPath()
        outline.addRoundedRect(QRectF(groove), 3., 3.)
        painter.fillPath(outline, self._rail_color)

    def _position_at(self, x):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        centers = []
        for value in (self.minimum(), self.maximum()):
            option.sliderPosition = option.sliderValue = value
            handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                                QStyle.SubControl.SC_SliderHandle, self)
            centers.append(handle.center().x())
        low, high = sorted(centers)
        return QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
            round(x) - low, high - low, centers[0] > centers[1])

    def mousePressEvent(self, event):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                            QStyle.SubControl.SC_SliderHandle, self)
        if event.button() == Qt.MouseButton.LeftButton and not handle.contains(event.position().toPoint()):
            # Native groove clicks page by a fixed amount. Seek to the clicked
            # position instead, with the same drag/commit signals as handle drag.
            self._jump_drag = True
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.setSliderDown(True)
            self.setSliderPosition(self._position_at(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._jump_drag:
            self.setSliderPosition(self._position_at(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._jump_drag and event.button() == Qt.MouseButton.LeftButton:
            self.setSliderPosition(self._position_at(event.position().x()))
            self._jump_drag = False
            self.setSliderDown(False)
            self.setValue(self.sliderPosition())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class BlockSlider(SeekSlider):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('blockSeek')
        self._tool_runs = ()
        self._computed = 0

    def set_program(self, program):
        from cutsim.rendering.gaussian_attributes import machining_tool_colors
        # Group consecutive uses, including a return to an earlier tool.
        runs = []
        start = 0
        while start < len(program):
            stop = next_tool_block(program, start)
            if stop is None:
                stop = len(program)
            cmd = program[start]
            color = QColor.fromRgbF(*machining_tool_colors([cmd.tool_id])[0])
            label = f'T{cmd.tool_id} · {cmd.tool_type.title()} · D{2 * cmd.radius:g} mm'
            runs.append((start + 1, stop, color, label))
            start = stop
        self._tool_runs = tuple(runs)
        self._computed = 0
        self.update()

    def set_completed(self, count):
        total = self._tool_runs[-1][1] if self._tool_runs else 0
        count = max(0, min(int(count), total))
        if count != self._computed:
            self._computed = count
            self.update()

    @staticmethod
    def _pending_color(color):
        # Dim pending ranges while completed ranges retain the surface tool color.
        return QColor.fromRgbF(*(c * .40 for c in color.getRgbF()[:3]))

    def color_for_block(self, block):
        color = next((color for start, stop, color, _ in self._tool_runs
                      if start <= block <= stop), None)
        return self._pending_color(color) if color is not None and block > self._computed else color

    def _completed_rect(self, rail):
        if self._computed <= 0:
            return QRect()
        if self._computed >= self._tool_runs[-1][1]:
            return rail
        # Use the same next-block boundary as the tool bands, in either direction.
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        option.sliderPosition = option.sliderValue = self._computed + 1
        edge = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                          QStyle.SubControl.SC_SliderHandle, self).center().x()
        if option.upsideDown:
            return QRect(edge, rail.top(), rail.right() + 1 - edge, rail.height())
        return QRect(rail.left(), rail.top(), edge - rail.left(), rail.height())

    def _bands(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderHandle, self)
        def x_at(block):
            option.sliderPosition = option.sliderValue = block
            return style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                        QStyle.SubControl.SC_SliderHandle, self).center().x()
        bands = []
        for i, (start, stop, color, label) in enumerate(self._tool_runs):
            if stop < self.minimum() or start > self.maximum():
                continue
            left = groove.left() if i == 0 else x_at(start)
            right = groove.right() + 1 if i == len(self._tool_runs) - 1 else x_at(stop + 1)
            # Qt's native handle geometry also handles reversed layouts.
            if option.upsideDown:
                left = groove.right() + 1 if i == 0 else x_at(start)
                right = groove.left() if i == len(self._tool_runs) - 1 else x_at(stop + 1)
            rect = QRect(min(left, right), groove.top(), abs(right - left), groove.height())
            bands.append((rect, color, label, None if i == 0 else left))
        return bands, handle

    def _paint_rail(self, painter, option):
        bands, _ = self._bands()
        if not bands:
            return
        rail = bands[0][0].united(bands[-1][0])
        outline = QPainterPath()
        outline.addRoundedRect(QRectF(rail), 3., 3.)
        painter.setClipPath(outline, Qt.ClipOperation.IntersectClip)
        completed = self._completed_rect(rail)
        for rect, color, _, _ in bands:
            painter.fillRect(rect, self._pending_color(color))
            painter.fillRect(rect.intersected(completed), color)
        # The handle is drawn afterwards, so its curved edge naturally occludes lines.
        painter.setClipping(False)
        painter.setPen(QPen(QColor('#b6c8dc'), 2))
        for rect, _, _, boundary in bands:
            if boundary is not None:
                painter.drawLine(boundary, rect.top() - 7, boundary, rect.bottom() + 7)

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            bands, _ = self._bands()
            pos = event.pos()
            # Boundaries are visual separators, not tooltip targets.
            on_boundary = any(boundary is not None and abs(pos.x() - boundary) <= 2
                              and rect.top() - 5 <= pos.y() <= rect.bottom() + 5
                              for rect, _, _, boundary in bands)
            label = None if on_boundary else next(
                (label for rect, _, label, _ in bands if rect.adjusted(0, -5, 0, 5).contains(pos)), None)
            if label:
                QToolTip.showText(event.globalPos(), label, self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)
