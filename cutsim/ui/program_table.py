"""Instruction list: tool colors, operation separators and playback tracking."""
from PySide6.QtCore import Qt, QSignalBlocker, QEvent, QPoint
from PySide6.QtGui import QColor, QPen, QFont, QFontMetrics, QPolygon
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QStyle, QStyledItemDelegate, QTableWidget, QTableWidgetItem

from cutsim.rendering.gaussian_attributes import machining_tool_colors
from cutsim.ui.program_commands import instruction_rows


def instruction_color(tool_id):
    return QColor.fromRgbF(*machining_tool_colors([tool_id])[0]).darker(260)


def draw_row_outline(painter, rect, index, color='white'):
    table = index.model()
    painter.setPen(QPen(QColor(color), 1))
    painter.drawLine(rect.topLeft(), rect.topRight())
    painter.drawLine(rect.bottomLeft(), rect.bottomRight())
    if index.column() == 0:
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
    if index.column() == table.columnCount() - 1:
        painter.drawLine(rect.topRight(), rect.bottomRight())


def completed_instruction_color(tool_id):
    return instruction_color(tool_id).lighter(165)


class InstructionDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        table = self.parent()
        entry = table.entries[index.row()]
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        playing = index.row() == table.playing_row
        inspecting = index.row() == table.inspection_row
        painter.save()
        color = instruction_color(entry.tool_id)
        if entry.heading:
            color = completed_instruction_color(entry.tool_id) if table.heading_is_completed(index.row()) else color.darker(145)
        elif entry.block_index is None:
            threshold = table.passed_instruction_thresholds.get(index.row())
            if threshold is not None and table.completed_count >= threshold:
                color = completed_instruction_color(entry.tool_id)
            else:
                color = QColor('#303742')
        else:
            color = (completed_instruction_color(entry.tool_id) if entry.block_index < table.completed_count
                     else color.darker(140))
        painter.fillRect(option.rect, color)
        font = option.font
        font.setBold(entry.heading or inspecting)
        painter.setFont(font)
        painter.setPen(QColor('#70efff' if inspecting else '#e5eaf2'))
        rect = option.rect.adjusted(4, 0, -4, 0)
        if index.column() == 0 and not entry.heading:
            rect.adjust(10, 0, 0, 0)
            if playing:
                x, y = option.rect.left() + 4, option.rect.center().y()
                painter.save()
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor('white'))
                painter.drawPolygon(QPolygon([QPoint(x, y - 4), QPoint(x, y + 4), QPoint(x + 6, y)]))
                painter.restore()
        text = str(index.data() or '')
        align = Qt.AlignmentFlag.AlignVCenter | (
            Qt.AlignmentFlag.AlignRight if index.column() >= 2 and not entry.heading
            else Qt.AlignmentFlag.AlignLeft)
        metrics = painter.fontMetrics()
        if index.column() >= 2 and not entry.heading and metrics.horizontalAdvance(text) > rect.width():
            # Remove insignificant decimal zeroes before eliding a coordinate.
            # The stored value and its full-precision tooltip stay unchanged.
            if '.' in text:
                text = text.rstrip('0').rstrip('.')
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(rect, align, text)
        if selected or playing or inspecting:
            draw_row_outline(painter, option.rect, index, '#70efff' if inspecting else 'white')
        painter.restore()


class ProgramTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 7)
        self.entries = []
        self.block_rows = {}
        self.passed_instruction_thresholds = {}
        self._heading_completion = {}
        self.playing_row = -1
        self.inspection_row = -1
        self.completed_count = 0
        self._last_block = None
        self.setHorizontalHeaderLabels(['Line', 'Type', 'X', 'Y', 'Z', 'A', 'B'])
        for column in range(2, 7):
            unit = 'mm' if column < 5 else '°'
            self.horizontalHeaderItem(column).setToolTip(f'End {"XYZAB"[column-2]} ({unit})')
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(28)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setMinimumSectionSize(24)
        self._coordinate_min_width = 24
        self.setItemDelegate(InstructionDelegate(self))

    def set_program(self, program, project=None):
        self.setUpdatesEnabled(False)
        try:
            with QSignalBlocker(self):
                self.clearSpans()
                self.clearContents()
                self.entries = instruction_rows(program, project)
                self.passed_instruction_thresholds.clear()
                # The next completed cut confirms progress past preceding
                # source instructions. Trailing instructions follow the final cut.
                next_cut = len(program) or None
                for row in reversed(range(len(self.entries))):
                    entry = self.entries[row]
                    if entry.block_index is not None:
                        next_cut = entry.block_index + 1
                    elif not entry.heading and next_cut is not None:
                        self.passed_instruction_thresholds[row] = next_cut
                self.block_rows = {}
                self._heading_completion = {}
                heading_row = None
                self.setRowCount(len(self.entries))
                for row, entry in enumerate(self.entries):
                    if entry.heading:
                        heading_row = row
                        self.setSpan(row, 0, 1, self.columnCount())
                    elif entry.block_index is not None:
                        self.block_rows[entry.block_index] = row
                        # Computation completes a prefix of the cutting blocks.
                        # Empty (e.g. rapid-only) groups have no completion threshold.
                        if heading_row is not None:
                            self._heading_completion[heading_row] = max(
                                self._heading_completion.get(heading_row, 0), entry.block_index + 1)
                    for column, text in enumerate(entry.values):
                        item = QTableWidgetItem(text)
                        if entry.heading:
                            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                            item.setToolTip(text)
                        else:
                            detail = f'Line {entry.source_index+1} · {entry.values[1]}'
                            if entry.block_index is not None:
                                detail += f' · Block {entry.block_index+1}'
                            else:
                                detail += ' · Non-cutting instruction'
                            if entry.source.get('label'):
                                detail += '\n' + str(entry.source['label'])
                            if column >= 2:
                                unit = 'mm' if column < 5 else '°'
                                detail += f'\n{"XYZAB"[column-2]}: {text} {unit}'
                            item.setToolTip(detail)
                        self.setItem(row, column, item)
                self.reset_column_widths()
                self._last_block = None
                self.playing_row = -1
                self.inspection_row = -1
                self.completed_count = 0
                self.clearSelection()
                self.setCurrentCell(-1, -1)
        finally:
            self.setUpdatesEnabled(True)

    def reset_column_widths(self):
        # Size from instructions only, excluding spanning tool headings.
        font = QFont(self.font())
        font.setBold(True)  # Reserve room for wider header text.
        metrics = QFontMetrics(font)
        for column in range(2):
            strings = [e.values[column] for e in self.entries if not e.heading]
            header_width = metrics.horizontalAdvance(self.horizontalHeaderItem(column).text()) + 20
            value_width = max([0] + [metrics.horizontalAdvance(s) for s in strings])
            self.setColumnWidth(column, max(60, header_width, value_width + (26 if column == 0 else 20)))
        self._fit_coordinate_columns()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange) and getattr(self, 'entries', None):
            self.reset_column_widths()

    def _fit_coordinate_columns(self):
        available = self.viewport().width() - self.columnWidth(0) - self.columnWidth(1)
        width, remainder = divmod(max(available, 5*self._coordinate_min_width), 5)
        for offset in range(5):
            self.setColumnWidth(offset+2, width + (offset < remainder))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_coordinate_min_width'):
            self._fit_coordinate_columns()

    def current_instruction(self):
        row = self.currentRow()
        return self.entries[row] if 0 <= row < len(self.entries) and not self.entries[row].heading else None

    def set_completed(self, count):
        # The published count is in simulation block order, independent of
        # which source instructions are represented in this table.
        count = max(0, int(count))
        if count != self.completed_count:
            self.completed_count = count
            self.viewport().update()

    def heading_is_completed(self, row):
        threshold = self._heading_completion.get(row)
        return threshold is not None and self.completed_count >= threshold

    def follow_playback(self, selected, playing_block=None, *, inspected_block=None):
        playing_row = self.block_rows.get(playing_block, -1)
        inspection_row = self.block_rows.get(inspected_block, -1)
        inspection_changed = self.inspection_row != inspection_row
        changed = self.playing_row != playing_row or inspection_changed
        self.playing_row = playing_row
        self.inspection_row = inspection_row
        # Inspect a generating operation without seeking the simulation back
        # to that operation (which would remove the picked GS/overlay).
        if inspection_row >= 0:
            selected = inspected_block
        if selected != self._last_block or inspection_changed or changed and playing_row >= 0:
            row = self.block_rows.get(selected, -1)
            with QSignalBlocker(self):
                if row >= 0:
                    self.setCurrentCell(row, 0)
                    self.selectRow(row)
                    self.scrollToItem(self.item(row, 0))
                else:
                    self.clearSelection()
                    self.setCurrentCell(-1, -1)
        self._last_block = selected
        if changed:
            self.viewport().update()
