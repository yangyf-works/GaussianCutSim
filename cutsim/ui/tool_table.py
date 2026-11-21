"""Tool catalogue using the machining list's palette and selection treatment."""
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QStyle, QStyledItemDelegate, QTableWidget, QTableWidgetItem

from cutsim.ui.program_table import instruction_color, draw_row_outline, completed_instruction_color
from cutsim.ui.workspace_model import path_length


class ToolDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        tool_id = index.data(Qt.ItemDataRole.UserRole)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        color = instruction_color(tool_id)
        color = completed_instruction_color(tool_id) if self.parent().row_is_completed(index.row()) else color.darker(140)
        painter.save()
        painter.fillRect(option.rect, color)
        painter.setFont(option.font)
        painter.setPen(QColor('#e5eaf2'))
        rect = option.rect.adjusted(6, 0, -6, 0)
        text = painter.fontMetrics().elidedText(str(index.data()), Qt.TextElideMode.ElideRight, rect.width())
        alignment = Qt.AlignmentFlag.AlignRight if index.column() >= 2 else Qt.AlignmentFlag.AlignLeft
        painter.drawText(rect, alignment | Qt.AlignmentFlag.AlignVCenter, text)
        if selected:
            draw_row_outline(painter, option.rect, index)
        painter.restore()


class ToolTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 6)
        self.completed_count = 0
        self._completion_thresholds = []
        self.setHorizontalHeaderLabels(['Tool', 'Type', 'D\n/ mm', 'Cutting\nrange / mm', 'Cut length\n/ mm', 'Blocks'])
        self.horizontalHeaderItem(3).setToolTip('Cutting range from tool tip (mm)')
        self.horizontalHeaderItem(4).setToolTip('Total cut length across all blocks using this tool (mm)')
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(28)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setItemDelegate(ToolDelegate(self))

    def set_tools(self, tools, program=None):
        with QSignalBlocker(self):
            self.completed_count = 0
            self._completion_thresholds = [max(tool.blocks) + 1 if tool.blocks else None for tool in tools]
            self.clearContents()
            self.setRowCount(len(tools))
            for row, tool in enumerate(tools):
                cutting = tool.cutting_range
                cutting_text = f'{cutting.start:g}–{cutting.end:g}' if cutting is not None else '—'
                length = f'{sum(path_length(program[i]) for i in tool.blocks):.2f}' if program is not None else '—'
                for column, text in enumerate((f'T{tool.tool_id}', tool.kind.title(),
                                               f'{tool.diameter:g}', cutting_text, length, str(len(tool.blocks)))):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, tool.tool_id)
                    if column == 3:
                        item.setToolTip(f'From tool tip: {cutting_text} mm' if cutting is not None else '—')
                    self.setItem(row, column, item)
            self.clearSelection()
            self.setCurrentCell(-1, -1)

    def set_completed(self, count):
        count = max(0, int(count))
        if count != self.completed_count:
            self.completed_count = count
            self.viewport().update()

    def row_is_completed(self, row):
        threshold = self._completion_thresholds[row]
        return threshold is not None and self.completed_count >= threshold
