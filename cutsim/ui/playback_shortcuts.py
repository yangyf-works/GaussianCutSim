"""Space controls playback rather than activating the focused Qt button."""
from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import QApplication, QWidget, QLineEdit, QTextEdit, QPlainTextEdit


class PlaybackSpaceFilter(QObject):
    def __init__(self, window, callback):
        super().__init__(window)
        self.window, self.callback = window, callback
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        if event.key() != Qt.Key.Key_Space or event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        if not isinstance(watched, QWidget):
            return False
        owner = watched
        # QWidget.isAncestorOf stops at a floating dock's window boundary.
        while owner is not None and owner is not self.window:
            owner = owner.parentWidget()
        if owner is None:
            return False
        if QApplication.activeModalWidget() is not None or QApplication.activePopupWidget() is not None:
            return False
        if isinstance(watched, (QLineEdit, QTextEdit, QPlainTextEdit)) and not watched.isReadOnly():
            return False
        event.accept()
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self.callback()
        return True
