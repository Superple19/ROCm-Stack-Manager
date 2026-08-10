"""Small Qt thread-pool helpers for non-blocking Manager operations."""

from PySide6 import QtCore


class TaskSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class Task(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.finished.emit(self.function())
        except Exception as error:
            self.signals.failed.emit(f"{type(error).__name__}: {error}")
