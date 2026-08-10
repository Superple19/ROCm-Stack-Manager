"""PySide6 application entry point for ROCm Stack Manager."""

import sys

from PySide6 import QtWidgets

from .main_window import MainWindow


def main(argv=None):
    app = QtWidgets.QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("ROCm Stack Manager")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
