"""Qt presentation models for Matrix candidate records."""

from PySide6 import QtCore


class CandidateTableModel(QtCore.QAbstractTableModel):
    HEADERS = (
        "Channel",
        "ROCm",
        "Torch",
        "TorchVision",
        "Python",
        "Kind",
        "Evidence",
        "Profile",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        candidate = self._rows[index.row()]
        values = (
            candidate.get("channel") or "unknown",
            candidate.get("rocm_version") or "unknown",
            candidate.get("torch_version") or "unknown",
            candidate.get("torchvision_version") or "unknown",
            candidate.get("python_compatibility") or "unknown",
            candidate.get("candidate_kind") or "unknown",
            candidate.get("resolver_status") or candidate.get("status") or "unknown",
            candidate.get("profile_status") or "unknown",
        )
        return values[index.column()]

    def set_rows(self, rows):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def candidate_at(self, row):
        return self._rows[row]
