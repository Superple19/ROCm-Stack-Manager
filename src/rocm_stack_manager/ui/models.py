"""Qt presentation models for Matrix candidate records."""

from PySide6 import QtCore


class CandidateTableModel(QtCore.QAbstractTableModel):
    HEADERS = (
        "Version set",
        "GFX",
        "Python",
        "Platform",
        "Kind",
        "Evidence",
        "Resolver",
        "Warnings",
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
        if not index.isValid() or role not in {
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        }:
            return None
        candidate = self._rows[index.row()]
        warnings = list(candidate.get("profile_warnings") or candidate.get("warnings") or ())
        resolver_status = candidate.get("resolver_status") or "not_collected"
        evidence = candidate.get("evidence_level") or candidate.get("evidence_status") or "unknown"
        if isinstance(evidence, dict):
            evidence = evidence.get("overall") or evidence.get("artifact") or "unknown"
        version_set = " / ".join(
            str(candidate.get(key) or "unknown")
            for key in ("rocm_version", "torch_version", "torchvision_version")
        )
        values = (
            version_set,
            candidate.get("gfx") or "unknown",
            candidate.get("python_compatibility") or "unknown",
            candidate.get("platform") or "unknown",
            candidate.get("candidate_kind") or "unknown",
            str(evidence),
            resolver_status,
            "; ".join(str(warning) for warning in warnings) or "none",
        )
        return values[index.column()]

    def set_rows(self, rows):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def candidate_at(self, row):
        return self._rows[row]
