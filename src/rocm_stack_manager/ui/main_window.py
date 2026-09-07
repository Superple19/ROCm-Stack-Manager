"""QtWidgets main window for read-only ROCm candidate and plan inspection."""

import json
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..adapters.registry import available_adapters
from ..core.backup import BackupError, load_backup
from ..core.platforms import SUPPORTED_PLATFORMS, host_platform
from .models import CandidateTableModel
from .services import ManagerService, detected_gfx_targets
from .workers import Task


class WorkspaceStack(QtWidgets.QStackedWidget):
    """Stacked workspace pages with tab-like titles for settings compatibility."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page_titles = []

    def add_page(self, widget, title):
        self._page_titles.append(title)
        return self.addWidget(widget)

    def tabText(self, index):
        return self._page_titles[index]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, service=None, parent=None, settings=None):
        super().__init__(parent)
        self.service = service or ManagerService()
        self.settings = settings
        self._saved_gfx = None
        self._startup_catalog_path = None
        self._startup_autoload_candidates = False
        self.thread_pool = QtCore.QThreadPool(self)
        self._tasks = set()
        self._generation = 0
        self._candidates = []
        self._candidate = None
        self._core_plan = None
        self._extension_plan_result = None
        self._restore_plan = None
        self._restore_path = None
        self._restore_kind = None
        self._restore_network_allowed = False
        self._mutation_in_progress = False
        self._theme_name = "System"
        self._density_name = "Comfortable"
        self._last_target_details = {}
        self._last_inventory_details = {}
        self._last_catalog_details = {}
        self._extension_records = []
        self._activity_records = []
        self._build_ui()
        self._restore_ui_settings()
        if self.settings is not None and self.startup_snapshot_check.isChecked():
            QtCore.QTimer.singleShot(0, self._start_saved_snapshot)

    def closeEvent(self, event):
        if self._mutation_in_progress:
            QtWidgets.QMessageBox.warning(
                self,
                "Operation in progress",
                "Wait for the package operation to finish before closing the manager.",
            )
            event.ignore()
            return
        self._generation += 1
        for task in self._tasks:
            task.signals.blockSignals(True)
        self.thread_pool.clear()
        self.thread_pool.waitForDone(2000)
        self._tasks.clear()
        self._save_ui_settings()
        super().closeEvent(event)

    def _restore_ui_settings(self):
        if self.settings is None:
            return

        geometry = self.settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("window/state")
        if state:
            self.restoreState(state)

        adapter = self.settings.value("configuration/adapter")
        if adapter in available_adapters():
            self.adapter_combo.blockSignals(True)
            self.adapter_combo.setCurrentText(str(adapter))
            self.adapter_combo.blockSignals(False)
            if self.service.adapter_name != adapter:
                self.service.set_adapter(str(adapter))
        legacy_restore_paths = self.settings.value("startup/restore_paths", True, type=bool)
        self.restore_target_path_check.setChecked(
            self.settings.value("startup/restore_target_path", legacy_restore_paths, type=bool)
        )
        self.restore_catalog_path_check.setChecked(
            self.settings.value("startup/restore_catalog_path", legacy_restore_paths, type=bool)
        )
        self.startup_snapshot_check.setChecked(
            self.settings.value("startup/read_only_snapshot", True, type=bool)
        )
        self.local_catalog_check.setChecked(
            self.settings.value("startup/local_catalog", True, type=bool)
        )
        self.never_refresh_network_check.setChecked(
            self.settings.value("startup/never_refresh_network", True, type=bool)
        )
        if self.restore_target_path_check.isChecked():
            self.target_edit.setText(str(self.settings.value("configuration/target_path", "")))
        if self.restore_catalog_path_check.isChecked():
            self.catalog_edit.setText(str(self.settings.value("configuration/catalog_path", "")))

        self._restore_combo_value(self.platform_combo, "configuration/platform")
        self._saved_gfx = str(self.settings.value("configuration/gfx", "")).strip() or None
        if self._saved_gfx:
            self.gfx_edit.setEditText(self._saved_gfx)
        self._restore_combo_value(self.channel_combo, "configuration/channel")
        self.rocm_edit.setText(str(self.settings.value("configuration/rocm", "")))
        self._restore_combo_value(self.family_combo, "configuration/family")
        self._restore_combo_value(self.lifecycle_combo, "configuration/lifecycle")
        self._restore_combo_value(self.kind_combo, "configuration/candidate_kind")
        self._restore_combo_value(self.theme_combo, "appearance/theme")
        self._restore_combo_value(self.density_combo, "appearance/density")
        self.sidebar_labels_check.setChecked(
            self.settings.value("appearance/sidebar_labels", True, type=bool)
        )
        self.activity_retention_spin.setValue(
            self.settings.value("advanced/activity_retention", 100, type=int)
        )
        self.remember_candidate_filters_check.setChecked(
            self.settings.value("advanced/remember_candidate_filters", True, type=bool)
        )
        self.show_unknown_evidence_check.setChecked(
            self.settings.value("advanced/show_unknown_evidence", True, type=bool)
        )
        self.developer_diagnostics_check.setChecked(
            self.settings.value("advanced/developer_diagnostics", False, type=bool)
        )
        self._theme_name = self.theme_combo.currentText()
        self._density_name = self.density_combo.currentText()
        self._toggle_sidebar_labels(self.sidebar_labels_check.isChecked())
        self._apply_visual_theme()

        tab_index = self.settings.value("window/tab", 0, type=int)
        if 0 <= tab_index < self.tabs.count():
            self.tabs.setCurrentIndex(tab_index)

    def _save_ui_settings(self):
        if self.settings is None:
            return

        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/tab", self.tabs.currentIndex())
        self.settings.setValue("configuration/adapter", self.adapter_combo.currentText())
        self.settings.setValue("configuration/target_path", self.target_edit.text().strip())
        self.settings.setValue("configuration/catalog_path", self.catalog_edit.text().strip())
        restore_target = self.restore_target_path_check.isChecked()
        restore_catalog = self.restore_catalog_path_check.isChecked()
        self.settings.setValue("startup/restore_paths", restore_target and restore_catalog)
        self.settings.setValue("startup/restore_target_path", restore_target)
        self.settings.setValue("startup/restore_catalog_path", restore_catalog)
        self.settings.setValue("startup/read_only_snapshot", self.startup_snapshot_check.isChecked())
        self.settings.setValue("startup/local_catalog", self.local_catalog_check.isChecked())
        self.settings.setValue("startup/never_refresh_network", self.never_refresh_network_check.isChecked())
        if self.remember_candidate_filters_check.isChecked():
            self.settings.setValue("configuration/platform", self.platform_combo.currentText())
            self.settings.setValue("configuration/gfx", self.gfx_edit.currentText().strip())
            self.settings.setValue("configuration/channel", self.channel_combo.currentText())
            self.settings.setValue("configuration/rocm", self.rocm_edit.text().strip())
            self.settings.setValue("configuration/family", self.family_combo.currentText())
            self.settings.setValue("configuration/lifecycle", self.lifecycle_combo.currentText())
            self.settings.setValue("configuration/candidate_kind", self.kind_combo.currentText())
        else:
            for key in (
                "configuration/platform",
                "configuration/gfx",
                "configuration/channel",
                "configuration/rocm",
                "configuration/family",
                "configuration/lifecycle",
                "configuration/candidate_kind",
            ):
                self.settings.remove(key)
        self.settings.setValue("appearance/theme", self.theme_combo.currentText())
        self.settings.setValue("appearance/density", self.density_combo.currentText())
        self.settings.setValue("appearance/sidebar_labels", self.sidebar_labels_check.isChecked())
        self.settings.setValue("advanced/activity_retention", self.activity_retention_spin.value())
        self.settings.setValue(
            "advanced/remember_candidate_filters",
            self.remember_candidate_filters_check.isChecked(),
        )
        self.settings.setValue("advanced/show_unknown_evidence", self.show_unknown_evidence_check.isChecked())
        self.settings.setValue("advanced/developer_diagnostics", self.developer_diagnostics_check.isChecked())
        self.settings.sync()

    def _restore_combo_value(self, combo, key):
        if self.settings is None:
            return
        value = self.settings.value(key)
        if value is not None and combo.findText(str(value)) >= 0:
            combo.setCurrentText(str(value))

    def _build_ui(self):
        self.setWindowTitle("ROCm Stack Manager")
        self.resize(1280, 800)

        central = QtWidgets.QWidget(self)
        central.setObjectName("mainSurface")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        global_bar = QtWidgets.QFrame()
        global_bar.setObjectName("globalBar")
        global_layout = QtWidgets.QVBoxLayout(global_bar)
        global_layout.setContentsMargins(14, 12, 14, 12)
        global_layout.setSpacing(10)
        global_top = QtWidgets.QHBoxLayout()
        global_top.setSpacing(10)
        global_title = QtWidgets.QLabel("ROCm Stack Manager")
        global_title.setObjectName("globalTitle")
        global_top.addWidget(global_title)
        global_top.addStretch(1)
        self.target_header_status = QtWidgets.QLabel("Not detected")
        self.target_header_status.setObjectName("statusChip")
        self.catalog_header_status = QtWidgets.QLabel("Catalog not loaded")
        self.catalog_header_status.setObjectName("statusChip")
        global_top.addWidget(self.target_header_status)
        global_top.addWidget(self.catalog_header_status)
        global_layout.addLayout(global_top)

        target_bar = QtWidgets.QHBoxLayout()
        target_bar.setSpacing(8)
        adapter_label = QtWidgets.QLabel("Adapter")
        adapter_label.setObjectName("fieldLabel")
        self.adapter_combo = QtWidgets.QComboBox()
        self.adapter_combo.addItems(available_adapters())
        self.adapter_combo.setCurrentText(self.service.adapter_name)
        self.adapter_combo.currentTextChanged.connect(self._adapter_changed)
        self.adapter_combo.setMinimumWidth(132)
        self.target_edit = QtWidgets.QLineEdit()
        self.target_edit.setPlaceholderText("Select a ComfyUI portable, venv, or source directory")
        self.target_browse_button = QtWidgets.QPushButton("Browse…")
        self.detect_button = QtWidgets.QPushButton("Detect")
        self.detect_button.setObjectName("primaryButton")
        self.target_browse_button.clicked.connect(self._browse_target)
        self.detect_button.clicked.connect(self._detect)
        target_bar.addWidget(adapter_label)
        target_bar.addWidget(self.adapter_combo)
        path_label = QtWidgets.QLabel("Target")
        path_label.setObjectName("fieldLabel")
        target_bar.addWidget(path_label)
        target_bar.addWidget(self.target_edit, 1)
        target_bar.addWidget(self.target_browse_button)
        target_bar.addWidget(self.detect_button)
        global_layout.addLayout(target_bar)
        root.addWidget(global_bar)

        workspace = QtWidgets.QHBoxLayout()
        workspace.setSpacing(12)
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setMinimumWidth(168)
        self.sidebar.setMaximumWidth(208)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 10, 8, 10)
        sidebar_layout.setSpacing(4)
        self.sidebar_label_widget = QtWidgets.QLabel("WORKSPACE")
        self.sidebar_label_widget.setObjectName("sidebarLabel")
        sidebar_layout.addWidget(self.sidebar_label_widget)
        self.navigation_buttons = []
        for index, label in enumerate(("Overview", "Candidates", "Extensions", "Activity", "Settings")):
            button = QtWidgets.QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.clicked.connect(lambda _checked, page=index: self._select_page(page))
            self.navigation_buttons.append(button)
            sidebar_layout.addWidget(button)
        sidebar_layout.addStretch(1)
        workspace.addWidget(self.sidebar)

        self.tabs = WorkspaceStack()
        self.tabs.currentChanged.connect(self._page_changed)
        workspace.addWidget(self.tabs, 1)
        root.addLayout(workspace, 1)

        def page_header(eyebrow_text, title_text, subtitle_text):
            header = QtWidgets.QWidget()
            header_layout = QtWidgets.QVBoxLayout(header)
            header_layout.setContentsMargins(2, 0, 2, 4)
            header_layout.setSpacing(3)
            eyebrow = QtWidgets.QLabel(eyebrow_text)
            eyebrow.setObjectName("eyebrow")
            title = QtWidgets.QLabel(title_text)
            title.setObjectName("pageTitle")
            subtitle = QtWidgets.QLabel(subtitle_text)
            subtitle.setObjectName("pageSubtitle")
            subtitle.setWordWrap(True)
            header_layout.addWidget(eyebrow)
            header_layout.addWidget(title)
            header_layout.addWidget(subtitle)
            return header

        def status_card(title, label, details_button):
            card = QtWidgets.QFrame()
            card.setObjectName("summaryCard")
            layout = QtWidgets.QVBoxLayout(card)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(7)
            heading = QtWidgets.QLabel(title)
            heading.setObjectName("cardHeading")
            layout.addWidget(heading)
            label.setObjectName("summaryStatus")
            self._configure_status_label(label)
            label.setWordWrap(True)
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            details_button.setProperty("compact", True)
            layout.addWidget(label)
            layout.addWidget(details_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            return card

        overview_page = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout(overview_page)
        overview_layout.setContentsMargins(4, 12, 4, 4)
        overview_layout.setSpacing(16)
        overview_layout.addWidget(
            page_header(
                "OVERVIEW",
                "Manage a target environment",
                "Inspect evidence, choose a candidate, and keep every change explicit.",
            )
        )

        self.target_summary = QtWidgets.QLabel("No target detected")
        self.snapshot_summary = QtWidgets.QLabel(
            "Inventory not loaded • Verify not run"
        )
        self.snapshot_metric_values = {}
        snapshot_card = QtWidgets.QFrame()
        snapshot_card.setObjectName("summaryCard")
        snapshot_card.setMinimumWidth(0)
        snapshot_card.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        snapshot_layout = QtWidgets.QVBoxLayout(snapshot_card)
        snapshot_layout.setContentsMargins(14, 12, 14, 12)
        snapshot_layout.setSpacing(8)
        snapshot_heading = QtWidgets.QLabel("Package snapshot")
        snapshot_heading.setObjectName("cardHeading")
        snapshot_layout.addWidget(snapshot_heading)
        snapshot_grid = QtWidgets.QGridLayout()
        snapshot_grid.setContentsMargins(0, 0, 0, 0)
        snapshot_grid.setHorizontalSpacing(8)
        snapshot_grid.setVerticalSpacing(8)
        for index, (label_text, package_name) in enumerate(
            (
                ("ROCm", "rocm"),
                ("Torch", "torch"),
                ("TorchVision", "torchvision"),
                ("TorchAudio", "torchaudio"),
            )
        ):
            metric = QtWidgets.QFrame()
            metric.setObjectName("snapshotMetric")
            metric.setMinimumWidth(0)
            metric.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            metric_layout = QtWidgets.QVBoxLayout(metric)
            metric_layout.setContentsMargins(9, 6, 9, 6)
            metric_layout.setSpacing(2)
            metric_label = QtWidgets.QLabel(label_text)
            metric_label.setObjectName("snapshotMetricLabel")
            metric_value = QtWidgets.QLabel("—")
            metric_value.setObjectName("snapshotMetricValue")
            metric_value.setWordWrap(True)
            metric_value.setMinimumWidth(0)
            metric_value.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            metric_value.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            metric_value.setToolTip("—")
            metric_layout.addWidget(metric_label)
            metric_layout.addWidget(metric_value, 1)
            snapshot_grid.addWidget(metric, index // 2, index % 2)
            self.snapshot_metric_values[package_name] = metric_value
        snapshot_grid.setColumnStretch(0, 1)
        snapshot_grid.setColumnStretch(1, 1)
        snapshot_layout.addLayout(snapshot_grid)
        self._configure_status_label(self.snapshot_summary)
        self.snapshot_summary.setObjectName("snapshotMeta")
        snapshot_layout.addWidget(self.snapshot_summary)
        self.target_details_button = QtWidgets.QPushButton("View details")
        self.target_details_button.setEnabled(False)
        self.target_details_button.clicked.connect(
            lambda: self._show_details_dialog("Target details", self._last_target_details)
        )
        self.package_details_button = QtWidgets.QPushButton("View details")
        self.package_details_button.setEnabled(False)
        self.package_details_button.clicked.connect(
            lambda: self._show_details_dialog("Package inventory", self._last_inventory_details)
        )
        self.catalog_summary = QtWidgets.QLabel("No catalog loaded")
        self.catalog_details_button = QtWidgets.QPushButton("View details")
        self.catalog_details_button.setEnabled(False)
        self.catalog_details_button.clicked.connect(
            lambda: self._show_details_dialog("Catalog details", self._last_catalog_details)
        )
        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(12)
        status_row.addWidget(status_card("Target", self.target_summary, self.target_details_button), 1)
        snapshot_layout.addWidget(self.package_details_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.package_details_button.setProperty("compact", True)
        status_row.addWidget(snapshot_card, 1)
        status_row.addWidget(status_card("Matrix catalog", self.catalog_summary, self.catalog_details_button), 1)
        overview_layout.addLayout(status_row)

        catalog_card = QtWidgets.QFrame()
        catalog_card.setObjectName("surfacePanel")
        catalog_layout = QtWidgets.QVBoxLayout(catalog_card)
        catalog_layout.setContentsMargins(14, 12, 14, 12)
        catalog_layout.setSpacing(8)
        catalog_heading = QtWidgets.QLabel("Catalog source")
        catalog_heading.setObjectName("sectionHeading")
        catalog_layout.addWidget(catalog_heading)
        catalog_row = QtWidgets.QHBoxLayout()
        catalog_row.setSpacing(8)
        self.catalog_edit = QtWidgets.QLineEdit()
        self.catalog_edit.setPlaceholderText("Optional local catalog.json or matrix.json")
        self.catalog_browse_button = QtWidgets.QPushButton("Browse…")
        self.load_catalog_button = QtWidgets.QPushButton("Load")
        self.refresh_catalog_button = QtWidgets.QPushButton("Refresh official")
        self.refresh_catalog_button.setObjectName("primaryButton")
        self.catalog_browse_button.clicked.connect(self._browse_catalog)
        self.load_catalog_button.clicked.connect(lambda: self._load_catalog(False))
        self.refresh_catalog_button.clicked.connect(lambda: self._load_catalog(True))
        catalog_row.addWidget(self.catalog_edit, 1)
        catalog_row.addWidget(self.catalog_browse_button)
        catalog_row.addWidget(self.load_catalog_button)
        catalog_row.addWidget(self.refresh_catalog_button)
        catalog_layout.addLayout(catalog_row)
        overview_layout.addWidget(catalog_card)

        quick_actions = QtWidgets.QFrame()
        quick_actions.setObjectName("actionBar")
        quick_layout = QtWidgets.QHBoxLayout(quick_actions)
        quick_layout.setContentsMargins(12, 10, 12, 10)
        quick_layout.setSpacing(8)
        quick_label = QtWidgets.QLabel("Next action")
        quick_label.setObjectName("mutedLabel")
        quick_layout.addWidget(quick_label)
        self.overview_detect_button = QtWidgets.QPushButton("Detect target")
        self.overview_detect_button.clicked.connect(self._detect)
        quick_layout.addWidget(self.overview_detect_button)
        self.overview_load_catalog_button = QtWidgets.QPushButton("Load catalog")
        self.overview_load_catalog_button.clicked.connect(lambda: self._load_catalog(False))
        quick_layout.addWidget(self.overview_load_catalog_button)
        self.overview_find_button = QtWidgets.QPushButton("Find candidates")
        self.overview_find_button.setObjectName("primaryButton")
        self.overview_find_button.clicked.connect(self._find_candidates)
        quick_layout.addWidget(self.overview_find_button)
        self.overview_verify_button = QtWidgets.QPushButton("Verify target")
        self.overview_verify_button.clicked.connect(self._verify)
        self.overview_verify_button.setEnabled(False)
        self.overview_verify_button.setVisible(False)
        quick_layout.addWidget(self.overview_verify_button)
        quick_layout.addStretch(1)
        overview_layout.addWidget(quick_actions)
        overview_layout.addStretch(1)
        self.tabs.add_page(overview_page, "Overview")

        candidates_page = QtWidgets.QWidget()
        candidates_layout = QtWidgets.QVBoxLayout(candidates_page)
        candidates_layout.setContentsMargins(4, 12, 4, 4)
        candidates_layout.setSpacing(12)
        candidates_layout.addWidget(
            page_header("CANDIDATES", "Choose a package set", "Filter evidence, inspect provenance, then build a dry-run.")
        )
        candidate_toolbar = QtWidgets.QHBoxLayout()
        self.filters_button = QtWidgets.QPushButton("Hide filters")
        self.filters_button.clicked.connect(self._toggle_filter_rail)
        candidate_toolbar.addWidget(self.filters_button)
        candidate_toolbar.addStretch(1)
        candidates_layout.addLayout(candidate_toolbar)
        candidate_workspace = QtWidgets.QHBoxLayout()
        candidate_workspace.setSpacing(10)
        self.filter_rail = QtWidgets.QScrollArea()
        self.filter_rail.setObjectName("filterRail")
        self.filter_rail.setWidgetResizable(True)
        self.filter_rail.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.filter_rail.setMinimumWidth(210)
        self.filter_rail.setMaximumWidth(260)
        filter_content = QtWidgets.QFrame()
        filter_content.setObjectName("filterRailContent")
        filter_layout = QtWidgets.QFormLayout(filter_content)
        filter_layout.setContentsMargins(12, 12, 12, 12)
        filter_layout.setHorizontalSpacing(8)
        filter_layout.setVerticalSpacing(8)
        filter_heading = QtWidgets.QLabel("Filters")
        filter_heading.setObjectName("sectionHeading")
        filter_layout.addRow(filter_heading)
        self.platform_combo = QtWidgets.QComboBox()
        self.platform_combo.addItems(("windows", "linux"))
        detected_platform = host_platform()
        if detected_platform in SUPPORTED_PLATFORMS:
            self.platform_combo.setCurrentText(detected_platform)
        else:
            self.platform_combo.setCurrentIndex(-1)
            self.platform_combo.setPlaceholderText(
                f"Unsupported host platform: {detected_platform}"
            )
            self.platform_combo.setEnabled(False)
        self.gfx_edit = QtWidgets.QComboBox()
        self.gfx_edit.setEditable(True)
        self._set_gfx_placeholder("Detected GFX or enter manually")
        self.gfx_status = QtWidgets.QLabel("No GFX detected")
        self._configure_status_label(self.gfx_status)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(("all", "stable", "nightly", "staging"))
        self.rocm_edit = QtWidgets.QLineEdit()
        self.rocm_edit.setPlaceholderText("Optional exact ROCm version")
        self.family_combo = QtWidgets.QComboBox()
        self.family_combo.addItems(("all", "therock", "legacy"))
        self.lifecycle_combo = QtWidgets.QComboBox()
        self.lifecycle_combo.addItems(("all", "current", "historical"))
        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.addItems(("all", "installable", "artifact_only", "unavailable"))
        self.candidate_summary = QtWidgets.QLabel("No candidates loaded")
        self._configure_status_label(self.candidate_summary)
        self.find_button = QtWidgets.QPushButton("Find candidates")
        self.find_button.setObjectName("primaryButton")
        self.find_button.clicked.connect(self._find_candidates)
        filter_layout.addRow("Platform", self.platform_combo)
        filter_layout.addRow("GFX", self.gfx_edit)
        filter_layout.addRow(self.gfx_status)
        filter_layout.addRow("Channel", self.channel_combo)
        filter_layout.addRow("ROCm", self.rocm_edit)
        filter_layout.addRow("Family", self.family_combo)
        filter_layout.addRow("Lifecycle", self.lifecycle_combo)
        filter_layout.addRow("State", self.kind_combo)
        filter_layout.addRow(self.find_button)
        filter_layout.addRow(self.candidate_summary)
        self.filter_rail.setWidget(filter_content)
        candidate_workspace.addWidget(self.filter_rail)

        candidate_center = QtWidgets.QVBoxLayout()
        candidate_center.setSpacing(8)
        self.candidate_model = CandidateTableModel(self)
        self.candidate_view = QtWidgets.QTableView()
        self.candidate_view.setModel(self.candidate_model)
        self.candidate_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_view.setSortingEnabled(False)
        self.candidate_view.selectionModel().selectionChanged.connect(self._candidate_selected)
        self.candidate_view.horizontalHeader().setStretchLastSection(True)
        self.candidate_view.setMinimumHeight(140)
        self.candidate_view.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        candidate_center.addWidget(self.candidate_view, 1)
        candidate_workspace.addLayout(candidate_center, 1)

        self.candidate_details = QtWidgets.QPlainTextEdit()
        self.candidate_details.setObjectName("detailsOutput")
        self.candidate_details.setReadOnly(True)
        self.candidate_details.setPlaceholderText("Select a candidate to inspect identity and provenance")
        self.candidate_details.setMinimumWidth(220)
        self.candidate_details.setMaximumWidth(280)
        candidate_workspace.addWidget(self.candidate_details)
        candidates_layout.addLayout(candidate_workspace, 1)

        candidate_actions = QtWidgets.QFrame()
        candidate_actions.setObjectName("actionBar")
        candidate_actions_layout = QtWidgets.QHBoxLayout(candidate_actions)
        candidate_actions_layout.setContentsMargins(12, 10, 12, 10)
        candidate_actions_layout.setSpacing(8)
        self.verify_button = QtWidgets.QPushButton("Verify")
        self.verify_button.setToolTip("Verify target")
        self.inventory_button = QtWidgets.QPushButton("Inventory")
        self.plan_button = QtWidgets.QPushButton("Install dry-run")
        self.plan_button.setObjectName("primaryButton")
        self.apply_core_button = QtWidgets.QPushButton("Apply core")
        self.apply_core_button.setObjectName("warningButton")
        self.allow_unverified = QtWidgets.QCheckBox("Allow unverified")
        self.allow_unverified.setToolTip("Allow unverified evidence")
        for button in (
            self.verify_button,
            self.inventory_button,
            self.plan_button,
            self.apply_core_button,
        ):
            button.setProperty("compact", True)
            button.setEnabled(False)
            candidate_actions_layout.addWidget(button)
        self.allow_unverified.setProperty("compact", True)
        candidate_actions_layout.addWidget(self.allow_unverified)
        candidate_actions_layout.addStretch(1)
        candidates_layout.addWidget(candidate_actions)
        self.tabs.add_page(candidates_page, "Candidates")

        extensions_page = QtWidgets.QWidget()
        extensions_layout = QtWidgets.QVBoxLayout(extensions_page)
        extensions_layout.setContentsMargins(4, 12, 4, 4)
        extensions_layout.setSpacing(12)
        extensions_layout.addWidget(
            page_header("EXTENSIONS", "Extension inventory", "Review installed extensions separately from core packages.")
        )
        extension_panel = QtWidgets.QFrame()
        extension_panel.setObjectName("surfacePanel")
        extension_layout = QtWidgets.QVBoxLayout(extension_panel)
        extension_layout.setContentsMargins(12, 12, 12, 12)
        extension_layout.setSpacing(8)
        self.extension_summary = QtWidgets.QLabel("No target extension inventory")
        self._configure_status_label(self.extension_summary)
        self.extension_table = QtWidgets.QTableWidget(0, 5)
        self.extension_table.setHorizontalHeaderLabels(
            ("Extension", "Status", "Installed", "Target match", "Evidence")
        )
        self.extension_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.extension_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.extension_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.extension_table.itemSelectionChanged.connect(self._extension_selected)
        self.extension_table.horizontalHeader().setStretchLastSection(True)
        self.extension_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.extension_table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self.extension_table.setWordWrap(False)
        self.extension_table.setMinimumHeight(120)
        extension_layout.addWidget(self.extension_summary)
        extension_workspace = QtWidgets.QHBoxLayout()
        extension_workspace.setSpacing(10)
        extension_workspace.addWidget(self.extension_table, 1)
        self.extension_details = QtWidgets.QPlainTextEdit()
        self.extension_details.setObjectName("detailsOutput")
        self.extension_details.setReadOnly(True)
        self.extension_details.setPlaceholderText("Select an extension to inspect evidence and reason")
        self.extension_details.setMinimumWidth(230)
        self.extension_details.setMaximumWidth(300)
        extension_workspace.addWidget(self.extension_details)
        extension_layout.addLayout(extension_workspace, 1)
        extensions_layout.addWidget(extension_panel, 1)

        extension_actions = QtWidgets.QFrame()
        extension_actions.setObjectName("actionBar")
        extension_actions_layout = QtWidgets.QHBoxLayout(extension_actions)
        extension_actions_layout.setContentsMargins(14, 12, 14, 12)
        extension_actions_layout.setSpacing(8)
        self.refresh_extension_button = QtWidgets.QPushButton("Refresh inventory")
        self.refresh_extension_button.clicked.connect(self._refresh_extensions)
        self.extension_button = QtWidgets.QPushButton("Extension plan")
        self.extension_button.setObjectName("primaryButton")
        self.apply_extension_button = QtWidgets.QPushButton("Apply extension")
        self.apply_extension_button.setObjectName("warningButton")
        self.extension_button.setEnabled(False)
        self.apply_extension_button.setEnabled(False)
        extension_actions_layout.addWidget(self.refresh_extension_button)
        extension_actions_layout.addWidget(self.extension_button)
        extension_actions_layout.addWidget(self.apply_extension_button)
        extension_actions_layout.addStretch(1)
        extensions_layout.addWidget(extension_actions)
        self.tabs.add_page(extensions_page, "Extensions")

        activity_page = QtWidgets.QWidget()
        activity_layout = QtWidgets.QVBoxLayout(activity_page)
        activity_layout.setContentsMargins(4, 12, 4, 4)
        activity_layout.setSpacing(12)
        activity_layout.addWidget(
            page_header("ACTIVITY", "Operations and restore", "Review summaries first; open details only when needed.")
        )
        restore_group = QtWidgets.QFrame()
        restore_group.setObjectName("actionBar")
        restore_layout = QtWidgets.QHBoxLayout(restore_group)
        restore_layout.setContentsMargins(14, 12, 14, 12)
        restore_layout.setSpacing(8)
        self.restore_button = QtWidgets.QPushButton("Restore dry-run…")
        self.restore_button.setObjectName("primaryButton")
        self.apply_restore_button = QtWidgets.QPushButton("Apply restore")
        self.apply_restore_button.setObjectName("warningButton")
        self.restore_button.setEnabled(False)
        self.apply_restore_button.setEnabled(False)
        restore_layout.addWidget(self.restore_button)
        restore_layout.addWidget(self.apply_restore_button)
        restore_layout.addStretch(1)
        activity_layout.addWidget(restore_group)
        activity_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.activity_list = QtWidgets.QListWidget()
        self.activity_list.setObjectName("activityList")
        self.activity_list.currentItemChanged.connect(self._activity_selected)
        activity_splitter.addWidget(self.activity_list)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setObjectName("activityOutput")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Select an operation to view details")
        activity_splitter.addWidget(self.output)
        activity_splitter.setStretchFactor(0, 1)
        activity_splitter.setStretchFactor(1, 2)
        activity_layout.addWidget(activity_splitter, 1)
        activity_actions = QtWidgets.QHBoxLayout()
        self.copy_activity_button = QtWidgets.QPushButton("Copy details")
        self.copy_activity_button.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(self.output.toPlainText())
        )
        self.clear_activity_button = QtWidgets.QPushButton("Clear")
        self.clear_activity_button.clicked.connect(self._clear_activity)
        activity_actions.addWidget(self.copy_activity_button)
        activity_actions.addWidget(self.clear_activity_button)
        activity_actions.addStretch(1)
        activity_layout.addLayout(activity_actions)
        self.tabs.add_page(activity_page, "Activity")

        settings_page = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(4, 12, 4, 4)
        settings_layout.setSpacing(12)
        settings_layout.addWidget(
            page_header(
                "SETTINGS",
                "Preferences",
                "Configure startup, appearance, and diagnostics without restoring plans or applying changes.",
            )
        )
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setObjectName("settingsScroll")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        settings_content = QtWidgets.QWidget()
        settings_content_layout = QtWidgets.QVBoxLayout(settings_content)
        settings_content_layout.setContentsMargins(0, 0, 4, 0)
        settings_content_layout.setSpacing(12)

        startup_panel = QtWidgets.QFrame()
        startup_panel.setObjectName("surfacePanel")
        startup_layout = QtWidgets.QVBoxLayout(startup_panel)
        startup_layout.setContentsMargins(14, 12, 14, 12)
        startup_layout.setSpacing(8)
        startup_heading = QtWidgets.QLabel("Startup")
        startup_heading.setObjectName("sectionHeading")
        startup_layout.addWidget(startup_heading)
        self.restore_target_path_check = QtWidgets.QCheckBox("Restore last target path")
        self.restore_target_path_check.setChecked(True)
        self.restore_catalog_path_check = QtWidgets.QCheckBox("Restore last catalog path")
        self.restore_catalog_path_check.setChecked(True)
        self.startup_snapshot_check = QtWidgets.QCheckBox("Load a read-only target snapshot on startup")
        self.startup_snapshot_check.setChecked(True)
        self.local_catalog_check = QtWidgets.QCheckBox("Load local catalog automatically")
        self.local_catalog_check.setChecked(True)
        self.never_refresh_network_check = QtWidgets.QCheckBox("Never refresh the network automatically")
        self.never_refresh_network_check.setChecked(True)
        for check in (
            self.restore_target_path_check,
            self.restore_catalog_path_check,
            self.startup_snapshot_check,
            self.local_catalog_check,
            self.never_refresh_network_check,
        ):
            startup_layout.addWidget(check)
        settings_content_layout.addWidget(startup_panel)

        appearance_panel = QtWidgets.QFrame()
        appearance_panel.setObjectName("surfacePanel")
        appearance_layout = QtWidgets.QFormLayout(appearance_panel)
        appearance_layout.setContentsMargins(14, 12, 14, 12)
        appearance_layout.setHorizontalSpacing(12)
        appearance_layout.setVerticalSpacing(8)
        appearance_heading = QtWidgets.QLabel("Appearance")
        appearance_heading.setObjectName("sectionHeading")
        appearance_layout.addRow(appearance_heading)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(("System", "Dark", "Light"))
        self.theme_combo.currentTextChanged.connect(self._appearance_changed)
        self.density_combo = QtWidgets.QComboBox()
        self.density_combo.addItems(("Comfortable", "Compact"))
        self.density_combo.currentTextChanged.connect(self._appearance_changed)
        self.sidebar_labels_check = QtWidgets.QCheckBox("Show sidebar labels")
        self.sidebar_labels_check.setChecked(True)
        self.sidebar_labels_check.toggled.connect(self._toggle_sidebar_labels)
        appearance_layout.addRow("Theme", self.theme_combo)
        appearance_layout.addRow("Density", self.density_combo)
        appearance_layout.addRow(self.sidebar_labels_check)
        settings_content_layout.addWidget(appearance_panel)

        advanced_panel = QtWidgets.QFrame()
        advanced_panel.setObjectName("surfacePanel")
        advanced_layout = QtWidgets.QFormLayout(advanced_panel)
        advanced_layout.setContentsMargins(14, 12, 14, 12)
        advanced_layout.setHorizontalSpacing(12)
        advanced_layout.setVerticalSpacing(8)
        advanced_heading = QtWidgets.QLabel("Advanced")
        advanced_heading.setObjectName("sectionHeading")
        advanced_layout.addRow(advanced_heading)
        self.activity_retention_spin = QtWidgets.QSpinBox()
        self.activity_retention_spin.setRange(10, 1000)
        self.activity_retention_spin.setValue(100)
        self.remember_candidate_filters_check = QtWidgets.QCheckBox("Remember default candidate filters")
        self.remember_candidate_filters_check.setChecked(True)
        self.show_unknown_evidence_check = QtWidgets.QCheckBox("Show unknown evidence")
        self.show_unknown_evidence_check.setChecked(True)
        self.developer_diagnostics_check = QtWidgets.QCheckBox("Developer diagnostics")
        self.developer_diagnostics_check.setChecked(False)
        advanced_layout.addRow("Activity retention", self.activity_retention_spin)
        advanced_layout.addRow(self.remember_candidate_filters_check)
        advanced_layout.addRow(self.show_unknown_evidence_check)
        advanced_layout.addRow(self.developer_diagnostics_check)
        settings_content_layout.addWidget(advanced_panel)
        settings_content_layout.addStretch(1)
        settings_scroll.setWidget(settings_content)
        settings_layout.addWidget(settings_scroll, 1)
        self.tabs.add_page(settings_page, "Settings")

        self.restore_button.setEnabled(False)
        self.verify_button.clicked.connect(self._verify)
        self.inventory_button.clicked.connect(self._inventory)
        self.plan_button.clicked.connect(self._plan)
        self.extension_button.clicked.connect(self._extension_plan)
        self.apply_core_button.clicked.connect(self._apply_core)
        self.apply_extension_button.clicked.connect(self._apply_extensions)
        self.restore_button.clicked.connect(self._restore_dry_run)
        self.apply_restore_button.clicked.connect(self._apply_restore)
        self.statusBar().showMessage("Ready")
        self._select_page(0)
        self._apply_visual_theme()

    def _select_page(self, index):
        if 0 <= index < self.tabs.count():
            self.tabs.setCurrentIndex(index)

    def _page_changed(self, index):
        for page_index, button in enumerate(self.navigation_buttons):
            button.setChecked(page_index == index)

    def _toggle_filter_rail(self):
        visible = not self.filter_rail.isVisible()
        self.filter_rail.setVisible(visible)
        self.filters_button.setText("Hide filters" if visible else "Show filters")

    def _toggle_sidebar_labels(self, visible):
        if hasattr(self, "sidebar_label_widget"):
            self.sidebar_label_widget.setVisible(bool(visible))

    def _appearance_changed(self):
        if hasattr(self, "theme_combo"):
            self._theme_name = self.theme_combo.currentText()
            self._density_name = self.density_combo.currentText()
            self._apply_visual_theme()

    def _update_header_status(self, *, target=None, catalog=None):
        if target is not None:
            self.target_header_status.setText(target)
            self.target_header_status.setProperty("state", target.casefold().replace(" ", "-"))
            self.target_header_status.style().unpolish(self.target_header_status)
            self.target_header_status.style().polish(self.target_header_status)
        if catalog is not None:
            self.catalog_header_status.setText(catalog)
            catalog_state = "catalog-loading" if "loading" in catalog.casefold() else (
                "catalog-ready" if "artifacts" in catalog.casefold() else "catalog-idle"
            )
            self.catalog_header_status.setProperty("state", catalog_state)
            self.catalog_header_status.style().unpolish(self.catalog_header_status)
            self.catalog_header_status.style().polish(self.catalog_header_status)

    def _mark_operation_error(self, label):
        normalized = label.casefold()
        if "catalog" in normalized:
            self._update_header_status(catalog="Catalog error")
        elif any(token in normalized for token in ("target", "detect", "snapshot")):
            self._update_header_status(target="Target error")

    def _apply_visual_theme(self):
        stylesheet = """
            QMainWindow, QWidget#mainSurface {
                background: #101318;
                color: #e6eaf0;
                font-size: 13px;
            }
            QFrame#globalBar {
                background: #171c24;
                border: 1px solid #283241;
                border-radius: 10px;
            }
            QLabel#globalTitle {
                color: #f4f7fb;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#fieldLabel, QLabel#mutedLabel {
                color: #8e99aa;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#statusChip {
                background: #202936;
                border: 1px solid #354052;
                border-radius: 12px;
                color: #b9c7d9;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#statusChip[state="target-ready"], QLabel#statusChip[state="catalog-ready"] {
                background: #17382b;
                border-color: #2e7654;
                color: #b4f0cf;
            }
            QLabel#statusChip[state="target-error"], QLabel#statusChip[state="catalog-error"] {
                background: #43232a;
                border-color: #9b4c5a;
                color: #ffd1d8;
            }
            QLabel#statusChip[state="detecting"], QLabel#statusChip[state="catalog-loading"] {
                background: #3a321c;
                border-color: #806d32;
                color: #f1dda0;
            }
            QFrame#sidebar {
                background: #141a22;
                border: 1px solid #283241;
                border-radius: 10px;
            }
            QLabel#sidebarLabel {
                color: #6fa7ff;
                font-size: 10px;
                font-weight: 700;
                padding: 6px 8px 10px 8px;
            }
            QPushButton#navButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #9aa6b8;
                min-height: 36px;
                padding: 0 12px;
                text-align: left;
            }
            QPushButton#navButton:hover {
                background: #1d2632;
                color: #e6eaf0;
            }
            QPushButton#navButton:checked {
                background: #243d61;
                border-color: #3f72b6;
                color: #ffffff;
            }
            QFrame#summaryCard, QFrame#surfacePanel {
                background: #171c24;
                border: 1px solid #283241;
                border-radius: 8px;
            }
            QFrame#snapshotMetric {
                background: #121820;
                border: 1px solid #263143;
                border-radius: 6px;
            }
            QFrame#actionBar {
                background: #141a22;
                border: 1px solid #263143;
                border-radius: 8px;
            }
            QScrollArea#filterRail {
                background: #171c24;
                border: 1px solid #283241;
                border-radius: 8px;
            }
            QFrame#filterRailContent {
                background: transparent;
            }
            QLabel#cardHeading, QLabel#sectionHeading {
                color: #c7d1df;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#snapshotMetricLabel {
                color: #8798ad;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#snapshotMetricValue {
                color: #eef4fb;
                font-size: 12px;
                font-weight: 600;
            }
            QPlainTextEdit#detailsOutput {
                background: #11161e;
                border: 1px solid #2a3545;
                border-radius: 8px;
                color: #b9c7d9;
                padding: 10px;
            }
            QTabWidget::pane {
                background: #101318;
                border: none;
                top: -1px;
            }
            QTabBar {
                qproperty-drawBase: 0;
            }
            QTabBar::tab {
                background: transparent;
                color: #8e99aa;
                padding: 11px 18px 10px 18px;
                margin-right: 4px;
                border-bottom: 2px solid transparent;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                color: #dce4f0;
                background: #171c25;
            }
            QTabBar::tab:selected {
                color: #f4f7fb;
                border-bottom: 2px solid #5b9cff;
            }
            QGroupBox {
                background: transparent;
                border: none;
                margin-top: 8px;
                padding-top: 12px;
                font-weight: 600;
                color: #b7c1d1;
            }
            QGroupBox#cardGroup {
                background: #171c24;
                border: 1px solid #283241;
                border-radius: 10px;
            }
            QGroupBox#actionGroup {
                background: #141a22;
                border: 1px solid #263143;
                border-radius: 9px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                background: #171c24;
                padding: 0 7px;
                color: #aeb9ca;
            }
            QLabel#eyebrow {
                color: #6fa7ff;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#pageTitle {
                color: #f4f7fb;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#pageSubtitle {
                color: #8792a4;
                font-size: 13px;
            }
            QLabel#targetStatus, QLabel#catalogStatus, QLabel#summaryStatus,
            QLabel#snapshotMeta {
                background: #121820;
                border: 1px solid #263143;
                border-radius: 6px;
                color: #aeb9ca;
                padding: 8px 10px;
            }
            QLineEdit, QComboBox {
                background: #10151c;
                border: 1px solid #354052;
                border-radius: 7px;
                color: #e6eaf0;
                min-height: 34px;
                padding: 0 10px;
                selection-background-color: #3f7fdb;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #5b9cff;
            }
            QLineEdit:disabled, QComboBox:disabled {
                background: #151a21;
                color: #667285;
            }
            QComboBox::drop-down {
                border: none;
                width: 26px;
            }
            QPushButton {
                background: #222a36;
                border: 1px solid #3a4658;
                border-radius: 7px;
                color: #e6eaf0;
                min-height: 34px;
                padding: 0 14px;
                font-weight: 600;
            }
            QPushButton[compact="true"] {
                font-size: 11px;
                padding: 0 8px;
            }
            QCheckBox[compact="true"] {
                font-size: 11px;
            }
            QPushButton:hover {
                background: #2b3748;
                border-color: #61728b;
            }
            QPushButton:pressed {
                background: #1b2430;
            }
            QPushButton#primaryButton {
                background: #2463b5;
                border-color: #4d8dde;
                color: #ffffff;
            }
            QPushButton#primaryButton:hover {
                background: #2d73ce;
            }
            QPushButton#warningButton {
                background: #5b3528;
                border-color: #a66446;
                color: #ffe9de;
            }
            QPushButton#warningButton:hover {
                background: #744331;
            }
            QPushButton:disabled, QCheckBox:disabled {
                background: #191f28;
                border-color: #293342;
                color: #647084;
            }
            QCheckBox {
                color: #b6c1d1;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #4a586d;
                border-radius: 4px;
                background: #10151c;
            }
            QCheckBox::indicator:checked {
                background: #3f7fdb;
                border-color: #6fa7ff;
            }
            QTableView, QTableWidget {
                background: #11161e;
                alternate-background-color: #151b24;
                border: 1px solid #2a3545;
                border-radius: 8px;
                color: #dce4ef;
                gridline-color: #25303e;
                selection-background-color: #244c7d;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #1b2430;
                border: none;
                border-bottom: 1px solid #303d4f;
                color: #aebbd0;
                padding: 9px 10px;
                font-weight: 600;
            }
            QPlainTextEdit#activityOutput {
                background: #0d1117;
                border: 1px solid #283241;
                border-radius: 8px;
                color: #b9c7d9;
                padding: 10px;
                selection-background-color: #244c7d;
            }
            QStatusBar {
                background: #0c0f14;
                border-top: 1px solid #202936;
                color: #8e99aa;
            }
            QScrollBar:vertical {
                background: #11161e;
                width: 12px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #3a4658;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
            """
        if self._theme_name.casefold() == "light":
            stylesheet += """
            QMainWindow, QWidget#mainSurface { background: #f4f7fb; color: #1f2937; }
            QFrame#globalBar, QFrame#summaryCard, QFrame#surfacePanel,
            QScrollArea#filterRail { background: #ffffff; border-color: #d6dee9; }
            QFrame#snapshotMetric { background: #f5f8fb; border-color: #d6dee9; }
            QFrame#sidebar, QFrame#actionBar { background: #edf2f7; border-color: #d6dee9; }
            QLineEdit, QComboBox, QPlainTextEdit#detailsOutput,
            QPlainTextEdit#activityOutput, QTableView, QTableWidget {
                background: #ffffff; border-color: #c7d1df; color: #1f2937;
            }
            QHeaderView::section { background: #e8eef5; color: #334155; }
            QLabel#globalTitle, QLabel#pageTitle { color: #172033; }
            QLabel#snapshotMetricLabel { color: #64748b; }
            QLabel#snapshotMetricValue { color: #172033; }
            QLabel#pageSubtitle, QLabel#fieldLabel, QLabel#mutedLabel { color: #526174; }
            QPushButton { background: #e9eef5; border-color: #bec9d8; color: #1f2937; }
            """
        if self._density_name.casefold() == "compact":
            stylesheet += """
            QLineEdit, QComboBox, QPushButton { min-height: 30px; }
            QFrame#actionBar { padding: 2px; }
            """
        self.setStyleSheet(stylesheet)

    @staticmethod
    def _configure_status_label(label):
        label.setWordWrap(False)
        label.setMinimumWidth(0)
        label.setMinimumHeight(34)
        label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    @staticmethod
    def _set_status_label(label, text, details=None):
        label.setText(text)
        # Keep hover text short; full records are available through View details
        # or the selected Activity entry instead of a tooltip.
        label.setToolTip(text)

    @staticmethod
    def _compact_version(version):
        """Keep overview metrics compact while retaining the full value in a tooltip."""

        text = str(version)
        if len(text) <= 18:
            return text
        prefix, separator, accelerator = text.partition("+rocm")
        if separator:
            suffix = accelerator[:4] + ("…" if len(accelerator) > 4 else "")
            return f"{prefix}+rocm{suffix}"
        return f"{text[:17]}…"

    def _show_details_dialog(self, title, value):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        details = QtWidgets.QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText(json.dumps(value or {}, indent=2, sort_keys=True, default=str))
        layout.addWidget(details, 1)
        actions = QtWidgets.QHBoxLayout()
        copy_button = QtWidgets.QPushButton("Copy")
        copy_button.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(details.toPlainText())
        )
        save_button = QtWidgets.QPushButton("Save…")

        def save_details():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog,
                "Save details",
                "details.json",
                "JSON files (*.json);;Text files (*.txt)",
            )
            if path:
                try:
                    Path(path).write_text(details.toPlainText(), encoding="utf-8")
                except OSError as error:
                    self._show_error(f"Could not save details: {error}")

        save_button.clicked.connect(save_details)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        actions.addWidget(copy_button)
        actions.addWidget(save_button)
        actions.addStretch(1)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        dialog.exec()

    def _start_saved_snapshot(self):
        """Start only the read-only work justified by persisted UI preferences."""

        if self._generation != 0 or self._mutation_in_progress:
            return
        target_path = self.target_edit.text().strip()
        self._startup_catalog_path = self.catalog_edit.text().strip() or None
        self._startup_autoload_candidates = bool(target_path)
        if not target_path:
            self._continue_saved_catalog_load()
            return

        resolved_target = Path(target_path).expanduser()
        if not resolved_target.exists():
            self._update_header_status(target="Target unavailable")
            self._set_status_label(
                self.target_summary,
                "Saved target unavailable",
                str(resolved_target),
            )
            self._set_status_label(
                self.snapshot_summary,
                "Inventory not loaded • Verify not run",
            )
            self.statusBar().showMessage("Saved target path is unavailable")
            self._continue_saved_catalog_load()
            return

        self._bump_generation()
        self._run(
            "Read saved target snapshot",
            lambda: self.service.prepare_target_snapshot(resolved_target),
            self._commit_saved_snapshot,
            error_callback=self._continue_saved_catalog_load,
        )

    def _continue_saved_catalog_load(self):
        if not self.local_catalog_check.isChecked():
            self._startup_catalog_path = None
            self._startup_autoload_candidates = False
            return
        catalog_path = self._startup_catalog_path
        self._startup_catalog_path = None
        if not catalog_path:
            self._startup_autoload_candidates = False
            return
        resolved_catalog = Path(catalog_path).expanduser()
        if not resolved_catalog.is_file():
            self._update_header_status(catalog="Catalog unavailable")
            self._set_status_label(
                self.catalog_summary,
                "Saved local catalog unavailable",
                str(resolved_catalog),
            )
            self.statusBar().showMessage("Saved catalog path is unavailable")
            self._startup_autoload_candidates = False
            return
        self._load_catalog(False, startup=True)

    def _commit_saved_snapshot(self, result):
        target, inventory, extension_report = result
        self.service.commit_target(target)
        self._display_target(target)
        self._display_package_inventory(inventory)
        if extension_report is not None:
            self._display_extension_inventory(extension_report)
        self.statusBar().showMessage("Saved target snapshot loaded")
        self._continue_saved_catalog_load()

    def _display_package_inventory(self, inventory):
        values = inventory.as_dict()
        self._last_inventory_details = values
        self.package_details_button.setEnabled(True)
        core = {}
        extensions = []
        for package in values.get("packages", ()):
            name = str(package.get("name") or "unknown")
            normalized = name.casefold().replace("_", "-")
            version = str(package.get("version") or "unknown")
            if (
                normalized in {"torch", "torchvision", "torchaudio", "rocm"}
                or normalized.startswith(("rocm-sdk", "amd-torch-device", "amd-torchvision-device"))
            ):
                core[normalized] = version
            elif package.get("compiled_files"):
                extensions.append(f"{name}=={version}")
        status = values.get("status") or "unknown"
        unknown_count = sum(
            1 for package in values.get("packages", ()) if package.get("status") == "unknown"
        )
        core_text = ", ".join(
            f"{name}=={version}" for name, version in sorted(core.items())
        ) or "none detected"
        for package_name, metric_value in self.snapshot_metric_values.items():
            version = core.get(package_name, "—")
            metric_value.setText(self._compact_version(version))
            metric_value.setToolTip(version)
        tooltip = "\n".join(
            (
                f"Inventory: {status}",
                f"Core packages: {core_text}",
                f"Compiled/extension packages: {len(extensions)}",
                f"Unknown packages: {unknown_count}",
                "Runtime verification: not run",
                "Hardware/tensor: not run",
            )
        )
        self._set_status_label(
            self.snapshot_summary,
            f"{len(extensions)} extensions • {status} • Verify not run",
            tooltip,
        )
        self._append_json(
            {
                "target_package_inventory": {
                    "status": status,
                    "target_root": values.get("target_root"),
                    "python_executable": values.get("python_executable"),
                    "core_packages": dict(sorted(core.items())),
                    "compiled_extension_count": len(extensions),
                    "unknown_package_count": unknown_count,
                }
            }
        )

    def _browse_target(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select ComfyUI target")
        if path:
            self.target_edit.setText(path)

    def _adapter_changed(self, adapter_name):
        self._startup_catalog_path = None
        self._startup_autoload_candidates = False
        self._bump_generation()
        self.service.set_adapter(adapter_name)
        self._reset_target_state()
        self.extension_summary.setText("Extension inventory is unavailable for this adapter")
        self.extension_table.setRowCount(0)
        self._extension_records = []
        self.extension_details.clear()

    def _browse_catalog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Matrix catalog",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            self.catalog_edit.setText(path)

    def _bump_generation(self):
        self._generation += 1
        return self._generation

    def _run(self, label, function, callback, *, mutation=False, error_callback=None):
        generation = None if mutation else self._generation
        if mutation:
            self._set_mutation_locked(True)
        self.statusBar().showMessage(f"{label}…")
        task = Task(function)
        self._tasks.add(task)
        task.signals.finished.connect(
            lambda result: self._finish_task(
                task, label, callback, result, generation, mutation
            )
        )
        task.signals.failed.connect(
            lambda error: self._fail_task(
                task, label, error, generation, mutation, error_callback
            )
        )
        self.thread_pool.start(task)

    def _finish_task(
        self, task, label, callback, result, generation=None, mutation=False
    ):
        self._tasks.discard(task)
        if generation is not None and generation != self._generation:
            self.statusBar().showMessage(f"{label} result discarded (target changed)")
            return
        try:
            failed = self._result_failed(result)
            self.statusBar().showMessage(f"{label} {'failed' if failed else 'complete'}")
            if failed:
                self._mark_operation_error(label)
            callback(result)
        finally:
            if mutation:
                self._set_mutation_locked(False)

    @staticmethod
    def _result_failed(result):
        returncode = getattr(result, "returncode", None)
        if getattr(result, "applied", False) and returncode != 0:
            return True
        runtime_status = getattr(result, "runtime_status", None)
        hardware_status = getattr(result, "hardware_status", None)
        tensor_status = getattr(result, "tensor_smoke_status", None)
        if runtime_status is not None and (
            runtime_status != "detected" or hardware_status != "detected" or tensor_status == "failed"
        ):
            return True
        if isinstance(result, dict):
            if result.get("verification_level") == "unknown":
                return True
            if any(
                extension.get("status") in {"runtime_failed", "hardware_failed"}
                for extension in result.get("extensions", ())
            ):
                return True
        return False

    def _fail_task(
        self,
        task,
        label,
        error,
        generation=None,
        mutation=False,
        error_callback=None,
    ):
        self._tasks.discard(task)
        if generation is not None and generation != self._generation:
            self.statusBar().showMessage(f"{label} error discarded (target changed)")
            return
        try:
            self.statusBar().showMessage(f"{label} failed")
            self._mark_operation_error(label)
            self._show_error(error)
            if error_callback is not None:
                error_callback()
        finally:
            if mutation:
                self._set_mutation_locked(False)

    def _set_mutation_locked(self, locked):
        self._mutation_in_progress = locked
        for widget in (
            self.adapter_combo,
            self.target_edit,
            self.target_browse_button,
            self.detect_button,
            self.catalog_edit,
            self.catalog_browse_button,
            self.load_catalog_button,
            self.refresh_catalog_button,
            self.find_button,
            self.overview_find_button,
            self.overview_detect_button,
            self.overview_load_catalog_button,
            self.overview_verify_button,
            self.refresh_extension_button,
            self.restore_button,
            self.restore_target_path_check,
            self.restore_catalog_path_check,
            self.startup_snapshot_check,
            self.local_catalog_check,
            self.never_refresh_network_check,
        ):
            widget.setEnabled(not locked)
        if locked:
            self.apply_core_button.setEnabled(False)
            self.apply_extension_button.setEnabled(False)
            self.apply_restore_button.setEnabled(False)
        else:
            self.verify_button.setEnabled(self.service.target is not None)
            self.restore_button.setEnabled(self.service.target is not None)
            self._update_candidate_actions()

    def _detect(self):
        self._startup_catalog_path = None
        self._startup_autoload_candidates = False
        self._bump_generation()
        path = self.target_edit.text().strip() or "."
        self._update_header_status(target="Detecting")
        self.service.clear_target()
        self._reset_target_state("Detecting target…")
        self._run(
            "Detect",
            lambda: self.service.detect(path),
            self._commit_and_display_target,
        )

    def _commit_and_display_target(self, target):
        self.service.commit_target(target)
        self._display_target_and_probe(target)

    def _display_target_and_probe(self, target):
        self._display_target(target)
        self._run(
            "Read target package snapshot",
            lambda: self.service.prepare_detected_target_snapshot(target),
            self._display_detected_snapshot,
        )
        self._run("Inspect runtime and hardware", self.service.inspect, self._display_inspection)

    def _display_detected_snapshot(self, result):
        inventory, extension_report = result
        self._display_package_inventory(inventory)
        if extension_report is not None:
            self._display_extension_inventory(extension_report)

    def _display_target(self, target):
        self._candidates = []
        self._candidate = None
        self._clear_plan_state()
        self.candidate_model.set_rows(())
        self.candidate_summary.setText("No candidates loaded")
        values = target.as_dict()
        self._last_target_details = values
        self.target_details_button.setEnabled(True)
        python_state = "Python found" if values.get("python_executable") else "Python not found"
        comfyui_state = "ComfyUI found" if values.get("comfyui_dir") else "ComfyUI not found"
        self._set_status_label(
            self.target_summary,
            f"Detected • {values.get('layout', 'unknown')} • {python_state} • {comfyui_state}",
            json.dumps(values, indent=2, sort_keys=True, default=str),
        )
        self._update_header_status(target="Target ready")
        for metric_value in self.snapshot_metric_values.values():
            metric_value.setText("Loading…")
            metric_value.setToolTip("Inventory loading…")
        self._set_status_label(
            self.snapshot_summary,
            "Inventory loading… • Verify available",
        )
        self.verify_button.setEnabled(True)
        self.restore_button.setEnabled(True)
        self._update_candidate_actions()

    def _reset_target_state(self, summary="No target detected"):
        self._candidates = []
        self._candidate = None
        self._clear_plan_state()
        self._restore_plan = None
        self._restore_path = None
        self._restore_kind = None
        self._restore_network_allowed = False
        self._last_target_details = {}
        self._last_inventory_details = {}
        self.target_details_button.setEnabled(False)
        self.package_details_button.setEnabled(False)
        for metric_value in self.snapshot_metric_values.values():
            metric_value.setText("—")
            metric_value.setToolTip("—")
        self.apply_restore_button.setEnabled(False)
        self.candidate_model.set_rows(())
        self.candidate_summary.setText("No candidates loaded")
        self._set_status_label(self.target_summary, summary)
        self._set_status_label(self.snapshot_summary, "Inventory not loaded • Verify not run")
        self._update_header_status(target="Not detected")
        self.verify_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        self._update_candidate_actions()

    def _probe_host_hardware(self):
        self._run("Probe host hardware", self.service.host_hardware, self._display_host_hardware)

    def _set_gfx_observation(self, observation, label):
        values = observation.as_dict()
        detected = detected_gfx_targets(observation)
        self.gfx_edit.blockSignals(True)
        self.gfx_edit.clear()
        self.gfx_edit.addItems(detected)
        if len(detected) == 1:
            self.gfx_edit.setCurrentIndex(0)
            self.gfx_status.setText(f"{label}: {detected[0]}")
        elif detected:
            self.gfx_edit.setCurrentIndex(-1)
            self._set_gfx_placeholder("Select a detected GFX target")
            self.gfx_status.setText(f"{label}: {len(detected)} GFX targets")
        else:
            self.gfx_edit.setCurrentIndex(-1)
            self._set_gfx_placeholder("Detected GFX unavailable; enter manually")
            self.gfx_status.setText(f"{label}: no GFX; manual entry allowed")
        self.gfx_edit.blockSignals(False)
        if label == "Host detected" and self._saved_gfx:
            self.gfx_edit.setEditText(self._saved_gfx)
        host_platform = values.get("host_platform")
        if host_platform in {self.platform_combo.itemText(index) for index in range(self.platform_combo.count())}:
            self.platform_combo.setCurrentText(host_platform)

    def _set_gfx_placeholder(self, text):
        line_edit = self.gfx_edit.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText(text)

    def _display_host_hardware(self, observation):
        self._set_gfx_observation(observation, "Host detected")
        self._append_json({"host_hardware_probe": observation.as_dict()})

    def _display_runtime(self, observation, *, append=True):
        values = observation.as_dict()
        detected = detected_gfx_targets(observation)
        self._set_gfx_observation(observation, "Target runtime")
        if append:
            self._append_json({"runtime_probe": values, "detected_gfx": list(detected)})

    def _display_inspection(self, result):
        runtime, hardware, runtime_error = result
        if runtime is not None:
            runtime_values = runtime.as_dict()
            self._display_runtime(runtime, append=False)
        else:
            runtime_values = {"status": "unavailable", "error": runtime_error}
            self._set_gfx_observation(hardware, "Target hardware")
        hardware_values = hardware.as_dict()
        self._append_json(
            {
                "runtime_probe": runtime_values,
                "hardware_probe": hardware_values,
                "runtime_error": runtime_error,
            }
        )

    def _load_catalog(self, refresh, *, startup=False):
        self._bump_generation()
        self.service.clear_catalog()
        self._last_catalog_details = {}
        self.catalog_details_button.setEnabled(False)
        self._candidates = []
        self._candidate = None
        self._clear_plan_state()
        self.candidate_model.set_rows(())
        self.candidate_summary.setText("No candidates loaded")
        self._set_status_label(self.catalog_summary, "Loading Matrix catalog…")
        self._update_header_status(catalog="Catalog loading")
        path = None if refresh else (self.catalog_edit.text().strip() or None)
        self._run(
            "Load Matrix catalog",
            lambda: self.service.load_catalog(path, refresh=refresh),
            lambda result: self._commit_and_display_catalog(result, startup=startup),
        )

    def _commit_and_display_catalog(self, result, *, startup=False):
        self.service.commit_catalog(result)
        self._display_catalog(result, startup=startup)

    def _display_catalog(self, result, *, startup=False):
        _, state = result
        fetched = state.fetched_at or "unknown"
        artifacts = state.artifact_count if state.artifact_count is not None else "unknown"
        age = "unknown"
        if state.cache_age_seconds is not None:
            age = f"{state.cache_age_seconds / 86400:.1f} days"
        failures = (
            state.source_failure_count
            if state.source_failure_count is not None
            else "unknown"
        )
        failure_ids = ", ".join(state.source_failures) or "none"
        hash_short = f"{state.catalog_sha256[:12]}…" if state.catalog_sha256 else "missing"
        self._last_catalog_details = {
            "path": str(state.path),
            "source": state.source,
            "fetched_at": state.fetched_at,
            "catalog_sha256": state.catalog_sha256,
            "artifact_count": state.artifact_count,
            "cache_age_seconds": state.cache_age_seconds,
            "source_failure_count": state.source_failure_count,
            "source_failures": list(state.source_failures),
            "timestamp_source": state.timestamp_source,
            "refreshed": state.refreshed,
        }
        self.catalog_details_button.setEnabled(True)
        self._set_status_label(
            self.catalog_summary,
            f"{state.source} • {artifacts} artifacts • {failures} source failures • "
            f"SHA-256 {hash_short}",
            "\n".join(
                (
                    f"Path: {state.path}",
                    f"Source: {state.source}",
                    f"Fetched: {fetched} ({age} old)",
                    f"Artifacts: {artifacts}",
                    f"Source failures: {failures} ({failure_ids})",
                    f"Refresh requested: {'yes' if state.refreshed else 'no'}",
                    f"SHA-256: {state.catalog_sha256 or 'missing'}",
                )
            ),
        )
        self._update_header_status(catalog=f"Catalog • {artifacts} artifacts")
        self._append_json(
            {
                "catalog": str(state.path),
                "source": state.source,
                "fetched_at": state.fetched_at,
                "catalog_sha256": state.catalog_sha256,
                "artifact_count": state.artifact_count,
                "cache_age_seconds": state.cache_age_seconds,
                "source_failure_count": state.source_failure_count,
                "source_failures": list(state.source_failures),
                "timestamp_source": state.timestamp_source,
            }
        )
        self._update_candidate_actions()
        if self.service.target is not None and getattr(self.service, "extension_capable", lambda: False)():
            callback = self._display_extension_inventory
            if startup:
                callback = self._display_startup_extensions_and_candidates
            self._run(
                "Refresh ComfyUI extensions",
                self.service.extension_inventory,
                callback,
            )
        elif startup:
            self._maybe_start_saved_candidates()

    def _display_startup_extensions_and_candidates(self, report):
        self._display_extension_inventory(report)
        self._maybe_start_saved_candidates()

    def _maybe_start_saved_candidates(self):
        if not self._startup_autoload_candidates:
            return
        self._startup_autoload_candidates = False
        if self.service.target is None or self.service.catalog is None:
            return
        if not self.gfx_edit.currentText().strip():
            return
        self._find_candidates()

    def _find_candidates(self):
        self._bump_generation()
        platform = self.platform_combo.currentText()
        if platform not in SUPPORTED_PLATFORMS:
            self._show_error(
                "unsupported host platform; ROCm Stack Manager supports Windows and Linux only"
            )
            return
        if self.service.catalog is None:
            self._show_error("CatalogError: load a Matrix catalog first")
            return
        gfx = self.gfx_edit.currentText().strip()
        if not gfx:
            self._show_error("ValueError: enter a GFX target")
            return
        channel = self.channel_combo.currentText()
        channel = None if channel == "all" else channel
        rocm_version = self.rocm_edit.text().strip() or None
        family = self.family_combo.currentText()
        family = None if family == "all" else family
        lifecycle = self.lifecycle_combo.currentText()
        lifecycle = None if lifecycle == "all" else lifecycle
        candidate_kind = self.kind_combo.currentText()
        candidate_kind = None if candidate_kind == "all" else candidate_kind
        self._run(
            "Find candidates",
            lambda: self.service.candidates(
                platform=platform,
                gfx=gfx,
                channel=channel,
                rocm_version=rocm_version,
                include_unavailable=True,
                include_incompatible=True,
                distribution_family=family,
                lifecycle=lifecycle,
                candidate_kind=candidate_kind,
            ),
            self._display_candidates,
        )

    def _display_candidates(self, candidates):
        self._candidates = list(candidates)
        self._candidate = None
        self._clear_plan_state()
        self.candidate_details.clear()
        self.candidate_model.set_rows(self._candidates)
        current = sum(item.get("lifecycle") == "current" for item in self._candidates)
        historical = sum(item.get("lifecycle") == "historical" for item in self._candidates)
        installable = sum(item.get("candidate_kind") == "installable" for item in self._candidates)
        artifact_only = sum(item.get("candidate_kind") == "artifact_only" for item in self._candidates)
        self.candidate_summary.setText(
            f"{len(self._candidates)} total | current {current} | historical {historical} | "
            f"installable {installable} | artifact-only {artifact_only}"
        )
        self._update_candidate_actions()
        self._select_page(1)
        self.statusBar().showMessage(f"{len(self._candidates)} candidates")

    def _candidate_selected(self, selected, _deselected):
        self._bump_generation()
        indexes = selected.indexes()
        self._candidate = self.candidate_model.candidate_at(indexes[0].row()) if indexes else None
        self._clear_plan_state()
        candidate = self._candidate or {}
        catalog_state = getattr(self.service, "catalog_state", None)
        details = {
            "identity": {
                "stable_id": candidate.get("id"),
                "distribution_family": candidate.get("distribution_family"),
                "lifecycle": candidate.get("lifecycle"),
                "channel": candidate.get("channel"),
            },
            "binding": {
                "catalog_hash": (
                    candidate.get("catalog_sha256")
                    or getattr(catalog_state, "catalog_sha256", None)
                ),
                "adapter_id": getattr(self.service, "adapter_name", None),
                "target_gfx": candidate.get("gfx") or self.gfx_edit.currentText().strip() or None,
            },
            "package_set": candidate.get("package_specs") or [],
            "provenance": {
                "source_url": candidate.get("index_url") or candidate.get("source_url"),
                "evidence_status": candidate.get("evidence_status"),
                "resolver_status": candidate.get("resolver_status") or "not_collected",
                "resolver_provenance": candidate.get("resolver_provenance"),
            },
            "warnings": candidate.get("profile_warnings") or candidate.get("warnings") or [],
            "candidate": candidate,
        }
        self.candidate_details.setPlainText(
            json.dumps(details, indent=2, sort_keys=True, default=str)
        )
        self._update_candidate_actions()

    def _update_candidate_actions(self):
        enabled = self.service.target is not None and self._candidate is not None
        self.inventory_button.setEnabled(enabled)
        self.plan_button.setEnabled(enabled)
        self.refresh_extension_button.setEnabled(
            self.service.target is not None
            and getattr(self.service, "extension_capable", lambda: False)()
        )
        self.extension_button.setEnabled(enabled and self.service.catalog is not None)
        self.overview_verify_button.setEnabled(self.service.target is not None)
        self.overview_verify_button.setVisible(self.service.target is not None)
        self.overview_find_button.setEnabled(
            self.service.target is not None and self.service.catalog is not None
        )

    def _clear_plan_state(self):
        self._core_plan = None
        self._extension_plan_result = None
        self.apply_core_button.setEnabled(False)
        self.apply_extension_button.setEnabled(False)

    def _verify(self):
        self._bump_generation()
        self._run("Inspect runtime and hardware", self.service.inspect, self._display_inspection)

    def _inventory(self):
        self._bump_generation()
        self._run(
            "Inventory packages",
            lambda: self.service.inventory(self._candidate),
            lambda result: self._append_json(result.as_dict()),
        )

    def _plan(self):
        self._bump_generation()
        self._run(
            "Build install dry-run",
            lambda: self.service.plan(self._candidate),
            self._display_core_plan,
        )

    def _refresh_extensions(self):
        if self.service.target is None:
            self._show_error("detect a target before refreshing extensions")
            return
        self._bump_generation()
        self._run(
            "Refresh ComfyUI extensions",
            lambda: self.service.extension_inventory(self._candidate),
            self._display_extension_inventory,
        )

    def _extension_plan(self):
        self._bump_generation()
        selections = self._selected_extensions()
        if not selections:
            self._show_error("select at least one extension before building an extension plan")
            return
        self._run(
            "Build extension plan",
            lambda: self.service.extension_plan(self._candidate, selections),
            self._display_extension_plan,
        )

    def _selected_extensions(self):
        selected = []
        for row in range(self.extension_table.rowCount()):
            item = self.extension_table.item(row, 0)
            if item and item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text())
        return tuple(selected)

    def _display_extension_inventory(self, report):
        records = report.get("extensions", [])
        self._extension_records = list(records)
        self.extension_table.setRowCount(len(records))
        for row, extension in enumerate(records):
            installed = ", ".join(
                f"{package['name']}=={package['version']}"
                for package in extension.get("installed", [])
            ) or "not installed"
            evidence = ", ".join(extension.get("catalog_evidence_refs") or extension.get("evidence_refs") or []) or "none"
            values = (
                extension.get("name") or extension.get("id") or "unknown",
                extension.get("status") or "unknown",
                installed,
                extension.get("target_match") or "unknown",
                evidence,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if column == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, extension.get("id"))
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
                self.extension_table.setItem(row, column, item)
        status = report.get("status", "unknown")
        self.extension_summary.setText(f"Inventory: {status} | {len(records)} known extensions")
        self.refresh_extension_button.setEnabled(
            self.service.target is not None
            and getattr(self.service, "extension_capable", lambda: False)()
        )
        self.extension_details.clear()
        self._append_json({"extension_inventory": report})

    def _extension_selected(self):
        row = self.extension_table.currentRow()
        if 0 <= row < len(self._extension_records):
            self.extension_details.setPlainText(
                json.dumps(
                    self._extension_records[row],
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
        else:
            self.extension_details.clear()

    def _display_core_plan(self, result):
        self._core_plan = result
        self.apply_core_button.setEnabled(bool(result.plan.command))
        self._append_json(result.as_dict())

    def _display_extension_plan(self, result):
        self._extension_plan_result = result
        installable = result.extensions and all(
            extension.get("status") == "installable" for extension in result.extensions
        )
        self.apply_extension_button.setEnabled(bool(installable and result.commands))
        self._append_json(result.as_dict())

    def _apply_core(self):
        if self._core_plan is None or self._candidate is None:
            return
        warnings = "\n".join(self._core_plan.plan.warnings) or "none"
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Confirm core package installation",
            f"Target:\n{self._core_plan.plan.target_root}\n\n"
            f"Candidate:\n{self._candidate['id']}\n\n"
            f"Warnings:\n{warnings}\n\n"
            "A target-local package backup will be created before pip runs.\n"
            "Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run(
            "Apply core packages",
            lambda: self.service.apply_core(
                self._core_plan,
                allow_unverified=self.allow_unverified.isChecked(),
            ),
            self._display_apply_result,
            mutation=True,
        )

    def _apply_extensions(self):
        if self._extension_plan_result is None:
            return
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Confirm extension installation",
            "Only Matrix evidence-backed extensions will be installed.\n"
            f"Selected extensions: {', '.join(self._extension_plan_result.selections)}\n"
            f"Selection hash: {self._extension_plan_result.selection_hash}\n\n"
            "A separate extension backup will be created first.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run(
            "Apply extensions",
            lambda: self.service.apply_extensions(self._extension_plan_result),
            self._display_apply_result,
            mutation=True,
        )

    def _display_apply_result(self, result):
        self.apply_core_button.setEnabled(False)
        self.apply_extension_button.setEnabled(False)
        self._append_json(result.as_dict())

    def _restore_dry_run(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select package or extension backup",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self._show_error(f"BackupError: {error}")
            return
        self._restore_path = Path(path)
        self._restore_kind = "extensions" if document.get("kind") == "extensions" else "core"
        self._restore_network_allowed = False
        if self._restore_kind == "extensions":
            def function():
                return self.service.restore_extensions(path)
        else:
            try:
                load_backup(path, materialize=False)
            except BackupError as error:
                if "--allow-network-restore" not in str(error):
                    self._show_error(f"BackupError: {error}")
                    return
                answer = QtWidgets.QMessageBox.warning(
                    self,
                    "Offline restore unavailable",
                    f"{error}\n\nBuild a version-pinned restore plan that may access the network?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                self._restore_network_allowed = True

            def function():
                return self.service.restore_core(
                    path,
                    allow_network_restore=self._restore_network_allowed,
                )
        self._run("Build restore dry-run", function, self._display_restore_plan)

    def _display_restore_plan(self, result):
        self._restore_plan = result
        self.apply_restore_button.setEnabled(bool(result.plan.command))
        self._append_json(result.as_dict())

    def _apply_restore(self):
        if self._restore_plan is None or self._restore_path is None:
            return
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Confirm restore",
            f"Restore target package state from:\n{self._restore_path}\n\n"
            "Extra packages will not be removed."
            + (
                " This restore explicitly allows network access."
                if self._restore_network_allowed
                else ""
            )
            + " Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if self._restore_kind == "extensions":
            def function():
                return self.service.restore_extensions(self._restore_path, apply=True)
        else:
            def function():
                return self.service.restore_core(
                    self._restore_path,
                    apply=True,
                    allow_network_restore=self._restore_network_allowed,
                )
        self._run(
            "Apply restore",
            function,
            self._display_restore_result,
            mutation=True,
        )

    def _display_restore_result(self, result):
        self.apply_restore_button.setEnabled(False)
        self._append_json(result.as_dict())

    def _append_json(self, value):
        details = json.dumps(value, indent=2, sort_keys=True, default=str)
        key = next(iter(value), "operation") if isinstance(value, dict) else "operation"
        payload = value.get(key) if isinstance(value, dict) else value
        summary = key.replace("_", " ").title()
        if isinstance(payload, dict):
            if payload.get("status"):
                summary += f" • {payload['status']}"
            if key == "target_package_inventory":
                summary += f" • {payload.get('compiled_extension_count', 0)} extensions"
            elif key == "extension_inventory":
                summary += f" • {len(payload.get('extensions', ())) } extensions"
            elif key == "catalog":
                summary += f" • {payload.get('artifact_count', 'unknown')} artifacts"
        record = {"summary": summary, "details": details}
        self._activity_records.append(record)
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QtWidgets.QListWidgetItem(f"{timestamp}  {summary}")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, details)
        self.activity_list.addItem(item)
        self._trim_activity()

    def _activity_selected(self, current, _previous):
        if current is None:
            self.output.clear()
            return
        self.output.setPlainText(
            str(current.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        )

    def _clear_activity(self):
        self._activity_records.clear()
        self.activity_list.clear()
        self.output.clear()

    def _trim_activity(self):
        limit = self.activity_retention_spin.value() if hasattr(self, "activity_retention_spin") else 100
        while len(self._activity_records) > limit:
            self._activity_records.pop(0)
            self.activity_list.takeItem(0)

    def _show_error(self, error):
        message = f"ERROR: {error}"
        self.output.setPlainText(message)
        self._activity_records.append({"summary": "Error", "details": message})
        item = QtWidgets.QListWidgetItem(message)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, message)
        self.activity_list.addItem(item)
        self.activity_list.setCurrentItem(item)
        self._trim_activity()
        self.statusBar().showMessage("Error")
