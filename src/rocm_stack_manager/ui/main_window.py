"""QtWidgets main window for read-only ROCm candidate and plan inspection."""

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..adapters.registry import available_adapters
from ..core.backup import BackupError, load_backup
from ..core.platforms import SUPPORTED_PLATFORMS, host_platform
from .models import CandidateTableModel
from .services import ManagerService, detected_gfx_targets
from .workers import Task


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
        self._build_ui()
        self._restore_ui_settings()
        if self.settings is not None:
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
        self.target_edit.setText(str(self.settings.value("configuration/target_path", "")))
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
        self.settings.setValue("configuration/platform", self.platform_combo.currentText())
        self.settings.setValue("configuration/gfx", self.gfx_edit.currentText().strip())
        self.settings.setValue("configuration/channel", self.channel_combo.currentText())
        self.settings.setValue("configuration/rocm", self.rocm_edit.text().strip())
        self.settings.setValue("configuration/family", self.family_combo.currentText())
        self.settings.setValue("configuration/lifecycle", self.lifecycle_combo.currentText())
        self.settings.setValue("configuration/candidate_kind", self.kind_combo.currentText())
        self.settings.sync()

    def _restore_combo_value(self, combo, key):
        if self.settings is None:
            return
        value = self.settings.value(key)
        if value is not None and combo.findText(str(value)) >= 0:
            combo.setCurrentText(str(value))

    def _build_ui(self):
        self.setWindowTitle("ROCm Stack Manager")
        self.resize(1180, 760)

        central = QtWidgets.QWidget(self)
        central.setObjectName("mainSurface")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        setup_page = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(setup_page)
        setup_layout.setContentsMargins(4, 18, 4, 4)
        setup_layout.setSpacing(14)

        header = QtWidgets.QWidget()
        header.setObjectName("pageHeader")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(2, 0, 2, 4)
        header_layout.setSpacing(3)
        eyebrow = QtWidgets.QLabel("ROCM STACK MANAGER")
        eyebrow.setObjectName("eyebrow")
        title = QtWidgets.QLabel("Manage a target environment")
        title.setObjectName("pageTitle")
        subtitle = QtWidgets.QLabel(
            "Inspect package evidence, build a dry-run plan, and keep every change explicit."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(eyebrow)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        setup_layout.addWidget(header)

        target_group = QtWidgets.QGroupBox("Target")
        target_group.setObjectName("cardGroup")
        target_layout = QtWidgets.QGridLayout(target_group)
        target_layout.setContentsMargins(16, 20, 16, 16)
        target_layout.setHorizontalSpacing(12)
        target_layout.setVerticalSpacing(10)
        self.adapter_combo = QtWidgets.QComboBox()
        self.adapter_combo.addItems(available_adapters())
        self.adapter_combo.setCurrentText(self.service.adapter_name)
        self.adapter_combo.currentTextChanged.connect(self._adapter_changed)
        self.target_edit = QtWidgets.QLineEdit()
        self.target_edit.setPlaceholderText("Select a ComfyUI portable, venv, or source directory")
        self.target_browse_button = QtWidgets.QPushButton("Browse…")
        self.detect_button = QtWidgets.QPushButton("Detect")
        self.detect_button.setObjectName("primaryButton")
        self.target_browse_button.clicked.connect(self._browse_target)
        self.detect_button.clicked.connect(self._detect)
        target_layout.addWidget(QtWidgets.QLabel("Adapter"), 0, 0)
        target_layout.addWidget(self.adapter_combo, 0, 1)
        target_layout.addWidget(QtWidgets.QLabel("Path"), 1, 0)
        target_layout.addWidget(self.target_edit, 1, 1, 1, 2)
        target_layout.addWidget(self.target_browse_button, 1, 3)
        target_layout.addWidget(self.detect_button, 1, 4)
        self.target_summary = QtWidgets.QLabel("No target detected")
        self.target_summary.setObjectName("targetStatus")
        self.target_summary.setWordWrap(True)
        self.target_summary.setMinimumWidth(0)
        self.target_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        target_layout.addWidget(self.target_summary, 2, 0, 1, 5)
        self.snapshot_summary = QtWidgets.QLabel(
            "Installed packages: not loaded | Extensions: not loaded | "
            "Runtime verification: not run | Hardware/tensor: not run"
        )
        self.snapshot_summary.setObjectName("summaryStatus")
        self.snapshot_summary.setWordWrap(True)
        self.snapshot_summary.setMinimumWidth(0)
        self.snapshot_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        target_layout.addWidget(self.snapshot_summary, 3, 0, 1, 5)
        setup_layout.addWidget(target_group)

        catalog_group = QtWidgets.QGroupBox("Matrix catalog")
        catalog_group.setObjectName("cardGroup")
        catalog_layout = QtWidgets.QGridLayout(catalog_group)
        catalog_layout.setContentsMargins(16, 20, 16, 16)
        catalog_layout.setHorizontalSpacing(12)
        catalog_layout.setVerticalSpacing(10)
        self.catalog_edit = QtWidgets.QLineEdit()
        self.catalog_edit.setPlaceholderText("Optional local catalog.json or matrix.json")
        self.catalog_browse_button = QtWidgets.QPushButton("Browse…")
        self.load_catalog_button = QtWidgets.QPushButton("Load")
        self.refresh_catalog_button = QtWidgets.QPushButton("Refresh official")
        self.refresh_catalog_button.setObjectName("primaryButton")
        self.catalog_browse_button.clicked.connect(self._browse_catalog)
        self.load_catalog_button.clicked.connect(lambda: self._load_catalog(False))
        self.refresh_catalog_button.clicked.connect(lambda: self._load_catalog(True))
        catalog_layout.addWidget(QtWidgets.QLabel("Path"), 0, 0)
        catalog_layout.addWidget(self.catalog_edit, 0, 1)
        catalog_layout.addWidget(self.catalog_browse_button, 0, 2)
        catalog_layout.addWidget(self.load_catalog_button, 0, 3)
        catalog_layout.addWidget(self.refresh_catalog_button, 0, 4)
        self.catalog_summary = QtWidgets.QLabel("No catalog loaded")
        self.catalog_summary.setObjectName("catalogStatus")
        self.catalog_summary.setWordWrap(True)
        self.catalog_summary.setMinimumWidth(0)
        self.catalog_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        catalog_layout.addWidget(self.catalog_summary, 1, 0, 1, 5)
        setup_layout.addWidget(catalog_group)

        filters = QtWidgets.QGroupBox("Candidate filters")
        filters.setObjectName("cardGroup")
        filter_layout = QtWidgets.QGridLayout(filters)
        filter_layout.setContentsMargins(16, 20, 16, 16)
        filter_layout.setHorizontalSpacing(12)
        filter_layout.setVerticalSpacing(10)
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
        self.gfx_status.setObjectName("summaryStatus")
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
        self.candidate_summary.setObjectName("summaryStatus")
        self.candidate_summary.setWordWrap(True)
        self.candidate_summary.setMinimumWidth(0)
        self.candidate_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.find_button = QtWidgets.QPushButton("Find candidates")
        self.find_button.setObjectName("primaryButton")
        self.find_button.clicked.connect(self._find_candidates)
        filter_layout.addWidget(QtWidgets.QLabel("Platform"), 0, 0)
        filter_layout.addWidget(self.platform_combo, 0, 1)
        filter_layout.addWidget(QtWidgets.QLabel("GFX"), 0, 2)
        filter_layout.addWidget(self.gfx_edit, 0, 3)
        filter_layout.addWidget(self.gfx_status, 0, 4, 1, 2)
        filter_layout.addWidget(QtWidgets.QLabel("Channel"), 1, 0)
        filter_layout.addWidget(self.channel_combo, 1, 1)
        filter_layout.addWidget(QtWidgets.QLabel("ROCm"), 1, 2)
        filter_layout.addWidget(self.rocm_edit, 1, 3)
        filter_layout.addWidget(QtWidgets.QLabel("Family"), 1, 4)
        filter_layout.addWidget(self.family_combo, 1, 5)
        filter_layout.addWidget(QtWidgets.QLabel("Lifecycle"), 2, 0)
        filter_layout.addWidget(self.lifecycle_combo, 2, 1)
        filter_layout.addWidget(QtWidgets.QLabel("Candidate state"), 2, 2)
        filter_layout.addWidget(self.kind_combo, 2, 3)
        filter_layout.addWidget(self.find_button, 2, 4)
        filter_layout.addWidget(self.candidate_summary, 3, 0, 1, 6)
        for column in (1, 3, 5):
            filter_layout.setColumnStretch(column, 1)
        setup_layout.addWidget(filters)
        setup_layout.addStretch(1)
        self.tabs.addTab(setup_page, "Setup")

        candidates_page = QtWidgets.QWidget()
        candidates_layout = QtWidgets.QVBoxLayout(candidates_page)
        candidates_layout.setContentsMargins(4, 18, 4, 4)
        candidates_layout.setSpacing(14)
        self.candidate_model = CandidateTableModel(self)
        self.candidate_view = QtWidgets.QTableView()
        self.candidate_view.setModel(self.candidate_model)
        self.candidate_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_view.setSortingEnabled(False)
        self.candidate_view.selectionModel().selectionChanged.connect(self._candidate_selected)
        self.candidate_view.horizontalHeader().setStretchLastSection(True)
        self.candidate_view.setMinimumHeight(220)
        self.candidate_view.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        candidates_layout.addWidget(self.candidate_view, 1)

        candidate_actions = QtWidgets.QGroupBox("Candidate actions")
        candidate_actions.setObjectName("actionGroup")
        candidate_actions_layout = QtWidgets.QHBoxLayout(candidate_actions)
        candidate_actions_layout.setContentsMargins(14, 12, 14, 12)
        candidate_actions_layout.setSpacing(8)
        self.verify_button = QtWidgets.QPushButton("Verify target")
        self.inventory_button = QtWidgets.QPushButton("Inventory")
        self.plan_button = QtWidgets.QPushButton("Install dry-run")
        self.plan_button.setObjectName("primaryButton")
        self.apply_core_button = QtWidgets.QPushButton("Apply core")
        self.apply_core_button.setObjectName("warningButton")
        self.allow_unverified = QtWidgets.QCheckBox("Allow unverified evidence")
        for button in (
            self.verify_button,
            self.inventory_button,
            self.plan_button,
            self.apply_core_button,
        ):
            button.setEnabled(False)
            candidate_actions_layout.addWidget(button)
        candidate_actions_layout.addWidget(self.allow_unverified)
        candidate_actions_layout.addStretch(1)
        candidates_layout.addWidget(candidate_actions)
        self.tabs.addTab(candidates_page, "Candidates")

        extensions_page = QtWidgets.QWidget()
        extensions_layout = QtWidgets.QVBoxLayout(extensions_page)
        extensions_layout.setContentsMargins(4, 18, 4, 4)
        extensions_layout.setSpacing(14)
        extension_group = QtWidgets.QGroupBox("ComfyUI extension inventory")
        extension_group.setObjectName("cardGroup")
        extension_layout = QtWidgets.QVBoxLayout(extension_group)
        extension_layout.setContentsMargins(16, 20, 16, 16)
        extension_layout.setSpacing(10)
        self.extension_summary = QtWidgets.QLabel("No target extension inventory")
        self.extension_summary.setObjectName("summaryStatus")
        self.extension_summary.setWordWrap(True)
        self.extension_summary.setMinimumWidth(0)
        self.extension_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.extension_table = QtWidgets.QTableWidget(0, 10)
        self.extension_table.setHorizontalHeaderLabels(
            ("Extension", "Status", "Installed", "Target match", "Latest artifact", "Artifact platform", "Python/ABI", "Matrix claim", "Evidence", "Reason")
        )
        self.extension_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.extension_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.extension_table.horizontalHeader().setStretchLastSection(True)
        self.extension_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.extension_table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self.extension_table.setWordWrap(False)
        self.extension_table.setMinimumHeight(150)
        extension_layout.addWidget(self.extension_summary)
        extension_layout.addWidget(self.extension_table, 1)
        extensions_layout.addWidget(extension_group, 1)

        extension_actions = QtWidgets.QGroupBox("Extension actions")
        extension_actions.setObjectName("actionGroup")
        extension_actions_layout = QtWidgets.QHBoxLayout(extension_actions)
        extension_actions_layout.setContentsMargins(14, 12, 14, 12)
        extension_actions_layout.setSpacing(8)
        self.extension_button = QtWidgets.QPushButton("Extension plan")
        self.extension_button.setObjectName("primaryButton")
        self.apply_extension_button = QtWidgets.QPushButton("Apply extension")
        self.apply_extension_button.setObjectName("warningButton")
        self.extension_button.setEnabled(False)
        self.apply_extension_button.setEnabled(False)
        extension_actions_layout.addWidget(self.extension_button)
        extension_actions_layout.addWidget(self.apply_extension_button)
        extension_actions_layout.addStretch(1)
        extensions_layout.addWidget(extension_actions)
        self.tabs.addTab(extensions_page, "Extensions")

        activity_page = QtWidgets.QWidget()
        activity_layout = QtWidgets.QVBoxLayout(activity_page)
        activity_layout.setContentsMargins(4, 18, 4, 4)
        activity_layout.setSpacing(14)
        restore_group = QtWidgets.QGroupBox("Restore")
        restore_group.setObjectName("actionGroup")
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

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setObjectName("activityOutput")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Operation results and warnings appear here")
        activity_layout.addWidget(self.output, 1)
        self.tabs.addTab(activity_page, "Activity")

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
        self._apply_visual_theme()

    def _apply_visual_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget#mainSurface {
                background: #101318;
                color: #e6eaf0;
                font-size: 13px;
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
            QLabel#targetStatus, QLabel#catalogStatus, QLabel#summaryStatus {
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
        )

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
            self.target_summary.setText(f"Saved target unavailable: {resolved_target}")
            self.snapshot_summary.setText(
                "Installed packages: not loaded | Extensions: not loaded | "
                "Runtime verification: not run | Hardware/tensor: not run"
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
        catalog_path = self._startup_catalog_path
        self._startup_catalog_path = None
        if not catalog_path:
            self._startup_autoload_candidates = False
            return
        resolved_catalog = Path(catalog_path).expanduser()
        if not resolved_catalog.is_file():
            self.catalog_summary.setText(f"Saved local catalog unavailable: {resolved_catalog}")
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
                core[name] = version
            elif package.get("compiled_files"):
                extensions.append(f"{name}=={version}")
        core_text = ", ".join(f"{name}=={version}" for name, version in sorted(core.items())) or "none detected"
        extension_text = ", ".join(sorted(extensions)) or "none detected"
        self.snapshot_summary.setText(
            f"Installed packages: {core_text} | Extensions: {extension_text} | "
            "Runtime verification: not run | Hardware/tensor: not run"
        )
        self.snapshot_summary.setToolTip(json.dumps(values, indent=2, sort_keys=True, default=str))
        self._append_json({"target_package_inventory": values})

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
            self.statusBar().showMessage(
                f"{label} {'failed' if self._result_failed(result) else 'complete'}"
            )
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
            self.restore_button,
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
        self.target_summary.setText(
            f"Layout: {values['layout']} | Python: {values['python_executable'] or 'not found'} | "
            f"ComfyUI: {values['comfyui_dir']}"
        )
        self.snapshot_summary.setText(
            "Installed packages: not loaded | Extensions: pending | "
            "Runtime verification: available from Verify target | "
            "Hardware/tensor: available from Verify target"
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
        self.apply_restore_button.setEnabled(False)
        self.candidate_model.set_rows(())
        self.candidate_summary.setText("No candidates loaded")
        self.target_summary.setText(summary)
        self.snapshot_summary.setText(
            "Installed packages: not loaded | Extensions: not loaded | "
            "Runtime verification: not run | Hardware/tensor: not run"
        )
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
        self._candidates = []
        self._candidate = None
        self._clear_plan_state()
        self.candidate_model.set_rows(())
        self.candidate_summary.setText("No candidates loaded")
        self.catalog_summary.setText("Loading Matrix catalog…")
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
        self.catalog_summary.setText(
            f"Loaded: {state.path} | Source: {state.source} | "
            f"Fetched: {fetched} ({age} old) | Artifacts: {artifacts} | "
            f"Source failures: {failures} ({failure_ids}) | "
            f"Refresh requested: {'yes' if state.refreshed else 'no'}"
        )
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
        self.statusBar().showMessage(f"{len(self._candidates)} candidates")

    def _candidate_selected(self, selected, _deselected):
        self._bump_generation()
        indexes = selected.indexes()
        self._candidate = self.candidate_model.candidate_at(indexes[0].row()) if indexes else None
        self._clear_plan_state()
        self._update_candidate_actions()

    def _update_candidate_actions(self):
        enabled = self.service.target is not None and self._candidate is not None
        self.inventory_button.setEnabled(enabled)
        self.plan_button.setEnabled(enabled)
        self.extension_button.setEnabled(enabled and self.service.catalog is not None)

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
        self.extension_table.setRowCount(len(records))
        for row, extension in enumerate(records):
            installed = ", ".join(
                f"{package['name']}=={package['version']}"
                for package in extension.get("installed", [])
            ) or "not installed"
            claim = extension.get("matrix_claim_status") or extension.get("claim_status") or "unknown"
            latest = extension.get("latest_artifact") or {}
            latest_label = latest.get("version") or "not collected"
            artifact_platform = ", ".join(latest.get("platform_tags") or []) or "unknown"
            artifact_python_abi = "; ".join(
                value
                for value in (
                    ", ".join(latest.get("python_tags") or []),
                    ", ".join(latest.get("abi_tags") or []),
                )
                if value
            ) or "unknown"
            evidence = ", ".join(extension.get("catalog_evidence_refs") or extension.get("evidence_refs") or []) or "none"
            values = (
                extension.get("name") or extension.get("id") or "unknown",
                extension.get("status") or "unknown",
                installed,
                extension.get("target_match") or "unknown",
                latest_label,
                artifact_platform,
                artifact_python_abi,
                claim,
                evidence,
                extension.get("reason") or "",
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
        self._append_json({"extension_inventory": report})

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
        self.output.appendPlainText(json.dumps(value, indent=2, sort_keys=True, default=str))

    def _show_error(self, error):
        self.output.appendPlainText(f"ERROR: {error}")
        self.statusBar().showMessage("Error")
