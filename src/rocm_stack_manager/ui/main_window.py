"""QtWidgets main window for read-only ROCm candidate and plan inspection."""

import json
import os
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..adapters.registry import available_adapters
from .models import CandidateTableModel
from .services import ManagerService
from .workers import Task


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, service=None, parent=None):
        super().__init__(parent)
        self.service = service or ManagerService()
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._tasks = set()
        self._candidates = []
        self._candidate = None
        self._core_plan = None
        self._extension_plan_result = None
        self._restore_plan = None
        self._restore_path = None
        self._restore_kind = None
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("ROCm Stack Manager")
        self.resize(1180, 760)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        target_group = QtWidgets.QGroupBox("Target")
        target_layout = QtWidgets.QGridLayout(target_group)
        self.adapter_combo = QtWidgets.QComboBox()
        self.adapter_combo.addItems(available_adapters())
        self.adapter_combo.setCurrentText(self.service.adapter_name)
        self.adapter_combo.currentTextChanged.connect(self._adapter_changed)
        self.target_edit = QtWidgets.QLineEdit()
        self.target_edit.setPlaceholderText("Select a ComfyUI portable, venv, or source directory")
        browse = QtWidgets.QPushButton("Browse…")
        detect = QtWidgets.QPushButton("Detect")
        browse.clicked.connect(self._browse_target)
        detect.clicked.connect(self._detect)
        target_layout.addWidget(QtWidgets.QLabel("Adapter"), 0, 0)
        target_layout.addWidget(self.adapter_combo, 0, 1)
        target_layout.addWidget(QtWidgets.QLabel("Path"), 1, 0)
        target_layout.addWidget(self.target_edit, 1, 1, 1, 2)
        target_layout.addWidget(browse, 1, 3)
        target_layout.addWidget(detect, 1, 4)
        self.target_summary = QtWidgets.QLabel("No target detected")
        self.target_summary.setWordWrap(True)
        target_layout.addWidget(self.target_summary, 2, 0, 1, 5)
        root.addWidget(target_group)

        catalog_group = QtWidgets.QGroupBox("Matrix catalog")
        catalog_layout = QtWidgets.QGridLayout(catalog_group)
        self.catalog_edit = QtWidgets.QLineEdit()
        self.catalog_edit.setPlaceholderText("Optional local catalog.json or matrix.json")
        catalog_browse = QtWidgets.QPushButton("Browse…")
        load_catalog = QtWidgets.QPushButton("Load")
        refresh_catalog = QtWidgets.QPushButton("Refresh official")
        catalog_browse.clicked.connect(self._browse_catalog)
        load_catalog.clicked.connect(lambda: self._load_catalog(False))
        refresh_catalog.clicked.connect(lambda: self._load_catalog(True))
        catalog_layout.addWidget(QtWidgets.QLabel("Path"), 0, 0)
        catalog_layout.addWidget(self.catalog_edit, 0, 1)
        catalog_layout.addWidget(catalog_browse, 0, 2)
        catalog_layout.addWidget(load_catalog, 0, 3)
        catalog_layout.addWidget(refresh_catalog, 0, 4)
        self.catalog_summary = QtWidgets.QLabel("No catalog loaded")
        catalog_layout.addWidget(self.catalog_summary, 1, 0, 1, 5)
        root.addWidget(catalog_group)

        filters = QtWidgets.QGroupBox("Candidate filters")
        filter_layout = QtWidgets.QGridLayout(filters)
        self.platform_combo = QtWidgets.QComboBox()
        self.platform_combo.addItems(("windows", "linux"))
        self.platform_combo.setCurrentText("windows" if os.name == "nt" else "linux")
        self.gfx_edit = QtWidgets.QLineEdit()
        self.gfx_edit.setPlaceholderText("gfx1201")
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(("all", "stable", "nightly", "staging"))
        self.rocm_edit = QtWidgets.QLineEdit()
        self.rocm_edit.setPlaceholderText("Optional exact ROCm version")
        find = QtWidgets.QPushButton("Find candidates")
        find.clicked.connect(self._find_candidates)
        filter_layout.addWidget(QtWidgets.QLabel("Platform"), 0, 0)
        filter_layout.addWidget(self.platform_combo, 0, 1)
        filter_layout.addWidget(QtWidgets.QLabel("GFX"), 0, 2)
        filter_layout.addWidget(self.gfx_edit, 0, 3)
        filter_layout.addWidget(QtWidgets.QLabel("Channel"), 0, 4)
        filter_layout.addWidget(self.channel_combo, 0, 5)
        filter_layout.addWidget(QtWidgets.QLabel("ROCm"), 0, 6)
        filter_layout.addWidget(self.rocm_edit, 0, 7)
        filter_layout.addWidget(find, 0, 8)
        root.addWidget(filters)

        self.candidate_model = CandidateTableModel(self)
        self.candidate_view = QtWidgets.QTableView()
        self.candidate_view.setModel(self.candidate_model)
        self.candidate_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_view.setSortingEnabled(False)
        self.candidate_view.selectionModel().selectionChanged.connect(self._candidate_selected)
        self.candidate_view.horizontalHeader().setStretchLastSection(True)
        self.candidate_view.setMinimumHeight(220)
        root.addWidget(self.candidate_view, 1)

        actions = QtWidgets.QHBoxLayout()
        self.verify_button = QtWidgets.QPushButton("Verify target")
        self.inventory_button = QtWidgets.QPushButton("Inventory")
        self.plan_button = QtWidgets.QPushButton("Install dry-run")
        self.extension_button = QtWidgets.QPushButton("Extension plan")
        self.apply_core_button = QtWidgets.QPushButton("Apply core")
        self.apply_extension_button = QtWidgets.QPushButton("Apply extension")
        self.restore_button = QtWidgets.QPushButton("Restore dry-run…")
        self.apply_restore_button = QtWidgets.QPushButton("Apply restore")
        self.allow_unverified = QtWidgets.QCheckBox("Allow failed resolver evidence")
        for button in (
            self.verify_button,
            self.inventory_button,
            self.plan_button,
            self.extension_button,
            self.apply_core_button,
            self.apply_extension_button,
            self.apply_restore_button,
        ):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addWidget(self.restore_button)
        self.restore_button.setEnabled(False)
        actions.addWidget(self.allow_unverified)
        self.verify_button.clicked.connect(self._verify)
        self.inventory_button.clicked.connect(self._inventory)
        self.plan_button.clicked.connect(self._plan)
        self.extension_button.clicked.connect(self._extension_plan)
        self.apply_core_button.clicked.connect(self._apply_core)
        self.apply_extension_button.clicked.connect(self._apply_extensions)
        self.restore_button.clicked.connect(self._restore_dry_run)
        self.apply_restore_button.clicked.connect(self._apply_restore)
        root.addLayout(actions)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Operation results and warnings appear here")
        root.addWidget(self.output, 1)
        self.statusBar().showMessage("Ready")

    def _browse_target(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select ComfyUI target")
        if path:
            self.target_edit.setText(path)

    def _adapter_changed(self, adapter_name):
        self.service.set_adapter(adapter_name)
        self.target_summary.setText("No target detected")
        self._candidates = []
        self._candidate = None
        self._clear_plan_state()
        self.candidate_model.set_rows(())
        self.verify_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        self._update_candidate_actions()

    def _browse_catalog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Matrix catalog",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            self.catalog_edit.setText(path)

    def _run(self, label, function, callback):
        self.statusBar().showMessage(f"{label}…")
        task = Task(function)
        self._tasks.add(task)
        task.signals.finished.connect(lambda result: self._finish_task(task, label, callback, result))
        task.signals.failed.connect(lambda error: self._fail_task(task, label, error))
        self.thread_pool.start(task)

    def _finish_task(self, task, label, callback, result):
        self._tasks.discard(task)
        self.statusBar().showMessage(f"{label} complete")
        callback(result)

    def _fail_task(self, task, label, error):
        self._tasks.discard(task)
        self.statusBar().showMessage(f"{label} failed")
        self._show_error(error)

    def _detect(self):
        path = self.target_edit.text().strip() or "."
        self._run("Detect", lambda: self.service.detect(path), self._display_target)

    def _display_target(self, target):
        values = target.as_dict()
        self.target_summary.setText(
            f"Layout: {values['layout']} | Python: {values['python_executable'] or 'not found'} | "
            f"ComfyUI: {values['comfyui_dir']}"
        )
        self.verify_button.setEnabled(True)
        self.restore_button.setEnabled(True)
        self._update_candidate_actions()

    def _load_catalog(self, refresh):
        path = None if refresh else (self.catalog_edit.text().strip() or None)
        self._run(
            "Load Matrix catalog",
            lambda: self.service.load_catalog(path, refresh=refresh),
            self._display_catalog,
        )

    def _display_catalog(self, result):
        _, state = result
        self.catalog_summary.setText(
            f"Loaded: {state.path} | Source: {state.source} | "
            f"Refresh requested: {'yes' if state.refreshed else 'no'}"
        )
        self._append_json({"catalog": str(state.path), "source": state.source})

    def _find_candidates(self):
        if self.service.catalog is None:
            self._show_error("CatalogError: load a Matrix catalog first")
            return
        gfx = self.gfx_edit.text().strip()
        if not gfx:
            self._show_error("ValueError: enter a GFX target")
            return
        channel = self.channel_combo.currentText()
        channel = None if channel == "all" else channel
        rocm_version = self.rocm_edit.text().strip() or None
        self._run(
            "Find candidates",
            lambda: self.service.candidates(
                platform=self.platform_combo.currentText(),
                gfx=gfx,
                channel=channel,
                rocm_version=rocm_version,
            ),
            self._display_candidates,
        )

    def _display_candidates(self, candidates):
        self._candidates = list(candidates)
        self._candidate = None
        self._clear_plan_state()
        self.candidate_model.set_rows(self._candidates)
        self._update_candidate_actions()
        self.statusBar().showMessage(f"{len(self._candidates)} candidates")

    def _candidate_selected(self, selected, _deselected):
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
        self._run("Verify target", self.service.verify, lambda result: self._append_json(result.as_dict()))

    def _inventory(self):
        self._run(
            "Inventory packages",
            lambda: self.service.inventory(self._candidate),
            lambda result: self._append_json(result.as_dict()),
        )

    def _plan(self):
        self._run(
            "Build install dry-run",
            lambda: self.service.plan(self._candidate),
            self._display_core_plan,
        )

    def _extension_plan(self):
        self._run(
            "Build extension plan",
            lambda: self.service.extension_plan(self._candidate),
            self._display_extension_plan,
        )

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
                self._candidate,
                allow_unverified=self.allow_unverified.isChecked(),
            ),
            self._display_apply_result,
        )

    def _apply_extensions(self):
        if self._extension_plan_result is None:
            return
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Confirm extension installation",
            "Only Matrix evidence-backed extensions will be installed.\n"
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
        if self._restore_kind == "extensions":
            function = lambda: self.service.restore_extensions(path)
        else:
            function = lambda: self.service.restore_core(path)
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
            "Extra packages will not be removed. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if self._restore_kind == "extensions":
            function = lambda: self.service.restore_extensions(self._restore_path, apply=True)
        else:
            function = lambda: self.service.restore_core(self._restore_path, apply=True)
        self._run("Apply restore", function, self._display_restore_result)

    def _display_restore_result(self, result):
        self.apply_restore_button.setEnabled(False)
        self._append_json(result.as_dict())

    def _append_json(self, value):
        self.output.appendPlainText(json.dumps(value, indent=2, sort_keys=True, default=str))

    def _show_error(self, error):
        self.output.appendPlainText(f"ERROR: {error}")
        self.statusBar().showMessage("Error")
