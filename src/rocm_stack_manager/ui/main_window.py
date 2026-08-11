"""QtWidgets main window for read-only ROCm candidate and plan inspection."""

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..adapters.registry import available_adapters
from ..core.platforms import SUPPORTED_PLATFORMS, host_platform
from .models import CandidateTableModel
from .services import ManagerService, detected_gfx_targets
from .workers import Task


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, service=None, parent=None):
        super().__init__(parent)
        self.service = service or ManagerService()
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._tasks = set()
        self._generation = 0
        self._candidates = []
        self._candidate = None
        self._core_plan = None
        self._extension_plan_result = None
        self._restore_plan = None
        self._restore_path = None
        self._restore_kind = None
        self._build_ui()
        if hasattr(self.service, "host_hardware"):
            QtCore.QTimer.singleShot(0, self._probe_host_hardware)

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
        find = QtWidgets.QPushButton("Find candidates")
        find.clicked.connect(self._find_candidates)
        filter_layout.addWidget(QtWidgets.QLabel("Platform"), 0, 0)
        filter_layout.addWidget(self.platform_combo, 0, 1)
        filter_layout.addWidget(QtWidgets.QLabel("GFX"), 0, 2)
        filter_layout.addWidget(self.gfx_edit, 0, 3)
        filter_layout.addWidget(self.gfx_status, 0, 4)
        filter_layout.addWidget(QtWidgets.QLabel("Channel"), 0, 5)
        filter_layout.addWidget(self.channel_combo, 0, 6)
        filter_layout.addWidget(QtWidgets.QLabel("ROCm"), 0, 7)
        filter_layout.addWidget(self.rocm_edit, 0, 8)
        filter_layout.addWidget(find, 0, 9)
        filter_layout.addWidget(QtWidgets.QLabel("Family"), 1, 0)
        filter_layout.addWidget(self.family_combo, 1, 1)
        filter_layout.addWidget(QtWidgets.QLabel("Lifecycle"), 1, 2)
        filter_layout.addWidget(self.lifecycle_combo, 1, 3)
        filter_layout.addWidget(QtWidgets.QLabel("Candidate state"), 1, 5)
        filter_layout.addWidget(self.kind_combo, 1, 6)
        filter_layout.addWidget(self.candidate_summary, 1, 7, 1, 3)
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

        extension_group = QtWidgets.QGroupBox("ComfyUI extension inventory")
        extension_layout = QtWidgets.QVBoxLayout(extension_group)
        self.extension_summary = QtWidgets.QLabel("No target extension inventory")
        self.extension_table = QtWidgets.QTableWidget(0, 8)
        self.extension_table.setHorizontalHeaderLabels(
            ("Extension", "Status", "Installed", "Target match", "Latest artifact", "Matrix claim", "Evidence", "Reason")
        )
        self.extension_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.extension_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.extension_table.horizontalHeader().setStretchLastSection(True)
        self.extension_table.setMinimumHeight(150)
        extension_layout.addWidget(self.extension_summary)
        extension_layout.addWidget(self.extension_table)
        root.addWidget(extension_group)

        actions = QtWidgets.QHBoxLayout()
        self.verify_button = QtWidgets.QPushButton("Verify target")
        self.inventory_button = QtWidgets.QPushButton("Inventory")
        self.plan_button = QtWidgets.QPushButton("Install dry-run")
        self.extension_button = QtWidgets.QPushButton("Extension plan")
        self.apply_core_button = QtWidgets.QPushButton("Apply core")
        self.apply_extension_button = QtWidgets.QPushButton("Apply extension")
        self.restore_button = QtWidgets.QPushButton("Restore dry-run…")
        self.apply_restore_button = QtWidgets.QPushButton("Apply restore")
        self.allow_unverified = QtWidgets.QCheckBox("Allow unverified evidence")
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
        self._bump_generation()
        self.service.set_adapter(adapter_name)
        self.target_summary.setText("No target detected")
        self._candidates = []
        self._candidate = None
        self._clear_plan_state()
        self.candidate_model.set_rows(())
        self.candidate_summary.setText("No candidates loaded")
        self.extension_summary.setText("Extension inventory is unavailable for this adapter")
        self.extension_table.setRowCount(0)
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

    def _bump_generation(self):
        self._generation += 1
        return self._generation

    def _run(self, label, function, callback):
        generation = self._generation
        self.statusBar().showMessage(f"{label}…")
        task = Task(function)
        self._tasks.add(task)
        task.signals.finished.connect(
            lambda result: self._finish_task(task, label, callback, result, generation)
        )
        task.signals.failed.connect(lambda error: self._fail_task(task, label, error, generation))
        self.thread_pool.start(task)

    def _finish_task(self, task, label, callback, result, generation=None):
        self._tasks.discard(task)
        if generation is not None and generation != self._generation:
            self.statusBar().showMessage(f"{label} result discarded (target changed)")
            return
        self.statusBar().showMessage(f"{label} {'failed' if self._result_failed(result) else 'complete'}")
        callback(result)

    @staticmethod
    def _result_failed(result):
        returncode = getattr(result, "returncode", None)
        if returncode is not None and returncode != 0:
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

    def _fail_task(self, task, label, error, generation=None):
        self._tasks.discard(task)
        if generation is not None and generation != self._generation:
            self.statusBar().showMessage(f"{label} error discarded (target changed)")
            return
        self.statusBar().showMessage(f"{label} failed")
        self._show_error(error)

    def _detect(self):
        self._bump_generation()
        path = self.target_edit.text().strip() or "."
        self._run("Detect", lambda: self.service.detect(path), self._display_target_and_probe)

    def _display_target_and_probe(self, target):
        self._display_target(target)
        self._run("Inspect runtime and hardware", self.service.inspect, self._display_inspection)
        if getattr(self.service, "extension_capable", lambda: False)():
            self._run(
                "Inspect ComfyUI extensions",
                self.service.extension_inventory,
                self._display_extension_inventory,
            )

    def _display_target(self, target):
        self._bump_generation()
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
        self.verify_button.setEnabled(True)
        self.restore_button.setEnabled(True)
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

    def _load_catalog(self, refresh):
        self._bump_generation()
        path = None if refresh else (self.catalog_edit.text().strip() or None)
        self._run(
            "Load Matrix catalog",
            lambda: self.service.load_catalog(path, refresh=refresh),
            self._display_catalog,
        )

    def _display_catalog(self, result):
        _, state = result
        fetched = state.fetched_at or "unknown"
        artifacts = state.artifact_count if state.artifact_count is not None else "unknown"
        age = "unknown"
        if state.cache_age_seconds is not None:
            age = f"{state.cache_age_seconds / 86400:.1f} days"
        failures = state.source_failure_count
        self.catalog_summary.setText(
            f"Loaded: {state.path} | Source: {state.source} | "
            f"Fetched: {fetched} ({age} old) | Artifacts: {artifacts} | "
            f"Source failures: {failures} | "
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
            }
        )
        if self.service.target is not None and getattr(self.service, "extension_capable", lambda: False)():
            self._run(
                "Refresh ComfyUI extensions",
                self.service.extension_inventory,
                self._display_extension_inventory,
            )

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
            evidence = ", ".join(extension.get("catalog_evidence_refs") or extension.get("evidence_refs") or []) or "none"
            values = (
                extension.get("name") or extension.get("id") or "unknown",
                extension.get("status") or "unknown",
                installed,
                extension.get("target_match") or "unknown",
                latest_label,
                claim,
                evidence,
                extension.get("reason") or "",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
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
            def function():
                return self.service.restore_extensions(path)
        else:
            def function():
                return self.service.restore_core(path)
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
            def function():
                return self.service.restore_extensions(self._restore_path, apply=True)
        else:
            def function():
                return self.service.restore_core(self._restore_path, apply=True)
        self._run("Apply restore", function, self._display_restore_result)

    def _display_restore_result(self, result):
        self.apply_restore_button.setEnabled(False)
        self._append_json(result.as_dict())

    def _append_json(self, value):
        self.output.appendPlainText(json.dumps(value, indent=2, sort_keys=True, default=str))

    def _show_error(self, error):
        self.output.appendPlainText(f"ERROR: {error}")
        self.statusBar().showMessage("Error")
