import importlib.util
import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PY_SIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PY_SIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets

    from rocm_stack_manager.core.detection import TargetLayout
    from rocm_stack_manager.core.catalog import CatalogError
    from rocm_stack_manager.core.verify import RuntimeObservation
    from rocm_stack_manager.ui.main_window import MainWindow
    from rocm_stack_manager.ui.services import ManagerService, detected_gfx_targets
    from rocm_stack_manager.ui.workers import Task


@unittest.skipUnless(PY_SIDE_AVAILABLE, "PySide6 optional dependency is not installed")
class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _target(self, root):
        comfyui = root / "ComfyUI"
        comfyui.mkdir()
        (comfyui / "main.py").write_text("", encoding="utf-8")
        return TargetLayout(
            root=root,
            comfyui_dir=comfyui,
            python_executable=None,
            launchers=(),
            layout="source",
        )

    def test_window_starts_read_only(self):
        window = MainWindow()
        self.assertEqual(window.windowTitle(), "ROCm Stack Manager")
        self.assertEqual(
            [window.tabs.tabText(index) for index in range(window.tabs.count())],
            ["Overview", "Candidates", "Extensions", "Activity", "Settings"],
        )
        self.assertEqual(window.adapter_combo.currentText(), "comfyui")
        self.assertEqual(
            [window.adapter_combo.itemText(index) for index in range(window.adapter_combo.count())],
            ["comfyui", "ollama"],
        )
        self.assertFalse(window.plan_button.isEnabled())
        self.assertFalse(window.extension_button.isEnabled())
        self.assertFalse(window.apply_core_button.isEnabled())
        self.assertFalse(window.apply_extension_button.isEnabled())
        self.assertFalse(window.restore_button.isEnabled())
        self.assertFalse(window.apply_restore_button.isEnabled())
        self.assertFalse(any(button.text() == "Apply" for button in window.findChildren(QtWidgets.QPushButton)))
        self.assertTrue(window.catalog_summary.wordWrap())
        self.assertEqual(
            window.catalog_summary.sizePolicy().horizontalPolicy(),
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.assertIs(window.tabs.widget(3).findChild(QtWidgets.QPlainTextEdit), window.output)
        window.close()

    def test_workspace_sidebar_switches_stacked_pages(self):
        window = MainWindow()
        self.assertIsInstance(window.tabs, QtWidgets.QStackedWidget)
        for index, button in enumerate(window.navigation_buttons):
            button.click()
            self.assertEqual(window.tabs.currentIndex(), index)
            self.assertTrue(button.isChecked())
        window.close()

    def test_workspace_surfaces_match_the_five_stage_plan(self):
        window = MainWindow()
        self.assertEqual(
            [window.overview_detect_button.text(), window.overview_load_catalog_button.text(), window.overview_find_button.text()],
            ["Detect target", "Load catalog", "Find candidates"],
        )
        self.assertFalse(window.overview_verify_button.isVisible())
        self.assertEqual(window.refresh_extension_button.text(), "Refresh inventory")
        self.assertEqual(window.candidate_model.HEADERS, (
            "Version set", "GFX", "Python", "Platform", "Kind", "Evidence", "Resolver", "Warnings"
        ))
        self.assertEqual(window.restore_target_path_check.text(), "Restore last target path")
        self.assertEqual(window.restore_catalog_path_check.text(), "Restore last catalog path")
        self.assertTrue(window.never_refresh_network_check.isChecked())
        self.assertEqual(window.theme_combo.currentText(), "System")
        self.assertEqual(window.density_combo.currentText(), "Comfortable")
        window.close()

    def test_candidate_rows_show_compact_identity_and_warning(self):
        window = MainWindow()
        window.candidate_model.set_rows([
            {
                "rocm_version": "7.14.0",
                "torch_version": "2.12.0",
                "torchvision_version": "0.27.0",
                "gfx": "gfx1201",
                "python_compatibility": "cp312",
                "platform": "windows",
                "candidate_kind": "artifact_only",
                "evidence_level": "artifact_available",
                "resolver_status": "not_collected",
                "profile_warnings": ["resolver evidence not collected"],
            }
        ])
        values = [
            window.candidate_model.data(window.candidate_model.index(0, column))
            for column in range(window.candidate_model.columnCount())
        ]
        self.assertEqual(values[0], "7.14.0 / 2.12.0 / 0.27.0")
        self.assertEqual(values[4], "artifact_only")
        self.assertEqual(values[6], "not_collected")
        self.assertIn("resolver evidence", values[7])
        window.close()

    def test_activity_retention_keeps_summary_records_bounded(self):
        window = MainWindow()
        window.activity_retention_spin.setValue(10)
        for index in range(11):
            window._append_json({f"operation_{index}": {"status": "ok"}})
        self.assertEqual(window.activity_list.count(), 10)
        self.assertEqual(len(window._activity_records), 10)
        self.assertIn("Operation 10", window.activity_list.item(9).text())
        window.close()

    def test_candidate_filter_rail_can_be_hidden(self):
        window = MainWindow()
        window.show()
        window.tabs.setCurrentIndex(1)
        self.application.processEvents()
        self.assertTrue(window.filter_rail.isVisible())
        window.filters_button.click()
        self.assertFalse(window.filter_rail.isVisible())
        self.assertEqual(window.filters_button.text(), "Show filters")
        window.filters_button.click()
        self.assertTrue(window.filter_rail.isVisible())
        window.close()

    def test_ui_settings_restore_configuration_without_restoring_plans(self):
        settings = QtCore.QSettings(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            "ROCmStackManagerTests",
            "UiSettings",
        )
        settings.clear()
        first = MainWindow(settings=settings)
        first.target_edit.setText("C:/targets/comfyui")
        first.catalog_edit.setText("C:/catalogs/catalog.json")
        first.platform_combo.setCurrentText("linux")
        first.channel_combo.setCurrentText("nightly")
        first.rocm_edit.setText("7.14.0")
        first.gfx_edit.setEditText("gfx1201")
        first.tabs.setCurrentIndex(2)
        first._core_plan = object()
        first.apply_core_button.setEnabled(True)
        first.close()

        second = MainWindow(settings=settings)
        self.assertEqual(second.target_edit.text(), "C:/targets/comfyui")
        self.assertEqual(second.catalog_edit.text(), "C:/catalogs/catalog.json")
        self.assertEqual(second.platform_combo.currentText(), "linux")
        self.assertEqual(second.channel_combo.currentText(), "nightly")
        self.assertEqual(second.rocm_edit.text(), "7.14.0")
        self.assertEqual(second.gfx_edit.currentText(), "gfx1201")
        self.assertEqual(second.tabs.currentIndex(), 2)
        self.assertIsNone(second._core_plan)
        self.assertFalse(second.apply_core_button.isEnabled())
        second.close()
        settings.clear()

    def test_saved_target_startup_loads_metadata_without_full_verification(self):
        class Inventory:
            def as_dict(self):
                return {
                    "status": "detected",
                    "packages": [
                        {"name": "torch", "version": "2.12.0+rocm7.14.0"},
                        {"name": "bitsandbytes", "version": "0.50.0", "compiled_files": ["x.pyd"]},
                    ],
                }

        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            service = ManagerService()
            window = MainWindow(service=service)
            window.target_edit.setText(directory)
            service.prepare_target_snapshot = lambda path: (
                target,
                Inventory(),
                {"status": "detected", "extensions": [], "network_access": False},
            )
            calls = []

            def run_inline(label, function, callback, **kwargs):
                calls.append(label)
                callback(function())

            with patch.object(window, "_run", side_effect=run_inline), patch.object(
                service, "inspect"
            ) as inspect:
                window._start_saved_snapshot()

            self.assertEqual(service.target, target)
            self.assertEqual(
                window.snapshot_metric_values["torch"].text(),
                "2.12.0+rocm7.14.0",
            )
            self.assertEqual(window.snapshot_summary.toolTip(), window.snapshot_summary.text())
            self.assertNotIn('"requires"', window.snapshot_summary.toolTip())
            self.assertIn("Read saved target snapshot", calls)
            activity_details = "\n".join(record["details"] for record in window._activity_records)
            self.assertIn('"compiled_extension_count": 1', activity_details)
            self.assertNotIn('"requires"', activity_details)
            inspect.assert_not_called()
            window.close()

    def test_saved_local_catalog_is_loaded_without_network_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.json"
            catalog_path.write_text("{}", encoding="utf-8")
            window = MainWindow()
            window.catalog_edit.setText(str(catalog_path))
            with patch.object(window, "_load_catalog") as load_catalog:
                window._start_saved_snapshot()
            load_catalog.assert_called_once_with(False, startup=True)
            window.close()

    def test_detect_button_loads_target_snapshot(self):
        class Inventory:
            def as_dict(self):
                return {
                    "status": "detected",
                    "packages": [{"name": "torch", "version": "2.12.0+rocm7.14.0"}],
                }

        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            service = ManagerService()
            service.detect = lambda path: target
            service.prepare_detected_target_snapshot = lambda detected: (
                Inventory(),
                {"status": "detected", "extensions": [], "network_access": False},
            )
            window = MainWindow(service=service)
            window.target_edit.setText(directory)
            calls = []

            def run_inline(label, function, callback, **kwargs):
                calls.append(label)
                if label != "Inspect runtime and hardware":
                    callback(function())

            with patch.object(window, "_run", side_effect=run_inline):
                window.detect_button.click()

            self.assertEqual(service.target, target)
            self.assertIn("Read target package snapshot", calls)
            self.assertEqual(
                window.snapshot_metric_values["torch"].text(),
                "2.12.0+rocm7.14.0",
            )
            self.assertNotIn('"requires"', window.snapshot_summary.toolTip())
            activity_details = "\n".join(record["details"] for record in window._activity_records)
            self.assertIn('"compiled_extension_count": 0', activity_details)
            window.close()

    def test_service_prepares_startup_snapshot_without_runtime_or_hardware_probe(self):
        inventory = object()

        class Adapter:
            id = "fake"

            def detect(self, path):
                return "target"

            def inventory(self, target, candidate=None):
                return inventory

            def extension_inventory(
                self,
                target,
                candidate=None,
                profile_documents=None,
                extension_catalog=None,
                inventory=None,
            ):
                return {"inventory_identity": id(inventory), "network_access": False}

            def extension_plan(self, *args, **kwargs):
                raise AssertionError("startup snapshot must not build an extension plan")

        service = ManagerService()
        service.adapter = Adapter()
        target, prepared_inventory, extension_report = service.prepare_target_snapshot("target")
        self.assertEqual(target, "target")
        self.assertIs(prepared_inventory, inventory)
        self.assertEqual(extension_report["inventory_identity"], id(inventory))
        self.assertFalse(extension_report["network_access"])

    def test_startup_candidate_listing_never_selects_or_plans(self):
        window = MainWindow()
        window.service.target = object()
        window.service.catalog = {"targets": []}
        window.gfx_edit.setEditText("gfx1201")
        window._startup_autoload_candidates = True
        with patch.object(window, "_find_candidates") as find_candidates:
            window._maybe_start_saved_candidates()
        find_candidates.assert_called_once_with()
        self.assertIsNone(window._candidate)
        self.assertIsNone(window._core_plan)
        window.close()

    def test_failed_result_is_not_reported_as_complete(self):
        window = MainWindow()
        result = RuntimeObservation(
            target_root=Path("target"),
            python_executable=None,
            host_platform="windows",
            runtime_status="not_detected",
            hardware_status="not_detected",
        )
        self.assertTrue(window._result_failed(result))
        window.close()

    def test_applied_result_without_return_code_is_failure(self):
        window = MainWindow()
        failed = type("Result", (), {"applied": True, "returncode": None})()
        dry_run = type("Result", (), {"applied": False, "returncode": None})()
        self.assertTrue(window._result_failed(failed))
        self.assertFalse(window._result_failed(dry_run))
        window.close()

    def test_worker_is_not_deleted_before_completion_signal(self):
        task = Task(lambda: None)
        self.assertFalse(task.autoDelete())

    def test_stale_async_result_is_discarded_after_generation_changes(self):
        window = MainWindow()
        marker = []
        task = object()
        window._tasks.add(task)
        window._generation = 2
        window._finish_task(task, "Find candidates", lambda result: marker.append(result), {"old": True}, 1)
        self.assertEqual(marker, [])
        self.assertIn("discarded", window.statusBar().currentMessage())
        window.close()

    def test_candidate_selection_enables_plan_views_only(self):
        class FakeService:
            adapter_name = "comfyui"
            target = None
            catalog = {"_comfyui_extension_profiles": {}}

        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            service = FakeService()
            window = MainWindow(service=service)
            service.target = target
            window._display_target(target)
            window._display_candidates(
                [
                    {
                        "id": "therock:windows:stable:gfx1201:7.14.0:2.12.0",
                        "channel": "stable",
                        "rocm_version": "7.14.0",
                        "torch_version": "2.12.0+rocm7.14.0",
                        "torchvision_version": "0.27.0+rocm7.14.0",
                        "python_compatibility": "compatible",
                        "candidate_kind": "installable",
                        "status": "artifact_available",
                        "profile_status": "documented",
                    }
                ]
            )
            window.candidate_view.selectRow(0)
            self.application.processEvents()
            self.assertTrue(window.plan_button.isEnabled())
            self.assertTrue(window.inventory_button.isEnabled())
            self.assertTrue(window.extension_button.isEnabled())
            self.assertFalse(window.apply_core_button.isEnabled())
            self.assertFalse(window.apply_extension_button.isEnabled())
            details = window.candidate_details.toPlainText()
            self.assertIn('"stable_id":', details)
            self.assertIn('"adapter_id": "comfyui"', details)
            self.assertIn('"target_gfx":', details)
            self.assertFalse(window.overview_verify_button.isHidden())
            window.close()

    def test_candidate_summary_includes_lifecycle_and_state_counts(self):
        window = MainWindow()
        window._display_candidates(
            [
                {"lifecycle": "current", "candidate_kind": "installable"},
                {"lifecycle": "historical", "candidate_kind": "artifact_only"},
            ]
        )
        self.assertIn("2 total", window.candidate_summary.text())
        self.assertIn("current 1", window.candidate_summary.text())
        self.assertIn("historical 1", window.candidate_summary.text())
        self.assertIn("installable 1", window.candidate_summary.text())
        window.close()

    def test_extension_inventory_is_rendered_separately(self):
        window = MainWindow()
        window._display_extension_inventory(
            {
                "status": "detected",
                "extensions": [
                    {
                        "id": "bitsandbytes",
                        "name": "bitsandbytes",
                        "status": "unknown",
                        "installed": [{"name": "bitsandbytes", "version": "0.46.1"}],
                        "matrix_claim_status": "unverified",
                        "reason": "missing ABI evidence",
                    }
                ],
            }
        )
        self.assertEqual(window.extension_table.rowCount(), 1)
        self.assertEqual(window.extension_table.item(0, 1).text(), "unknown")
        self.assertEqual(window.extension_table.columnCount(), 5)
        self.assertEqual(window.extension_table.horizontalHeaderItem(4).text(), "Evidence")
        self.assertIn("1 known extensions", window.extension_summary.text())
        window.extension_table.selectRow(0)
        self.application.processEvents()
        self.assertIn("missing ABI evidence", window.extension_details.toPlainText())
        window.close()

    def test_activity_shows_summary_before_details(self):
        window = MainWindow()
        window._append_json(
            {"target_package_inventory": {"status": "detected", "compiled_extension_count": 3}}
        )
        self.assertEqual(window.activity_list.count(), 1)
        self.assertEqual(window.output.toPlainText(), "")
        window.activity_list.setCurrentRow(0)
        self.application.processEvents()
        self.assertIn('"compiled_extension_count": 3', window.output.toPlainText())
        window.close()

    def test_service_loads_explicit_local_catalog_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = {
                "schema_version": 1,
                "generated_at": "2026-08-11T00:00:00Z",
                "targets": [],
            }
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps({
                    "schema_version": 1,
                    "artifacts": [{"id": "compatibility_matrix", "path": "matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest()}],
                }),
                encoding="utf-8",
            )
            catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
            (root / "source-manifest.json").write_text(
                json.dumps(
                    {
                        "fetched_at": "2026-08-11T00:00:00Z",
                        "catalog_sha256": catalog_hash,
                        "artifact_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            service = ManagerService()
            loaded, state = service.load_catalog(catalog)
            self.assertIsNone(service.catalog)
            service.commit_catalog((loaded, state))
            self.assertEqual(loaded["targets"], [])
            self.assertEqual(state.path, catalog.resolve())
            self.assertEqual(state.source, "local file")
            self.assertEqual(state.fetched_at, "2026-08-11T00:00:00Z")
            self.assertEqual(state.artifact_count, 4)
            self.assertEqual(state.catalog_sha256, catalog_hash)
            self.assertIsNone(state.source_failure_count)
            self.assertIs(service.catalog, loaded)

    def test_service_rejects_local_catalog_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps({"schema_version": 1, "targets": []}),
                encoding="utf-8",
            )
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifacts": [
                            {
                                "id": "compatibility_matrix",
                                "path": "matrix.json",
                                "schema": "schemas/compatibility-matrix.schema.json",
                                "schema_version": 1,
                                "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "source-manifest.json").write_text(
                json.dumps({"catalog_sha256": "0" * 64}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CatalogError, "catalog hash"):
                ManagerService().load_catalog(catalog)

    def test_direct_matrix_ignores_unrelated_parent_catalog_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            matrix_path = data / "matrix.json"
            matrix_path.write_text(
                json.dumps({"schema_version": 1, "targets": []}),
                encoding="utf-8",
            )
            (root / "source-manifest.json").write_text(
                json.dumps({"catalog_sha256": "0" * 64}),
                encoding="utf-8",
            )

            _loaded, state = ManagerService().load_catalog(matrix_path)

            self.assertEqual(
                state.catalog_sha256,
                hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            )
            self.assertIsNone(state.source_failure_count)

    def test_service_reports_collection_failures_from_catalog_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps({"schema_version": 1, "generated_at": "2026-08-11T00:00:00Z", "targets": []}),
                encoding="utf-8",
            )
            status_path = root / "status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "results": [
                            {"source_id": "therock-github-workflows", "status": "failed"},
                            {"source_id": "therock-releases", "status": "failed"},
                            {"source_id": "stable", "status": "passed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifacts": [
                            {
                                "id": "compatibility_matrix",
                                "path": "matrix.json",
                                "schema": "schemas/compatibility-matrix.schema.json",
                                "schema_version": 1,
                                "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                            },
                            {
                                "id": "collection_status:therock",
                                "path": "status.json",
                                "schema": "schemas/collection-status.schema.json",
                                "schema_version": 1,
                                "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            _loaded, state = ManagerService().load_catalog(catalog)

            self.assertEqual(state.source_failure_count, 2)
            self.assertEqual(
                state.source_failures,
                ("therock-github-workflows", "therock-releases"),
            )

    def test_stale_callback_cannot_commit_service_target(self):
        window = MainWindow()
        service = window.service
        service.commit_target("new-target")
        task = object()
        window._tasks.add(task)
        window._generation = 2

        window._finish_task(
            task,
            "Detect",
            service.commit_target,
            "old-target",
            generation=1,
        )

        self.assertEqual(service.target, "new-target")
        window.close()

    def test_stale_callback_cannot_commit_service_catalog(self):
        window = MainWindow()
        service = window.service
        service.commit_catalog(({"id": "new"}, "new-state"))
        task = object()
        window._tasks.add(task)
        window._generation = 2

        window._finish_task(
            task,
            "Load Matrix catalog",
            service.commit_catalog,
            ({"id": "old"}, "old-state"),
            generation=1,
        )

        self.assertEqual(service.catalog, {"id": "new"})
        self.assertEqual(service.catalog_state, "new-state")
        window.close()

    def test_mutation_failure_is_visible_and_unlocks_after_generation_change(self):
        window = MainWindow()
        task = object()
        window._tasks.add(task)
        window._set_mutation_locked(True)
        window._generation = 3

        window._fail_task(
            task,
            "Apply core packages",
            "subprocess failed",
            generation=None,
            mutation=True,
        )

        self.assertIn("ERROR: subprocess failed", window.output.toPlainText())
        self.assertFalse(window._mutation_in_progress)
        self.assertTrue(window.target_edit.isEnabled())
        self.assertFalse(window.apply_core_button.isEnabled())
        window.close()

    def test_catalog_reload_immediately_clears_candidates_and_plans(self):
        window = MainWindow()
        window.service.catalog = {"old": True}
        window._candidates = [{"id": "old"}]
        window._candidate = {"id": "old"}
        window._core_plan = object()
        window._extension_plan_result = object()
        window.apply_core_button.setEnabled(True)
        window.apply_extension_button.setEnabled(True)

        with patch.object(window, "_run") as run:
            window._load_catalog(False)

        self.assertIsNone(window.service.catalog)
        self.assertEqual(window._candidates, [])
        self.assertIsNone(window._candidate)
        self.assertIsNone(window._core_plan)
        self.assertIsNone(window._extension_plan_result)
        self.assertFalse(window.apply_core_button.isEnabled())
        self.assertFalse(window.apply_extension_button.isEnabled())
        run.assert_called_once()
        window.close()

    def test_extracts_unique_normalized_gfx_targets(self):
        observation = RuntimeObservation(
            target_root=Path("target"),
            python_executable=None,
            host_platform="windows",
            runtime_status="detected",
            hardware_status="detected",
            devices=(
                {"gfx": "gfx1201:sramecc+"},
                {"gfx": "GFX1201"},
                {"gfx": "gfx1100"},
            ),
        )
        self.assertEqual(detected_gfx_targets(observation), ("gfx1201", "gfx1100"))

    def test_runtime_probe_populates_single_gfx(self):
        window = MainWindow()
        observation = RuntimeObservation(
            target_root=Path("target"),
            python_executable=None,
            host_platform="windows",
            runtime_status="detected",
            hardware_status="detected",
            devices=({"gfx": "gfx1201"},),
        )
        window._display_runtime(observation)
        self.assertEqual(window.gfx_edit.currentText(), "gfx1201")
        self.assertEqual(window.gfx_edit.count(), 1)
        self.assertIn("gfx1201", window.gfx_status.text())
        window.close()

    def test_host_probe_is_marked_provisional(self):
        window = MainWindow()
        from rocm_stack_manager.core.hardware import HardwareObservation

        window._display_host_hardware(
            HardwareObservation(
                scope="host",
                host_platform="windows",
                status="detected",
                gfx_targets=("gfx1201",),
            )
        )
        self.assertEqual(window.gfx_edit.currentText(), "gfx1201")
        self.assertIn("Host detected", window.gfx_status.text())
        window._display_runtime(
            RuntimeObservation(
                target_root=Path("target"),
                python_executable=None,
                host_platform="windows",
                runtime_status="detected",
                hardware_status="detected",
                devices=({"gfx": "gfx1201"},),
            ),
            append=False,
        )
        self.assertIn("Target runtime", window.gfx_status.text())
        window.close()

    def test_runtime_probe_keeps_multiple_gfx_unselected(self):
        window = MainWindow()
        observation = RuntimeObservation(
            target_root=Path("target"),
            python_executable=None,
            host_platform="linux",
            runtime_status="detected",
            hardware_status="detected",
            devices=({"gfx": "gfx1201"}, {"gfx": "gfx1100"}),
        )
        window._display_runtime(observation)
        self.assertEqual(window.gfx_edit.currentText(), "")
        self.assertEqual(
            [window.gfx_edit.itemText(index) for index in range(window.gfx_edit.count())],
            ["gfx1201", "gfx1100"],
        )
        self.assertIn("2 GFX", window.gfx_status.text())
        window.close()

    def test_inspection_labels_runtime_and_hardware_sources_separately(self):
        from rocm_stack_manager.core.hardware import HardwareObservation

        window = MainWindow()
        window._display_inspection(
            (
                None,
                HardwareObservation(
                    scope="target-runtime",
                    host_platform="windows",
                    status="detected",
                    gfx_targets=("gfx1201",),
                ),
                "target Python unavailable",
            )
        )

        self.assertIn("Target hardware", window.gfx_status.text())
        self.assertNotIn("Target runtime", window.gfx_status.text())
        window.close()

    def test_runtime_probe_allows_manual_gfx_when_missing(self):
        window = MainWindow()
        observation = RuntimeObservation(
            target_root=Path("target"),
            python_executable=None,
            host_platform="windows",
            runtime_status="not_detected",
            hardware_status="not_detected",
            devices=(),
        )
        window._display_runtime(observation)
        window.gfx_edit.setEditText("gfx900")
        self.assertEqual(window.gfx_edit.currentText(), "gfx900")
        self.assertIn("manual", window.gfx_status.text())
        window.close()
