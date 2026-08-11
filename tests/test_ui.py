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
    from PySide6 import QtWidgets

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
        self.assertEqual(window.extension_table.columnCount(), 10)
        self.assertEqual(window.extension_table.horizontalHeaderItem(5).text(), "Artifact platform")
        self.assertIn("1 known extensions", window.extension_summary.text())
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
