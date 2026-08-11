import importlib.util
import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path


PY_SIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PY_SIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from rocm_stack_manager.core.detection import TargetLayout
    from rocm_stack_manager.core.verify import RuntimeObservation
    from rocm_stack_manager.ui.main_window import MainWindow
    from rocm_stack_manager.ui.services import ManagerService, detected_gfx_targets


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
            (root / "source-manifest.json").write_text(
                json.dumps(
                    {
                        "fetched_at": "2026-08-11T00:00:00Z",
                        "catalog_sha256": "abc123",
                        "artifact_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            service = ManagerService()
            loaded, state = service.load_catalog(catalog)
            self.assertEqual(loaded["targets"], [])
            self.assertEqual(state.path, catalog.resolve())
            self.assertEqual(state.source, "local file")
            self.assertEqual(state.fetched_at, "2026-08-11T00:00:00Z")
            self.assertEqual(state.artifact_count, 4)

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
