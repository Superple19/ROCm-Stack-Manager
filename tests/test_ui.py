import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


PY_SIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PY_SIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from rocm_stack_manager.core.detection import TargetLayout
    from rocm_stack_manager.ui.main_window import MainWindow
    from rocm_stack_manager.ui.services import ManagerService


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

    def test_service_loads_explicit_local_catalog_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = {
                "targets": [],
            }
            (root / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps({
                    "artifacts": [{"id": "compatibility_matrix", "path": "matrix.json"}],
                }),
                encoding="utf-8",
            )
            service = ManagerService()
            loaded, state = service.load_catalog(catalog)
            self.assertEqual(loaded["targets"], [])
            self.assertEqual(state.path, catalog.resolve())
            self.assertEqual(state.source, "local file")
