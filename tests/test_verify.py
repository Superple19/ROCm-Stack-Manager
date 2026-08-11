import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.verify import probe_target, target_python_tag


class RuntimeProbeTests(unittest.TestCase):
    def _target(self, root):
        (root / "ComfyUI").mkdir()
        (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
        (root / "venv" / "Scripts").mkdir(parents=True)
        python = root / "venv" / "Scripts" / "python.exe"
        python.write_text("", encoding="utf-8")
        return detect_target(root)

    def test_probe_is_target_scoped_and_preserves_hip_version(self):
        payload = {
            "python_version": "3.12.10",
            "torch_version": "2.12.0+rocm7.14.0",
            "torch_rocm_tag": "7.14.0",
            "hip_version": "7.15.26290",
            "torch_file": "C:/target/Lib/site-packages/torch/__init__.py",
            "rocm_packages": {"rocm-sdk": "7.14.0"},
            "cuda_available": True,
            "device_count": 1,
            "devices": [{"index": 0, "name": "AMD Radeon RX 9070 XT", "gfx": "gfx1201"}],
            "tensor_smoke_status": "passed",
            "tensor_smoke_error": None,
            "torch_error": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            completed = type("Completed", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
            with patch("rocm_stack_manager.core.verify.subprocess.run", return_value=completed) as run:
                observation = probe_target(target)

            self.assertEqual(observation.as_dict()["runtime_scope"], "target")
            self.assertEqual(observation.hip_version, "7.15.26290")
            self.assertEqual(observation.torch_rocm_tag, "7.14.0")
            self.assertEqual(observation.rocm_packages["rocm-sdk"], "7.14.0")
            self.assertEqual(observation.devices[0]["gfx"], "gfx1201")
            self.assertEqual(observation.tensor_smoke_status, "passed")
            self.assertIsNone(observation.tensor_smoke_error)
            self.assertEqual(run.call_args.args[0][0], str(target.python_executable))

    def test_tensor_smoke_failure_is_kept_separate_from_runtime_detection(self):
        payload = {
            "python_version": "3.12.10",
            "torch_version": "2.12.0+rocm7.14.0",
            "cuda_available": True,
            "device_count": 1,
            "devices": [{"index": 0, "gfx": "gfx1201"}],
            "tensor_smoke_status": "failed",
            "tensor_smoke_error": "RuntimeError: kernel failed",
            "torch_error": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            completed = type("Completed", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
            with patch("rocm_stack_manager.core.verify.subprocess.run", return_value=completed):
                observation = probe_target(target)

        self.assertEqual(observation.runtime_status, "detected")
        self.assertEqual(observation.hardware_status, "detected")
        self.assertEqual(observation.tensor_smoke_status, "failed")
        self.assertIn("kernel failed", observation.tensor_smoke_error)

    def test_missing_target_python_does_not_use_global_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ComfyUI").mkdir()
            (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
            target = detect_target(root)

            observation = probe_target(target)

            self.assertEqual(observation.runtime_status, "not_detected")
            self.assertEqual(observation.hardware_status, "not_detected")
            self.assertIn("target Python", observation.error)

    def test_reads_python_tag_from_target_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            completed = type("Completed", (), {"returncode": 0, "stdout": "cp312\n", "stderr": ""})()
            with patch("rocm_stack_manager.core.verify.subprocess.run", return_value=completed):
                self.assertEqual(target_python_tag(target), "cp312")
