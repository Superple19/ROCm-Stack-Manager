import tempfile
import unittest
from pathlib import Path

from rocm_stack_manager.core.detection import detect_target


class TargetDetectionTests(unittest.TestCase):
    def test_detects_portable_root_without_an_absolute_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ComfyUI").mkdir()
            (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
            (root / "python_embeded").mkdir()
            (root / "python_embeded" / "python.exe").write_text("", encoding="utf-8")
            (root / "run_amd_gpu.bat").write_text("", encoding="utf-8")

            target = detect_target(root)

            self.assertEqual(target.root, root.resolve())
            self.assertEqual(target.comfyui_dir, (root / "ComfyUI").resolve())
            self.assertEqual(target.layout, "portable")
            self.assertEqual(target.python_executable, (root / "python_embeded" / "python.exe").resolve())
            self.assertEqual([path.name for path in target.launchers], ["run_amd_gpu.bat"])

    def test_accepts_nested_comfyui_directory_and_venv_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfyui = root / "ComfyUI"
            comfyui.mkdir()
            (comfyui / "main.py").write_text("", encoding="utf-8")
            (root / ".venv" / "Scripts").mkdir(parents=True)
            python = root / ".venv" / "Scripts" / "python.exe"
            python.write_text("", encoding="utf-8")

            target = detect_target(comfyui)

            self.assertEqual(target.root, root.resolve())
            self.assertEqual(target.layout, "venv")
            self.assertEqual(target.python_executable, python.resolve())

    def test_accepts_python_env_root_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("", encoding="utf-8")
            python = root / "python_env" / "python.exe"
            python.parent.mkdir()
            python.write_text("", encoding="utf-8")

            target = detect_target(root)

            self.assertEqual(target.layout, "venv")
            self.assertEqual(target.python_executable, python.resolve())
