import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.inventory import classify_packages, collect_inventory


class InventoryTests(unittest.TestCase):
    def test_classifies_core_compiled_and_pure_python_packages(self):
        packages = classify_packages(
            (
                {"name": "torch", "version": "2.12.0", "requires": [], "compiled_files": []},
                {"name": "bitsandbytes", "version": "0.46.1", "requires": [], "compiled_files": ["lib.dll"]},
                {"name": "custom-node", "version": "1.0", "requires": ["torch==2.11.0"], "compiled_files": []},
                {"name": "optional-node", "version": "1.0", "requires": ['torch>=2.6; extra == "torch"'], "compiled_files": []},
                {"name": "requests", "version": "2.32.0", "requires": [], "compiled_files": []},
            ),
            {"torch_version": "2.12.0"},
        )

        statuses = {package["name"]: package["status"] for package in packages}

        self.assertEqual(statuses["torch"], "compatible")
        self.assertEqual(statuses["bitsandbytes"], "unknown")
        self.assertEqual(statuses["custom-node"], "conflict")
        self.assertEqual(statuses["optional-node"], "compatible")
        self.assertEqual(statuses["requests"], "compatible")

    def test_collects_inventory_from_target_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ComfyUI").mkdir()
            (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
            python = root / "venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            target = detect_target(root)
            payload = {"packages": [{"name": "torch", "version": "2.12.0", "requires": [], "compiled_files": []}]}
            completed = type("Completed", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()

            with patch("rocm_stack_manager.core.inventory.subprocess.run", return_value=completed):
                inventory = collect_inventory(target, {"torch_version": "2.12.0"})

            self.assertEqual(inventory.status, "detected")
            self.assertEqual(inventory.packages[0]["status"], "compatible")
