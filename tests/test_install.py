import tempfile
import unittest
from pathlib import Path

from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.install import InstallationError, build_install_plan, dry_run_install


class InstallPlanTests(unittest.TestCase):
    def _target(self, root):
        (root / "ComfyUI").mkdir()
        (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
        python = root / "python_env" / "python.exe"
        python.parent.mkdir()
        python.write_text("", encoding="utf-8")
        return detect_target(root)

    def test_historical_wheels_create_warning_only_dry_run(self):
        candidate = {
            "id": "legacy:stable:7.2.1",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "resolver_status": "resolver_failed",
            "wheel_urls": ["https://example.test/torch.whl"],
            "distribution_family": "legacy",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = dry_run_install(self._target(Path(directory)), candidate)

        self.assertFalse(result.applied)
        self.assertIn("--no-input", result.plan.command)
        self.assertIn("resolver evidence is failed", result.plan.warnings[0])
        self.assertIn("--allow-unverified", result.plan.warnings[-1])

    def test_current_candidate_uses_target_python_and_index(self):
        candidate = {
            "id": "therock:windows:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "package_specs": ["torch==2.12.0"],
            "index_url": "https://repo.example.test/",
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            plan = build_install_plan(target, candidate)

        self.assertEqual(plan.command[0], str(target.python_executable))
        self.assertIn("--index-url", plan.command)
        self.assertIn("torch==2.12.0", plan.command)

    def test_incompatible_candidate_is_rejected(self):
        candidate = {
            "id": "candidate",
            "artifact_available": True,
            "python_compatibility": "incompatible",
            "wheel_urls": ["https://example.test/torch.whl"],
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InstallationError):
                build_install_plan(self._target(Path(directory)), candidate)
