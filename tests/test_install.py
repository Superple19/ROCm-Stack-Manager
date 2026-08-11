import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.backup import BackupSnapshot
from rocm_stack_manager.core.install import InstallationError, apply_install, build_install_plan, dry_run_install
from rocm_stack_manager.core.planning import PlanningError, validate_plan_binding
from rocm_stack_manager.core.platforms import UnsupportedPlatformError


class InstallPlanTests(unittest.TestCase):
    def _target(self, root):
        (root / "ComfyUI").mkdir()
        (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
        python = root / "python_env" / "python.exe"
        python.parent.mkdir()
        python.write_text("", encoding="utf-8")
        return detect_target(root)

    def _staged(self, root):
        artifacts = root / "staging" / "wheelhouse"
        artifacts.mkdir(parents=True)
        wheel = artifacts / "torch-2.12.0-cp312-cp312-win_amd64.whl"
        wheel.write_bytes(b"test-wheel")
        requirements = artifacts.parent / "requirements-hashed.txt"
        import hashlib
        requirements.write_text(
            f"torch==2.12.0 --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n",
            encoding="utf-8",
        )
        manifest = artifacts.parent / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                    "files": [{"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            root=artifacts.parent,
            artifacts_path=artifacts,
            requirements_path=requirements,
            manifest_path=manifest,
            files=(
                {
                    "filename": wheel.name,
                    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                },
            ),
            install_command=("--no-index", "--find-links", str(artifacts), "--require-hashes", "--requirement", str(requirements)),
        )

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
        self.assertTrue(any("--allow-unverified" in warning for warning in result.plan.warnings))

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
        self.assertIn("--extra-index-url", plan.command)
        self.assertNotIn("--index-url", plan.command)
        self.assertIn("torch==2.12.0", plan.command)

    def test_warns_when_torchaudio_is_not_included(self):
        candidate = {
            "id": "therock:windows:nightly:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "package_specs": ["torch==2.14.0"],
            "index_url": "https://repo.example.test/",
        }

        with tempfile.TemporaryDirectory() as directory:
            result = dry_run_install(self._target(Path(directory)), candidate)

        self.assertTrue(any("torchaudio is not included" in warning for warning in result.plan.warnings))

    def test_does_not_warn_when_torchaudio_is_included(self):
        candidate = {
            "id": "legacy:windows:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "wheel_urls": ["https://example.test/torchaudio-2.9.1.whl"],
        }

        with tempfile.TemporaryDirectory() as directory:
            result = dry_run_install(self._target(Path(directory)), candidate)

        self.assertFalse(any("torchaudio is not included" in warning for warning in result.plan.warnings))

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

    def test_plan_records_target_and_candidate_binding(self):
        candidate = {
            "id": "therock:windows:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "gfx": "gfx1201",
            "package_specs": ["torch==2.12.0"],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            plan = build_install_plan(
                target,
                candidate,
                catalog_hash="a" * 64,
                adapter_id="comfyui",
            )

        self.assertEqual(plan.binding["target_root"], str(target.root))
        self.assertEqual(plan.binding["target_python"], str(target.python_executable))
        self.assertEqual(plan.binding["target_gfx"], "gfx1201")
        self.assertEqual(plan.binding["catalog_hash"], "a" * 64)
        self.assertEqual(plan.binding["adapter_id"], "comfyui")

    def test_apply_rejects_plan_for_different_target(self):
        candidate = {
            "id": "therock:windows:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "gfx": "gfx1201",
            "wheel_urls": ["https://example.test/torch.whl"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            target_a = self._target(root / "a")
            target_b = self._target(root / "b")
            plan = build_install_plan(target_a, candidate, adapter_id="comfyui")
            backup = BackupSnapshot(root / "backup.json", root / "requirements.txt", "", ())

            with self.assertRaises(PlanningError):
                apply_install(target_b, candidate, backup, plan=plan)

    def test_plan_rejects_changed_catalog_binding(self):
        candidate = {
            "id": "therock:windows:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "gfx": "gfx1201",
            "package_specs": ["torch==2.12.0"],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            plan = build_install_plan(target, candidate, catalog_hash="a" * 64)

            with self.assertRaises(PlanningError):
                validate_plan_binding(plan, target, candidate, catalog_hash="b" * 64)

    def test_apply_rejects_cross_platform_candidate(self):
        candidate = {
            "id": "therock:linux:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "platform": "linux",
            "gfx": "gfx1201",
            "package_specs": ["torch==2.12.0"],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            backup = BackupSnapshot(Path(directory) / "backup.json", Path(directory) / "requirements.txt", "", ())

            with self.assertRaises(InstallationError):
                apply_install(target, candidate, backup)

    def test_apply_rejects_unsupported_host_platform(self):
        candidate = {
            "id": "therock:windows:stable:gfx1201",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "platform": "windows",
            "gfx": "gfx1201",
            "package_specs": ["torch==2.12.0"],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            backup = BackupSnapshot(Path(directory) / "backup.json", Path(directory) / "requirements.txt", "", ())
            with patch(
                "rocm_stack_manager.core.install.require_supported_host_platform",
                side_effect=UnsupportedPlatformError("unsupported host platform"),
            ):
                with self.assertRaisesRegex(InstallationError, "unsupported host platform"):
                    apply_install(target, candidate, backup)

    def test_resolver_failure_blocks_cleanup_and_install(self):
        candidate = {
            "id": "candidate",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "wheel_urls": ["https://example.test/torch.whl"],
        }
        failed = type("Resolver", (), {"status": "resolver_failed", "binding": {}})()
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            backup = BackupSnapshot(Path(directory) / "backup.json", Path(directory) / "requirements.txt", "", ())
            with patch("rocm_stack_manager.core.install.collect_inventory") as inventory:
                with patch("rocm_stack_manager.core.install.subprocess.run") as run:
                    with self.assertRaisesRegex(InstallationError, "fresh target-bound resolver"):
                        apply_install(
                            target,
                            candidate,
                            backup,
                            resolver_result=failed,
                        )
            inventory.assert_not_called()
            run.assert_not_called()

    def test_artifact_only_candidate_is_rejected(self):
        candidate = {
            "id": "therock:nightly:10.1.0:artifact-only",
            "artifact_available": True,
            "candidate_kind": "artifact_only",
            "python_compatibility": "compatible",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InstallationError):
                build_install_plan(self._target(Path(directory)), candidate)

    def test_apply_removes_stale_managed_packages_before_install(self):
        candidate = {
            "id": "legacy:stable:7.2.1",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "wheel_urls": [
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
            ],
        }
        completed = type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        inventory = type(
            "Inventory",
            (),
            {
                "status": "detected",
                "packages": (
                    {"name": "rocm", "version": "10.1.0"},
                    {"name": "torch", "version": "2.14.0"},
                    {"name": "amd-torch-device-gfx1201", "version": "2.14.0"},
                ),
            },
        )()
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            backup = BackupSnapshot(Path(directory) / "backup.json", Path(directory) / "requirements.txt", "", ())
            with patch("rocm_stack_manager.core.install.collect_inventory", return_value=inventory):
                with patch("rocm_stack_manager.core.install.stage_candidate", return_value=self._staged(Path(directory))):
                    with patch("rocm_stack_manager.core.install.subprocess.run", return_value=completed) as run:
                        result = apply_install(target, candidate, backup)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(run.call_args_list[0].args[0][3], "install")
        self.assertIn("--dry-run", run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[1].args[0][3], "uninstall")
        self.assertIn("amd-torch-device-gfx1201", run.call_args_list[1].args[0])
        self.assertEqual(run.call_args_list[2].args[0][3], "install")
        self.assertIn("Removed stale packages", result.output)
