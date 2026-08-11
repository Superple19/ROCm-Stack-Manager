import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.core.backup import (
    BackupError,
    BackupSnapshot,
    ExtensionBackupSnapshot,
    create_backup,
    create_extension_backup,
    load_backup,
    load_extension_backup,
    migrate_backup,
)
from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.install import (
    InstallationError,
    apply_extension_restore,
    apply_install,
    apply_restore,
    build_extension_restore_plan,
    build_restore_plan,
)
from rocm_stack_manager.cli import parse_args
from rocm_stack_manager.ui.services import ManagerService


class BackupAndApplyTests(unittest.TestCase):
    def _target(self, root):
        (root / "ComfyUI").mkdir()
        (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
        python = root / "python_env" / "python.exe"
        python.parent.mkdir()
        python.write_text("", encoding="utf-8")
        return detect_target(root)

    def test_backup_writes_freeze_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self._target(root)
            completed = type("Completed", (), {"returncode": 0, "stdout": "torch==2.12.0\n", "stderr": ""})()
            with patch("rocm_stack_manager.core.backup.subprocess.run", return_value=completed):
                backup = create_backup(target, root / "backups")

            document = json.loads(backup.path.read_text(encoding="utf-8"))
            self.assertEqual(document["requirements"], ["torch==2.12.0"])
            self.assertTrue(backup.requirements_path.is_file())
            self.assertFalse(backup.path.with_suffix(".json.tmp").exists())

    def test_apply_requires_explicit_unverified_approval(self):
        candidate = {
            "id": "legacy",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "resolver_status": "resolver_failed",
            "wheel_urls": ["https://example.test/torch.whl"],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            backup = BackupSnapshot(
                Path(directory) / "backup.json",
                Path(directory) / "requirements.txt",
                "2026-01-01T00:00:00Z",
                ("torch==2.12.0",),
            )
            with self.assertRaises(InstallationError):
                apply_install(target, candidate, backup)

    def test_apply_uses_backed_up_target_python(self):
        candidate = {
            "id": "candidate",
            "artifact_available": True,
            "python_compatibility": "compatible",
            "wheel_urls": ["https://example.test/torch.whl"],
        }
        completed = type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            backup = BackupSnapshot(
                Path(directory) / "backup.json",
                Path(directory) / "requirements.txt",
                "2026-01-01T00:00:00Z",
                ("torch==2.12.0",),
            )
            with patch("rocm_stack_manager.core.install.subprocess.run", return_value=completed) as run:
                result = apply_install(target, candidate, backup)

        self.assertTrue(result.applied)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(run.call_args.args[0][0], str(target.python_executable))

    def test_restore_plan_uses_recorded_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self._target(root)
            requirements = root / "requirements.txt"
            requirements.write_text("torch==2.12.0\n", encoding="utf-8")
            backup = BackupSnapshot(root / "backup.json", requirements, "2026-01-01T00:00:00Z", ("torch==2.12.0",))

            plan = build_restore_plan(target, backup)

            self.assertIn("--force-reinstall", plan.command)
            self.assertIn(str(requirements), plan.command)
            self.assertTrue(any("artifact bytes" in warning for warning in plan.warnings))

    def test_restore_rejects_backup_from_another_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir()
            target = self._target(root / "target")
            requirements = root / "requirements.txt"
            requirements.write_text("torch==2.12.0\n", encoding="utf-8")
            backup = BackupSnapshot(
                root / "backup.json",
                requirements,
                "2026-01-01T00:00:00Z",
                ("torch==2.12.0",),
                target_root=str(root / "other-target"),
            )

            with self.assertRaises(InstallationError):
                build_restore_plan(target, backup)

    def test_load_backup_rejects_legacy_schema_without_materializing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup_path = root / "backup.json"
            backup_path.write_text(
                json.dumps({"created_at": "2026-01-01T00:00:00Z", "requirements": ["torch==2.12.0"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BackupError, "migrate-backup"):
                load_backup(backup_path)
            self.assertFalse((root / "backup.txt").exists())

    def test_read_only_backup_load_does_not_create_missing_requirements_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup_path = root / "backup.json"
            backup_path.write_text(
                json.dumps({"created_at": "2026-01-01T00:00:00Z", "requirements": ["torch==2.12.0"]}),
                encoding="utf-8",
            )

            requirements_path = root / "backup.txt"
            with self.assertRaisesRegex(BackupError, "migrate-backup"):
                load_backup(backup_path, materialize=False)
            self.assertFalse(requirements_path.exists())

    def test_ui_extension_restore_dry_run_does_not_materialize_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir()
            target = self._target(root / "target")
            backup_path = root / "extensions.json"
            backup_path.write_text(
                json.dumps({
                    "kind": "extensions",
                    "requirements": ["bitsandbytes==0.50.0"],
                    "requirements_path": "extensions.txt",
                }),
                encoding="utf-8",
            )
            service = ManagerService()
            service.target = target
            with self.assertRaises(BackupError):
                service.restore_extensions(backup_path)
            self.assertFalse((root / "extensions.txt").exists())

    def test_load_backup_resolves_relative_requirements_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup_path = root / "backup.json"
            (root / "recorded.txt").write_text("torch==2.12.0\n", encoding="utf-8")
            backup_path.write_text(
                json.dumps(
                    {
                        "created_at": "2026-01-01T00:00:00Z",
                        "requirements_path": "recorded.txt",
                        "requirements": ["torch==2.12.0"],
                        "schema_version": 1,
                        "requirements_sha256": hashlib.sha256(
                            (root / "recorded.txt").read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            backup = load_backup(backup_path)

            self.assertEqual(backup.requirements_path, root / "recorded.txt")

    def test_load_backup_rejects_external_requirements_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup_path = root / "backup.json"
            backup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "requirements": ["torch==2.12.0"],
                        "requirements_path": "../outside.txt",
                        "requirements_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(BackupError):
                load_backup(backup_path)

    def test_load_backup_rejects_modified_hashed_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements_path = root / "backup.txt"
            requirements_path.write_text("torch==2.12.0\n", encoding="utf-8")
            backup_path = root / "backup.json"
            backup_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "requirements": ["torch==2.12.0"],
                    "requirements_path": requirements_path.name,
                    "requirements_sha256": "0" * 64,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(BackupError):
                load_backup(backup_path)

    def test_apply_restore_uses_target_python(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self._target(root)
            requirements = root / "requirements.txt"
            requirements.write_text("torch==2.12.0\n", encoding="utf-8")
            backup = BackupSnapshot(root / "backup.json", requirements, "2026-01-01T00:00:00Z", ("torch==2.12.0",))
            with patch("rocm_stack_manager.core.install.subprocess.run", return_value=completed) as run:
                result = apply_restore(target, backup)

            self.assertTrue(result.applied)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(run.call_args.args[0][0], str(target.python_executable))

    def test_rollback_is_restore_alias(self):
        args = parse_args(["rollback", "--target", "target", "--backup", "backup.json"])

        self.assertEqual(args.command, "rollback")
        self.assertFalse(args.apply)

    def test_extension_backup_filters_to_selected_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self._target(root)
            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "torch==2.14.0\nbitsandbytes==0.50.0\nsageattention==1.0.6\n",
                    "stderr": "",
                },
            )()
            with patch("rocm_stack_manager.core.backup.subprocess.run", return_value=completed):
                backup = create_extension_backup(
                    target,
                    ("bitsandbytes",),
                    extension_ids=("bitsandbytes",),
                    candidate_id="candidate",
                    destination=root / "backups",
                )

            self.assertEqual(backup.requirements, ("bitsandbytes==0.50.0",))
            document = backup.path.read_text(encoding="utf-8")
            self.assertIn('"kind": "extensions"', document)
            self.assertEqual(load_extension_backup(backup.path).extension_ids, ("bitsandbytes",))

    def test_empty_extension_backup_has_no_restore_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "extensions.json"
            requirements = root / "extensions.txt"
            requirements.write_text("", encoding="utf-8")
            path.write_text(
                json.dumps(
                    {
                        "kind": "extensions",
                        "schema_version": 1,
                        "requirements": [],
                        "requirements_path": "extensions.txt",
                        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            (root / "target").mkdir()
            target = self._target(root / "target")

            plan = build_extension_restore_plan(target, load_extension_backup(path))

            self.assertEqual(plan.command, ())
            self.assertTrue(any("nothing to restore" in warning for warning in plan.warnings))

    def test_extension_restore_uses_target_python(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir()
            target = self._target(root / "target")
            requirements = root / "extensions.txt"
            requirements.write_text("bitsandbytes==0.50.0\n", encoding="utf-8")
            backup = ExtensionBackupSnapshot(
                root / "extensions.json",
                requirements,
                "2026-01-01T00:00:00Z",
                ("bitsandbytes==0.50.0",),
                target_root=str(target.root),
            )
            with patch("rocm_stack_manager.core.install.subprocess.run", return_value=completed) as run:
                result = apply_extension_restore(target, backup)

            self.assertEqual(result.returncode, 0)
            self.assertEqual(run.call_args.args[0][0], str(target.python_executable))

    def test_extension_cli_actions_are_explicit(self):
        args = parse_args(["extensions", "apply", "--target", "target", "--apply"])

        self.assertEqual(args.action, "apply")
        self.assertTrue(args.apply)

    def test_migrate_backup_cli_requires_distinct_output(self):
        args = parse_args(
            ["migrate-backup", "--backup", "legacy.json", "--output", "current.json"]
        )

        self.assertEqual(args.command, "migrate-backup")
        self.assertEqual(args.backup, Path("legacy.json"))
        self.assertEqual(args.output, Path("current.json"))

    def test_migrate_backup_writes_current_schema_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps(
                    {
                        "created_at": "2026-01-01T00:00:00Z",
                        "requirements": ["torch==2.12.0"],
                        "target_root": str(root / "target"),
                    }
                ),
                encoding="utf-8",
            )
            migrated = migrate_backup(legacy, root / "migrated.json")

            document = json.loads(migrated.path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(
                document["requirements_sha256"],
                hashlib.sha256(migrated.requirements_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(load_backup(migrated.path).requirements, ("torch==2.12.0",))
            self.assertTrue(legacy.exists())

    def test_migrate_backup_does_not_overwrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.json"
            legacy.write_text(json.dumps({"requirements": ["torch==2.12.0"]}), encoding="utf-8")

            with self.assertRaisesRegex(BackupError, "different from the source"):
                migrate_backup(legacy, legacy)

    def test_current_backup_requires_hash_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "backup.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "requirements": ["torch==2.12.0"],
                        "requirements_path": "backup.txt",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BackupError, "hash is missing"):
                load_backup(path)
