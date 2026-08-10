import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.adapters.comfyui.extensions import (
    ExtensionInstallResult,
    ExtensionPlan,
    apply_extension_plan,
    build_extension_plan,
    build_extension_report,
)
from rocm_stack_manager.core.inventory import PackageInventory


class ExtensionReportTests(unittest.TestCase):
    def test_compiled_extensions_remain_unknown(self):
        inventory = PackageInventory(
            Path("C:/target"),
            Path("C:/target/python.exe"),
            (
                {
                    "name": "bitsandbytes",
                    "version": "0.50.0",
                    "status": "unknown",
                    "compiled_files": ["bitsandbytes/libbitsandbytes_rocm.dll"],
                },
            ),
            "detected",
        )

        report = build_extension_report(inventory)
        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")

        self.assertEqual(bitsandbytes["status"], "unknown")
        self.assertEqual(bitsandbytes["claim_status"], "unverified")
        self.assertFalse(report["network_access"])
        self.assertFalse(report["installation_performed"])

    def test_missing_extensions_are_not_reported_compatible(self):
        report = build_extension_report(PackageInventory(Path("C:/target"), None, (), "detected"))

        self.assertTrue(all(item["status"] == "not_installed" for item in report["extensions"]))
        self.assertTrue(all(item["claim_status"] == "unverified" for item in report["extensions"]))

    def test_core_conflict_is_preserved_for_extension(self):
        inventory = PackageInventory(
            Path("C:/target"),
            None,
            (
                {
                    "name": "triton-windows",
                    "version": "3.7.0",
                    "status": "conflict",
                    "compiled_files": ["triton/_C.pyd"],
                },
            ),
            "detected",
        )

        report = build_extension_report(inventory)
        triton = next(item for item in report["extensions"] if item["id"] == "triton")

        self.assertEqual(triton["status"], "conflict")

    def test_installed_extension_without_conflict_is_still_unverified(self):
        inventory = PackageInventory(
            Path("C:/target"),
            None,
            ({"name": "sageattention", "version": "1.0.6", "status": "compatible"},),
            "detected",
        )

        report = build_extension_report(inventory)
        sageattention = next(item for item in report["extensions"] if item["id"] == "sageattention")

        self.assertEqual(sageattention["status"], "unknown")
        self.assertEqual(sageattention["claim_status"], "unverified")

    def test_matrix_extension_profile_metadata_is_preserved(self):
        inventory = PackageInventory(Path("C:/target"), None, (), "detected")
        profile = {
            "id": "comfyui.bitsandbytes",
            "metadata": {"core_required": False, "status": "unverified"},
            "evidence_refs": ["source:test"],
            "constraints": [{"claim_status": "unverified"}],
        }

        report = build_extension_report(inventory, {"bitsandbytes": profile})
        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")

        self.assertEqual(bitsandbytes["matrix_profile_id"], "comfyui.bitsandbytes")
        self.assertEqual(bitsandbytes["matrix_claim_status"], "unverified")
        self.assertEqual(bitsandbytes["matrix_evidence_refs"], ["source:test"])

    def test_unverified_profiles_cannot_create_install_commands(self):
        target = type("Target", (), {"root": Path("C:/target"), "python_executable": Path("C:/target/python.exe")})()
        inventory = PackageInventory(Path("C:/target"), target.python_executable, (), "detected")
        candidate = {
            "id": "therock:windows:nightly:gfx1201:10.1.0:2.14.0",
            "artifact_available": True,
            "candidate_kind": "installable",
            "platform": "windows",
            "gfx": "gfx1201",
            "python_tag": "cp312",
            "torch_version": "2.14.0",
            "rocm_version": "10.1.0",
        }

        plan = build_extension_plan(target, inventory, candidate)

        self.assertTrue(all(item["status"] == "unverified" for item in plan.extensions))
        self.assertEqual(plan.commands, ())

    def test_exact_evidence_and_constraints_make_extension_installable(self):
        target = type("Target", (), {"root": Path("C:/target"), "python_executable": Path("C:/target/python.exe")})()
        inventory = PackageInventory(Path("C:/target"), target.python_executable, (), "detected")
        candidate = {
            "id": "therock:windows:nightly:gfx1201:10.1.0:2.14.0",
            "artifact_available": True,
            "candidate_kind": "installable",
            "platform": "windows",
            "gfx": "gfx1201",
            "python_tag": "cp312",
            "torch_version": "2.14.0",
            "rocm_version": "10.1.0",
        }
        profile = {
            "id": "comfyui.bitsandbytes",
            "metadata": {"status": "artifact_available"},
            "evidence_refs": ["artifact:bitsandbytes"],
            "constraints": [
                {
                    "kind": "extension",
                    "claim_status": "artifact_available",
                    "evidence_refs": ["artifact:bitsandbytes"],
                    "value": {
                        "install_sources": ["https://example.test/bitsandbytes.whl"],
                        "supported_os": ["windows"],
                        "python_tags": ["cp312"],
                        "gfx_targets": ["gfx1201"],
                        "torch_rocm_hip_abi": {
                            "torch": "2.14.0",
                            "rocm": "10.1.0",
                        },
                    },
                }
            ],
        }

        plan = build_extension_plan(
            target,
            inventory,
            candidate,
            {"bitsandbytes": profile},
            ("bitsandbytes",),
        )

        self.assertEqual(plan.extensions[0]["status"], "installable")
        self.assertEqual(plan.commands[0]["extension_id"], "bitsandbytes")
        self.assertIn("https://example.test/bitsandbytes.whl", plan.commands[0]["command"])

    def test_extension_constraint_mismatch_blocks_install(self):
        target = type("Target", (), {"root": Path("C:/target"), "python_executable": Path("C:/target/python.exe")})()
        inventory = PackageInventory(Path("C:/target"), target.python_executable, (), "detected")
        candidate = {
            "id": "candidate",
            "artifact_available": True,
            "candidate_kind": "installable",
            "platform": "windows",
            "gfx": "gfx1201",
            "python_tag": "cp312",
            "torch_version": "2.14.0",
            "rocm_version": "10.1.0",
        }
        profile = {
            "id": "comfyui.bitsandbytes",
            "evidence_refs": ["artifact:test"],
            "constraints": [
                {
                    "kind": "extension",
                    "claim_status": "artifact_available",
                    "value": {
                        "install_sources": ["https://example.test/bitsandbytes.whl"],
                        "supported_os": ["linux"],
                    },
                }
            ],
        }

        plan = build_extension_plan(
            target,
            inventory,
            candidate,
            {"bitsandbytes": profile},
            ("bitsandbytes",),
        )

        self.assertEqual(plan.extensions[0]["status"], "blocked")
        self.assertEqual(plan.commands, ())

    def test_apply_rejects_non_installable_selection(self):
        plan = ExtensionPlan(
            Path("C:/target"),
            {"id": "candidate"},
            ({"id": "bitsandbytes", "status": "unverified"},),
        )
        backup = type("Backup", (), {"path": Path("C:/backup.json")})()
        target = type("Target", (), {"comfyui_dir": Path("C:/target")})()

        with self.assertRaises(ValueError):
            apply_extension_plan(target, plan, backup)

    def test_apply_runs_only_planned_commands(self):
        plan = ExtensionPlan(
            Path("C:/target"),
            {"id": "candidate"},
            ({"id": "bitsandbytes", "status": "installable"},),
            ({"extension_id": "bitsandbytes", "command": ["python", "-m", "pip", "install", "pkg"]},),
        )
        backup = type("Backup", (), {"path": Path("C:/backup.json")})()
        target = type(
            "Target",
            (),
            {
                "root": Path("C:/target"),
                "comfyui_dir": Path("C:/target"),
                "python_executable": Path("C:/target/python.exe"),
            },
        )()
        completed = type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        with patch(
            "rocm_stack_manager.adapters.comfyui.extensions.subprocess.run",
            return_value=completed,
        ) as run:
            result = apply_extension_plan(target, plan, backup)

        self.assertIsInstance(result, ExtensionInstallResult)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(run.call_args.args[0][0], "python")
