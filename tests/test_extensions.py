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
    def test_catalog_artifacts_are_reported_with_target_match(self):
        inventory = PackageInventory(Path("C:/target"), None, (), "detected")
        catalog = {
            "extensions": [{
                "id": "extension:bitsandbytes:0.48.2",
                "extension": "bitsandbytes",
                "package_name": "bitsandbytes",
                "version": "0.48.2",
                "python_tags": ["cp312"],
                "abi_tags": ["cp312"],
                "platform_tags": ["win_amd64"],
                "artifact_urls": ["https://example.test/bitsandbytes.whl"],
            }]
        }
        report = build_extension_report(
            inventory,
            candidate={"platform": "windows", "python_tag": "cp312", "abi_tags": ["cp312"], "rocm_version": "10.1.0"},
            extension_catalog=catalog,
        )
        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")
        self.assertEqual(bitsandbytes["target_match"], "matched")
        self.assertEqual(bitsandbytes["available_versions"], ["0.48.2"])
        self.assertEqual(bitsandbytes["latest_artifact"]["version"], "0.48.2")

    def test_target_platform_tags_select_one_windows_architecture(self):
        inventory = PackageInventory(Path("C:/target"), None, (), "detected")
        catalog = {
            "extensions": [{
                "id": "extension:bitsandbytes:0.50.0",
                "extension": "bitsandbytes",
                "package_name": "bitsandbytes",
                "version": "0.50.0",
                "python_tags": ["py3"],
                "abi_tags": ["none"],
                "platform_tags": ["win_amd64", "win_arm64"],
                "artifacts": [
                    {"candidate_id": "amd64", "python_tag": "py3", "abi_tag": "none", "platform_tag": "win_amd64", "url": "https://example.test/amd64.whl", "requires_dist": []},
                    {"candidate_id": "arm64", "python_tag": "py3", "abi_tag": "none", "platform_tag": "win_arm64", "url": "https://example.test/arm64.whl", "requires_dist": []},
                ],
                "artifact_urls": ["https://example.test/amd64.whl", "https://example.test/arm64.whl"],
            }]
        }
        candidate = {
            "platform": "windows",
            "python_tag": "cp312",
            "abi_tags": ["cp312", "none"],
            "platform_tags": ["win_amd64"],
            "rocm_version": "10.1.0",
        }
        report = build_extension_report(inventory, candidate=candidate, extension_catalog=catalog)
        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")
        self.assertEqual(bitsandbytes["target_match"], "matched")
        self.assertEqual(bitsandbytes["latest_artifact"]["artifacts"][0]["url"], "https://example.test/amd64.whl")
        plan = build_extension_plan(
            type("Target", (), {"root": Path("C:/target"), "python_executable": Path("C:/target/python.exe")})(),
            inventory,
            {**candidate, "artifact_available": True, "candidate_kind": "installable"},
            extension_catalog=catalog,
            selections=("bitsandbytes",),
            allow_unverified=True,
        )
        self.assertEqual(plan.commands[0]["command"][-1], "https://example.test/amd64.whl")

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

    def test_report_exposes_pinned_official_source(self):
        report = build_extension_report(PackageInventory(Path("C:/target"), None, (), "detected"))
        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")

        self.assertEqual(bitsandbytes["official_source"]["revision_type"], "commit")
        self.assertEqual(bitsandbytes["official_source"]["package_names"], ["bitsandbytes"])

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
            "abi_tags": ["cp312"],
            "torch_version": "2.14.0",
            "rocm_version": "10.1.0",
        }

        plan = build_extension_plan(target, inventory, candidate)

        self.assertTrue(all(item["status"] == "unverified" for item in plan.extensions))
        self.assertEqual(plan.commands, ())

    def test_abi_mismatch_is_incompatible(self):
        inventory = PackageInventory(Path("C:/target"), None, (), "detected")
        catalog = {
            "extensions": [{
                "id": "extension:bitsandbytes:0.50.0",
                "extension": "bitsandbytes",
                "package_name": "bitsandbytes",
                "version": "0.50.0",
                "python_tags": ["cp311"],
                "abi_tags": ["cp311"],
                "platform_tags": ["win_amd64"],
                "artifacts": [{
                    "candidate_id": "cp311",
                    "python_tag": "cp311",
                    "abi_tag": "cp311",
                    "platform_tag": "win_amd64",
                    "url": "https://example.test/cp311.whl",
                    "requires_dist": [],
                }],
                "artifact_urls": ["https://example.test/cp311.whl"],
            }],
        }
        report = build_extension_report(
            inventory,
            candidate={
                "platform": "windows",
                "python_tag": "cp312",
                "abi_tags": ["cp312"],
                "platform_tags": ["win_amd64"],
            },
            extension_catalog=catalog,
        )

        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")
        self.assertEqual(bitsandbytes["target_match"], "incompatible")

    def test_abi3_and_none_are_compatible_with_target(self):
        inventory = PackageInventory(Path("C:/target"), None, (), "detected")
        catalog = {
            "extensions": [{
                "id": "extension:bitsandbytes:0.50.0",
                "extension": "bitsandbytes",
                "package_name": "bitsandbytes",
                "version": "0.50.0",
                "python_tags": ["cp312", "py3"],
                "abi_tags": ["abi3", "none"],
                "platform_tags": ["win_amd64"],
                "artifacts": [
                    {"candidate_id": "abi3", "python_tag": "cp312", "abi_tag": "abi3", "platform_tag": "win_amd64", "url": "https://example.test/abi3.whl", "requires_dist": []},
                    {"candidate_id": "none", "python_tag": "py3", "abi_tag": "none", "platform_tag": "win_amd64", "url": "https://example.test/none.whl", "requires_dist": []},
                ],
                "artifact_urls": ["https://example.test/abi3.whl", "https://example.test/none.whl"],
            }],
        }
        report = build_extension_report(
            inventory,
            candidate={
                "platform": "windows",
                "python_tag": "cp312",
                "abi_tags": ["cp312"],
                "platform_tags": ["win_amd64"],
            },
            extension_catalog=catalog,
        )

        bitsandbytes = next(item for item in report["extensions"] if item["id"] == "bitsandbytes")
        self.assertEqual(bitsandbytes["target_match"], "matched")

    def test_unverified_matching_artifact_is_preflightable_but_not_installable(self):
        target = type("Target", (), {"root": Path("C:/target"), "python_executable": Path("C:/target/python.exe")})()
        inventory = PackageInventory(Path("C:/target"), target.python_executable, (), "detected")
        candidate = {
            "id": "therock:windows:nightly:gfx1201:10.1.0:2.14.0",
            "candidate_hash": "core-hash",
            "artifact_available": True,
            "candidate_kind": "installable",
            "platform": "windows",
            "gfx": "gfx1201",
            "python_tag": "cp312",
            "abi_tags": ["cp312"],
            "torch_version": "2.14.0",
            "rocm_version": "10.1.0",
        }
        catalog = {
            "extensions": [{
                "id": "extension:bitsandbytes:0.50.0",
                "extension": "bitsandbytes",
                "package_name": "bitsandbytes",
                "version": "0.50.0",
                "source_id": "packages-pypi-bitsandbytes",
                "artifacts": [{
                    "candidate_id": "extension:bitsandbytes:0.50.0:py3:win_amd64:hash",
                    "python_tag": "py3",
                    "abi_tag": "none",
                    "platform_tag": "win_amd64",
                    "url": "https://files.example/bitsandbytes.whl",
                    "requires_dist": [],
                }],
                "candidate_ids": ["extension:bitsandbytes:0.50.0:py3:win_amd64:hash"],
                "python_tags": ["py3"],
                "abi_tags": ["none"],
                "platform_tags": ["win_amd64"],
                "artifact_urls": ["https://files.example/bitsandbytes.whl"],
                "requires_dist": [],
                "gfx_targets": [],
                "artifact_available": True,
            }]
        }

        plan = build_extension_plan(
            target,
            inventory,
            candidate,
            extension_catalog=catalog,
            selections=("bitsandbytes",),
        )

        record = plan.extensions[0]
        self.assertEqual(record["status"], "unverified")
        self.assertTrue(record["preflight_eligible"])
        self.assertEqual(plan.commands, ())

        approved = build_extension_plan(
            target,
            inventory,
            candidate,
            extension_catalog=catalog,
            selections=("bitsandbytes",),
            allow_unverified=True,
        )
        self.assertTrue(approved.allow_unverified)
        self.assertTrue(approved.commands[0]["experimental"])

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
            "abi_tags": ["cp312"],
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
        extension_catalog = {
            "extensions": [
                {
                    "id": "extension:bitsandbytes:0.46.1",
                    "extension": "bitsandbytes",
                    "package_name": "bitsandbytes",
                    "version": "0.46.1",
                    "platform": "windows",
                    "python_tags": ["cp312"],
                    "platform_tags": ["win_amd64"],
                    "abi_tags": ["cp312"],
                    "artifacts": [
                        {
                            "candidate_id": "extension-candidate",
                            "python_tag": "cp312",
                            "abi_tag": "cp312",
                            "platform_tag": "win_amd64",
                            "url": "https://example.test/bitsandbytes.whl",
                        }
                    ],
                    "artifact_urls": ["https://example.test/bitsandbytes.whl"],
                    "source_id": "packages-test",
                    "rocm_version": "10.1.0",
                    "requires_dist": [],
                    "gfx_targets": [],
                    "evidence_status": "artifact_available",
                }
            ]
        }

        plan = build_extension_plan(
            target,
            inventory,
            candidate,
            {"bitsandbytes": profile},
            ("bitsandbytes",),
            extension_catalog=extension_catalog,
        )

        self.assertEqual(plan.extensions[0]["status"], "installable")
        self.assertEqual(plan.commands[0]["extension_id"], "bitsandbytes")
        self.assertIn("https://example.test/bitsandbytes.whl", plan.commands[0]["command"])

    def test_profile_source_without_catalog_artifact_is_blocked(self):
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
            "metadata": {"status": "artifact_available"},
            "evidence_refs": ["artifact:profile-only"],
            "constraints": [
                {
                    "kind": "extension",
                    "claim_status": "artifact_available",
                    "value": {
                        "install_sources": ["https://example.test/bitsandbytes.whl"],
                        "supported_os": ["windows"],
                        "python_tags": ["cp312"],
                    },
                }
            ],
        }
        plan = build_extension_plan(target, inventory, candidate, {"bitsandbytes": profile}, ("bitsandbytes",))
        self.assertEqual(plan.extensions[0]["status"], "blocked")
        self.assertEqual(plan.commands, ())

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
            selections=("bitsandbytes",),
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
            selections=("bitsandbytes",),
            selection_hash="selection",
            resolver_results=(
                {
                    "extension_id": "bitsandbytes",
                    "status": "resolver_verified",
                    "selection_hash": "selection",
                },
            ),
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

    def test_pep440_version_order_does_not_use_lexical_comparison(self):
        from rocm_stack_manager.adapters.comfyui.extensions import _version_compare

        self.assertGreater(_version_compare("2.10", "2.9"), 0)
        self.assertTrue(
            __import__("rocm_stack_manager.adapters.comfyui.extensions", fromlist=["_requirement_satisfied"])
            ._requirement_satisfied(">=2.9,<3", "2.10")
        )

    def test_apply_requires_explicit_selection(self):
        plan = ExtensionPlan(
            Path("C:/target"),
            {"id": "candidate"},
            ({"id": "bitsandbytes", "status": "installable"},),
            ({"extension_id": "bitsandbytes", "command": ["python", "-m", "pip", "install", "pkg"]},),
        )
        backup = type("Backup", (), {"path": Path("C:/backup.json")})()
        target = type("Target", (), {"comfyui_dir": Path("C:/target")})()

        with self.assertRaisesRegex(ValueError, "explicit extension selections"):
            apply_extension_plan(target, plan, backup)
