import unittest
from pathlib import Path

from rocm_stack_manager.adapters.comfyui.extensions import build_extension_report
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
