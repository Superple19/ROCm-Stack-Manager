import unittest
from pathlib import Path

from rocm_stack_manager.core.extension_verification import build_extension_verification


class ExtensionVerificationTests(unittest.TestCase):
    def test_export_binds_core_extension_and_runtime_identity(self):
        target = type(
            "Target",
            (),
            {
                "root": Path("C:/target"),
                "comfyui_dir": Path("C:/target"),
                "python_executable": Path("C:/target/python.exe"),
            },
        )()
        candidate = {
            "id": "therock:windows:nightly:gfx1201:10.1.0:2.14.0",
            "candidate_hash": "core-hash",
            "platform": "windows",
            "gfx": "gfx1201",
            "python_tag": "cp312",
            "torch_version": "2.14.0",
            "rocm_version": "10.1.0",
        }
        runtime = {
            "runtime_status": "detected",
            "hardware_status": "detected",
            "torch_version": "2.14.0",
            "hip_version": "7.15.26312",
            "devices": [{"gfx": "gfx1201"}],
        }
        hardware = {"status": "detected", "gfx_targets": ["gfx1201"], "driver_version": "31.0"}
        extension = {
            "id": "bitsandbytes",
            "extension_candidate_id": "extension:bitsandbytes:0.50.0:cp312:any:hash",
            "claim_status": "artifact_available",
        }

        def runner(command, **kwargs):
            output = '{"result": [2.0]}' if "torch.tensor" in command[2] else '{"version": "0.50.0"}'
            return type("Completed", (), {"returncode": 0, "stdout": output, "stderr": ""})()

        evidence = build_extension_verification(
            target,
            candidate,
            [extension],
            {"bitsandbytes": ("bitsandbytes",)},
            runtime,
            hardware,
            runner=runner,
        )
        self.assertEqual(evidence["candidate_hash"], "core-hash")
        self.assertEqual(evidence["extensions"][0]["extension_candidate_id"], extension["extension_candidate_id"])
        self.assertEqual(evidence["verification_level"], "hardware_verified")
        self.assertEqual(evidence["promotion"], "none")


if __name__ == "__main__":
    unittest.main()
