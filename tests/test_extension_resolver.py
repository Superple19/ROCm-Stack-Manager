import unittest
from pathlib import Path

from rocm_stack_manager.core.extension_resolver import build_extension_resolver_command, run_extension_resolver


class ExtensionResolverTests(unittest.TestCase):
    def setUp(self):
        self.target = type(
            "Target",
            (),
            {
                "root": Path("C:/target"),
                "comfyui_dir": Path("C:/target"),
                "python_executable": Path("C:/target/python.exe"),
            },
        )()
        self.candidate = {
            "id": "therock:windows:nightly:gfx1201:10.1.0:2.14.0",
            "candidate_hash": "core-hash",
            "platform": "windows",
            "package_specs": ["torch==2.14.0"],
            "index_url": "https://repo.example/simple/",
        }
        self.extension = {
            "id": "bitsandbytes",
            "status": "installable",
            "extension_candidate_id": "extension:bitsandbytes:0.50.0:cp312:any:hash",
            "sources": ["https://files.example/bitsandbytes.whl"],
        }

    def test_command_is_non_mutating_and_binds_core_and_extension(self):
        command = build_extension_resolver_command(self.target, self.candidate, self.extension)
        self.assertIn("--dry-run", command)
        self.assertIn("--ignore-installed", command)
        self.assertIn("--extra-index-url", command)
        self.assertNotIn("--index-url", command)
        self.assertNotIn("--report", command)
        self.assertIn("torch==2.14.0", command)
        self.assertIn("https://files.example/bitsandbytes.whl", command)

    def test_runner_preserves_exact_candidate_identity(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": "resolved", "stderr": ""})()
        results = run_extension_resolver(self.target, self.candidate, [self.extension], runner=lambda *args, **kwargs: completed)
        self.assertEqual(results[0].status, "resolver_verified")
        self.assertEqual(results[0].candidate_hash, "core-hash")
        self.assertEqual(results[0].extension_candidate_id, self.extension["extension_candidate_id"])

    def test_unverified_artifact_can_run_preflight_without_install_approval(self):
        extension = dict(self.extension)
        extension.update({
            "status": "unverified",
            "claim_status": "unverified",
            "preflight_eligible": True,
        })
        command = build_extension_resolver_command(self.target, self.candidate, extension)
        self.assertIn("https://files.example/bitsandbytes.whl", command)
        completed = type("Completed", (), {"returncode": 0, "stdout": "resolved", "stderr": ""})()
        result = run_extension_resolver(
            self.target,
            self.candidate,
            [extension],
            runner=lambda *args, **kwargs: completed,
        )[0]
        self.assertEqual(result.status, "resolver_verified")
        self.assertEqual(result.claim_status, "unverified")
        self.assertTrue(result.preflight)
        self.assertEqual(result.as_dict()["promotion"], "none")


if __name__ == "__main__":
    unittest.main()
