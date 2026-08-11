import subprocess
import unittest
from pathlib import Path

from rocm_stack_manager.core.resolver import build_resolver_command, run_resolver
from rocm_stack_manager.core.identity import candidate_hash


class CoreResolverTests(unittest.TestCase):
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
            "id": "therock:windows:stable:gfx1201:7.14.0:2.12.0+rocm7.14.0",
            "candidate_hash": "core-hash",
            "platform": "windows",
            "package_specs": ["torch==2.12.0+rocm7.14.0"],
            "index_url": "https://repo.example/rocm/",
            "artifact_available": True,
            "candidate_kind": "installable",
            "python_compatibility": "compatible",
        }

    def test_command_is_a_non_mutating_core_preflight(self):
        command = build_resolver_command(self.target, self.candidate)

        self.assertIn("--dry-run", command)
        self.assertIn("--ignore-installed", command)
        self.assertIn("--extra-index-url", command)
        self.assertNotIn("--report", command)
        self.assertIn("torch==2.12.0+rocm7.14.0", command)

    def test_runner_preserves_candidate_identity_and_success(self):
        completed = subprocess.CompletedProcess(
            ["python", "-m", "pip"],
            0,
            stdout="Would install torch",
            stderr="",
        )
        result = run_resolver(
            self.target,
            self.candidate,
            runner=lambda *args, **kwargs: completed,
        )

        self.assertEqual(result.status, "resolver_verified")
        self.assertEqual(result.candidate_id, self.candidate["id"])
        self.assertEqual(result.candidate_hash, "core-hash")
        self.assertFalse(result.as_dict()["installation_performed"])
        self.assertEqual(result.as_dict()["promotion"], "none")

    def test_runner_preserves_target_catalog_and_adapter_binding(self):
        completed = subprocess.CompletedProcess(
            ["python", "-m", "pip"],
            0,
            stdout="Would install torch",
            stderr="",
        )
        result = run_resolver(
            self.target,
            self.candidate,
            catalog_hash="catalog-hash",
            adapter_id="comfyui",
            target_gfx="gfx1201",
            runner=lambda *args, **kwargs: completed,
        )

        self.assertEqual(
            result.binding,
            {
                "target_root": "C:\\target",
                "target_python": "C:\\target\\python.exe",
                "target_platform": "windows",
                "target_gfx": "gfx1201",
                "candidate_hash": candidate_hash(self.candidate),
                "catalog_hash": "catalog-hash",
                "adapter_id": "comfyui",
            },
        )

    def test_artifact_only_candidate_cannot_resolve(self):
        candidate = dict(self.candidate, candidate_kind="artifact_only", package_specs=[])

        result = run_resolver(self.target, candidate)

        self.assertEqual(result.status, "resolver_failed")
        self.assertEqual(result.command, ())
        self.assertIn("artifact evidence only", result.output)


if __name__ == "__main__":
    unittest.main()
