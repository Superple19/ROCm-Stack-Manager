import tempfile
import unittest
from pathlib import Path

from rocm_stack_manager.core.staging import StagingError, stage_candidate, validate_staged_candidate


class StagingTests(unittest.TestCase):
    def _target(self, root):
        comfyui = root / "ComfyUI"
        comfyui.mkdir()
        python = root / "python_env" / "python.exe"
        python.parent.mkdir()
        python.write_text("", encoding="utf-8")
        return type(
            "Target",
            (),
            {"root": root, "comfyui_dir": comfyui, "python_executable": python},
        )()

    def test_stages_and_hashes_downloaded_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self._target(root)

            def runner(command, **_kwargs):
                destination = Path(command[command.index("--dest") + 1])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "torch-2.12.0-cp312-cp312-win_amd64.whl").write_bytes(b"wheel")
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            staged = stage_candidate(
                target,
                {"id": "candidate", "package_specs": ["torch==2.12.0"]},
                runner=runner,
            )
            self.assertTrue(validate_staged_candidate(staged))
            self.assertIn("--require-hashes", staged.install_command)
            staged_artifact = staged.artifacts_path / "torch-2.12.0-cp312-cp312-win_amd64.whl"
            staged_artifact.write_bytes(b"corrupted")
            with self.assertRaisesRegex(StagingError, "hash mismatch"):
                validate_staged_candidate(staged)

    def test_staging_rejects_empty_download(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._target(Path(directory))
            completed = type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            with self.assertRaisesRegex(StagingError, "no installable artifacts"):
                stage_candidate(
                    target,
                    {"id": "candidate", "package_specs": ["torch==2.12.0"]},
                    runner=lambda *_args, **_kwargs: completed,
                )


if __name__ == "__main__":
    unittest.main()
