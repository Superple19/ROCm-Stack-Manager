import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rocm_stack_manager import cli


class CliTests(unittest.TestCase):
    def test_extension_report_works_without_matrix_catalog(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            comfyui = root / "ComfyUI"
            comfyui.mkdir()
            (comfyui / "main.py").write_text("", encoding="utf-8")
            output = io.StringIO()
            error = io.StringIO()
            with patch(
                "rocm_stack_manager.adapters.comfyui.adapter.collect_inventory",
                return_value=type(
                    "Inventory",
                    (),
                    {
                        "target_root": root,
                        "status": "detected",
                        "error": None,
                        "packages": (),
                    },
                )(),
            ):
                with redirect_stdout(output), redirect_stderr(error):
                    result = cli.main(["extensions", "report", "--target", str(root)])

            self.assertEqual(result, 0)
            self.assertEqual(error.getvalue(), "")
            self.assertIn("Mode: local inventory only", output.getvalue())
            self.assertIn("not installed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
