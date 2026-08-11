import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rocm_stack_manager import cli
from rocm_stack_manager.core.verify import RuntimeObservation


class CliTests(unittest.TestCase):
    def test_candidate_gfx_is_optional_for_target_probe(self):
        args = cli.parse_args(["candidates", "--target", "target"])

        self.assertIsNone(args.gfx)

    def test_candidates_infer_single_target_gfx(self):
        class FakeAdapter:
            id = "fake"

            def detect(self, path):
                return type(
                    "Target",
                    (),
                    {
                        "root": Path(path),
                        "comfyui_dir": Path(path),
                        "python_executable": None,
                    },
                )()

            def python_tag(self, target):
                return "cp312"

            def verify(self, target):
                return RuntimeObservation(
                    target_root=target.root,
                    python_executable=None,
                    host_platform="windows",
                    runtime_status="detected",
                    hardware_status="detected",
                    devices=({"gfx": "gfx1201"},),
                )

        with patch("rocm_stack_manager.cli.get_adapter", return_value=FakeAdapter()), patch(
            "rocm_stack_manager.cli.ensure_catalog", return_value=Path("catalog.json")
        ), patch("rocm_stack_manager.cli.load_catalog", return_value={}), patch(
            "rocm_stack_manager.cli.iter_candidates", return_value=[]
        ) as iter_candidates:
            result = cli.main(["candidates", "--target", "target", "--json"])

        self.assertEqual(result, 0)
        self.assertEqual(iter_candidates.call_args.kwargs["gfx"], "gfx1201")

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
