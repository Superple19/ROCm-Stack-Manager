import json
import tempfile
import unittest
from pathlib import Path

from rocm_stack_manager.core.catalog import iter_candidates, load_catalog
from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.planning import PlanningError, build_plan


def _matrix():
    return {
        "targets": [
            {
                "gfx": "gfx1201",
                "platforms": {
                    "windows": {
                        "package_channels": {
                            "stable": {
                                "all_device_packages_available": True,
                                "rocm_device_version": "7.14.0",
                                "torch_device_version": "2.12.0+rocm7.14.0",
                                "torchvision_device_version": "0.27.0+rocm7.14.0",
                                "source_id": "packages-stable",
                            },
                            "nightly": {
                                "all_device_packages_available": False,
                                "rocm_device_version": None,
                                "torch_device_version": None,
                                "torchvision_device_version": None,
                                "source_id": "packages-nightly",
                            },
                        }
                    }
                },
            }
        ]
    }


class CatalogTests(unittest.TestCase):
    def test_loads_matrix_from_catalog_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(_matrix()), encoding="utf-8")
            index_path = root / "catalog.json"
            index_path.write_text(
                json.dumps({"artifacts": [{"id": "compatibility_matrix", "path": "matrix.json"}]}),
                encoding="utf-8",
            )

            loaded = load_catalog(index_path)

            self.assertEqual(loaded["targets"][0]["gfx"], "gfx1201")

    def test_filters_available_candidates(self):
        candidates = iter_candidates(_matrix(), platform="windows", gfx="gfx1201")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["channel"], "stable")
        self.assertTrue(candidates[0]["id"].startswith("therock:windows:stable:gfx1201:"))

    def test_plan_rejects_unavailable_candidate(self):
        candidate = iter_candidates(
            _matrix(), platform="windows", gfx="gfx1201", include_unavailable=True, channel="nightly"
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ComfyUI").mkdir()
            (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
            target = detect_target(root)

            with self.assertRaises(PlanningError):
                build_plan(target, candidate)
