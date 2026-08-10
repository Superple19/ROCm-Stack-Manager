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

    def test_filters_candidates_by_target_python_tag(self):
        catalog = _matrix()
        catalog["_package_snapshots"] = {
            "package_snapshots:stable": {
                "packages": {
                    name: [
                        {"python_tag": "cp312", "platform_tag": "win_amd64"},
                    ]
                    for name in (
                        "torch",
                        "torchvision",
                        "amd-torch-device-gfx1201",
                        "amd-torchvision-device-gfx1201",
                        "rocm-sdk-device-gfx1201",
                    )
                }
            }
        }

        compatible = iter_candidates(catalog, platform="windows", gfx="gfx1201", python_tag="cp312")
        incompatible = iter_candidates(catalog, platform="windows", gfx="gfx1201", python_tag="cp311")
        visible_incompatible = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            python_tag="cp311",
            include_incompatible=True,
        )

        self.assertEqual(len(compatible), 1)
        self.assertEqual(compatible[0]["python_compatibility"], "compatible")
        self.assertEqual(incompatible, [])
        self.assertEqual(visible_incompatible[0]["python_compatibility"], "incompatible")

    def test_exposes_historical_candidate_by_exact_rocm_version(self):
        catalog = _matrix()
        catalog["_historical_candidates"] = [
            {
                "id": "legacy:stable:7.2.1:gfx1201",
                "distribution_family": "legacy",
                "platform": "windows",
                "channel": "stable",
                "rocm_version": "7.2.1",
                "torch_version": "2.9.1+rocm7.2.1",
                "torchvision_version": "0.24.1+rocm7.2.1",
                "torchaudio_version": "2.9.1+rocm7.2.1",
                "python_tags": ["cp312"],
                "available_gfx_targets": ["gfx1201"],
                "artifact_available": True,
                "wheel_urls": ["https://example.test/torch.whl"],
            }
        ]

        candidates = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            rocm_version="7.2.1",
            python_tag="cp312",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["distribution_family"], "legacy")
        self.assertEqual(candidates[0]["python_compatibility"], "compatible")
        self.assertEqual(candidates[0]["wheel_urls"], ["https://example.test/torch.whl"])

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
