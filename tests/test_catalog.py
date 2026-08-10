import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.core.catalog import (
    DEFAULT_MATRIX_RAW_BASE_URL,
    CatalogError,
    ensure_catalog,
    iter_candidates,
    load_catalog,
)
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
    def test_automatic_catalog_fetch_populates_cache(self):
        document = {
            "artifacts": [{"id": "compatibility_matrix", "path": "data/matrix.json"}],
            "schema_version": 1,
        }
        responses = {
            "https://matrix.test/data/catalog.json": json.dumps(document).encode("utf-8"),
            "https://matrix.test/data/matrix.json": json.dumps(_matrix()).encode("utf-8"),
        }

        def download(url):
            if url in responses:
                return responses[url]
            raise CatalogError(f"optional source missing: {url}")

        with tempfile.TemporaryDirectory() as directory:
            with patch("rocm_stack_manager.core.catalog._download_bytes", side_effect=download) as fetch:
                path = ensure_catalog(None, cache_dir=Path(directory) / "cache", base_url="https://matrix.test")
                first_fetch_count = fetch.call_count
                cached_again = ensure_catalog(
                    None,
                    cache_dir=Path(directory) / "cache",
                    base_url="https://matrix.test",
                )

            self.assertEqual(path, cached_again)
            self.assertTrue(path.is_file())
            self.assertTrue((path.parent / "matrix.json").is_file())
            self.assertTrue((path.parents[1] / "source-manifest.json").is_file())
            self.assertGreaterEqual(first_fetch_count, 2)
            self.assertEqual(fetch.call_count, first_fetch_count)

    def test_explicit_catalog_never_fetches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text("{}", encoding="utf-8")
            with patch("rocm_stack_manager.core.catalog._download_bytes") as fetch:
                resolved = ensure_catalog(path)

            self.assertEqual(resolved, path.resolve())
            fetch.assert_not_called()

    def test_default_source_is_used_when_base_url_is_none(self):
        document = {
            "artifacts": [{"id": "compatibility_matrix", "path": "data/matrix.json"}],
        }
        responses = {
            f"{DEFAULT_MATRIX_RAW_BASE_URL}/data/catalog.json": json.dumps(document).encode("utf-8"),
            f"{DEFAULT_MATRIX_RAW_BASE_URL}/data/matrix.json": json.dumps(_matrix()).encode("utf-8"),
        }

        def download(url):
            if url in responses:
                return responses[url]
            raise CatalogError(f"optional source missing: {url}")

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "rocm_stack_manager.core.catalog._download_bytes",
                side_effect=download,
            ) as fetch:
                path = ensure_catalog(
                    None,
                    cache_dir=Path(directory) / "cache",
                    base_url=None,
                )
                self.assertTrue(path.is_file())
                self.assertIn(
                    f"{DEFAULT_MATRIX_RAW_BASE_URL}/data/catalog.json",
                    [call.args[0] for call in fetch.call_args_list],
                )

    def test_refresh_failure_preserves_existing_cache(self):
        document = {"artifacts": [], "schema_version": 1}

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "cache"
            catalog_path = cache_root / "data" / "catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text(json.dumps(document), encoding="utf-8")
            with patch(
                "rocm_stack_manager.core.catalog._download_bytes",
                side_effect=CatalogError("temporary network failure"),
            ):
                with self.assertRaises(CatalogError):
                    ensure_catalog(None, cache_dir=cache_root, refresh=True)

            self.assertEqual(json.loads(catalog_path.read_text(encoding="utf-8")), document)

    def test_rejects_catalog_path_traversal(self):
        document = {"artifacts": [{"id": "bad", "path": "../outside.json"}]}

        with tempfile.TemporaryDirectory() as directory:
            responses = {
                "https://matrix.test/data/catalog.json": json.dumps(document).encode("utf-8"),
            }

            def download(url):
                return responses[url]

            with patch(
                "rocm_stack_manager.core.catalog._download_bytes",
                side_effect=download,
            ):
                with self.assertRaises(CatalogError):
                    ensure_catalog(
                        None,
                        cache_dir=Path(directory) / "cache",
                        base_url="https://matrix.test",
                    )

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

    def test_loads_extension_artifact_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            matrix_path = root / "data" / "matrix.json"
            matrix_path.write_text(json.dumps(_matrix()), encoding="utf-8")
            extension_path = root / "data" / "extensions.json"
            extension_path.write_text(
                json.dumps({"schema_version": 1, "extensions": [{"id": "extension:one"}]}),
                encoding="utf-8",
            )
            catalog_path = root / "data" / "catalog.json"
            catalog_path.write_text(
                json.dumps({
                    "artifacts": [
                        {"id": "compatibility_matrix", "path": "data/matrix.json"},
                        {"id": "extension_catalog", "path": "data/extensions.json"},
                    ]
                }),
                encoding="utf-8",
            )

            loaded = load_catalog(catalog_path)

            self.assertEqual(loaded["_extension_catalog"]["extensions"][0]["id"], "extension:one")

    def test_filters_available_candidates(self):
        candidates = iter_candidates(_matrix(), platform="windows", gfx="gfx1201")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["channel"], "stable")
        self.assertTrue(candidates[0]["id"].startswith("therock:windows:stable:gfx1201:"))
        self.assertEqual(candidates[0]["candidate_kind"], "installable")

    def test_historical_artifact_without_install_source_is_artifact_only(self):
        catalog = _matrix()
        catalog["_historical_candidates"] = [
            {
                "id": "therock:nightly:10.1.0:torch",
                "distribution_family": "therock",
                "platform": "windows",
                "channel": "nightly",
                "rocm_version": "10.1.0",
                "torch_version": "2.14.0",
                "python_tags": ["cp312"],
                "available_gfx_targets": ["gfx1201"],
                "artifact_available": True,
            }
        ]

        candidates = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            channel="nightly",
            python_tag="cp312",
        )

        self.assertEqual(candidates[0]["candidate_kind"], "artifact_only")

    def test_filters_candidates_by_family_lifecycle_and_kind(self):
        catalog = _matrix()
        catalog["_historical_candidates"] = [
            {
                "id": "legacy:stable:7.2.1:gfx1201",
                "distribution_family": "legacy",
                "platform": "windows",
                "channel": "stable",
                "rocm_version": "7.2.1",
                "torch_version": "2.9.1+rocm7.2.1",
                "python_tags": ["cp312"],
                "available_gfx_targets": ["gfx1201"],
                "artifact_available": True,
                "wheel_urls": ["https://example.test/torch.whl"],
            }
        ]

        legacy = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            distribution_family="legacy",
            lifecycle="historical",
            candidate_kind="installable",
            python_tag="cp312",
        )

        self.assertEqual(len(legacy), 1)
        self.assertEqual(legacy[0]["distribution_family"], "legacy")
        self.assertEqual(legacy[0]["lifecycle"], "historical")

    def test_newer_versions_sort_before_older_versions(self):
        catalog = _matrix()
        catalog["_historical_candidates"] = [
            {
                "id": "therock:stable:7.13.0:torch",
                "distribution_family": "therock",
                "platform": "windows",
                "channel": "stable",
                "rocm_version": "7.13.0",
                "torch_version": "2.11.0",
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
            distribution_family="therock",
            channel="stable",
            python_tag="cp312",
        )

        self.assertEqual(candidates[0]["rocm_version"], "7.14.0")

    def test_candidate_consumes_comfyui_profile(self):
        catalog = _matrix()
        catalog["_comfyui_profile"] = {
            "id": "comfyui",
            "constraints": [
                {"id": "supported-platform", "value": ["windows", "linux"]},
                {"id": "rocm-channel", "value": {"allowed": ["stable", "nightly", "staging"]}},
                {"id": "rocm-package-candidate", "value": {"distribution_families": ["therock", "legacy"]}},
                {"id": "torch-rocm", "value": {"torch_series": ["2.12"]}},
            ],
        }

        candidate = iter_candidates(catalog, platform="windows", gfx="gfx1201")[0]

        self.assertEqual(candidate["profile_id"], "comfyui")
        self.assertEqual(candidate["profile_status"], "documented")
        self.assertEqual(candidate["profile_warnings"], [])

    def test_candidate_profile_warns_for_out_of_scope_torch_series(self):
        catalog = _matrix()
        catalog["_comfyui_profile"] = {
            "id": "comfyui",
            "constraints": [
                {"id": "torch-rocm", "value": {"torch_series": ["2.14"]}},
            ],
        }

        candidate = iter_candidates(catalog, platform="windows", gfx="gfx1201")[0]

        self.assertEqual(candidate["profile_status"], "out_of_profile")
        self.assertTrue(any("Torch series 2.12" in warning for warning in candidate["profile_warnings"]))

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

    def test_adds_legacy_rocm_source_artifact(self):
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
                "python_tags": ["cp312"],
                "available_gfx_targets": ["gfx1201"],
                "artifact_available": True,
                "wheel_urls": [
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
                ],
            }
        ]

        candidate = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            rocm_version="7.2.1",
            python_tag="cp312",
        )[0]

        self.assertEqual(candidate["wheel_urls"][0], "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz")

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
