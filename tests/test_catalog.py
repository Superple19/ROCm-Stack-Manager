import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.core.catalog import (
    DEFAULT_MATRIX_REVISION_URL,
    MATRIX_CATALOG_URL_ENV,
    CatalogError,
    ensure_catalog,
    iter_candidates,
    load_catalog,
)
from rocm_stack_manager.core.detection import detect_target
from rocm_stack_manager.core.planning import PlanningError, build_plan


def _matrix():
    matrix = {
        "schema_version": 1,
        "generated_at": "2026-08-08T00:00:00Z",
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
                                "torchaudio_version": "2.12.0+rocm7.14.0",
                                "source_id": "packages-stable",
                            },
                            "nightly": {
                                "all_device_packages_available": False,
                                "rocm_device_version": None,
                                "torch_device_version": None,
                                "torchvision_device_version": None,
                                "torchaudio_version": None,
                                "source_id": "packages-nightly",
                            },
                        }
                    }
                },
            }
        ]
    }
    versions = {
        "rocm": "7.14.0",
        "rocm-sdk-core": "7.14.0",
        "rocm-sdk-libraries": "7.14.0",
        "rocm-sdk-device-gfx1201": "7.14.0",
        "torch": "2.12.0+rocm7.14.0",
        "amd-torch-device-gfx1201": "2.12.0+rocm7.14.0",
        "torchvision": "0.27.0+rocm7.14.0",
        "amd-torchvision-device-gfx1201": "0.27.0+rocm7.14.0",
        "torchaudio": "2.12.0+rocm7.14.0",
    }
    matrix["_package_snapshots"] = {
        "package_snapshots:stable": {
            "packages": {
                name: [{"version": version, "python_tag": "cp312", "platform_tag": "win_amd64"}]
                for name, version in versions.items()
            }
        }
    }
    return matrix


class CatalogTests(unittest.TestCase):
    def test_automatic_catalog_fetch_populates_cache(self):
        matrix_bytes = json.dumps(_matrix()).encode("utf-8")
        document = {
            "schema_version": 1,
            "artifacts": [{"id": "compatibility_matrix", "path": "data/matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_bytes).hexdigest()}],
        }
        responses = {
            "https://matrix.test/data/catalog.json": json.dumps(document).encode("utf-8"),
            "https://matrix.test/data/matrix.json": matrix_bytes,
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
            self.assertTrue((Path(directory) / "cache" / "source-manifest.json").is_file())
            self.assertTrue((Path(directory) / "cache" / "current.json").is_file())
            self.assertGreaterEqual(first_fetch_count, 2)
            self.assertEqual(fetch.call_count, first_fetch_count)

    def test_corrupt_cached_artifact_is_replaced(self):
        matrix_bytes = json.dumps(_matrix()).encode("utf-8")
        document = {
            "schema_version": 1,
            "artifacts": [{"id": "compatibility_matrix", "path": "data/matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_bytes).hexdigest()}],
        }
        responses = {
            "https://matrix.test/data/catalog.json": json.dumps(document).encode("utf-8"),
            "https://matrix.test/data/matrix.json": matrix_bytes,
        }

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "cache"
            with patch("rocm_stack_manager.core.catalog._download_bytes", side_effect=lambda url: responses[url]):
                path = ensure_catalog(None, cache_dir=cache_root, base_url="https://matrix.test")
            (path.parent / "matrix.json").write_bytes(b"corrupt")
            with patch("rocm_stack_manager.core.catalog._download_bytes", side_effect=lambda url: responses[url]) as fetch:
                repaired = ensure_catalog(None, cache_dir=cache_root, base_url="https://matrix.test")

            self.assertEqual((repaired.parent / "matrix.json").read_bytes(), matrix_bytes)
            self.assertGreaterEqual(fetch.call_count, 2)

    def test_explicit_catalog_never_fetches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text("{}", encoding="utf-8")
            with patch("rocm_stack_manager.core.catalog._download_bytes") as fetch:
                resolved = ensure_catalog(path)

            self.assertEqual(resolved, path.resolve())
            fetch.assert_not_called()

    def test_default_source_is_used_when_base_url_is_none(self):
        matrix_bytes = json.dumps(_matrix()).encode("utf-8")
        revision = "a" * 40
        revision_base = f"https://raw.githubusercontent.com/Superple19/rocm-evidence-matrix/{revision}"
        document = {
            "schema_version": 1,
            "artifacts": [{"id": "compatibility_matrix", "path": "data/matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_bytes).hexdigest()}],
        }
        responses = {
            DEFAULT_MATRIX_REVISION_URL: json.dumps({"sha": revision}).encode("utf-8"),
            f"{revision_base}/data/catalog.json": json.dumps(document).encode("utf-8"),
            f"{revision_base}/data/matrix.json": matrix_bytes,
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
                manifest = json.loads((Path(directory) / "cache" / "source-manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["revision"], revision)
                self.assertIn(f"{DEFAULT_MATRIX_REVISION_URL}", [call.args[0] for call in fetch.call_args_list])
                self.assertIn(f"{revision_base}/data/catalog.json", [call.args[0] for call in fetch.call_args_list])

    def test_environment_source_overrides_default(self):
        matrix_bytes = json.dumps(_matrix()).encode("utf-8")
        document = {"artifacts": [{"id": "compatibility_matrix", "path": "data/matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_bytes).hexdigest()}], "schema_version": 1}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {MATRIX_CATALOG_URL_ENV: "https://mirror.test/matrix"}
        ):
            with patch(
                "rocm_stack_manager.core.catalog._download_bytes",
                side_effect=[json.dumps(document).encode("utf-8"), matrix_bytes],
            ) as fetch:
                ensure_catalog(None, cache_dir=Path(directory) / "cache", base_url=None)

            self.assertEqual(
                fetch.call_args_list[0].args[0],
                "https://mirror.test/matrix/data/catalog.json",
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
                json.dumps({"schema_version": 1, "artifacts": [{"id": "compatibility_matrix", "path": "matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest()}]}),
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
                        {"id": "compatibility_matrix", "path": "data/matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest()},
                        {"id": "extension_catalog", "path": "data/extensions.json", "schema": "schemas/extension-catalog.schema.json", "schema_version": 1, "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest()},
                    ],
                    "schema_version": 1
                }),
                encoding="utf-8",
            )

            loaded = load_catalog(catalog_path)

            self.assertEqual(loaded["_extension_catalog"]["extensions"][0]["id"], "extension:one")

    def test_rejects_malformed_package_snapshot_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            matrix_path = root / "data" / "matrix.json"
            matrix_path.write_text(json.dumps(_matrix()), encoding="utf-8")
            snapshot_path = root / "data" / "stable.json"
            snapshot_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            catalog_path = root / "data" / "catalog.json"
            catalog_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "artifacts": [
                        {"id": "compatibility_matrix", "path": "data/matrix.json", "schema": "schemas/compatibility-matrix.schema.json", "schema_version": 1, "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest()},
                        {"id": "package_snapshots:stable", "path": "data/stable.json", "schema": "schemas/package-snapshot.schema.json", "schema_version": 1, "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest()},
                    ],
                }),
                encoding="utf-8",
            )

            with self.assertRaises(CatalogError):
                load_catalog(catalog_path)

    def test_filters_available_candidates(self):
        candidates = iter_candidates(_matrix(), platform="windows", gfx="gfx1201")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["channel"], "stable")
        self.assertTrue(candidates[0]["id"].startswith("therock:windows:stable:gfx1201:"))
        self.assertEqual(candidates[0]["candidate_kind"], "installable")

    def test_candidate_identity_includes_torchvision(self):
        first = iter_candidates(_matrix(), platform="windows", gfx="gfx1201")[0]
        changed = _matrix()
        changed["targets"][0]["platforms"]["windows"]["package_channels"]["stable"][
            "torchvision_device_version"
        ] = "0.28.0+rocm7.14.0"
        second = iter_candidates(changed, platform="windows", gfx="gfx1201")[0]

        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["candidate_hash"], second["candidate_hash"])

    def test_malformed_platform_record_is_rejected(self):
        matrix = _matrix()
        matrix["targets"][0]["platforms"]["windows"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaises(CatalogError):
                load_catalog(path)

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

    def test_rejects_package_version_mismatch_even_when_tags_match(self):
        catalog = _matrix()
        catalog["_package_snapshots"]["package_snapshots:stable"]["packages"]["torch"][0]["version"] = "2.13.0+rocm7.14.0"

        visible = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            python_tag="cp312",
            include_incompatible=True,
        )

        self.assertEqual(visible[0]["python_compatibility"], "incompatible")
        self.assertEqual(iter_candidates(catalog, platform="windows", gfx="gfx1201", python_tag="cp312"), [])

    def test_rejects_missing_package_map(self):
        catalog = _matrix()
        del catalog["_package_snapshots"]["package_snapshots:stable"]["packages"]["torchaudio"]

        visible = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            python_tag="cp312",
            include_incompatible=True,
        )

        self.assertEqual(visible[0]["python_compatibility"], "incompatible")

    def test_rejects_versionless_artifact(self):
        catalog = _matrix()
        catalog["_package_snapshots"]["package_snapshots:stable"]["packages"]["torch"][0]["version"] = None

        visible = iter_candidates(
            catalog,
            platform="windows",
            gfx="gfx1201",
            python_tag="cp312",
            include_incompatible=True,
        )

        self.assertEqual(visible[0]["python_compatibility"], "incompatible")

    def test_rejects_linux_gfx1250_stable_when_generic_versions_do_not_match(self):
        catalog = _matrix()
        catalog["targets"].append({
            "gfx": "gfx1250",
            "platforms": {
                "linux": {
                    "package_channels": {
                        "stable": {
                            "all_device_packages_available": True,
                            "rocm_device_version": "7.14.0",
                            "torch_device_version": "2.11.0+rocm7.14.0",
                            "torchvision_device_version": "0.26.0+rocm7.14.0",
                            "torchaudio_version": "2.11.0.2+rocm7.14.0",
                            "source_id": "packages-stable-linux",
                        }
                    }
                }
            },
        })
        expected = {
            "rocm": "7.14.0",
            "rocm-sdk-core": "7.14.0",
            "rocm-sdk-libraries": "7.14.0",
            "rocm-sdk-device-gfx1250": "7.14.0",
            "torch": "2.11.0+rocm7.14.0",
            "amd-torch-device-gfx1250": "2.11.0+rocm7.14.0",
            "torchvision": "0.26.0+rocm7.14.0",
            "amd-torchvision-device-gfx1250": "0.26.0+rocm7.14.0",
            "torchaudio": "2.11.0.2+rocm7.14.0",
        }
        packages = {
            name: [{"version": version, "python_tag": "cp312", "platform_tag": "linux_x86_64"}]
            for name, version in expected.items()
        }
        packages["torch"][0]["version"] = "2.13.0+rocm7.14.0"
        packages["torchvision"][0]["version"] = "0.28.0+rocm7.14.0"
        catalog["_package_snapshots"]["package_snapshots:stable-linux"] = {"packages": packages}

        visible = iter_candidates(
            catalog,
            platform="linux",
            gfx="gfx1250",
            channel="stable",
            python_tag="cp312",
            include_incompatible=True,
        )

        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["python_compatibility"], "incompatible")

    def test_rejects_linux_gfx90c_nightly_when_generic_versions_do_not_match(self):
        catalog = _matrix()
        catalog["targets"].append({
            "gfx": "gfx90c",
            "platforms": {
                "linux": {
                    "package_channels": {
                        "nightly": {
                            "all_device_packages_available": True,
                            "rocm_device_version": "10.1.0a20260807",
                            "torch_device_version": "2.12.0+rocm10.1.0a20260807",
                            "torchvision_device_version": "0.27.0+rocm10.1.0a20260807",
                            "torchaudio_version": "2.11.0+rocm10.1.0a20260807",
                            "source_id": "packages-nightly-linux",
                        }
                    }
                }
            },
        })
        expected = {
            "rocm": "10.1.0a20260807",
            "rocm-sdk-core": "10.1.0a20260807",
            "rocm-sdk-libraries": "10.1.0a20260807",
            "rocm-sdk-device-gfx90c": "10.1.0a20260807",
            "torch": "2.12.0+rocm10.1.0a20260807",
            "amd-torch-device-gfx90c": "2.12.0+rocm10.1.0a20260807",
            "torchvision": "0.27.0+rocm10.1.0a20260807",
            "amd-torchvision-device-gfx90c": "0.27.0+rocm10.1.0a20260807",
            "torchaudio": "2.11.0+rocm10.1.0a20260807",
        }
        packages = {
            name: [{"version": version, "python_tag": "cp312", "platform_tag": "linux_x86_64"}]
            for name, version in expected.items()
        }
        packages["torch"][0]["version"] = "2.14.0a0+rocm10.1.0a20260807"
        packages["torchvision"][0]["version"] = "0.29.0a0+rocm10.1.0a20260807"
        catalog["_package_snapshots"]["package_snapshots:nightly-linux"] = {"packages": packages}

        visible = iter_candidates(
            catalog,
            platform="linux",
            gfx="gfx90c",
            channel="nightly",
            python_tag="cp312",
            include_incompatible=True,
        )

        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["python_compatibility"], "incompatible")

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
