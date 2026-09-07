import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from rocm_stack_manager.core.catalog import CatalogError, ensure_catalog, iter_candidates, load_catalog


BUNDLE_FIXTURE = Path(__file__).parent / "fixtures" / "matrix-bundle-v1"


def _zip_fixture(destination, transform=None):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(BUNDLE_FIXTURE.rglob("*")):
            if not path.is_file():
                continue
            content = path.read_bytes()
            if path.name == "manifest.json" and transform:
                manifest = transform(json.loads(content))
                content = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
            archive.writestr(path.relative_to(BUNDLE_FIXTURE).as_posix(), content)


class CatalogBundleTests(unittest.TestCase):
    def test_directory_bundle_loads_catalog_profile_history_and_candidates(self):
        catalog_path = ensure_catalog(bundle_path=BUNDLE_FIXTURE)
        catalog = load_catalog(catalog_path)

        self.assertEqual(catalog["_comfyui_profile"]["id"], "comfyui")
        self.assertGreater(len(catalog["_historical_candidates"]), 0)
        self.assertTrue(
            iter_candidates(
                catalog,
                platform="windows",
                gfx="gfx1250",
                include_unavailable=True,
                include_incompatible=True,
            )
        )

    def test_zip_bundle_is_verified_and_cached(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "catalog.zip"
            cache = Path(directory) / "cache"
            _zip_fixture(bundle)

            first = ensure_catalog(bundle_path=bundle, cache_dir=cache)
            second = ensure_catalog(bundle_path=bundle, cache_dir=cache)

            self.assertEqual(first, second)
            self.assertEqual(load_catalog(first)["_comfyui_profile"]["id"], "comfyui")

    def test_bundle_rejects_digest_mismatch(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "catalog.zip"
            broken = Path(directory) / "broken.zip"
            _zip_fixture(bundle)
            with zipfile.ZipFile(bundle) as original, zipfile.ZipFile(broken, "w") as rewritten:
                for name in original.namelist():
                    content = b"tampered" if name == "data/matrix.json" else original.read(name)
                    rewritten.writestr(name, content)

            with self.assertRaises(CatalogError):
                ensure_catalog(bundle_path=broken, cache_dir=Path(directory) / "cache")

    def test_bundle_rejects_contract_path_and_compatibility_failures(self):
        transforms = (
            lambda manifest: {**manifest, "contract_version": 2},
            lambda manifest: {
                **manifest,
                "artifacts": [{**manifest["artifacts"][0], "path": "../escape.json"}, *manifest["artifacts"][1:]],
            },
            lambda manifest: {**manifest, "manager_compatibility": {"min": "99.0.0"}},
        )
        for transform in transforms:
            with self.subTest(transform=transform), TemporaryDirectory() as directory:
                bundle = Path(directory) / "catalog.zip"
                _zip_fixture(bundle, transform)
                with self.assertRaises(CatalogError):
                    ensure_catalog(bundle_path=bundle, cache_dir=Path(directory) / "cache")


if __name__ == "__main__":
    unittest.main()
