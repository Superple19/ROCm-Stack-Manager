import json
import tempfile
import unittest
from pathlib import Path

from rocm_stack_manager.core.extension_sources import ExtensionSourceError, load_extension_sources, validate_extension_sources


class ExtensionSourceTests(unittest.TestCase):
    def test_loads_bundled_pinned_sources(self):
        sources = load_extension_sources()

        self.assertEqual(set(sources), {"bitsandbytes", "flash-attention", "aiter", "sageattention", "triton"})
        self.assertTrue(all(len(source["revision"]) == 40 for source in sources.values()))

    def test_rejects_moving_branch(self):
        document = {
            "schema_version": 1,
            "extensions": [{
                "id": "example",
                "repository": "https://github.com/example/project",
                "revision": "main",
                "revision_type": "tag",
                "package_names": ["example"],
                "install_methods": ["source"],
            }],
        }

        with self.assertRaisesRegex(ExtensionSourceError, "moving branch"):
            validate_extension_sources(document)

    def test_rejects_partial_commit(self):
        document = {
            "schema_version": 1,
            "extensions": [{
                "id": "example",
                "repository": "https://github.com/example/project",
                "revision": "abc123",
                "revision_type": "commit",
                "package_names": ["example"],
                "install_methods": ["source"],
            }],
        }

        with self.assertRaisesRegex(ExtensionSourceError, "full SHA-1"):
            validate_extension_sources(document)

    def test_rejects_duplicate_package_names(self):
        document = {
            "schema_version": 1,
            "extensions": [
                {
                    "id": "one",
                    "repository": "https://github.com/example/one",
                    "revision": "a" * 40,
                    "revision_type": "commit",
                    "package_names": ["shared"],
                    "install_methods": ["source"],
                },
                {
                    "id": "two",
                    "repository": "https://github.com/example/two",
                    "revision": "b" * 40,
                    "revision_type": "commit",
                    "package_names": ["shared"],
                    "install_methods": ["source"],
                },
            ],
        }

        with self.assertRaisesRegex(ExtensionSourceError, "multiple extension sources"):
            validate_extension_sources(document)

    def test_loads_manifest_from_explicit_path(self):
        document = {
            "schema_version": 1,
            "extensions": [{
                "id": "example",
                "repository": "https://github.com/example/project",
                "revision": "c" * 40,
                "revision_type": "commit",
                "package_names": ["example"],
                "install_methods": ["source"],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            self.assertEqual(load_extension_sources(path)["example"]["revision"], "c" * 40)


if __name__ == "__main__":
    unittest.main()
