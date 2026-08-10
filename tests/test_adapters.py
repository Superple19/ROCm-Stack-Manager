import tempfile
import unittest
from pathlib import Path

from rocm_stack_manager.adapters.registry import available_adapters, get_adapter
from rocm_stack_manager.cli import parse_args
from rocm_stack_manager.core.adapter import CapabilityUnavailable, RuntimeAdapter
from rocm_stack_manager.core.detection import detect_target


class AdapterContractTests(unittest.TestCase):
    def _target(self, root):
        (root / "ComfyUI").mkdir()
        (root / "ComfyUI" / "main.py").write_text("", encoding="utf-8")
        python = root / "python_env" / "python.exe"
        python.parent.mkdir()
        python.write_text("", encoding="utf-8")
        return detect_target(root)

    def test_registry_contains_comfyui_and_ollama(self):
        self.assertEqual(available_adapters(), ("comfyui", "ollama"))
        self.assertEqual(get_adapter("comfyui").id, "comfyui")
        self.assertEqual(get_adapter("ollama").id, "ollama")

    def test_comfyui_adapter_satisfies_runtime_contract(self):
        self.assertIsInstance(get_adapter("comfyui"), RuntimeAdapter)

    def test_ollama_capabilities_are_explicitly_unavailable(self):
        adapter = get_adapter("ollama")
        with self.assertRaises(CapabilityUnavailable):
            adapter.detect(Path("."))

    def test_cli_defaults_to_comfyui_adapter(self):
        args = parse_args(["detect"])
        self.assertEqual(args.adapter, "comfyui")

    def test_candidate_cli_uses_automatic_catalog_by_default(self):
        args = parse_args(
            ["candidates", "--target", "target", "--gfx", "gfx1201"]
        )
        self.assertIsNone(args.catalog)
        self.assertIsNone(args.catalog_url)
        self.assertFalse(args.refresh_catalog)
