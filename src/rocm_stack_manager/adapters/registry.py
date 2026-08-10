"""Registry for application-specific ROCm adapters."""

from .comfyui.adapter import ComfyUIAdapter
from .ollama.adapter import OllamaAdapter


_ADAPTERS = {
    "comfyui": ComfyUIAdapter,
    "ollama": OllamaAdapter,
}


def available_adapters():
    """Return registered adapter identifiers in deterministic order."""

    return tuple(sorted(_ADAPTERS))


def get_adapter(name):
    """Create an adapter by identifier."""

    try:
        adapter_type = _ADAPTERS[name]
    except KeyError as error:
        choices = ", ".join(available_adapters())
        raise ValueError(f"unknown adapter {name!r}; available adapters: {choices}") from error
    return adapter_type()
