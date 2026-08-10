"""ComfyUI-specific target detection hooks."""

from ...core.detection import detect_target


def detect_comfyui(path="."):
    """Detect a ComfyUI target using the shared filesystem detector."""

    return detect_target(path)
