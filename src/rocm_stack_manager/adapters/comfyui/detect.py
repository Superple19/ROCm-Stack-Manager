"""ComfyUI-specific target detection hooks."""

from ...core.detection import detect_target
from ...core.verify import probe_target, target_python_tag


def detect_comfyui(path="."):
    """Detect a ComfyUI target using the shared filesystem detector."""

    return detect_target(path)


def verify_comfyui(target):
    """Probe runtime and hardware through the selected ComfyUI Python."""

    return probe_target(target)


def python_tag_comfyui(target):
    """Return the target interpreter's CPython wheel tag."""

    return target_python_tag(target)
