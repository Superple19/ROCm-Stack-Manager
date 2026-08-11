"""Host platform normalization for supported target operations."""

import sys


SUPPORTED_PLATFORMS = ("windows", "linux")
UNSUPPORTED_PLATFORM = "unsupported_platform"


class UnsupportedPlatformError(ValueError):
    """Raised when a target operation is requested on an unsupported host."""


def host_platform(value=None):
    """Normalize a platform token without treating unknown hosts as Linux."""

    value = sys.platform if value is None else str(value)
    if value.startswith("win"):
        return "windows"
    if value.startswith("linux"):
        return "linux"
    return UNSUPPORTED_PLATFORM


def require_supported_host_platform(value=None):
    """Return the normalized host platform or raise a clear capability error."""

    resolved = host_platform(value)
    if resolved not in SUPPORTED_PLATFORMS:
        raise UnsupportedPlatformError(
            "unsupported host platform; ROCm Stack Manager supports Windows and Linux only"
        )
    return resolved
