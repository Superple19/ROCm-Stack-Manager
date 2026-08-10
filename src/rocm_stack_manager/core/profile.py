"""Load and apply machine-readable application profile constraints."""

import json
from pathlib import Path


class ProfileError(ValueError):
    """Raised when an application profile cannot be loaded."""


def load_profile(path):
    profile_path = Path(path).expanduser().resolve()
    try:
        with profile_path.open(encoding="utf-8") as handle:
            profile = json.load(handle)
    except OSError as error:
        raise ProfileError(f"cannot read profile: {profile_path}") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid profile: {profile_path}") from error
    if profile.get("id") != "comfyui" or profile.get("profile_kind") != "application":
        raise ProfileError(f"unsupported ComfyUI profile: {profile_path}")
    if not isinstance(profile.get("constraints"), list):
        raise ProfileError(f"ComfyUI profile has no constraints: {profile_path}")
    return profile


def _constraints(profile):
    return {constraint.get("id"): constraint for constraint in profile.get("constraints", ())}


def evaluate_candidate(candidate, profile):
    """Evaluate documented profile scope without claiming runtime success."""

    if not profile:
        return "unknown", (), None
    constraints = _constraints(profile)
    warnings = []

    supported_platforms = constraints.get("supported-platform", {}).get("value")
    if isinstance(supported_platforms, list) and candidate.get("platform") not in supported_platforms:
        warnings.append(f"platform {candidate.get('platform')} is outside the ComfyUI profile")

    channel_value = constraints.get("rocm-channel", {}).get("value", {})
    allowed_channels = channel_value.get("allowed") if isinstance(channel_value, dict) else None
    if isinstance(allowed_channels, list) and candidate.get("channel") not in allowed_channels:
        warnings.append(f"channel {candidate.get('channel')} is outside the ComfyUI profile")

    package_value = constraints.get("rocm-package-candidate", {}).get("value", {})
    if isinstance(package_value, dict):
        families = package_value.get("distribution_families")
        if isinstance(families, list) and candidate.get("distribution_family") not in families:
            warnings.append(f"distribution family {candidate.get('distribution_family')} is outside the ComfyUI profile")

    torch_value = constraints.get("torch-rocm", {}).get("value", {})
    allowed_series = torch_value.get("torch_series") if isinstance(torch_value, dict) else None
    torch_version = str(candidate.get("torch_version") or "")
    torch_series = torch_version.split("+", 1)[0].split(".")
    torch_series = ".".join(torch_series[:2]) if len(torch_series) >= 2 else ""
    if isinstance(allowed_series, list) and torch_series and torch_series not in allowed_series:
        warnings.append(f"Torch series {torch_series} is outside the documented ComfyUI profile")

    return ("out_of_profile" if warnings else "documented"), tuple(warnings), profile.get("id")
