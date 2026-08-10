"""Read and filter local ROCm Evidence Matrix catalog snapshots."""

import json
from pathlib import Path


class CatalogError(ValueError):
    """Raised when a Matrix catalog cannot be loaded or interpreted."""


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise CatalogError(f"cannot read catalog: {path}") from error
    except json.JSONDecodeError as error:
        raise CatalogError(f"invalid JSON catalog: {path}") from error


def load_catalog(path):
    """Load a Matrix index or matrix snapshot from a local path.

    A Matrix ``data/catalog.json`` is an index of generated artifacts. The
    manager consumes its ``compatibility_matrix`` artifact automatically, so
    callers may also pass ``data/matrix.json`` directly.
    """

    catalog_path = Path(path).expanduser().resolve()
    document = _read_json(catalog_path)
    if "targets" in document:
        return document

    artifacts = document.get("artifacts", [])
    matrix_artifact = next(
        (item for item in artifacts if item.get("id") == "compatibility_matrix"),
        None,
    )
    if not matrix_artifact or not matrix_artifact.get("path"):
        raise CatalogError("catalog does not reference a compatibility matrix")

    relative_path = Path(matrix_artifact["path"])
    path_candidates = [catalog_path.parent / relative_path]
    if catalog_path.parent.name == "data":
        path_candidates.append(catalog_path.parent.parent / relative_path)
    matrix_path = next((candidate.resolve() for candidate in path_candidates if candidate.is_file()), None)
    if matrix_path is None:
        raise CatalogError(f"compatibility matrix is missing: {path_candidates[-1].resolve()}")
    matrix = _read_json(matrix_path)
    if not isinstance(matrix.get("targets"), list):
        raise CatalogError(f"compatibility matrix has no targets: {matrix_path}")
    return matrix


def _candidate_id(platform, channel, gfx, rocm_version, torch_version):
    values = (platform, channel, gfx, rocm_version or "unknown", torch_version or "unknown")
    return "therock:" + ":".join(str(value).replace(":", "_") for value in values)


def _candidate_from_channel(target, platform, channel, details):
    rocm_version = details.get("rocm_device_version")
    torch_version = details.get("torch_device_version")
    torchvision_version = details.get("torchvision_device_version")
    available = bool(details.get("all_device_packages_available"))
    return {
        "id": _candidate_id(platform, channel, target["gfx"], rocm_version, torch_version),
        "distribution_family": "therock",
        "platform": platform,
        "channel": channel,
        "gfx": target["gfx"],
        "rocm_version": rocm_version,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "source_id": details.get("source_id"),
        "artifact_available": available,
        "status": "artifact_available" if available else "artifact_unavailable",
    }


def iter_candidates(catalog, *, platform, gfx=None, channel=None, include_unavailable=False):
    """Return install candidates for one platform and optional filters.

    The Matrix records package channels either under a platform or, in older
    snapshots, directly on a target. Both shapes are accepted. This function
    only reports artifact evidence; it never promotes a candidate to runtime or
    hardware compatibility.
    """

    candidates = []
    for target in catalog.get("targets", []):
        target_gfx = target.get("gfx")
        if gfx and target_gfx != gfx:
            continue
        platform_record = (target.get("platforms") or {}).get(platform)
        channels = (platform_record or {}).get("package_channels")
        if channels is None:
            channels = target.get("package_channels", {}) if platform == "windows" else {}
        for channel_name, details in channels.items():
            if channel and channel_name != channel:
                continue
            candidate = _candidate_from_channel(target, platform, channel_name, details or {})
            if include_unavailable or candidate["artifact_available"]:
                candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            item["gfx"],
            {"stable": 0, "nightly": 1, "staging": 2}.get(item["channel"], 9),
            item["rocm_version"] or "",
            item["torch_version"] or "",
        ),
    )


def find_candidate(catalog, candidate_id):
    """Find an exact candidate ID, including unavailable artifact records."""

    for platform in ("windows", "linux"):
        for candidate in iter_candidates(catalog, platform=platform, include_unavailable=True):
            if candidate["id"] == candidate_id:
                return candidate
    raise CatalogError(f"candidate not found: {candidate_id}")
