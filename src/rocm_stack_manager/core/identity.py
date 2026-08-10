"""Stable identities for core and extension evidence records."""

import hashlib
import json


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def candidate_hash(candidate):
    """Hash only immutable core candidate identity fields."""

    identity = {
        "channel": candidate.get("channel"),
        "distribution_family": candidate.get("distribution_family"),
        "gfx": candidate.get("gfx"),
        "index_url": candidate.get("index_url"),
        "package_specs": sorted(candidate.get("package_specs") or ()),
        "platform": candidate.get("platform"),
        "python_tag": candidate.get("python_tag"),
        "rocm_version": candidate.get("rocm_version"),
        "torch_version": candidate.get("torch_version"),
        "torchvision_version": candidate.get("torchvision_version"),
        "wheel_urls": sorted(candidate.get("wheel_urls") or ()),
    }
    return _digest(identity)


def extension_candidate_id(extension, version, python_tag, platform_tag, source_id, artifact_url):
    """Match Matrix's deterministic identity for one exact extension artifact."""

    identity = {
        "artifact_url": str(artifact_url),
        "extension": str(extension),
        "platform_tag": str(platform_tag),
        "python_tag": str(python_tag),
        "source_id": str(source_id),
        "version": str(version),
    }
    digest = _digest(identity)[:16]
    safe = lambda value: str(value).replace(":", "_")
    return f"extension:{safe(extension)}:{safe(version)}:{safe(python_tag)}:{safe(platform_tag)}:{digest}"
