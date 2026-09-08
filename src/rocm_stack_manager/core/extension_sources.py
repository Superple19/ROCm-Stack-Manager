"""Validate the pinned upstream sources used by ComfyUI extensions."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse


class ExtensionSourceError(ValueError):
    """Raised when the pinned extension source manifest is invalid."""


_MANIFEST_RESOURCE = "data/extension-sources.json"
_REVISION_TYPES = {"commit", "tag"}
_INSTALL_METHODS = {"matrix", "pypi", "source"}


def _read_manifest(path: Path | None) -> dict:
    if path is None:
        try:
            text = resources.files("rocm_stack_manager").joinpath(_MANIFEST_RESOURCE).read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError, OSError) as error:
            raise ExtensionSourceError("bundled extension source manifest is missing") from error
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ExtensionSourceError(f"cannot read extension source manifest: {path}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ExtensionSourceError("extension source manifest is invalid JSON") from error
    if not isinstance(document, dict):
        raise ExtensionSourceError("extension source manifest must be an object")
    return document


def _validate_repository(value, extension_id):
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.strip("/"):
        raise ExtensionSourceError(f"invalid official repository for {extension_id}")
    if parsed.query or parsed.fragment:
        raise ExtensionSourceError(f"repository URL must not contain query or fragment: {extension_id}")


def validate_extension_sources(document):
    if document.get("schema_version") != 1:
        raise ExtensionSourceError("unsupported extension source manifest schema")
    entries = document.get("extensions")
    if not isinstance(entries, list) or not entries:
        raise ExtensionSourceError("extension source manifest has no extensions")

    seen_ids = set()
    seen_packages = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ExtensionSourceError("extension source entry must be an object")
        required = {"id", "repository", "revision", "revision_type", "package_names", "install_methods"}
        if set(entry) != required:
            raise ExtensionSourceError(f"invalid extension source fields: {entry.get('id', 'unknown')}")
        extension_id = entry["id"]
        if not isinstance(extension_id, str) or not extension_id or extension_id in seen_ids:
            raise ExtensionSourceError(f"duplicate or invalid extension source ID: {extension_id}")
        seen_ids.add(extension_id)
        _validate_repository(entry["repository"], extension_id)

        revision_type = entry["revision_type"]
        revision = entry["revision"]
        if revision_type not in _REVISION_TYPES or not isinstance(revision, str) or not revision:
            raise ExtensionSourceError(f"invalid pinned revision: {extension_id}")
        if revision_type == "commit" and not (len(revision) == 40 and all(char in "0123456789abcdef" for char in revision)):
            raise ExtensionSourceError(f"commit revision must be a full SHA-1: {extension_id}")
        if revision_type == "tag" and revision.casefold() in {"main", "master", "develop", "latest", "head"}:
            raise ExtensionSourceError(f"moving branch is not a valid tag revision: {extension_id}")

        package_names = entry["package_names"]
        if not isinstance(package_names, list) or not package_names or any(not isinstance(name, str) or not name for name in package_names):
            raise ExtensionSourceError(f"extension source has invalid package names: {extension_id}")
        for package_name in package_names:
            normalized = package_name.casefold().replace("_", "-")
            if normalized in seen_packages:
                raise ExtensionSourceError(f"package belongs to multiple extension sources: {package_name}")
            seen_packages.add(normalized)

        install_methods = entry["install_methods"]
        if not isinstance(install_methods, list) or not install_methods or not set(install_methods).issubset(_INSTALL_METHODS):
            raise ExtensionSourceError(f"extension source has invalid install methods: {extension_id}")
    return document


def load_extension_sources(path: str | Path | None = None) -> dict[str, dict]:
    """Load the validated pinned extension source map."""

    document = validate_extension_sources(_read_manifest(Path(path) if path is not None else None))
    return {entry["id"]: entry for entry in document["extensions"]}
