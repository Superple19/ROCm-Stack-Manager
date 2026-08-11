"""Create target-local package backups before a mutating install."""

import json
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .verify import _clean_environment


class BackupError(RuntimeError):
    """Raised when a target package backup cannot be created."""


BACKUP_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(frozen=True)
class BackupSnapshot:
    path: Path
    requirements_path: Path
    created_at: str
    requirements: tuple[str, ...]
    target_root: str | None = None
    python_executable: str | None = None
    requirements_sha256: str | None = None

    def as_dict(self):
        return {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "path": str(self.path),
            "requirements_path": str(self.requirements_path),
            "created_at": self.created_at,
            "requirements": list(self.requirements),
            "target_root": self.target_root,
            "python_executable": self.python_executable,
            "requirements_sha256": self.requirements_sha256,
        }


@dataclass(frozen=True)
class ExtensionBackupSnapshot:
    """Target-local backup containing only selected extension packages."""

    path: Path
    requirements_path: Path
    created_at: str
    requirements: tuple[str, ...]
    extension_ids: tuple[str, ...] = ()
    candidate_id: str | None = None
    target_root: str | None = None
    python_executable: str | None = None
    requirements_sha256: str | None = None

    def as_dict(self):
        return {
            "kind": "extensions",
            "schema_version": BACKUP_SCHEMA_VERSION,
            "path": str(self.path),
            "requirements_path": str(self.requirements_path),
            "created_at": self.created_at,
            "requirements": list(self.requirements),
            "extension_ids": list(self.extension_ids),
            "candidate_id": self.candidate_id,
            "target_root": self.target_root,
            "python_executable": self.python_executable,
            "requirements_sha256": self.requirements_sha256,
        }


def _atomic_write(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _requirements_text(requirements, trailing_newline=True):
    text = "\n".join(requirements)
    if trailing_newline and text:
        text += "\n"
    return text


def _resolve_requirements_path(backup_path, value):
    root = backup_path.parent.resolve()
    candidate = Path(value) if value else backup_path.with_suffix(".txt")
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise BackupError("backup requirements path must stay beside the backup JSON") from error
    return candidate


def create_backup(target, destination=None, timeout=60):
    """Save target Python's exact ``pip freeze --all`` output."""

    if target.python_executable is None:
        raise BackupError("target Python executable was not found")
    try:
        completed = subprocess.run(
            [str(target.python_executable), "-m", "pip", "freeze", "--all"],
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BackupError(f"package backup failed: {type(error).__name__}: {error}") from error
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "pip freeze failed").strip()
        raise BackupError(error[-1000:])

    requirements = tuple(line for line in completed.stdout.splitlines() if line.strip())
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backup_dir = Path(destination) if destination else target.root / ".rocm-stack-manager" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    filename = "packages-" + created_at.replace(":", "").replace("-", "") + ".json"
    path = backup_dir / filename
    requirements_path = path.with_suffix(".txt")
    requirements_text = _requirements_text(requirements)
    _atomic_write_text(requirements_path, requirements_text)
    requirements_sha256 = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
    _atomic_write(
        path,
        {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "created_at": created_at,
            "target_root": str(target.root),
            "python_executable": str(target.python_executable),
            "requirements_path": requirements_path.name,
            "requirements": list(requirements),
            "requirements_sha256": requirements_sha256,
        },
    )
    return BackupSnapshot(
        path=path,
        requirements_path=requirements_path,
        created_at=created_at,
        requirements=requirements,
        target_root=str(target.root),
        python_executable=str(target.python_executable),
        requirements_sha256=requirements_sha256,
    )


def _normalize_package_name(value):
    return value.casefold().replace("_", "-").replace(".", "-")


def _freeze_package_name(requirement):
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*==", requirement)
    return _normalize_package_name(match.group(1)) if match else None


def create_extension_backup(
    target,
    package_names,
    *,
    extension_ids=(),
    candidate_id=None,
    destination=None,
    timeout=60,
):
    """Save only selected extension packages from the target interpreter."""

    if target.python_executable is None:
        raise BackupError("target Python executable was not found")
    try:
        completed = subprocess.run(
            [str(target.python_executable), "-m", "pip", "freeze", "--all"],
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BackupError(f"extension backup failed: {type(error).__name__}: {error}") from error
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "pip freeze failed").strip()
        raise BackupError(error[-1000:])

    selected_names = {_normalize_package_name(str(name)) for name in package_names}
    requirements = tuple(
        line
        for line in completed.stdout.splitlines()
        if line.strip() and _freeze_package_name(line) in selected_names
    )
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backup_dir = Path(destination) if destination else target.root / ".rocm-stack-manager" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    filename = "extensions-" + created_at.replace(":", "").replace("-", "") + ".json"
    path = backup_dir / filename
    requirements_path = path.with_suffix(".txt")
    requirements_text = _requirements_text(requirements)
    _atomic_write_text(requirements_path, requirements_text)
    requirements_sha256 = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
    document = {
        "kind": "extensions",
        "schema_version": BACKUP_SCHEMA_VERSION,
        "created_at": created_at,
        "target_root": str(target.root),
        "python_executable": str(target.python_executable),
        "requirements_path": requirements_path.name,
        "requirements": list(requirements),
        "extension_ids": list(extension_ids),
        "candidate_id": candidate_id,
        "requirements_sha256": requirements_sha256,
    }
    _atomic_write(path, document)
    return ExtensionBackupSnapshot(
        path=path,
        requirements_path=requirements_path,
        created_at=created_at,
        requirements=requirements,
        extension_ids=tuple(extension_ids),
        candidate_id=candidate_id,
        target_root=str(target.root),
        python_executable=str(target.python_executable),
        requirements_sha256=requirements_sha256,
    )


def _read_backup_document(path):
    backup_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(backup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackupError(f"cannot read backup: {backup_path}") from error
    if not isinstance(document, dict):
        raise BackupError(f"backup document must be an object: {backup_path}")
    return backup_path, document


def _validated_requirements(backup_path, document, *, extensions):
    if document.get("schema_version") != BACKUP_SCHEMA_VERSION:
        raise BackupError(
            f"legacy backup schema at {backup_path}; run migrate-backup before restoring"
        )
    if extensions and document.get("kind") != "extensions":
        raise BackupError(f"backup is not an extension backup: {backup_path}")
    if not extensions and document.get("kind") == "extensions":
        raise BackupError(f"backup is an extension backup: {backup_path}")
    requirements_value = document.get("requirements")
    if not isinstance(requirements_value, list) or any(
        not isinstance(requirement, str) or not requirement.strip()
        for requirement in requirements_value
    ):
        raise BackupError(f"backup requirements must be a list of strings: {backup_path}")
    requirements = tuple(requirements_value)
    if not extensions and not requirements:
        raise BackupError(f"backup contains no requirements: {backup_path}")
    expected_hash = document.get("requirements_sha256")
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        raise BackupError(f"backup requirements hash is missing or invalid: {backup_path}")
    requirements_path = _resolve_requirements_path(backup_path, document.get("requirements_path"))
    if not requirements_path.is_file():
        raise BackupError(f"backup requirements file is missing: {requirements_path}")
    requirements_bytes = requirements_path.read_bytes()
    if hashlib.sha256(requirements_bytes).hexdigest() != expected_hash.casefold():
        raise BackupError(f"backup requirements hash mismatch: {requirements_path}")
    try:
        recorded_text = requirements_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise BackupError(f"backup requirements file is not UTF-8: {requirements_path}") from error
    if recorded_text != _requirements_text(requirements):
        raise BackupError(f"backup requirements do not match JSON record: {requirements_path}")
    return requirements, requirements_path, expected_hash.casefold()


def load_backup(path, *, materialize=True):
    """Load a current package backup without changing it.

    ``materialize`` is retained for API compatibility. Current backups must
    already contain their hashed requirements sidecar; missing files require
    explicit ``migrate-backup`` instead of being created during restore.
    """

    del materialize
    backup_path, document = _read_backup_document(path)
    requirements, requirements_path, expected_hash = _validated_requirements(
        backup_path, document, extensions=False
    )
    return BackupSnapshot(
        path=backup_path,
        requirements_path=requirements_path,
        created_at=document.get("created_at", ""),
        requirements=requirements,
        target_root=document.get("target_root"),
        python_executable=document.get("python_executable"),
        requirements_sha256=expected_hash,
    )


def load_extension_backup(path, *, materialize=True):
    """Load a current extension backup without changing it."""

    del materialize
    backup_path, document = _read_backup_document(path)
    requirements, requirements_path, expected_hash = _validated_requirements(
        backup_path, document, extensions=True
    )
    return ExtensionBackupSnapshot(
        path=backup_path,
        requirements_path=requirements_path,
        created_at=document.get("created_at", ""),
        requirements=requirements,
        extension_ids=tuple(document.get("extension_ids") or ()),
        candidate_id=document.get("candidate_id"),
        target_root=document.get("target_root"),
        python_executable=document.get("python_executable"),
        requirements_sha256=expected_hash,
    )


def migrate_backup(path, destination, *, kind=None):
    """Convert a legacy backup into the current hashed sidecar format.

    Migration never overwrites the source backup and does not install or
    restore packages. The returned snapshot is validated through the same
    strict loader used by restore.
    """

    source_path, document = _read_backup_document(path)
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise BackupError("legacy migration requires an output path different from the source")
    document_kind = document.get("kind")
    if document_kind not in (None, "extensions"):
        raise BackupError(f"unsupported legacy backup kind: {document_kind}")
    inferred_kind = "extensions" if document_kind == "extensions" else "core"
    selected_kind = kind or inferred_kind
    if selected_kind not in {"core", "extensions"}:
        raise BackupError(f"unsupported backup kind: {selected_kind}")
    if kind and document_kind and kind != document_kind:
        raise BackupError(f"backup kind does not match --kind: {source_path}")
    if document.get("schema_version") == BACKUP_SCHEMA_VERSION and document.get("requirements_sha256"):
        raise BackupError(f"backup already uses the current schema: {source_path}")
    requirements_value = document.get("requirements")
    if not isinstance(requirements_value, list) or any(
        not isinstance(requirement, str) or not requirement.strip()
        for requirement in requirements_value
    ):
        raise BackupError(f"legacy backup requirements must be a list of strings: {source_path}")
    requirements = tuple(requirements_value)
    if selected_kind == "core" and not requirements:
        raise BackupError(f"legacy backup contains no requirements: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    requirements_path = destination_path.with_suffix(".txt")
    _atomic_write_text(requirements_path, _requirements_text(requirements))
    requirements_sha256 = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
    migrated = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "created_at": document.get("created_at", ""),
        "target_root": document.get("target_root"),
        "python_executable": document.get("python_executable"),
        "requirements_path": requirements_path.name,
        "requirements": list(requirements),
        "requirements_sha256": requirements_sha256,
    }
    if selected_kind == "extensions":
        migrated.update(
            {
                "kind": "extensions",
                "extension_ids": list(document.get("extension_ids") or ()),
                "candidate_id": document.get("candidate_id"),
            }
        )
    _atomic_write(destination_path, migrated)
    return (
        load_extension_backup(destination_path, materialize=False)
        if selected_kind == "extensions"
        else load_backup(destination_path, materialize=False)
    )
