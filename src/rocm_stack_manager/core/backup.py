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
            "schema_version": 1,
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
            "schema_version": 1,
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
        "schema_version": 1,
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


def load_backup(path, *, materialize=True):
    """Load and validate a JSON package backup."""

    backup_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(backup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackupError(f"cannot read backup: {backup_path}") from error
    requirements = tuple(document.get("requirements") or ())
    requirements_path = _resolve_requirements_path(backup_path, document.get("requirements_path"))
    if not requirements:
        raise BackupError(f"backup contains no requirements: {backup_path}")
    if not requirements_path.is_file() and materialize:
        sibling_path = backup_path.with_suffix(".txt")
        if requirements_path != sibling_path:
            requirements_path = sibling_path
        _atomic_write_text(requirements_path, _requirements_text(requirements))
    if not requirements_path.is_file():
        raise BackupError(f"backup requirements file is missing: {requirements_path}")
    expected_hash = document.get("requirements_sha256")
    if expected_hash and hashlib.sha256(requirements_path.read_bytes()).hexdigest() != expected_hash:
        raise BackupError(f"backup requirements hash mismatch: {requirements_path}")
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
    """Load and validate an extension-only backup."""

    backup_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(backup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackupError(f"cannot read extension backup: {backup_path}") from error
    if document.get("kind") != "extensions":
        raise BackupError(f"backup is not an extension backup: {backup_path}")
    requirements = tuple(document.get("requirements") or ())
    requirements_path = _resolve_requirements_path(backup_path, document.get("requirements_path"))
    if not requirements_path.is_file() and materialize:
        _atomic_write_text(requirements_path, _requirements_text(requirements))
    if not requirements_path.is_file():
        raise BackupError(f"extension backup requirements file is missing: {requirements_path}")
    expected_hash = document.get("requirements_sha256")
    if expected_hash and hashlib.sha256(requirements_path.read_bytes()).hexdigest() != expected_hash:
        raise BackupError(f"extension backup requirements hash mismatch: {requirements_path}")
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
