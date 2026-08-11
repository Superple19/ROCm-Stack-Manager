"""Create target-local package backups before a mutating install."""

import json
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename

from .verify import _clean_environment


class BackupError(RuntimeError):
    """Raised when a target package backup cannot be created."""


CORE_BACKUP_SCHEMA_VERSION = 2
EXTENSION_BACKUP_SCHEMA_VERSION = 1
# Kept as the public core-backup version for callers that imported the old name.
BACKUP_SCHEMA_VERSION = CORE_BACKUP_SCHEMA_VERSION
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_EXACT_REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s+(?:--hash=sha256:[0-9a-fA-F]{64}\s*)+)?$"
)


@dataclass(frozen=True)
class BackupSnapshot:
    path: Path
    requirements_path: Path
    created_at: str
    requirements: tuple[str, ...]
    target_root: str | None = None
    python_executable: str | None = None
    requirements_sha256: str | None = None
    wheelhouse_path: Path | None = None
    wheelhouse_requirements_path: Path | None = None
    schema_version: int = CORE_BACKUP_SCHEMA_VERSION
    restore_mode: str = "network_version_pinned"
    wheelhouse_error: str | None = None

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "requirements_path": str(self.requirements_path),
            "created_at": self.created_at,
            "requirements": list(self.requirements),
            "target_root": self.target_root,
            "python_executable": self.python_executable,
            "requirements_sha256": self.requirements_sha256,
            "wheelhouse_path": str(self.wheelhouse_path) if self.wheelhouse_path else None,
            "wheelhouse_requirements_path": (
                str(self.wheelhouse_requirements_path) if self.wheelhouse_requirements_path else None
            ),
            "restore_mode": self.restore_mode,
            "wheelhouse_error": self.wheelhouse_error,
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
            "schema_version": EXTENSION_BACKUP_SCHEMA_VERSION,
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


def _exact_requirement_versions(requirements, *, label):
    versions = {}
    for requirement in requirements:
        match = _EXACT_REQUIREMENT_RE.fullmatch(str(requirement))
        if not match:
            raise BackupError(
                f"{label} contains a non-exact requirement that cannot be archived safely: {requirement}"
            )
        name = canonicalize_name(match.group(1))
        version = match.group(2)
        if name in versions and versions[name] != version:
            raise BackupError(f"{label} contains conflicting versions for {name}")
        versions[name] = version
    return versions


def _artifact_identity(path):
    try:
        package, version, _build, _tags = parse_wheel_filename(path.name)
        return canonicalize_name(str(package)), str(version)
    except (TypeError, ValueError):
        try:
            package, version = parse_sdist_filename(path.name)
        except (TypeError, ValueError) as error:
            raise BackupError(f"backup download has no supported package filename: {path.name}") from error
        return canonicalize_name(str(package)), str(version)


@dataclass(frozen=True)
class StagedBackupArtifacts:
    root: Path
    artifacts_path: Path
    requirements_path: Path
    files: tuple[dict, ...]


def _resolve_requirements_path(backup_path, value):
    root = backup_path.parent.resolve()
    candidate = Path(value) if value else backup_path.with_suffix(".txt")
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise BackupError("backup requirements path must stay beside the backup JSON") from error
    return candidate


def _resolve_backup_path(backup_path, value):
    if not value:
        return None
    root = backup_path.parent.resolve()
    candidate = (root / str(value)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise BackupError("backup wheelhouse path must stay beside the backup JSON") from error
    return candidate


def stage_backup_wheelhouse(target, backup, *, timeout=1800, runner=subprocess.run):
    """Download the pre-change environment into a disposable wheelhouse."""

    if target.python_executable is None:
        raise BackupError("target Python executable was not found")
    expected = _exact_requirement_versions(backup.requirements, label="backup requirements")
    if not expected:
        raise BackupError("backup contains no requirements to archive")
    root = Path(tempfile.mkdtemp(prefix=f".{backup.path.stem}-archive-", dir=backup.path.parent))
    artifacts_path = root / "wheelhouse"
    artifacts_path.mkdir()
    command = [
        str(target.python_executable),
        "-m",
        "pip",
        "download",
        "--dest",
        str(artifacts_path),
        "--no-input",
        "--disable-pip-version-check",
        "--requirement",
        str(backup.requirements_path),
    ]
    try:
        completed = runner(
            command,
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        shutil.rmtree(root, ignore_errors=True)
        raise BackupError(f"pre-change package archive failed: {type(error).__name__}: {error}") from error
    if completed.returncode != 0:
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        shutil.rmtree(root, ignore_errors=True)
        raise BackupError(f"pre-change package archive failed: {output[-1000:]}")

    files = []
    archived = {}
    for path in sorted(artifacts_path.iterdir()):
        if not path.is_file() or path.name.endswith(".metadata"):
            continue
        package, version = _artifact_identity(path)
        if package in archived and archived[package] != version:
            shutil.rmtree(root, ignore_errors=True)
            raise BackupError(f"pre-change archive contains conflicting versions for {package}")
        archived[package] = version
        files.append(
            {
                "filename": path.name,
                "package": package,
                "version": version,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if archived != expected:
        shutil.rmtree(root, ignore_errors=True)
        missing = sorted(set(expected) - set(archived))
        unexpected = sorted(set(archived) - set(expected))
        mismatched = sorted(
            name for name in set(expected).intersection(archived) if expected[name] != archived[name]
        )
        raise BackupError(
            "pre-change archive does not match recorded requirements: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    requirements_path = root / "requirements-hashed.txt"
    lines = [
        f"{name}=={version} "
        + " ".join(
            f"--hash=sha256:{item['sha256']}"
            for item in files
            if item["package"] == name and item["version"] == version
        )
        for name, version in sorted(expected.items())
    ]
    requirements_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return StagedBackupArtifacts(root, artifacts_path, requirements_path, tuple(files))


def attach_wheelhouse(backup, staged):
    """Atomically attach verified pre-change artifacts to a core backup."""

    expected = _exact_requirement_versions(backup.requirements, label="backup requirements")
    staged_requirements = tuple(
        line for line in staged.requirements_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    archived = _exact_requirement_versions(staged_requirements, label="wheelhouse requirements")
    if archived != expected:
        raise BackupError("wheelhouse requirements do not match the recorded pre-change environment")

    wheelhouse = backup.path.parent / f"{backup.path.stem}-wheelhouse"
    temporary_wheelhouse = backup.path.parent / f".{backup.path.stem}-wheelhouse.tmp"
    requirements_path = backup.path.parent / f"{backup.path.stem}-wheelhouse-requirements.txt"
    if wheelhouse.exists() or temporary_wheelhouse.exists() or requirements_path.exists():
        raise BackupError(f"backup wheelhouse already exists: {wheelhouse}")
    temporary_wheelhouse.mkdir(parents=True)
    try:
        for item in staged.files:
            source = staged.artifacts_path / item["filename"]
            if not source.is_file():
                raise BackupError(f"staged backup artifact is missing: {source}")
            shutil.copy2(source, temporary_wheelhouse / source.name)
        requirements_text = staged.requirements_path.read_text(encoding="utf-8")
        document = json.loads(backup.path.read_text(encoding="utf-8"))
        document["schema_version"] = CORE_BACKUP_SCHEMA_VERSION
        document["wheelhouse_role"] = "pre_change_environment"
        document["restore_mode"] = "offline_hash_verified"
        document["wheelhouse_path"] = wheelhouse.name
        document["wheelhouse_requirements_path"] = requirements_path.name
        document["wheelhouse_manifest"] = {
            "schema_version": 1,
            "files": [
                {
                    "filename": item["filename"],
                    "sha256": hashlib.sha256(
                        (temporary_wheelhouse / item["filename"]).read_bytes()
                    ).hexdigest(),
                }
                for item in staged.files
            ],
        }
        os.replace(temporary_wheelhouse, wheelhouse)
        _atomic_write_text(requirements_path, requirements_text)
        document["wheelhouse_requirements_sha256"] = hashlib.sha256(
            requirements_path.read_bytes()
        ).hexdigest()
        _atomic_write(backup.path, document)
    except (BackupError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        shutil.rmtree(temporary_wheelhouse, ignore_errors=True)
        shutil.rmtree(wheelhouse, ignore_errors=True)
        requirements_path.unlink(missing_ok=True)
        raise
    return BackupSnapshot(
        path=backup.path,
        requirements_path=backup.requirements_path,
        created_at=backup.created_at,
        requirements=backup.requirements,
        target_root=backup.target_root,
        python_executable=backup.python_executable,
        requirements_sha256=backup.requirements_sha256,
        wheelhouse_path=wheelhouse,
        wheelhouse_requirements_path=requirements_path,
        schema_version=CORE_BACKUP_SCHEMA_VERSION,
        restore_mode="offline_hash_verified",
    )


def _validate_wheelhouse(backup_path, document, requirements):
    wheelhouse = _resolve_backup_path(backup_path, document.get("wheelhouse_path"))
    hashed_requirements = _resolve_backup_path(backup_path, document.get("wheelhouse_requirements_path"))
    manifest = document.get("wheelhouse_manifest")
    if wheelhouse is None and hashed_requirements is None and manifest is None:
        return None, None
    if document.get("schema_version") == CORE_BACKUP_SCHEMA_VERSION:
        if document.get("wheelhouse_role") != "pre_change_environment":
            raise BackupError(f"backup wheelhouse has no pre-change role: {backup_path}")
        if document.get("restore_mode") != "offline_hash_verified":
            raise BackupError(f"backup wheelhouse has an invalid restore mode: {backup_path}")
    expected_requirements_hash = document.get("wheelhouse_requirements_sha256")
    if (
        wheelhouse is None
        or hashed_requirements is None
        or not isinstance(manifest, dict)
        or not isinstance(expected_requirements_hash, str)
        or not _SHA256_RE.fullmatch(expected_requirements_hash)
    ):
        raise BackupError(f"backup wheelhouse metadata is incomplete: {backup_path}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise BackupError(f"backup wheelhouse manifest has no artifacts: {backup_path}")
    if not wheelhouse.is_dir() or not hashed_requirements.is_file():
        raise BackupError(f"backup wheelhouse is missing: {wheelhouse}")
    if hashlib.sha256(hashed_requirements.read_bytes()).hexdigest() != expected_requirements_hash.casefold():
        raise BackupError(f"backup wheelhouse requirements hash mismatch: {hashed_requirements}")
    try:
        hashed_lines = tuple(
            line
            for line in hashed_requirements.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeDecodeError) as error:
        raise BackupError(
            f"backup wheelhouse requirements are not valid UTF-8: {hashed_requirements}"
        ) from error
    if _exact_requirement_versions(
        requirements, label="backup requirements"
    ) != _exact_requirement_versions(hashed_lines, label="wheelhouse requirements"):
        raise BackupError("backup wheelhouse requirements do not match the recorded environment")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str) or not _SHA256_RE.fullmatch(str(item.get("sha256", ""))):
            raise BackupError(f"backup wheelhouse manifest is invalid: {backup_path}")
        artifact = (wheelhouse / item["filename"]).resolve()
        try:
            artifact.relative_to(wheelhouse.resolve())
        except ValueError as error:
            raise BackupError(f"backup wheelhouse artifact escapes its directory: {item['filename']}") from error
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
            raise BackupError(f"backup wheelhouse artifact hash mismatch: {item['filename']}")
    return wheelhouse, hashed_requirements


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
            "schema_version": CORE_BACKUP_SCHEMA_VERSION,
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
        schema_version=CORE_BACKUP_SCHEMA_VERSION,
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
        "schema_version": EXTENSION_BACKUP_SCHEMA_VERSION,
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
    schema_version = document.get("schema_version")
    supported_versions = (
        {EXTENSION_BACKUP_SCHEMA_VERSION}
        if extensions
        else {1, CORE_BACKUP_SCHEMA_VERSION}
    )
    if schema_version not in supported_versions:
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


def load_backup(path, *, materialize=True, allow_network_restore=False):
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
    schema_version = document.get("schema_version")
    if not isinstance(schema_version, int):
        raise BackupError(f"backup schema version is invalid: {backup_path}")
    wheelhouse_error = None
    try:
        wheelhouse_path, wheelhouse_requirements_path = _validate_wheelhouse(
            backup_path, document, requirements
        )
        if wheelhouse_path is None:
            raise BackupError("backup has no hash-verified pre-change wheelhouse")
    except BackupError as error:
        if "stay beside" in str(error) or "escapes its directory" in str(error):
            raise
        if not allow_network_restore:
            raise BackupError(
                f"{error}; pass --allow-network-restore to use the recorded versions from the network"
            ) from error
        wheelhouse_path = None
        wheelhouse_requirements_path = None
        wheelhouse_error = str(error)
    return BackupSnapshot(
        path=backup_path,
        requirements_path=requirements_path,
        created_at=document.get("created_at", ""),
        requirements=requirements,
        target_root=document.get("target_root"),
        python_executable=document.get("python_executable"),
        requirements_sha256=expected_hash,
        wheelhouse_path=wheelhouse_path,
        wheelhouse_requirements_path=wheelhouse_requirements_path,
        schema_version=schema_version,
        restore_mode=(
            "offline_hash_verified" if wheelhouse_path else "network_version_pinned"
        ),
        wheelhouse_error=wheelhouse_error,
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
    current_version = (
        EXTENSION_BACKUP_SCHEMA_VERSION
        if selected_kind == "extensions"
        else CORE_BACKUP_SCHEMA_VERSION
    )
    if document.get("schema_version") == current_version and document.get("requirements_sha256"):
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
        "schema_version": current_version,
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
        else load_backup(
            destination_path,
            materialize=False,
            allow_network_restore=True,
        )
    )
