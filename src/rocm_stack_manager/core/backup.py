"""Create target-local package backups before a mutating install."""

import json
import os
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

    def as_dict(self):
        return {
            "path": str(self.path),
            "requirements_path": str(self.requirements_path),
            "created_at": self.created_at,
            "requirements": list(self.requirements),
            "target_root": self.target_root,
            "python_executable": self.python_executable,
        }


def _atomic_write(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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
    _atomic_write_text(requirements_path, "\n".join(requirements) + "\n")
    _atomic_write(
        path,
        {
            "schema_version": 1,
            "created_at": created_at,
            "target_root": str(target.root),
            "python_executable": str(target.python_executable),
            "requirements_path": requirements_path.name,
            "requirements": list(requirements),
        },
    )
    return BackupSnapshot(
        path=path,
        requirements_path=requirements_path,
        created_at=created_at,
        requirements=requirements,
        target_root=str(target.root),
        python_executable=str(target.python_executable),
    )


def load_backup(path):
    """Load and validate a JSON package backup."""

    backup_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(backup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackupError(f"cannot read backup: {backup_path}") from error
    requirements = tuple(document.get("requirements") or ())
    requirements_value = document.get("requirements_path")
    if requirements_value:
        requirements_path = Path(requirements_value)
        if not requirements_path.is_absolute():
            requirements_path = backup_path.parent / requirements_path
    else:
        requirements_path = backup_path.with_suffix(".txt")
    if not requirements:
        raise BackupError(f"backup contains no requirements: {backup_path}")
    if not requirements_path.is_file():
        sibling_path = backup_path.with_suffix(".txt")
        if requirements_path != sibling_path:
            requirements_path = sibling_path
        _atomic_write_text(requirements_path, "\n".join(requirements) + "\n")
    return BackupSnapshot(
        path=backup_path,
        requirements_path=requirements_path,
        created_at=document.get("created_at", ""),
        requirements=requirements,
        target_root=document.get("target_root"),
        python_executable=document.get("python_executable"),
    )
