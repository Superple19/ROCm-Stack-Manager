"""Stage exact candidate artifacts before a mutating installation."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse

from packaging.utils import parse_sdist_filename, parse_wheel_filename

from .identity import candidate_hash
from .verify import _clean_environment


class StagingError(RuntimeError):
    """Raised when candidate artifacts cannot be staged safely."""


@dataclass(frozen=True)
class StagedCandidate:
    root: Path
    artifacts_path: Path
    requirements_path: Path
    manifest_path: Path
    files: tuple[dict, ...]

    @property
    def install_command(self):
        return (
            "--no-index",
            "--find-links",
            str(self.artifacts_path),
            "--require-hashes",
            "--requirement",
            str(self.requirements_path),
        )


def _atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_identity(path):
    name = path.name
    try:
        package, version, build, tags = parse_wheel_filename(name)
        return str(package), str(version), {"kind": "wheel", "build": list(build), "tags": sorted(str(tag) for tag in tags)}
    except (ValueError, TypeError):
        try:
            package, version = parse_sdist_filename(name)
        except (ValueError, TypeError) as error:
            raise StagingError(f"downloaded artifact has no supported package filename: {name}") from error
        return str(package), str(version), {"kind": "sdist"}


def _requirements(files):
    grouped = {}
    for item in files:
        key = (item["package"], item["version"])
        grouped.setdefault(key, set()).add(item["sha256"])
    lines = []
    for (package, version), hashes in sorted(grouped.items()):
        lines.append(f"{package}=={version} " + " ".join(f"--hash=sha256:{digest}" for digest in sorted(hashes)))
    return "\n".join(lines) + ("\n" if lines else "")


def stage_candidate(target, candidate, destination=None, *, timeout=1800, runner=subprocess.run):
    """Download a candidate and its resolved dependencies into a hash manifest."""

    if target.python_executable is None:
        raise StagingError("target Python executable was not found")
    source_values = candidate.get("wheel_urls") or candidate.get("package_specs") or ()
    sources: tuple[str, ...] = tuple(str(value) for value in source_values)
    if not sources:
        raise StagingError("candidate has no package sources to stage")
    digest = candidate_hash(candidate)
    root = Path(destination) if destination else target.root / ".rocm-stack-manager" / "staging" / digest
    artifacts_path = root / "wheelhouse"
    requirements_path = root / "requirements-hashed.txt"
    manifest_path = root / "manifest.json"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    for existing in artifacts_path.iterdir():
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()
    command = [
        str(target.python_executable),
        "-m",
        "pip",
        "download",
        "--dest",
        str(artifacts_path),
        "--no-input",
        "--disable-pip-version-check",
    ]
    if candidate.get("index_url") and not candidate.get("wheel_urls"):
        command.extend(("--extra-index-url", candidate["index_url"]))
    command.extend(sources)
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
        raise StagingError(f"candidate staging failed: {type(error).__name__}: {error}") from error
    if completed.returncode != 0:
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        raise StagingError(f"candidate staging failed: {output[-1000:]}")
    files = []
    for path in sorted(artifacts_path.iterdir()):
        if not path.is_file() or path.name.endswith(".metadata"):
            continue
        package, version, metadata = _artifact_identity(path)
        source = None
        for candidate_source in sources:
            if Path(urlparse(candidate_source).path).name == path.name:
                source = candidate_source
                break
        files.append(
            {
                "filename": path.name,
                "package": package,
                "version": version,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source": source,
                **metadata,
            }
        )
    if not files:
        raise StagingError("candidate staging returned no installable artifacts")
    requirements_path.write_text(_requirements(files), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "candidate_hash": digest,
        "candidate_id": candidate.get("id"),
        "target_root": str(target.root),
        "target_python": str(target.python_executable),
        "sources": list(sources),
        "requirements_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        "files": files,
    }
    _atomic_write(manifest_path, manifest)
    return StagedCandidate(root, artifacts_path, requirements_path, manifest_path, tuple(files))


def validate_staged_candidate(staged):
    """Validate every staged file against its manifest before installation."""

    try:
        document = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StagingError(f"cannot read staging manifest: {staged.manifest_path}") from error
    items = document.get("files")
    if document.get("schema_version") != 1 or not isinstance(items, list):
        raise StagingError(f"staging manifest is invalid: {staged.manifest_path}")
    if not staged.requirements_path.is_file():
        raise StagingError(f"staging requirements file is missing: {staged.requirements_path}")
    requirements_hash = document.get("requirements_sha256")
    if not isinstance(requirements_hash, str) or hashlib.sha256(staged.requirements_path.read_bytes()).hexdigest() != requirements_hash:
        raise StagingError(f"staging requirements hash mismatch: {staged.requirements_path}")
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise StagingError(f"staging manifest contains an invalid artifact: {staged.manifest_path}")
        path = staged.artifacts_path / item["filename"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise StagingError(f"staged artifact hash mismatch: {path.name}")
    return True
