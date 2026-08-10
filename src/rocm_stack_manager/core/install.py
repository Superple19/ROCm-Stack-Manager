"""Create and optionally apply safe, target-scoped install plans."""

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .backup import BackupSnapshot
from .planning import InstallPlan, PlanningError
from .verify import _clean_environment


class InstallationError(PlanningError):
    """Raised when a candidate cannot produce a safe install plan."""


@dataclass(frozen=True)
class InstallResult:
    """Result of a dry-run or explicitly applied package operation."""

    plan: InstallPlan
    applied: bool = False
    returncode: int | None = None
    backup_path: str | None = None
    output: str = ""

    def as_dict(self):
        values = self.plan.as_dict()
        values.update(
            {
                "applied": self.applied,
                "returncode": self.returncode,
                "backup_path": self.backup_path,
                "output": self.output,
            }
        )
        return values


def build_install_plan(target, candidate, *, allow_unverified=False):
    """Build a pip command without modifying the target environment."""

    if not candidate.get("artifact_available"):
        raise InstallationError(f"candidate is not artifact-available: {candidate.get('id', 'unknown')}")
    if candidate.get("python_compatibility") == "incompatible":
        raise InstallationError(f"candidate is incompatible with target Python: {candidate.get('id', 'unknown')}")

    warnings = []
    resolver_status = candidate.get("resolver_status")
    if resolver_status == "resolver_failed":
        warnings.append("resolver evidence is failed; this candidate requires explicit unverified approval")
    elif resolver_status in {"not_collected", "unknown"}:
        warnings.append("resolver evidence is not available")
    if candidate.get("python_compatibility") == "unknown":
        warnings.append("Python ABI compatibility is unknown")
    if resolver_status == "resolver_failed" and not allow_unverified:
        warnings.append("apply would require --allow-unverified")

    python = target.python_executable
    if python is None:
        raise InstallationError("target Python executable was not found")

    wheel_urls = tuple(candidate.get("wheel_urls") or ())
    package_specs = tuple(candidate.get("package_specs") or ())
    if wheel_urls:
        requirements = wheel_urls
    elif package_specs:
        requirements = package_specs
    else:
        raise InstallationError("candidate has no package URLs or exact package specifications")

    command = [str(python), "-m", "pip", "install", "--no-input"]
    if candidate.get("index_url") and not wheel_urls:
        command.extend(("--index-url", candidate["index_url"]))
    command.extend(requirements)
    return InstallPlan(
        target_root=target.root,
        candidate=candidate,
        command=tuple(command),
        warnings=tuple(warnings),
    )


def dry_run_install(target, candidate, *, allow_unverified=False):
    """Return an install command without invoking pip."""

    return InstallResult(plan=build_install_plan(target, candidate, allow_unverified=allow_unverified))


def apply_install(target, candidate, backup, *, allow_unverified=False, timeout=3600):
    """Run the planned pip command after an explicit backup and approval."""

    plan = build_install_plan(target, candidate, allow_unverified=allow_unverified)
    if candidate.get("resolver_status") == "resolver_failed" and not allow_unverified:
        raise InstallationError("resolver evidence failed; pass --allow-unverified to apply")
    try:
        completed = subprocess.run(
            list(plan.command),
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return InstallResult(
            plan=plan,
            applied=True,
            backup_path=str(backup.path),
            output=f"{type(error).__name__}: {error}",
        )
    output = (completed.stdout or "") + (completed.stderr or "")
    return InstallResult(
        plan=plan,
        applied=True,
        returncode=completed.returncode,
        backup_path=str(backup.path),
        output=output,
    )


def build_restore_plan(target, backup):
    """Build a version-restore command without running pip."""

    if not isinstance(backup, BackupSnapshot):
        raise InstallationError("invalid package backup")
    if target.python_executable is None:
        raise InstallationError("target Python executable was not found")
    if backup.target_root:
        try:
            backup_root = Path(backup.target_root).expanduser().resolve()
            target_root = Path(target.root).expanduser().resolve()
        except OSError as error:
            raise InstallationError(f"cannot resolve restore target identity: {error}") from error
        if backup_root != target_root:
            raise InstallationError(
                f"backup belongs to a different target: {backup_root}"
            )
    if not backup.requirements_path.is_file():
        raise InstallationError(f"backup requirements file is missing: {backup.requirements_path}")
    command = (
        str(target.python_executable),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--force-reinstall",
        "--requirement",
        str(backup.requirements_path),
    )
    return InstallPlan(
        target_root=target.root,
        candidate={"id": f"backup:{backup.path.name}"},
        command=command,
        warnings=("restore reinstalls recorded versions but does not prune extra packages",),
    )


def apply_restore(target, backup, timeout=3600):
    """Run a restore command after explicit user approval."""

    plan = build_restore_plan(target, backup)
    try:
        completed = subprocess.run(
            list(plan.command),
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return InstallResult(plan=plan, applied=True, output=f"{type(error).__name__}: {error}")
    return InstallResult(
        plan=plan,
        applied=True,
        returncode=completed.returncode,
        backup_path=str(backup.path),
        output=(completed.stdout or "") + (completed.stderr or ""),
    )
