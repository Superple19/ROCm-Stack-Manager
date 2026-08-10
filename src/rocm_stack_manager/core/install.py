"""Create and optionally apply safe, target-scoped install plans."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import re
from urllib.parse import unquote, urlparse

from .backup import BackupSnapshot
from .inventory import collect_inventory
from .planning import InstallPlan, PlanningError
from .verify import _clean_environment


class InstallationError(PlanningError):
    """Raised when a candidate cannot produce a safe install plan."""


_MANAGED_EXACT_NAMES = {
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-libraries",
    "rocm-sdk-libraries-custom",
    "torch",
    "torchvision",
    "torchaudio",
}


def _normalize_package_name(name):
    return str(name).casefold().replace("_", "-")


def _is_managed_package(name):
    normalized = _normalize_package_name(name)
    return (
        normalized in _MANAGED_EXACT_NAMES
        or normalized.startswith("rocm-sdk-device-")
        or normalized.startswith("amd-torch-device-")
        or normalized.startswith("amd-torchvision-device-")
    )


def _package_name_from_requirement(requirement):
    value = unquote(str(requirement)).split("#", 1)[0]
    if "://" in value:
        value = urlparse(value).path.rsplit("/", 1)[-1]
    value = re.sub(r"\.(?:whl|tar\.gz|zip)$", "", value, flags=re.IGNORECASE)
    match = re.match(r"^(.+?)-\d", value)
    return _normalize_package_name(match.group(1) if match else value.split("==", 1)[0])


def _candidate_managed_packages(candidate):
    requirements = tuple(candidate.get("wheel_urls") or ()) + tuple(candidate.get("package_specs") or ())
    return {
        _package_name_from_requirement(requirement)
        for requirement in requirements
        if _is_managed_package(_package_name_from_requirement(requirement))
    }


def _stale_managed_packages(target, candidate):
    inventory = collect_inventory(target, candidate)
    if inventory.status != "detected":
        return ()
    desired = _candidate_managed_packages(candidate)
    stale = {
        package["name"]
        for package in inventory.packages
        if _is_managed_package(package.get("name", ""))
        and _normalize_package_name(package["name"]) not in desired
    }
    return tuple(sorted(stale, key=str.casefold))


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
    if "torchaudio" not in _candidate_managed_packages(candidate):
        warnings.append("torchaudio is not included; audio workflows may be unavailable")
    warnings.extend(f"ComfyUI profile: {warning}" for warning in candidate.get("profile_warnings", ()))
    warnings.append("apply removes stale managed ROCm/Torch packages before install")

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
    stale_packages = _stale_managed_packages(target, candidate)
    output_prefix = ""
    if stale_packages:
        cleanup_command = [str(target.python_executable), "-m", "pip", "uninstall", "--yes", *stale_packages]
        try:
            cleanup = subprocess.run(
                cleanup_command,
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
                output=f"stale package cleanup failed: {type(error).__name__}: {error}",
            )
        cleanup_output = (cleanup.stdout or "") + (cleanup.stderr or "")
        if cleanup.returncode != 0:
            return InstallResult(
                plan=plan,
                applied=True,
                returncode=cleanup.returncode,
                backup_path=str(backup.path),
                output=f"stale package cleanup failed:\n{cleanup_output}",
            )
        output_prefix = f"Removed stale packages: {', '.join(stale_packages)}\n"
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
            output=output_prefix + f"{type(error).__name__}: {error}",
        )
    output = output_prefix + (completed.stdout or "") + (completed.stderr or "")
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
