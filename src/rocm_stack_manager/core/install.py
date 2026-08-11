"""Create and optionally apply safe, target-scoped install plans."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import re
from urllib.parse import unquote, urlparse

from .backup import BackupError, BackupSnapshot, ExtensionBackupSnapshot, attach_wheelhouse, create_backup
from .inventory import collect_inventory
from .planning import InstallPlan, PlanningError, build_plan_binding, validate_plan_binding
from .platforms import UnsupportedPlatformError, require_supported_host_platform
from .staging import StagingError, stage_candidate, validate_staged_candidate
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
    staging_path: str | None = None
    staged_command: tuple[str, ...] = ()
    output: str = ""

    def as_dict(self):
        values = self.plan.as_dict()
        values.update(
            {
                "applied": self.applied,
                "returncode": self.returncode,
                "backup_path": self.backup_path,
                "staging_path": self.staging_path,
                "staged_command": list(self.staged_command),
                "output": self.output,
            }
        )
        return values


def build_install_plan(
    target,
    candidate,
    *,
    allow_unverified=False,
    catalog_hash=None,
    adapter_id=None,
    target_platform=None,
    target_gfx=None,
):
    """Build a pip command without modifying the target environment."""

    if not candidate.get("artifact_available"):
        raise InstallationError(f"candidate is not artifact-available: {candidate.get('id', 'unknown')}")
    if candidate.get("candidate_kind") == "artifact_only":
        raise InstallationError(
            f"candidate has artifact evidence only and no install source: {candidate.get('id', 'unknown')}"
        )
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
        command.extend(("--extra-index-url", candidate["index_url"]))
    command.extend(requirements)
    return InstallPlan(
        target_root=target.root,
        candidate=candidate,
        command=tuple(command),
        warnings=tuple(warnings),
        binding=build_plan_binding(
            target,
            candidate,
            catalog_hash=catalog_hash,
            adapter_id=adapter_id,
            target_platform=target_platform,
            target_gfx=target_gfx,
        ),
    )


def dry_run_install(target, candidate, *, allow_unverified=False, **binding):
    """Return an install command without invoking pip."""

    return InstallResult(
        plan=build_install_plan(
            target,
            candidate,
            allow_unverified=allow_unverified,
            **binding,
        )
    )


def apply_install(
    target,
    candidate,
    backup=None,
    *,
    allow_unverified=False,
    timeout=3600,
    plan=None,
    backup_dir=None,
    resolver_result=None,
    resolver_runner=None,
    staging_runner=None,
    staging_dir=None,
    **binding,
):
    """Run the planned pip command after an explicit backup and approval."""

    try:
        current_platform = require_supported_host_platform()
    except UnsupportedPlatformError as error:
        raise InstallationError(str(error)) from error
    candidate_platform = candidate.get("platform")
    if candidate_platform and candidate_platform != current_platform:
        raise InstallationError(
            f"cross-platform apply is not allowed: candidate={candidate_platform}, target={current_platform}"
        )
    if plan is None:
        plan = build_install_plan(
            target,
            candidate,
            allow_unverified=allow_unverified,
            **binding,
        )
    else:
        validate_plan_binding(plan, target, candidate, **binding)
    if not allow_unverified:
        from .resolver import resolver_matches_target, run_resolver

        if resolver_result is None:
            resolver_result = run_resolver(
                target,
                candidate,
                catalog_hash=binding.get("catalog_hash"),
                adapter_id=binding.get("adapter_id"),
                target_gfx=binding.get("target_gfx"),
                runner=resolver_runner or subprocess.run,
                timeout=min(timeout, 900),
            )
        if not resolver_matches_target(
            resolver_result,
            target,
            candidate,
            catalog_hash=binding.get("catalog_hash"),
            adapter_id=binding.get("adapter_id"),
            target_gfx=binding.get("target_gfx"),
        ):
            status = getattr(resolver_result, "status", "not_collected")
            raise InstallationError(f"fresh target-bound resolver preflight required; status={status}")
    try:
        staged = stage_candidate(
            target,
            candidate,
            staging_dir,
            timeout=min(timeout, 1800),
            runner=staging_runner or subprocess.run,
        )
        validate_staged_candidate(staged)
    except StagingError as error:
        raise InstallationError(str(error)) from error
    if backup is None:
        backup = create_backup(target, backup_dir)
    if backup.path.is_file():
        try:
            backup = attach_wheelhouse(backup, staged)
        except (BackupError, OSError, ValueError) as error:
            raise InstallationError(f"cannot preserve staged artifacts in backup: {error}") from error
    staged_command = (
        str(target.python_executable),
        "-m",
        "pip",
        "install",
        "--no-input",
        *staged.install_command,
    )
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
                returncode=1,
                backup_path=str(backup.path),
                staging_path=str(staged.root),
                staged_command=staged_command,
                output=f"stale package cleanup failed: {type(error).__name__}: {error}",
            )
        cleanup_output = (cleanup.stdout or "") + (cleanup.stderr or "")
        if cleanup.returncode != 0:
            return InstallResult(
                plan=plan,
                applied=True,
                returncode=cleanup.returncode,
                backup_path=str(backup.path),
                staging_path=str(staged.root),
                staged_command=staged_command,
                output=f"stale package cleanup failed:\n{cleanup_output}",
            )
        output_prefix = f"Removed stale packages: {', '.join(stale_packages)}\n"
    try:
        completed = subprocess.run(
            list(staged_command),
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
            returncode=1,
            backup_path=str(backup.path),
            staging_path=str(staged.root),
            staged_command=staged_command,
            output=output_prefix + f"{type(error).__name__}: {error}",
        )
    output = output_prefix + (completed.stdout or "") + (completed.stderr or "")
    return InstallResult(
        plan=plan,
        applied=True,
        returncode=completed.returncode,
        backup_path=str(backup.path),
        staging_path=str(staged.root),
        staged_command=staged_command,
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
    requirements_source = backup.wheelhouse_requirements_path or backup.requirements_path
    command = (
        str(target.python_executable),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--force-reinstall",
        "--requirement",
        str(requirements_source),
    )
    if backup.wheelhouse_path:
        command = (
            str(target.python_executable),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--force-reinstall",
            "--no-index",
            "--find-links",
            str(backup.wheelhouse_path),
            "--require-hashes",
            "--requirement",
            str(requirements_source),
        )
    return InstallPlan(
        target_root=target.root,
        candidate={"id": f"backup:{backup.path.name}"},
        command=command,
        warnings=(
            "restore reinstalls recorded versions but does not prune extra packages",
            "restore uses hash-verified local artifacts" if backup.wheelhouse_path else "restore is version-pinned; package artifact bytes are not archived",
        ),
    )


def build_extension_restore_plan(target, backup: ExtensionBackupSnapshot):
    """Build a restore plan for an extension-only backup."""

    if not isinstance(backup, ExtensionBackupSnapshot):
        raise InstallationError("invalid extension package backup")
    if target.python_executable is None:
        raise InstallationError("target Python executable was not found")
    if backup.target_root:
        if Path(backup.target_root).expanduser().resolve() != Path(target.root).expanduser().resolve():
            raise InstallationError(f"backup belongs to a different target: {backup.target_root}")
    if not backup.requirements:
        return InstallPlan(
            target_root=target.root,
            candidate={"id": f"backup:{backup.path.name}"},
            command=(),
            warnings=(
                "no extension packages were recorded; nothing to restore",
                "restore is version-pinned; package artifact bytes are not archived",
            ),
        )
    if not backup.requirements_path.is_file():
        raise InstallationError(f"extension backup requirements file is missing: {backup.requirements_path}")
    return InstallPlan(
        target_root=target.root,
        candidate={"id": f"backup:{backup.path.name}"},
        command=(
            str(target.python_executable),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--force-reinstall",
            "--requirement",
            str(backup.requirements_path),
        ),
        warnings=(
            "restore reinstalls recorded extensions but does not prune extra packages",
            "restore is version-pinned; package artifact bytes are not archived",
        ),
    )


def apply_restore(target, backup, timeout=3600):
    """Run a restore command after explicit user approval."""

    try:
        require_supported_host_platform()
    except UnsupportedPlatformError as error:
        raise InstallationError(str(error)) from error
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
        return InstallResult(plan=plan, applied=True, returncode=1, output=f"{type(error).__name__}: {error}")
    return InstallResult(
        plan=plan,
        applied=True,
        returncode=completed.returncode,
        backup_path=str(backup.path),
        output=(completed.stdout or "") + (completed.stderr or ""),
    )


def apply_extension_restore(target, backup: ExtensionBackupSnapshot, timeout=3600):
    """Apply an extension-only restore after explicit approval."""

    try:
        require_supported_host_platform()
    except UnsupportedPlatformError as error:
        raise InstallationError(str(error)) from error
    plan = build_extension_restore_plan(target, backup)
    if not plan.command:
        return InstallResult(
            plan=plan,
            applied=True,
            returncode=0,
            backup_path=str(backup.path),
        )
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
            returncode=1,
            backup_path=str(backup.path),
            output=f"{type(error).__name__}: {error}",
        )
    return InstallResult(
        plan=plan,
        applied=True,
        returncode=completed.returncode,
        backup_path=str(backup.path),
        output=(completed.stdout or "") + (completed.stderr or ""),
    )
