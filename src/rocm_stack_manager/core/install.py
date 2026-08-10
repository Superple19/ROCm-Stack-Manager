"""Create safe, target-scoped ROCm package installation plans."""

from dataclasses import dataclass

from .planning import InstallPlan, PlanningError


class InstallationError(PlanningError):
    """Raised when a candidate cannot produce a safe install plan."""


@dataclass(frozen=True)
class InstallResult:
    """Result of a dry-run plan; no subprocess is run by this module."""

    plan: InstallPlan
    applied: bool = False
    output: str = ""

    def as_dict(self):
        values = self.plan.as_dict()
        values.update({"applied": self.applied, "output": self.output})
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
