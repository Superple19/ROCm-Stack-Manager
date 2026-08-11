"""Targeted, non-mutating dependency checks for one core candidate."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import subprocess

from .install import build_install_plan
from .verify import _clean_environment


@dataclass(frozen=True)
class ResolverResult:
    """Result of resolving one candidate without changing the target."""

    candidate_id: str
    candidate_hash: str | None
    status: str
    command: tuple[str, ...]
    returncode: int | None = None
    output: str = ""
    observed_at: str = ""
    binding: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "status": self.status,
            "command": list(self.command),
            "returncode": self.returncode,
            "output": self.output,
            "observed_at": self.observed_at,
            "verification_scope": "resolver",
            "promotion": "none",
            "network_access": True,
            "installation_performed": False,
            "binding": dict(self.binding or {}),
        }


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_resolver_command(target, candidate):
    """Build a pip dry-run command from the normal core install plan."""

    plan = build_install_plan(target, candidate)
    command = list(plan.command)
    install_index = command.index("install")
    command[install_index + 1:install_index + 1] = (
        "--dry-run",
        "--ignore-installed",
        "--disable-pip-version-check",
    )
    return tuple(command)


def run_resolver(
    target,
    candidate,
    *,
    catalog_hash=None,
    adapter_id=None,
    target_gfx=None,
    timeout=900,
    runner=subprocess.run,
):
    """Resolve one selected candidate without installing it."""

    observed_at = _utc_now()
    from .planning import build_plan_binding

    binding = build_plan_binding(
        target,
        candidate,
        catalog_hash=catalog_hash,
        adapter_id=adapter_id,
        target_gfx=target_gfx,
    )
    try:
        command = build_resolver_command(target, candidate)
    except (OSError, ValueError) as error:
        return ResolverResult(
            candidate_id=candidate.get("id", "unknown"),
            candidate_hash=candidate.get("candidate_hash"),
            status="resolver_failed",
            command=(),
            output=str(error),
            observed_at=observed_at,
            binding=binding,
        )
    try:
        completed = runner(
            list(command),
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ResolverResult(
            candidate_id=candidate.get("id", "unknown"),
            candidate_hash=candidate.get("candidate_hash"),
            status="resolver_failed",
            command=command,
            output=f"{type(error).__name__}: {error}",
            observed_at=observed_at,
            binding=binding,
        )
    output = ((completed.stdout or "") + (completed.stderr or ""))[-12000:]
    return ResolverResult(
        candidate_id=candidate.get("id", "unknown"),
        candidate_hash=candidate.get("candidate_hash"),
        status="resolver_verified" if completed.returncode == 0 else "resolver_failed",
        command=command,
        returncode=completed.returncode,
        output=output,
        observed_at=observed_at,
        binding=binding,
    )


def resolver_matches_target(result, target, candidate, *, catalog_hash=None, adapter_id=None, target_gfx=None):
    """Return whether resolver evidence belongs to this exact target and candidate."""

    if result is None or result.status != "resolver_verified":
        return False
    from .planning import build_plan_binding

    expected = build_plan_binding(
        target,
        candidate,
        catalog_hash=catalog_hash,
        adapter_id=adapter_id,
        target_gfx=target_gfx,
    )
    return result.binding == expected
