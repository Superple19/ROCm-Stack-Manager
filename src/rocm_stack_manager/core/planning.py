"""Build non-mutating installation plans from catalog candidates."""

from dataclasses import dataclass, field
import sys
from pathlib import Path

from .identity import candidate_hash


def host_platform():
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def build_plan_binding(
    target,
    candidate,
    *,
    catalog_hash=None,
    adapter_id=None,
    target_platform=None,
    target_gfx=None,
):
    return {
        "target_root": str(Path(target.root).expanduser().resolve()),
        "target_python": (
            str(Path(target.python_executable).expanduser().resolve())
            if target.python_executable
            else None
        ),
        "target_platform": target_platform or host_platform(),
        "target_gfx": target_gfx if target_gfx is not None else candidate.get("gfx"),
        "candidate_hash": candidate_hash(candidate),
        "catalog_hash": catalog_hash,
        "adapter_id": adapter_id or "unknown",
    }


def validate_plan_binding(
    plan,
    target,
    candidate=None,
    *,
    catalog_hash=None,
    adapter_id=None,
    target_platform=None,
    target_gfx=None,
):
    expected = build_plan_binding(
        target,
        candidate or plan.candidate,
        catalog_hash=catalog_hash,
        adapter_id=adapter_id,
        target_platform=target_platform,
        target_gfx=target_gfx,
    )
    actual = plan.binding
    candidate_platform = (candidate or plan.candidate).get("platform")
    if candidate_platform and candidate_platform != expected.get("target_platform"):
        raise PlanningError(
            f"cross-platform apply is not allowed: candidate={candidate_platform}, target={expected.get('target_platform')}"
        )
    for field in ("target_root", "target_python", "target_platform", "target_gfx", "candidate_hash"):
        if actual.get(field) != expected.get(field):
            raise PlanningError(f"installation plan binding changed: {field}")
    if catalog_hash is not None and actual.get("catalog_hash") != expected.get("catalog_hash"):
        raise PlanningError("installation plan binding changed: catalog_hash")
    if adapter_id is not None and actual.get("adapter_id") != expected.get("adapter_id"):
        raise PlanningError("installation plan binding changed: adapter_id")


@dataclass(frozen=True)
class InstallPlan:
    """A user-approved candidate and optional extension selection."""

    target_root: Path
    candidate: dict
    extensions: tuple[str, ...] = field(default_factory=tuple)
    command: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    binding: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "target_root": str(self.target_root),
            "candidate": self.candidate,
            "extensions": list(self.extensions),
            "command": list(self.command),
            "warnings": list(self.warnings),
            "binding": dict(self.binding),
        }


class PlanningError(ValueError):
    """Raised when a candidate cannot produce an installation plan."""


def build_plan(target, candidate, extensions=()):
    """Create a plan without installing or modifying the target."""

    if not candidate.get("artifact_available"):
        raise PlanningError(
            f"candidate is not artifact-available: {candidate.get('id', 'unknown')}"
        )
    if candidate.get("candidate_kind") == "artifact_only":
        raise PlanningError(
            f"candidate has artifact evidence only and no install source: {candidate.get('id', 'unknown')}"
        )
    if candidate.get("python_compatibility") == "incompatible":
        raise PlanningError(
            f"candidate is incompatible with target Python: {candidate.get('id', 'unknown')}"
        )
    return InstallPlan(
        target_root=target.root,
        candidate=candidate,
        extensions=tuple(extensions),
        binding=build_plan_binding(target, candidate),
    )
