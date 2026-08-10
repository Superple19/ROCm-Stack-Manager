"""Build non-mutating installation plans from catalog candidates."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InstallPlan:
    """A user-approved candidate and optional extension selection."""

    target_root: Path
    candidate: dict
    extensions: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self):
        return {
            "target_root": str(self.target_root),
            "candidate": self.candidate,
            "extensions": list(self.extensions),
        }


class PlanningError(ValueError):
    """Raised when a candidate cannot produce an installation plan."""


def build_plan(target, candidate, extensions=()):
    """Create a plan without installing or modifying the target."""

    if not candidate.get("artifact_available"):
        raise PlanningError(
            f"candidate is not artifact-available: {candidate.get('id', 'unknown')}"
        )
    if candidate.get("python_compatibility") == "incompatible":
        raise PlanningError(
            f"candidate is incompatible with target Python: {candidate.get('id', 'unknown')}"
        )
    return InstallPlan(target_root=target.root, candidate=candidate, extensions=tuple(extensions))
