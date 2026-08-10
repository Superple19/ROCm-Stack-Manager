"""Installation plan data structures."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InstallPlan:
    """A user-approved candidate and optional extension selection."""

    target_root: Path
    candidate: dict
    extensions: tuple[str, ...] = field(default_factory=tuple)
