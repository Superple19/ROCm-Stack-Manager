"""Application launch service boundary."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LaunchOptions:
    """Explicit launch arguments supplied by a caller or future UI."""

    arguments: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LaunchPlan:
    """A launch command that has not been executed."""

    target_root: Path
    command: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self):
        return {
            "target_root": str(self.target_root),
            "command": list(self.command),
            "environment": dict(self.environment),
            "warnings": list(self.warnings),
        }
