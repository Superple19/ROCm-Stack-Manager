"""Detection of user-selected ComfyUI installation layouts."""

from dataclasses import dataclass
from pathlib import Path


class TargetDetectionError(ValueError):
    """Raised when a selected directory is not a recognizable ComfyUI target."""


@dataclass(frozen=True)
class TargetLayout:
    """Resolved paths and metadata for one ComfyUI installation."""

    root: Path
    comfyui_dir: Path
    python_executable: Path | None
    launchers: tuple[Path, ...]
    layout: str

    def as_dict(self):
        return {
            "root": str(self.root),
            "comfyui_dir": str(self.comfyui_dir),
            "python_executable": str(self.python_executable) if self.python_executable else None,
            "launchers": [str(path) for path in self.launchers],
            "layout": self.layout,
        }


def _comfyui_dir(path: Path) -> Path | None:
    candidates = (path, path / "ComfyUI")
    for candidate in candidates:
        if (candidate / "main.py").is_file():
            return candidate
    return None


def _normalize_root(selected: Path) -> tuple[Path, Path]:
    selected_comfyui = _comfyui_dir(selected)
    if selected_comfyui is None:
        raise TargetDetectionError(
            f"No ComfyUI source directory found under target: {selected}"
        )

    if selected_comfyui == selected and selected.name.casefold() == "comfyui":
        parent = selected.parent
        if _comfyui_dir(parent) == selected:
            return parent, selected
    return selected, selected_comfyui


def _python_candidates(root: Path, comfyui_dir: Path):
    relative_paths = (
        ("python_embeded", "python.exe"),
        ("python_embedded", "python.exe"),
        ("python", "python.exe"),
        ("python_env", "python.exe"),
        ("venv", "Scripts", "python.exe"),
        (".venv", "Scripts", "python.exe"),
        ("python_env", "Scripts", "python.exe"),
        ("venv", "bin", "python"),
        (".venv", "bin", "python"),
        ("python_env", "bin", "python"),
    )
    seen = set()
    for base in (root, comfyui_dir):
        for parts in relative_paths:
            candidate = base.joinpath(*parts)
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _find_python(root: Path, comfyui_dir: Path) -> Path | None:
    for candidate in _python_candidates(root, comfyui_dir):
        if candidate.is_file():
            return candidate
    return None


def _layout_name(root: Path, python_executable: Path | None) -> str:
    if (root / "python_embeded").is_dir() or (root / "python_embedded").is_dir():
        return "portable"
    if python_executable and any(
        part.casefold() in {"venv", ".venv", "python_env"}
        for part in python_executable.parts
    ):
        return "venv"
    return "source"


def detect_target(selected_path=".") -> TargetLayout:
    """Resolve a portable or virtual-environment ComfyUI target.

    ``selected_path`` may be the portable root or its nested ``ComfyUI``
    directory. No machine-specific absolute path is used.
    """

    selected = Path(selected_path).expanduser().resolve()
    if not selected.is_dir():
        raise TargetDetectionError(f"Target directory does not exist: {selected}")

    root, comfyui_dir = _normalize_root(selected)
    python_executable = _find_python(root, comfyui_dir)
    launchers = tuple(sorted(root.glob("run_*.bat")))
    return TargetLayout(
        root=root,
        comfyui_dir=comfyui_dir,
        python_executable=python_executable,
        launchers=launchers,
        layout=_layout_name(root, python_executable),
    )
