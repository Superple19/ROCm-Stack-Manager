"""Application-independent GPU, GFX, and ROCm tool detection."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess

from .platforms import host_platform as _host_platform


_GFX_RE = re.compile(r"(?i)\b(gfx[0-9a-f]+)\b")
_DRIVER_RE = re.compile(r"(?im)^\s*(?:driver|driver version|version)\s*[:=]\s*(\S+)")
_TOOL_NAMES = ("hipInfo", "hipinfo", "rocminfo")


@dataclass(frozen=True)
class HardwareObservation:
    """A hardware observation independent of an application runtime."""

    scope: str
    host_platform: str
    status: str
    devices: tuple[dict, ...] = ()
    gfx_targets: tuple[str, ...] = ()
    source: str | None = None
    tool: str | None = None
    command: tuple[str, ...] = ()
    driver_version: str | None = None
    target_root: Path | None = None
    error: str | None = None

    def as_dict(self):
        return {
            "scope": self.scope,
            "host_platform": self.host_platform,
            "status": self.status,
            "devices": list(self.devices),
            "gfx_targets": list(self.gfx_targets),
            "source": self.source,
            "tool": self.tool,
            "command": list(self.command),
            "driver_version": self.driver_version,
            "target_root": str(self.target_root) if self.target_root else None,
            "error": self.error,
        }


def normalize_gfx(value):
    """Normalize a GFX architecture token without inferring from a GPU name."""

    match = _GFX_RE.search(str(value or "").strip())
    return match.group(1).lower() if match else None


def detected_gfx_targets(observation):
    """Return unique GFX values from a hardware or runtime observation."""

    if isinstance(observation, dict):
        values = observation.get("gfx_targets") or ()
        if values:
            return tuple(dict.fromkeys(filter(None, (normalize_gfx(value) for value in values))))
        devices = observation.get("devices") or ()
    else:
        values = getattr(observation, "gfx_targets", ())
        if values:
            return tuple(dict.fromkeys(filter(None, (normalize_gfx(value) for value in values))))
        devices = getattr(observation, "devices", ())
    targets = []
    for device in devices:
        value = device.get("gfx") if isinstance(device, dict) else None
        target = normalize_gfx(value)
        if target and target not in targets:
            targets.append(target)
    return tuple(targets)


def hardware_from_devices(
    devices,
    *,
    scope="target-runtime",
    source="runtime",
    target_root=None,
    host_platform=None,
):
    """Create a hardware observation from an adapter/runtime device report."""

    normalized_devices = []
    for index, device in enumerate(devices or ()):
        if not isinstance(device, dict):
            continue
        item = dict(device)
        item.setdefault("index", index)
        item["gfx"] = normalize_gfx(item.get("gfx"))
        normalized_devices.append(item)
    targets = detected_gfx_targets({"devices": normalized_devices})
    return HardwareObservation(
        scope=scope,
        host_platform=host_platform or _host_platform(),
        status="detected" if targets else "not_detected",
        devices=tuple(normalized_devices),
        gfx_targets=targets,
        source=source,
        target_root=target_root,
        error=None if targets else "runtime reported no GFX architecture",
    )


def _candidate_tool_paths(target):
    if target is None:
        return ()
    roots = [target.root]
    if target.python_executable:
        roots.extend(
            (
                target.python_executable.parent,
                target.python_executable.parent / "Scripts",
                target.python_executable.parent / "bin",
            )
        )
    roots.extend((target.root / "Scripts", target.root / "bin"))
    paths: list[Path] = []
    for root in roots:
        for name in _TOOL_NAMES:
            for suffix in (".exe", ""):
                path = root / f"{name}{suffix}"
                if path.is_file() and path not in paths:
                    paths.append(path)
    return tuple(paths)


def _tool_candidates(target=None, tool_path=None):
    if tool_path:
        return (Path(tool_path).expanduser(),)
    candidates = list(_candidate_tool_paths(target))
    for name in _TOOL_NAMES:
        resolved = shutil.which(name)
        if resolved:
            path = Path(resolved)
            if path not in candidates:
                candidates.append(path)
    return tuple(candidates)


def _parse_tool_output(output):
    targets = []
    for match in _GFX_RE.finditer(output):
        target = match.group(1).lower()
        if target not in targets:
            targets.append(target)
    driver_match = _DRIVER_RE.search(output)
    devices = tuple({"index": index, "gfx": target} for index, target in enumerate(targets))
    return devices, tuple(targets), driver_match.group(1) if driver_match else None


def probe_hardware(target=None, *, tool_path=None, timeout=10):
    """Probe GFX through target-local or explicitly available ROCm tools.

    Target-local tools are preferred. A system ``hipInfo``/``rocminfo`` is only
    used as a fallback; no ROCm installation path is assumed or invented.
    """

    candidates = _tool_candidates(target, tool_path)
    if not candidates:
        return HardwareObservation(
            scope="host",
            host_platform=_host_platform(),
            status="not_detected",
            target_root=target.root if target else None,
            error="hipInfo or rocminfo was not found",
        )
    errors = []
    for path in candidates:
        command = (str(path),)
        try:
            completed = subprocess.run(
                list(command),
                cwd=str(target.comfyui_dir) if target else None,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        output = (completed.stdout or "") + (completed.stderr or "")
        devices, targets, driver_version = _parse_tool_output(output)
        if completed.returncode == 0 and targets:
            return HardwareObservation(
                scope="target-tool" if target and path in _candidate_tool_paths(target) else "host",
                host_platform=_host_platform(),
                status="detected",
                devices=devices,
                gfx_targets=targets,
                source="rocm-tool",
                tool=path.name,
                command=command,
                driver_version=driver_version,
                target_root=target.root if target else None,
            )
        errors.append(f"{path.name}: exit {completed.returncode}; no GFX target reported")
    return HardwareObservation(
        scope="host",
        host_platform=_host_platform(),
        status="not_detected",
        target_root=target.root if target else None,
        error="; ".join(errors)[-1000:],
    )
