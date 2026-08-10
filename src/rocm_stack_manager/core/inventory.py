"""Inventory target-local packages and classify candidate impact."""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .verify import _clean_environment


_INVENTORY_SCRIPT = r'''
import importlib.metadata
import json

packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        continue
    files = []
    for file in distribution.files or ():
        suffix = str(file).lower()
        if suffix.endswith((".pyd", ".so", ".dll", ".dylib")):
            files.append(str(file))
    packages.append({
        "name": name,
        "version": distribution.version,
        "requires": list(distribution.requires or ()),
        "compiled_files": files,
    })
print(json.dumps({"packages": packages}, sort_keys=True))
'''

_CORE_NAMES = {
    "torch": "torch_version",
    "torchvision": "torchvision_version",
    "torchaudio": "torchaudio_version",
    "rocm": "rocm_version",
}
_COMPILED_EXTENSION_NAMES = {
    "amd-aiter",
    "aiter",
    "bitsandbytes",
    "flash-attn",
    "flash_attn",
    "sageattention",
    "sage-attention",
    "triton",
    "triton-windows",
}
_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*(.*)$")


@dataclass(frozen=True)
class PackageInventory:
    """Packages observed through the selected target interpreter."""

    target_root: Path
    python_executable: Path | None
    packages: tuple[dict, ...]
    status: str
    error: str | None = None

    def as_dict(self):
        counts = {state: 0 for state in ("compatible", "conflict", "unknown")}
        for package in self.packages:
            state = package.get("status")
            if state in counts:
                counts[state] += 1
        return {
            "target_root": str(self.target_root),
            "python_executable": str(self.python_executable) if self.python_executable else None,
            "status": self.status,
            "summary": counts,
            "packages": list(self.packages),
            "error": self.error,
        }


def _parse_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        for line in reversed(output.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "packages" in value:
                return value
        raise


def _is_compiled(package):
    name = package["name"].casefold().replace("_", "-")
    return name in _COMPILED_EXTENSION_NAMES or bool(package.get("compiled_files"))


def _core_expected_key(normalized):
    if normalized in _CORE_NAMES:
        return _CORE_NAMES[normalized]
    if normalized.startswith("amd-torch-device-"):
        return "torch_version"
    if normalized.startswith("amd-torchvision-device-"):
        return "torchvision_version"
    if normalized.startswith("rocm-sdk"):
        return "rocm_version"
    return None


def _requirement_conflict(requirement, candidate):
    requirement = requirement.split(";", 1)[0].strip()
    match = _REQUIREMENT_RE.match(requirement)
    if not match:
        return False
    name, specifier = match.groups()
    normalized = name.casefold().replace("_", "-")
    expected_key = _core_expected_key(normalized)
    expected = candidate.get(expected_key) if expected_key else None
    if not expected or not specifier:
        return False
    for operator, version in re.findall(r"(==|!=)\s*([^,;\s]+)", specifier):
        if operator == "==" and version != expected:
            return True
        if operator == "!=" and version == expected:
            return True
    return False


def classify_packages(packages, candidate=None):
    """Classify packages conservatively against an optional candidate."""

    candidate = candidate or {}
    classified = []
    for package in packages:
        name = package.get("name", "")
        normalized = name.casefold().replace("_", "-")
        record = dict(package)
        expected_key = _core_expected_key(normalized)
        if expected_key:
            expected = candidate.get(expected_key)
            record["status"] = "unknown" if not expected else (
                "compatible" if package.get("version") == expected else "conflict"
            )
            record["reason"] = "matches candidate core version" if record["status"] == "compatible" else (
                "installed core version differs from candidate" if record["status"] == "conflict" else "candidate core version unavailable"
            )
        elif _is_compiled(package):
            record["status"] = "unknown"
            record["reason"] = "compiled extension requires separate compatibility evidence"
        elif any(_requirement_conflict(requirement, candidate) for requirement in package.get("requires", ())):
            record["status"] = "conflict"
            record["reason"] = "declared dependency excludes candidate core version"
        else:
            record["status"] = "compatible"
            record["reason"] = "no compiled artifact or conflicting core constraint observed"
        classified.append(record)
    return tuple(sorted(classified, key=lambda package: package["name"].casefold()))


def collect_inventory(target, candidate=None, timeout=30):
    """Collect packages using only the selected target Python interpreter."""

    if target.python_executable is None:
        return PackageInventory(target.root, None, (), "not_detected", "target Python executable was not found")
    try:
        completed = subprocess.run(
            [str(target.python_executable), "-c", _INVENTORY_SCRIPT],
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return PackageInventory(target.root, target.python_executable, (), "not_detected", f"{type(error).__name__}: {error}")
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "target package inventory failed").strip()
        return PackageInventory(target.root, target.python_executable, (), "not_detected", error[-1000:])
    try:
        document = _parse_output(completed.stdout)
    except json.JSONDecodeError as error:
        return PackageInventory(target.root, target.python_executable, (), "not_detected", f"invalid inventory output: {error}")
    packages = classify_packages(document.get("packages", ()), candidate)
    return PackageInventory(target.root, target.python_executable, packages, "detected")
