"""Probe ROCm runtime and GPU state from the selected target environment."""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


_PROBE_SCRIPT = r'''
import json
import platform
import sys

result = {
    "python_version": platform.python_version(),
    "torch_version": None,
    "torch_rocm_tag": None,
    "hip_version": None,
    "torch_file": None,
    "rocm_packages": {},
    "cuda_available": False,
    "device_count": 0,
    "devices": [],
    "tensor_smoke_status": "not_applicable",
    "tensor_smoke_error": None,
    "torch_error": None,
}
try:
    import torch
except Exception as error:
    result["torch_error"] = f"{type(error).__name__}: {error}"
else:
    result["torch_version"] = getattr(torch, "__version__", None)
    if result["torch_version"] and "+rocm" in result["torch_version"]:
        result["torch_rocm_tag"] = result["torch_version"].split("+rocm", 1)[1]
    result["hip_version"] = getattr(getattr(torch, "version", None), "hip", None)
    result["torch_file"] = getattr(torch, "__file__", None)
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["device_count"] = int(torch.cuda.device_count())
        for index in range(result["device_count"]):
            properties = torch.cuda.get_device_properties(index)
            result["devices"].append({
                "index": index,
                "name": getattr(properties, "name", None),
                "gfx": getattr(properties, "gcnArchName", None),
            })
        if result["cuda_available"] and result["device_count"]:
            try:
                value = torch.ones((2, 2), device="cuda")
                product = value @ value
                torch.cuda.synchronize()
                expected = torch.full((2, 2), 2.0, device="cuda")
                if not bool(torch.allclose(product, expected)):
                    raise RuntimeError("tensor result did not match expected value")
                result["tensor_smoke_status"] = "passed"
            except Exception as error:
                result["tensor_smoke_status"] = "failed"
                result["tensor_smoke_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["torch_error"] = f"{type(error).__name__}: {error}"

try:
    from importlib import metadata
    for distribution in metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower()
        if name.startswith(("rocm", "amd-torch-device", "amd-torchvision-device")):
            result["rocm_packages"][name] = distribution.version
except Exception as error:
    result["torch_error"] = result["torch_error"] or f"{type(error).__name__}: {error}"

print(json.dumps(result, sort_keys=True))
'''


@dataclass(frozen=True)
class RuntimeObservation:
    """Target-scoped runtime and hardware observation."""

    target_root: Path
    python_executable: Path | None
    host_platform: str
    runtime_status: str
    hardware_status: str
    python_version: str | None = None
    torch_version: str | None = None
    torch_rocm_tag: str | None = None
    hip_version: str | None = None
    torch_file: str | None = None
    rocm_packages: dict | None = None
    cuda_available: bool = False
    device_count: int = 0
    devices: tuple[dict, ...] = ()
    tensor_smoke_status: str | None = None
    tensor_smoke_error: str | None = None
    error: str | None = None

    def as_dict(self):
        return {
            "target_root": str(self.target_root),
            "python_executable": str(self.python_executable) if self.python_executable else None,
            "runtime_scope": "target",
            "host_platform": self.host_platform,
            "runtime_status": self.runtime_status,
            "hardware_status": self.hardware_status,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_rocm_tag": self.torch_rocm_tag,
            "hip_version": self.hip_version,
            "torch_file": self.torch_file,
            "rocm_packages": self.rocm_packages or {},
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "devices": list(self.devices),
            "tensor_smoke_status": self.tensor_smoke_status,
            "tensor_smoke_error": self.tensor_smoke_error,
            "error": self.error,
        }


def _host_platform():
    return "windows" if os.name == "nt" else "linux"


def _clean_environment(target):
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        environment.pop(name, None)
    local_paths = [
        target.root,
        target.python_executable.parent if target.python_executable else None,
        target.root / "bin",
        target.root / "Library" / "bin",
        target.root / "Scripts",
    ]
    local_path_text = os.pathsep.join(str(path) for path in local_paths if path)
    if local_path_text:
        environment["PATH"] = local_path_text + os.pathsep + environment.get("PATH", "")
    return environment


def target_python_tag(target, timeout=10):
    """Return the selected interpreter's CPython wheel tag when available."""

    if target.python_executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                str(target.python_executable),
                "-c",
                "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
            ],
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    for line in reversed(completed.stdout.splitlines()):
        value = line.strip()
        if value.startswith("cp") and value[2:].isdigit():
            return value
    return None


def _parse_probe_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        for line in reversed(output.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "torch_version" in value:
                return value
        raise


def probe_target(target, timeout=30):
    """Probe only the Python and libraries belonging to ``target``.

    No global ROCm executable is invoked. If the target Python cannot import
    Torch or enumerate a device, the corresponding status remains negative.
    """

    if target.python_executable is None:
        return RuntimeObservation(
            target_root=target.root,
            python_executable=None,
            host_platform=_host_platform(),
            runtime_status="not_detected",
            hardware_status="not_detected",
            error="target Python executable was not found",
        )

    try:
        completed = subprocess.run(
            [str(target.python_executable), "-c", _PROBE_SCRIPT],
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return RuntimeObservation(
            target_root=target.root,
            python_executable=target.python_executable,
            host_platform=_host_platform(),
            runtime_status="not_detected",
            hardware_status="not_detected",
            error=f"{type(error).__name__}: {error}",
        )

    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "target Python probe failed").strip()
        return RuntimeObservation(
            target_root=target.root,
            python_executable=target.python_executable,
            host_platform=_host_platform(),
            runtime_status="not_detected",
            hardware_status="not_detected",
            error=error[-1000:],
        )

    try:
        result = _parse_probe_output(completed.stdout)
    except json.JSONDecodeError as error:
        return RuntimeObservation(
            target_root=target.root,
            python_executable=target.python_executable,
            host_platform=_host_platform(),
            runtime_status="not_detected",
            hardware_status="not_detected",
            error=f"invalid target probe output: {error}",
        )

    torch_version = result.get("torch_version")
    device_count = int(result.get("device_count") or 0)
    cuda_available = bool(result.get("cuda_available"))
    devices = tuple(result.get("devices") or ())
    return RuntimeObservation(
        target_root=target.root,
        python_executable=target.python_executable,
        host_platform=_host_platform(),
        runtime_status="detected" if torch_version else "not_detected",
        hardware_status="detected" if cuda_available and device_count else "not_detected",
        python_version=result.get("python_version"),
        torch_version=torch_version,
        torch_rocm_tag=result.get("torch_rocm_tag"),
        hip_version=result.get("hip_version"),
        torch_file=result.get("torch_file"),
        rocm_packages=result.get("rocm_packages") or {},
        cuda_available=cuda_available,
        device_count=device_count,
        devices=devices,
        tensor_smoke_status=result.get("tensor_smoke_status"),
        tensor_smoke_error=result.get("tensor_smoke_error"),
        error=result.get("torch_error"),
    )
