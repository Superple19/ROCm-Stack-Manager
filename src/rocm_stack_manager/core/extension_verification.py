"""Export target-local extension runtime and hardware verification evidence."""

from datetime import datetime, timezone
import json
import subprocess

from .verify import _clean_environment


_IMPORT_SCRIPT = r'''
import importlib
import json
import sys
module = importlib.import_module(sys.argv[1])
print(json.dumps({"module": sys.argv[1], "version": getattr(module, "__version__", None)}))
'''

_TENSOR_SCRIPT = r'''
import json
import torch
value = torch.tensor([1.0], device="cuda")
result = (value + 1.0).cpu().tolist()
torch.cuda.synchronize()
print(json.dumps({"result": result, "device": str(value.device)}))
'''


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_dict(value):
    return value.as_dict() if hasattr(value, "as_dict") else dict(value or {})


def _run_import(target, module, *, timeout=120, runner=subprocess.run):
    if target.python_executable is None:
        return {"status": "not_applicable", "module": module, "error": "target Python executable was not found"}
    command = [str(target.python_executable), "-c", _IMPORT_SCRIPT, module]
    try:
        completed = runner(
            command,
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "runtime_failed", "module": module, "command": command, "error": f"{type(error).__name__}: {error}"}
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return {
        "status": "runtime_verified" if completed.returncode == 0 else "runtime_failed",
        "module": module,
        "command": command,
        "returncode": completed.returncode,
        "output": output[-4000:],
    }


def _run_tensor_smoke(target, *, timeout=120, runner=subprocess.run):
    if target.python_executable is None:
        return {"status": "not_applicable", "error": "target Python executable was not found"}
    command = [str(target.python_executable), "-c", _TENSOR_SCRIPT]
    try:
        completed = runner(
            command,
            cwd=str(target.comfyui_dir),
            env=_clean_environment(target),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "hardware_failed", "command": command, "error": f"{type(error).__name__}: {error}"}
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return {
        "status": "hardware_verified" if completed.returncode == 0 else "hardware_failed",
        "command": command,
        "returncode": completed.returncode,
        "output": output[-4000:],
    }


def build_extension_verification(
    target,
    candidate,
    extension_records,
    import_names,
    runtime,
    hardware,
    *,
    timeout=120,
    runner=subprocess.run,
):
    """Verify selected imports and one target tensor smoke test without promotion."""

    runtime_values = _as_dict(runtime)
    hardware_values = _as_dict(hardware)
    results = []
    for record in extension_records:
        extension_id = record.get("id", "unknown")
        modules = tuple(import_names.get(extension_id) or ())
        imports = [_run_import(target, module, timeout=timeout, runner=runner) for module in modules]
        import_ok = bool(imports) and all(item["status"] == "runtime_verified" for item in imports)
        if not modules:
            import_status = "not_applicable"
        elif import_ok and runtime_values.get("runtime_status") == "detected":
            import_status = "runtime_verified"
        else:
            import_status = "runtime_failed"
        results.append(
            {
                "extension_id": extension_id,
                "extension_candidate_id": record.get("extension_candidate_id"),
                "status": import_status,
                "imports": imports,
                "claim_status": record.get("claim_status", "unverified"),
            }
        )

    if runtime_values.get("hardware_status") == "detected" and hardware_values.get("gfx_targets"):
        tensor = _run_tensor_smoke(target, timeout=timeout, runner=runner)
    else:
        tensor = {"status": "not_applicable", "error": "target hardware was not detected"}
    runtime_verified = (
        runtime_values.get("runtime_status") == "detected"
        and bool(results)
        and all(item["status"] == "runtime_verified" for item in results)
    )
    level = "hardware_verified" if tensor["status"] == "hardware_verified" and runtime_verified else (
        "runtime_verified" if runtime_verified else "unknown"
    )
    return {
        "schema_version": 1,
        "observed_at": _utc_now(),
        "target_root": str(target.root),
        "python_executable": str(target.python_executable) if target.python_executable else None,
        "python_tag": candidate.get("python_tag"),
        "platform": candidate.get("platform"),
        "gfx": candidate.get("gfx"),
        "candidate_id": candidate.get("id"),
        "candidate_hash": candidate.get("candidate_hash"),
        "torch_version": candidate.get("torch_version") or runtime_values.get("torch_version"),
        "rocm_version": candidate.get("rocm_version"),
        "hip_version": runtime_values.get("hip_version"),
        "driver_version": hardware_values.get("driver_version"),
        "runtime": runtime_values,
        "hardware": hardware_values,
        "extensions": results,
        "tensor_smoke": tensor,
        "verification_level": level,
        "promotion": "none",
    }
