"""Independent ComfyUI extension profiles and evidence-gated planning."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from urllib.parse import urlparse

from ...core.backup import ExtensionBackupSnapshot
from ...core.verify import _clean_environment

@dataclass(frozen=True)
class ExtensionProfile:
    id: str
    display_name: str
    package_names: tuple[str, ...]
    import_names: tuple[str, ...]
    role: str
    install_policy: str = "explicit_only"


PROFILES = (
    ExtensionProfile(
        "bitsandbytes",
        "bitsandbytes",
        ("bitsandbytes",),
        ("bitsandbytes",),
        "optional_quantization",
    ),
    ExtensionProfile(
        "flash-attention",
        "Flash Attention",
        ("flash-attn", "flash-attention"),
        ("flash_attn",),
        "optional_attention",
    ),
    ExtensionProfile(
        "aiter",
        "AITER",
        ("amd-aiter", "aiter"),
        ("aiter",),
        "optional_attention",
    ),
    ExtensionProfile(
        "sageattention",
        "SageAttention",
        ("sageattention", "sage-attention"),
        ("sageattention",),
        "optional_attention",
    ),
    ExtensionProfile(
        "triton",
        "Triton",
        ("triton-windows", "triton"),
        ("triton",),
        "optional_backend",
    ),
)


def _normalize(value):
    return value.casefold().replace("_", "-")


def _matches(profile, package):
    name = _normalize(package.get("name", ""))
    return name in {_normalize(value) for value in profile.package_names}


def _matrix_constraint(document):
    constraints = document.get("constraints") or []
    return next((item for item in constraints if item.get("kind") == "extension"), {})


def _claim_status(document, constraint):
    return constraint.get("claim_status") or (document.get("metadata") or {}).get("status", "unverified")


def _evidence_refs(document, constraint):
    return list(document.get("evidence_refs") or constraint.get("evidence_refs") or [])


def _exact_sources(document, constraint):
    value = constraint.get("value") or {}
    metadata = document.get("metadata") or {}
    sources = []
    for key in ("install_sources", "wheel_urls", "package_specs"):
        values = value.get(key) or metadata.get(key) or []
        if isinstance(values, str):
            values = [values]
        sources.extend(str(item) for item in values if str(item).strip())
    source = value.get("source") or metadata.get("source")
    if source:
        sources.append(str(source))
    exact = all(
        "==" in source
        or urlparse(source).scheme in {"http", "https"}
        for source in sources
    )
    return tuple(dict.fromkeys(sources)) if sources and exact else ()


def _constraint_mismatches(document, constraint, candidate):
    value = constraint.get("value") or {}
    mismatches = []
    platform = candidate.get("platform")
    supported_os = value.get("supported_os") or value.get("platforms")
    if isinstance(supported_os, str):
        supported_os = [supported_os]
    if supported_os and platform not in supported_os:
        mismatches.append(f"platform {platform} is not supported by the profile")

    python_tags = value.get("python_tags")
    if isinstance(python_tags, str):
        python_tags = [python_tags]
    if python_tags and candidate.get("python_tag") not in python_tags:
        mismatches.append(f"Python tag {candidate.get('python_tag') or 'unknown'} is not supported")
    elif value.get("python") == "candidate-specific":
        mismatches.append("Python ABI is not established by the profile")

    gfx_targets = value.get("gfx_targets") or value.get("gfx")
    if isinstance(gfx_targets, str):
        gfx_targets = [gfx_targets]
    if gfx_targets == ["not-established"] or gfx_targets == "not-established":
        mismatches.append("GFX compatibility is not established by the profile")
    elif gfx_targets and candidate.get("gfx") not in gfx_targets:
        mismatches.append(f"GFX {candidate.get('gfx') or 'unknown'} is not supported")

    abi = value.get("torch_rocm_hip_abi")
    if abi == "not-established":
        mismatches.append("Torch/ROCm/HIP ABI is not established by the profile")
    elif isinstance(abi, dict):
        expected_values = {
            "torch": candidate.get("torch_version"),
            "torch_version": candidate.get("torch_version"),
            "rocm": candidate.get("rocm_version"),
            "rocm_version": candidate.get("rocm_version"),
            "hip": candidate.get("hip_version"),
            "hip_version": candidate.get("hip_version"),
        }
        for key, expected in abi.items():
            if key in expected_values and expected not in {None, "candidate-specific"}:
                allowed = expected if isinstance(expected, list) else [expected]
                if expected_values[key] not in allowed:
                    mismatches.append(f"{key} does not match the selected candidate")
    return mismatches


def _extension_plan_record(profile, document, candidate, installed):
    constraint = _matrix_constraint(document)
    claim_status = _claim_status(document, constraint)
    evidence_refs = _evidence_refs(document, constraint)
    sources = _exact_sources(document, constraint)
    reasons = []
    if installed and any(package.get("status") == "conflict" for package in installed):
        status = "conflict"
        reasons.append("installed package conflicts with the selected core candidate")
    elif claim_status == "unverified":
        status = "unverified"
        reasons.append("Matrix profile is unverified")
    elif claim_status not in {
        "artifact_available",
        "resolver_verified",
        "runtime_verified",
        "hardware_verified",
    }:
        status = "blocked"
        reasons.append(f"Matrix claim status is {claim_status}")
    elif not evidence_refs:
        status = "blocked"
        reasons.append("profile has no evidence references")
    elif not sources:
        status = "blocked"
        reasons.append("profile has no exact wheel URL or package specification")
    else:
        reasons.extend(_constraint_mismatches(document, constraint, candidate))
        status = "blocked" if reasons else "installable"
    return {
        "id": profile.id,
        "name": profile.display_name,
        "role": profile.role,
        "status": status,
        "claim_status": claim_status,
        "matrix_profile_id": document.get("id"),
        "evidence_refs": evidence_refs,
        "installed": [
            {"name": package["name"], "version": package["version"]}
            for package in installed
        ],
        "already_installed": bool(installed),
        "sources": list(sources),
        "reason": "; ".join(reasons) if reasons else "exact source and target constraints match",
    }


@dataclass(frozen=True)
class ExtensionPlan:
    """A read-only plan for evidence-backed ComfyUI extensions."""

    target_root: Path
    candidate: dict
    extensions: tuple[dict, ...]
    commands: tuple[dict, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "target_root": str(self.target_root),
            "candidate": self.candidate,
            "extensions": list(self.extensions),
            "commands": list(self.commands),
            "warnings": list(self.warnings),
            "network_access": False,
            "installation_performed": False,
        }


@dataclass(frozen=True)
class ExtensionInstallResult:
    """Result of an explicitly applied ComfyUI extension plan."""

    plan: ExtensionPlan
    applied: bool = False
    returncode: int | None = None
    backup_path: str | None = None
    output: str = ""

    def as_dict(self):
        values = self.plan.as_dict()
        values.update(
            {
                "applied": self.applied,
                "returncode": self.returncode,
                "backup_path": self.backup_path,
                "output": self.output,
            }
        )
        return values


def build_extension_report(inventory, profile_documents=None, candidate=None):
    """Classify known extensions using only target-local package evidence.

    A compiled extension is never promoted to compatible without a matching
    Matrix/runtime evidence record. This report deliberately performs no
    network access and produces no installation command.
    """

    profile_documents = profile_documents or {}
    records = []
    for profile in PROFILES:
        matrix_profile = profile_documents.get(profile.id) or {}
        metadata = matrix_profile.get("metadata") or {}
        constraints = matrix_profile.get("constraints") or []
        claim_status = next(
            (constraint.get("claim_status") for constraint in constraints if constraint.get("claim_status")),
            metadata.get("status", "unverified"),
        )
        evidence_refs = list(matrix_profile.get("evidence_refs") or [])
        core_required = bool(metadata.get("core_required", False))
        installed = tuple(package for package in inventory.packages if _matches(profile, package))
        if not installed:
            records.append(
                {
                    "id": profile.id,
                    "name": profile.display_name,
                    "role": profile.role,
                    "status": "not_installed",
                    "claim_status": "unverified",
                    "matrix_profile_id": matrix_profile.get("id"),
                    "matrix_claim_status": claim_status,
                    "matrix_evidence_refs": evidence_refs,
                    "core_required": core_required,
                    "installed": [],
                    "reason": "package is not installed in the target environment",
                    "install_policy": profile.install_policy,
                    "evidence_refs": [],
                }
            )
            continue

        if any(package.get("status") == "conflict" for package in installed):
            status = "conflict"
            reason = "installed package has a declared conflict with the selected core candidate"
        else:
            status = "unknown"
            reason = "installed extension lacks matching ABI/runtime evidence"
        records.append(
            {
                "id": profile.id,
                "name": profile.display_name,
                "role": profile.role,
                "status": status,
                "claim_status": "unverified",
                "matrix_profile_id": matrix_profile.get("id"),
                "matrix_claim_status": claim_status,
                "matrix_evidence_refs": evidence_refs,
                "core_required": core_required,
                "installed": [
                    {"name": package["name"], "version": package["version"]}
                    for package in installed
                ],
                "reason": reason,
                "install_policy": profile.install_policy,
                "evidence_refs": [],
            }
        )
    return {
        "target_root": str(inventory.target_root),
        "status": inventory.status,
        "error": inventory.error,
        "network_access": False,
        "installation_performed": False,
        "extensions": records,
    }


def build_extension_plan(
    target,
    inventory,
    candidate,
    profile_documents=None,
    selections=(),
):
    """Build a read-only, evidence-gated extension installation plan."""

    profile_documents = profile_documents or {}
    selected = set(selections)
    known = {profile.id for profile in PROFILES}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown ComfyUI extension: {', '.join(sorted(unknown))}")
    if not candidate.get("artifact_available"):
        raise ValueError("selected core candidate is not artifact-available")
    if candidate.get("candidate_kind") == "artifact_only":
        raise ValueError("selected core candidate has artifact evidence only")

    records = []
    commands = []
    for profile in PROFILES:
        if selected and profile.id not in selected:
            continue
        document = profile_documents.get(profile.id) or {}
        installed = tuple(package for package in inventory.packages if _matches(profile, package))
        record = _extension_plan_record(profile, document, candidate, installed)
        records.append(record)
        if record["status"] == "installable" and not record["already_installed"]:
            command = [str(target.python_executable), "-m", "pip", "install", "--no-input"]
            command.extend(record["sources"])
            commands.append({"extension_id": profile.id, "command": command})

    warnings = []
    if not commands:
        warnings.append("no extension has sufficient Matrix evidence for installation")
    return ExtensionPlan(
        target_root=target.root,
        candidate=candidate,
        extensions=tuple(records),
        commands=tuple(commands),
        warnings=tuple(warnings),
    )


def package_names_for_extensions(extension_ids=()):
    """Return package aliases owned by selected ComfyUI extensions."""

    selected = set(extension_ids)
    return tuple(
        package_name
        for profile in PROFILES
        if not selected or profile.id in selected
        for package_name in profile.package_names
    )


def apply_extension_plan(target, plan, backup, timeout=3600):
    """Apply only a fully installable extension plan."""

    blocked = [item for item in plan.extensions if item["status"] != "installable"]
    if blocked:
        names = ", ".join(item["id"] for item in blocked)
        raise ValueError(f"extension plan contains non-installable selections: {names}")
    if not plan.commands:
        return ExtensionInstallResult(
            plan=plan,
            applied=True,
            returncode=0,
            backup_path=str(backup.path),
        )

    output_parts = []
    for command_record in plan.commands:
        try:
            completed = subprocess.run(
                command_record["command"],
                cwd=str(target.comfyui_dir),
                env=_clean_environment(target),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            output_parts.append(f"{command_record['extension_id']}: {type(error).__name__}: {error}")
            return ExtensionInstallResult(
                plan=plan,
                applied=True,
                backup_path=str(backup.path),
                output="\n".join(output_parts),
            )
        output = (completed.stdout or "") + (completed.stderr or "")
        output_parts.append(f"{command_record['extension_id']}:\n{output}".rstrip())
        if completed.returncode != 0:
            return ExtensionInstallResult(
                plan=plan,
                applied=True,
                returncode=completed.returncode,
                backup_path=str(backup.path),
                output="\n".join(output_parts),
            )
    return ExtensionInstallResult(
        plan=plan,
        applied=True,
        returncode=0,
        backup_path=str(backup.path),
        output="\n".join(output_parts),
    )
