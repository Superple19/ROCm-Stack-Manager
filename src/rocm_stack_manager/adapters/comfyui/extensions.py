"""Independent ComfyUI extension profiles and local compatibility reporting."""

from dataclasses import dataclass


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


def build_extension_report(inventory, profile_documents=None):
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
