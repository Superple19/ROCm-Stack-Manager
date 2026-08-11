"""Command-line entry point for ROCM Stack Manager."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .adapters.registry import available_adapters, get_adapter
from .core.adapter import (
    CapabilityUnavailable,
    ExtensionInstaller,
    ExtensionProvider,
    ExtensionVerifier,
    PythonPackageAdapter,
    RuntimeAdapter,
    TargetProbeAdapter,
)
from .core.catalog import CatalogError, ensure_catalog, iter_candidates, load_catalog
from .core.backup import BackupError, load_backup, migrate_backup
from .core.detection import TargetDetectionError
from .core.hardware import detected_gfx_targets, normalize_gfx
from .core.install import (
    InstallationError,
    InstallResult,
    apply_install,
    apply_restore,
    build_restore_plan,
    dry_run_install,
)
from .core.extension_resolver import run_extension_resolver
from .core.resolver import run_resolver
from .core.planning import PlanningError
from .core.platforms import SUPPORTED_PLATFORMS, UNSUPPORTED_PLATFORM, host_platform


def _host_platform():
    return host_platform()


def _add_catalog_options(parser):
    parser.add_argument("--catalog", type=Path, help="Optional local Matrix catalog.json or matrix.json")
    parser.add_argument(
        "--catalog-url",
        default=None,
        help="Matrix raw repository base URL (or ROCM_MATRIX_CATALOG_URL) used when --catalog is omitted",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Refresh the cached Matrix catalog before use",
    )
    parser.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    detected_platform = _host_platform()
    parser.add_argument(
        "--platform",
        choices=SUPPORTED_PLATFORMS,
        default=None if detected_platform == UNSUPPORTED_PLATFORM else detected_platform,
    )
    parser.add_argument(
        "--gfx",
        help="GFX target, for example gfx1201; infer a single target-local GFX when omitted",
    )
    parser.add_argument("--channel", choices=("stable", "nightly", "staging"))
    parser.add_argument("--rocm", dest="rocm_version", help="Exact ROCm version filter")


def _add_adapter_option(parser):
    parser.add_argument(
        "--adapter",
        choices=available_adapters(),
        default="comfyui",
        help="Application adapter (default: comfyui)",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="rocm-stack-manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Inspect an existing ComfyUI installation")
    detect.add_argument("--target", type=Path, default=Path("."), help="Portable root or ComfyUI directory")
    _add_adapter_option(detect)
    detect.add_argument("--json", action="store_true", dest="json_output")

    verify = subparsers.add_parser("verify", help="Probe target-local ROCm runtime and GPU")
    verify.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    _add_adapter_option(verify)
    verify.add_argument("--json", action="store_true", dest="json_output")

    resolve = subparsers.add_parser("resolve", help="Run a targeted core pip dry-run")
    _add_catalog_options(resolve)
    _add_adapter_option(resolve)
    resolve.add_argument("--candidate", required=True, help="Exact Matrix candidate ID")
    resolve.add_argument("--json", action="store_true", dest="json_output")
    resolve.add_argument("--output", type=Path, help="Optional JSON output path")

    candidates = subparsers.add_parser("candidates", help="List Matrix package candidates")
    _add_catalog_options(candidates)
    _add_adapter_option(candidates)
    candidates.add_argument("--json", action="store_true", dest="json_output")
    candidates.add_argument(
        "--candidate-kind",
        choices=("installable", "artifact_only", "unavailable", "all"),
        default="installable",
        help="Show only this record kind (default: installable; use all for provenance records)",
    )
    candidates.add_argument("--include-unavailable", action="store_true")
    candidates.add_argument("--include-incompatible", action="store_true")

    plan = subparsers.add_parser("plan", help="Create a non-mutating installation plan")
    _add_catalog_options(plan)
    _add_adapter_option(plan)
    plan.add_argument("--candidate", required=True, help="Exact Matrix candidate ID")
    plan.add_argument("--json", action="store_true", dest="json_output")

    inventory = subparsers.add_parser("inventory", help="Inspect target-local packages")
    _add_catalog_options(inventory)
    _add_adapter_option(inventory)
    inventory.add_argument("--candidate", required=True, help="Exact Matrix candidate ID")
    inventory.add_argument("--json", action="store_true", dest="json_output")

    extensions = subparsers.add_parser("extensions", help="Report, plan, or manage ComfyUI extensions")
    extensions.add_argument(
        "action",
        nargs="?",
        choices=("report", "plan", "resolve", "verify", "apply", "restore"),
        default="report",
    )
    extensions.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    extensions.add_argument("--catalog", type=Path, help="Optional Matrix catalog.json for profile evidence")
    extensions.add_argument(
        "--catalog-url",
        help="Matrix raw repository base URL (or ROCM_MATRIX_CATALOG_URL) used when catalog is omitted",
    )
    extensions.add_argument("--refresh-catalog", action="store_true")
    extensions.add_argument(
        "--offline",
        action="store_true",
        help="Do not fetch Matrix data; report only target-local inventory",
    )
    extensions.add_argument("--candidate", help="Optional exact Matrix candidate ID")
    extensions.add_argument("--extension", dest="selections", action="append", default=[])
    extensions.add_argument("--backup", type=Path, help="Extension backup JSON for restore")
    extensions.add_argument("--backup-dir", type=Path, help="Optional extension backup directory")
    extensions.add_argument("--apply", action="store_true", help="Apply or restore after an explicit dry-run")
    extensions.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow an artifact-matched unverified extension after manual preflight review",
    )
    _add_adapter_option(extensions)
    extensions.add_argument("--json", action="store_true", dest="json_output")
    extensions.add_argument("--output", type=Path, help="Optional JSON output path for extension verification")

    install = subparsers.add_parser("install", help="Create a target-local package install dry-run")
    _add_catalog_options(install)
    _add_adapter_option(install)
    install.add_argument("--candidate", required=True, help="Exact Matrix candidate ID")
    install.add_argument("--allow-unverified", action="store_true")
    install.add_argument("--apply", action="store_true", help="Apply after creating a package backup")
    install.add_argument("--backup-dir", type=Path, help="Optional backup directory")
    install.add_argument("--json", action="store_true", dest="json_output")

    restore = subparsers.add_parser(
        "restore",
        aliases=("rollback",),
        help="Restore recorded target package versions",
    )
    restore.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    restore.add_argument("--backup", type=Path, required=True, help="Backup JSON created by install --apply")
    _add_adapter_option(restore)
    restore.add_argument("--apply", action="store_true", help="Apply the restore; default is dry-run")
    restore.add_argument(
        "--allow-network-restore",
        action="store_true",
        help=(
            "Explicitly allow a version-pinned network restore when no valid pre-change "
            "wheelhouse is available"
        ),
    )
    restore.add_argument("--json", action="store_true", dest="json_output")

    migrate = subparsers.add_parser(
        "migrate-backup",
        help="Convert a legacy backup to the current hashed sidecar format",
    )
    migrate.add_argument("--backup", type=Path, required=True, help="Legacy backup JSON")
    migrate.add_argument("--output", type=Path, required=True, help="New backup JSON path")
    migrate.add_argument(
        "--kind",
        choices=("core", "extensions"),
        help="Backup kind when the legacy JSON does not identify it",
    )
    migrate.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _print_target(target, json_output):
    if json_output:
        print(json.dumps(target.as_dict(), indent=2, sort_keys=True))
        return
    values = target.as_dict()
    print(f"Target root: {values['root']}")
    print(f"ComfyUI: {values['comfyui_dir']}")
    print(f"Layout: {values['layout']}")
    print(f"Python: {values['python_executable'] or 'not found'}")
    print(f"Launchers: {', '.join(values['launchers']) or 'none found'}")


def _print_candidates(candidates, json_output):
    if json_output:
        print(json.dumps(candidates, indent=2, sort_keys=True))
        return
    if not candidates:
        print("No matching candidates.")
        return
    installable = [item for item in candidates if item.get("candidate_kind") == "installable"]
    artifact_only = [item for item in candidates if item.get("candidate_kind") == "artifact_only"]
    unavailable = [item for item in candidates if item.get("candidate_kind") == "unavailable"]

    def print_group(title, items):
        if not items:
            return
        print(title)
        for candidate in items:
            resolver = candidate.get("resolver_status")
            resolver_text = f" | Resolver {resolver}" if resolver else ""
            print(
                f"{candidate['id']} | {candidate['rocm_version'] or 'unknown'} | "
                f"Torch {candidate['torch_version'] or 'unknown'} | "
                f"TorchAudio {candidate.get('torchaudio_version') or 'unknown'} | "
                f"Python {candidate['python_compatibility']} | "
                f"{candidate['distribution_family']} | {candidate['status']}{resolver_text}"
            )

    print_group("Installable candidates:", installable)
    print_group("Artifact-only records (not installable):", artifact_only)
    print_group("Unavailable records:", unavailable)


def _candidate_for_plan(catalog, args, python_tag):
    candidates = iter_candidates(
        catalog,
        platform=args.platform,
        gfx=args.gfx,
        channel=args.channel,
        rocm_version=args.rocm_version,
        python_tag=python_tag,
        include_unavailable=True,
        include_incompatible=True,
    )
    for candidate in candidates:
        if candidate["id"] == args.candidate:
            return candidate
    raise CatalogError(f"candidate not found for {args.platform}/{args.gfx}: {args.candidate}")


def _resolve_gfx(target, adapter, requested):
    """Normalize an explicit GFX or infer one unambiguous target-local value."""

    if requested:
        normalized = normalize_gfx(requested)
        if normalized is None:
            raise ValueError(f"invalid GFX target: {requested}")
        return normalized, False
    if not isinstance(adapter, TargetProbeAdapter):
        raise ValueError("GFX was not provided and this adapter cannot probe a target Python")
    observation = adapter.verify(target)
    detected = detected_gfx_targets(observation)
    if len(detected) == 1:
        return detected[0], True
    if len(detected) > 1:
        values = ", ".join(detected)
        raise ValueError(f"multiple target GFX values detected ({values}); pass --gfx to choose one")
    detail = getattr(observation, "error", None)
    suffix = f" ({detail})" if detail else ""
    raise ValueError(f"target GFX was not detected; pass --gfx gfx1201 manually{suffix}")


def _print_inventory(inventory, json_output):
    if json_output:
        print(json.dumps(inventory.as_dict(), indent=2, sort_keys=True))
        return
    values = inventory.as_dict()
    print(f"Target: {values['target_root']}")
    print(f"Inventory: {values['status']}")
    print(f"Compatible: {values['summary']['compatible']}")
    print(f"Conflict: {values['summary']['conflict']}")
    print(f"Unknown: {values['summary']['unknown']}")
    for package in values["packages"]:
        print(f"{package['name']}=={package['version']} | {package['status']} | {package['reason']}")


def _summarize_values(values, limit=3):
    values = [str(value) for value in values if value]
    if len(values) <= limit:
        return ", ".join(values) or "none"
    return f"{', '.join(values[:limit])} (+{len(values) - limit} more)"


def _operation_exit_code(result):
    """Return a failure code only after a mutating operation was attempted."""

    if not getattr(result, "applied", False):
        return 0
    return 0 if result.returncode == 0 else 1


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "migrate-backup":
            snapshot = migrate_backup(args.backup, args.output, kind=args.kind)
            if args.json_output:
                print(json.dumps(snapshot.as_dict(), indent=2, sort_keys=True))
            else:
                print(f"Migrated backup: {snapshot.path}")
                print(f"Requirements: {snapshot.requirements_path}")
                print(f"SHA-256: {snapshot.requirements_sha256}")
            return 0
        adapter = get_adapter(args.adapter)
        target = adapter.detect(args.target)
        if args.command == "detect":
            _print_target(target, args.json_output)
            return 0
        if args.command == "verify":
            observation = adapter.verify(target)
            if args.json_output:
                print(json.dumps(observation.as_dict(), indent=2, sort_keys=True))
            else:
                values = observation.as_dict()
                print(f"Target: {values['target_root']}")
                print(f"Runtime scope: {values['runtime_scope']}")
                print(f"Runtime: {values['runtime_status']}")
                print(f"Hardware: {values['hardware_status']}")
                print(f"Torch: {values['torch_version'] or 'not detected'}")
                print(f"Torch ROCm tag: {values['torch_rocm_tag'] or 'not detected'}")
                print(f"HIP: {values['hip_version'] or 'not detected'}")
                print(f"ROCm packages: {', '.join(values['rocm_packages']) or 'not detected'}")
                print(f"Devices: {values['device_count']}")
                print(f"Tensor smoke: {values['tensor_smoke_status'] or 'not run'}")
                if values.get("tensor_smoke_error"):
                    print(f"Tensor smoke error: {values['tensor_smoke_error']}")
            failed = (
                observation.runtime_status != "detected"
                or observation.hardware_status != "detected"
                or observation.tensor_smoke_status == "failed"
            )
            return 1 if failed else 0

        candidate_commands = {"candidates", "plan", "inventory", "install", "resolve"}
        if args.command in candidate_commands and args.platform is None:
            raise CapabilityUnavailable(
                "unsupported host platform; Manager supports Windows and Linux only; "
                "pass --platform windows or --platform linux for read-only catalog inspection"
            )

        if args.command in {"restore", "rollback"}:
            backup = load_backup(
                args.backup,
                materialize=args.apply,
                allow_network_restore=args.allow_network_restore,
            )
            if args.apply:
                result = apply_restore(target, backup)
            else:
                result = InstallResult(plan=build_restore_plan(target, backup))
            if args.json_output:
                print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            else:
                print(f"Target: {result.plan.target_root}")
                print(f"Backup: {backup.path}")
                print(f"Command: {subprocess.list2cmdline(result.plan.command)}")
                print("Mode: applied" if args.apply else "Mode: dry-run (pip was not executed)")
                if args.apply:
                    print(f"Return code: {result.returncode}")
                for warning in result.plan.warnings:
                    print(f"Warning: {warning}")
                if result.output:
                    print(result.output.rstrip())
            return _operation_exit_code(result)

        if args.command == "extensions":
            if args.action == "restore":
                if not args.backup:
                    raise BackupError("extensions restore requires --backup")
                if not isinstance(adapter, ExtensionInstaller):
                    raise CapabilityUnavailable(
                        f"adapter does not provide extension restore: {adapter.id}"
                    )
                result = adapter.restore_extensions(target, args.backup, args.apply)
                if args.json_output:
                    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
                else:
                    print(f"Target: {result.target_root}")
                    print(f"Command: {subprocess.list2cmdline(result.command) or 'none'}")
                    print("Mode: applied" if args.apply else "Mode: dry-run (pip was not executed)")
                    for warning in result.warnings:
                        print(f"Warning: {warning}")
                    if getattr(result, "output", ""):
                        print(result.output.rstrip())
                return _operation_exit_code(result)

            extension_profiles = {}
            catalog = None
            needs_catalog = bool(args.catalog) or (
                not args.offline
                and (args.action in {"report", "plan", "resolve", "verify", "apply"} or bool(args.candidate))
            )
            if needs_catalog:
                catalog_path = ensure_catalog(
                    args.catalog,
                    base_url=args.catalog_url,
                    refresh=args.refresh_catalog,
                )
                catalog = load_catalog(catalog_path)
            extension_profiles = (catalog or {}).get("_comfyui_extension_profiles", {})
            extension_catalog = (catalog or {}).get("_extension_catalog", {})
            if not isinstance(adapter, ExtensionProvider):
                raise CapabilityUnavailable(
                    f"adapter does not provide ComfyUI extension operations: {adapter.id}"
                )
            candidate = None
            if args.candidate:
                if catalog is None:
                    raise CatalogError("--candidate requires a Matrix catalog")
                if _host_platform() not in SUPPORTED_PLATFORMS:
                    raise CapabilityUnavailable(
                        "unsupported host platform; extension candidate operations support Windows and Linux only"
                    )
                python_tag = (
                    adapter.python_tag(target)
                    if isinstance(adapter, PythonPackageAdapter)
                    else None
                )
                candidate = _candidate_for_plan(
                    catalog,
                    argparse.Namespace(
                        platform=_host_platform(),
                        gfx=None,
                        channel=None,
                        rocm_version=None,
                        candidate=args.candidate,
                    ),
                    python_tag,
                )
            if args.action == "resolve":
                if candidate is None:
                    raise CatalogError("extensions resolve requires a Matrix catalog and --candidate")
                if not args.selections:
                    raise ValueError("extensions resolve requires at least one --extension")
                plan = adapter.extension_plan(
                    target,
                    candidate,
                    tuple(args.selections),
                    extension_profiles,
                    extension_catalog,
                    args.allow_unverified,
                )
                results = run_extension_resolver(target, candidate, plan.extensions)
                payload = {
                    "candidate_id": candidate.get("id"),
                    "candidate_hash": candidate.get("candidate_hash"),
                    "results": [result.as_dict() for result in results],
                }
                if args.json_output:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    for result in results:
                        print(f"{result.extension_id} | {result.status}")
                        if result.output:
                            print(result.output.rstrip())
                return 0 if all(result.status in {"resolver_verified", "not_applicable"} for result in results) else 2
            if args.action == "verify":
                if candidate is None:
                    raise CatalogError("extensions verify requires a Matrix catalog and --candidate")
                if not args.selections:
                    raise ValueError("extensions verify requires at least one --extension")
                if not isinstance(adapter, ExtensionVerifier):
                    raise CapabilityUnavailable(
                        f"adapter does not provide extension verification: {adapter.id}"
                    )
                evidence = adapter.extension_verify(
                    target,
                    candidate,
                    tuple(args.selections),
                    extension_profiles,
                    extension_catalog,
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    temporary.replace(args.output)
                if args.json_output or args.output:
                    print(json.dumps(evidence, indent=2, sort_keys=True))
                else:
                    print(f"Candidate: {evidence['candidate_id']}")
                    print(f"Verification: {evidence['verification_level']}")
                    for extension in evidence["extensions"]:
                        print(f"{extension['extension_id']} | {extension['status']}")
                failed = evidence.get("verification_level") == "unknown" or any(
                    extension.get("status") in {"runtime_failed", "hardware_failed"}
                    for extension in evidence.get("extensions", ())
                )
                return 1 if failed else 0
            if args.action == "apply":
                if not args.apply:
                    raise InstallationError("extensions apply requires --apply")
                if not args.selections:
                    raise InstallationError("extensions apply requires at least one --extension")
                if candidate is None:
                    raise CatalogError("extensions apply requires a Matrix catalog and --candidate")
                if not isinstance(adapter, ExtensionInstaller):
                    raise CapabilityUnavailable(
                        f"adapter does not provide extension installation: {adapter.id}"
                    )
                plan = adapter.extension_plan(
                    target,
                    candidate,
                    tuple(args.selections),
                    extension_profiles,
                    extension_catalog,
                    args.allow_unverified,
                )
                resolver_results = run_extension_resolver(
                    target,
                    candidate,
                    plan.extensions,
                    selection_hash=plan.selection_hash,
                )
                plan = replace(plan, resolver_results=resolver_results)
                backup = adapter.create_extension_backup(target, plan, args.backup_dir)
                result = adapter.apply_extension_plan(target, plan, backup)
                if args.json_output:
                    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
                else:
                    print(f"Target: {result.plan.target_root}")
                    print(f"Candidate: {candidate['id']}")
                    print(f"Backup: {result.backup_path}")
                    print("Mode: applied")
                    print(f"Return code: {result.returncode}")
                    if result.output:
                        print(result.output.rstrip())
                return _operation_exit_code(result)
            if args.action == "plan":
                if candidate is None:
                    raise CatalogError("extensions plan requires a Matrix catalog and --candidate")
                plan = adapter.extension_plan(
                    target,
                    candidate,
                    tuple(args.selections),
                    extension_profiles,
                    extension_catalog,
                    args.allow_unverified,
                )
                if args.json_output:
                    print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
                else:
                    print(f"Target: {plan.target_root}")
                    print(f"Candidate: {candidate['id']}")
                    print("Mode: dry-run (no network, no installation)")
                    for extension in plan.extensions:
                        print(f"{extension['name']} | {extension['status']}")
                        print(f"  Reason: {extension['reason']}")
                        if extension["sources"]:
                            print(f"  Sources: {', '.join(extension['sources'])}")
                    for command in plan.commands:
                        print(
                            f"Command ({command['extension_id']}): "
                            f"{subprocess.list2cmdline(command['command'])}"
                        )
                    for warning in plan.warnings:
                        print(f"Warning: {warning}")
                return 0
            report = adapter.extension_inventory(target, candidate, extension_profiles, extension_catalog)
            if args.json_output:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(f"Target: {report['target_root']}")
                if catalog is None:
                    print("Mode: local inventory only (no network, no installation)")
                else:
                    print("Mode: target inventory + Matrix catalog (no extension installation)")
                for extension in report["extensions"]:
                    installed = ", ".join(
                        f"{package['name']}=={package['version']}"
                        for package in extension["installed"]
                    ) or "not installed"
                    print(f"{extension['name']} | {extension['status']} | {installed}")
                    if extension.get("matrix_profile_id"):
                        print(
                            f"  Matrix profile: {extension['matrix_profile_id']} | "
                            f"claim={extension['matrix_claim_status']}"
                        )
                    versions = _summarize_values(extension.get("available_versions") or ())
                    if not extension.get("available_versions"):
                        versions = "not collected"
                    target_match = extension.get("target_match") or "unknown"
                    evidence = _summarize_values(extension.get("catalog_evidence_refs") or ())
                    print(f"  Matrix artifacts: {versions} | target={target_match}")
                    print(f"  Evidence: {evidence}")
                    print(f"  Reason: {extension['reason']}")
            return 0

        catalog_path = ensure_catalog(
            args.catalog,
            base_url=args.catalog_url,
            refresh=args.refresh_catalog,
        )
        catalog = load_catalog(catalog_path)
        if not isinstance(adapter, PythonPackageAdapter):
            raise CapabilityUnavailable(
                f"adapter does not provide Python package candidate operations: {adapter.id}"
            )
        python_tag = adapter.python_tag(target)
        inferred_gfx = False
        if args.command in {"candidates", "plan", "inventory", "install", "resolve"}:
            args.gfx, inferred_gfx = _resolve_gfx(target, adapter, args.gfx)
            if inferred_gfx and not getattr(args, "json_output", False):
                print(f"Detected target GFX: {args.gfx}")
        if args.command == "candidates":
            candidates = iter_candidates(
                catalog,
                platform=args.platform,
                gfx=args.gfx,
                channel=args.channel,
                rocm_version=args.rocm_version,
                python_tag=python_tag,
                include_unavailable=args.include_unavailable,
                include_incompatible=args.include_incompatible,
                candidate_kind=None if args.candidate_kind == "all" else args.candidate_kind,
            )
            _print_candidates(candidates, args.json_output)
            return 0

        candidate = _candidate_for_plan(catalog, args, python_tag)
        catalog_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        if args.command == "resolve":
            result = run_resolver(
                target,
                candidate,
                catalog_hash=catalog_hash,
                adapter_id=adapter.id,
                target_gfx=args.gfx,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.replace(args.output)
            if args.json_output or args.output:
                print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            else:
                print(f"Candidate: {result.candidate_id}")
                print(f"Resolver: {result.status}")
                print("Mode: dry-run (pip was not executed for installation)")
                if result.output:
                    print(result.output.rstrip())
            return 0 if result.status == "resolver_verified" else 2
        if args.command == "inventory":
            if not isinstance(adapter, RuntimeAdapter):
                raise CapabilityUnavailable(
                    f"adapter does not provide inventory operations: {adapter.id}"
                )
            _print_inventory(adapter.inventory(target, candidate), args.json_output)
            return 0
        if args.command == "install":
            if args.apply:
                result = apply_install(
                    target,
                    candidate,
                    allow_unverified=args.allow_unverified,
                    backup_dir=args.backup_dir,
                )
            else:
                result = dry_run_install(target, candidate, allow_unverified=args.allow_unverified)
            if args.json_output:
                print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            else:
                print(f"Target: {result.plan.target_root}")
                print(f"Candidate: {candidate['id']}")
                print(f"Distribution: {candidate['distribution_family']}")
                print(f"Command: {subprocess.list2cmdline(result.plan.command)}")
                print("Mode: applied" if args.apply else "Mode: dry-run (pip was not executed)")
                if result.backup_path:
                    print(f"Backup: {result.backup_path}")
                if args.apply:
                    print(f"Return code: {result.returncode}")
                for warning in result.plan.warnings:
                    print(f"Warning: {warning}")
                if result.output:
                    print(result.output.rstrip())
            return _operation_exit_code(result)
        if not isinstance(adapter, RuntimeAdapter):
            raise CapabilityUnavailable(
                f"adapter does not provide plan operations: {adapter.id}"
            )
        plan = adapter.plan(
            target,
            candidate,
            catalog_hash=catalog_hash,
            adapter_id=adapter.id,
            target_gfx=candidate.get("gfx"),
        )
        if args.json_output:
            print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        else:
            print(f"Target: {plan.target_root}")
            print(f"Candidate: {candidate['id']}")
            print(f"Platform/GFX: {candidate['platform']} / {candidate['gfx']}")
            print(f"Distribution: {candidate['distribution_family']}")
            print(f"ROCm: {candidate['rocm_version']}")
            print(f"Torch: {candidate['torch_version']}")
            if candidate.get("resolver_status"):
                print(f"Resolver evidence: {candidate['resolver_status']}")
            if candidate.get("wheel_urls"):
                print(f"Wheel URLs: {len(candidate['wheel_urls'])}")
            print("Mode: dry-run (no files changed)")
        return 0
    except (
        TargetDetectionError,
        CatalogError,
        PlanningError,
        InstallationError,
        BackupError,
        CapabilityUnavailable,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
