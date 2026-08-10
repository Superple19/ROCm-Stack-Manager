"""Command-line entry point for ROCM Stack Manager."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .adapters.registry import available_adapters, get_adapter
from .core.adapter import (
    CapabilityUnavailable,
    ExtensionProvider,
    PythonPackageAdapter,
)
from .core.catalog import CatalogError, iter_candidates, load_catalog
from .core.backup import BackupError, create_backup, load_backup
from .core.detection import TargetDetectionError
from .core.install import (
    InstallationError,
    InstallResult,
    apply_install,
    apply_restore,
    build_restore_plan,
    dry_run_install,
)
from .core.planning import PlanningError


def _host_platform():
    return "windows" if os.name == "nt" else "linux"


def _add_catalog_options(parser):
    parser.add_argument("--catalog", type=Path, required=True, help="Matrix catalog.json or matrix.json")
    parser.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    parser.add_argument("--platform", choices=("windows", "linux"), default=_host_platform())
    parser.add_argument("--gfx", required=True, help="GFX target, for example gfx1201")
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

    candidates = subparsers.add_parser("candidates", help="List Matrix package candidates")
    _add_catalog_options(candidates)
    _add_adapter_option(candidates)
    candidates.add_argument("--json", action="store_true", dest="json_output")
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

    extensions = subparsers.add_parser("extensions", help="Report or plan ComfyUI extensions")
    extensions.add_argument("action", nargs="?", choices=("report", "plan"), default="report")
    extensions.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    extensions.add_argument("--catalog", type=Path, help="Optional Matrix catalog.json for profile evidence")
    extensions.add_argument("--candidate", help="Optional exact Matrix candidate ID")
    extensions.add_argument("--extension", dest="selections", action="append", default=[])
    _add_adapter_option(extensions)
    extensions.add_argument("--json", action="store_true", dest="json_output")

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
    restore.add_argument("--json", action="store_true", dest="json_output")
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


def main(argv=None):
    args = parse_args(argv)
    try:
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
            return 0

        if args.command in {"restore", "rollback"}:
            backup = load_backup(args.backup)
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
            return 0

        if args.command == "extensions":
            extension_profiles = {}
            catalog = None
            if args.catalog:
                catalog = load_catalog(args.catalog)
                extension_profiles = catalog.get("_comfyui_extension_profiles", {})
            if not isinstance(adapter, ExtensionProvider):
                raise CapabilityUnavailable(
                    f"adapter does not provide ComfyUI extension operations: {adapter.id}"
                )
            candidate = None
            if args.candidate:
                if catalog is None:
                    raise CatalogError("--candidate requires --catalog")
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
            if args.action == "plan":
                if candidate is None:
                    raise CatalogError("extensions plan requires --catalog and --candidate")
                plan = adapter.extension_plan(
                    target,
                    candidate,
                    tuple(args.selections),
                    extension_profiles,
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
            report = adapter.extension_inventory(target, candidate, extension_profiles)
            if args.json_output:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(f"Target: {report['target_root']}")
                print("Mode: local inventory only (no network, no installation)")
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
                    print(f"  Reason: {extension['reason']}")
            return 0

        catalog = load_catalog(args.catalog)
        if not isinstance(adapter, PythonPackageAdapter):
            raise CapabilityUnavailable(
                f"adapter does not provide Python package candidate operations: {adapter.id}"
            )
        python_tag = adapter.python_tag(target)
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
            )
            _print_candidates(candidates, args.json_output)
            return 0

        candidate = _candidate_for_plan(catalog, args, python_tag)
        if args.command == "inventory":
            _print_inventory(adapter.inventory(target, candidate), args.json_output)
            return 0
        if args.command == "install":
            if args.apply:
                if candidate.get("resolver_status") == "resolver_failed" and not args.allow_unverified:
                    raise InstallationError("resolver evidence failed; pass --allow-unverified to apply")
                backup = create_backup(target, args.backup_dir)
                result = apply_install(
                    target,
                    candidate,
                    backup,
                    allow_unverified=args.allow_unverified,
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
            return 0
        plan = adapter.plan(target, candidate)
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
