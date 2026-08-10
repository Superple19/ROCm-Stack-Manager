"""Command-line entry point for ROCM Stack Manager."""

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters.comfyui.detect import detect_comfyui, python_tag_comfyui, verify_comfyui
from .core.catalog import CatalogError, iter_candidates, load_catalog
from .core.detection import TargetDetectionError
from .core.inventory import collect_inventory
from .core.planning import PlanningError, build_plan


def _host_platform():
    return "windows" if os.name == "nt" else "linux"


def _add_catalog_options(parser):
    parser.add_argument("--catalog", type=Path, required=True, help="Matrix catalog.json or matrix.json")
    parser.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    parser.add_argument("--platform", choices=("windows", "linux"), default=_host_platform())
    parser.add_argument("--gfx", required=True, help="GFX target, for example gfx1201")
    parser.add_argument("--channel", choices=("stable", "nightly", "staging"))
    parser.add_argument("--rocm", dest="rocm_version", help="Exact ROCm version filter")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="rocm-stack-manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Inspect an existing ComfyUI installation")
    detect.add_argument("--target", type=Path, default=Path("."), help="Portable root or ComfyUI directory")
    detect.add_argument("--json", action="store_true", dest="json_output")

    verify = subparsers.add_parser("verify", help="Probe target-local ROCm runtime and GPU")
    verify.add_argument("--target", type=Path, required=True, help="Portable root or ComfyUI directory")
    verify.add_argument("--json", action="store_true", dest="json_output")

    candidates = subparsers.add_parser("candidates", help="List Matrix package candidates")
    _add_catalog_options(candidates)
    candidates.add_argument("--json", action="store_true", dest="json_output")
    candidates.add_argument("--include-unavailable", action="store_true")
    candidates.add_argument("--include-incompatible", action="store_true")

    plan = subparsers.add_parser("plan", help="Create a non-mutating installation plan")
    _add_catalog_options(plan)
    plan.add_argument("--candidate", required=True, help="Exact Matrix candidate ID")
    plan.add_argument("--json", action="store_true", dest="json_output")

    inventory = subparsers.add_parser("inventory", help="Inspect target-local packages")
    _add_catalog_options(inventory)
    inventory.add_argument("--candidate", required=True, help="Exact Matrix candidate ID")
    inventory.add_argument("--json", action="store_true", dest="json_output")
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
    for candidate in candidates:
        resolver = candidate.get("resolver_status")
        resolver_text = f" | Resolver {resolver}" if resolver else ""
        print(
            f"{candidate['id']} | {candidate['rocm_version'] or 'unknown'} | "
            f"Torch {candidate['torch_version'] or 'unknown'} | "
            f"Python {candidate['python_compatibility']} | "
            f"{candidate['distribution_family']} | {candidate['status']}{resolver_text}"
        )


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
        target = detect_comfyui(args.target)
        if args.command == "detect":
            _print_target(target, args.json_output)
            return 0
        if args.command == "verify":
            observation = verify_comfyui(target)
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

        catalog = load_catalog(args.catalog)
        python_tag = python_tag_comfyui(target)
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
            _print_inventory(collect_inventory(target, candidate), args.json_output)
            return 0
        plan = build_plan(target, candidate)
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
    except (TargetDetectionError, CatalogError, PlanningError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
