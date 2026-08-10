"""Command-line entry point for ROCm Stack Manager."""

import argparse
import json
import sys
from pathlib import Path

from .core.detection import TargetDetectionError, detect_target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="rocm-stack-manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Inspect an existing ComfyUI installation")
    detect.add_argument("--target", type=Path, default=Path("."), help="Portable root or ComfyUI directory")
    detect.add_argument("--json", action="store_true", dest="json_output")
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


def main(argv=None):
    args = parse_args(argv)
    try:
        target = detect_target(args.target)
    except TargetDetectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _print_target(target, args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
