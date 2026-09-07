#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH." >&2
    echo "Install uv, then run this launcher again." >&2
    exit 1
fi

cd -- "${ROOT_DIR}"
exec uv run --locked --extra ui python -m rocm_stack_manager.ui.app "$@"
