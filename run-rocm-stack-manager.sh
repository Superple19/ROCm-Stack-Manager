#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ROCm Stack Manager venv was not found: ${PYTHON}" >&2
    echo "Create it with: python3 -m venv .venv" >&2
    echo "Then install the UI with: .venv/bin/python -m pip install --editable '.[ui]'" >&2
    exit 1
fi

if ! "${PYTHON}" -c 'import PySide6' >/dev/null 2>&1; then
    echo "PySide6 is not installed in the Manager venv." >&2
    echo "Install it with: .venv/bin/python -m pip install --editable \".[ui]\"" >&2
    exit 1
fi

exec "${PYTHON}" -m rocm_stack_manager.ui.app "$@"
