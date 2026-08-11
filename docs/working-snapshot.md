# Working snapshot usage

This repository is a working snapshot, not a release. It does not publish a
package, tag a version, or install anything in the background.

## First run

Create and install the local virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --editable ".[ui]"
```

On Linux, use `python3`, `.venv/bin/python`, and the `.venv/bin` launcher.
The repository launchers start only the local PySide6 UI:

```powershell
.\run-rocm-stack-manager.bat
```

```bash
./run-rocm-stack-manager.sh
```

The CLI is also available without the UI:

```text
python -m rocm_stack_manager --help
```

## Matrix catalog

The manager consumes a generated Matrix catalog; it does not clone or modify
the Matrix repository. Pass a local catalog for an offline or private setup:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager candidates `
  --target C:\path\to\ComfyUI_windows_portable `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --channel stable
```

Without `--catalog`, the first catalog command downloads the configured public
snapshot into the platform cache. Use `ROCM_MATRIX_CATALOG_URL` for a trusted
mirror. A failed download never replaces an existing cached snapshot.

## Safe core flow

```text
detect
  -> verify target Python, Torch, HIP, device, and tensor smoke
  -> candidates
  -> plan (read-only)
  -> install --apply (explicit, creates backup first)
  -> restore (dry-run by default)
```

Examples:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager detect --target C:\path\to\target
.\.venv\Scripts\python.exe -m rocm_stack_manager verify --target C:\path\to\target --json
.\.venv\Scripts\python.exe -m rocm_stack_manager candidates --target C:\path\to\target --catalog C:\path\to\catalog.json --channel stable
.\.venv\Scripts\python.exe -m rocm_stack_manager plan --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id>
.\.venv\Scripts\python.exe -m rocm_stack_manager install --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id>
```

The last command is a dry-run. `--apply` is required to execute pip. The
manager creates a target-local package backup before an apply and does not
automatically delete or restore packages after a failed install. Restore is
explicit and does not prune unrelated packages.

The target probe is scoped to the selected interpreter. A result such as
`runtime_status=detected`, `hardware_status=detected`, and
`tensor_smoke_status=passed` describes that target only; it is not a claim for
all GFX targets.

## Extension flow

Extensions are independent from core ROCm/Torch packages:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions report --target C:\path\to\target --catalog C:\path\to\catalog.json
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions plan --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id> --extension bitsandbytes
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions resolve --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id> --extension bitsandbytes
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions verify --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id> --extension bitsandbytes --output evidence.json
```

`report` is local inventory. `plan` is read-only and emits an install command
only when exact source, ABI, platform, GFX, and Matrix evidence agree.
`resolve` is a targeted `pip --dry-run`; it never installs or promotes an
extension. `verify` runs selected imports and a small tensor smoke test and
exports evidence for human review.

An extension with no exact source or evidence remains `blocked` or
`unverified`. It is never treated as generally compatible. An experimental
apply requires both `--allow-unverified` and `--apply`, after reviewing a
successful targeted preflight. Extension backups and restores are separate
from core package backups.

## Troubleshooting

- `target GFX was not detected`: pass `--gfx` explicitly; no GPU name is used
  to infer a GFX target.
- `cannot fetch Matrix catalog source`: use `--catalog` for a local snapshot or
  set `ROCM_MATRIX_CATALOG_URL` to a reachable mirror.
- `artifact-only`: the Matrix recorded an artifact but no safe install source;
  it is visible for provenance and cannot produce an install command.
- `resolver_failed` or `unverified`: inspect the warning and run a targeted
  preflight; do not treat it as runtime or hardware verification.
