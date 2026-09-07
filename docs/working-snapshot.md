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

Use a pinned Matrix bundle instead of a standalone catalog when available:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager candidates `
  --target C:\path\to\ComfyUI_windows_portable `
  --catalog-bundle C:\path\to\rocm-matrix-catalog-2026.09.07.zip `
  --channel stable
```

For a remote release asset, use an exact versioned ZIP URL:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager candidates `
  --target C:\path\to\ComfyUI_windows_portable `
  --catalog-url https://github.com/Superple19/rocm-evidence-matrix/releases/download/catalog-2026.09.07/rocm-matrix-catalog-2026.09.07.zip `
  --channel stable
```

Without `--catalog` or `--catalog-bundle`, use `--catalog-url` with a
version-pinned Matrix Release bundle asset, or set
`ROCM_MATRIX_CATALOG_URL`. The Manager verifies the ZIP before placing it in
the platform cache. A failed download or validation never replaces an existing
verified bundle. Local and cached catalog files are always SHA-256-bound to
plans. Validated Matrix collection-status artifacts supply the UI's
failed-source list; when a direct matrix file has no status artifacts, the UI
reports that state as unknown.

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
.\.venv\Scripts\python.exe -m rocm_stack_manager resolve --target C:\path\to\target --candidate <candidate-id>
```

The last command is a dry-run. `--apply` is required to execute pip. An apply
first runs a fresh target-bound resolver preflight, stages exact artifacts in a
target-local wheelhouse, records the current environment, and archives a
separate hash-verified wheelhouse for those pre-change requirements. If an
exact requirement or package artifact cannot be archived, apply stops before
uninstalling or installing anything. The manager does not automatically delete
or restore packages after a failed install. Restore is explicit and does not
prune unrelated packages.

The target probe is scoped to the selected interpreter. A result such as
`runtime_status=detected`, `hardware_status=detected`, and
`tensor_smoke_status=passed` describes that target only; it is not a claim for
all GFX targets.

`plan` only builds the exact install command. `resolve` is the separate
targeted `pip --dry-run --ignore-installed` preflight and does not install or
promote evidence to Matrix.

Installable TheRock candidates come from Matrix `package_history` and retain
its resolver status. Independently observed current-channel versions remain
`artifact_only` until the exact set appears in package history; artifact-only
records cannot produce plan, resolve, or install commands.

Core backup schema 2 identifies its wheelhouse as the pre-change environment.
Before a restore plan is created, the manager verifies the JSON schema version,
requirements sidecar SHA-256, package-version equivalence, wheelhouse role,
manifest, and artifact hashes.
Restore dry-runs are read-only and never create a missing sidecar. Convert an
older backup explicitly, without overwriting it:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager migrate-backup `
  --backup C:\path\to\old-backup.json `
  --output C:\path\to\backup\migrated.json
```

The migration command only writes the new JSON and sidecar; it never installs,
restores, or uploads anything. New applies archive hash-verified pre-change
package bytes for offline restore. Core v1 wheelhouses are accepted only when
they match the recorded requirements. Otherwise the backup is rejected unless
the user explicitly requests a version-pinned network plan:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager restore `
  --target C:\path\to\target `
  --backup C:\path\to\backup.json `
  --allow-network-restore
```

Network restore remains a dry-run unless `--apply` is also present. Restore
does not prune extra packages, so the guarantee covers archived package bytes,
not byte-identical Python environment state.

## Extension flow

Extensions are independent from core ROCm/Torch packages:

```powershell
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions report --target C:\path\to\target
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions report --target C:\path\to\target --offline
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions plan --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id> --extension bitsandbytes
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions resolve --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id> --extension bitsandbytes
.\.venv\Scripts\python.exe -m rocm_stack_manager extensions verify --target C:\path\to\target --catalog C:\path\to\catalog.json --candidate <candidate-id> --extension bitsandbytes --output evidence.json
```

`report` uses a configured or cached verified Matrix bundle and combines it
with local inventory. Add `--offline` for local inventory only. `plan` is read-only
and emits an install command only when exact source, ABI, platform, GFX, and
Matrix evidence agree.
`resolve` is a targeted `pip --dry-run`; it never installs or promotes an
extension. `verify` runs selected imports and a small tensor smoke test and
exports evidence to a local file for the user's inspection. No command in this
working snapshot uploads verification data or sends telemetry, and the manager
does not operate a community intake or maintainer review workflow.

An extension with no exact source or evidence remains `blocked` or
`unverified`. It is never treated as generally compatible. An experimental
apply requires both `--allow-unverified` and `--apply`, after reviewing a
successful targeted preflight. Extension backups and restores are separate
from core package backups.

## PySide6 workspace

The optional GUI uses a sidebar workspace rather than a single form page:

- `Overview` shows compact Target, Package snapshot, and Matrix catalog status
  cards. The package snapshot keeps ROCm, Torch, TorchVision, and TorchAudio in
  separate metrics; catalog status includes the hash prefix, artifact count,
  and source failures. Its quick actions are `Detect target`, `Load catalog`,
  and `Find candidates`; use `View details` for formatted target, inventory, or
  catalog JSON.
- `Candidates` uses a collapsible filter rail, compact version/GFX/Python rows,
  a provenance details pane, and a bottom action bar. Selecting a row never
  applies or selects a package automatically.
- `Extensions` keeps extension status separate from core packages. Use `Refresh
  inventory`, then select a row to inspect artifact, ABI, Matrix claim, and reason
  details.
- `Activity` is summary-first. Select an operation to view its details, then
  use `Copy details` or save a local JSON file. Raw JSON is not displayed in
  status-label hover tooltips.
- `Settings` controls independent target/catalog path restoration, read-only
  startup snapshots, no automatic network refresh, appearance, activity
  retention, and diagnostic preferences. Candidates, plans, approvals, and apply
  results are never restored.

The global target bar is available on every page. `Detect` only discovers the
selected target and reads local metadata; runtime/hardware verification remains
an explicit action. Catalog refresh is also explicit and never runs in the
background. Apply and restore remain disabled until their respective dry-runs
and confirmation steps succeed.

## Troubleshooting

- `target GFX was not detected`: pass `--gfx` explicitly; no GPU name is used
  to infer a GFX target.
- `cannot fetch Matrix catalog source`: use `--catalog` or `--catalog-bundle` for
  a local snapshot, or set `ROCM_MATRIX_CATALOG_URL` to a reachable versioned
  Release asset or trusted HTTPS mirror.
- `artifact-only`: the Matrix recorded an artifact but no safe install source;
  it is visible for provenance and cannot produce an install command.
- `resolver_failed` or `unverified`: inspect the warning and run a targeted
  preflight; do not treat it as runtime or hardware verification.
