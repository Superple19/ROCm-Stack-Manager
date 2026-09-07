# ROCm Stack Manager

An unofficial ROCm environment manager for ComfyUI installations. Other ROCm
application adapters are reserved scaffolds and are not functional targets.

The manager never assumes a machine-specific installation path. A target is
provided explicitly or resolved relative to the current working directory.

## Privacy and evidence boundary

All target detection, package inventory, resolver preflight, runtime probes,
and tensor checks run locally on the selected machine. The manager has no
telemetry, background reporting, crash upload, or automatic evidence upload.
Network access is limited to an explicitly invoked catalog refresh and the
package commands the user requests, such as a resolver dry-run or install.

`verify` and `extensions verify` write evidence only to the user-selected
local output path. They never transmit that file. The manager does not operate
a community intake or maintainer review workflow. No local path, token,
hostname, driver detail, GPU result, or environment value is sent automatically.

## Install from a clone

The repository uses a standard `src` package layout. `uv` manages the local
environment and lockfile; do not set `PYTHONPATH` manually.

```powershell
git clone <repository-url>
cd rocm-stack-manager
uv sync --locked
uv run --locked rocm-stack-manager --help
uv sync --locked --extra ui
uv run --locked rocm-stack-manager-gui
# Or use the repository launcher:
.\run-rocm-stack-manager.bat
```

The examples below use `uv run --locked`, so activation is not required. On
Linux, use the same commands and `./run-rocm-stack-manager.sh` for the UI.

The first command after installation detects an existing portable or
virtual-environment layout:

```powershell
uv run --locked python -m rocm_stack_manager detect --target C:\path\to\comfyui-portable
uv run --locked python -m rocm_stack_manager verify --target C:\path\to\comfyui-portable --json
uv run --locked python -m rocm_stack_manager extensions report `
  --target C:\path\to\comfyui-portable
uv run --locked python -m rocm_stack_manager extensions plan `
  --target C:\path\to\comfyui-portable `
  --candidate <candidate-id>
uv run --locked python -m rocm_stack_manager extensions resolve `
  --target C:\path\to\comfyui-portable `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --candidate <candidate-id> --extension bitsandbytes
uv run --locked python -m rocm_stack_manager extensions verify `
  --target C:\path\to\comfyui-portable `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --candidate <candidate-id> --extension bitsandbytes `
  --output .rocm-stack-manager\evidence\bitsandbytes.json
uv run --locked python -m rocm_stack_manager extensions apply `
  --target C:\path\to\comfyui-portable `
  --candidate <candidate-id> --extension <extension-id> --apply
uv run --locked python -m rocm_stack_manager extensions restore `
  --target C:\path\to\comfyui-portable `
  --backup C:\path\to\extensions-<timestamp>.json
uv run --locked python -m rocm_stack_manager inventory `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 `
  --candidate <candidate-id>
uv run --locked python -m rocm_stack_manager install `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 `
  --candidate <candidate-id>
uv run --locked python -m rocm_stack_manager candidates `
  --target C:\path\to\comfyui-portable `
  --platform windows --channel stable
uv run --locked python -m rocm_stack_manager candidates `
  --target C:\path\to\comfyui-portable `
  --platform windows --rocm 7.2.1
uv run --locked python -m rocm_stack_manager plan `
  --target C:\path\to\comfyui-portable `
  --platform windows --candidate <candidate-id>
uv run --locked python -m rocm_stack_manager resolve `
  --target C:\path\to\comfyui-portable `
  --platform windows --candidate <candidate-id>
uv run --locked python -m rocm_stack_manager restore `
  --target C:\path\to\comfyui-portable `
  --backup C:\path\to\backup\packages-<timestamp>.json
uv run --locked python -m rocm_stack_manager migrate-backup `
  --backup C:\path\to\old-backup.json `
  --output C:\path\to\backup\migrated.json
```

The selected target is expected to contain a `ComfyUI` directory with
`main.py`, or to be the ComfyUI source directory itself. Python executables
are discovered from common portable and virtual-environment locations.

This repository is independent of the ComfyUI source tree. It does not bundle
ComfyUI, ROCm, PyTorch, or third-party wheels. The `candidates` and `plan`
commands are read-only: they consume a local Matrix catalog and produce a
dry-run plan without changing the target environment.
Candidate listing reads the selected interpreter's CPython tag and excludes
artifact candidates whose recorded wheel tags do not support it. Missing ABI
evidence remains explicitly marked as `unknown`.
Candidate output distinguishes `installable` records, which have exact
package specifications or direct wheel URLs, from `artifact_only` records,
which preserve availability evidence but cannot produce an install command.
TheRock installable package sets are derived from Matrix `package_history`,
including its resolver evidence. A current-channel combination that has not
yet appeared in package history remains `artifact_only` instead of being
assembled from independently latest package versions. Every candidate reports
`resolver_status`; `not_collected` and `resolver_failed` never imply runtime or
hardware compatibility.
The `candidates` command shows only `installable` records by default. Use
`--candidate-kind artifact_only` or `--candidate-kind all` when reviewing
provenance; artifact-only records remain rejected by plan and install commands.
When the catalog is from ROCM Evidence Matrix, the adjacent
`profiles/comfyui/profile.json` is loaded automatically. Candidate output then
includes the profile status and warnings without promoting a candidate to
runtime compatibility.
The `--rocm` filter includes historical candidates when the Matrix catalog has
preserved their complete artifact evidence. A historical version is not shown
as installable merely because a release name exists in documentation.
When `--catalog` and `--catalog-bundle` are omitted, Matrix data is read from
a verified per-user cache. On Windows the default cache is
`%LOCALAPPDATA%\rocm-stack-manager\matrix`; on Linux it is
`~/.cache/rocm-stack-manager/matrix` (or `$XDG_CACHE_HOME`). To populate or
refresh that cache, pass `--catalog-url` with a version-pinned Matrix Release
bundle asset, or set `ROCM_MATRIX_CATALOG_URL`. The URL must point to a
versioned ZIP, not `main` or `latest`. A failed download or validation never
replaces an existing verified bundle. Use `--catalog` for a fully offline JSON
file. When `--gfx` is omitted, a single GFX target reported by the selected
target Python is used automatically; multiple or undetected targets require an
explicit `--gfx` value.
For a pinned Matrix release bundle, pass `--catalog-bundle` with either the
ZIP asset or an extracted bundle directory. The Manager verifies its manifest,
contract version, artifact paths, compatibility, and SHA-256 digests before
loading `data/catalog.json`.

Remote bundle loading requires the configured Release asset or trusted HTTPS
mirror to be reachable and publicly readable. If it is unavailable, use
`--catalog` or `--catalog-bundle` with a local source. Every local or cached
catalog is bound to plans by the SHA-256 of the selected catalog file. The UI
also reads validated `collection_status:*` artifacts and shows failed source
IDs; a snapshot without those artifacts reports the source failure state as
unknown rather than zero.

The `inventory` command reads installed distributions from the selected target
Python and classifies them against one candidate. Compiled extensions without
matching evidence remain `unknown`; they are never assumed compatible.
The `extensions` command reports known ComfyUI extensions from local inventory
and Matrix-observed artifact versions and target wheel-tag matches. It uses a
configured or cached verified Matrix bundle unless `--offline` is provided. It
does not contact package hosts for extension artifacts, install extensions, or
treat an installed extension as compatible without ABI/runtime evidence.
`not_collected`, `artifact_available`, and compatibility evidence remain
separate states.
`extensions plan` adds the selected core candidate and produces a read-only
extension installation plan. It emits commands only when the Matrix profile
contains an exact source, evidence reference, artifact claim, and matching
platform/Python/GFX/ABI constraints. The current extension profiles are
unverified and therefore remain blocked from installation.
`extensions resolve` runs a targeted `pip --dry-run` for only the selected
core candidate and extension artifacts. It records the exact core candidate
hash and extension candidate ID, never installs packages, and does not test
all historical combinations. An artifact-matched but `unverified` profile may
run this preflight; the result keeps `claim_status: unverified` and is never an
automatic install approval. `extensions verify` runs selected extension
imports and a small target-local GPU tensor smoke test when the target runtime
and hardware are detected. It writes a user-reviewable evidence JSON with
runtime and hardware details; the export is explicitly not promoted to Matrix
compatibility evidence automatically.
`extensions apply` requires both the `apply` action and `--apply`; it creates
an extension-only backup before running any command. If any selected
extension is blocked, unverified, conflicting, or missing evidence, the whole
operation is rejected. An artifact-matched `unverified` extension can be
explicitly approved with `--allow-unverified` after reviewing a successful
targeted `extensions resolve` preflight; this does not promote the profile to
verified. `extensions restore` is dry-run by default and never prunes unrelated
packages.
The `install` command emits a target-local pip command by default. It never
executes pip unless `--apply` is explicitly provided. An apply first runs a
fresh target-bound resolver preflight, stages exact artifacts into a local
wheelhouse, records the current environment, and downloads a separate
hash-verified wheelhouse for that pre-change environment. No uninstall or
install runs unless the pre-change archive is complete and its normalized
package versions match the recorded requirements.
A normal apply always runs a fresh target-bound resolver and proceeds only when
that exact preflight succeeds. `--allow-unverified` is an explicit bypass of
that fresh resolver gate; it does not promote catalog evidence.
During an apply, stale managed ROCm and PyTorch packages are removed before
the selected candidate is installed; unrelated application extensions remain.
Plans warn when the selected package set does not include `torchaudio`; this is
safe for image-only ComfyUI use but may affect audio workflows.
The `restore` command (also available as `rollback`) accepts a current backup
JSON path and is dry-run by default. Backups created by an apply include a
target-local pre-change wheelhouse and SHA-256 manifest; restore uses `--no-index`,
`--find-links`, and `--require-hashes` against those local artifacts. Older
core v1 backups are accepted only when their wheelhouse matches the recorded
requirements. A missing, damaged, or mismatched v1 wheelhouse is rejected by
default; `--allow-network-restore` explicitly opts into a version-pinned
network restore using the original requirements sidecar after its SHA-256 and
content have been validated.
Restore does not remove extra packages. The backup schema, requirements
sidecar, wheelhouse role, package-version equivalence, manifest, and hashes are validated before a plan is
produced. A dry-run never creates or repairs backup files. Backups created
without a hashed requirements sidecar must first be converted with
`migrate-backup`, which never overwrites the source. Hash verification applies
to archived package bytes; restore does not claim byte-identical Python state
because extra packages are not pruned.
The `verify` command executes only the selected target's Python interpreter. It
does not call a globally installed ROCm executable; host GPU state is reported
separately from target-local Torch, HIP, and ROCm package metadata.
The `resolve` command runs one selected core candidate through
`pip --dry-run --ignore-installed`. It never modifies the target, does not
write Matrix evidence, and returns `resolver_verified` or `resolver_failed`
for that candidate only. Use `--output` to save an explicit local result.

The optional PySide6 interface is an inspection and safety-gated operation
layer over the same core services. Its workspace shell has an `Overview`
page, `Candidates`, `Extensions`, `Activity`, and `Settings` pages, with a
global adapter/target bar and a sidebar for navigation. Overview shows compact
Target, Package snapshot, and Matrix catalog status surfaces; the package card
keeps ROCm, Torch, TorchVision, and TorchAudio as separate metrics. Long package
records are available through `View details`, not hover tooltips. Candidates
uses a filter rail, a list-detail layout, and an action bar for verification and
  dry-runs. Candidate rows show version set, GFX, Python, platform, candidate kind,
  evidence, resolver state, and text warnings; stable identity and provenance stay
  in the details pane. Extensions shows a compact status table with a separate
  evidence details pane and an explicit `Refresh inventory` action. Activity is
  summary-first: each operation is a selectable timeline item whose formatted JSON
  can be copied or saved on demand. Candidate filters
include distribution family, current/historical lifecycle, channel, ROCm
version, and candidate state. Core package apply and package/extension restore
require a completed dry-run, target-local backup, and explicit confirmation.
Extension apply remains disabled unless every selected extension has exact Matrix
source and evidence.
  The UI persists harmless preferences such as the last target/catalog paths
  independently, filters, selected page, startup snapshot and no-auto-refresh
  preferences, appearance choices, activity retention, and window geometry through
the platform settings store.
It does not persist detection results, candidates, dry-run plans, restore plans,
or apply approvals. A saved path is used at startup only for the read-only
snapshot described below. When a
saved target still exists, startup detects its layout and reads target-local
package metadata plus the extension inventory. When a saved catalog path is a
real local file, startup loads and hash-validates it without network access.
If the saved target, catalog, and GFX filter are all available, the UI may
populate the candidate table, but it never selects a candidate automatically.
Startup does not import Torch for runtime verification, probe hardware, run a
tensor smoke test, refresh the official catalog, resolve packages, or create a
plan. Full runtime/hardware verification remains an explicit `Verify target`
action.
Target and catalog workers prepare results without mutating shared state; only
the latest generation is committed. Changing either context invalidates all
candidate and restore plans. Target/catalog controls are locked while an apply
or restore runs, and operation failures remain visible.
After target detection, the UI also runs a local ComfyUI extension inventory
without network access. Installed extensions are shown separately from Matrix
claim status; an installed extension with missing ABI evidence remains
`unknown` and cannot produce an install command.
The UI does not perform a host hardware probe on startup. After explicit target
detection, `Verify target` probes the target Python and target-local tools for
runtime and hardware evidence. Multiple detected GFX targets remain a manual
selection; no GFX value is inferred from a GPU name or hardcoded into the
interface. The shared hardware layer can also use target-local or explicitly
available `hipInfo`/`rocminfo` tools for native runtimes such as Ollama.

On Linux, install the optional UI with `uv sync --locked --extra ui`, then run
`./run-rocm-stack-manager.sh`. The launcher resolves its own repository root
and passes additional arguments to the UI.

## Repository layout

```text
rocm-stack-manager/
├─ src/
│  └─ rocm_stack_manager/
│     ├─ core/
│     │  ├─ catalog.py
│     │  ├─ hardware.py
│     │  ├─ detection.py
│     │  ├─ planning.py
│     │  ├─ install.py
│     │  ├─ backup.py
│     │  ├─ verify.py
│     │  └─ launch.py
│     ├─ adapters/
│     │  ├─ comfyui/
│     │  └─ ollama/
│     ├─ cli.py
│     └─ ui/
├─ tests/
├─ docs/
├─ LICENSE
├─ NOTICE
├─ THIRD_PARTY_NOTICES.md
└─ pyproject.toml
```

Compatibility facts remain canonical in ROCm Evidence Matrix. The adapters
consume those profiles and keep application-specific installation and launch
behavior separate.

## Development

```powershell
uv run --locked python -m unittest discover -s tests -v
```

Install the optional development quality tools with the project extra:

```powershell
uv sync --locked --extra dev
```

Install the `prek` commit-message hook once per checkout:

```powershell
uv run --locked prek install --force
```

Commit messages require a Conventional Commit subject, one blank separator
line, and consecutive `-` body bullets.

Run the required lint gate and the advisory analyses:

```powershell
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
uv run --locked deptry .
uv run --locked vulture src tests --min-confidence 100 --ignore-names options
```

`ruff check` and `pyright` are required in CI. Pyright remains in basic mode.
Formatting, dependency analysis, and dead-code analysis are advisory while
the existing codebase is being cleaned up. Do not run `ruff format --fix` as part of an unrelated
behavior change. Vulture findings require review because CLI entry points and
dynamic adapter loading can look unused to static analysis. The tools are
development-only; the normal Manager installation does not include them.
Import Linter remains intentionally deferred until the adapter boundaries are
stable.
