# ROCm Stack Manager

An unofficial ROCm environment manager for ComfyUI installations and other
ROCm application runtimes.

The manager never assumes a machine-specific installation path. A target is
provided explicitly or resolved relative to the current working directory.

## Install from a clone

The repository uses a standard `src` package layout. Install it into a
virtual environment before invoking the module or console script; do not set
`PYTHONPATH` manually.

```powershell
git clone <repository-url>
cd rocm-stack-manager
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --editable .
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m rocm_stack_manager --help
.\.venv\Scripts\rocm-stack-manager.exe --help
.\.venv\Scripts\python.exe -m pip install --editable ".[ui]"
.\.venv\Scripts\rocm-stack-manager-gui.exe
# Or use the repository launcher:
.\run-rocm-stack-manager.bat
```

After activation, the examples below can use `python -m rocm_stack_manager`.
On Linux, use `.venv/bin/activate`, `.venv/bin/python`, and
`.venv/bin/rocm-stack-manager` instead.

The first command after installation detects an existing portable or
virtual-environment layout:

```powershell
python -m rocm_stack_manager detect --target C:\path\to\comfyui-portable
python -m rocm_stack_manager verify --target C:\path\to\comfyui-portable --json
python -m rocm_stack_manager extensions `
  --target C:\path\to\comfyui-portable
python -m rocm_stack_manager extensions plan `
  --target C:\path\to\comfyui-portable `
  --candidate <candidate-id>
python -m rocm_stack_manager extensions resolve `
  --target C:\path\to\comfyui-portable `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --candidate <candidate-id> --extension bitsandbytes
python -m rocm_stack_manager extensions verify `
  --target C:\path\to\comfyui-portable `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --candidate <candidate-id> --extension bitsandbytes `
  --output .rocm-stack-manager\evidence\bitsandbytes.json
python -m rocm_stack_manager extensions apply `
  --target C:\path\to\comfyui-portable `
  --candidate <candidate-id> --extension <extension-id> --apply
python -m rocm_stack_manager extensions restore `
  --target C:\path\to\comfyui-portable `
  --backup C:\path\to\extensions-<timestamp>.json
python -m rocm_stack_manager inventory `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 `
  --candidate <candidate-id>
python -m rocm_stack_manager install `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 `
  --candidate <candidate-id>
python -m rocm_stack_manager candidates `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 --channel stable
python -m rocm_stack_manager candidates `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 --rocm 7.2.1
python -m rocm_stack_manager plan `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 --candidate <candidate-id>
python -m rocm_stack_manager restore `
  --target C:\path\to\comfyui-portable `
  --backup C:\path\to\backup\packages-<timestamp>.json
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
Artifact-only records remain visible for provenance and are rejected by plan
and install commands.
When the catalog is from ROCM Evidence Matrix, the adjacent
`profiles/comfyui/profile.json` is loaded automatically. Candidate output then
includes the profile status and warnings without promoting a candidate to
runtime compatibility.
The `--rocm` filter includes historical candidates when the Matrix catalog has
preserved their complete artifact evidence. A historical version is not shown
as installable merely because a release name exists in documentation.
When `--catalog` is omitted, the first command that needs Matrix data fetches
the official generated catalog and referenced evidence files from
`rocm-evidence-matrix` into a per-user cache. On Windows the default cache is
`%LOCALAPPDATA%\rocm-stack-manager\matrix`; on Linux it is
`~/.cache/rocm-stack-manager/matrix` (or `$XDG_CACHE_HOME`). Later commands
reuse that snapshot. Use `--refresh-catalog` to request a new snapshot, or
pass `--catalog` for a fully offline local file. `--catalog-url` is available
for a trusted mirror or a local test server.

The `inventory` command reads installed distributions from the selected target
Python and classifies them against one candidate. Compiled extensions without
matching evidence remain `unknown`; they are never assumed compatible.
The `extensions` command reports known ComfyUI extensions from local inventory
and, when available, Matrix-observed artifact versions and target wheel-tag
matches. With `--catalog`, it also displays the Matrix extension profile and
its claim/evidence status. It does not contact package hosts, install
extensions, or treat an installed extension as compatible without ABI/runtime
evidence. `not_collected`, `artifact_available`, and compatibility evidence
remain separate states.
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
executes pip unless `--apply` is explicitly provided, and candidates with failed
resolver evidence produce an explicit warning requiring `--allow-unverified`.
Passing `--apply` explicitly creates a target-local package backup before
running pip. Applying a failed-resolver candidate additionally requires
`--allow-unverified`.
During an apply, stale managed ROCm and PyTorch packages are removed before
the selected candidate is installed; unrelated application extensions remain.
Plans warn when the selected package set does not include `torchaudio`; this is
safe for image-only ComfyUI use but may affect audio workflows.
The `restore` command (also available as `rollback`) accepts a backup JSON path
and is dry-run by default. It reinstalls recorded versions with
`--force-reinstall`; it does not remove extra packages.
The `verify` command executes only the selected target's Python interpreter. It
does not call a globally installed ROCm executable; host GPU state is reported
separately from target-local Torch, HIP, and ROCm package metadata.

The optional PySide6 interface is an inspection and safety-gated operation
layer over the same core services. It supports target detection, Matrix
catalog loading, candidate filtering, inventory, runtime verification, and
core/extension dry-runs. Candidate filters include distribution family,
current/historical lifecycle, channel, ROCm version, and candidate state. The
table shows the exact candidate ID so historical artifact records are not
silently collapsed into one row per channel. Core package apply and
package/extension restore require a completed dry-run,
target-local backup, and explicit confirmation. Extension apply remains
disabled unless every selected extension has exact Matrix source and evidence.
After target detection, the UI also runs a local ComfyUI extension inventory
without network access. Installed extensions are shown separately from Matrix
claim status; an installed extension with missing ABI evidence remains
`unknown` and cannot produce an install command.
When the UI starts, it performs a host-scoped `hipInfo`/`rocminfo` probe when
one is available and shows the result as a provisional GFX hint. This does not
claim that a candidate is compatible. After target detection, the UI probes
the target Python and target-local tools and replaces the hint with target
runtime evidence. Multiple detected GFX targets remain a manual selection; no
GFX value is inferred from a GPU name or hardcoded into the interface. The
shared hardware layer can also use target-local or explicitly available
`hipInfo`/`rocminfo` tools for native runtimes such as Ollama.

On Linux, create the venv with `python3 -m venv .venv`, install the optional UI
with `.venv/bin/python -m pip install --editable '.[ui]'`, and run
`./run-rocm-stack-manager.sh`. The launcher resolves its own repository root,
uses only the local `.venv`, and passes additional arguments to the UI.

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
│     └─ gui/
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
python -m unittest discover -s tests -v
```
