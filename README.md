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
  --target C:\path\to\comfyui-portable `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json
python -m rocm_stack_manager inventory `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 `
  --candidate <candidate-id>
python -m rocm_stack_manager install `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 `
  --candidate <candidate-id>
python -m rocm_stack_manager candidates `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 --channel stable
python -m rocm_stack_manager candidates `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
  --target C:\path\to\comfyui-portable `
  --platform windows --gfx gfx1201 --rocm 7.2.1
python -m rocm_stack_manager plan `
  --catalog C:\path\to\rocm-evidence-matrix\data\catalog.json `
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
The `inventory` command reads installed distributions from the selected target
Python and classifies them against one candidate. Compiled extensions without
matching evidence remain `unknown`; they are never assumed compatible.
The `extensions` command reports known ComfyUI extensions from local inventory
only. With `--catalog`, it also displays the Matrix extension profile and its
claim/evidence status. It does not contact package hosts, install extensions,
or treat an installed extension as compatible without ABI/runtime evidence.
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

## Repository layout

```text
rocm-stack-manager/
├─ src/
│  └─ rocm_stack_manager/
│     ├─ core/
│     │  ├─ catalog.py
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
