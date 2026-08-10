# ROCm Stack Manager

An unofficial ROCm environment manager for ComfyUI installations and other
ROCm application runtimes.

The manager never assumes a machine-specific installation path. A target is
provided explicitly or resolved relative to the current working directory.
The first command detects an existing portable or virtual-environment layout:

```powershell
python -m rocm_stack_manager detect --target C:\path\to\comfyui-portable
python -m rocm_stack_manager verify --target C:\path\to\comfyui-portable --json
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
The `--rocm` filter includes historical candidates when the Matrix catalog has
preserved their complete artifact evidence. A historical version is not shown
as installable merely because a release name exists in documentation.
The `inventory` command reads installed distributions from the selected target
Python and classifies them against one candidate. Compiled extensions without
matching evidence remain `unknown`; they are never assumed compatible.
The `install` command emits a target-local pip command by default. It never
executes pip unless `--apply` is explicitly provided, and candidates with failed
resolver evidence produce an explicit warning requiring `--allow-unverified`.
Passing `--apply` explicitly creates a target-local package backup before
running pip. Applying a failed-resolver candidate additionally requires
`--allow-unverified`.
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
