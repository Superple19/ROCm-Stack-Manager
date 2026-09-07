# Matrix bundle E2E smoke test

This smoke test validates the real Windows Manager flow against the pinned
Matrix bundle fixture without touching a user target.

## Disposable target

Create a ComfyUI-shaped directory and a target-local Python environment:

```powershell
$target = Join-Path ${env:TEMP} "rocm-manager-e2e"
New-Item -ItemType Directory -Force "$target\ComfyUI" | Out-Null
Set-Content "$target\ComfyUI\main.py" "# disposable target"
uv venv "$target\.venv" --python 3.12
uv pip install --python "$target\.venv\Scripts\python.exe" pip
```

Use the pinned Matrix fixture for every Manager command:

```powershell
$bundle = "tests\fixtures\matrix-bundle-v1"
$candidate = "therock:stable:7.14.0:2.12.0+rocm7.14.0:0.27.0+rocm7.14.0:2.11.0+rocm7.14.0:cp310,cp311,cp312,cp313,cp314"
```

## Flow

```powershell
uv run --locked rocm-stack-manager detect --target $target
uv run --locked rocm-stack-manager candidates `
  --target $target --catalog-bundle $bundle `
  --platform windows --gfx gfx1201 --candidate-kind installable
uv run --locked rocm-stack-manager resolve `
  --target $target --catalog-bundle $bundle `
  --platform windows --gfx gfx1201 --candidate $candidate --json
uv run --locked rocm-stack-manager install `
  --target $target --catalog-bundle $bundle `
  --platform windows --gfx gfx1201 --candidate $candidate --apply --json
uv run --locked rocm-stack-manager verify --target $target --json
uv run --locked rocm-stack-manager restore `
  --target $target --backup "$backup" --json
```

The install command must create a target-local backup and the verify result
must report detected runtime, hardware, and a passed tensor smoke. The restore
command is a dry-run unless `--apply` is explicitly supplied; an install-created
backup should produce a hash-verified wheelhouse restore command.

## GFX bootstrap

Manager probes host/target ROCm tools before asking the target Python to infer
GFX. If no tool reports an unambiguous GFX value, the command must require an
explicit `--gfx` instead of guessing. On the validation host, `amd-smi` was
present but failed to load its runtime, so the smoke test explicitly used
`gfx1201` for the RX 9070 XT. Selecting `gfx1100` produced an expected invalid
kernel image; selecting `gfx1201` completed tensor verification successfully.
