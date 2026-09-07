# Existing ComfyUI E2E

This smoke test validates Manager against a real, version-pinned ComfyUI
source checkout and a disposable target environment. It does not modify a
user's production ComfyUI installation.

## Target

- Source: `https://github.com/Comfy-Org/ComfyUI`
- Tag: `v0.34.0`
- Commit: `12d5279438bfefc058a269eae805ceab6047777f`
- Python: 3.12 target-local `.venv`
- Matrix input: `tests/fixtures/matrix-bundle-v1`
- Validation GPU: AMD Radeon RX 9070 XT / `gfx1201`

Prepare a disposable checkout and target environment:

```powershell
$target = "C:\path\to\ComfyUI-v0.34.0"
git clone --branch v0.34.0 --depth 1 `
  https://github.com/Comfy-Org/ComfyUI.git $target
python -m venv "$target\.venv"
& "$target\.venv\Scripts\python.exe" -m pip install -r "$target\requirements.txt"
```

The target Python must be used for every target operation. The Manager's own
`.venv` is only used to invoke the Manager CLI.

## Flow

```powershell
$manager = ".\.venv\Scripts\python.exe"
$bundle = "tests\fixtures\matrix-bundle-v1"
$candidate = "therock:stable:7.14.0:2.12.0+rocm7.14.0:0.27.0+rocm7.14.0:2.11.0+rocm7.14.0:cp310,cp311,cp312,cp313,cp314"

& $manager -m rocm_stack_manager detect --target $target --json
& $manager -m rocm_stack_manager plan `
  --target $target --catalog-bundle $bundle --platform windows `
  --gfx gfx1201 --candidate $candidate --json
& $manager -m rocm_stack_manager resolve `
  --target $target --catalog-bundle $bundle --platform windows `
  --gfx gfx1201 --candidate $candidate --json
& $manager -m rocm_stack_manager install `
  --target $target --catalog-bundle $bundle --platform windows `
  --gfx gfx1201 --candidate $candidate --apply --json
& $manager -m rocm_stack_manager verify --target $target --json
```

The ComfyUI startup smoke should run with custom and API nodes disabled:

```powershell
& "$target\.venv\Scripts\python.exe" "$target\main.py" `
  --quick-test-for-ci --disable-all-custom-nodes --disable-api-nodes `
  --disable-auto-launch
```

For the minimal API workflow smoke, start ComfyUI on a disposable port and
submit an `EmptyImage -> PreviewImage` prompt. This exercises node loading,
GPU tensor creation, prompt validation, queueing, and output completion without
requiring a model checkpoint.

Restore with the backup emitted by `install --apply`:

```powershell
& $manager -m rocm_stack_manager restore `
  --target $target --backup $backup --json
& $manager -m rocm_stack_manager restore `
  --target $target --backup $backup --apply --json
```

After restore, run `main.py --quick-test-for-ci --cpu` and inspect the target
inventory. Restore reinstalls the recorded, hash-verified package set but does
not prune unrelated extra packages.

## Observed result

The pinned source target passed:

- target detection selected the target-local `.venv`
- pip dry-run resolved the Matrix candidate
- install created a package backup and applied the candidate from a hashed local wheelhouse
- runtime and tensor verification passed with ROCm 7.14 and Torch 2.12.0
- ComfyUI startup passed with `gfx1201`
- the minimal `EmptyImage -> PreviewImage` workflow completed successfully
- restore dry-run and apply passed
- post-restore CPU startup passed and the original Torch 2.14.0 environment was restored

The official `ComfyUI_windows_portable_amd.7z` asset was tested separately for
read-only detection and verification. Its embedded `pip freeze` contains a
build-machine-local `file://` wheel path, so Manager correctly refuses to apply
changes when it cannot create a safe backup. Portable apply/restore is not
silently supported by stripping that metadata.
