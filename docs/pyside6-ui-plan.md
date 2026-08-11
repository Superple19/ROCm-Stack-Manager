# PySide6 UI Plan

## Purpose

Add an optional desktop interface for ROCm Stack Manager without moving
installation policy into the UI. The interface will call the existing core
services and application adapters, present their results, and require explicit
user approval for every mutating operation.

The first UI target is ComfyUI. The design must leave room for Ollama and
AI Toolkit adapters without adding their fields to the shared core model.

## Goals

- Let a user select and inspect a ComfyUI target without typing a path.
- Show the detected layout, target Python, host platform, GPU, GFX, ROCm, HIP,
  and Torch information when available.
- Load a local Matrix catalog or explicitly request the official catalog.
- Filter and display candidates by platform, GFX, channel, ROCm version, and
  Python ABI.
- Show the difference between artifact availability, resolver evidence, runtime
  evidence, and hardware evidence.
- Preview package and extension plans before any change.
- Keep apply, restore, and extension installation behind explicit confirmation.
- Preserve CLI behavior and keep the CLI usable without PySide6.

## Non-goals

- No automatic background refresh, telemetry, update check, upload, community
  intake, or maintainer review workflow. Evidence export is local-only.
- No exhaustive resolver or GPU testing from the UI.
- No automatic selection of a candidate, extension, source, GFX, or platform.
- No copied third-party installer scripts or patched wheels.
- No ComfyUI core modifications.
- No Ollama or AI Toolkit implementation in the first UI milestone.

## Dependency and packaging policy

PySide6 is an optional dependency. The base package and CLI remain usable in
an environment where Qt is not installed.

The current development floor is the latest stable PyPI release, PySide6
6.11.1. The package constraint allows newer Qt 6 releases while excluding a
future Qt 7 API break.

```text
pip install --editable ".[ui]"
rocm-stack-manager-gui
```

The UI should use QtWidgets rather than QtQuick/QML for the first version.
Target selection, tables, forms, confirmation dialogs, and log output are
standard desktop widgets and do not require a second presentation technology.

PySide6 and its Qt runtime notices must be included in the packaged
application's third-party notices. The repository's own license remains
independent of ComfyUI and Matrix licenses.

## Layering

```text
QtWidgets UI
    ↓
UI service/facade (thread and presentation boundary)
    ↓
adapter registry → ComfyUIAdapter
    ↓
core detection, catalog, inventory, planning, install, backup, restore, verify
```

The UI must not call `pip`, inspect package metadata, or parse Matrix JSON
directly. It receives typed results from the service boundary and renders the
resulting status, warnings, commands, and evidence references.

Suggested layout:

```text
src/rocm_stack_manager/ui/
  __init__.py
  app.py              # QApplication setup and entry point
  main_window.py      # QtWidgets layout and signal wiring
  models.py           # table/list presentation models
  services.py         # calls adapters and core services
  workers.py          # cancellable background jobs
```

The service boundary should accept plain paths and immutable request values,
then return existing domain dataclasses or JSON-compatible dictionaries. It
must not create a second installation or candidate model.

## User workflow

### 1. Target selection

The initial window contains:

- target path field
- folder picker
- adapter selector, initially `ComfyUI`
- `Detect` button

Detection is local-only. The result shows portable, embedded-Python, venv, or
source layout, the selected Python executable, and actionable errors if no
supported layout is found.

### 2. Runtime inspection

After detection, the UI presents a summary card:

- host OS and architecture
- target Python and Python ABI
- GPU name and GFX target, when detected
- target Torch and HIP versions
- ROCm package metadata
- runtime and hardware status

Unknown values remain `unknown`; the UI must not infer them from a filename or
from the selected candidate.

### 3. Matrix catalog

The user can choose one of three modes:

- `Local catalog`: choose `catalog.json` or `matrix.json` with no network.
- `Use cached catalog`: use the existing per-user Matrix cache.
- `Refresh official catalog`: an explicit button that fetches the configured
  Matrix source and reports the cache location and fetch error, if any.

The UI must show the catalog source and observation time. A catalog refresh is
never performed on application startup or in the background.

### 4. Candidate browser

The candidate table should include:

| Column | Meaning |
|---|---|
| Family | TheRock or legacy distribution family |
| Platform | Target operating system; never inferred from the host fallback |
| Lifecycle | current or historical observation |
| Channel | stable, nightly, or staging |
| GFX | Exact target architecture |
| ROCm | Candidate ROCm package version |
| Torch | Candidate Torch version |
| TorchVision | Candidate TorchVision version |
| Python | compatible, incompatible, or unknown |
| Candidate kind | installable, artifact-only, or unavailable |
| Evidence | documented/resolver/runtime/hardware state |
| Profile | ComfyUI profile status and warnings |
| Candidate ID | Exact immutable candidate identifier |

Filters are explicit and visible: platform, GFX, distribution family,
current/historical lifecycle, channel, ROCm version, and candidate state. The
UI shows total/current/historical/installable counts after each search. A row
selected in the UI is an exact candidate ID, not a fuzzy version request.

### 5. Plan and inventory

Selecting a candidate enables:

- `Inventory`: inspect target packages against that exact candidate.
- `Dry-run install`: show the target Python command, package specs, index URL,
  warnings, and backup scope.
- `Extension plan`: show each extension as `installable`, `blocked`,
  `unverified`, `conflict`, or `not_installed`.

The extension inventory also shows the latest Matrix artifact platform tags and
Python/ABI tags. Unknown tags remain visible as `unknown` and cannot be treated
as a compatibility claim.

After target detection, the UI also runs the ComfyUI extension inventory
without a candidate and without network access. This inventory shows the
installed package versions and Matrix claim status independently from the
candidate browser. Loading or refreshing the Matrix catalog refreshes this
extension view so a missing local profile is not mistaken for a missing
installed package.

The plan view must visibly state `No files changed` and `No package command was
executed` for dry-run results.

### 6. Apply and restore

Apply is disabled until a dry-run has completed. Before enabling it, the UI
must show:

- exact target path and Python executable
- exact candidate ID
- packages to remove and install
- backup destination
- warnings, including missing TorchAudio or failed resolver evidence

The confirmation dialog requires a second explicit confirmation. Extension
changes use a separate plan and extension-only backup. Restore starts as a
dry-run and requires its own `Apply restore` confirmation.

## State model

The UI should represent these states directly rather than collapsing them into
one compatibility badge:

```text
not_loaded
loading
loaded
documented
artifact_available
resolver_verified
resolver_failed
runtime_verified
hardware_verified
installable
blocked
unverified
conflict
not_installed
error
```

`artifact_available` is not displayed as runtime or hardware compatibility.
`unverified` and `blocked` rows never produce an apply command.

## Error and cancellation behavior

- Network, catalog, detection, and pip errors appear in a visible error panel
  with the command or source that failed.
- Authorization headers, tokens, and environment secrets are never displayed.
- A running catalog fetch or inventory operation can be cancelled before it
  commits its result to the UI.
- A failed apply preserves the backup path and return code; the UI does not
  silently run restore.
- Closing the window does not terminate a running package process without a
  confirmation.

## Implementation milestones

### UI-1: Optional shell

- Add the optional PySide6 extra and `rocm-stack-manager-gui` entry point.
- Open a window with target selection, adapter selection, and a log panel.
- Keep all controls read-only.

### UI-2: Detection and catalog

- Connect target detection through the adapter registry.
- Add local, cached, and explicit refresh catalog actions.
- Add runtime summary and catalog provenance display.

### UI-3: Candidate and plan views

- Add candidate table and visible filters.
- Connect inventory, core dry-run, and extension plan services.
- Add status rendering and evidence warnings.

### UI-4: Safe apply and restore

- Add confirmation flow, package backup display, apply output, and restore
  dry-run.
- Keep extension apply and core package apply as separate operations.
- Add cancellation and failure-state handling.

### UI-5: Adapter-ready polish

- Ensure the UI obtains labels and capabilities from the selected adapter.
- Show Ollama and AI Toolkit as unavailable until their adapters implement the
  required capabilities.
- Do not add application-specific controls to the shared core screens.

## Tests and acceptance criteria

The default test suite must not require a display, network, Matrix checkout, or
real GPU. Use fake services and Qt's offscreen platform for widget tests.

Required tests:

- UI starts without PySide6 installed when only the CLI is used.
- Target selection passes the chosen path unchanged to the adapter.
- Local catalog mode never calls the network.
- Refresh is explicit and reports failures without replacing a valid cache.
- Candidate filters preserve exact platform, GFX, Python, channel, and ROCm
  values.
- Artifact-only and unverified candidates cannot enable Apply.
- Core and extension plans remain separate.
- Apply requires a completed dry-run and explicit confirmation.
- Backup and restore paths are shown and passed unchanged.
- Secrets and absolute environment credentials are redacted from logs.
- An unavailable adapter produces a clear capability message rather than an
  empty success screen.

Completion means a user can select a target, inspect exact candidates, produce
core and extension dry-runs, and understand why an operation is blocked without
the UI making an unsupported compatibility claim or mutating the environment.
