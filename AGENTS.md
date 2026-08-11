# AGENTS.md

## Project Scope

- Build an independent ROCm environment manager for user-selected application environments.
- Keep installation, backup, restore, verification, and launch logic separate from application adapters.
- Treat `rocm-evidence-matrix` as a read-only evidence and candidate source; do not duplicate its dataset.
- Keep ComfyUI and future runtime integrations behind adapters. The core must remain application-agnostic.
- Never copy application-specific installer scripts, patched wheels, or source code from third-party projects.
- The manager is unofficial and community-maintained; it is not an AMD, ComfyUI, or Ollama product.

## Language

- Write source code, documentation, schemas, user-facing text, tests, and commit messages in English.
- Do not add translated duplicates unless localization becomes an explicit feature.

## Candidate and Evidence Rules

- A catalog candidate describes an installable package set; it is not a compatibility guarantee.
- Preserve exact ROCm, HIP, Torch, TorchVision, TorchAudio, Python ABI, platform, GFX, channel, and source identifiers.
- Keep these states distinct: `documented`, `artifact_available`, `resolver_verified`, `runtime_verified`, `hardware_verified`, `unknown`, `conflict`, and `not_installed`.
- Never promote a candidate from catalog evidence to runtime or hardware success without an exact match for candidate hash, platform, GFX, Torch, and ROCm identity.
- Treat missing evidence as unknown, not unsupported.
- Warn when a package set omits optional or workflow-relevant packages such as TorchAudio.
- Do not silently select a different platform, Python ABI, GFX target, or distribution family.

## Target Detection and Installation

- Never hard-code a ComfyUI or application path. Accept an explicit target path and discover supported layouts.
- Detect portable, virtualenv, embedded-Python, and source layouts without assuming a global ROCm installation.
- Prefer the target environment's Python executable and package manager.
- Installation is user-initiated only. Show a plan and dry-run before applying changes.
- Create a target-local backup before package changes. Restore must be explicit and must not delete unrelated packages.
- Prune only packages owned by the manager and preserve application extensions unless the user explicitly requests otherwise.
- Keep core package installation separate from optional extensions such as bitsandbytes, AITER, Flash Attention, SageAttention, and Triton.
- Unverified extensions must remain `unknown` and must never be installed as automatically compatible.

## Network and Source Policy

- Network access is allowed only for an explicit user action such as catalog refresh or package installation.
- Use documented upstream indexes and the Matrix catalog. Do not add telemetry, background update checks, or uploads.
- Use explicit timeouts, actionable errors, and bounded retries. Never log credentials or authorization headers.
- Do not execute downloaded code during catalog discovery.
- Keep local inventories, backups, logs, caches, and virtual environments out of Git.

## Windows subprocess encoding

- When running Python subprocesses on Windows, force UTF-8 output with `python -X utf8` or `PYTHONIOENCODING=utf-8`.
- Do not infer file corruption from mojibake in external-process output or from the display column of `Format-Hex`.
- Validate file bytes and decoding separately with an explicit UTF-8 reader before changing a file.

## Architecture

- `core/` owns target detection, catalog selection, planning, install, backup, restore, verification, and launch contracts.
- `adapters/` owns application-specific detection, extension policy, and launch behavior.
- `cli.py` owns argument parsing and presentation only; it must not contain package policy.
- Keep planning side-effect free. Apply, restore, and launch operations must be explicit.
- Keep local inventory and verification results separate from Matrix evidence.
- Prefer small, direct helpers and standard-library dependencies.

## Testing and Safety

- Add regression tests for candidate filtering, platform/GFX/Python matching, backup and restore, dry-run plans, extension status, and exact evidence binding.
- Use fixtures and fake runners for unit tests; do not require a real GPU or network in the default suite.
- Run the full unit suite, Python compilation, and `git diff --check` before committing.
- Never commit credentials, absolute machine paths, target-local state, generated caches, or downloaded artifacts.

## Documentation

- Document supported target layouts, dry-run/apply/restore behavior, warning states, and limitations.
- Explain that package availability, resolver success, runtime success, and hardware success are different claims.
- Keep examples path-agnostic and use placeholders rather than a contributor's local path.

## Commit Messages

- Use Conventional Commits with `type: description` or `type(scope): description`.
- Use these types when applicable: `build`, `chore`, `ci`, `data`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, or `test`.
- Use a short imperative subject.
- When a body is useful, leave exactly one blank line after the subject, then use consecutive `-` bullet lines with no blank lines between bullets.
- Pass the body as one multiline message; do not use separate `-m` options for individual bullets.
- Keep one coherent change per commit.

Example:

```text
feat: add target package inventory

- Detect portable and virtualenv Python layouts
- Record installed ROCm and Torch package versions
- Keep local paths out of exported evidence
```

## Pull Requests

- Keep pull requests focused and describe behavior, safety implications, tests, and remaining limitations.
- Do not combine unrelated application adapters, package policy, and broad refactors without a clear reason.
