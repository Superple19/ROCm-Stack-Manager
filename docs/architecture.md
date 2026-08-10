# Architecture

```text
ROCm Evidence Matrix catalog
            |
            v
      core services
            |
      +-----+------+
      |            |
 shared hardware  adapters
      |        +---+---+
      |        |       |
   ComfyUI   Ollama  future
   runtime   native  adapters
```

The Matrix owns compatibility facts and evidence. This repository owns target
detection, installation planning, backups, verification, and application
launching. Adapters remain separate because ComfyUI uses a Python environment
while Ollama uses a native runtime.

Hardware detection is shared and returns a `HardwareObservation` containing
GPU, GFX, driver, tool, scope, and provenance fields. The UI may collect a
host-scoped probe at startup as a provisional hint. After a target is selected,
an adapter runtime probe is preferred when it reports devices; `hipInfo` or
`rocminfo` is then used as a target-local fallback. Target evidence replaces
the host hint, and a GPU model name never infers a GFX target.

ComfyUI extension evidence follows a separate path:

```text
Matrix extension catalog
        |
        v
ComfyUI extension adapter
        |
inventory -> plan -> targeted resolver -> optional local verification export
```

An extension artifact receives its own exact candidate ID. A plan may use that
artifact only when the selected core candidate, target platform, Python ABI,
and dependency requirements match. Resolver checks are limited to the
selected core/extension pair and never install packages. Runtime and hardware
exports retain the core candidate hash and extension candidate IDs, but are
marked for manual review and are not promoted to Matrix compatibility claims
automatically.
