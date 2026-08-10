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
