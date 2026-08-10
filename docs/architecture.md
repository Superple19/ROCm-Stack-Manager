# Architecture

```text
ROCm Evidence Matrix catalog
            |
            v
      core services
            |
      +-----+------+
      |            |
   ComfyUI       Ollama
   adapter       adapter
```

The Matrix owns compatibility facts and evidence. This repository owns target
detection, installation planning, backups, verification, and application
launching. Adapters remain separate because ComfyUI uses a Python environment
while Ollama uses a native runtime.
