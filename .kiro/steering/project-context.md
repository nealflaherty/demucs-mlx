---
title: Project Context
description: Key context about the demucs-mlx project for AI assistants
inclusion: always
---

# Project: demucs-mlx

A 1:1 C++ port of Meta's Demucs (HTDemucs) music source separation model, using Apple's MLX framework for GPU-accelerated inference on Apple Silicon.

## Architecture

- `cpp/demucs/` — C++ source files, mirroring the Python module structure from the reference `demucs` submodule
- `cpp/tests/` — Google Test unit tests for the C++ implementation
- `cpp/third_party/mlx` — MLX git submodule (C++ build dependency)
- `cpp/CMakeLists.txt` — C++ build configuration
- `swift/` — Swift port using MLX Swift (SwiftPM package)
- `tools/` — Shell scripts (build, test, download model, benchmark) and a shared Python venv
- `models/` — Downloaded model weights (gitignored, created by `tools/download_model.sh`)

## Key Rules

1. **Do not modify the math** of the Python reference implementation. The C++ code must produce identical numerical results.

2. **Keep `demucs/` as the source directory name** (not `src/`). This preserves the 1:1 mapping with the Python module.

3. **Temporary files go in `.wip/`** — drafts, one-off scripts, analysis docs, debug logs, experiments. The `.wip/` folder is gitignored. Do NOT put source code, README, configs, or tracked tests in `.wip/`.

4. **Use conventional commits** — see `conventional-commits.md` for the full spec.

5. **Shared Python venv** — all tools use `tools/.venv` with deps from `tools/requirements.txt`.

## Build & Test

```bash
./tools/build.sh          # Release build (auto-downloads model if missing)
./tools/build.sh debug    # Debug build
./tools/test.sh           # Run all tests
./tools/test.sh -R Audio  # Run tests matching "Audio"
```
