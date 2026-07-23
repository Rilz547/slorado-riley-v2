# Bench checkout: Overlap v1.1 (2026-07-21)

## Current tree
- **Branch:** `bench-overlap-v1.1` (`f27f772`)
- **Slorado algorithm:** `9f8db24` (Overlap v1.1)
- **Openfish:** `cf182a7` (Overlap v1.1 persistent host buffers) — pinned in the bench commit
- **Runner:** current `rileys-runner.py` (from v1.2 tree); not the algorithm under test
- **`develop` remains at v1.2:** `7cec6c5` + openfish `223f26c`

## Why not `git submodule update` alone?
`9f8db24` still recorded openfish `3d95491` (pre-stream / pre-persist). That would **not** be true v1.1. The bench commit corrects the pin to `cf182a7`.

## Before running
Rebuild (venv + TORCH_PATH), then set runner APPEND e.g. `-overlap-v1.1` and `EXTRA_ARGS='--overlap-decode=yes'`.

## Return to v1.2
```bash
git checkout develop
git submodule update --init openfish   # should land on 223f26c
```
