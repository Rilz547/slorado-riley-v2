# Bench checkout: Overlap v1.0 (2026-07-21)

## Current tree
- **Branch:** `bench-overlap-v1.0` (`ad973cf`)
- **Slorado algorithm:** `e00bada` (stream-aware infer∥decode overlap; still calls `openfish_decode_free_host` per batch)
- **Openfish:** `59e12e9` (stream-aware decode; per-batch pinned host alloc/free) — pinned in the bench commit
- **Runner:** current `rileys-runner.py` kept
- **`develop`:** v1.2 at `7cec6c5` + openfish `223f26c`
- **`bench-overlap-v1.1`:** still at `f27f772` + openfish `cf182a7`

## Before running
Rebuild, then e.g. `FILENAME_APPEND_FLAG="-overlap-v1.0"` / `"-baseline-v1.0"` with matching `EXTRA_ARGS`.
