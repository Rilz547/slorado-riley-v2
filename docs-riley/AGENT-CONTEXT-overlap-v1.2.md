# Agent context: overlap decode v1.0 → v1.2 (Jetson Orin)

> Personal handoff note for future Cursor chats. Not part of the upstream project story unless Riley commits it.
> Last updated: 2026-07-21

## Repo layout

| Path | Branch / note |
|------|----------------|
| `~/slorado-riley-v2` | `develop` — slorado |
| `openfish/` submodule | `Rilz547/openfish-riley`, branch `decode-profiling` |
| Riley notes / tooling | `docs-riley/`, `rileys-runner.py` (often uncommitted) |

**Riley builds/runs himself** — agents should not auto-`make` or kick long benches unless asked.

Build (from repo root, venv + TORCH_PATH set):

```bash
source slorado_venv/bin/activate
export TORCH_PATH=/home/riley/slorado_venv/lib/python3.10/site-packages/torch
make clean; make -j6 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH
```

## Optimization versions (what “v1.x” means)

### v1.0 — Infer∥decode overlap
- Depth-1 ping-pong: decode(N−1) on `decode_stream` while infer(N) on `infer_stream`.
- CLI: `--overlap-decode=yes|no`.
- Critical fix: NTC score layout for openfish (needed for `-C > 1`).
- Big FAST wall win; HAC smaller (infer-bound).

### v1.1 — Persistent pinned host buffers
- Commits: slorado `9f8db24`, openfish `cf182a7` (tagged message “Overlap v1.1”).
- Stop per-batch `cudaHostAlloc` / `cudaFreeHost` of moves/seq/qstring.
- Alloc once on `gpubuf`, free in `openfish_gpubuf_free`.
- Killed FreeHost hotspot; much tighter run-to-run timing under nsys.

### v1.2 — P5-lite (current uncommitted work as of this note)
- **Still one GPU decode at a time** (shared device scratch).
- **Two pinned host slots** (`OPENFISH_HOST_RING=2`).
- `openfish_decode_gpu(..., stream, host_slot)`.
- Slorado overlap schedule:
  1. sync decode(prev)
  2. launch decode(curr) into the *other* host slot
  3. CPU `write_decode_results(prev)` while curr decode runs
- Expected: modest extra wall (~1–4% FAST, ~0–2% HAC) on top of v1.1.
- HIP/Metal stubs updated for new `host_slot` API (serial path uses slot 0).

### Explored but not kept / deferred
- Int8 scores: implemented then **reverted** (slower).
- TensorRT FP16: high effort, low expected upside — deferred.
- True flip removal: needs Dorado/koi-style reverse LSTM — not a small patch.
- Full depth-2 dual-decode / P2 sync gating: deferred.

## Benchmarking plan (next)

1. **Done:** v1.2 full suite with `rileys-runner.py`:
   - overlap: `FILENAME_APPEND_FLAG="-overlap-v1.2"`, `EXTRA_ARGS="--overlap-decode=yes"`
   - baseline: `FILENAME_APPEND_FLAG="-baseline-v1.2"`, `EXTRA_ARGS=""`
2. **Next:** Keep `rileys-runner.py` (and its knobs/timeouts). Revert **code** (slorado + openfish submodule) to **v1.1** commits and re-bench overlap (and optionally baseline) with a new APPEND e.g. `-overlap-v1.1`.
3. Compare timed `/dev/null` stats (not nsys/accuracy pass) across APPEND labels.
4. Accuracy: FASTQs + matching nsys from accuracy pass; Riley’s desktop `fetch-and-minimap.py` — update naming for APPEND suffixes; no baseline↔overlap pairing in that tool.

### Known good v1.1 commit pins
- slorado: `9f8db24` — `(Overlap v1.1) Stop freeing openfish host decode buffers...`
- openfish: `cf182a7` — `(Overlap v1.1) Persist pinned host decode buffers...`

## `rileys-runner.py` (keep across code reverts)

Edit knobs at top of file. Important behaviours:

| Behaviour | Detail |
|-----------|--------|
| Accuracy pass | If `NSIGHT=yes`: one nsys+FASTQ run **excluded from stats**. Then `NUM_RUNS_*` timed `/dev/null` runs feed mean/stdev/CV%. |
| Warmup | Per-config; uncounted; `/dev/null`. HAC 20k warmup default **yes**. |
| Progress | Parses `[basecaller_main::…]` lines. |
| Hang recovery | Stall timeout (default 600s silence) **and** per-config wall budget; kill PG → `killall -9 slorado nsys` → nvidia-smi note → retry once → mark failed → continue. |
| Wall budgets | From `EXPECTED_TIMED_WALL_S` × `WALL_MARGIN` (1.25); nsys accuracy × `NSYS_WALL_FACTOR` (2.0). FAST 20k base 350s; HAC 20k base 1200s. |
| Resume | `python3 rileys-runner.py --resume` using `riley-runner-state{APPEND}.json`. |
| Lock | `riley-runner.lock` — one suite at a time. |
| Flush | Report + CSV + state rewritten after every run. |

### Outputs (per APPEND)

- `riley-runner-output{APPEND}.txt` — human report
- `riley-runner-timings{APPEND}.csv` — `config,phase,run,real_s,cpu_s,peak_ram_gb,status,attempts,nsight,fastq,note`
- `riley-runner-{fast,hac}-console{APPEND}.txt` — full console if `RECORD_CONSOLE=yes`
- `riley-runner-state{APPEND}.json` — resume state
- FASTQ: `output_{fast\|hac}_{1k\|20k}[_overlap]{APPEND}.fastq` (accuracy pass only)
- nsys: `nsys_{fast\|hac}_{1k\|20k}_{overlap\|base}{APPEND}.nsys-rep`

**Tip:** prefer APPEND like `-v1.2` or `-baseline-v1.2`; mode tags already add `_overlap` / `_base` where needed. Using `-overlap-v1.2` with overlap mode yields `..._overlap-overlap-v1.2...` (works, just ugly).

### Recorded v1.2 data on disk (as of 2026-07-21)

- `riley-runner-output-baseline-v1.2.txt` + `riley-runner-timings-baseline-v1.2.csv`
- `riley-runner-output-overlap-v1.2.txt` + `riley-runner-timings-overlap-v1.2.csv`
- Matching state JSON files; FASTQs/nsys as named in those reports.

## Report / accuracy pipeline notes

- Stats for jitter/speedup claims: **timed phase rows only** (`phase=timed`, `status=ok`) from CSV or the “Summary (ok timed runs only…)” blocks.
- Accuracy/nsys pass is listed but **must not** enter mean/stdev/CV%.
- With n=3 on 20k, prefer mean + range; lead variance claims with 1k n=10.
- Desktop accuracy tool should map FASTQ↔nsys by APPEND + mode tags; compare experiments offline.

## Jetson hygiene (optional)

- Swapfile + `vm.swappiness=10`; disable `nvzramconfig` if fighting swap.
- After a hung CUDA context, reboot (or `sudo nvidia-smi --gpu-reset` if available) before `--resume`.

## What not to do unless asked

- Do not commit secrets / huge FASTQs / nsys dumps unless Riley asks.
- Do not force-push or amend without explicit request.
- Do not auto-run full 20k suites.
