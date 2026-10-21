# Agent handoff: Slorado Jetson optimizations + power campaign

**Owner:** Riley (Jetson Orin Nano / thesis work)  
**Repo:** `~/slorado-riley-v2`  
**Last updated:** 2026-10-05  
**Purpose:** Resume without prior chat history. Covers *what* was built, *how* it works, *what was measured*, and *how to operate* the tooling.

**Do not commit or push unless Riley explicitly asks.** Much of `docs-riley/` is personal tooling/results and is intentionally untracked.

---

## 1. One-paragraph summary

Riley is optimizing **slorado** (Nanopore basecaller using LibTorch + **openfish** CUDA decode) on a **Jetson Orin** (25W nvpmodel, `jetson_clocks` OFF). Work progressed through several decode/pipeline optimizations (overlap infer∥decode, load-imbalance narrow batches, speculative greedy→beam decode, a cpu-beam hybrid experiment). In Aug 2026 a **tegrastats power campaign** recorded wall time, peak/mean power, and energy for A/B configs across FAST/HAC × 1k/20k. Results were pulled to a main PC for charts. **Current git branch:** `cpu-beam` @ `664bb1f`. **Current on-disk power results:** full cpu-beam A/B under `docs-riley/power-runs/` (keep these).

---

## 2. Environment & conventions

### Machine / fairness (power + timing)

| Knob | Value / rule |
|------|----------------|
| Board | Jetson Orin (Nano-class, ~8 GB) |
| Power mode | `nvpmodel` **25W** — keep fixed across A/Bs |
| Clocks | **`jetson_clocks` OFF** for all runtime/power work |
| Output | Prefer `-o /dev/null` for timed/power runs (no FASTQ I/O) |
| Concurrency | One heavy job at a time |
| Wall metric | Prefer slorado **`[main] Real time`** (excludes process/Torch startup), not shell wall |

### Build (Riley usually runs this himself — don’t auto-make unless asked)

```bash
source ~/slorado_venv/bin/activate   # path may be ~/slorado_venv or similar
export TORCH_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch
# from repo root:
make clean
make -j6 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH
```

See also `docs-riley/howto.md`.

### Models & data

- Models: `models/dna_r10.4.1_e8.2_400bps_{fast,hac}@v5.0.0`
- Reads: `test/PGXXXX230339/reads_{1k,20k}.blow5` (`reads_20k.blow5` may be untracked/large)

### Shared “campaign” batch flags

Unless a stage predates a flag, timed/power runs use:

```text
-C 128 -c 12288 -K 4096 -p 150
```

Plus stage-specific flags documented below.

### Important branches / commits

| Ref | Role |
|-----|------|
| `8873209` | Pre-overlap (“colour outputs”); no `--overlap-decode` |
| `7cec6c5` / `develop` | **Overlap v1.2** |
| `574db27` | Load imbalance (narrow / `--flush-threshold` / `--fixed-c-batch`) |
| `65246f2` / `load-imbalance` | Adds `docs-riley/best` runner |
| `40f3b9e` / `speculative-decode` | Speculative decode WIP + power-stats commit note |
| `664bb1f` / `cpu-beam` | CPU beam hybrid experiment (**current HEAD**) |

Openfish is a **submodule**. After every checkout: `git submodule update --init --force`. Speculative greedy path lived in openfish (`greedy_search_cuda.h`, `openfish_decode_gpu_greedy`); cpu-beam openfish tip differs.

---

## 3. Optimization stack — inner workings

Story order (oldest → newest on the main Riley line): **pre-overlap → overlap v1.x → load imbalance → speculative decode**. **cpu-beam** is a side branch / throwaway experiment (not stacked on speculative).

### 3.1 Overlap decode (v1.0 → v1.2)

**Problem:** Serial pipeline does infer(batch) then decode(batch). On FAST, decode is expensive; GPU idle during parts of that.

**Idea:** Run **GPU infer of batch N** concurrent with **GPU decode of batch N−1** (depth-1 ping-pong on separate CUDA streams).

| Version | Commit(s) | What changed |
|---------|-----------|--------------|
| **v1.0** | `e00bada` era | Infer∥decode overlap; CLI `--overlap-decode=yes\|no` (default **no**). Needed correct NTC score layout for `-C > 1`. |
| **v1.1** | slorado `9f8db24`, openfish `cf182a7` | Persist pinned host decode buffers — stop per-batch `cudaHostAlloc`/`cudaFreeHost` (nsys FreeHost hotspot). |
| **v1.2** | `7cec6c5` | Still one GPU decode at a time; **2-slot pinned host ring**; after launching decode(curr), CPU `write_decode_results(prev)` overlaps under decode(curr). |

**Bound:** Big FAST wall win; HAC smaller (more infer-bound).

**Power A/B labels:** `ov_off` = `--overlap-decode=no`, `ov_on` = `--overlap-decode=yes` (on commit `7cec6c5`; no fixed-c/flush yet).

Deeper notes: `docs-riley/context.md`.

### 3.2 Load imbalance (narrow GPU batches)

**Problem:** Streaming / partial flushes still launched **full `-C`-wide** GPU batches padded with dummy chunks → wasted compute on partials.

**Idea (commit `574db27`):**

- `--fixed-c-batch=no` (default when using “best”): launch only **N real chunks** (narrow).
- `--fixed-c-batch=yes`: old padded full-C behaviour (baseline for A/B).
- `--flush-threshold N`: flush a GPU batch once ≥ N chunks queued (FAST best uses **64**, HAC **128**); `0` ⇒ pack full C.

**“Best” runner:** `docs-riley/best` (on `load-imbalance`):

```bash
./slorado basecaller -C 128 -c 12288 -K 4096 -p 150 \
  --overlap-decode=yes --fixed-c-batch=no --flush-threshold={64|128} \
  -o /dev/null MODEL READS
```

**Power A/B:** `lb_baseline` (overlap on + `--fixed-c-batch=yes`) vs `lb_best` (overlap on + narrow + flush). Measured effect: clear on **FAST 1k**; ~flat on 20k/HAC.

### 3.3 Speculative decode

**Branch:** `speculative-decode` (from `load-imbalance`).  
**Full write-up:** `docs-riley/2026-08-02-speculative-decode-report.md`  
**Short index:** `docs-riley/2026-08-02-speculative-decode-plan.md`

**Motivation:** On best FAST path, wall is dominated by **CRF beam (width 32)**, not LSTM. Scores already on GPU after infer.

**Pipeline:**

```text
infer → greedy draft (all chunks) → Q/margin gate → full beam on hard subset only → stitch
optional brave: repair(k) ‖ infer(k+1)
```

| Piece | Implementation |
|-------|------------------|
| Draft | `openfish_decode_gpu_greedy` (width-1); margins for gate |
| Gate | Repair if mean draft Phred Q < `--spec-repair-threshold` **or** mean decision margin < `--spec-margin-threshold` |
| Repair | Full `openfish_decode_gpu` beam on hard subset |
| Brave | `--spec-overlap-repair=yes` |

**Headline config (`q10_m2_brave`):**

```text
--speculative-decode=yes --spec-repair-threshold=10 \
--spec-margin-threshold=2 --spec-overlap-repair=yes
```

Plus the **best** load-imb flags as baseline.

**Timing/accuracy (from report, not power):**

| Model | 20k wall vs best baseline | Δ median identity |
|-------|---------------------------|-------------------|
| FAST | ~9–10% faster | ~−0.45 to −0.55 pp |
| HAC | ~0.6% faster | ~−0.15 pp |

HAC is **infer-bound** → decode savings barely move wall. Phase-0: greedy≠beam chunk identity (~0%); accept is approximate (Q+margin), not bit-identical.

**Key code:**

- `openfish/src/greedy_search_cuda.h`, `openfish/src/decode_cuda.c`
- `src/basecall.cpp` (gate/repair/brave), CLI in `src/basecaller_main.cpp` / `src/slorado.h`

**Harness:**

- Jetson: `docs-riley/spec-fastq-sweep` / `spec-sweep` — time (`/dev/null`) then keep FASTQ
- PC: `docs-riley/fetch-and-minimap-spec.py` → `overlapping/spec-decode/<stamp>/`

### 3.4 CPU beam hybrid (`cpu-beam` branch)

**Not** full CPU decode. Only **beam search** on host; scan/qual/gen stay on GPU.

**CLI:** `--cpu-beam=yes|no` (yes implies `--stream`). Needs GPU build; do **not** use `-x cpu`.

Overhead: sync after bwd → D2H scores/bwd → threaded CPU beam → H2D states/moves → continue GPU path.

**Timing report (Jul 2026, on develop@overlap v1.2 throwaway):** ~**2.5–3.3×** slower — see `docs-riley/2026-07-22-cpu-beam-hybrid-report.md`. Verdict: not worthwhile for FAST/HAC on this Jetson.

**Power campaign (Aug 2026, branch `cpu-beam` @ `664bb1f`):** milder wall deltas than that report (see §5). Treat the **July timing report** as the wall-clock story for the experiment as originally measured; treat **Aug power-runs** as the energy capture on the `cpu-beam` tip. If reconciling, re-check openfish pin / flags / whether beam actually ran on CPU (`slorado.log`).

---

## 4. Power campaign — tooling

### 4.1 Files (mostly untracked under `docs-riley/`)

| File | Role |
|------|------|
| `power-record` | One run: start tegrastats @ 1 Hz → basecall → parse → `samples.csv` + `summary.csv`; append `all_summaries.csv` on success |
| `power-sweep` | Matrix driver: `--pre` / `--overlap` / `--loadimb` / `--cpu-beam` (+ `--only fast_1k,...`) |
| `parse_tegrastats.py` | Log → CSV; **total power = `VDD_IN`** (not sum of rails — avoids double-count on this Orin) |
| `power-record.md` | Short usage |
| `fetch-and-plot-power.py` | **Main PC:** rsync `power-runs/` → new folder + matplotlib charts (CPU, GPU, RAM/swap, power; summary bars) |
| `power-runs/` | Per-run dirs + rolling `all_summaries.csv` |

Energy ≈ Σ `VDD_IN_mW` × 1 s → joules. Failed runs (e.g. HAC 20k OOM) go to `_failed_*` and are **not** appended to `all_summaries.csv`. `power-sweep` sleeps ~8s between cells to reduce NvMap/CUDA pressure.

### 4.2 Sweep modes ↔ git checkout

| Mode | Checkout | What each cell runs |
|------|----------|---------------------|
| `--pre` | `8873209` | label `pre_overlap`: `-C/-c/-K/-p` only |
| `--overlap` | `7cec6c5` | `ov_off` then `ov_on` |
| `--loadimb` | `load-imbalance` / `65246f2` | `lb_baseline` then `lb_best` |
| (spec, manual) | `speculative-decode` | e.g. label `best_fast` then `spec_q10_m2_brave` + spec flags after `--` |
| `--cpu-beam` | `cpu-beam` / `664bb1f` | `cb_off` then `cb_on` |

**Always rebuild after checkout** so the binary matches the tree (stale binaries can expose flags that source doesn’t have, or vice versa).

### 4.3 PC pull + charts

```bash
# copy fetch-and-plot-power.py to analysis tree; edit CONFIG (host/user/paths)
python3 -m venv .venv-power && source .venv-power/bin/activate
pip install matplotlib
python fetch-and-plot-power.py
```

Writes `overlapping/power-runs/<stamp>/` with `raw/` + `charts/`.

### 4.4 Typical A/B workflow Riley used

1. Transfer/keep prior results on PC if needed.  
2. `rm -rf docs-riley/power-runs` (when starting a new stage).  
3. `git checkout <commit/branch> && git submodule update --init --force`.  
4. Rebuild.  
5. `./docs-riley/power-sweep --<mode>`.  
6. PC: `fetch-and-plot-power.py`.  
7. Agent updates harness for next stage’s flags.

---

## 5. Power results archive (numbers)

Earlier stage CSVs were **deleted on Jetson after PC transfer**. Numbers below are from that campaign (chat + docs). **cpu-beam** numbers are still on disk in `docs-riley/power-runs/all_summaries.csv`.

### 5.1 Pre-overlap (`8873209`, label `pre_overlap`)

| cell | wall_s | peak_W | energy_J |
|------|--------|--------|----------|
| FAST 1k | 12.97 | 14.4 | 177 |
| HAC 1k | 48.56 | 17.5 | 802 |
| FAST 20k | 217.1 | 14.6 | 2981 |
| HAC 20k | 934.3 | 18.2 | 15845 |

### 5.2 Overlap v1.2 (`7cec6c5`)

| cell | ov_off wall / E | ov_on wall / E | rough Δ wall |
|------|-----------------|----------------|--------------|
| FAST 1k | 12.36 / 168 | 10.23 / 152 | **~−17%** |
| HAC 1k | 46.59 / 774 | 45.11 / 779 | ~−3% |
| FAST 20k | 206.1 / 2829 | 174.2 / 2649 | **~−15%** |
| HAC 20k | 902.5 / 15345 | 875.2 / 15328 | ~−3% |

### 5.3 Load imbalance (`65246f2`)

| cell | lb_baseline wall / E | lb_best wall / E | rough Δ wall |
|------|----------------------|------------------|--------------|
| FAST 1k | 11.06 / 162 | 9.78 / 143 | **~−12%** |
| HAC 1k | 45.03 / 779 | 44.69 / 767 | ~−1% |
| FAST 20k | 175.1 / 2667 | 173.8 / 2661 | ~0 |
| HAC 20k | 867.6 / 15257 | 870.5 / 15262 | ~0 |

### 5.4 Speculative (`speculative-decode`, vs same-commit “best” flags)

| cell | best wall / E | q10_m2_brave wall / E | rough Δ wall |
|------|---------------|------------------------|--------------|
| FAST 1k | 9.78 / 142 | 9.15 / 139 | **~−6%** |
| HAC 1k | 45.82 / 780 | 45.37 / 784 | ~−1% |
| FAST 20k | 174.2 / 2651 | 161.0 / 2501 | **~−7%** |
| HAC 20k | 873.9 / 15307 | 872.3 / 15366 | ~0 |

### 5.5 cpu-beam (on disk now @ `664bb1f`)

| cell | cb_off wall / E | cb_on wall / E | Δ wall (this capture) |
|------|-----------------|----------------|------------------------|
| FAST 1k | 11.93 / 165 | 12.21 / 181 | +2% |
| HAC 1k | 46.06 / 776 | 54.50 / 882 | +18% |
| FAST 20k | 198.3 / 2765 | 219.2 / 3313 | +11% |
| HAC 20k | 893.7 / 15223 | 1059.5 / 17513 | +19% |

July timing report showed ~2.5–3× for `cb_on`; this power capture is milder — see §3.4 caveat.

### 5.6 Cascade energy takeaway (presentation-friendly)

Comparing **ov_off** pack vs **+spec** pack (sum of all 4 cells’ energy):

- Dominated by **HAC 20k** → total energy only ~**−2%**.  
- **FAST-only** (1k+20k energy): ~**−12%**.  
- Wins are mostly **shorter wall → less energy**, not lower peak (peaks stay ~14–19 W).

Stack narrative for FAST wall: overlap helps a lot → load-imb helps short FAST → spec adds another ~7–10% on FAST; HAC stays near-flat after overlap.

---

## 6. Untracked / working-tree clutter (what it’s for)

| Path | Purpose |
|------|---------|
| `power-record`, `power-sweep`, `parse_tegrastats.py`, `power-record.md` | Power harness |
| `fetch-and-plot-power.py` | PC charts |
| `power-runs/` | **Keep** — current cpu-beam results |
| `2026-08-02-speculative-decode-*.md` | Spec report/plan |
| `spec-fastq-sweep`, `spec-sweep`, `spec-decode-sweep/` | Spec timing/accuracy sweeps |
| `fetch-and-minimap-spec.py` | PC minimap for spec FASTQs |
| `context.md` | Older overlap v1.x agent notes |
| `howto.md` | Riley Jetson how-to (build/swap/etc.) |
| `__pycache__/` | Safe to delete |
| `test/.../reads_20k.blow5` | Large dataset; usually not committed |

---

## 7. Related but out of scope of the Aug power loop

- `rileys-runner.py` — multi-run timing suite with hang recovery / resume (see `context.md`).  
- Cascade / overlap-depth2 / flip-spike branches — separate experiments.  
- Desktop analysis often under something like `~/Desktop/thesisB-data-and-diagrams/` with `overlapping/power-runs/` and `overlapping/spec-decode/`.

---

## 8. Rules of engagement for agents

1. **Ask before** long 20k suites, commits, pushes, hard resets, or deleting `power-runs/` if Riley wants to keep them.  
2. Prefer editing **`docs-riley/` harness** over core unless implementing a feature.  
3. After checkout: submodule update + rebuild reminder.  
4. Power total rail: **`VDD_IN`**.  
5. Don’t treat `parse_tegrastats.py` as the orchestrator — that’s `power-record` / `power-sweep`.  
6. Speculative decode is **WIP** on `speculative-decode`; feature code may not be present on `cpu-beam` / older checkouts.

---

## 9. Quick command cheat sheet

```bash
# status
git status -sb; git log -1 --oneline; git submodule status

# power (after correct checkout + rebuild)
./docs-riley/power-sweep --pre
./docs-riley/power-sweep --overlap
./docs-riley/power-sweep --loadimb
./docs-riley/power-sweep --cpu-beam
./docs-riley/power-sweep --cpu-beam --only fast_1k,hac_1k   # smoke

# single cell examples
./docs-riley/power-record --pre --fast 1k
./docs-riley/power-record --ov on --hac 20k
./docs-riley/power-record --lb best --fast 1k
./docs-riley/power-record --cb on --fast 1k
./docs-riley/power-record --label spec_q10_m2_brave --fast 1k -- \
  --speculative-decode=yes --spec-repair-threshold=10 \
  --spec-margin-threshold=2 --spec-overlap-repair=yes
# (only on speculative-decode tree with best flags baked into power-record --lb best / label mode)

# PC
python fetch-and-plot-power.py
python fetch-and-minimap-spec.py --stamp <STAMP>
```

---

## 10. Suggested reading order for a new agent

1. This file  
2. `docs-riley/power-record.md` + skim `power-record` / `power-sweep`  
3. `docs-riley/2026-08-02-speculative-decode-report.md`  
4. `docs-riley/context.md` (overlap versions)  
5. `docs-riley/2026-07-22-cpu-beam-hybrid-report.md`  
6. `docs-riley/howto.md`  
7. On-disk `docs-riley/power-runs/all_summaries.csv` for latest cpu-beam numbers  
