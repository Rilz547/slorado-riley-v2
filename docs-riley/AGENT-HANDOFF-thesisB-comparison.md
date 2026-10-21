# Agent handoff (PC side): catch up and cross-check against thesisB-data-and-diagrams

**Owner:** Riley (thesis work, slorado on a Jetson Orin)
**Written:** 2026-10-21 by the Claude Code agent running on the Jetson
**Read this first, top to bottom.** You have no chat history. This file is the whole briefing.
**Companion file (older, still valid):** `docs-riley/AGENT-HANDOFF-power-campaign.md` — the power-campaign tooling and the
per-stage power results. This file does not repeat it; it points to it and adds everything that happened afterwards.

---

## 0. Your job in one paragraph

You are running on Riley's **main PC**, which has the folder `thesisB-data-and-diagrams` (older power-run charts, accuracy CSVs,
nsys stats, minimap2 results and diagrams). The Jetson-side agent has **no access** to that folder. Your job is to
(A) inventory what is in it, (B) cross-check the claims and numbers in this file against it and flag every mismatch, and
(C) say which comparison gaps it can fill. Then write a reply file (format in section 7) that Riley will paste back to the Jetson agent.

## 1. Rules

- Do **not** commit or push anything, in any repo. Riley commits.
- Treat every number here as a claim to verify, not as truth. The Jetson agent made at least one mistake in this session (section 5).
- If something in the PC data disagrees with this file, **report both values and their source file**; do not silently pick one.
- Do not delete or rewrite anything under `thesisB-data-and-diagrams`. Read-only unless Riley says otherwise.
- Prefer reading existing CSVs/charts over re-running anything. Only re-run minimap2 if a number cannot be recovered from existing outputs.
- No secrets are needed; do not ask Riley for credentials.

## 2. Environment map

| thing | where |
|---|---|
| Jetson (Orin, Nano-class, ~7.6 GB shared) | `riley@192.168.8.211` (LAN). Repo: `~/slorado-riley-v2`. Power mode `nvpmodel` 25W, `jetson_clocks` OFF |
| PC data | `~/Desktop/thesisB-data-and-diagrams/` (per the older handoff: `overlapping/power-runs/<stamp>/`, `overlapping/spec-decode/<stamp>/`, `overlapping/overlapv1.0/{accuracy,nsight-stats,runner-summaries}/`) — **verify; layout may differ** |
| PC tools | minimap2 at `/home/riley/Desktop/slorado_analysis/mm2-gb/minimap2`; reference `~/Desktop/slorado_analysis/ref/hg38noAlt.fa` |
| PC scripts | `fetch-and-minimap.py` (accuracy pass: pulls FASTQs + runner summaries from the Jetson, runs minimap2, writes `minimap_accuracy{APPEND}.csv`), `fetch-and-plot-power.py`, `fetch-and-minimap-spec.py` |

**Stale config you should fix in memory (not in files):** `fetch-and-minimap.py` has `JETSON_HOST = "192.168.4.36"`. The Jetson is now
**`192.168.8.211`**. Do not edit Riley's script without asking; just note it.

To fetch Jetson-side files to the PC (read-only pulls):
```bash
mkdir -p ~/jetson-pull && cd ~/jetson-pull
rsync -av riley@192.168.8.211:~/slorado-riley-v2/docs-riley/2026-10-21-optimisation-summary.md .
rsync -av riley@192.168.8.211:~/slorado-riley-v2/docs-riley/2026-10-05-hac-lstm-profile.md .
rsync -av riley@192.168.8.211:~/slorado-riley-v2/docs-riley/lstm-profile/ lstm-profile/
rsync -av "riley@192.168.8.211:~/slorado-riley-v2/riley-runner-*-spec-q10m2brave.*" .
rsync -av riley@192.168.8.211:~/slorado-riley-v2/docs-riley/power-runs/all_summaries.csv .
rsync -av riley@192.168.8.211:~/slorado/benchmark_runs/ jun23-upstream-baseline/   # ~330 MB, includes FASTQs
```
The Jetson requires an SSH login (password or key) — Riley will provide it; do not guess.

## 3. Current state of the code (as of 2026-10-21)

- slorado repo (`~/slorado-riley-v2`, GitHub `Rilz547/slorado-riley-v2`): branch **`speculative-decode` @ `49ac641`** ("Redid speculative-decode so it works."), pushed.
  Riley is now on a local branch `temp` (same commit plus staged docs).
- openfish (submodule, GitHub `Rilz547/openfish-riley`): branch **`speculative-greedy` @ `5109673`**, pushed. Based on the overlap-v1.2 commit `223f26c`.
- **What happened:** the original speculative-decode openfish code (greedy kernel `openfish_decode_gpu_greedy`, `greedy_search_cuda.h`, a `time_draft`
  stat, margin buffers) was never committed and was lost. It was **re-created** from the slorado call sites and the design in
  `2026-08-02-speculative-decode-report.md`. The re-created code reproduced the original report's accept rates (α 0.759 / 0.761 / 0.093),
  wall times and median identities to within noise on all four configs (section 4.2). It is functionally equivalent, not bit-identical to the lost version.
- ROCm is not supported by the new kernel (CUDA only). Riley does not care about ROCm.
- Quantisation: not available for the v5.0.0 LSTM models (int8 path exists only for v6 factored-LSTM/transformer on the `cpu-beam` lineage, and its fused kernel was removed).

## 4. Numbers to verify

### 4.1 Final timings — `rileys-runner.py`, speculative `q10_m2_brave`, 5 Oct 2026 (single run each)
Flags: `-C 128 -c 12288 -K 4096 -p 150 --overlap-decode=yes --fixed-c-batch=no --speculative-decode=yes --spec-repair-threshold=10 --spec-margin-threshold=2 --spec-overlap-repair=yes`, plus `--flush-threshold=64` (FAST) / `128` (HAC). Output order per config: FASTQ run (acts as warm-up), then one timed `/dev/null` run.

| config | timed run (s) | median identity (minimap2, PC) | map rate | reads |
|---|---|---|---|---|
| FAST 1k | 9.023 | 0.935276 | 103.2 % | 1000 |
| HAC 1k | 44.679 | 0.975490 | 110.5 % | 1000 |
| FAST 20k | 158.803 | 0.934853 | 103.6 % | 20000 |
| HAC 20k | 865.431 | 0.975521 | 111.3 % | 20000 |

Source on the Jetson: `riley-runner-timings-spec-q10m2brave.csv`, `riley-runner-output-spec-q10m2brave.txt`. Accuracy rows came from the PC script's summary table (`minimap_accuracy-spec-q10m2brave.csv` should exist in the PC `ACCURACY_DIR` if Riley saved it).

### 4.2 Original speculative-decode report (Aug 2026) — what the re-created code was checked against
From `2026-08-02-speculative-decode-report.md` (`q10_m2_brave`): FAST 1k 9.033 s / α 0.759 / median 0.935274; FAST 20k 157.768 s / α 0.730 / 0.934857;
HAC 1k 44.593 s / α 0.827 / 0.975490; HAC 20k 866.079 s / α 0.809 / 0.975516.
Baselines in that report (non-speculative, narrow batches): FAST 1k 9.827 s, median 0.940772; FAST 20k 174.065 s, 0.939375; HAC 1k 45.305 s, 0.976852; HAC 20k 871.297 s, 0.977333.
Re-created code, same configs, Jetson 1k tests on 5 Oct: α 0.759 (`q10_m2_brave`), 0.761 (`q10_m2` serial), 0.093 (`q10_m3_brave`); FAST 1k median 0.935276 vs baseline 0.940772.

### 4.3 Power-campaign chain (single runs, power-record + tegrastats; from the older handoff §5)
| stage (label, commit) | FAST 1k | HAC 1k | FAST 20k | HAC 20k |
|---|---|---|---|---|
| `pre_overlap` `8873209` | 12.97 s / 177 J | 48.56 / 802 | 217.1 / 2981 | 934.3 / 15845 |
| `ov_off` `7cec6c5` | 12.36 / 168 | 46.59 / 774 | 206.1 / 2829 | 902.5 / 15345 |
| `ov_on` | 10.23 / 152 | 45.11 / 779 | 174.2 / 2649 | 875.2 / 15328 |
| `lb_baseline` `65246f2` | 11.06 / 162 | 45.03 / 779 | 175.1 / 2667 | 867.6 / 15257 |
| `lb_best` | 9.78 / 143 | 44.69 / 767 | 173.8 / 2661 | 870.5 / 15262 |
| `best` (spec branch) | 9.78 / 142 | 45.82 / 780 | 174.2 / 2651 | 873.9 / 15307 |
| spec `q10_m2_brave` | 9.15 / 139 | 45.37 / 784 | 161.0 / 2501 | 872.3 / 15366 |
| `cb_off` (cpu-beam `664bb1f`) | 11.93 / 165 | 46.06 / 776 | 198.3 / 2765 | 893.7 / 15223 |
| `cb_on` | 12.21 / 181 | 54.50 / 882 | 219.2 / 3313 | 1059.5 / 17513 |

Note: speculative `q10_m2_brave` ran on the real (lost-then-recreated) code in August; the Oct runner numbers in 4.1 are a *re-run on the re-created code* and agree within ~1–2 %.
The older Jetson CSVs for earlier stages were deleted after transfer to the PC — **your PC data is the only copy**. Please confirm they match the table above.

### 4.4 June 23 2026 upstream baseline (unmodified Bonson `slorado` `main` @ `754a74d`), Jetson `~/slorado/benchmark_runs/`
| config | wall (s) | command differences |
|---|---|---|
| FAST 1k | 14.212 | `-t 8 -K 4096 -C 256 -B 512M -c 12288 -p 150`, writes a real FASTQ |
| HAC 1k | 49.805 | same but `-C 128` |
| FAST 20k | 247.872 | `-C 256` |
| HAC 20k | 955.152 | `-C 128` |
These are the closest thing to a "true original", but are **not** same-harness (FASTQ output, telemetry logging, FAST at `-C 256`, no warm-up). Also contain FASTQs, `console.log`, `run.json`, `telemetry.csv`.

### 4.5 Expected-median discrepancy in `fetch-and-minimap.py`
The script's built-in `EXPECTED_MEDIAN` is FAST `0.940696`, HAC `0.976852` (1k only). The Aug report's non-speculative narrow-batch baseline for FAST 1k is **0.940772**
(HAC 1k matches at 0.976852). The 0.000076 FAST difference makes the script's printed "delta" for speculative FAST 1k read −0.005420 instead of −0.005496 against the report baseline.
Please find out which run/commit produced 0.940696 and say which baseline each is for.

## 5. A correction you must not repeat (HAC inference profile)

Document: `docs-riley/2026-10-05-hac-lstm-profile.md` (+ raw data in `docs-riley/lstm-profile/`).
Valid, measured at the **production batch size N = 128**: a standalone cuDNN LSTM scaling benchmark (one layer at N=128, T=2048 takes 159.6 ms, matching slorado's 159 ms/layer-batch)
and an `nsys` kernel breakdown of a real HAC 1k run (recurrent GEMM 29.0 %, `elemWiseRNNcell` cell/gate kernel 21.7 % at 23.3 µs × 409 600 launches, input GEMM 14.6 % of GPU kernel time; LSTM ≈ 83 % of HAC wall time).
**Provisional / wrong scope:** two Nsight Compute passes were run on a 1-read file = **2 real chunks**, and slorado narrows the forward pass to the real chunks, so those kernels ran at **N = 2, not N = 128**
(the Jetson agent first mis-stated this as a 128-wide batch). Only the cell-kernel conclusion survives (its duration is ~22–23 µs at both N=2 and N=128). The GEMM-level ncu numbers (13 % tensor pipe, L1/L2 ≈ 58 %, 48 blocks)
describe a small-batch kernel variant and should not be cited. DRAM throughput cannot be measured with ncu on this Orin (`n/a`).

## 6. Summary of what was tried (details: `docs-riley/2026-10-21-optimisation-summary.md`)

- **Worked:** overlap decode (FAST −15 to −17 %, HAC ≈ −3 %); load imbalance / narrow batches (FAST 1k ≈ −12 %, ~0 elsewhere); speculative decode (FAST ≈ −7 to −10 % for ≈ −0.5 pp identity; HAC < 1 % for ≈ −0.15 pp).
- **Dead ends:** CPU beam hybrid (`cpu-beam`, slower everywhere), overlap depth-2, FAST→HAC cascade (negligible or worse), `cuda-lstm-gemm` / `flip-spike` (personal tests), speculative `m = 3`.
- **Ruled out:** larger batches (`-C 256` OOM on HAC), int8 quantisation for v5 LSTM.
- **Best remaining lead (unbuilt, unmeasured):** replacing the near-idle cuDNN cell/gate kernel — estimated ≈ −20 % of HAC wall.

## 7. Tasks and the reply format

### A. Inventory (do this first)
List what exists under `thesisB-data-and-diagrams`: stages/versions covered (pre-overlap, overlap v1.0/1.1/1.2, load-imbalance, speculative, cpu-beam, cascade, upstream/June), per stage what data types exist
(timing CSVs, runner summaries, power/energy CSVs, minimap accuracy CSVs, nsys stats, diagrams), file paths, dates, and the config/flags each used if recorded.

### B. Cross-check
For **each** row in sections 4.1–4.4, find the corresponding figure in the PC data and record: PC value, source file, match / near-match (< 2 %) / mismatch. Also check:
- the 4.3 table against the PC power CSVs (all stages, both wall time and energy);
- whether the PC holds baseline accuracy (median identity, map rate) for each stage and config, and whether 4.2's baselines (0.940772 / 0.939375 / 0.976852 / 0.977333) appear there;
- which baseline the value `0.940696` belongs to (section 4.5);
- anything in the PC data that contradicts the "worked / dead end" ordering in section 6.

### C. Gaps
List what the PC data **can** supply for a fair "original vs now" comparison (same harness for both ends, matched flags, matched accuracy) and what is still missing, e.g.:
same-harness pre-overlap or June-upstream timings with warm-up; accuracy for the non-speculative best config on HAC and on 20k in the same session as the speculative runs; energy for the Oct re-run.
Say plainly if the data cannot support a claim.

### Reply format (one markdown file, e.g. `REPLY-from-PC-agent.md`; Riley will paste it back)
1. **Inventory table** (stage × data types × path).
2. **Cross-check table** (claim | Jetson value | PC value | file | verdict).
3. **Mismatches and surprises** (numbered, most important first).
4. **Gaps** (what is missing for a fair original-vs-now comparison, and the cheapest way to fill each).
5. **Recommended next steps** for the report (max ~6 bullets). Do not implement them; Riley decides.
Keep it factual and concise. Include exact file paths so the Jetson agent can check a claim without access to your machine.

## 8. Where things live on the Jetson (for reference)

- `docs-riley/2026-10-21-optimisation-summary.md` — worked / dead ends / ruled out, plus pseudocode for overlap decode, load imbalance, speculative decode.
- `docs-riley/2026-10-05-hac-lstm-profile.md` and `docs-riley/lstm-profile/` — LSTM profiling (scripts, `kern.csv`, raw ncu output).
- `docs-riley/2026-08-02-speculative-decode-report.md`, `...-plan.md` — original speculative-decode results (the baseline for 4.2).
- `docs-riley/2026-07-22-cpu-beam-hybrid-report.md` — July timing report for the cpu-beam experiment (different implementation from the Aug power-run tip).
- `docs-riley/AGENT-HANDOFF-power-campaign.md` — power tooling and per-stage power numbers.
- `rileys-runner.py` — timing/accuracy runner; Oct run outputs are `riley-runner-*-spec-q10m2brave.*` in the repo root (gitignored).
