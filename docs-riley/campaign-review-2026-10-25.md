# Campaign review (2026-10-25): completeness, failures, open decisions

Campaign ran 2026-10-23 13:40 → 2026-10-25 01:01 (≈ 35 h) plus fix-ups 2026-10-25 12:30 → 15:31. Harness stamp `d5704e0ec3e6` throughout. Data: `bench-runs/runs.jsonl` (380 timed runs ok), `bench-runs/rerun-ov_on-fast20k.jsonl` (replicate block), `bench-runs/campaign.log`, `campaign-fixups.log`. Tables: `analyze.py tables` → `bench-runs/tables.md`.

## Completeness
380 of 387 planned timed runs recorded. Missing: `spec` DNA FAST 500k (1 run) and `cb_on` DNA HAC 20k (6 runs, modes A and B).

## Failures, by cause
| what | runs | evidence | status |
|---|---|---|---|
| `cb_on` HAC 20k: killed by the 900 s stall limit | 18 attempts, all end at 901.5 s | CPU-beam pipeline prints no progress; a HAC 20k run needs ≈ 53 min | **harness limitation, not a hang.** Left unmeasured (dead end; FAST 1k/20k and HAC 1k already show it is slower). Raising the limit would change a stamped file |
| GPU hangs (stall or wall_cap) | `lb_baseline` FAST 20k, `cb_off` FAST 20k, `ov_on` FAST 20k (fix-up block), `spec` FAST 500k ×4 | all show the same signature before the kill: power ≈ 6.0 W (from ≈ 13–16 W), CPU ≈ 2 %, **GPU load reads 99.9 %**, no output; the three 20k runs succeeded on retry | **unexplained; all 7 occurred in overlap-decode-on stages and none in 4.6 h of overlap-off FAST running (`upstream`, `ov_off`, `paper_xavier`); none on HAC.** Hypothesis (untested): a rare deadlock in the overlap-decode / spec-repair stream code. Cost ≈ 16 min per 20k hang |
| `spec` HAC 20k crash (SIGABRT) | 1 attempt | console starts with `NvMapMemAllocInternalTagged ... error 12` (device memory allocation failure), retry succeeded | same family as the August NvMap OOM on HAC 20k spec |
| `spec` FAST 500k | **4 of 4 attempts hung** at 36, 75, 98 and 99 of ≈ 123 batches (≈ 20 %, 61 %, 80 %, 80 % of the dataset) | RAM rose from ≈ 5.9 GB to 6.6–7.4 GB of 7.6 GB (upstream at 500k: 5.7 → 6.5 GB) | **no `spec` 500k result.** Report as "did not complete (4/4 hangs)" or run a reduced-memory variant (see decisions) |

## Reduced-`-K` attempt for `spec` 500k (2026-10-25 16:17, `reduced_k_500k.py`, `-K 2000`, one attempt)
**Failed: hung again** after 146 of ≈ 250 batches (≈ 292k reads, 2,327 s into the run). Same signature (power ≈ 6.0 W, CPU 0 %, GPU load 99.9 %, no output). **RAM stayed at 5.5 → 5.9 GB of 7.6 GB, so the memory hypothesis is refuted.** `upstream` at `-K 2000` was skipped. Record: `bench-runs/reducedK-500k.jsonl`.
All five `spec` 500k attempts hung, after 1,164, 2,430, 3,188, 3,221 and 2,327 s (≈ 3.4 h of run time before 5 hangs; 36, 75, 98, 99 and 146 batches), at different data positions. That looks like a time-driven hazard of roughly one hang per 40–50 min of `spec` FAST running (a 20k run, ≈ 2.7 min, hangs with ≈ 6 % probability; a 500k run needs ≈ 66 min, success chance ≈ 20 %), not a data- or memory-dependent fault. Estimate only.

## Hang captured under gdb (2026-10-25 23:02 – 2026-10-26 00:43, `hang_hunt.py`, `spec` FAST 500k, campaign flags, `-o /dev/null`)
Files: `bench-runs/hang-hunt/20261025T230242-spec-dna_fast_500k-{1,2}/`.
- **Run 1 completed:** `[main] Real time: 3990.994 sec`, 123 batches, accept rate α = 0.727, peak RAM 5.466 GB. Run under gdb with no power window, so it is **not** a measurement; indicative only (about −37 % against `upstream`'s 6,331 s at the same `-K`, in line with −37.5 % at 20k).
- **Run 2 hung** after 56 batches (2,009 s). At the hang: VDD_IN 6.08 W, GPU load node 999 ‰, 12 threads idle on futexes, and **one thread blocked in `cudaStreamSynchronize` called from `overlap_sync_repair_and_stitch` (`src/basecall.cpp:874`, via `basecall_chunks_overlap`, `pthread_single_basecall`)**, itself waiting in an `ioctl` to the Tegra host1x driver (`libnvrm_host1x.so`; kernel wait `dma_fence_default_wait`). So the host is waiting on the decode/repair stream's GPU fence and it never signals; the GPU is not doing work (6 W) but reports full load. `free -m` at the hang: 7,133 MB used, 236 MB available, swap 2,168 MB (the reduced-`-K` hang had only 5.9 GB used, so memory pressure is not required).
- **Candidate cause (hypothesis, untested):** a cross-stream lifetime race. Scores are produced on `infer_stream` (caching-allocator blocks belong to that stream) and consumed on `decode_stream` (`overlap_launch_decode` waits on `infer_event[slot]`, which orders the read after the write), but there is no `record_stream` anywhere in `basecall.cpp`. In the repair path, `hold->scores_NTC = hard_scores;` (≈ line 856) drops the original scores tensor right after queueing `index_select` on `decode_stream`; the allocator may hand that block to the next inference on `infer_stream` while the queued read has not run. A corrupted read on Tegra could fault the GPU channel (hang at idle power) instead of raising an error. The hangs in `ov_on`, `lb_baseline` and `cb_off` (no repair) would need the same rule applied to `pending->scores_NTC`. Cheap tests: keep the original tensor alive until the repair sync, and call `record_stream(decode_stream)` on scores consumed by the decode stream; then repeat `hang_hunt.py` (a 3-hour run without a hang would be ≈ 4 expected hangs avoided).

## `ov_on` DNA FAST 20k (outlier cell)
Campaign block: mode A 174.06, 173.60, **193.18** (+11 %); mode B **198.16** (+13 %), 174.83, 175.30. Replicate block (same stamp, canary 160.1 vs 160.0 ms before): mode A 177.71, 176.20, 175.72; mode B 176.24, 176.54, 176.85 (one hang, retried). Means: campaign without the two flagged runs A 173.83 / B 175.06; replicate A 176.54 / B 176.54 (+1.6 % / +0.8 %, inside the 2 % block-to-block tolerance). Effect on the chain (mode A): `ov_off`→`ov_on` −14.5 % (campaign, outliers excluded) or −13.2 % (replicate), versus −11.4 % from the contaminated mean; `ov_on`→`lb_baseline` +0.6 % or −0.9 % (versus −3.0 %).

## Other validation exceptions on the campaign data (informational, none affects `Real time` or the energy window)
- V3 time CV > 1 %: `ov_on` FAST 20k A and B (the outliers above), `upstream` HAC 1k B (+3.4 % on rep 1, coincident with `gnome-system-monitor` being launched at 13:53 on 2026-10-23), `spec` FAST 20k B (1.05 %).
- V4 exit gap > 3 s: 5 HAC 20k runs (max 29 s, `ov_on` A rep 2); the upper bound was calibrated on 1k runs.
- V4 shoulder: one run +0.79 W (`lb_best` RNA HAC 20k B rep 3).
- V1 has no qualifying runs (no tegrastats in the campaign; V1 passed in validation).
- Passed: canary drift, idle drift, A/B order, start-epoch spread, anchor cross-check, rail coverage; V7 output equivalence passes for every stage.

## Finding for the report: `upstream` → `ov_off` is −17.3 % (FAST 1k) / −19.6 % (FAST 20k) with overlap switched off
Larger than overlap's own step (−11.3 % / −11.4 %). The commits between `754a74d` and `7cec6c5` include pointing openfish at the fork (`b9569c6`), stream-aware decode, and "stop freeing openfish host decode buffers after each GPU batch" (v1.1). Not attributable to overlap; the split between those changes is unmeasured.

## Decision made (Riley, 2026-10-26): outlier handling
Runs more than 3 % above their (stage, cell, mode) median are excluded from `runs.csv` and every table (`analyze.py outliers --write` → `bench-runs/excluded-runs.json`, listed in each table's footnote; `validate` and `equiv` still see every run). The rule found exactly three runs: `ov_on` DNA FAST 20k mode A rep 3 (+11.0 %) and mode B rep 1 (+13.0 %), and `upstream` DNA HAC 1k mode B rep 1 (+3.4 %, coincident with `gnome-system-monitor`). The third was excluded because the same rule applies to every cell; delete its line in `excluded-runs.json` to undo. The replicate block stays as an independent check, not averaged in. Effect: `ov_on` FAST 20k = 173.83 s (n = 2), `ov_off`→`ov_on` −14.5 %, `ov_on`→`lb_baseline` +0.6 %.

## Decisions pending (Riley)
1. `ov_on` FAST 20k headline: campaign block with the two flagged runs excluded (pre-stated rule: runs > 3 % above the cell median are flagged), with the replicate block as an independent check. Recommended.
2. `spec` 500k: report as not completed, or spend ≈ 3 h on a reduced-memory comparison (`spec` and `upstream` at the same smaller `-K`, different flags from the other 500k rows).
3. Whether to investigate the hang (a patched `pre_overlap` stage for the upstream→`ov_off` question is a separate, optional item).
