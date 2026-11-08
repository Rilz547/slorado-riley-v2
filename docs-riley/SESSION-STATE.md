# Session state — where the final-campaign work stands (2026-10-23, updated 2026-11-08)

Purpose: let a fresh Claude session (or Riley, or the PC agent) pick this up without the original conversation. Read this first, then `HARNESS-FROZEN.md`.

## Goal
Thesis on slorado (Nanopore basecaller) optimisation on a Jetson Orin Nano (25 W, `jetson_clocks` off). Redo all timing, power, profiling and accuracy tests on one trusted, frozen harness, then write the results sections (`report-results-outline.md`).

## Done
- `speculative-decode` branch rebuilt (greedy kernel re-created in openfish `5109673`, slorado `49ac641`), validated against the August report.
- Harness in `docs-riley/bench/` (own INA3221 power logger, per-rail and net energy, H1 `/dev/null` and H2 FASTQ modes, idle baseline, canary, window recovered from slorado's progress lines). Stage worktrees in `~/bench-stages/`. Datasets in `~/data/` (DNA 1k/20k/500k, RNA 1k/20k, GENCODE v40).
- Findings: `pre_overlap` (`8873209`) is INVALID (score-layout bug, fixed in v1.0 `e00bada`); August "original → now" chains built on it are withdrawn. RNA speculative decode: Q gate dominates, no setting both faster and accurate on RNA FAST (`spec-threshold-exploration-rna.md`).
- Validation: run 1 (`bench-runs/validation.jsonl`) failed 3 checks because the Jetson desktop was in use; quiet re-run (`validation-rerun.jsonl`) passed all, with the V4 shoulder check amended to one-sided (decision by Riley 2026-10-23). Details: `validation-run1-2026-10-22.md`, `HARNESS-FROZEN.md` (harness stamp `d5704e0ec3e6`).
- PC accuracy pipeline cross-check passed (`upstream` FAST 0.940763, HAC 0.976852; `spec` −0.55 / −0.14 pp).

## Current state (2026-11-08)
**Read first:** `report-draft-results.md` (4 Nov, the most complete source of final numbers, incl. D.4c–D.4f and E.5), then `exp/hang-root-cause.md`.
- **GPU hangs: root cause found (2026-11-02, `exp/hang-root-cause.md`).** A missing `__syncthreads()` in openfish's beam-search kernel: after the beam-width cut-off loop ends at the 10-guess cap, tid 0 rewrites `bs_active` before every warp has read it, and the block's barriers desynchronise (`src/beam_search_cuda.h`; same pattern in `beam_search_hip.h`). Introduced by BonsonW's "optimise beam search" (openfish `1f26d77`, 2026-06-22), not by this fork's code; it entered this repo when `e00bada` moved the openfish pin from `2530c2a` (upstream slorado's pin, before `1f26d77`) to the fork. Isolated test (`exp/beam_race_stress.py`): 12/40 bad runs without the barrier, 0/340 with it. The `record_stream` lifetime-race hypothesis below is superseded.
- **The fix is an experiment only:** `exp/barrier-fix.patch` (CUDA kernel only), applied in `~/bench-stages/exp_src/openfish`, built as `~/bench-stages/x_barrier/slorado` (`49ac641` + patch, `exp/build_exp.sh barrier`). Not committed to the openfish fork (`5109673`) or to slorado. FAST 1k output read-for-read identical to the campaign's `spec`, no measurable time cost.
- **`spec` 500k measured with the fix** (`x_barrier`, `exp/run_fixed_500k.py`, both on attempt 1): campaign flags 3,970.1 s, 15.74 W, 62.5 kJ, α 0.727, median identity 0.934579 (−0.455 pp against `upstream` 500k 0.939130), 1.59× faster than `upstream` (6,331 s); preprint flags (`--tag paperflags`) 3,935.3 s, 61.8 kJ. Records `bench-runs/variants-fix500k.jsonl`, `variants-fix500k-paperflags.jsonl`. Reported as a labelled variant (report D.4c, E.5), not as a campaign stage.
- **November follow-ups:** `-C` / `-c` sweeps (`bench-runs/csweep-*.jsonl`, report D.4f) and power mode / clock pinning incl. MAXN_SUPER (`bench-runs/climits-*.jsonl`, report D.4e).
- **Accuracy recheck with minimap2 2.26** (PC commit `36fa15f`, `accuracy-mm2-2.26/`): every DNA median identical, every RNA median within 0.005 pp; the reported accuracy numbers stand.
- **Attribution correction (2026-11-08):** Bonson's `dev` (`c2ca899`) pins openfish `991561a`, which branched from this lineage at `9b656e0` (2026-05-07) and does not contain `1f26d77` (its beam search is the pre-June version, no `bs_active`). So "`dev` is not faster than `upstream`" says nothing about the June beam-search rewrite, which may be much of the `e00bada` gain. Not measured. Revised in `report-draft-results.md` D.1, D.4d and E.5.
- **Trial merge of Bonson's `dev` (2026-11-08, future work, not measured):** branch `merge-bonson-dev`, worktree `~/bench-stages/merge_dev` (merge resolved and staged, Riley commits; openfish `5109673` + barrier patch in the build tree only). `--async=no` matches `spec` (FAST 1k 9.09 s); `--async=yes` is ≈ 16 % slower because `pipeline.cpp` calls the serial `basecall_chunks` and loses overlap. Write-up in report D.4d.
- **Still open:** SUP overlap decode (`ov_on`) output differs from `upstream` even with the barrier (separate decode/inference data hazard, `exp/sup_stage_probe.py`); SUP `ov_off` is exact.

## Earlier state (2026-10-26)
**Done:** timing/power campaign (380/387 timed runs; `bench-runs/runs.jsonl`), fix-ups, DNA accuracy on the PC (33 files, incl. `paper_xavier` 500k = 0.939825 vs the preprint's 0.939837), RNA accuracy on the Jetson (16 files), final tables (`bench-runs/tables-with-accuracy.md`, `runs.csv`), a hang captured under gdb (`bench-runs/hang-hunt/`), results copied to the PC (`final-campaign/jetson-campaign-runs/`).
**Read first:** `campaign-review-2026-10-25.md` (failures, hang analysis, outlier cell, decisions), `HARNESS-FROZEN.md`, `campaign-flow.md`, `POST-CAMPAIGN-STEPS.md`.
**Headline (H1, vs `upstream`):** `spec` DNA FAST −34 % (1k) / −37.5 % (20k) at −0.55 / −0.45 pp; DNA HAC −9 to −10 % at −0.14 / −0.18 pp; RNA FAST −33 / −39 % at **−4.2 / −4.3 pp** (not worthwhile); RNA HAC −9 % at −0.35 pp. `upstream`→`ov_off` alone is −17 % FAST (not overlap; openfish fork + buffer changes).

## Open items
1. **Commit and push** the staged docs/scripts on the Jetson `temp` branch (Riley does the commits; no Claude co-author line).
2. ~~`ov_on` FAST 20k headline~~ **Decided 2026-10-26:** exclude outliers by the 3 % rule (`bench-runs/excluded-runs.json`, 3 runs); replicate block kept as a check only.
2b. **Baseline pairing / upstream comparison (DONE 2026-10-27/30):** spec pairing closed (same-binary −7.3 / −8.3 / −0.7 %); Bonson's `dev` not faster than `upstream` (`--async` −2.5 to −2.8 % FAST); bundle split run: the whole `upstream`→`ov_off` gain (−17 % FAST 1k, −19 % FAST 20k, −5 % HAC) is commit `e00bada` (stream-aware decode + fork openfish), instrumentation +3 %, pinned buffers 0 % (`variants-bundle.jsonl`; written into report D.1 and the review doc). **New finding: 5 of 16 FAST 20k attempts hung with `--overlap-decode=no` on the `e00bada`+ lineage (0 of 8 on `upstream`/`u_timing`), so hangs are not specific to overlap.** (Explained 2026-11-02: the openfish beam-search barrier, see Current state.)
3. ~~**`spec` 500k:** no valid measurement~~ **Resolved 2026-11-02:** root cause is the missing beam-search barrier (not the `record_stream` race once suspected here); the campaign's `spec` binary still has no 500k row (5/5 attempts hung), and the number comes from the fixed `x_barrier` build (3,970.1 s, see Current state).
3b. **Back up the Jetson-only November files to the PC** (5 × `csweep-*.jsonl`, `climits-*.jsonl`, `variants-fix500k*.jsonl`, newer `power/` and `console/` traces, optionally `kept-sweeps/`): `docs-riley/pc/backup-check.sh` (dry run) on the PC, then copy and commit in the PC repo.
3c. **Barrier fix:** report it to BonsonW/openfish (CUDA and HIP kernels); decide whether to commit it to the openfish fork.
3d. **Report draft:** the `dev` inference was revised 2026-11-08 (D.1, D.4d, E.5). Still open: D.6 RNA energy uses `spec`, while the presentation's RNA headline is `lb_best` (decided 2026-11-08).
4. **Ask Hasindu:** the preprint's Xavier `nvpmodel` mode and whether `jetson_clocks` was on (needed before claiming a like-for-like 500k comparison: here 25 W, `jetson_clocks` off).
5. Report: first draft of sections D/E and limitations is in `report-draft-results.md` (Riley to revise, **[check]** marks open items); PC-agent chart handoff is `AGENT-HANDOFF-charts-from-runs.md` (Riley to paste to the PC agent after re-copying the Jetson `bench-runs/`).
6. Not done: nsys profile pass per stage (`profile_stage.py` not written); the `upstream`→`ov_off` attribution (cheapest test now: standalone openfish decode time at `c7c0983` against `1f26d77` + barrier, i.e. before and after the June beam-search rewrite); `chr22`; `cb_on` HAC 20k (stall detector limitation). (The `-c 10000` test was done on `upstream`, report D.4f.)

## Standing rules from Riley
- Riley starts campaign runs himself; Claude may run only short permitted checks (smoke, validation, self-tests).
- Claude stages files but never commits or pushes; commits carry no Claude co-author line.
- No `sudo`, no credentials; no desktop use on the Jetson while it measures.

## Where things live
- Jetson repo `~/slorado-riley-v2`, branch `temp`. Run data `docs-riley/bench-runs/` is gitignored (copied to the PC up to the October campaign; November files see open item 3b).
- Post-campaign experiments: scripts and notes in `docs-riley/exp/`, build trees in `~/bench-stages/exp_src/` and `~/bench-stages/x_*` (`x_barrier` = the fixed `spec` binary).
- PC repo `~/Desktop/thesisB-data-and-diagrams` (`jetson-handoff2/` docs, `final-campaign/accuracy`, `final-campaign/jetson-campaign-runs`, `fetch-and-minimap-final.py`).
- Conversation transcript (Jetson, local only): `~/.claude/projects/-home-riley-slorado-riley-v2/d0e94262-12e7-4e7e-b3c3-5201ec536501.jsonl`; resume with `claude --resume`.
