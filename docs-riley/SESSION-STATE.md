# Session state — where the final-campaign work stands (2026-10-23)

Purpose: let a fresh Claude session (or Riley, or the PC agent) pick this up without the original conversation. Read this first, then `HARNESS-FROZEN.md`.

## Goal
Thesis on slorado (Nanopore basecaller) optimisation on a Jetson Orin Nano (25 W, `jetson_clocks` off). Redo all timing, power, profiling and accuracy tests on one trusted, frozen harness, then write the results sections (`report-results-outline.md`).

## Done
- `speculative-decode` branch rebuilt (greedy kernel re-created in openfish `5109673`, slorado `49ac641`), validated against the August report.
- Harness in `docs-riley/bench/` (own INA3221 power logger, per-rail and net energy, H1 `/dev/null` and H2 FASTQ modes, idle baseline, canary, window recovered from slorado's progress lines). Stage worktrees in `~/bench-stages/`. Datasets in `~/data/` (DNA 1k/20k/500k, RNA 1k/20k, GENCODE v40).
- Findings: `pre_overlap` (`8873209`) is INVALID (score-layout bug, fixed in v1.0 `e00bada`); August "original → now" chains built on it are withdrawn. RNA speculative decode: Q gate dominates, no setting both faster and accurate on RNA FAST (`spec-threshold-exploration-rna.md`).
- Validation: run 1 (`bench-runs/validation.jsonl`) failed 3 checks because the Jetson desktop was in use; quiet re-run (`validation-rerun.jsonl`) passed all, with the V4 shoulder check amended to one-sided (decision by Riley 2026-10-23). Details: `validation-run1-2026-10-22.md`, `HARNESS-FROZEN.md` (harness stamp `d5704e0ec3e6`).
- PC accuracy pipeline cross-check passed (`upstream` FAST 0.940763, HAC 0.976852; `spec` −0.55 / −0.14 pp).

## Current state (2026-10-26)
**Done:** timing/power campaign (380/387 timed runs; `bench-runs/runs.jsonl`), fix-ups, DNA accuracy on the PC (33 files, incl. `paper_xavier` 500k = 0.939825 vs the preprint's 0.939837), RNA accuracy on the Jetson (16 files), final tables (`bench-runs/tables-with-accuracy.md`, `runs.csv`), a hang captured under gdb (`bench-runs/hang-hunt/`), results copied to the PC (`final-campaign/jetson-campaign-runs/`).
**Read first:** `campaign-review-2026-10-25.md` (failures, hang analysis, outlier cell, decisions), `HARNESS-FROZEN.md`, `campaign-flow.md`, `POST-CAMPAIGN-STEPS.md`.
**Headline (H1, vs `upstream`):** `spec` DNA FAST −34 % (1k) / −37.5 % (20k) at −0.55 / −0.45 pp; DNA HAC −9 to −10 % at −0.14 / −0.18 pp; RNA FAST −33 / −39 % at **−4.2 / −4.3 pp** (not worthwhile); RNA HAC −9 % at −0.35 pp. `upstream`→`ov_off` alone is −17 % FAST (not overlap; openfish fork + buffer changes).

## Open items
1. **Commit and push** the staged docs/scripts on the Jetson `temp` branch (Riley does the commits; no Claude co-author line).
2. ~~`ov_on` FAST 20k headline~~ **Decided 2026-10-26:** exclude outliers by the 3 % rule (`bench-runs/excluded-runs.json`, 3 runs); replicate block kept as a check only.
3. **`spec` 500k:** no valid measurement (5/5 attempts hung; run 1 under gdb finished in 3,991 s, indicative only). Hang location: `cudaStreamSynchronize` in `overlap_sync_repair_and_stitch` (`src/basecall.cpp:874`); candidate cause: cross-stream tensor lifetime race (no `record_stream`). Untested fix: keep the original scores alive until the repair sync + `record_stream(decode_stream)`, then re-run `hang_hunt.py`.
4. **Ask Hasindu:** the preprint's Xavier `nvpmodel` mode and whether `jetson_clocks` was on (needed before claiming a like-for-like 500k comparison: here 25 W, `jetson_clocks` off).
5. Report: first draft of sections D/E and limitations is in `report-draft-results.md` (Riley to revise, **[check]** marks open items); PC-agent chart handoff is `AGENT-HANDOFF-charts-from-runs.md` (Riley to paste to the PC agent after re-copying the Jetson `bench-runs/`).
6. Not done: nsys profile pass per stage (`profile_stage.py` not written); the `upstream`→`ov_off` attribution (a patched `pre_overlap` stage could split it); isolated `-c 10000` test; `chr22`; `cb_on` HAC 20k (stall detector limitation).

## Standing rules from Riley
- Riley starts campaign runs himself; Claude may run only short permitted checks (smoke, validation, self-tests).
- Claude stages files but never commits or pushes; commits carry no Claude co-author line.
- No `sudo`, no credentials; no desktop use on the Jetson while it measures.

## Where things live
- Jetson repo `~/slorado-riley-v2`, branch `temp`. Run data `docs-riley/bench-runs/` is gitignored (copied to the PC).
- PC repo `~/Desktop/thesisB-data-and-diagrams` (`jetson-handoff2/` docs, `final-campaign/accuracy`, `final-campaign/jetson-campaign-runs`, `fetch-and-minimap-final.py`).
- Conversation transcript (Jetson, local only): `~/.claude/projects/-home-riley-slorado-riley-v2/d0e94262-12e7-4e7e-b3c3-5201ec536501.jsonl`; resume with `claude --resume`.
