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

## Not done / next (in order)
1. Riley reads `HARNESS-FROZEN.md`, ticks sign-off.
2. Archive `bench-runs/kept/` → `kept-prefreeze`; on the PC move `final-campaign/accuracy` → `accuracy-prefreeze`.
3. Riley starts the campaign (`bench.py run --stage <stage> --resume`, ≈ 32 h total; 1k cells first, then 20k, then 500k FAST on `paper_xavier`/`upstream`/`spec`). No desktop use on the Jetson; monitor over SSH. Not started yet (`bench-runs/runs.jsonl` does not exist).
4. nsys profile pass per stage (`profile_stage.py` is not written yet).
5. After the campaign: copy kept FASTQs to the PC; DNA accuracy with `fetch-and-minimap.py` (DNA only; RNA FASTQs must NOT go through it), RNA accuracy on the Jetson with `bench/accuracy.py --dtype rna`; `analyze.py --accuracy minimap_accuracy-*.csv tables`; write the post-campaign handoff for the PC agent; fill the report sections.
6. Deferred: chr22 tier; a single isolated `-c 10000` test.

## Standing rules from Riley
- Riley starts campaign runs himself; Claude may run only the permitted smoke and validation runs.
- Claude stages files but never commits or pushes; commits carry no Claude co-author line.
- No `sudo`, no credentials; copy rather than move from the Trash.

## Where things live
- Jetson repo `~/slorado-riley-v2`, branch `temp` (docs and harness staged, not committed). Run data in `docs-riley/bench-runs/` is gitignored: back it up to the PC (`*.jsonl`, `power/`, `console/`, `kept/`).
- PC repo `~/Desktop/thesisB-data-and-diagrams` (`jetson-handoff2/` docs, `final-campaign/`, `fetch-and-minimap.py`).
- Full conversation transcript (Jetson, local only): `~/.claude/projects/-home-riley-slorado-riley-v2/d0e94262-12e7-4e7e-b3c3-5201ec536501.jsonl`; resume with `claude --resume` from `~/slorado-riley-v2`.
