# Handoff for the PC agent: charts and cross-checks from the final campaign data

**Who this is for:** the Claude agent on Riley's main PC (repo `~/Desktop/thesisB-data-and-diagrams`). It is written to be pasted as the first message. The earlier handoffs (`AGENT-CONTEXT-final-test-campaign.md`) described the plan; this one is for when the data exists. Riley commits and pushes himself; you do not.

## Your task
1. **Cross-check the numbers** (before any chart). Re-compute 12 values from the raw CSVs and compare with `tables-with-accuracy.md`; report any mismatch.
2. **Build the charts** listed below from `runs.csv`, `summary-by-cell.csv` and the accuracy CSVs, as PNG + SVG in `final-campaign/charts/`, plus the small CSV behind each chart.
3. **Write `REPLY3-from-PC-agent.md`** in the repo root: what you checked, what you built, anything inconsistent.
Rules: read-only on all data; do not edit `fetch-and-minimap*.py` or the CSVs; no commits or pushes; if something is ambiguous, write the question in the reply file and pick the conservative reading.

## Where the data is
Riley has re-copied the Jetson `bench-runs/` folder to `final-campaign/jetson-campaign-runs/` (check the modification dates: the copy must be **after 2026-10-26**, because `excluded-runs.json`, `summary-by-cell.csv` and the final `runs.csv` were produced then; if `runs.csv` has 380 rows instead of 377, it is the old copy: stop and ask).
| file | what |
|---|---|
| `runs.csv` | 377 timed runs, 56 columns (schema below); the 3 excluded outlier runs are already removed |
| `summary-by-cell.csv` | one row per cell, mode, stage: `n, mean_s, sd_s, min_s, max_s`, change and speed-up against `upstream` |
| `tables-with-accuracy.md` | every table in markdown, including accuracy; **this is the reference for the cross-check** |
| `excluded-runs.json` | the 3 runs removed by the rule "more than 3 % above their cell median" |
| `runs.jsonl` | all raw records (528), including failed attempts, idle baselines and canaries |
| `final-campaign/accuracy/minimap_accuracy-dna-<stage>.csv` | your DNA accuracy (33 rows); `accuracy-rna/minimap_accuracy-rna-<stage>.csv` (16 rows, from the Jetson) |
| `hang-hunt/`, `reducedK-500k.jsonl`, `rerun-ov_on-fast20k.jsonl` | diagnostic captures; **not for charts** (except as noted) |

`runs.csv` columns of interest: `stage, dtype, cell, config(fast|hac), size, mode(A|B), rep, t_real_s, wall_process_s, samples_per_s, E_in_J, Pmean_in_W, E_net_in_J, E_cg_J, E_soc_J, E_resid_J, J_per_1k_reads, J_per_Mbase, spec_alpha, spec_draft_s, spec_repair_s`. Cell ids: `dna_fast_1k`, `rna_hac_20k`, `dna_fast_500k`, etc. Stage order everywhere: `upstream, ov_off, ov_on, lb_baseline, lb_best, spec, cb_off, cb_on, paper_xavier`.

## Conventions (must match the thesis)
- **Mode A** = `-o /dev/null` ("H1 raw speed"); **mode B** = FASTQ written ("H2 end-to-end speed"). Never mix them in one comparison. Default to mode A, show B where stated.
- Time change `d = (t_new − t_base)/t_base`, shown as a negative percent when faster, **together with** speed-up `S = t_base/t_new` (×). Do not call a time reduction a "speed increase".
- Always label the baseline. The baseline is `upstream` unless the chart says otherwise (`ov_off` for the overlap step, `lb_baseline` for load imbalance, `lb_best` for speculative decode).
- Mean ± sd with n in the caption; error bars are 1 sd. Cells with n < 3 get a † marker; the 500k points are single runs and must be drawn without error bars.
- Accuracy is the **absolute** change in median identity in percentage points (pp) against `upstream`, not a relative change.
- Never use `pre_overlap` (`8873209`); it is withdrawn and is not in the data.

## Charts
1. **Stage chain, time:** for each of DNA FAST 1k/20k, DNA HAC 1k/20k: bars `upstream → ov_off → ov_on → lb_best → spec` (omit `lb_baseline` here), labelled with seconds and cumulative `d` against `upstream`.
2. **Same chain for RNA** (stages `upstream, ov_on, lb_best, spec`; RNA FAST/HAC, 1k/20k).
3. **Energy:** per cell, grouped bars for gross and net energy (`E_in_J`, `E_net_in_J`) and a line for mean power (`Pmean_in_W`) for `upstream, ov_on, lb_best, spec` (mode A). Add a panel of J per 1000 reads. Caption: "the board has one shared CPU+GPU+CV rail; the residual is reported separately".
4. **Accuracy vs speed trade-off:** scatter, x = time change vs `upstream`, y = accuracy change in pp, one point per (stage, cell) for `ov_on, lb_best, spec, cb_on` (colour = stage, marker shape = DNA FAST / DNA HAC / RNA FAST / RNA HAC). The RNA FAST `spec` points (≈ −4.2 pp) will stand out; annotate them.
5. **500k panel:** bars for `upstream` and `paper_xavier` (mode B, single runs): `Real time`, throughput in M samples/s, gross energy; add the **preprint's** Xavier value as a separate hatched bar (144.4 min, ≈ 3.24 M samples/s, no energy) and print median identity (0.939130, 0.939825, preprint 0.939837). Caption must say: different devices, Xavier power mode not known, single runs; `spec` did not complete at 500k.
6. **Speculative-decode detail:** accept rate α per cell (`spec_alpha`, mean over runs), plus the split of `spec` run time into draft and repair seconds (`spec_draft_s`, `spec_repair_s`) for the DNA FAST cells.
7. **Dead end:** `cb_off` vs `cb_on` time and energy for DNA FAST 1k/20k and HAC 1k (HAC 20k was not measured; say so in the caption).
8. **Mode cost:** mode B minus mode A time change per stage and cell (all within ≈ ±1.5 %), as one compact figure.

## Cross-check list (report each as OK / mismatch with the value you computed)
Recompute from `runs.csv` (mode A means, then `d` and `S`) and compare with `tables-with-accuracy.md`:
`upstream` DNA FAST 1k 13.842 s; `spec` DNA FAST 1k 9.122 s (−34.1 %, 1.52×); `ov_on` DNA FAST 20k 173.828 s with **n = 2**; `spec` DNA FAST 20k 158.260 s (−37.5 %); `lb_best` DNA HAC 20k 872.756 s; `spec` RNA FAST 20k 121.068 s (−39.4 %);
gross energy `upstream` DNA FAST 20k and `spec` (−22.9 %), net energy change −12.8 %; `spec_alpha` DNA FAST 1k 0.759; 500k `upstream` 6,331.195 s and `paper_xavier` 5,495.826 s (mode B);
accuracy: `spec` DNA FAST 1k 0.935276 (−0.549 pp), RNA FAST 20k 0.889894 (−4.319 pp), `paper_xavier` 500k 0.939825.
Also confirm: the rows excluded from `runs.csv` match `excluded-runs.json` (3 run ids), and no `pre_overlap` rows exist.

## Things that look odd but are known (do not "fix")
- `ov_on` DNA FAST 20k has n = 2 per mode (two outlier runs excluded by the pre-stated 3 % rule); an independent replicate block (`rerun-ov_on-fast20k.jsonl`, 176.5 s mean, n = 3) agrees within 1.6 %. Mention it in the caption; do not average the two blocks.
- `spec` has no 500k row (5 hung attempts); `cb_on` has no HAC 20k row (the harness stall limit); `paper_xavier` exists only at 500k.
- `upstream → ov_off` is −17 % on FAST *with overlap off*: it is the largest single step and is **not** attributable to overlap (the fork also changed openfish and buffer handling). Caption it as "pre-overlap changes", not "overlap".
- Map rate above 100 % in the accuracy CSVs is PAF lines per read (supplementary alignments), not an error.
