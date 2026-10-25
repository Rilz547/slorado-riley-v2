# Results — draft text for the thesis (sections D, E and limitations)

**Status:** first draft for Riley to revise; every number comes from `bench-runs/tables-with-accuracy.md` / `runs.csv` (frozen harness `d5704e0ec3e6`, campaign 2026-10-23 to 2026-10-25). Section letters follow `report-results-outline.md`. Figures are listed in the PC-agent handoff; `[Fig. n]` marks where they go. Anything marked **[check]** needs a source or a decision before submission.

---

## D. Results: methods that improved performance

### D.0 Setup of the comparison
All results are on a Jetson Orin Nano (25 W power mode, `jetson_clocks` off). Each configuration ("stage") is a separate build of the same code base; the baseline for every comparison is the unmodified upstream release (`upstream`, commit `754a74d`). Cells are named `<dna|rna>_<fast|hac>_<size>`: DNA R10.4.1 (v5.0.0 models) on 1,000 and 20,000 HG002 reads and 500,000 reads (FAST only); RNA004 (v5.1.0 models) on 1,000 and 20,000 UHRR reads. The headline metric is slorado's `Real time`, reported for two output modes: **H1 raw speed** (`-o /dev/null`) and **H2 end-to-end speed** (FASTQ written to NVMe). The two differ by less than about 1.5 % in every cell (writing the FASTQ adds ≈ 0.02–0.07 s of `sync`), so tables below give H1 and note H2 where it matters. Repeats are n = 5 (1k) and n = 3 (20k) per mode after one uncounted warm-up; values are mean ± sd. Time change `d = (t_new − t_base)/t_base` (negative is faster) and speed-up `S = t_base/t_new` are reported together; cumulative factors multiply. Power is integrated over exactly slorado's timed window from the board's INA3221 sensor, per rail, with net energy defined as gross minus idle power × window.

### D.1 Overlap decode (`ov_off` → `ov_on`)
Decoding batch *N−1* on the GPU concurrently with inference on batch *N*, using two CUDA streams and a pinned host ring buffer.

| cell | `ov_off` (s) | `ov_on` (s) | change | speed-up |
|---|---|---|---|---|
| DNA FAST 1k | 11.444 ± 0.015 | 10.155 ± 0.019 | −11.3 % | 1.13× |
| DNA FAST 20k | 203.413 ± 0.213 | 173.828 ± 0.321 (n = 2) | −14.5 % | 1.17× |
| DNA HAC 1k | 46.621 ± 0.045 | 45.241 ± 0.032 | −3.0 % | 1.03× |
| DNA HAC 20k | 907.605 ± 5.548 | 877.488 ± 2.799 | −3.3 % | 1.03× |

FAST is decode-bound, so hiding the decode behind inference helps; HAC is inference-bound (the LSTM is about 83 % of its wall time), so the same overlap hides only the small decode share. **Note on the 20k FAST `ov_on` cell:** two of its six runs were 11 % and 13 % slower than the rest for reasons not found in any log; they are excluded by a rule fixed beforehand (any run more than 3 % above its cell median) and an independent replicate block gave 176.5 s (n = 3), 1.6 % above the figure shown, within the harness's 2 % block-to-block tolerance. The −14.5 % would be −13.2 % using the replicate.

**The first step is larger than overlap.** Going from `upstream` to `ov_off` — overlap switched *off* — already gives −17.3 % (1k) and −19.6 % (20k) on DNA FAST (−5.8 % and −5.1 % on HAC). `ov_off` is not upstream with a flag: between the two commits the fork pointed openfish at its own branch, added stream-aware decode, and stopped freeing openfish's host decode buffers after every batch. The split between these changes was **not measured**; the gain cannot be attributed to overlap. **[check: decide whether to build a patched pre-overlap stage to split it]**

### D.2 Load imbalance (`lb_baseline` → `lb_best`)
Running partial batches at their real width (narrow batches) with a flush threshold (64 for FAST, 128 for HAC) instead of padding them to the full GPU batch width.

| cell | `lb_baseline` (s) | `lb_best` (s) | change |
|---|---|---|---|
| DNA FAST 1k | 10.096 ± 0.011 | 9.847 ± 0.020 | −2.5 % |
| DNA FAST 20k | 174.882 ± 0.941 | 172.027 ± 0.088 | −1.6 % |
| DNA HAC 1k / 20k | 45.229 / 878.808 | 45.119 / 872.756 | −0.2 % / −0.7 % |

With the whole input available, padding waste is small, so the offline gain is 1–3 % on FAST and negligible on HAC. The method matters for streaming and partial-batch workloads: an earlier flush-threshold sweep (PC data, single runs) measured −45 % at flush 64 and −77 % at flush 4 against padded fixed-width batches. **[check: cite the sweep file]** Accuracy is unchanged (median identity differs by ≤ 0.001 pp).

### D.3 Speculative (draft-and-repair) decoding (`lb_best` → `spec`)
A cheap greedy draft decodes every chunk; chunks whose mean draft quality is below Q10 or whose mean decision margin is below 2 are repaired by the full beam search, with the repair overlapped with the next batch's inference.

| cell | `lb_best` (s) | `spec` (s) | change | accept rate α | accuracy change |
|---|---|---|---|---|---|
| DNA FAST 1k | 9.847 | 9.122 ± 0.022 | −7.4 % | 0.759 | −0.549 pp |
| DNA FAST 20k | 172.027 | 158.260 ± 0.367 | −8.0 % | 0.730 | −0.452 pp |
| DNA HAC 1k | 45.119 | 44.748 ± 0.067 | −0.8 % | 0.827 | −0.136 pp |
| DNA HAC 20k | 872.756 | 870.638 ± 1.047 | −0.2 % | 0.809 | −0.181 pp |
| RNA FAST 1k | 7.753 | 7.116 ± 0.033 | −8.2 % | 0.905 | **−4.164 pp** |
| RNA FAST 20k | 135.987 | 121.068 ± 0.112 | −11.0 % | 0.905 | **−4.319 pp** |
| RNA HAC 1k / 20k | 35.471 / 708.302 | 34.967 / 700.137 | −1.4 % / −1.2 % | 0.930 / 0.925 | −0.346 / −0.349 pp |

The gain is concentrated where decoding dominates (FAST). On DNA the accuracy cost is about half a percentage point for FAST and 0.1–0.2 pp for HAC. **On RNA FAST the method is not worthwhile:** thresholds tuned on DNA accept 90 % of drafts, but the accepted drafts are poor (output is 8 % shorter), costing 4.2–4.3 pp. A 13-setting threshold grid on 1k RNA reads found no setting that was both faster and accurate (the quality gate dominates; `spec-threshold-exploration-rna.md`). Accuracy is the median of PAF `matches / block length` over all alignments (minimap2 `map-ont` against hg38; `splice -uf -k14` against GENCODE v40 for RNA, both `--secondary=no`), following the Slorado/Openfish preprint; the Δ is in absolute percentage points.

### D.4 Combined effect
Time change against `upstream` (H1), cumulative:

| cell | `ov_on` | `lb_best` | `spec` | `spec` speed-up |
|---|---|---|---|---|
| DNA FAST 1k | −26.6 % | −28.9 % | **−34.1 %** | 1.52× |
| DNA FAST 20k | −31.3 % | −32.0 % | **−37.5 %** | 1.60× |
| DNA HAC 1k | −8.6 % | −8.8 % | −9.6 % | 1.11× |
| DNA HAC 20k | −8.3 % | −8.8 % | −9.0 % | 1.10× |
| RNA FAST 1k | −20.8 % | −27.4 % | −33.4 % | 1.50× |
| RNA FAST 20k | −27.2 % | −31.9 % | **−39.4 %** | 1.65× |
| RNA HAC 1k / 20k | −7.5 % / −7.5 % | −7.6 % / −8.1 % | −8.9 % / −9.1 % | 1.10× |

End-to-end (H2) gives the same picture: `upstream` → `spec` is −33.7 % / −37.0 % (DNA FAST 1k / 20k) and −9.4 % / −8.5 % (DNA HAC). Absolute throughput on DNA FAST 20k rises from 4.43 to 7.08 M samples/s. [Fig. 1, Fig. 2]

### D.4c Large dataset: 500,000 DNA reads, FAST, end-to-end
| | `Real time` | throughput | gross energy | median identity |
|---|---|---|---|---|
| `upstream` (`-C 128 -c 12288 -K 4096 -p 150`) | 6,331 s (105.5 min) | 4.44 M samples/s | 79.4 kJ | 0.939130 |
| `paper_xavier` (commit `406ab8c`, the preprint's flags `-t 8 -B 500M -K 2000 -c 10000 -C 200`) | 5,496 s (91.6 min) | 5.11 M samples/s | 74.3 kJ | **0.939825** |
| preprint, Jetson AGX Xavier, same commit and flags | 144.4 min | ≈ 3.24 M samples/s | not reported | 0.939837 |

With the preprint's commit and command line, the Orin Nano reproduces the published median identity to 0.000012 and is 37 % faster in wall time than the preprint's Xavier. This is a comparison of two different devices, not a controlled one: the Xavier's power mode and `jetson_clocks` setting are not stated in the sources available to us and the software stacks differ. **[check: confirm with the preprint's authors before claiming a like-for-like comparison]** The larger chunk size (`-c 10000`) alone accounts for the 0.07 pp accuracy difference between the two rows here. Both rows are single runs. **`spec` did not complete at 500k** (see E.5). [Fig. 5]

### D.5 Accuracy summary
Median identity change in percentage points against `upstream`: overlap, load imbalance and the CPU/offline variants change nothing (≤ 0.001 pp, with 0.023 pp for the CPU-beam hybrid on DNA FAST 1k); `spec` costs −0.45 to −0.55 pp (DNA FAST), −0.14 to −0.18 pp (DNA HAC), −4.2 to −4.3 pp (RNA FAST) and −0.35 pp (RNA HAC). [Fig. 4]

### D.6 Power and energy
Gross energy is the integral of `VDD_IN` over the timed window; net energy subtracts the block's idle power (≈ 5.1 W) times the window. Because the faster stages draw more power (the GPU is busier), energy falls by less than time:

| cell | time | mean power (W) | gross energy | net energy |
|---|---|---|---|---|
| DNA FAST 1k (`upstream` → `spec`) | −34.1 % | 11.82 → 13.75 | −23.3 % | −15.1 % |
| DNA FAST 20k | −37.5 % | 12.58 → 15.52 | −22.9 % | −12.8 % |
| DNA HAC 1k / 20k | −9.6 % / −9.0 % | 16.35 → 17.00 / 16.67 → 17.46 | −6.0 % / −4.7 % | −4.2 % / −2.7 % |
| RNA FAST 1k / 20k | −33.4 % / −39.4 % | 11.41 → 13.26 / 12.38 → 15.41 | −22.6 % / −24.5 % | −13.7 % / −14.0 % |
| RNA HAC 1k / 20k | −8.9 % / −9.1 % | 16.28 → 16.87 / 16.93 → 17.52 | −5.6 % / −6.0 % | −4.0 % / −4.6 % |

Energy per 1,000 reads on DNA FAST 20k falls from 159.2 J (`upstream`) to 122.8 J (`spec`). The board has one shared CPU+GPU+CV rail and no DRAM rail, so energy cannot be split between GPU, CPU and memory; the residual (board, DRAM, regulator loss, NVMe) is about a third of total energy and is reported as a separate column. The earlier single-run August power results are superseded by these multi-run figures. [Fig. 3]

---

## E. Results: methods that did not improve performance

### E.1 CPU beam search hybrid (`cb_off` → `cb_on`)
Moving the beam search to the CPU was slower and used more energy in every cell measured: DNA FAST 1k +9.6 % time and +20.6 % gross energy; DNA HAC 1k +22.0 % and +17.2 %; DNA FAST 20k +11.3 % and +19.9 %. Accuracy was unchanged (−0.023 to +0.003 pp). The HAC 20k cell was **not measured**: the CPU-beam pipeline prints no progress output and the harness's 900 s stall detector killed the 50-minute run on every attempt, a limitation of the measurement code, not of the method.

### E.2 – E.4 Not re-measured
Overlap with depth 2, the FAST→HAC cascade (an accuracy/speed trade-off, always slower than FAST), speculative-decode variants (`m = 3`), larger batches (`-C 256/512` runs out of device memory with overlap on) and int8 quantisation (not available for the v5 LSTM models in this code base) were assessed during development and are described from the earlier documentation (`2026-10-21-optimisation-summary.md`); they are not part of the final campaign data. **[check: wording, and whether to include numbers from the older runs]**

### E.5 Intermittent GPU hangs and the missing 500k `spec` result
Seven runs in the campaign and follow-ups hung and were killed by the harness: `lb_baseline`, `cb_off` and `ov_on` on FAST 20k (each recovered on retry) and `spec` on FAST 500k, which hung on all five attempts (after 20–80 % of the reads, at different positions each time) including a run at a smaller read batch (`-K 2000`). Every hang had the same signature: power falls from about 13–16 W to about 6 W, the CPU idles, and the GPU load counter reads 99.9 % with no output. All occurred in stages with overlap decode enabled (none in 4.6 hours of overlap-off FAST running, none on HAC). Under `gdb`, one hang was captured: the single active thread was blocked in `cudaStreamSynchronize` on the decode/repair stream at `src/basecall.cpp:874`, waiting on a GPU fence that never signalled. The cause is not established; a cross-stream memory-lifetime race in the overlap code is the leading hypothesis (no `record_stream` is used) and has not been tested. One run of the 500k `spec` configuration did complete under `gdb` in 3,991 s (−37 % against `upstream`, in line with the 20k result), but it was not a measured run and is not tabulated. This is a reliability limitation of the overlap-decode implementation: the 20k-and-below results are unaffected (hung runs were repeated and excluded as failures), but the method is not yet dependable for runs of an hour or more.

---

## Limitations and threats to validity
- **Hardware and platform:** a single Jetson Orin Nano in one power mode; power cannot be split between GPU, CPU and DRAM.
- **Run counts:** n = 5 (1k) and n = 3 (20k) per mode; the 500k results are single runs; the `ov_on` 20k FAST cell has n = 2 after excluding two outliers (replicate n = 3 agrees within 1.6 %).
- **Measurement disturbances:** desktop activity on the Jetson caused outliers in an earlier validation run; the campaign was run over SSH, but one run coincided with a system-monitor launch (excluded by the outlier rule, with two others; `excluded-runs.json`). Validation criteria not met on the campaign data: run-time CV above 1 % in 4 of 94 groups (three because of the outliers), process-exit gaps above 3 s in five HAC 20k runs (max 29 s), one idle shoulder +0.79 W; none affects `Real time` or the energy window.
- **Baseline pairing:** the `upstream` → `ov_off` step bundles several changes (D.1).
- **Accuracy method:** median identity over all alignment lines (including supplementary alignments), against hg38 / GENCODE v40, following the preprint; relative error rates are not reported.
- **Withdrawn data:** the earlier pre-overlap stage (`8873209`) decoded garbage because of a score-layout mismatch and is excluded from every comparison.
- **Speculative decode on RNA FAST** is reported as a negative result with DNA-tuned thresholds; re-tuning was explored only on 1k reads.

## Reproducibility
Scripts, configuration and the frozen-harness record: `docs-riley/bench/`, `HARNESS-FROZEN.md`; stage builds `~/bench-stages/` (commits in `stages.json`); run data `bench-runs/runs.jsonl` (528 records), `runs.csv` (377 timed runs after exclusions), accuracy CSVs, hang captures; analysis `analyze.py csv|tables|outliers|validate|equiv`.
