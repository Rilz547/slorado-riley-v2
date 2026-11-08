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

**The first step is not overlap.** Going from `upstream` to `ov_off` (overlap switched *off*) already gives −17.3 % (1k) and −19.6 % (20k) on DNA FAST, and −5.8 % and −5.1 % on HAC. `ov_off` is not upstream with a flag: it is commit `7cec6c5`, nine slorado commits and an openfish update ahead of `upstream`. A split run of the intermediate commits (same harness, interleaved, DNA, mode A) places the whole gain at one commit:

| step (commit) | DNA FAST 1k | DNA HAC 1k | DNA FAST 20k |
|---|---|---|---|
| `upstream` (`754a74d`) | 13.815 s | 49.043 s | 252.30 s |
| + LSTM layer-timing instrumentation (`3403f19`) | +3.2 % | +0.7 % | +3.8 % |
| + stream-aware decode and the fork's openfish (`e00bada`) | **−19.7 %** | −6.0 % | **−21.8 %** |
| + persistent pinned host decode buffers (`9f8db24`) | +0.2 % | 0.0 % | −0.3 % |
| = `ov_off` (`7cec6c5`) | −0.2 % | −0.1 % | −0.1 % |
| cumulative against `upstream` | −17.2 % | −5.4 % | −19.1 % |

(Each row is the change against the row above; 5 runs per cell at 1k and 3 at 20k, 2 for `e00bada` on 20k. The instrumentation commit is measurement code that adds about 3 % on FAST; the stray decoder-option line in that commit was reverted in the build tree so that it builds with upstream's openfish.) The step `e00bada` changed the slorado decode path (`basecall.cpp`, 353 lines: decode on its own CUDA stream, separate buffers) and moved openfish from upstream's pin to the fork (44 further openfish commits, 41 of them the maintainers'). Upstream's own current `dev` branch (v0.7.0 in progress, `c2ca899`), built and run with the same flags, is **not faster than `upstream`** (+0.2 % at 1k and −0.2 % at 20k on FAST, +0.1 % on HAC; D.4d), but that does not test the openfish side of `e00bada`: `dev`'s openfish (`991561a`) branched from the fork's openfish lineage in May 2026 and does not contain the maintainers' June 2026 rewrite of the beam-search kernel (`1f26d77`, "optimise beam search"), which the fork's pin does contain. That rewrite is therefore a plausible source of much of the gain. Which part of `e00bada` produces it (the slorado-side restructuring or the openfish update it depends on, in particular the beam-search rewrite) is **not separated**, because that commit cannot be built against upstream's old openfish. The gain is therefore reported as "pre-overlap changes" and is not credited to overlap or to the author's methods. **[check: optional follow-up, time the standalone openfish decoder before and after `1f26d77`, or port the slorado side onto upstream's openfish]**

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

A same-binary check isolates the method from the other differences between the two commits: the `49ac641` build with `--speculative-decode=no` versus `=yes`, interleaved run by run in two independent blocks, gives −7.3 % (DNA FAST 1k), −8.3 % (FAST 20k) and −0.7 % (HAC 1k) with block-to-block agreement within 0.5 percentage points, essentially the same as the `lb_best` → `spec` figures above (−7.4 %, −8.0 %, −0.8 %); the speculative-off outputs are read-for-read identical to `lb_best`'s, and the speculative-on outputs to the campaign's `spec`. The gain is concentrated where decoding dominates (FAST). On DNA the accuracy cost is about half a percentage point for FAST and 0.1–0.2 pp for HAC. **On RNA FAST the method is not worthwhile:** thresholds tuned on DNA accept 90 % of drafts, but the accepted drafts are poor (output is 8 % shorter), costing 4.2–4.3 pp. A 13-setting threshold grid on 1k RNA reads found no setting that was both faster and accurate (the quality gate dominates; `spec-threshold-exploration-rna.md`). Accuracy is the median of PAF `matches / block length` over all alignments (minimap2 `map-ont` against hg38; `splice -uf -k14` against GENCODE v40 for RNA, both `--secondary=no`), following the Slorado/Openfish preprint; the Δ is in absolute percentage points.

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
| `spec`, patched build (`x_barrier`: `49ac641` plus a one-line kernel fix, E.5; the campaign's `spec` flags) | **3,970 s (66.2 min)** | **7.08 M samples/s** | 62.5 kJ | 0.934579 |
| `spec`, patched build at the preprint's flags (`x_barrier` with `-t 8 -B 500M -K 2000 -c 10000 -C 200` added) | **3,935 s (65.6 min)** | **7.14 M samples/s** | 61.8 kJ | 0.934641 |
| `paper_xavier` (commit `406ab8c`, the preprint's flags `-t 8 -B 500M -K 2000 -c 10000 -C 200`) | 5,496 s (91.6 min) | 5.11 M samples/s | 74.3 kJ | **0.939825** |
| preprint, Jetson AGX Xavier, same commit and flags | 144.4 min | ≈ 3.24 M samples/s | not reported | 0.939837 |

With the preprint's commit and command line, the Orin Nano reproduces the published median identity to 0.000012 and is 37 % faster in wall time than the preprint's Xavier. This is a comparison of two different devices, not a controlled one: the Xavier's power mode and `jetson_clocks` setting are not stated in the sources available to us and the software stacks differ. **[check: confirm with the preprint's authors before claiming a like-for-like comparison]** The two rows differ in commit, chunk length (`-c 10000` against `-c 12288`), GPU batch (`-C 200` against `128`) and read-batch settings, and their median identities differ by 0.07 pp (0.939825 against 0.939130); the identity gap is not a like-for-like comparison, because `paper_xavier` produces different output from `upstream`: 2.093 G bases against 2.035 G (+2.8 %) and 653,458 alignments for the 500,000 reads (130.7 %) against 518,542 (103.7 %), so the two medians are taken over differently composed alignment sets. The flags are not the cause: the patched `spec` build writes 2.033 G bases at the campaign's flags and 2.034 G at the preprint's flags. The patched build's output at the preprint's flags confirms it: 518,836 alignments (103.8 %), in line with `upstream` and the patched build at the campaign's flags (517,795), and median identity 0.934641 against 0.934579 at the campaign's flags (+0.006 pp). The extra alignments and bases are therefore a property of commit `406ab8c`'s code, whose cause was not investigated. The chunk length does not explain the identity gap either: on `upstream`, `-c 10000` changes the 20k-read median by +0.004 pp (D.4f). In timing, an isolated test on `upstream` (D.4f) shows `-c 10000` alone is 2.3–3.0 % faster than 12,288, so the chunk length explains only a small part of the 13 % by which `paper_xavier` is faster than `upstream` (5,496 s against 6,331 s); the rest (commit, `-C`, `-K`, `-B`, `-t`) has not been isolated. All rows are single runs. (The PC mapping script's CSV for the patched build shows 500,002 reads, not 500,000: it counts every line that begins with `@`, and two quality lines in that file begin with `@`, a valid Phred-31 character. The file has exactly 500,000 records, as the harness's own count says; `upstream`'s 500k file has the same two lines, but its CSV, written on 2026-10-25, says 500,000, so the script's counting must have changed since. The median identity is not affected.) **The campaign's own `spec` binary could not complete a 500k run** because of a GPU hang (E.5); the `spec` row is the same code and flags built with a one-line fix to the defect behind those hangs and run through the same harness. It completed on its first attempt (canary +0.9 %, idle power +0.08 W, 500,000 reads written, accept rate 0.727). It is 37.3 % faster than `upstream` (1.59×) with 21 % less energy, the same as at 20k (−37.5 %), and costs 0.455 pp of median identity (error rate 6.09 % → 6.54 %, +7.5 % relative), against 0.452 pp at 20k. Run again with the preprint's flags added (`-t 8 -B 500M -K 2000 -c 10000 -C 200`; the later options override the campaign's `-C 128 -c 12288 -K 4096`), the same binary completed on its first attempt in 3,935 s (canary 159.4 → 163.4 ms, idle 5.17 → 5.20 W, 500,000 reads written, accept rate 0.725, 61.8 kJ, peak memory 4.1 GB against 5.1 GB). That is 0.9 % faster than at the campaign's flags, so the flags matter little to `spec` (its flush threshold of 64 caps the real batch, so `-C 200` only sets buffer size; the shorter chunks account for the small gain, D.4f). At the same flags it takes 28.4 % less time than `paper_xavier` (1.40×) with 16.8 % less energy, 37.8 % less than `upstream` (1.61×), and 54.6 % less than the preprint's Xavier figure (2.20×); the last comparison crosses two devices with unknown power settings and is not controlled. [Fig. 5]

### D.4d Against upstream's own latest development branch (`dev`, v0.7.0 in progress)
Upstream's `dev` branch (commit `c2ca899`, 2026-10-19; 105 commits ahead of the base used here) adds an experimental `--async` pipeline that overlaps I/O, preprocessing, inference and output, an automatic read batch size and a newer openfish release (`991561a`; it does not include the June 2026 beam-search rewrite that the fork uses from `e00bada` on, see D.1). It was built on the same Jetson and run beside the base release and the author's `spec`, interleaved run by run, in two independent blocks (mode A, same flags; times pooled over both blocks, which agree within 1 %):

| cell | `upstream` (s) | `dev` | `dev --async=yes` | author's `spec` |
|---|---|---|---|---|
| DNA FAST 1k | 13.872 ± 0.07 | 13.899 (+0.2 %) | 13.476 (−2.8 %) | **9.131 (−34.2 %, 1.52×)** |
| DNA HAC 1k | 49.329 ± 0.30 | 49.389 (+0.1 %) | 49.079 (−0.5 %) | **44.813 (−9.2 %, 1.10×)** |
| DNA FAST 20k | 254.221 ± 1.04 | 253.624 (−0.2 %) | 247.834 (−2.5 %) | **159.534 (−37.2 %, 1.59×)** |

Against the best upstream configuration (`dev --async=yes`) the author's `spec` is still 32 % (FAST 1k), 36 % (FAST 20k) and 9 % (HAC 1k) faster. The `dev` build's output differs from the base release's in about 14 % of reads (lengths identical), but its accuracy is the same: median identity is 0.940763 / 0.939381 / 0.976852 (DNA FAST 1k / FAST 20k / HAC 1k) against `upstream`'s 0.940763 / 0.939375 / 0.976852, a difference of at most 0.0006 pp, and `--async=yes` produces output with the same identities as plain `dev`. So the comparison is not confounded by an accuracy difference: `dev` is neither faster nor more accurate than the base release, while `spec` trades −0.45 to −0.55 pp (DNA FAST) and −0.14 pp (DNA HAC) for its speed. **[check: RNA not run on `dev`; one run per block set, no energy tabulated]**

**Combining `dev` with the author's methods (future work).** A trial merge of `dev` into the author's branch (`merge-bonson-dev`, openfish kept at the fork's pin because overlap and `spec` need its API) builds and runs. With `--async=no` it is as fast as `spec` (DNA FAST 1k 9.09 and 9.13 s against 9.08 s for `spec` with the barrier fix; smoke runs, not measured with the harness) and makes the same repair decisions (α 0.759); its output differs from `spec` in 117 of 1,000 reads, about as much as `dev` itself differs from `upstream` (131 of 1,000). With `--async=yes` it is about 16 % slower (10.51 and 10.54 s), because `dev`'s pipeline calls the serial decode path and so loses overlap decode. A real combination needs the async runner to drive the overlap decode path; its gain is bounded by what `--async` gives `dev` on its own (−2.5 to −2.8 % on FAST, −0.5 % on HAC).

### D.4e Sensitivity to power mode and clock pinning
All campaign results are at 25 W with `jetson_clocks` off, where the GPU averages 824–895 MHz against a 918 MHz limit. Two separate side experiments, outside the frozen harness (same measurement code, without its power-mode/clock pre-flight; own data files), check how much that setting matters. They are not part of the campaign tables.

*Clock pinning at 25 W* (`spec`, mode A, 5 runs per block, clocks off → on → off again; GPU pinned at 918 MHz, CPU and memory at maximum): DNA FAST 1k 9.140 ± 0.047 s → 8.817 ± 0.020 s (−3.5 %; the second clocks-off block 9.106 s). DNA HAC 1k 45.685 → 44.783 s (−2.0 %), which is no larger than the 1.4 % difference between the two clocks-off blocks. Energy per run is unchanged to +2.2 % (FAST) and idle power rises from 5.2 W to 7.0 W, so pinning trades power for a few percent of speed. An 11 % higher GPU clock buys about a third of that in time, consistent with the pipeline not being purely GPU-clock-bound.

*MAXN_SUPER with `jetson_clocks` on* (GPU 1020 MHz throughout, board junction ≤ 61 °C, canary 162.6 and 161.4 ms, no failed or retried run; interleaved, DNA 1k n = 3, 20k n = 1):

| cell | `upstream` | `ov_off` | `spec` | `ov_off` change (campaign) | `spec` change (campaign) |
|---|---|---|---|---|---|
| DNA FAST 1k | 11.816 ± 0.013 s | 9.718 ± 0.015 s | 8.062 ± 0.067 s | −17.8 % (−17.3 %) | **−31.8 %** (−34.1 %) |
| DNA HAC 1k | 48.036 ± 0.297 s | 45.381 ± 0.130 s | 44.159 ± 0.203 s | −5.5 % (−5.8 %) | **−8.1 %** (−9.6 %) |
| DNA FAST 20k (single run) | 223.8 s | 181.6 s | 148.3 s | −18.8 % (−19.6 %) | **−33.7 %** (−37.5 %) |

The stage ordering and the gain from stream-aware decode are unchanged; the gain of `spec` over `upstream` is smaller by 1.5–3.8 percentage points, because the faster clock helps `upstream` more (FAST 1k 13.8 → 11.8 s) than `spec` (9.1 → 8.1 s), which is already limited by inference. Energy per run keeps the same ordering (FAST 1k 163 J `upstream`, 144 J `ov_off`, 129 J `spec`). The board's over-current event counter rose by about 122,000 in the 28-minute MAXN_SUPER run (about 6,300 in the 25-minute clock-pinning experiment), so the 45 W supply is being hit hard in this mode; it is not recommended for unattended use. If the preprint's Xavier ran at maximum clocks, the comparison in D.4c would shift by a few percent at most. **[check: Xavier power mode and `jetson_clocks` state; confirm with the preprint's authors]**

### D.4f Tuning sweeps: GPU batch width (`-C`) and chunk length (`-c`)
Both sweeps ran on 2026-11-02 through the frozen harness's measurement code (`exp/c_sweep.py`; mode A unless stated; one warm-up and three timed runs per value; desktop session running, so about 0.5 GB of the 8 GB is not available). A value that ran out of device memory (`NvMap` allocation abort, return code −6) was recorded as failed, was not retried, and larger values were skipped. Other options as in the campaign (`-c 12288 -K 4096 -p 150`; `spec` with its campaign flags, flush threshold 64 for FAST and 128 for HAC).

**Batch width `-C`** (DNA 1k reads, mean of three timed runs in seconds; bold = fastest in the column; sd ≤ 0.5 s except `upstream` HAC at 64, 1.2 s):

| `-C` | FAST `upstream` | FAST `ov_on` | FAST `spec` | HAC `upstream` | HAC `ov_on` | HAC `spec` |
|---|---|---|---|---|---|---|
| 32 | 23.10 | 11.82 | 13.24 | 86.41 | 74.55 | 77.97 |
| 64 | 14.66 | **9.75** | 9.15 | 56.03 | 49.32 | 49.31 |
| 96 | 16.09 | 10.94 | 9.14 | 54.35 | 50.00 | 49.42 |
| 128 | **13.87** | 10.13 | **9.12** | **49.40** | **45.19** | **44.81** |
| 192 | 14.59 | 11.08 | 9.21 | 55.08 | 51.44 | 45.07 |
| 256 | 14.06 | out of memory | 9.26 | 53.66 | out of memory | 45.44 |
| 384 | 15.68 | not run | 9.30 | out of memory | not run | 46.27 |
| 512 | out of memory | not run | 9.40 | not run | not run | out of memory |

Three things follow. (1) `-C 128`, the value used throughout the campaign, is the fastest value in five of the six columns (in FAST `spec`, 64 and 96 are within 0.3 %); the exception is FAST `ov_on`, where `-C 64` is 4.0 % faster. (2) `-C 256` is not a second optimum: it is within 1.5 % of 128 for FAST `upstream` and `spec` but 8.6 % slower for HAC `upstream`, and with overlap on it does not run at all. (3) Too large means out of memory, and where that happens depends on the stage: on 1,000 reads with the desktop running, `upstream` fits up to 384 (FAST) and 256 (HAC), `ov_on` only up to 192 in both models (its peak memory is higher: 3.6 GB against 2.6 GB for FAST at `-C 128`), and `spec` up to at least 512 (FAST, 3.3 GB at 512) and 384 (HAC, 5.8 GB). `spec` is flat for FAST from 64 to 128 and loses 1–3 % at 192–512, because the flush threshold of 64 caps the real batch and `-C` then only enlarges the buffers; for HAC the flush threshold is 128, so 64 and 96 cut the batch and cost 10 %. `-C 32` is 21–75 % slower than the best value in every column. The curve is not smooth: 96 and 192 are slower than their power-of-two neighbours (FAST `upstream` 96: +16 % against 128, while 64 is +5.7 %), which suggests the GPU kernels prefer power-of-two batch widths; the mechanism was not investigated. Each point is one block of three runs on 1,000 reads, so differences of 1–2 % at the top of each column are small. **[check: the paper's `-C 200` lies between the unfavourable 192 and 256 values; its speed was not measured]**

**Largest `-C` that fits (20,000 reads).** The 1k grid only brackets the memory limit coarsely, and 1,000 reads understate the memory needed. A second queue on 2026-11-04 (`exp/c_limits.py`, started by `exp/queue_limits.sh`; 57 min) bisected the limit on the 20,000-read set to a resolution of 16. Each probe started the basecaller with the stage's campaign flags and the candidate `-C`, output to `/dev/null`, polled its memory every 0.5 s, and called the value "fits" if the process was still running after 100 s and "fails" if it aborted earlier. All nine failures were the `NvMap` allocation abort (return code −6) 4.6–15.1 s after start; there was no other kind of failure. Largest value that fits (the next probed value that failed in brackets):

| | `upstream` | `ov_on` | `spec` |
|---|---|---|---|
| FAST | 272 (288) | 176 (192) | 1024 (not reached: search limit) |
| HAC | 224 (240) | 176 (192) | 320 (336) |
| SUP (HAC build, SUP model) | 48 (64) | not tested | not tested |

Four points. (1) On 20,000 reads the limit is lower than the 1k grid suggested in every cell where both are known: `upstream` FAST 272 against 384 fitting at 1k, `upstream` HAC 224 against 256, `ov_on` 176 against 192 in both models, `spec` HAC 320 against 384. (2) The campaign's `-C 128` fits every FAST and HAC stage with at least 48 to spare (the tightest is `ov_on`, 176), so the campaign was not running near the edge. The preprint's `-C 200` fits `upstream` and `spec` but not `ov_on` (the 500k `spec` run at the preprint's flags completed, as expected). (3) SUP cannot use the campaign's 128 or even 64 on this board; 48 and 32 run. (4) `spec` on FAST has no practical limit in the range searched (1024 fitted; its peak memory grew from 5.29 GB at 512 to 5.98 GB at 1024), but that is not useful: the flush threshold of 64 caps the real batch, and the 1k sweep shows `-C` above 128 only costs time there. Caveats: "fits" means alive after 100 s, not run to completion (`c_limits.py --confirm` does that and was not run), so a value at the top of a range could still fail later in a long run; several fits were close to the edge (lowest available memory 41 MB for FAST `ov_on` at 176, 72 MB for HAC `spec` at 320, 94 MB for FAST `spec` at 1024); each value was probed once, so the limits are accurate to the 16-wide bracket and to whatever else held memory at the time (whether the desktop session was running was not recorded; the 2 November sweep had it running); and the peak memory of a failed probe is not comparable with that of a fit, because it aborted during start-up.

**Small `-C`** (`upstream`, 1k reads, mode A, one warm-up and two timed runs per value on 2026-11-04, joined to the 32–512 blocks of 2 November; sd ≤ 0.6 s). No small value failed; the cost is time, and it is large:

| `-C` | FAST (s) | against the best (`-C 128`) | HAC (s) | against the best (`-C 128`) |
|---|---|---|---|---|
| 8 | 70.52 | +408 % | not run | |
| 16 | 38.99 | +181 % | 142.54 | +189 % |
| 24 | 28.49 | +105 % | 101.19 | +105 % |
| 32 | 23.10 | +67 % | 86.41 | +75 % |
| 48 | 17.73 | +28 % | 66.81 | +35 % |
| 64 | 14.66 | +5.7 % | 56.03 | +13 % |
| 128 | 13.87 | | 49.40 | |

Below 64 the time is roughly inversely proportional to `-C`. For FAST, 7.5 s + 504 s / `-C` matches the measured values from 8 to 48 to within 1.5 % (the two constants were fitted to the points at 8 and 16; 24, 32 and 48 are the check). A fixed cost per batch would give that shape, but it was not decomposed. Peak memory of FAST `upstream` rises linearly with `-C`: 1.30 GB at 8, 2.59 GB at 128 and 5.40 GB at 384, about 11 MB per unit of `-C` on top of about 1.2 GB. There is therefore no lower limit in the sense of a failure. The lowest values tested were 8 (FAST) and 16 (HAC), and below those the cost grows too fast to be of interest.

**Chunk length `-c`** (`upstream` only; the value is rounded down to a multiple of the model stride, 6: 8000 → 7,998, 10000 → 9,996, 16000 → 15,996; change against the default 12,288):

| `-c` | FAST 1k (s) | FAST 20k (s, mode B, one run) | HAC 1k (s) | peak memory FAST 1k / HAC 1k |
|---|---|---|---|---|
| 6000 | 13.19 (−4.1 %) | 243.0 (−4.3 %) | 47.48 (−3.6 %) | 1.85 / 2.48 GB |
| 8000 | 13.34 (−3.0 %) | 247.2 (−2.6 %) | 47.90 (−2.8 %) | 2.09 / 3.23 GB |
| 10000 | 13.38 (−2.7 %) | 248.0 (−2.3 %) | 47.79 (−3.0 %) | 2.32 / 3.27 GB |
| 12288 (default) | 13.75 | 253.9 | 49.26 | 2.58 / 4.22 GB |
| 16000 | 15.10 (+9.8 %) | 275.6 (+8.5 %) | 52.61 (+6.8 %) | 3.02 / 5.05 GB |

Time falls as chunks get shorter and rises steeply above the default; memory is roughly linear in `-c`. The paper's `-c 10000` is 2.3–3.0 % faster than the default in all three cells, which is the whole timing effect of that one flag in these data. A plausible part of the trend is wasted work on the last, partly empty chunk of each read: in the 500k `spec` runs, chunks drafted × chunk length exceeds the number of input samples by 12.0 % at `-c 12288` (2,561,417 chunks) and by 10.0 % at `-c 10000` (3,092,266 chunks), i.e. 1.8 % less work against 0.9 % less time. That does not explain the rise at 16,000, where inference time itself grows (FAST 1k: 6.4 s at 6000, 6.9 s at 12,288, 7.8 s at 16,000); the cause was not investigated. Chunk length also changes the signal at chunk boundaries, so a time gain is only usable if accuracy holds. Median identity of the same outputs (PC, `minimap2` against hg38noAlt):

| `-c` | FAST 1k | FAST 20k | HAC 1k |
|---|---|---|---|
| 6000 | 0.939817 | 0.939377 | 0.977625 |
| 8000 | 0.940598 | 0.939437 | 0.977249 |
| 10000 | 0.940541 | 0.939419 | 0.977219 |
| 12288 (default) | 0.940763 | 0.939375 | 0.976852 |
| 16000 | 0.940270 | 0.939270 | 0.976852 |

On the 20,000-read set, the cell with the most reads, all five values lie within 0.017 pp of each other and the shorter chunks are not worse (`-c 10000`: +0.004 pp against the default). The 1,000-read cells move more (up to 0.095 pp for FAST at 6000, 0.077 pp for HAC), which is probably sampling noise (not measured); HAC 20k was not run, so the small rise in HAC identity with shorter chunks (+0.04 pp at 10,000) is not established. The `-c 12288` outputs reproduce the campaign's `upstream` medians exactly (0.940763 / 0.939375 / 0.976852). So on `upstream`, `-c 10000` buys 2.3–3.0 % of time with no measurable change in identity. The campaign kept the default, so none of its comparisons are affected. Whether the same holds under `spec` was not tested; its 500k run at the preprint's flags was 0.9 % faster than at the campaign's, with median identity 0.934641 against 0.934579. **[check: `spec` at `-c 10000` on 1k and 20k if a recommendation for `spec` is wanted]**


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

### E.5 Intermittent GPU hangs: cause, fix and the 500k `spec` result
Thirteen runs across the campaign, its follow-ups and the later split run hung and were killed by the harness. Eight in the campaign and follow-ups: `lb_baseline`, `cb_off` and `ov_on` on FAST 20k (each recovered on retry) and `spec` on FAST 500k, which hung on all five attempts (after 20–80 % of the reads, at different positions each time) including a run at a smaller read batch (`-K 2000`). Every hang had the same signature: power falls from about 13–16 W to about 6 W, the CPU idles, and the GPU load counter reads 99.9 % with no output. Those eight all occurred with overlap decode enabled; none on HAC. **The later split run (D.1) then produced five more hangs on DNA FAST 20k with `--overlap-decode=no`** (2 of 6 attempts at `7cec6c5`, 2 of 5 at `e00bada`, 1 of 5 at `9f8db24`), after 0–3 progress lines, with the same silence, while `upstream` and the instrumentation-only commit had none in 8 attempts. So the hang is not specific to overlap; it appears in the `e00bada`-and-later code lineage whether or not overlap is switched on. Under `gdb`, one hang was captured: the single active thread was blocked in `cudaStreamSynchronize` on the decode/repair stream at `src/basecall.cpp:874`, waiting on a GPU fence that never signalled. **Cause.** The common factor was the newer Openfish that the fork took up at `e00bada` (`upstream` and `paper_xavier` use the older pin and had no stall in 8.6 h of running). Its beam-search kernel, rewritten by the maintainers in June 2026 (openfish `1f26d77`, "optimise beam search"; the same code is still in openfish's `dev` branch as last fetched on 2026-11-03), has a missing barrier: every thread reads a shared flag to decide whether to leave the cut-off search loop, and when the loop ends at its cap of ten guesses thread 0 rewrites the flag with no barrier in between. A thread that reads late stays in the loop, the block's barriers fall out of step, and a stray thread can spin for ever, which fits the signature (load counter 99.9 %, power at idle level, host waiting in `cudaStreamSynchronize`). The same pattern is in the HIP kernel. In an isolated test (one real chunk tiled to 128 identical chunks, decoded repeatedly, no Slorado, no streams) the original kernel hung or decoded one chunk differently in 12 of 40 runs; with one added `__syncthreads()` none of 340 runs did (`exp/hang-root-cause.md`, `exp/barrier-fix.patch`). The defect probably also explains the rare reads that differ from `upstream` in stages that should be exact (1 of 20,000 for `ov_on` on FAST, 2 of 20,000 for `ov_off` on HAC; not proven). It does not affect the timings in this chapter: hung runs were killed, repeated and excluded as failures. A separate latent hazard in the overlap repair path (no `record_stream`, `src/basecall.cpp:856`) was not needed to explain the hangs and has not been tested.

**500k `spec`.** With the fix (`x_barrier`: `49ac641` plus the barrier; its 1k output is read-for-read identical to the campaign's `spec` and its time is unchanged, 9.10 s against 9.106 s) the 500k run completed on its first attempt in 3,970 s (D.4c), and a second 500k run at the preprint's flags also completed on its first attempt (3,935 s); before the fix all five harness attempts stalled, between 1,250 s and 3,500 s into the run. Two completed runs (about 2.2 h of running with the fix, none stalled) are consistent with the fix but are not proof of it; the isolated test is. An earlier run that completed under `gdb` in 3,991 s was not a valid measurement and is not tabulated. The batch-width sweep of D.4f also ran the unfixed `ov_on` and `spec` builds for 0.9 h (1k reads, 100 runs) without a stall; its only failures were out-of-memory aborts. That is short against the 14 stalls in about 29 h that these builds showed on the longer cells, so it neither supports nor contradicts the fix.

---

## Limitations and threats to validity
- **Hardware and platform:** a single Jetson Orin Nano in one power mode; power cannot be split between GPU, CPU and DRAM. A sensitivity run at MAXN_SUPER with `jetson_clocks` on (D.4e) kept the stage ordering and shrank the `spec` gain by 1.5–3.8 points; it has n = 3 (1k) and n = 1 (20k), and no 500k run.
- **Run counts:** n = 5 (1k) and n = 3 (20k) per mode; the 500k results are single runs; the `ov_on` 20k FAST cell has n = 2 after excluding two outliers (replicate n = 3 agrees within 1.6 %).
- **Measurement disturbances:** desktop activity on the Jetson caused outliers in an earlier validation run; the campaign was run over SSH, but one run coincided with a system-monitor launch (excluded by the outlier rule, with two others; `excluded-runs.json`). Validation criteria not met on the campaign data: run-time CV above 1 % in 4 of 94 groups (three because of the outliers), process-exit gaps above 3 s in five HAC 20k runs (max 29 s), one idle shoulder +0.79 W; none affects `Real time` or the energy window.
- **Baseline pairing:** `ov_off` ↔ `ov_on` and `lb_baseline` ↔ `lb_best` are flag-only comparisons of one binary; `lb_best` → `spec` compares two commits (a same-binary `--speculative-decode=no` run is the clean check); `upstream` → `ov_off` was split by commit: it is entirely the stream-aware-decode / fork-openfish commit `e00bada` (instrumentation +3 %, pinned buffers 0 %), but that commit is not further separable, so it is not attributable to a single method; upstream's current `dev` is not faster than the base release (D.1, D.4d).
- **Accuracy method:** median identity over all alignment lines (including supplementary alignments), against hg38 / GENCODE v40, following the preprint; relative error rates are not reported.
- **Withdrawn data:** the earlier pre-overlap stage (`8873209`) decoded garbage because of a score-layout mismatch and is excluded from every comparison.
- **Speculative decode on RNA FAST** is reported as a negative result with DNA-tuned thresholds; re-tuning was explored only on 1k reads.

## Reproducibility
Scripts, configuration and the frozen-harness record: `docs-riley/bench/`, `HARNESS-FROZEN.md`; stage builds `~/bench-stages/` (commits in `stages.json`); run data `bench-runs/runs.jsonl` (528 records), `runs.csv` (377 timed runs after exclusions), accuracy CSVs, hang captures; analysis `analyze.py csv|tables|outliers|validate|equiv`.
The D.4e sensitivity runs use `exp/clocks_ab.py` / `exp/run_clocks_ab.sh` and `exp/maxn_sens.py` / `exp/run_maxn_sens.sh` (the frozen harness's measurement functions without its pre-flight); data in `bench-runs/clocks-*.jsonl` and `bench-runs/maxn-*.jsonl`. The hang analysis and the 500k `spec` run use `exp/hang-root-cause.md`, `exp/barrier-fix.patch`, `exp/beam_race_stress.py` and `exp/run_fixed_500k.py` (records in `bench-runs/variants-fix500k.jsonl` and `variants-fix500k-paperflags.jsonl`, accuracy in `bench-runs/accuracy-variants/`). The D.4f sweeps use `exp/c_sweep.py` (`--flag C|c`) and were started with `exp/queue_after_500k.sh` (log `exp/queue.log`); records in `bench-runs/csweep-20261102T142917.jsonl` (`-c`, 1k), `csweep-20261102T150956.jsonl` (`-c`, FAST 20k) and `csweep-20261102T153435.jsonl` (`-C`); kept FASTQs in `bench-runs/kept-sweeps/`. `python3 exp/c_sweep.py --summary` prints the tables. The `-C` limits and small values (2026-11-04) use `exp/c_limits.py` and `exp/c_sweep.py`, started by `exp/queue_limits.sh` (log `exp/queue_limits.log`); records in `bench-runs/climits-20261104T155822.jsonl` (all probes), `csweep-20261104T165506.jsonl` (FAST, `-C` 8–48) and `csweep-20261104T170716.jsonl` (HAC, `-C` 16–48); `python3 exp/c_limits.py --summary` prints the limits table. **[check: `docs-riley/exp/` is untracked; commit these scripts and notes (not the decode probes) if the report cites them]**
