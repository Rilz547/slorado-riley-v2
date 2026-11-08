# Decode optimisation feasibility triage (2026-10-30)

**Status:** probe results only. Nothing here is a shipped change or part of the final campaign; no commits. Experiment tree: `~/bench-stages/exp_src` (49ac641 + probes, patch in `probes.patch`), binaries `~/bench-stages/x_*`.
**Setup:** `spec` configuration (q10_m2_brave), DNA FAST 1k, mode A, 5 interleaved reps per variant through the frozen harness (`variant_runs.py --set exp`, `bench-runs/variants-exp.jsonl`). Control `spec_on` = 9.118 ± 0.033 s. Noise: sd about 0.03 s, so differences below about 0.1 s (1 %) are not resolvable.
Probes: `dup_X` launches kernel X a second time (outputs unchanged, verified byte-identical to `kept/output_fast_1k-dna-spec_on.fastq`); `skip_repair_scans` skips bwd_scan + fwd_post_scan in the repair call (timing only, output wrong); `pack32` = R7.

## Raw results (80 batches per run)
| variant | wall (s) | vs control | extra event time of that kernel (s) | wall per extra event-second |
|---|---|---|---|---|
| spec_on (control) | 9.118 ± 0.033 | | | |
| x_sm87 (R0a) | 9.097 ± 0.031 | −0.2 % | | noise |
| x_probe (probe build, none active) | 9.104 ± 0.029 | −0.2 % | | noise (build is neutral) |
| dup_beam | 11.261 | +23.5 % | +2.90 | 0.74 |
| dup_fwd (fwd_post_scan) | 10.334 | +13.3 % | +1.41 | 0.86 |
| dup_bwd | 9.679 | +6.2 % | +0.59 | 0.95 |
| dup_draft (greedy) | 9.455 | +3.7 % | +0.29 | 1.15 |
| dup_qual | 10.031 | +10.0 % | +0.81 | 1.13 |
| dup_gen | 9.793 | +7.4 % | +0.54 | 1.26 |
| pack32 (R7) | 10.080 | +10.5 % | | negative |
| skip_repair_scans (R1 ceiling) | 9.113 | −0.1 % | −1.05 (bwd+fwd, repair launches); host-side repair wait 3.75 s -> 0.53 s | **0** |

## Reading
- Kernels in the **draft path** (bwd, greedy, fwd, qual, gen) convert to wall time about 1:1 (0.9-1.3). They sit on the critical path.
- The **repair path has slack**: removing 1.05 s of GPU work and 3.2 s of host wait from it changed nothing (−0.1 %, inside noise), while adding a whole extra beam launch costs 0.74 s of wall per second. So speeding up beam_search (R2) or the repair scans (R1) will not pay in this configuration unless the repair path becomes the limit.
- nsys (`exp/nsys/`) shows 96 % of decode GPU time overlaps inference, but overlap in time is not the same as being off the critical path; the probes are the better evidence.
- R0a: native sm_87 code gives no steady-state speed-up (−0.2 %, noise). It removes about 1 s of one-off JIT compile when the kernel cache is cold (9.0 s vs 10.0 s with `CUDA_CACHE_DISABLE=1`). Output byte-identical.

## Verdicts
| item | verdict | why |
|---|---|---|
| R0a sm_87 | adopt for cold start only; no speed claim | no steady-state effect |
| R1 repair without recomputing scans | **no-go** | measured ceiling 0 % |
| R2 beam barriers | **no-go for this config** (revisit if inference gets faster) | repair path has slack |
| R7 pack32 | **no-go** | +10.5 % slower |
| R5 greedy draft | skip | whole draft kernel 2.7 ms/batch; halving it is about 1.5 % |
| R3 bwd_scan | skip | about 1 % at best |
| R4 fwd_post_scan barrier trim / fusion | **drop** | same mechanism as A |
| new A: compute_qual_data parallel over timesteps | **built and measured: no gain** (see below) | kernel event time 0.83 -> 0.20 s (FAST), 0.96 -> 0.42 s (HAC), output byte-identical, wall +0.2 % (noise) |
| new B: generate_sequence parallel (prefix sum over moves) | **drop** (same mechanism as A) | 3.2 ms/launch, single thread; accumulation order per position must be preserved |
| new C: concurrent draft (own stream + gpubuf + 4-slot host ring) | **drop** unless the pipeline stops being inference-bound | draft chain is serialised after the repair sync because gpubuf is exclusive |
| R6 jetson_clocks | **measured: small** (see below) | FAST 1k -3.5 %, HAC 1k -2.0 % (within drift), no energy saving |
| R0b record_stream fix | not yet done | correctness fix; hang effect unknown |

## Result of the compute_qual_data prototype (the test that overturned the sensitivity estimates)
`compute_qual_data_par` (128 threads per chunk, timesteps spread across threads; identical arithmetic) vs the original kernel, FAST 1k, 7 interleaved reps:
spec_on 9.106 +- 0.025 (n=12), unchanged probe build 9.100 +- 0.037, qual_par **9.125 +- 0.023 (+0.2 %)**. The kernel got 4x faster (event time 0.83 -> 0.20 s) and the host-side repair wait fell from about 4.1 s to 0.7 s, yet wall time did not move.

## What this means
- Adding decode work slows the run (the dup_* probes), but removing decode work does not speed it up (skip_repair_scans, qual_par). The decode chains already fit inside the time inference takes, so the run sits just on the inference-bound side of the knee: shortening them gains nothing, lengthening them eventually spills over.
- Consistent with the nsys timeline: the inference side (conv + LSTM + GEMM + elementwise, all streams except decode) is busy about 7.5 s of the 8.5 s steady state, about 88 %.
- So decode-kernel optimisation (R1-R5, qual/gen parallelisation, concurrent draft) cannot help DNA FAST 1k in the final configuration. The lever is inference (the persistent LSTM kernels are about 3 s of the 7.5 s) or the GPU clock.
- Not yet tested: the GPU clock (R6), which speeds inference directly, and the R0b correctness fix.
- Caveats: one cell and one configuration (DNA FAST 1k, spec). A 20k run and HAC were not probed; HAC is even more inference-bound (inference 41 of 45 s).

## R6 jetson_clocks A/B (2026-10-31, `run_clocks_ab.sh`, `clocks_ab.py`; spec configuration, mode A, 5 runs per cell and block; blocks off -> on -> off2)
| cell | off | on | off2 |
|---|---|---|---|
| DNA FAST 1k time | 9.140 ± 0.047 s | **8.817 ± 0.020 s (-3.5 %)** | 9.106 ± 0.018 s (-0.4 %) |
| mean GPU clock | 824 MHz | 918 MHz | 824 MHz |
| mean power / energy | 13.86 W / 126.7 J | 14.69 W / 129.5 J (+2.2 %) | 13.89 W / 126.5 J |
| DNA HAC 1k time | 45.685 ± 0.254 s | 44.783 ± 0.332 s (-2.0 %) | 45.042 ± 0.128 s (-1.4 %) |
| mean GPU clock | 891 MHz | 918 MHz | 895 MHz |
| mean power / energy | 16.87 W / 770.7 J | 17.23 W / 771.8 J | 17.04 W / 767.3 J |
Idle power: 5.16 W off, 6.96-7.04 W on. Canary 158.9-164.1 ms in all blocks. tj max about 60 C in all blocks. Over-current event counter (`/sys/class/hwmon/hwmon3/oc3_event_cnt`): 3107 before block 2, 9396 after block 3 (the increase cannot be attributed to a single block).
Reading: pinning the clocks (GPU 824 -> 918 MHz, +11 %, plus CPU and memory at maximum) buys 3.5 % on FAST and about 2 % on HAC (the HAC difference is no larger than the 1.4 % between the two clocks-off blocks). Energy per run is unchanged to slightly higher and idle power rises by 1.8 W. Time falls by about a third of the clock increase, consistent with the pipeline not being purely GPU-clock-bound. Not worth recommending as an optimisation; MAXN_SUPER (GPU max 1020 MHz, +11 % over 918) would be expected to add at most a similar small amount. If the Xavier preprint runs had jetson_clocks on, the effect on the 500k comparison would be of this size (a few percent).
