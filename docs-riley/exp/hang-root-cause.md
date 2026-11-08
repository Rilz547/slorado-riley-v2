# Root cause of the GPU hangs: missing barrier in the openfish beam-search kernel

**Status 2026-11-02:** cause found and reproduced in isolation (hang and wrong output in seconds); a one-line fix removes both in the isolated test. **First full 500k run with the fix completed on attempt 1** (`x_barrier`, `run_fixed_500k.py`): Real 3,970.1 s (66.2 min), 500,000 reads written, accept rate alpha 0.727, mean 15.74 W, 62.5 kJ, canary 159.6 -> 161.0 ms, idle 5.13 -> 5.21 W (all drift gates pass); records `bench-runs/variants-fix500k.jsonl`, FASTQ `bench-runs/kept/output_fast_500k-dna-x_barrier.fastq`. Median identity 0.934579 (PC; -0.455 pp against upstream 500k 0.939130, -0.452 pp at 20k; `bench-runs/accuracy-variants/`). A second 500k run at the preprint's flags (`--tag paperflags`: `-t 8 -B 500M -K 2000 -c 10000 -C 200`) also completed on attempt 1: 3,935.3 s, 15.70 W, 61.8 kJ, 500,000 reads, accept rate 0.725, canary 159.4 -> 163.4 ms, idle 5.17 -> 5.20 W (`bench-runs/variants-fix500k-paperflags.jsonl`; FASTQ `bench-runs/kept-sweeps/`). Two successes are consistent with the fix but are not proof by themselves (before the fix 1 of 7 attempts completed, and the earlier stalls all fell between 1,250 s and 3,500 s, before this run's end); the isolated test (12/40 bad runs against 0/340) is the proof, and the queued runs add exposure. Experiment only: nothing committed, campaign data untouched.

## The defect
`openfish/src/beam_search_cuda.h`, in the beam-width cut-off search (introduced upstream in BonsonW's "optimise beam search", 2026-06-22):

```
while (true) {                    // loop at :486
    if (tid == 0) { ... bs_active = 1 or 0 ... }
    __syncthreads();
    if (!bs_active) break;        // :505  every thread READS bs_active
    ... recount, 3 more barriers ...
}
if (tid == 0) {                   // :534  tid 0 WRITES bs_active again, with no barrier in between
    bs_active = (num_guesses == MAX_GUESSES) ? 1 : 0;
    ...
}
```
When the loop ends because the 10-guess cap was reached (`num_guesses == MAX_GUESSES`), tid 0 leaves the loop, then immediately writes `bs_active = 1`. A warp that reads `bs_active` later than tid 0's write sees 1, does not leave the loop, and its `__syncthreads()` calls now pair with the wrong barriers of the other warps. The block desynchronises; a stray warp can spin in the loop for ever (nobody updates `bs_active` any more). The same pattern is in `beam_search_hip.h` (:511 read, :541 write).

Fix (`barrier-fix.patch`): one `__syncthreads()` between the end of the loop and the `if (tid == 0)` write.

## Why this matches everything seen
| observation | fits because |
|---|---|
| power falls to ~6 W (idle is 5.2 W) while the GPU load counter reads 99.9 %, host waits in `cudaStreamSynchronize` | a few blocks spinning on otherwise idle SMs, not a device-wide deadlock |
| only builds from `e00bada` on hang (14 stalls in about 29 h); `upstream` and `paper_xavier` have 0 in 8.6 h (chance if the rate were equal: about 1.5 %) | `e00bada` moved openfish from `2530c2a` (upstream's pin, before the June rewrite) to the fork lineage containing the parallel beam search (also the source of the 17 % speed-up) |
| hangs also with `--overlap-decode=no` | the defect is inside one kernel, not in stream handling |
| hang position random (batch 37, 56, 76, 99, 100, 147 of 123-150) | depends on warp timing |
| rare wrong reads: FAST 20k `ov_on` 1/20000, HAC 20k `ov_off` 2/20000 differ from `upstream` | a desynchronised block that finishes produces wrong output (consistent, not proven) |
| the cap path is not rare | instrumented counter: with the repo's test scores the loop ended at the cap 18 times in one 1,666-timestep chunk |

## Isolated reproduction (no slorado, no streams)
`beam_race_stress.py <openfish binary> 128 <runs>`: one real chunk of scores tiled to 128 identical chunks, decoded repeatedly; any run that hangs (>15 s, normal 0.5 s) or in which a chunk differs from chunk 0 is bad. Standalone openfish built from a clean copy with `make cuda=1 debug=1 CUDA_ARCH="-gencode arch=compute_87,code=sm_87"`.

| kernel | runs | hung | chunk disagrees | bad runs |
|---|---|---|---|---|
| original | 12 | 3 | 1 | 4 |
| original | 40 | 8 | 4 | 12 (30 %) |
| with the barrier | 40 | 0 | 0 | 0 |
| with the barrier | 300 | 0 | 0 | 0 (about 690,000 cap-path events) |

Chance of 0 bad in 340 runs if the fix did nothing: below 1e-60.

Caveat: the repo's test blob holds 512 values per timestep while `main.c` reads 256, so the scores are real model output read with a shifted layout (still model-like; it exercises the cap path). The CUDA sanitizer cannot be used on the Jetson ("GPU debugging features are disabled", needs root); `compute-sanitizer --tool racecheck` on a desktop GPU would confirm the hazard independently.

## In slorado
`~/bench-stages/x_barrier/slorado` = `49ac641` (`spec`) + the barrier (built by `build_exp.sh barrier`, 10 s). FAST 1k with the `spec` flags: reads identical to the campaign's kept `spec_on` output (1000/1000, twice); `Real time` 9.10 s (campaign `spec` 9.106 +/- 0.025 s), so no measurable cost.

## What it does not explain
SUP overlap decode (`sup_stage_probe.py`): `ov_on` still disagrees with `upstream` (65/100 and 58/100 identical, 46/100 between repeats) **with the barrier in place**. That is a separate defect (data hazard between decode and inference); SUP `ov_off` is exact. Open.

## Consequences for the campaign
- Timings: unaffected (hung runs were killed, recorded as failed and repeated; the barrier adds no measurable time).
- Accuracy: affected only through the rare wrong reads (at most a few reads in 20,000); medians unchanged to the reported precision.
- `spec` 500k never completed under the frozen harness (5 of 5 attempts stalled); the fixed binary is the way to get that number, reported as a separate labelled variant (`x_barrier`, same flags), with the number of hung attempts stated.
- Worth reporting upstream (BonsonW/openfish): CUDA and HIP kernels, fix above.

## Files (all untracked)
`exp/barrier-fix.patch` (apply inside `openfish/`), `exp/beam_race_stress.py`, `exp/run_fixed_500k.py`, `exp/build_exp.sh`, binary `~/bench-stages/x_barrier/slorado`. `exp_src/openfish` now holds the barrier fix only; the earlier probe state is `exp/probes+qualpar.patch`.
