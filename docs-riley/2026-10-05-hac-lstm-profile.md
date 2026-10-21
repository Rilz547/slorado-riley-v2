# HAC inference profile — where the LSTM time goes

**Date:** 2026-10-05
**Platform:** Jetson Orin (8 SMs, CC 8.7, 2 MB L2, ~7.6 GB shared), `nvpmodel` 25W, `jetson_clocks` OFF
**Software:** slorado `speculative-decode` @ `49ac641`, torch 2.11.0, CUDA 12.6, cuDNN LSTM (fp16)
**Model / data:** `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`, `reads_1k.blow5`, flags `-C 128 -c 12288 -K 4096 -p 150`
**Files:** `docs-riley/lstm-profile/` (scripts, raw nsys/ncu data)

---

## 1. Question

HAC is ~83 % LSTM by wall time and barely responds to the decode-side optimisations (overlap,
narrow batches, speculative decode). Is the LSTM limited by **tensor-core compute**, **memory
bandwidth**, or **per-step latency / launch overhead**? That decides whether quantisation (int8) or
a different kernel structure is the right next step.

## 2. Short answer

- The LSTM is **not tensor-core-compute-bound**: the per-step recurrent GEMM runs at ~5 TFLOP/s, about a third of the (assumed) ~15 TFLOP/s peak. By contrast the one-shot input GEMM reaches ~9.6 TFLOP/s on the same device.
- **~30 % of every LSTM layer is spent in a gate/cell elementwise kernel that does almost no work** (`elemWiseRNNcell`, 23 µs per step at both N = 2 and N = 128; ~3 % of compute/memory throughput in ncu at N = 2).
- Throughput plateaus at ~3.9 TFLOP/s per layer for N ≥ 128, with a fixed ≈ 25–28 µs per-step floor.
- Int8 quantisation would only address the GEMM part (~60 % of layer time) and none of the cell-kernel or launch overhead. A fused/rewritten cell update is the more direct target, but neither is built.

## 3. Setup facts

| item | value | how obtained |
|---|---|---|
| GPU clock cap (25W mode) | 918 MHz (idles at 306 MHz with DVFS) | `/sys/class/devfreq/17000000.gpu` |
| SMs / L2 | 8 / 2 MB | torch device properties, CUDA runtime |
| Assumed fp16 tensor peak | ≈ 15.0 TFLOP/s | **assumption**: 8 SMs × 2048 FLOP/clk/SM × 0.918 GHz; not read from the device |
| DRAM peak | not established | CUDA runtime `memoryClockRate` returned a value that gives an implausible 29 GB/s; documented Orin Nano figures are 68–102 GB/s |

The peak figure is only used for context. Nothing in the conclusions depends on its exact value, but
the ih-GEMM figure of 9.6 TFLOP/s (64 % of it) is a consistency check that the peak is not far off.

## 4. Experiment A — standalone cuDNN LSTM scaling

`lstm-profile/lstm_scaling.py`: one `torch.nn.LSTM(384, 384)` layer, fp16, cuDNN, after an 8 s warm-up so the GPU
reached 918 MHz. Median of 7 runs.

**Sanity check:** N = 128, T = 2048 takes **159.6 ms** standalone, versus **159 ms per layer-batch** measured inside slorado
(6.35 s per layer / ~40 batches). The benchmark represents the pipeline.

### Batch scaling (T = 2048)

| N | time (ms) | µs / step | TFLOP/s | ms / chunk |
|---|---|---|---|---|
| 8 | 58.0 | 28.3 | 0.67 | 7.25 |
| 16 | 69.3 | 33.8 | 1.12 | 4.33 |
| 32 | 80.6 | 39.4 | 1.92 | 2.52 |
| 64 | 95.7 | 46.7 | 3.23 | 1.50 |
| 96 | 144.1 | 70.4 | 3.22 | 1.50 |
| 128 | 159.6 | 77.9 | 3.87 | 1.25 |
| 160 | 209.4 | 102.3 | 3.69 | 1.31 |
| 192 | 236.4 | 115.4 | 3.93 | 1.23 |

### Sequence-length scaling (N = 128)

| T | time (ms) | µs / step | TFLOP/s |
|---|---|---|---|
| 256 | 20.8 | 81.4 | 3.71 |
| 512 | 42.1 | 82.2 | 3.67 |
| 1024 | 80.1 | 78.3 | 3.86 |
| 2048 | 159.3 | 77.8 | 3.88 |
| 4096 | 316.5 | 77.3 | 3.91 |

**Reading:**
- Cost per step is constant in T (77–82 µs): no startup effects, pure per-step cost.
- A crude two-point fit of the batch scan gives ≈ 28 µs fixed per step plus ≈ 0.47 µs per chunk per step.
- For N ≥ 128 throughput is flat at ~3.9 TFLOP/s, so adding chunks does not make it more efficient.
- N = 96 is an outlier (worse than N = 64 per chunk), probably tile alignment; N = 128 is a good size.

## 5. Experiment B — nsys kernel breakdown of a real run

`nsys profile -t cuda` on HAC 1k (wall 46.1 s under nsys; 44.8 s un-profiled). 40 batches × 5 layers = 200 layer-batches, each 2048 steps → 409 600 steps. Raw data: `lstm-profile/kern.csv`.

| kernel | launches | avg | total | % GPU kernel time | per layer-batch |
|---|---|---|---|---|---|
| recurrent GEMM `ampere_fp16_s16816gemm_128x64` | 409 600 | 31.2 µs | 12.76 s | 29.0 % | 63.8 ms |
| **cell update `elemWiseRNNcell`** | 409 600 | **23.3 µs** | **9.56 s** | **21.7 %** | **47.8 ms** |
| input GEMM `ampere_fp16_s16816gemm_128x128` | 12 800 | 504 µs | 6.45 s | 14.6 % | 32.3 ms |
| (sum of the three) | | | 28.77 s | 65.3 % | 143.9 ms |
| other (conv, flips, copies, decode, gaps) | | | | | 15.8 ms of the 159.6 ms layer time is unaccounted gaps |

Total kernel time is 44.08 s, i.e. the GPU is kept busy ~96 % of the run; the cost is the tiny kernels themselves, not an idle GPU.

Per layer-batch (159.6 ms): recurrent GEMM 40 %, cell kernel 30 %, input GEMM 20 %, gaps ~10 %.

Derived rates: input GEMM 309 GFLOP in 32.3 ms ≈ **9.6 TFLOP/s**; recurrent GEMM 151 MFLOP in 31.2 µs ≈ **4.8 TFLOP/s** per step.

## 6. Experiment C — ncu speed-of-light (first pass)

> **Batch-size caveat for both ncu passes (6 and 6b).** They profiled `reads_1.blow5`, which is one read = **2 real chunks**.
> Without `--fixed-c-batch=yes` slorado narrows the forward pass to the real chunks (`call_chunks`, `src/basecall.cpp:511-515`),
> so these kernels ran at **N = 2, not N = 128**. (The "126 padded slots" line in slorado's output is a statistics counter, not the
> launch width.) Evidence: the profiled recurrent GEMM is a 32×32-tile kernel with 48 blocks, whereas the real N = 128 run uses a
> 128×64-tile GEMM (nsys, section 5). GEMM-specific ncu figures below (grid size, tensor-pipe %, L1/L2 %, occupancy) therefore
> describe a small-batch kernel variant and are **not representative of the production batch**. The cell-kernel observation survives:
> it takes ~22 µs at N = 2 (ncu) and 23.3 µs at N = 128 (nsys), i.e. its time is independent of batch size. A re-run at N = 128 is in section 10.

`lstm-profile/ncu-sol-first-pass.txt` — one 1-read run, 40 kernels, `--section SpeedOfLight`. **ncu locked the SM clock to ~406 MHz**, so absolute durations are inflated; only the percentages and the clock-dependence below are used.

| kernel | grid | duration @406 MHz | memory thr. | compute (SM) thr. |
|---|---|---|---|---|
| recurrent GEMM (`cutlass…gemm_f16_32x32`) | 48 blocks (0.8 waves) | 56 µs | 58 % (L1/TEX 68 %, L2 58 %) | ~20 % |
| `elemWiseRNNcell` | 3 blocks (0.1 waves) | 22.8 µs | 3.7 % | 3.1 % |

Clock-dependence check:
- `elemWiseRNNcell`: 22.8 µs at 406 MHz vs 23.3 µs at 918 MHz (nsys) → **does not speed up with clock** (latency-bound).
- recurrent GEMM: 56 µs at 406 MHz vs 31 µs at 918 MHz → **scales with clock** (SM / on-chip-throughput-bound).

ncu itself flags both grids as too small to fill the GPU. The DRAM-throughput figure was not printed in this section, so DRAM utilisation is **not measured**.

## 6b. Experiment C2 — ncu metrics, second pass (`--clock-control none`)

`lstm-profile/ncu-clean-pass.txt` — 1-read run, 40 kernels (20 recurrent GEMMs, 20 cell kernels), metrics-only. Means over the samples (min/max within ~2 % of the mean):

| metric | recurrent GEMM (`cutlass…32x32`, 48 blocks) | `elemWiseRNNcell` (3 blocks) |
|---|---|---|
| duration (µs) | 54.4 | 22.3 |
| L1/TEX throughput (% of peak) | 57.7 | 1.0 |
| L2 throughput (% of peak) | 57.6 | 3.7 |
| tensor-pipe (HMMA) active (% of active cycles) | **13.2** | 0.2 |
| warps active (% of peak occupancy) | 54.7 | 25.8 |
| DRAM throughput | **n/a** (not exposed on this iGPU) | n/a |

**Caveats for this pass:**
- DRAM utilisation **cannot be measured with ncu on this Orin** (`dram__throughput… (!) n/a`), so a DRAM-bound explanation is neither confirmed nor ruled out. Cache-level numbers (L1/L2) are the closest available.
- The GEMM duration (54 µs) matches the first pass's locked-clock figure rather than the 31 µs seen in nsys at 918 MHz, so the GPU probably stayed at a low clock while ncu replayed kernels (device idled at 306 MHz when checked). Percentages are relative to elapsed cycles and remain usable; absolute times are not.

**What it adds:** the recurrent GEMM uses the tensor pipe only ~13 % of its active cycles while L1 and L2 sit at ~58 %, and occupancy is ~55 % (48 blocks on 8 SMs). The cell kernel has ~26 % occupancy, ~1 % L1 and ~4 % L2 utilisation: it is nearly idle hardware for 22 µs.

## 7. Interpretation

1. **Not tensor-compute-bound (supported at N = 128 only by the nsys rate).** The recurrent GEMM reaches ~4.8 TFLOP/s in the real run (nsys), about a third of the assumed peak. The ncu 20 % compute / 13 % tensor-pipe figures are from the N = 2 kernel variant and are provisional.
2. **The cell kernel is pure overhead.** An elementwise kernel that takes ~23 µs at N = 2 and at N = 128 alike, and regardless of clock, at ~3 % utilisation in ncu (N = 2), accounts for ~30 % of layer time. Its duration not growing with batch size is strong evidence it is latency/launch-bound.
3. **The recurrent GEMM is not tensor-rate-bound; what limits it at N = 128 is unresolved.** For the N = 2 variant ncu shows L1/TEX and L2 at ~58 %, tensor pipe ~13 %, occupancy ~55 % and only 48 blocks on 8 SMs; the first-pass clock comparison (56 µs at 406 MHz vs 31 µs at 918 MHz) suggests it scales with SM clock. None of that has been measured at N = 128, and DRAM cannot be measured with ncu on this chip.
4. **The input GEMM is already efficient** (~64 % of assumed peak); it is not where the time is lost.

### What each optimisation could do (estimates, not measurements)

| idea | what it touches | ceiling (share of one layer) | notes |
|---|---|---|---|
| Replace/fuse the cell update | cell kernel, 47.8 ms (30 %) | −30 % if free; **≈ −23 %** if replaced by a ~5 µs kernel | would be ≈ −20 % of HAC wall (LSTM ≈ 84 % of runner time). Needs a custom per-step loop instead of cuDNN's RNN. The repo already has a per-step "addmm + fused gate kernel" structure for factored LSTMs (`flstm_layer_forward`, `openfish_flstm_step_gpu`) on the `cpu-beam` lineage |
| int8 GEMMs | recurrent + input GEMM, 96 ms (60 %) | ≤ −30 % only if both GEMMs halve | unlikely to be reached: recurrent GEMM is not tensor-rate-bound; needs new int8 LSTM kernels, calibration, accuracy testing |
| Larger batch | amortise fixed step cost | none available | N = 256 runs out of memory (peak 5.0 GB at N = 128); throughput is flat for N ≥ 128 anyway |

## 8. Caveats

- Both ncu passes ran at **N = 2** (see the caveat at section 6), not the production N = 128, and are one read / 40 kernels at low GPU clocks (~406 MHz in the first, probably similar in the second); DRAM utilisation cannot be captured by ncu on this device, so DRAM-boundness is untested.
- Peak FLOP/s is an assumption; DRAM peak not established.
- The ceilings above assume the replacement kernel would cost what is stated; no replacement was built or measured.
- cuDNN's internal algorithm choice is not controllable from slorado, so the cell-kernel cost is only removable by replacing the RNN call.
- Single run of each measurement; the standalone benchmark and the in-pipeline number agree within 0.3 %.

## 9. Corrections to earlier reasoning in this session

- Early on I suggested int8 would help by halving memory traffic (bandwidth-bound LSTM). The measurements do not support bandwidth as the primary limit.
- I then described the LSTM as latency/launch-bound. That was only partly right: at N = 128 the recurrent GEMM is throughput-limited on-chip; the clean latency-bound piece is the cell kernel.

- I initially described the ncu passes as covering a full 128-wide batch. They did not (N = 2); the document now says so, and the GEMM-level ncu figures are provisional.

## 10. Reproduce

```bash
# A: scaling micro-benchmark (no root)
source ~/slorado_venv/bin/activate && python3 docs-riley/lstm-profile/lstm_scaling.py

# B: nsys kernel breakdown (no root)
nsys profile -t cuda -o hac1k --force-overwrite true --stats=false \
  ./slorado basecaller -C 128 -c 12288 -K 4096 -p 150 --overlap-decode=yes --fixed-c-batch=no \
  --flush-threshold=128 -o /dev/null models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 test/PGXXXX230339/reads_1k.blow5
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv hac1k.nsys-rep > kern.csv

# C: cleaner ncu pass — needs root (RmProfilingAdminOnly=1). Real clocks, DRAM + occupancy metrics.
sudo env "PATH=$PATH" ncu --clock-control none \
  --kernel-name regex:"gemm_fp16_128x64|elemWiseRNNcell|Kernel2" --launch-skip 2000 --launch-count 40 \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size \
  ./slorado basecaller -C 128 -c 12288 -K 4096 -p 150 -o /dev/null \
  models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 test/PGXXXX230339/reads_1.blow5 2>&1 | tee ncu-clean.txt
```

C was run (results in section 6b, raw output `lstm-profile/ncu-clean-pass.txt`) but **at N = 2**, and it did not yield a DRAM figure (n/a on this iGPU). To profile the production batch size, add `--fixed-c-batch=yes` so the forward runs C-wide even for one read, and skip further into the run, e.g.:

```bash
sudo env "PATH=$PATH" ncu --clock-control none \
  --kernel-name regex:"gemm_fp16_128x64|elemWiseRNNcell" --launch-skip 2000 --launch-count 40 \
  --metrics gpu__time_duration.sum,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size \
  ./slorado basecaller -C 128 -c 12288 -K 4096 -p 150 --fixed-c-batch=yes -o /dev/null \
  models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 test/PGXXXX230339/reads_1.blow5 2>&1 | tee ncu-n128.txt
```
(Not yet run.)
