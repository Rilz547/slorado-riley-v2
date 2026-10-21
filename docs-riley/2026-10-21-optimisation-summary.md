# Slorado on Jetson Orin — optimisations tried: what worked, what didn't

**Date:** 2026-10-21 (work done Jun–Oct 2026)
**Platform:** Jetson Orin (Nano-class, ~7.6 GB shared), `nvpmodel` 25W, `jetson_clocks` OFF
**Models / data:** DNA R10.4.1 e8.2 400bps `fast@v5.0.0` and `hac@v5.0.0`; `reads_1k` and `reads_20k`
**Code:** slorado `speculative-decode` (`49ac641`); openfish `speculative-greedy` (`5109673`)
**Shared flags:** `-C 128 -c 12288 -K 4096 -p 150`

All wall times are slorado's `[main] Real time` (excludes process/Torch startup), output to `/dev/null`.
**Every figure below is from a single run**, so differences of a percent or two are noise.

---

## 1. Summary

### What worked

| change | idea | measured effect |
|---|---|---|
| **Overlap decode** (v1.0 → v1.2) | run GPU decode of batch *N−1* concurrently with inference of batch *N* | FAST −15 to −17 %, HAC ≈ −3 % |
| **Load imbalance** (narrow batches + flush threshold) | stop padding partial batches to a full `-C`-wide GPU launch | FAST 1k ≈ −12 %; ≈ 0 on FAST 20k and HAC |
| **Speculative decode** | cheap greedy draft for every chunk; full beam search only for chunks the draft is unsure about | FAST ≈ −7 to −10 % for ≈ −0.5 pp identity; HAC < 1 % for ≈ −0.15 pp identity |

### Dead ends

| idea | outcome |
|---|---|
| **CPU beam hybrid** (`cpu-beam`) | slower in every cell: +2 % to +19 % wall in the Aug power runs (and 2.5–3.3× in the July timing report, which measured a different implementation) |
| **Overlap depth-2** (dual decode) | documented as a failed route |
| **FAST→HAC cascade** (promote hard reads to HAC) | negligible or worse |
| **`cuda-lstm-gemm`, `flip-spike` branches** | personal tests, no commits of their own |
| **Speculative `m = 3` margin** | FAST: near-beam identity but slower than baseline (accept rate ≈ 7–9 %). HAC: still accepts ≈ 67–71 % but 20k is slower than baseline |

### Investigated and ruled out

| idea | why |
|---|---|
| **Larger batches** (`-C 256`, `-C 512`) | HAC runs out of GPU memory; `-C 128` already peaks at ≈ 5.0 GB |
| **Int8 quantisation** | the codebase's int8 path covers only v6 factored-LSTM and transformer models; the fused int8 FLSTM kernel was removed, and `--quant` is never passed to the plain-LSTM loader used by v5.0.0 FAST/HAC. A real int8 LSTM would need new kernels |

### Overall (power-campaign chain, FAST 20k, same harness)

217.1 s (pre-overlap) → 206.1 (overlap off, newer build) → 174.2 (overlap on) → 173.8 (narrow batches) → 161.0 s (speculative) ≈ **−26 %**.
The runner measured 158.8 s for the final configuration. HAC 20k: 934.3 s → 865.4 s ≈ **−7 %**.
"Pre-overlap" is the start of this optimisation work (commit `8873209`), not upstream slorado.

### Final timings (rileys-runner.py, `q10_m2_brave`)

| config | time | median identity | Δ vs baseline |
|---|---|---|---|
| FAST 1k | 9.02 s | 0.935276 | −0.0055 |
| HAC 1k | 44.68 s | 0.975490 | −0.0014 |
| FAST 20k | 158.8 s | 0.934853 | −0.0045 |
| HAC 20k | 865.4 s | 0.975521 | −0.0018 |

---

## 2. Pseudocode of the three changes that worked

Notation: `N` = real chunks in a batch, `C` = `-C` (GPU batch size, 128), `T` = timesteps per chunk (2048 for HAC/FAST at `-c 12288`),
`NTC` = scores laid out `[batch, time, channels]`.

### 2.1 Overlap decode  (`basecall_chunks_overlap`, `src/basecall.cpp`; `--overlap-decode=yes`)

**Problem.** The serial path does `infer(batch)` then `decode(batch)`; the GPU decode and the CPU work to turn its output into
sequences leave inference idle.

**Idea.** A depth-1 software pipeline. Each runner owns two CUDA streams (inference, decode), two input buffers, two completion
events, and a 2-slot pinned host output ring in openfish. Decode of batch *k* runs while the CPU stages batch *k+1* and the GPU infers it.

```text
per-runner state:
    input[0..1]            # two CPU->GPU staging buffers, ping-ponged by `slot`
    infer_stream, decode_stream
    infer_event[0..1]      # "inference for this slot is finished"
    host_ring[0..1]        # persistent pinned host buffers for decode output (v1.1: never freed per batch;
                           #   v1.2: two slots so batch k's output survives while k-1 is still being written)
    pending                # the previous batch: its scores (on GPU) + chunk list + which slot it used

basecall_chunks_overlap(chunks):
    slot = runner.slot                              # 0 or 1, flips every batch
    copy each chunk's signal into input[slot]       # CPU, "accept"

    if pending exists and its decode is not yet launched:
        launch_decode(pending)                      # see below; asynchronous, no sync

    # --- inference of the CURRENT batch, concurrent with the decode of the PREVIOUS one ---
    with stream = infer_stream:
        x      = input[slot][0:N]                   # narrowed to real chunks (see 2.2)
        scores = model.forward(x)                   # NTC
        record(infer_event[slot])

    if pending exists:
        synchronize(decode_stream)                  # previous decode has finished
        write_decode_results(pending)               # CPU: moves/sequence/qstring -> chunk objects
        pending = none

    pending = { chunks, scores, slot, host_slot = slot & 1 }
    launch_decode(pending)                          # starts as soon as infer_event[slot] fires
    runner.slot = slot ^ 1                          # ping-pong

launch_decode(p):
    with stream = decode_stream:
        wait(infer_event[p.slot])                   # decode only starts after THIS batch's inference is done
        openfish_decode_gpu(p.scores, ..., host_slot = p.host_slot, stream = decode_stream)
            # bwd_scan -> beam_search -> fwd_post_scan -> qual -> gen_sequence -> async D2H into host_ring[host_slot]
            # no cudaStreamSynchronize inside: stream != NULL skips the per-phase device syncs

flush at end of input:
    launch (if needed) + synchronize + write the last pending batch serially (nothing left to overlap with)
```

Version history: **v1.0** infer∥decode with correct NTC layout for `C > 1`; **v1.1** persistent pinned host buffers
(removes per-batch `cudaHostAlloc/FreeHost`, which was an nsys hotspot); **v1.2** two-slot ring so one batch's CPU write-back
can overlap the next batch's decode.

### 2.2 Load imbalance — narrow batches + flush threshold  (`pthread_single_basecall`, `call_chunks`; `--fixed-c-batch`, `--flush-threshold`)

**Problem.** Every GPU launch used the full `C`-wide input tensor even when only `N < C` chunks were real (tail of the input, or a
partial flush), so inference and decode were spent on padding.

**Idea.** (a) Run the forward pass and decode only on the `N` real chunks; (b) let the batcher flush a batch earlier than `C`.

```text
flush_thr = flush_threshold  if 0 < flush_threshold <= C  else C

queue = []
for each read assigned to this runner:
    for each chunk of the read:
        queue.append(chunk)
        if len(queue) >= flush_thr:
            dispatch(queue); queue = []
if queue not empty:
    dispatch(queue)                                  # leftover tail batch (N < flush_thr)

dispatch(batch):
    N = len(batch)
    stats: total_batches++, total_chunks += N; if N < C: tail_batches++, padded_slots += C - N
    if overlap_decode: basecall_chunks_overlap(batch)      # 2.1
    else:              basecall_chunks(batch)

# inside inference (both paths):
input  = input_tensor                    if --fixed-c-batch=yes        # old behaviour: full C rows, padding included
       = input_tensor.narrow(0, 0, N)    otherwise                     # only the N real rows
scores = model.forward(input)            # N rows, so the CRF scores and the openfish decode are N-wide too
```

Notes: the "best" settings found were `--fixed-c-batch=no` with `--flush-threshold=64` for FAST and `128` for HAC. Smaller thresholds
flush sooner but run narrower (less efficient) batches; the benefit showed up on the short FAST 1k run where tail padding was a
large fraction of the work. slorado's "padded slots" statistic counts `C − N` per tail batch and is an accounting figure, not the
launch width when narrowing is on.

### 2.3 Speculative decode  (`overlap_*` and `decode_scores_speculative` in `src/basecall.cpp`; kernels in `openfish/src/greedy_search_cuda.h`, `decode_cuda.c`)

**Problem.** After the LSTM, FAST is dominated by the width-32 CRF beam search. Most chunks are easy.

**Idea.** Decode every chunk with a cheap width-1 *greedy* path; accept it when it looks confident; run the full beam only on the rest ("repair").

```text
# thresholds (flags): Q_thr = --spec-repair-threshold (10), M_thr = --spec-margin-threshold (2)

decode_speculative(scores):                       # scores: NTC for one batch
    (moves, seq, qstr, margin[]) = openfish_decode_gpu_greedy(scores)       # draft, ALL chunks
    write drafts into the chunk objects
    hard = []
    for i in 0 .. N-1:
        q = mean_phred(chunk[i].qstring)
        if q < Q_thr  or  margin[i] < M_thr:     # not confident enough
            hard.append(i)
    if hard is empty: return                      # every draft accepted; scores can be dropped
    hard_scores = scores.index_select(dim=0, index=hard)   # copy only the hard rows
    (moves2, seq2, qstr2) = openfish_decode_gpu(hard_scores)              # full beam, hard subset only
    for j, i in enumerate(hard):
        chunk[i] = result j of the repair                                 # overwrites the draft
```

The greedy draft itself (openfish, replaces `beam_search` inside the same decode pipeline):

```text
openfish_decode_gpu_greedy(scores):               # same pipeline as the beam decode, one kernel swapped
    bwd_scan(scores)           -> bwd[chunk][t][state]        # backward guide
    greedy_search(scores, bwd) -> states[], moves[], margin[chunk]     # <-- replaces beam_search
    fwd_post_scan(scores, bwd) -> posteriors
    compute_qual_data(posteriors, states)
    generate_sequence(...)     -> sequence, qstring
    async D2H of moves / sequence / qstring / margin into the pinned host ring slot

greedy_search (one GPU thread per chunk):
    state      = argmax over s of bwd[0][s]       # start where the backward guide is best
    margin_sum = 0
    for t in 0 .. T-1:
        candidates:
            for base in 0..3:                      # "step": append one base to the k-mer
                new_state = ((state << 2) & mask) | base
                score[base] = emission[t][move_index(state, new_state)] * scale + bwd[t+1][new_state]
            stay_score      = fixed_stay_score + bwd[t+1][state]        # "stay": repeat the same k-mer
        best, second = largest and second-largest of the five scores
        state = best's destination state
        states[t] = state;  moves[t] = 1 if best was a step else 0
        margin_sum += best - second
    moves[0] = 1                                   # always a step at the first event
    margin[chunk] = margin_sum / T                 # mean decision margin; low = unreliable chunk
```

The "brave" variant (`--spec-overlap-repair=yes`) also overlaps the repair with the next batch's inference, using the same
two-stream structure as 2.1:

```text
basecall_chunks_overlap(chunks):                    # speculative + brave
    accept chunks into input[slot]
    if repair_hold exists:   launch full-beam repair of repair_hold's hard subset on decode_stream   # repair(k-1) ‖ infer(k)
    elif pending exists:     launch greedy draft of pending on decode_stream
    infer(current) on infer_stream
    if repair_hold exists:   sync decode_stream; write repaired chunks; repair_hold = none
    if pending exists:
        if draft not launched and no repair queued: launch draft(pending)
        sync decode_stream; gate(pending)            # write draft, build `hard`
        if pending has hard chunks:
            repair_hold = pending                    # defer: repaired during the NEXT iteration's inference
        pending = none
    pending = current batch (scores kept alive)
    if repair_hold is empty: launch draft(pending) now      # otherwise delay, so the GPU buffers stay exclusive to the repair
```

Cost/benefit knobs: the margin gate dominates the accept rate (m = 2 → ≈ 73–76 % of chunks accepted on FAST, ≈ 81–83 % on HAC; m = 3 → ≈ 7–9 % on FAST and ≈ 67–71 % on HAC, and slower than baseline on FAST and on HAC 20k).
HAC gains little because its wall time is inference-bound (≈ 83 % LSTM), so saving decode time hardly shows.

---

## 3. What the profiling added (see `2026-10-05-hac-lstm-profile.md`)

- HAC wall time is ≈ 83 % LSTM, which none of the three changes above touch.
- About 30 % of each LSTM layer is a near-idle gate/cell elementwise kernel (`elemWiseRNNcell`, ≈ 23 µs/step, launched 409 600 times per 1k run).
- The best untested lead is replacing that kernel (estimate: ≈ −20 % of HAC wall; unbuilt and unmeasured).
- The ncu per-kernel numbers in that document were taken at N = 2, not N = 128, and are provisional; the nsys and micro-benchmark results are N = 128.

## 4. Open items

- Same-harness baseline: re-run the pre-overlap commit through `rileys-runner.py` so both ends of the comparison use the same harness.
- Baseline accuracy for the load-imbalance config on HAC/20k in the same session as the speculative runs.
- Optional: proof-of-concept fused LSTM cell kernel; re-run ncu at N = 128 (`--fixed-c-batch=yes`).
- Merge `speculative-decode` into `develop` (deferred while work continues).
