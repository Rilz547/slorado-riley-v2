# Experiment: full depth-2 decode overlap (P5-full)

**Date:** 2026-07-31  
**Branch:** `overlap-depth2` (from `load-imbalance`)  
**Status:** **FAILED ROUTE** — keep flag for reference; do not make default

## Goal

See if two in-flight GPU decodes beat v1.2 on FAST. If not, document and stop.

## What landed

- CLI `--overlap-depth=1|2` (default 1 = v1.2)
- Depth 2: second `gpubuf`, second decode stream, pending ring of 2
- Decode phase stats disabled when depth=2 (one-shot timing unsafe)
- Default path unchanged at depth=1

## Baseline command (sub‑10s load-balancer)

```bash
./slorado basecaller \
  -C 128 -c 12288 -K 4096 -p 150 \
  --overlap-decode=yes \
  --overlap-depth=DEPTH \
  --fixed-c-batch=no \
  --flush-threshold=64 \
  -o /dev/null \
  models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
  test/PGXXXX230339/reads_1k.blow5
```

## Result (Orin, 2026-07-31) — timed `/dev/null`, n=3 after warmup

| depth | run1 | run2 | run3 | mean | peak RAM |
|------:|-----:|-----:|-----:|-----:|---------:|
| 1 (v1.2) | 9.842 | 9.735 | 9.762 | **9.780 s** | ~2.55 GB |
| 2 | 9.864 | 9.842 | 9.810 | **9.839 s** | ~2.76 GB |

**Verdict:** depth-2 ≈ **0.6% slower** than depth-1 (noise / slightly worse), +~0.2 GB RAM. No wall win. Orin likely time-slices the second decode against the first rather than hiding more work.

## Why it didn’t help (short)

v1.2 already overlaps decode(N−1) with infer(N). Remaining exposed time is mostly `max(0, decode−infer)`. A second concurrent decode doesn’t shrink that on a saturated Orin GPU; it adds VRAM and scheduling overhead.

## Next

- Leave `--overlap-depth=2` available but unused for demos.
- Prefer: more load-imbalance testing, or P3 flip spike, or P4 beam sweep.
