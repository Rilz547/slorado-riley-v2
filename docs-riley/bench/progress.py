#!/usr/bin/env python3
"""Campaign progress and a rough ETA (reads bench-runs/runs.jsonl only; no load on the Jetson, safe to run any time).

    python3 progress.py [--no-extras]

Expected runs follow campaign.sh: per stage and cell, 1k = 5 reps x modes A,B; 20k = 3 reps x A,B; 500k = 1 rep, mode B;
plus one uncounted warm-up per cell (not for 500k). ETA = remaining runs x their expected duration (samples / the median
samples-per-second already measured for that stage and model, else for that model in any stage) + ~5 min per remaining block.
It is a rough guide, not a promise.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import bench

CORE = ["upstream", "ov_off", "ov_on", "lb_best", "spec"]
RNA_STAGES = {"upstream", "ov_on", "lb_best", "spec"}
EXTRAS = ["lb_baseline", "cb_off", "cb_on"]
BLOCK_OVERHEAD_S = 300  # 2 x 60 s idle + 2 canaries + gate
RUN_OVERHEAD_S = 15  # pre/post power shoulders, model load, exit, sync, thermal gate
# samples/s priors (from the validation runs) used until a cell type has real data: (model, dtype) -> rate
PRIOR_RATE = {("fast", "dna"): 4.0e6, ("hac", "dna"): 1.2e6, ("fast", "rna"): 3.7e6, ("hac", "rna"): 0.96e6}


def plan(extras: bool) -> list[tuple[str, str, str]]:
    """(phase, stage, cell) in campaign order."""
    out = []
    for size, phase in (("1k", "1"), ("20k", "2")):
        for s in CORE:
            cells = [f"dna_fast_{size}", f"dna_hac_{size}"] + ([f"rna_fast_{size}", f"rna_hac_{size}"] if s in RNA_STAGES else [])
            out += [(phase, s, c) for c in cells]
    out += [("3", s, "dna_fast_500k") for s in ("paper_xavier", "upstream", "spec")]
    if extras:
        for size in ("1k", "20k"):
            out += [("4", s, c) for s in EXTRAS for c in (f"dna_fast_{size}", f"dna_hac_{size}")]
    return out


def fmt_h(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-extras", action="store_true", help="leave out phase 4 (lb_baseline, cb_off, cb_on)")
    a = ap.parse_args()
    cfg = bench.load_config()
    recs = bench.load_records()
    done_t = {}
    done_w = set()
    rates: dict[tuple, list[float]] = {}
    for r in recs:
        if r.get("status") != "ok":
            continue
        if r.get("kind") == "timed":
            done_t[(r["stage"], r["cell"], r["mode"])] = done_t.get((r["stage"], r["cell"], r["mode"]), 0) + 1
            if r.get("samples_per_s"):
                rates.setdefault((r["stage"], r["config"], r["dtype"]), []).append(r["samples_per_s"])
                rates.setdefault(("*", r["config"], r["dtype"]), []).append(r["samples_per_s"])
        elif r.get("kind") == "warmup":
            done_w.add((r["stage"], r["cell"]))
    started = min((r.get("t_start_epoch") or r.get("window_start_epoch") or 1e18 for r in recs), default=None)
    total = remaining = n_prior_cells = 0
    rem_s = 0.0
    per_phase: dict[str, list[int]] = {}
    blocks_left = set()
    last_block = None
    for phase, stage, cell in plan(not a.no_extras):
        c = bench.parse_cell(cfg, cell)
        ds = c["ds"]
        modes, reps = ds["modes"], ds["reps"]
        r_list = rates.get((stage, c["model"], c["dtype"])) or rates.get(("*", c["model"], c["dtype"]))
        rate = st.median(r_list) if r_list else PRIOR_RATE[(c["model"], c["dtype"])]
        dur = ds["samples"] / rate + RUN_OVERHEAD_S
        n_prior_cells += 0 if r_list else 1
        n_exp = len(modes) * reps
        n_done = sum(min(done_t.get((stage, cell, m), 0), reps) for m in modes)
        pp = per_phase.setdefault(phase, [0, 0])
        pp[0] += n_exp
        pp[1] += n_done
        total += n_exp
        remaining += n_exp - n_done
        warm_left = 0 if (ds.get("warmup", True) is False or (stage, cell) in done_w or n_done) else 1
        if n_done < n_exp:
            blocks_left.add((phase, stage, cell.rsplit("_", 1)[1] if phase in ("1", "2", "4") else ""))
            rem_s += dur * (n_exp - n_done + warm_left)
    elapsed = (time.time() - started) if started else 0
    done = total - remaining
    print(f"timed runs done: {done} / {total}  ({100 * done / total:.1f} %)   elapsed: {fmt_h(elapsed)}")
    for ph, (n, d) in sorted(per_phase.items()):
        label = {"1": "1k cells", "2": "20k cells", "3": "500k FAST", "4": "extras"}[ph]
        print(f"  phase {ph} ({label}): {d} / {n}")
    eta = rem_s + BLOCK_OVERHEAD_S * len(blocks_left)
    print(f"remaining: {remaining} timed runs in {len(blocks_left)} blocks; rough ETA {fmt_h(eta)}"
          + (f"  ({n_prior_cells} of {len(plan(not a.no_extras))} cells still use prior rates; the ETA sharpens as data arrives)" if n_prior_cells else ""))
    failed = [r for r in recs if r.get("kind") in ("timed", "warmup") and r.get("status") != "ok"]
    print(f"failed attempts so far: {len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
