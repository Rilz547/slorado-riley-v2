#!/usr/bin/env python3
"""500k FAST at a reduced read batch size (-K), for the stages that did not complete at the campaign's -K 4096.

Why: `spec` hung on 4 of 4 attempts at 500k (RAM reached 6.6-7.4 GB of 7.6 GB; upstream stayed under 6.5 GB). This tests whether a
smaller -K (default 2000, the value the preprint's Xavier command line used and that `paper_xavier` ran with here) lets it finish.

It reuses bench.py's own measurement function (same power window, idle baselines, canaries, hang detection), so records are
comparable with the campaign. Only the shared flag `-K` is changed, in memory, for this process; no stamped file is edited.
Records go to bench-runs/reducedK-500k.jsonl (never runs.jsonl) with variant "K<value>"; kept FASTQs are named
output_fast_500k-dna-<stage>-K<value>.fastq. One attempt per stage (a hang is not retried: it costs up to an hour).

    python3 reduced_k_500k.py [--K 2000] [--stages spec,upstream] [--dry-run] [--continue-on-fail]

Default order is spec first; if it does not complete, `upstream` is skipped (use --continue-on-fail to run it anyway).
Do not use the Jetson desktop while it runs. ~70-80 min for spec, ~105 min for upstream.
"""

from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path

import bench

CELL = "dna_fast_500k"


def with_reduced_k(cfg: dict, k: int) -> None:
    flags = list(cfg["shared_flags"])
    i = flags.index("-K")
    flags[i + 1] = str(k)
    cfg["shared_flags"] = flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--stages", default="spec,upstream")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and run the pre-flight checks only")
    ap.add_argument("--continue-on-fail", action="store_true")
    ap.add_argument("--runs-file", default=str(bench.OUT_ROOT / "reducedK-500k.jsonl"))
    a = ap.parse_args()

    bench.RUNS_FILE = Path(a.runs_file)
    cfg = bench.load_config()
    with_reduced_k(cfg, a.K)
    stages = [s for s in a.stages.split(",") if s]
    variant = f"K{a.K}"

    snap, problems = bench.preflight(cfg, stages, True, [CELL])
    if problems:
        print("PRE-FLIGHT FAILED:\n  " + "\n  ".join(problems))
        return 2
    c = bench.parse_cell(cfg, CELL)
    for s in stages:
        cmd = bench.build_cmd(cfg, s, c, str(bench.BENCH_OUT / "tmp" / "<fastq>"), None, None)
        print(f"{s}: {' '.join(cmd)}")
    if a.dry_run:
        return 0

    bench.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(bench.LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2

    block = f"reducedK-{variant}-{time.strftime('%Y%m%dT%H%M%S')}"
    print(f"[reducedK] block {block}  harness {bench.harness_version()}  stages {stages}  -K {a.K}", flush=True)
    print("[reducedK] idle baseline (before)", flush=True)
    bench.record_idle(cfg, f"reducedK:before", block)
    print("[reducedK] canary (before)", flush=True)
    bench.record_canary(cfg, f"reducedK:before", block)

    failed = False
    for s in stages:
        if failed and not a.continue_on_fail:
            print(f"[reducedK] skipping {s} (an earlier stage did not complete)", flush=True)
            continue
        print(f"[reducedK] {s} {CELL} mode B, -K {a.K}", flush=True)
        rec = bench.run_slorado_measured(cfg, s, CELL, "B", 1, "timed", block, True, None, False, None,
                                         keep_as=f"output_fast_500k-dna-{s}-{variant}.fastq")
        rec["attempt"], rec["variant"] = 1, variant
        bench.append_record(rec)
        if rec["status"] == "ok":
            print(f"        Real {rec['real_s']:.1f}s ({rec['real_s'] / 60:.1f} min)  proc {rec['wall_process_s']:.0f}s  E_in {rec.get('E_in_J', float('nan')) / 1000:.1f} kJ  "
                  f"P {rec.get('Pmean_in_W', float('nan')):.2f} W  peakRAM {rec.get('peak_ram_gb')} GB", flush=True)
        else:
            failed = True
            print(f"        FAILED: {rec.get('error') or rec.get('killed')}", flush=True)

    print("[reducedK] canary (after)", flush=True)
    bench.record_canary(cfg, "reducedK:after", block)
    print("[reducedK] idle baseline (after)", flush=True)
    bench.record_idle(cfg, "reducedK:after", block)
    print("[reducedK] done", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
