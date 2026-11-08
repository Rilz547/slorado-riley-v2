#!/usr/bin/env python3
"""Paired / interleaved comparisons for the baseline-pairing questions. Same measurement function, power window, idle baselines,
canaries and hang handling as the campaign (bench.py); variants are added IN MEMORY only, so no stamped file is edited.
Records go to bench-runs/variants-<set>.jsonl (never runs.jsonl); stage label = variant name.

Variants run ABAB-interleaved per cell (rep 1 of every variant, then rep 2, ...) so slow drift hits all of them alike.

  --set pairing   spec_off vs spec_on        the SAME 49ac641 binary with --speculative-decode=no / yes   (no build needed)
  --set bundle    ref_upstream, u_timing, v10_off, v11_off, ref_ov_off   splits upstream -> ov_off (needs the 3 extra builds)
  --set dev       ref_upstream, bdev, bdev_async, spec_on    Bonson's dev (v0.7.0 in progress) vs upstream and spec  (needs the dev build)
  --set exp       spec_on + any ~/bench-stages/x_*/slorado experiment binary (pass --variants spec_on,x_sm87,...); FAST 1k by default

    python3 variant_runs.py --set pairing [--cells dna_fast_1k,dna_hac_1k,dna_fast_20k] [--reps N] [--with-b1] [--dry-run]

Builds: setup_extra_worktrees.sh. Do not use the Jetson desktop while this runs; nothing else on the GPU.
--with-b1 adds one mode-B run per variant and cell (rep 1, kept as output_<model>_<size>-dna-<variant>.fastq) to check output equivalence.
"""

from __future__ import annotations

import argparse
import fcntl
import statistics as st
import time
from pathlib import Path

import bench

EXTRA_WORKTREES = {
    "b_3403f19": {"commit": "3403f19", "note": "upstream + LSTM layer-timing instrumentation (before the openfish fork)"},
    "b_e00bada": {"commit": "e00bada", "note": "overlap v1.0: fork openfish pin + stream-aware decode (flag off here)"},
    "b_9f8db24": {"commit": "9f8db24", "note": "overlap v1.1: persistent pinned host decode buffers (flag off here)"},
    "bonson_dev": {"commit": "c2ca899", "note": "Bonson's dev, v0.7.0 in progress (openfish pin 991561a)"},
}
FLUSH = {"fast": ["--flush-threshold=64"], "hac": ["--flush-threshold=128"]}
VARIANTS = {  # name -> (worktree, args, model_args)
    "spec_off": ("spec", ["--overlap-decode=yes", "--fixed-c-batch=no", "--speculative-decode=no"], FLUSH),
    "spec_on": ("spec", ["--overlap-decode=yes", "--fixed-c-batch=no", "--speculative-decode=yes", "--spec-repair-threshold=10",
                         "--spec-margin-threshold=2", "--spec-overlap-repair=yes"], FLUSH),
    "ref_upstream": ("upstream", [], {}),
    "ref_ov_off": ("overlap", ["--overlap-decode=no"], {}),
    "u_timing": ("b_3403f19", [], {}),
    "v10_off": ("b_e00bada", ["--overlap-decode=no"], {}),
    "v11_off": ("b_9f8db24", ["--overlap-decode=no"], {}),
    "bdev": ("bonson_dev", [], {}),
    "bdev_async": ("bonson_dev", ["--async=yes"], {}),
}
SETS = {
    "pairing": (["spec_off", "spec_on"], ["dna_fast_1k", "dna_hac_1k", "dna_fast_20k"]),
    "bundle": (["ref_upstream", "u_timing", "v10_off", "v11_off", "ref_ov_off"], ["dna_fast_1k", "dna_hac_1k", "dna_fast_20k"]),
    "dev": (["ref_upstream", "bdev", "bdev_async", "spec_on"], ["dna_fast_1k", "dna_hac_1k", "dna_fast_20k"]),
    "exp": (["spec_on"], ["dna_fast_1k"]),  # use --variants spec_on,x_<name>,...  (see docs-riley/exp/)
}
PRIOR_RATE = {("fast", "dna"): 4.5e6, ("hac", "dna"): 1.2e6, ("fast", "rna"): 3.7e6, ("hac", "rna"): 0.96e6}


def discover_exp(cfg: dict) -> list[str]:
    """Experiment binaries: every ~/bench-stages/x_<name>/slorado becomes variant x_<name> with the spec_on flags (set `exp`,
    docs-riley/exp/). A binary may be a shell wrapper that sets an env var (OPENFISH_EXP=...) and execs the real build."""
    found = []
    for d in sorted(bench.expand(cfg["worktree_root"]).glob("x_*")):
        if (d / "slorado").exists():
            VARIANTS[d.name] = (d.name, list(VARIANTS["spec_on"][1]), FLUSH)
            EXTRA_WORKTREES[d.name] = {"commit": "exp", "note": "experiment binary (docs-riley/exp)"}
            found.append(d.name)
    return found


def register(cfg: dict) -> None:
    discover_exp(cfg)
    cfg["worktrees"].update(EXTRA_WORKTREES)
    for name, (wt, args, margs) in VARIANTS.items():
        cfg["stages"][name] = {"worktree": wt, "args": list(args), "model_args": {k: list(v) for k, v in margs.items()}}


def estimate_h(cfg: dict, variants: list[str], cells: list[str], reps: int | None, b1: bool) -> float:
    total = 0.0
    for cell in cells:
        c = bench.parse_cell(cfg, cell)
        n = reps or c["ds"]["reps"]
        per = c["ds"]["samples"] / PRIOR_RATE[(c["model"], c["dtype"])] + 15
        total += per * len(variants) * (n + 1 + (1 if b1 else 0))
    return (total + 300 * 2) / 3600


def summarise(path: Path, variants: list[str], cells: list[str]) -> None:
    recs = bench.load_records(path)
    print("\n== summary (mode A, mean ± sd, n) and change against the first variant in each row group ==")
    for cell in cells:
        base = None
        print(f"\n{cell}")
        for v in variants:
            x = [r["real_s"] for r in recs if r.get("kind") == "timed" and r.get("status") == "ok" and r["stage"] == v and r["cell"] == cell and r["mode"] == "A"]
            if not x:
                print(f"  {v:13} no runs")
                continue
            m, sd = st.mean(x), (st.stdev(x) if len(x) > 1 else 0.0)
            if base is None:
                base = m
            print(f"  {v:13} {m:9.3f} ± {sd:6.3f} (n={len(x)})   {100 * (m - base) / base:+6.1f} %   ({base / m:.3f}x)   [{min(x):.2f} .. {max(x):.2f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--variants", help="comma list overriding the set's variants")
    ap.add_argument("--cells", help="comma list overriding the set's cells (HAC 20k is ~15 min per run)")
    ap.add_argument("--reps", type=int, help="override repeats (default: 5 for 1k, 3 for 20k)")
    ap.add_argument("--with-b1", action="store_true")
    ap.add_argument("--runs-file")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    variants = a.variants.split(",") if a.variants else SETS[a.set][0]
    cells = a.cells.split(",") if a.cells else SETS[a.set][1]
    cfg = bench.load_config()
    register(cfg)
    bench.RUNS_FILE = Path(a.runs_file) if a.runs_file else bench.OUT_ROOT / f"variants-{a.set}.jsonl"

    snap, problems = bench.preflight(cfg, variants, True, cells)
    for v in variants:
        c = bench.parse_cell(cfg, cells[0])
        print(f"{v:13} {' '.join(bench.build_cmd(cfg, v, c, '/dev/null', None, None)).replace(str(Path.home()), '~')[:330]}")
    print(f"\nrough time: {estimate_h(cfg, variants, cells, a.reps, a.with_b1):.1f} h of Jetson time   records -> {bench.RUNS_FILE}")
    if problems:
        print("PRE-FLIGHT PROBLEMS:\n  " + "\n  ".join(problems))
    if a.dry_run:
        return 1 if problems else 0
    if problems:
        return 2

    bench.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(bench.LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2

    block = f"variants-{a.set}-{time.strftime('%Y%m%dT%H%M%S')}"
    print(f"[variants] block {block}  harness {bench.harness_version()}", flush=True)
    done = bench.done_keys(bench.load_records()) if a.resume else set()
    bench.record_idle(cfg, f"{a.set}:before", block)
    bench.record_canary(cfg, f"{a.set}:before", block)
    for cell in cells:
        c = bench.parse_cell(cfg, cell)
        n = a.reps or c["ds"]["reps"]
        for v in variants:
            if (v, cell, "A", "warmup", 0) not in done:
                print(f"[variants] {v} {cell} warm-up", flush=True)
                bench.run_with_retries(cfg, v, cell, "A", 0, "warmup", block, False, None, False)
        for rep in range(1, n + 1):
            for v in variants:
                if (v, cell, "A", "timed", rep) in done:
                    continue
                print(f"[variants] {cell} rep {rep}/{n} {v}", flush=True)
                rec = bench.run_with_retries(cfg, v, cell, "A", rep, "timed", block, False, None, False)
                print(f"        {'Real %.3f s  P %.2f W' % (rec['real_s'], rec.get('Pmean_in_W') or 0) if rec['status'] == 'ok' else 'FAILED: ' + str(rec.get('error'))}", flush=True)
        if a.with_b1:
            for v in variants:
                if (v, cell, "B", "timed", 1) in done:
                    continue
                print(f"[variants] {cell} mode B (kept) {v}", flush=True)
                bench.run_with_retries(cfg, v, cell, "B", 1, "timed", block, True, None, False,
                                       keep_as=f"output_{c['model']}_{c['ds']['size']}-{c['dtype']}-{v}.fastq")
    bench.record_canary(cfg, f"{a.set}:after", block)
    bench.record_idle(cfg, f"{a.set}:after", block)
    summarise(bench.RUNS_FILE, variants, cells)
    print("[variants] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
