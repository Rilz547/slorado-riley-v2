#!/usr/bin/env python3
"""One valid 500k measurement of an experiment binary, retried until a run completes. Side experiment, NOT part of the frozen campaign.

Same protocol as the campaign's 500k cells (mode B: FASTQ written and kept, 1 rep, no warm-up) with the same measurement function, power
window, idle baselines, canary and hang handling (bench.py). The variant is an experiment binary ~/bench-stages/x_<name>/slorado, which
variant_runs.py registers in memory with the `spec` flags. Records go to bench-runs/variants-fix500k.jsonl (never runs.jsonl); every attempt
is recorded, so the number of hung attempts is part of the result.

    python3 run_fixed_500k.py --dry-run                       # show the command and the pre-flight result
    python3 run_fixed_500k.py [--variant x_barrier] [--cell dna_fast_500k] [--max-attempts 6]
    # the paper's own settings with the optimised build (records -> variants-fix500k-paperflags.jsonl, FASTQ -> kept-sweeps/):
    python3 run_fixed_500k.py --tag paperflags --extra "-t 8 -B 500M -K 2000 -c 10000 -C 200" --kept-dir ~/slorado-riley-v2/docs-riley/bench-runs/kept-sweeps

The kept FASTQ is bench-runs/kept/output_fast_500k-dna-<variant>[_<tag>].fastq (about 4 GB). --extra options are appended after the stage's flags, so they override
the shared -C 128 -c 12288 -K 4096 (slorado rounds -c down to a multiple of the model stride: 10000 runs as 9996). Do not use the Jetson desktop while this runs.
"""

from __future__ import annotations

import argparse
import fcntl
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import bench  # noqa: E402
import variant_runs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="x_barrier")
    ap.add_argument("--cell", default="dna_fast_500k")
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--extra", default="", help='slorado options appended after the stage flags (later options override earlier ones)')
    ap.add_argument("--tag", default="", help="label for this variant of the run: names the records file and the kept FASTQ")
    ap.add_argument("--kept-dir", help="directory for the kept FASTQ (default bench-runs/kept)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = bench.load_config()
    variant_runs.register(cfg)
    bench.RUNS_FILE = bench.OUT_ROOT / (f"variants-fix500k-{a.tag}.jsonl" if a.tag else "variants-fix500k.jsonl")
    if a.kept_dir:
        bench.KEPT_DIR = Path(a.kept_dir).expanduser()
    extra = shlex.split(a.extra) or None
    suffix = f"_{a.tag}" if a.tag else ""
    c = bench.parse_cell(cfg, a.cell)
    snap, problems = bench.preflight(cfg, [a.variant], True, [a.cell])
    print(" ".join(bench.build_cmd(cfg, a.variant, c, "<fastq>", None, extra)).replace(str(Path.home()), "~"))
    print(f"records -> {bench.RUNS_FILE}   up to {a.max_attempts} attempts of about 70 min each (a hung attempt is killed after about 15 min of silence)")
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

    block = f"fix500k-{time.strftime('%Y%m%dT%H%M%S')}"
    print(f"[fix500k] block {block}  harness {bench.harness_version()}", flush=True)
    bench.record_idle(cfg, "fix500k:before", block)
    bench.record_canary(cfg, "fix500k:before", block)
    ok = None
    for attempt in range(1, a.max_attempts + 1):
        print(f"[fix500k] attempt {attempt}/{a.max_attempts} {a.variant} {a.cell}", flush=True)
        rec = bench.run_slorado_measured(cfg, a.variant, a.cell, "B", 1, "timed", block, True, None, False, extra_args=extra,
                                         keep_as=f"output_{c['model']}_{c['ds']['size']}-{c['dtype']}-{a.variant}{suffix}.fastq")
        rec["attempt"] = attempt
        bench.append_record(rec)
        if rec["status"] == "ok":
            ok = rec
            print(f"[fix500k] OK  Real {rec['real_s']:.1f} s  P {rec.get('Pmean_in_W') or 0:.2f} W", flush=True)
            break
        print(f"[fix500k] attempt {attempt} failed: {rec.get('error')}", flush=True)
        subprocess.run(["pkill", "-9", "-f", "slorado basecaller"], capture_output=True)
        time.sleep(15)
    bench.record_canary(cfg, "fix500k:after", block)
    bench.record_idle(cfg, "fix500k:after", block)
    print("[fix500k] " + (f"completed on attempt {ok['attempt']}" if ok else f"no completed run in {a.max_attempts} attempts"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
