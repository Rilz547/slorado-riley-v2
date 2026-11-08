#!/usr/bin/env python3
"""MAXN_SUPER + jetson_clocks sensitivity run: the campaign stages upstream / ov_off / spec on a few cells, interleaved,
through the frozen harness's measurement (bench.run_with_retries) but WITHOUT the 25 W / clocks-off pre-flight refusal.
NOT a campaign result: its own jsonl (bench-runs/maxn-<label>-<ts>.jsonl), reported as a separate sensitivity table.

  python3 maxn_sens.py --label maxn                                 # upstream,ov_off,spec x FAST 1k, HAC 1k (3 reps), FAST 20k (1 rep)
  python3 maxn_sens.py --summary                                    # per cell: mean per stage and change vs upstream, every maxn-*.jsonl block
Order per cell: warm-up for each stage, then rep 1 of every stage, rep 2 of every stage, ... (interleaved against drift).
"""
from __future__ import annotations
import argparse, fcntl, glob, json, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench

REPS = {"dna_fast_1k": 3, "dna_hac_1k": 3, "dna_fast_20k": 1}
OC = "/sys/class/hwmon/hwmon3/oc3_event_cnt"

def freq_state() -> str:
    d = "/sys/class/devfreq/17000000.gpu/"
    try:
        return f"gpu min {int(open(d + 'min_freq').read()) // 10**6} MHz, max {int(open(d + 'max_freq').read()) // 10**6} MHz, cur {int(open(d + 'cur_freq').read()) // 10**6} MHz"
    except OSError as e:
        return f"unreadable ({e})"

def oc_count() -> str:
    try:
        return open(OC).read().strip()
    except OSError:
        return "n/a"

def summary() -> None:
    rows = []
    for f in sorted(glob.glob(str(bench.OUT_ROOT / "maxn-*.jsonl"))):
        for ln in open(f):
            a = json.loads(ln)
            if a.get("kind") == "timed" and a.get("status") == "ok":
                rows.append((Path(f).stem, a))
    for label in sorted({l for l, _ in rows}):
        print(f"\n=== {label}")
        for cell in sorted({a["cell"] for l, a in rows if l == label}):
            print(cell)
            base = None
            for stage in ["upstream", "ov_off", "spec"]:
                x = [a for l, a in rows if l == label and a["cell"] == cell and a["stage"] == stage]
                if not x:
                    continue
                t = [a["real_s"] for a in x]
                m = st.mean(t)
                base = base or m
                e = [a["E_in_J"] for a in x if a.get("E_in_J")]
                g = [a["gpu_mhz_mean"] for a in x if a.get("gpu_mhz_mean")]
                print(f"  {stage:9} n={len(t)} {m:8.3f} ± {st.stdev(t) if len(t) > 1 else 0:5.3f} s  {100*(m-base)/base:+6.1f} % vs upstream"
                      f"  gpu {st.mean(g) if g else 0:5.0f} MHz  E {st.mean(e) if e else 0:7.1f} J")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="maxn")
    ap.add_argument("--stages", default="upstream,ov_off,spec")
    ap.add_argument("--cells", default=",".join(REPS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(); return 0
    cfg = bench.load_config()
    stages, cells = a.stages.split(","), a.cells.split(",")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    bench.RUNS_FILE = bench.OUT_ROOT / f"maxn-{a.label}-{stamp}.jsonl"
    print(f"[maxn] label {a.label}   {freq_state()}   nvpmodel: {' '.join(bench.sh(['nvpmodel', '-q']).split())}   oc_count {oc_count()}", flush=True)
    bench.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(bench.LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)"); return 2
    block = f"maxn-{a.label}-{stamp}"
    bench.record_idle(cfg, f"maxn-{a.label}:before", block)
    bench.record_canary(cfg, f"maxn-{a.label}:before", block)
    for cell in cells:
        reps = REPS.get(cell, 1)
        for s in stages:
            print(f"[maxn] {cell} {s} warm-up", flush=True)
            bench.run_with_retries(cfg, s, "dna_fast_1k" if cell == "dna_fast_20k" else cell, "A", 0, "warmup", block, False, None, False)
        for rep in range(1, reps + 1):
            for s in stages:
                rec = bench.run_with_retries(cfg, s, cell, "A", rep, "timed", block, False, None, False)
                print(f"   {cell} {s:9} rep {rep}/{reps}: " + (f"Real {rec['real_s']:.3f} s  P {rec.get('Pmean_in_W') or 0:.2f} W  gpu {rec.get('gpu_mhz_mean')} MHz" if rec["status"] == "ok" else f"FAILED {rec.get('error')}"), flush=True)
    bench.record_canary(cfg, f"maxn-{a.label}:after", block)
    bench.record_idle(cfg, f"maxn-{a.label}:after", block)
    print(f"[maxn] end state: {freq_state()}   oc_count {oc_count()}\n[maxn] done -> {bench.RUNS_FILE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
