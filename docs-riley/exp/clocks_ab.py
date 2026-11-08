#!/usr/bin/env python3
"""jetson_clocks A/B: same harness measurement (bench.run_with_retries: Real-time window, INA3221 power, idle baseline, canary),
but WITHOUT the pre-flight refusal when jetson_clocks is on.  Run once per clock state, label it, toggle clocks by hand between runs.

  python3 clocks_ab.py --label off --reps 5 [--cells dna_fast_1k,dna_hac_1k]    # records -> bench-runs/clocks-<label>-<time>.jsonl
  python3 clocks_ab.py --summary                                                  # compare every clocks-*.jsonl block

Variant: spec_on (the final configuration). Not a campaign result: report separately.
"""
from __future__ import annotations
import argparse, fcntl, glob, json, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench, variant_runs

def freq_state() -> str:
    d = "/sys/class/devfreq/17000000.gpu/"
    try:
        return f"gpu min {int(open(d + 'min_freq').read()) // 10**6} MHz, max {int(open(d + 'max_freq').read()) // 10**6} MHz, cur {int(open(d + 'cur_freq').read()) // 10**6} MHz"
    except OSError as e:
        return f"unreadable ({e})"

def summary() -> None:
    files = sorted(glob.glob(str(bench.OUT_ROOT / "clocks-*.jsonl")))
    rows = []
    for f in files:
        for ln in open(f):
            a = json.loads(ln)
            if a.get("kind") == "timed" and a.get("status") == "ok":
                rows.append((Path(f).stem, a))
    labels = sorted({l for l, _ in rows}, key=lambda s: s.split("-")[-1])
    for cell in sorted({a["cell"] for _, a in rows}):
        print(f"\n{cell}")
        base = None
        for l in labels:
            x = [a for ll, a in rows if ll == l and a["cell"] == cell]
            if not x:
                continue
            t = [a["real_s"] for a in x]
            m = st.mean(t)
            base = base or m
            e = [a["E_in_J"] for a in x if a.get("E_in_J")]
            p = [a["Pmean_in_W"] for a in x if a.get("Pmean_in_W")]
            g = [a["gpu_mhz_mean"] for a in x if a.get("gpu_mhz_mean")]
            print(f"  {l:28} n={len(t)} {m:8.3f} ± {st.stdev(t) if len(t) > 1 else 0:5.3f} s  {100*(m-base)/base:+6.1f} %  gpu {st.mean(g):5.0f} MHz  P {st.mean(p):5.2f} W  E {st.mean(e):7.1f} J")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="e.g. off, on, off2")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cells", default="dna_fast_1k,dna_hac_1k")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(); return 0
    if not a.label:
        ap.error("--label required")
    cfg = bench.load_config()
    variant_runs.register(cfg)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    bench.RUNS_FILE = bench.OUT_ROOT / f"clocks-{a.label}-{stamp}.jsonl"
    cells = a.cells.split(",")
    print(f"[clocks] label {a.label}   {freq_state()}   nvpmodel: {' '.join(bench.sh(['nvpmodel', '-q']).split())}", flush=True)
    bench.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(bench.LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)"); return 2
    block = f"clocks-{a.label}-{stamp}"
    bench.record_idle(cfg, f"clocks-{a.label}:before", block)
    bench.record_canary(cfg, f"clocks-{a.label}:before", block)
    for cell in cells:
        print(f"[clocks] {cell} warm-up", flush=True)
        bench.run_with_retries(cfg, "spec_on", cell, "A", 0, "warmup", block, False, None, False)
        for rep in range(1, a.reps + 1):
            rec = bench.run_with_retries(cfg, "spec_on", cell, "A", rep, "timed", block, False, None, False)
            print(f"   {cell} rep {rep}/{a.reps}: " + (f"Real {rec['real_s']:.3f} s  P {rec.get('Pmean_in_W') or 0:.2f} W  gpu {rec.get('gpu_mhz_mean')} MHz" if rec["status"] == "ok" else f"FAILED {rec.get('error')}"), flush=True)
    bench.record_canary(cfg, f"clocks-{a.label}:after", block)
    bench.record_idle(cfg, f"clocks-{a.label}:after", block)
    print(f"[clocks] end state: {freq_state()}\n[clocks] done -> {bench.RUNS_FILE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
