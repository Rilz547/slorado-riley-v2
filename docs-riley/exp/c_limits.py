#!/usr/bin/env python3
"""Find the largest GPU batch width -C that fits in the Jetson's 8 GB shared memory, per model and stage family. Side experiment, NOT a timing measurement.

c_sweep.py already showed where -C stops fitting on 1,000 reads, but only on a coarse grid (32, 64, ... 512) and with the read queues nearly empty. This probe
(1) runs on the 20,000-read set, where the input queues fill and peak memory is about 1.9 GB higher than on 1k reads, and (2) bisects between the last value that
fits and the first that does not, to --step (default 16).

A value "fits" if slorado is still running --seconds (default 100) after it starts, with the queues full and batches flowing; it does not fit if it aborts earlier
(NvMap / bad_alloc, return code -6 or -9). The process is then stopped. Nothing is timed and nothing is written to disk (-o /dev/null). The limit therefore means
"allocates and starts processing 20k reads", not "completes 20k reads"; --confirm additionally runs the largest fitting value to completion.

The probe child gets oom_score_adj 1000 so that, if the kernel's OOM killer does fire, it picks slorado and not your editor or ssh session.
Stage families are the same as c_sweep.py (the stage's own flags with only -C replaced). The SUP model is not a harness cell; --models sup builds the command from the
HAC cell with the SUP model path and runs the upstream family only.

    python3 c_limits.py --dry-run
    python3 c_limits.py                       # FAST and HAC, upstream + ov_on + spec, about 1.5 h
    python3 c_limits.py --models fast,hac,sup # adds SUP (upstream family only)
    python3 c_limits.py --summary

Do not use the Jetson desktop while this runs. Records: bench-runs/climits-<ts>.jsonl; child console logs: bench-runs/console/climits-*.log.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from c_sweep import FAMILIES, OOM_MARKS  # noqa: E402

# (lo, hi) per (model, family) from the 1k sweep (csweep-20261102T153435): last value that ran, first that did not (None = all ran up to 512).
SEEDS = {("fast", "upstream"): (384, 512), ("fast", "ov_on"): (192, 256), ("fast", "spec"): (512, None),
         ("hac", "upstream"): (256, 384), ("hac", "ov_on"): (192, 256), ("hac", "spec"): (384, 512),
         ("sup", "upstream"): (32, 64)}   # SUP from exp/sup-probe*.log (100 reads): -C 32 ran, 64 and 128 failed with NvMap errors
SUP_MODEL = "models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"


def mem_available_mb() -> tuple[float, float]:
    """(MemAvailable MB, swap used MB)."""
    kv = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":")
        kv[k] = float(v.split()[0]) / 1024
    return kv["MemAvailable"], kv["SwapTotal"] - kv["SwapFree"]


def hwm_gb(pid: int) -> float:
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmHWM"):
                return float(ln.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    subprocess.run(["pkill", "-9", "-f", "slorado basecaller"], capture_output=True)


def command(cfg: dict, model: str, fam: str, v: int) -> list[str]:
    c = bench.parse_cell(cfg, f"dna_{'hac' if model == 'sup' else model}_20k")
    cmd = bench.build_cmd(cfg, fam, c, "/dev/null", None, ["-C", str(v)])
    if model == "sup":
        old = str(bench.model_path(cfg, "dna", "hac"))
        cmd[cmd.index(old)] = str(bench.REPO / SUP_MODEL)
    return cmd


def probe(cfg: dict, model: str, fam: str, v: int, seconds: float, block: str, sink: Path) -> bool:
    log = bench.OUT_ROOT / "console" / f"climits-{block}-{model}-{fam}-C{v}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(cfg, model, fam, v)
    avail0, _ = mem_available_mb()
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                preexec_fn=lambda: open("/proc/self/oom_score_adj", "w").write("1000"), start_new_session=True)
        min_avail, max_swap, hwm = avail0, 0.0, 0.0
        while proc.poll() is None and time.time() - t0 < seconds:
            time.sleep(0.5)
            a, s = mem_available_mb()
            min_avail, max_swap, hwm = min(min_avail, a), max(max_swap, s), max(hwm, hwm_gb(proc.pid))
        alive = proc.poll() is None
        rc = None if alive else proc.returncode
        stop(proc)
    text = log.read_text(errors="replace")
    oom = any(m in text for m in OOM_MARKS) or rc in (-6, -9)
    if alive or rc == 0:
        outcome = "fits"
    elif oom:
        outcome = "oom"
    else:
        outcome = "other-failure"
    rec = {"kind": "limit_probe", "block": block, "model": model, "family": fam, "dataset": "dna_20k", "C": v, "outcome": outcome, "returncode": rc,
           "alive_s": round(time.time() - t0, 1), "cap_s": seconds, "vmhwm_gb": round(hwm, 2), "memavail_min_mb": round(min_avail), "swap_max_mb": round(max_swap),
           "console": str(log.relative_to(bench.REPO)), "cmd": " ".join(cmd)}
    with open(sink, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"   {model:4} {fam:9} -C {v:<5} {outcome:13} rc={rc}  alive {rec['alive_s']:5.1f} s  peak RSS {hwm:.2f} GB  min MemAvailable {min_avail:5.0f} MB  swap {max_swap:5.0f} MB",
          flush=True)
    if outcome == "other-failure":
        print(f"      !! not a memory failure; last console lines:\n" + "\n".join("      " + x for x in text.strip().splitlines()[-5:]), flush=True)
        raise SystemExit(3)
    time.sleep(20)   # let the GPU memory be released before the next probe
    return outcome == "fits"


def search(ok, lo: int, hi: int | None, step: int, max_c: int) -> tuple[int | None, int | None]:
    """Largest fitting value (to --step) and the smallest value known not to fit (None = nothing failed up to max_c)."""
    rnd = lambda x: max(step, int(x) // step * step)   # noqa: E731
    while not ok(lo):                      # the 1k limit does not hold at 20k: go down
        hi = lo
        if lo <= step:
            return None, hi
        lo = rnd(lo / 2)
    if hi is not None and ok(hi):          # the 1k failure does not reproduce: treat it as a fitting value and go up
        lo, hi = hi, None
    while hi is None:                      # find a failing value
        nxt = min(max_c, rnd(lo * 1.25) if rnd(lo * 1.25) > lo else lo + step)
        if nxt <= lo:
            return lo, None
        if ok(nxt):
            lo = nxt
        else:
            hi = nxt
    while hi - lo > step:                  # bisect
        mid = rnd((lo + hi) / 2)
        if mid <= lo:
            break
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def summary() -> None:
    probes, limits = [], {}
    for f in sorted(glob.glob(str(bench.OUT_ROOT / "climits-*.jsonl"))):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("kind") == "limit_probe":
                probes.append(r)
            elif r.get("kind") == "limit":
                limits.setdefault((r["model"], r["family"]), []).append(r)
    print("\nLargest -C that fits on 20,000 reads (alive after the probe time, queues full) and the smallest value that did not:")
    print(f"  {'model':5} {'family':9} {'max fit':>8} {'first fail':>11} {'peak RSS at max fit (GB)':>26} {'min MemAvailable (MB)':>22}")
    for (model, fam), rs in sorted(limits.items()):
        r = rs[-1]
        at = [p for p in probes if p["model"] == model and p["family"] == fam and p["C"] == r["max_fit"] and p["outcome"] == "fits"]
        print(f"  {model:5} {fam:9} {str(r['max_fit']):>8} {str(r['first_fail'] or '> ' + str(r['max_c'])):>11} "
              f"{max((p['vmhwm_gb'] for p in at), default=0):26.2f} {min((p['memavail_min_mb'] for p in at), default=0):22.0f}")
    print("\nAll probes:")
    for p in probes:
        print(f"  {p['model']:5} {p['family']:9} -C {p['C']:<5} {p['outcome']:8} peak RSS {p['vmhwm_gb']:.2f} GB  min MemAvailable {p['memavail_min_mb']} MB  ({p['block']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--models", default="fast,hac")
    ap.add_argument("--seconds", type=float, default=100, help="a value fits if slorado is still running this long after it starts")
    ap.add_argument("--step", type=int, default=16, help="bisection resolution")
    ap.add_argument("--max-c", type=int, default=1024)
    ap.add_argument("--confirm", action="store_true", help="also run the largest fitting value to completion (20k reads; slow for HAC)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
        return 0
    fams, models = a.families.split(","), a.models.split(",")
    combos = [(m, f) for m in models for f in fams if (m, f) in SEEDS and (m != "sup" or f == "upstream")]
    cfg = bench.load_config()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    sink = bench.OUT_ROOT / f"climits-{stamp}.jsonl"
    cells = sorted({f"dna_{'hac' if m == 'sup' else m}_20k" for m, _ in combos})
    snap, problems = bench.preflight(cfg, sorted({f for _, f in combos}), True, cells)
    for m, f in combos:
        print(f"{m:4} {f:9} seed lo/hi {SEEDS[(m, f)]}   " + " ".join(command(cfg, m, f, SEEDS[(m, f)][0])).replace(str(Path.home()), "~")[:230])
    print(f"\n{len(combos)} searches, about 5-8 probes each ({a.seconds:.0f} s for a value that fits, a few seconds plus 20 s settling for one that does not): "
          f"roughly {len(combos) * 6 * (a.seconds / 2 + 20) / 3600:.1f} h   records -> {sink}")
    if problems:
        print("PRE-FLIGHT PROBLEMS:\n  " + "\n  ".join(problems))
    if a.dry_run:
        return 1 if problems else 0
    if problems:
        return 2
    lock = open(bench.LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2
    block = stamp
    for model, fam in combos:
        print(f"[climits] {model} {fam}", flush=True)
        lo, hi = SEEDS[(model, fam)]
        cache: dict[int, bool] = {}

        def ok(v: int, model=model, fam=fam, cache=cache) -> bool:
            if v not in cache:
                cache[v] = probe(cfg, model, fam, v, a.seconds, block, sink)
            return cache[v]

        best, first_fail = search(ok, lo, hi, a.step, a.max_c)
        print(f"[climits] {model} {fam}: largest -C that fits = {best}, first that does not = {first_fail}", flush=True)
        with open(sink, "a") as f:
            f.write(json.dumps({"kind": "limit", "block": block, "model": model, "family": fam, "max_fit": best, "first_fail": first_fail, "max_c": a.max_c, "step": a.step}) + "\n")
        if a.confirm and best and model != "sup":
            bench.RUNS_FILE = sink   # keep anything the harness writes out of runs.jsonl
            rec = bench.run_slorado_measured(cfg, fam, f"dna_{model}_20k", "A", 0, "warmup", block, False, None, False, extra_args=["-C", str(best)])
            print(f"   confirm {model} {fam} -C {best}: {rec['status']}  Real {rec.get('real_s')}  peak RAM {rec.get('peak_ram_gb')} GB", flush=True)
            with open(sink, "a") as f:
                f.write(json.dumps({"kind": "limit_confirm", "model": model, "family": fam, "C": best, "status": rec["status"], "real_s": rec.get("real_s"),
                                    "peak_ram_gb": rec.get("peak_ram_gb")}) + "\n")
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
