#!/usr/bin/env python3
"""Single-option sweeps through the frozen harness's measurement: -C (GPU batch width) or -c (chunk length). Side experiment, NOT part of the frozen campaign.

Same measurement function as the campaign (bench.run_slorado_measured: temperature gate, power window, idle baselines and canary around the block, 25 W / clocks-off
pre-flight), but a failed run is NOT retried: an out-of-memory failure is a result ("too big"). Records go to bench-runs/csweep-<ts>.jsonl, never runs.jsonl.
FASTQs kept with --keep go to bench-runs/kept-sweeps/ (NOT kept/), named output_<model>_<size>-<dtype>-<family>_<flag><value>.fastq, for the PC accuracy step
(fetch-and-minimap.py --glob "docs-riley/bench-runs/kept-sweeps/output_*.fastq").

--flag C  (default)  GPU batch width. Families (the stage's own flags with only -C replaced; a later -C overrides the earlier one):
  upstream  754a74d, no overlap, always C-wide batches          C is the real batch width; the June baseline used 256 for FAST and 128 for HAC
  ov_on     7cec6c5, overlap decode on, always C-wide batches   shows the memory wall with overlap (two input buffers + decode buffers)
  spec      49ac641 final config (flush threshold 64 FAST / 128 HAC)   a batch never exceeds the flush threshold, so above it C only sets the allocation size
--flag c             chunk length in samples (slorado rounds it down to a multiple of the model stride, 6: -c 10000 runs as 9996). Run with --families upstream --keep.
                     Changes where chunks are stitched, so it affects accuracy as well as time; compare samples/s, not Real time, because the chunk count changes.

Per (family, cell): one warm-up per value, ascending (it also finds the memory limit; the first value that fails removes itself and every larger one), then --reps timed
mode-A runs interleaved over the values, then with --keep one mode-B run per value whose FASTQ is kept.

    python3 c_sweep.py --dry-run                                  # commands, estimate, pre-flight
    python3 c_sweep.py                                            # -C 32..512 on upstream,ov_on,spec x FAST 1k, HAC 1k, 3 reps  (about 1.5 h)
    python3 c_sweep.py --flag c --families upstream --reps 3 --keep                          # -c 6000..16000, FAST 1k and HAC 1k, accuracy FASTQs  (about 30 min)
    python3 c_sweep.py --flag c --families upstream --cells dna_fast_20k --reps 0 --no-warmup --keep   # 20k: one kept mode-B run per value (about 20 min)
    python3 c_sweep.py --summary                                  # tables from every csweep-*.jsonl

Do not use the Jetson desktop while this runs. Large -C or -c values can exhaust the 8 GB shared memory; earlier tests ended with an NvMap error (clean abort), but
keep ssh sessions light and expect the run to slow while swap is used.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench  # noqa: E402

DEFAULT_VALUES = {"C": [32, 64, 96, 128, 192, 256, 384, 512], "c": [6000, 8000, 10000, 12288, 16000]}
FAMILIES = ["upstream", "ov_on", "spec"]
CELLS = ["dna_fast_1k", "dna_hac_1k"]
PRIOR_S = {("fast", "1k"): 13, ("hac", "1k"): 50, ("fast", "20k"): 250, ("hac", "20k"): 950}   # upstream-like run time per cell, rough
OOM_MARKS = ("NvMap", "out of memory", "bad_alloc", "Killed")
KEPT_SWEEPS = bench.OUT_ROOT / "kept-sweeps"


def looks_like_oom(rec: dict) -> bool:
    try:
        text = (bench.REPO / rec["console"]).read_text(errors="replace")
    except (OSError, KeyError):
        return False
    return any(m in text for m in OOM_MARKS)


def summary() -> None:
    rows = []
    for f in sorted(glob.glob(str(bench.OUT_ROOT / "csweep-*.jsonl"))):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("kind") in ("timed", "warmup") and "sweep_value" in r:
                rows.append(r)
    for flag in ("C", "c"):
        for cell in sorted({r["cell"] for r in rows if r["sweep_flag"] == flag}):
            for fam in FAMILIES:
                sel = [r for r in rows if r["sweep_flag"] == flag and r["cell"] == cell and r["stage"] == fam]
                if not sel:
                    continue
                print(f"\n-{flag} sweep   {cell}   {fam}")
                print(f"  {'-' + flag:>6} {'ok/runs':>8} {'Real s (mode A)':>17} {'Msamp/s':>8} {'infer':>7} {'decode':>7} {'peak RAM GB':>11} {'swap MB':>8} {'B run s':>8}  outcome")
                table = []
                for v in sorted({r["sweep_value"] for r in sel}):
                    rs = [r for r in sel if r["sweep_value"] == v]
                    ok = [r for r in rs if r["status"] == "ok" and r["kind"] == "timed" and r["mode"] == "A"]
                    okb = [r for r in rs if r["status"] == "ok" and r["mode"] == "B"]
                    bad = [r for r in rs if r["status"] != "ok"]
                    t = [r["real_s"] for r in ok]
                    table.append((v, ok, okb, rs, bad, t, st.mean(t) if t else None))
                best = min((m for *_, m in table if m is not None), default=None)
                for v, ok, okb, rs, bad, t, m in table:
                    inf = [r["timings"].get("inference") for r in ok if r.get("timings")]
                    dec = [r["timings"].get("decode") for r in ok if r.get("timings")]
                    pk = [r["peak_ram_gb"] for r in ok + okb if r.get("peak_ram_gb")]
                    sw = [r["swap_used_mb_max"] for r in ok + okb if r.get("swap_used_mb_max") is not None]
                    sps = [r["samples_per_s"] for r in ok if r.get("samples_per_s")]
                    if bad:
                        outcome = "OOM (NvMap/alloc)" if any(looks_like_oom(r) for r in bad) else f"FAILED ({bad[0].get('error')})"
                    else:
                        outcome = "" if m is None or flag == "c" else f"{100 * (m - best) / best:+.1f} % vs best"
                    rt = f"{m:7.2f} ± {st.stdev(t) if len(t) > 1 else 0:5.2f}" if t else "-"
                    print(f"  {v:>6} {len(ok):>3}/{len(rs):<4} {rt:>17} {st.mean(sps) / 1e6 if sps else 0:8.2f} {st.mean(inf) if inf else 0:7.1f} {st.mean(dec) if dec else 0:7.1f}"
                          f" {st.mean(pk) if pk else 0:11.2f} {max(sw) if sw else 0:8.0f} {okb[0]['real_s'] if okb else 0:8.1f}  {outcome}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flag", choices=["C", "c"], default="C", help="option to sweep: C = GPU batch width, c = chunk length")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--values", "--c", dest="values", help="comma list of values, ascending (default depends on --flag)")
    ap.add_argument("--reps", type=int, default=3, help="timed mode-A runs per value (0 = none)")
    ap.add_argument("--keep", action="store_true", help="also one mode-B run per value, FASTQ kept in kept-sweeps/ for the PC accuracy step")
    ap.add_argument("--no-warmup", action="store_true", help="skip the per-value warm-up (use for 20k; the memory limit is then found by the first real run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
        return 0
    fams, cells = a.families.split(","), a.cells.split(",")
    vals = sorted(int(x) for x in a.values.split(",")) if a.values else DEFAULT_VALUES[a.flag]
    cfg = bench.load_config()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    bench.RUNS_FILE = bench.OUT_ROOT / f"csweep-{stamp}.jsonl"
    bench.KEPT_DIR = KEPT_SWEEPS

    snap, problems = bench.preflight(cfg, fams, True, cells)
    est = 0.0
    for cell in cells:
        c = bench.parse_cell(cfg, cell)
        for fam in fams:
            est += PRIOR_S[(c["model"], c["ds"]["size"])] * len(vals) * ((0 if a.no_warmup else 1) + a.reps + (1 if a.keep else 0))
            cmd = bench.build_cmd(cfg, fam, c, "/dev/null", None, [f"-{a.flag}", str(vals[0])])
            print(f"{cell:12} {fam:9} " + " ".join(cmd).replace(str(Path.home()), "~")[:260])
    print(f"\n-{a.flag} values {vals}, {a.reps} reps{'' if a.no_warmup else ' (+1 warm-up per value)'}{' + 1 kept mode-B run per value' if a.keep else ''}; "
          f"rough upper bound {est / 3600:.1f} h (failed values are skipped)   records -> {bench.RUNS_FILE}")
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
    block = f"csweep-{stamp}"
    bench.record_idle(cfg, "csweep:before", block)
    bench.record_canary(cfg, "csweep:before", block)

    def run(fam: str, cell: str, v: int, mode: str, rep: int, kind: str) -> dict:
        c = bench.parse_cell(cfg, cell)
        keep_as = f"output_{c['model']}_{c['ds']['size']}-{c['dtype']}-{fam}_{a.flag}{v}.fastq" if mode == "B" else None
        rec = bench.run_slorado_measured(cfg, fam, cell, mode, rep, kind, block, mode == "B", None, False, extra_args=[f"-{a.flag}", str(v)], keep_as=keep_as)
        rec.update({"attempt": 1, "sweep_flag": a.flag, "sweep_value": v})
        bench.append_record(rec)
        if rec["status"] == "ok":
            print(f"   {cell} {fam:9} -{a.flag} {v:<6} {kind:7} {mode}  Real {rec['real_s']:.3f} s  peak RAM {rec.get('peak_ram_gb')} GB  swap {rec.get('swap_used_mb_max')} MB", flush=True)
        else:
            print(f"   {cell} {fam:9} -{a.flag} {v:<6} {kind:7} {mode}  FAILED: {rec.get('error')}{'  (memory)' if looks_like_oom(rec) else ''}", flush=True)
            subprocess.run(["pkill", "-9", "-f", "slorado basecaller"], capture_output=True)
            time.sleep(20)
        return rec

    for cell in cells:
        for fam in fams:
            print(f"[csweep] {cell} {fam}", flush=True)
            alive = list(vals)
            if not a.no_warmup:
                for v in list(alive):
                    if v in alive and run(fam, cell, v, "A", 0, "warmup")["status"] != "ok":
                        alive = [x for x in alive if x < v]
            for rep in range(1, a.reps + 1):
                for v in list(alive):
                    if v in alive and run(fam, cell, v, "A", rep, "timed")["status"] != "ok":
                        alive = [x for x in alive if x < v]
            if a.keep:
                for v in list(alive):
                    if v in alive and run(fam, cell, v, "B", 1, "timed")["status"] != "ok":
                        alive = [x for x in alive if x < v]
    bench.record_canary(cfg, "csweep:after", block)
    bench.record_idle(cfg, "csweep:after", block)
    print(f"[csweep] done -> {bench.RUNS_FILE}")
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
