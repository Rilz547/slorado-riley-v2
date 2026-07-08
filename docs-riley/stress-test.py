#!/usr/bin/env python3
"""
Sweep slorado basecaller flags to find fast combos.

  python docs-riley/stress-test.py
  python docs-riley/stress-test.py --dry-run

Run from repo root with the venv active.

None in a sweep array = don't pass that flag, slorado uses its default.
  THREADS = [None]         -> no -t on the command line
  THREADS = [None, 4, 8] -> default plus 4 and 8

  Array        Flag   default
  THREADS      -t     8
  READ_BATCH   -K     4096
  GPU_BATCH    -C     512
  CHUNK_SIZE   -c     12288
  OVERLAP      -p     150
  BYTE_LIMIT   -B     512M

CSV results (optional): set STORE_OUTPUT_RESULTS = True.
Goes to docs-riley/stress-results/ as fast_1k.csv, hac_1k.csv, etc.
"""

import csv
import itertools
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SLORADO = "./slorado"
TITLE = "Riley's Stress Testa v1.0"
VERSION_TAG = "v1.0"

MODEL_FAST = "models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
MODEL_HAC = "models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
READS_1K = "test/PGXXXX230339/reads_1k.blow5"
READS_20K = "test/PGXXXX230339/reads_20k.blow5"

# change these before a run
MODEL = MODEL_HAC
INPUT = READS_20K

PROFILE_CSV = {
    (MODEL_FAST, READS_1K):  "fast_1k.csv",
    (MODEL_FAST, READS_20K): "fast_20k.csv",
    (MODEL_HAC, READS_1K):   "hac_1k.csv",
    (MODEL_HAC, READS_20K):  "hac_20k.csv",
}

DEFAULTS = {
    "threads": 8, "read_batch": 4096, "gpu_batch": 512,
    "chunk_size": 12288, "overlap": 150, "byte_limit": "512M",
}

# sweep arrays — None means leave that flag alone
THREADS = [None]
# READ_BATCH = [1024, 2048, 4096]
READ_BATCH = [1024]
# GPU_BATCH = [64, 96, 128, 160, 192, 224, 256, 288]
GPU_BATCH = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
# GPU_BATCH = [64, 96, 128]
CHUNK_SIZE = [12288]
OVERLAP = [None]
BYTE_LIMIT = [None]

# behaviour
COOLDOWN_SEC = 2
TIMEOUT_SEC = 1500
DRY_RUN = False
TOP_N = 10

RESULTS_DIR = "docs-riley/stress-results"
STORE_OUTPUT_RESULTS = True
SORT_CSV_AFTER_RUN = True

# basecalls go nowhere — don't want disk I/O skewing the timings
OUTPUT_SINK = "/dev/null"

CSV_COLUMNS = [
    "timestamp", "version_tag", "run_id", "status",
    "real_time_sec", "reads_per_sec", "peak_ram_gb", "cpu_time_sec", "processing_sec",
    "threads", "read_batch", "gpu_batch", "chunk_size", "overlap", "byte_limit",
]

REAL_TIME_RE = re.compile(
    r"\[main\] Real time: ([\d.]+) sec; CPU time: ([\d.]+) sec; Peak RAM: ([\d.]+) GB"
)
PROC_TIME_RE = re.compile(r"\[basecaller_main\] data processing: ([\d.]+) sec")
ENTRIES_RE = re.compile(r"\[basecaller_main\] total entries: (\d+)")

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def which_csv(model, input_path):
    key = (model, input_path)
    if key not in PROFILE_CSV:
        raise ValueError(f"don't know which csv for {model} + {input_path}")
    return Path(RESULTS_DIR) / PROFILE_CSV[key]


def resolved(cfg):
    return {k: DEFAULTS[k] if v is None else v for k, v in cfg.items()}


def show_val(v):
    return "default" if v is None else str(v)


def show_flags(cfg):
    return (
        f"-t {show_val(cfg['threads'])} -K {show_val(cfg['read_batch'])} "
        f"-C {show_val(cfg['gpu_batch'])} -c {show_val(cfg['chunk_size'])} "
        f"-p {show_val(cfg['overlap'])} -B {show_val(cfg['byte_limit'])}"
    )


def slorado_cmd(model, input_path, cfg):
    cmd = [SLORADO, "basecaller"]
    for flag, val in [
        ("-t", cfg["threads"]), ("-K", cfg["read_batch"]), ("-C", cfg["gpu_batch"]),
        ("-c", cfg["chunk_size"]), ("-p", cfg["overlap"]), ("-B", cfg["byte_limit"]),
    ]:
        if val is not None:
            cmd.extend([flag, str(val)])
    cmd.extend(["-o", OUTPUT_SINK, model, input_path])
    return cmd


def parse_log(text):
    out = {}
    m = REAL_TIME_RE.search(text)
    if m:
        out["real_time_sec"] = float(m.group(1))
        out["cpu_time_sec"] = float(m.group(2))
        out["peak_ram_gb"] = float(m.group(3))
    m = PROC_TIME_RE.search(text)
    if m:
        out["processing_sec"] = float(m.group(1))
    m = ENTRIES_RE.search(text)
    if m:
        out["total_entries"] = int(m.group(1))
    if out.get("real_time_sec") and out.get("total_entries"):
        out["reads_per_sec"] = out["total_entries"] / out["real_time_sec"]
    return out


def why_failed(metrics):
    if metrics.get("status") == "timeout":
        return f"timed out after {TIMEOUT_SEC}s"
    err = (metrics.get("error") or "").strip()
    if not err:
        return metrics.get("status", "unknown")
    for line in err.splitlines():
        line = line.strip()
        if line and ("ERROR" in line or "CUDA" in line or "Killed" in line):
            return line[:200]
    return err.splitlines()[-1][:200]


def say_passed(m):
    rps = m.get("reads_per_sec")
    ram = m.get("peak_ram_gb")
    rt = m.get("real_time_sec")
    print(f"    {GREEN}PASSED{RESET}  "
          f"{f'{rps:.1f} reads/s' if rps else '? reads/s'}  "
          f"{f'{ram:.3f} GB' if ram is not None else '? GB'}  "
          f"({f'{rt:.3f}s' if rt else '?'})")


def say_failed(m):
    print(f"    {RED}FAILED{RESET}  {why_failed(m)}")


def run_combo(model, input_path, cfg):
    print(f"  {show_flags(cfg)}")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            slorado_cmd(model, input_path, cfg),
            capture_output=True, text=True, timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {TIMEOUT_SEC}s"}
    except OSError as e:
        return {"status": "error", "error": str(e)}

    log = p.stdout + p.stderr
    m = {"status": "ok" if p.returncode == 0 else "failed", "wall_sec": round(time.perf_counter() - t0, 3)}
    m.update(parse_log(log))
    if p.returncode != 0:
        m["error"] = (p.stderr or p.stdout)[-800:].strip()
    return m


def csv_row(run_id, cfg, m):
    r = resolved(cfg)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version_tag": VERSION_TAG,
        "run_id": run_id,
        "status": m.get("status", ""),
        "real_time_sec": m.get("real_time_sec", ""),
        "reads_per_sec": m.get("reads_per_sec", ""),
        "peak_ram_gb": m.get("peak_ram_gb", ""),
        "cpu_time_sec": m.get("cpu_time_sec", ""),
        "processing_sec": m.get("processing_sec", ""),
        "threads": r["threads"], "read_batch": r["read_batch"],
        "gpu_batch": r["gpu_batch"], "chunk_size": r["chunk_size"],
        "overlap": r["overlap"], "byte_limit": r["byte_limit"],
    }


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def sort_csv(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    rows.sort(key=lambda r: float(r.get("real_time_sec") or 1e9))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def print_leaderboard(runs, n=TOP_N):
    ok = [r for r in runs if r.get("status") == "ok" and "real_time_sec" in r]
    if not ok:
        print("\nno successful runs to rank")
        return
    ok.sort(key=lambda r: r["real_time_sec"])
    print(f"\ntop {min(n, len(ok))} fastest:")
    print("-" * 72)
    for i, r in enumerate(ok[:n], 1):
        c = r["config"]
        rps = r.get("reads_per_sec")
        print(f"{i:2}. {r['real_time_sec']:7.3f}s  "
              f"{f'{rps:.1f} reads/s' if rps else '?':>14}  "
              f"ram {r.get('peak_ram_gb', '?'):>5} GB  "
              f"{show_flags(c)}")


def main():
    global DRY_RUN
    if "--dry-run" in sys.argv:
        DRY_RUN = True

    for label, path in [("binary", SLORADO), ("model", MODEL), ("input", INPUT)]:
        if not Path(path).exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 1

    combos = list(itertools.product(
        THREADS, READ_BATCH, GPU_BATCH, CHUNK_SIZE, OVERLAP, BYTE_LIMIT,
    ))
    csv_out = which_csv(MODEL, INPUT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    results = []

    print(TITLE)
    print(f"{len(combos)} combinations  |  model: {MODEL}  |  input: {INPUT}")
    if STORE_OUTPUT_RESULTS:
        print(f"csv: {csv_out}")
    print("-" * 72)

    for i, (t, k, c, chunk, overlap, bl) in enumerate(combos, 1):
        cfg = {
            "threads": t, "read_batch": k, "gpu_batch": c,
            "chunk_size": chunk, "overlap": overlap, "byte_limit": bl,
        }
        run_id = f"{stamp}_{i:04d}"

        print(f"[{i}/{len(combos)}]", end=" ")
        if DRY_RUN:
            print(f"  {show_flags(cfg)}")
            continue

        m = run_combo(MODEL, INPUT, cfg)
        results.append({"config": cfg, **m})

        if m.get("status") == "ok":
            say_passed(m)
        else:
            say_failed(m)

        if STORE_OUTPUT_RESULTS:
            append_csv(csv_out, csv_row(run_id, cfg, m))

        if COOLDOWN_SEC > 0 and i < len(combos):
            time.sleep(COOLDOWN_SEC)

    if DRY_RUN:
        print(f"\ndry run — {len(combos)} combos, nothing executed")
        return 0

    if STORE_OUTPUT_RESULTS and SORT_CSV_AFTER_RUN:
        sort_csv(csv_out)

    print(f"\nfinished — {len(results)} runs")
    print_leaderboard(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
