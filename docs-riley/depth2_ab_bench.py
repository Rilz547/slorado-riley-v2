#!/usr/bin/env python3
"""
A/B bench: overlap depth=1 vs depth=2.

Runs (per depth):
  - FAST 1k  × 5
  - HAC  1k  × 5
  - FAST 20k × 2

Edits the CONFIG block, then from repo root (venv active):

  python3 docs-riley/depth2_ab_bench.py
  python3 docs-riley/depth2_ab_bench.py --dry-run

Results append to RESULTS_TXT / RESULTS_CSV under docs-riley/.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
SLORADO = REPO / "slorado"

MODEL_FAST = REPO / "models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
MODEL_HAC = REPO / "models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
READS_1K = REPO / "test/PGXXXX230339/reads_1k.blow5"
READS_20K = REPO / "test/PGXXXX230339/reads_20k.blow5"

# Load-balancer / overlap settings that hit ~sub-10s on FAST 1k.
GPU_BATCH = 128
CHUNK_SIZE = 12288
READ_BATCH = 4096
OVERLAP = 150
FLUSH_THRESHOLD = 64
FIXED_C_BATCH = "no"  # narrow partials
OVERLAP_DECODE = "yes"

DEPTHS = (1, 2)

# (tag, model, reads, n_runs)
SUITE = [
    ("fast_1k", MODEL_FAST, READS_1K, 5),
    ("hac_1k", MODEL_HAC, READS_1K, 5),
    ("fast_20k", MODEL_FAST, READS_20K, 2),
]

WARMUP_PER_DEPTH = True  # one FAST 1k warmup per depth before timed suite

OUT_DIR = Path(__file__).resolve().parent
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_TXT = OUT_DIR / f"depth2_ab_results_{STAMP}.txt"
RESULTS_CSV = OUT_DIR / f"depth2_ab_results_{STAMP}.csv"

REAL_TIME_RE = re.compile(
    r"\[main\] Real time: ([\d.]+) sec; CPU time: ([\d.]+) sec; Peak RAM: ([\d.]+) GB"
)
PROC_RE = re.compile(r"\[basecaller_main\] data processing: ([\d.]+) sec")


def build_cmd(model: Path, reads: Path, depth: int) -> list[str]:
    return [
        str(SLORADO),
        "basecaller",
        "-C",
        str(GPU_BATCH),
        "-c",
        str(CHUNK_SIZE),
        "-K",
        str(READ_BATCH),
        "-p",
        str(OVERLAP),
        f"--overlap-decode={OVERLAP_DECODE}",
        f"--overlap-depth={depth}",
        f"--fixed-c-batch={FIXED_C_BATCH}",
        "--flush-threshold",
        str(FLUSH_THRESHOLD),
        "-o",
        "/dev/null",
        str(model),
        str(reads),
    ]


def parse_metrics(text: str) -> dict:
    out: dict = {}
    m = REAL_TIME_RE.search(text)
    if m:
        out["real_s"] = float(m.group(1))
        out["cpu_s"] = float(m.group(2))
        out["peak_ram_gb"] = float(m.group(3))
    m = PROC_RE.search(text)
    if m:
        out["proc_s"] = float(m.group(1))
    return out


def run_one(cmd: list[str]) -> tuple[dict, str]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall = time.time() - t0
    text = proc.stdout or ""
    metrics = parse_metrics(text)
    metrics["exit_code"] = proc.returncode
    metrics["host_wall_s"] = wall
    if proc.returncode != 0 and "real_s" not in metrics:
        metrics["status"] = "fail"
    else:
        metrics["status"] = "ok" if proc.returncode == 0 else "fail"
    return metrics, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print plan only")
    args = ap.parse_args()

    if not SLORADO.is_file():
        print(f"error: missing binary {SLORADO} — build first", file=sys.stderr)
        return 1
    for p in (MODEL_FAST, MODEL_HAC, READS_1K, READS_20K):
        if not p.exists():
            print(f"error: missing {p}", file=sys.stderr)
            return 1

    plan = []
    if WARMUP_PER_DEPTH:
        for d in DEPTHS:
            plan.append(("warmup_fast_1k", MODEL_FAST, READS_1K, d, 0))
    for d in DEPTHS:
        for tag, model, reads, n in SUITE:
            for run in range(1, n + 1):
                plan.append((tag, model, reads, d, run))

    print(f"Repo:    {REPO}")
    print(f"Binary:  {SLORADO}")
    print(f"Results: {RESULTS_TXT}")
    print(f"CSV:     {RESULTS_CSV}")
    print(f"Planned runs: {len(plan)} (including warmups)")
    print()
    for tag, model, reads, d, run in plan:
        label = "warmup" if run == 0 else f"run={run}"
        print(f"  depth={d}  {tag:12s}  {label}")
    if args.dry_run:
        return 0

    fieldnames = [
        "stamp_utc",
        "tag",
        "depth",
        "run",
        "status",
        "real_s",
        "cpu_s",
        "peak_ram_gb",
        "proc_s",
        "host_wall_s",
        "exit_code",
        "cmd",
    ]

    with RESULTS_TXT.open("w", encoding="utf-8") as ftxt, RESULTS_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        header = (
            f"# depth2 A/B bench\n"
            f"# started_utc={STAMP}\n"
            f"# C={GPU_BATCH} c={CHUNK_SIZE} K={READ_BATCH} p={OVERLAP} "
            f"flush={FLUSH_THRESHOLD} fixed-c={FIXED_C_BATCH} overlap-decode={OVERLAP_DECODE}\n"
            f"# suite: fast_1k×5, hac_1k×5, fast_20k×2  × depths {list(DEPTHS)}\n\n"
        )
        ftxt.write(header)
        ftxt.flush()
        print(header, end="")

        for tag, model, reads, depth, run in plan:
            cmd = build_cmd(model, reads, depth)
            cmd_s = " ".join(cmd)
            is_warmup = run == 0
            label = f"depth={depth} {tag} " + ("warmup" if is_warmup else f"run={run}")
            print(f"\n>>> {label}", flush=True)
            print(f"    {cmd_s}", flush=True)

            metrics, text = run_one(cmd)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Keep last ~40 lines of slorado log in the text dump for context.
            tail = "\n".join(text.strip().splitlines()[-40:])
            block = (
                f"=== {label}  status={metrics.get('status')}  "
                f"real_s={metrics.get('real_s', 'NA')}  "
                f"cpu_s={metrics.get('cpu_s', 'NA')}  "
                f"ram_gb={metrics.get('peak_ram_gb', 'NA')}  "
                f"proc_s={metrics.get('proc_s', 'NA')} ===\n"
                f"CMD: {cmd_s}\n"
                f"{tail}\n\n"
            )
            ftxt.write(block)
            ftxt.flush()
            print(
                f"    -> status={metrics.get('status')} real_s={metrics.get('real_s', 'NA')} "
                f"ram={metrics.get('peak_ram_gb', 'NA')}GB",
                flush=True,
            )

            if not is_warmup:
                writer.writerow(
                    {
                        "stamp_utc": stamp,
                        "tag": tag,
                        "depth": depth,
                        "run": run,
                        "status": metrics.get("status"),
                        "real_s": metrics.get("real_s", ""),
                        "cpu_s": metrics.get("cpu_s", ""),
                        "peak_ram_gb": metrics.get("peak_ram_gb", ""),
                        "proc_s": metrics.get("proc_s", ""),
                        "host_wall_s": f"{metrics.get('host_wall_s', 0):.3f}",
                        "exit_code": metrics.get("exit_code"),
                        "cmd": cmd_s,
                    }
                )
                fcsv.flush()

        ftxt.write(f"# finished_utc={datetime.now(timezone.utc).isoformat()}\n")
        ftxt.flush()

    print(f"\nDone.\nText: {RESULTS_TXT}\nCSV:  {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
