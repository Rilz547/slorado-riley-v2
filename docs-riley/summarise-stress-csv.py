#!/usr/bin/env python3
"""
Pick the best runs from a stress-test CSV.

Groups by gpu batch (-C), keeps the top N fastest per group (deduped by
read batch first if you've run the same combo more than once), then writes
a smaller CSV sorted by gpu batch then read batch.

  python docs-riley/summarise-stress-csv.py docs-riley/stress-results/hac_1k.csv
  python docs-riley/summarise-stress-csv.py hac_1k.csv -o hac_1k_picks.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

TOP_PER_GPU = 3

OUT_COLUMNS = [
    "real_time_sec", "read_batch", "gpu_batch", "status",
    "reads_per_sec", "peak_ram_gb", "chunk_size",
    "threads", "overlap", "byte_limit", "run_id",
]


def parse_row(row):
    try:
        row["_read_batch"] = int(row["read_batch"])
        row["_gpu_batch"] = int(row["gpu_batch"])
    except (KeyError, ValueError, TypeError):
        return None
    if row.get("status") == "ok":
        try:
            row["_time"] = float(row["real_time_sec"])
        except (ValueError, TypeError):
            return None
    else:
        row["_time"] = 0
        row["real_time_sec"] = 0
        row["reads_per_sec"] = 0
        row["peak_ram_gb"] = 0
    return row


def load_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        parsed = parse_row(row)
        if parsed is not None:
            out.append(parsed)
    return out


def best_per_combo(rows):
    """Same -K/-C run more than once? Keep the fastest successful one."""
    seen = {}
    for row in rows:
        key = (row["_gpu_batch"], row["_read_batch"])
        if key not in seen:
            seen[key] = row
            continue
        prev = seen[key]
        # ok beats failed; between two ok keep faster
        if prev.get("status") != "ok" and row.get("status") == "ok":
            seen[key] = row
        elif row.get("status") == "ok" and prev.get("status") == "ok" and row["_time"] < prev["_time"]:
            seen[key] = row
    return list(seen.values())


def pick_top_per_gpu(rows, n=TOP_PER_GPU):
    by_gpu = defaultdict(list)
    for row in rows:
        by_gpu[row["_gpu_batch"]].append(row)

    picked = []
    for gpu in sorted(by_gpu):
        # fastest n per gpu batch, then line up by read batch for the table
        group = sorted(by_gpu[gpu], key=lambda r: (r["_time"] if r.get("status") == "ok" else 1e9))[:n]
        group.sort(key=lambda r: r["_read_batch"])
        picked.extend(group)
    return picked


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def default_output(input_path):
    p = Path(input_path)
    return p.with_name(f"{p.stem}_summary{p.suffix}")


def main():
    p = argparse.ArgumentParser(description="summarise stress-test csv results")
    p.add_argument("input", help="csv from stress-test.py")
    p.add_argument("-o", "--output", help="output csv (default: <input>_summary.csv)")
    p.add_argument("-n", type=int, default=TOP_PER_GPU, help=f"fastest runs per gpu batch (default {TOP_PER_GPU})")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 1

    rows = load_rows(src)
    if not rows:
        print("no rows in that file", file=sys.stderr)
        return 1

    rows = best_per_combo(rows)
    picked = pick_top_per_gpu(rows, args.n)

    out = Path(args.output) if args.output else default_output(src)
    write_csv(out, picked)

    print(f"read {len(rows)} unique combos from {src}")
    print(f"wrote {len(picked)} rows -> {out}")
    print()
    print(f"{'time':>10}  {'read_batch':>10}  {'gpu_batch':>9}")
    print("-" * 34)
    for row in picked:
        t = row["_time"]
        mark = " (failed)" if row.get("status") != "ok" else ""
        print(f"{t:10.3f}  {row['_read_batch']:10}  {row['_gpu_batch']:9}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
