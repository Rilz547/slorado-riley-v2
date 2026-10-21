#!/usr/bin/env python3
"""Parse tegrastats logfile → samples.csv + summary.csv (+ append all_summaries.csv)."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

# Timestamp at start of line (MM-DD-YYYY HH:MM:SS or similar)
TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\s+(.*)$")
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
# cpu@46.187C  OR  CPU@53.562C
TEMP_RE = re.compile(r"([A-Za-z0-9]+)@(-?\d+(?:\.\d+)?)C")
# VDD_IN 5096mW/5096mW
POWER_RE = re.compile(r"((?:VDD|VIN)_[A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")


def parse_cpu_field(s: str) -> tuple[list[float], float | None]:
    """Return (per-core %, mean %). Entries like '2%@729' or 'off'."""
    vals: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part or part.lower() == "off":
            continue
        m = re.match(r"(\d+(?:\.\d+)?)%", part)
        if m:
            vals.append(float(m.group(1)))
    mean = statistics.mean(vals) if vals else None
    return vals, mean


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    m = TS_RE.match(line)
    if not m:
        return None
    ts, rest = m.group(1), m.group(2)
    row: dict = {"timestamp": ts, "raw": rest}

    rm = RAM_RE.search(rest)
    if rm:
        row["ram_used_mb"] = int(rm.group(1))
        row["ram_total_mb"] = int(rm.group(2))

    sm = SWAP_RE.search(rest)
    if sm:
        row["swap_used_mb"] = int(sm.group(1))
        row["swap_total_mb"] = int(sm.group(2))

    cm = CPU_RE.search(rest)
    if cm:
        cores, mean = parse_cpu_field(cm.group(1))
        row["cpu_mean_pct"] = mean
        for i, v in enumerate(cores):
            row[f"cpu{i}_pct"] = v

    gm = GR3D_RE.search(rest)
    if gm:
        row["gr3d_pct"] = float(gm.group(1))

    for name, cur, avg in POWER_RE.findall(rest):
        row[f"{name}_mW"] = int(cur)
        row[f"{name}_avg_mW"] = int(avg)

    for name, val in TEMP_RE.findall(rest):
        # skip fake -256C sensors
        f = float(val)
        if f < -100:
            continue
        row[f"temp_{name.lower()}_C"] = f

    # Total power: prefer VDD_IN (module input on Orin NX/Nano); else sum rails.
    power_keys = [k for k in row if k.startswith("VDD_") and k.endswith("_mW") and not k.endswith("_avg_mW")]
    power_keys += [k for k in row if k.startswith("VIN_") and k.endswith("_mW") and not k.endswith("_avg_mW")]
    if "VDD_IN_mW" in row:
        row["power_total_mW"] = row["VDD_IN_mW"]
        row["power_total_source"] = "VDD_IN"
    elif power_keys:
        row["power_total_mW"] = sum(row[k] for k in power_keys)
        row["power_total_source"] = "sum_rails"
    else:
        row["power_total_mW"] = None
        row["power_total_source"] = ""

    if power_keys:
        row["power_sum_rails_mW"] = sum(row[k] for k in power_keys)

    return row


def load_samples(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            r = parse_line(line)
            if r is not None:
                rows.append(r)
    return rows


def summarize(rows: list[dict], meta: dict, dt_s: float = 1.0) -> dict:
    if not rows:
        raise SystemExit("no tegrastats samples parsed")

    powers = [r["power_total_mW"] for r in rows if r.get("power_total_mW") is not None]
    if not powers:
        raise SystemExit("no power rails found in tegrastats log")

    energy_j = sum(p * dt_s / 1000.0 for p in powers)  # mW·s → J
    gr3d = [r["gr3d_pct"] for r in rows if r.get("gr3d_pct") is not None]
    ram = [r["ram_used_mb"] for r in rows if r.get("ram_used_mb") is not None]
    swap = [r["swap_used_mb"] for r in rows if r.get("swap_used_mb") is not None]

    temp_cols = [k for k in rows[0] if k.startswith("temp_") and k.endswith("_C")]
    max_temps = {}
    for col in sorted({k for r in rows for k in r if k.startswith("temp_") and k.endswith("_C")}):
        vals = [r[col] for r in rows if col in r]
        if vals:
            max_temps[col] = max(vals)

    out = {
        "label": meta.get("label", ""),
        "model": meta.get("model", ""),
        "size": meta.get("size", ""),
        "git_commit": meta.get("git_commit", ""),
        "git_branch": meta.get("git_branch", ""),
        "nvpmodel": meta.get("nvpmodel", ""),
        "wall_s": meta.get("wall_s", ""),
        "n_samples": len(rows),
        "dt_s": dt_s,
        "power_total_source": rows[0].get("power_total_source", ""),
        "peak_power_mW": max(powers),
        "mean_power_mW": statistics.mean(powers),
        "min_power_mW": min(powers),
        "energy_J": round(energy_j, 3),
        "energy_Wh": round(energy_j / 3600.0, 6),
        "peak_ram_mb": max(ram) if ram else "",
        "peak_swap_mb": max(swap) if swap else "",
        "peak_gr3d_pct": max(gr3d) if gr3d else "",
        "mean_gr3d_pct": round(statistics.mean(gr3d), 2) if gr3d else "",
        "command": meta.get("command", ""),
    }
    for k, v in max_temps.items():
        out[f"max_{k}"] = round(v, 2)
    return out


def write_samples_csv(rows: list[dict], path: Path) -> None:
    # Stable column order
    prefer = [
        "timestamp",
        "ram_used_mb",
        "ram_total_mb",
        "swap_used_mb",
        "swap_total_mb",
        "cpu_mean_pct",
        "gr3d_pct",
        "power_total_mW",
        "power_total_source",
        "power_sum_rails_mW",
    ]
    keys: list[str] = []
    seen = set()
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    all_keys.discard("raw")
    for k in prefer:
        if k in all_keys:
            keys.append(k)
            seen.add(k)
    for k in sorted(all_keys):
        if k not in seen:
            keys.append(k)
            seen.add(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def append_all_summaries(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fields = list(reader.fieldnames)
                for k in summary:
                    if k not in fields:
                        fields.append(k)

    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: summary.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tegrastats_log", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--all-summaries", type=Path, required=True)
    ap.add_argument("--dt", type=float, default=1.0, help="sample interval seconds")
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--size", default="")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--git-branch", default="")
    ap.add_argument("--nvpmodel", default="")
    ap.add_argument("--wall-s", default="")
    ap.add_argument("--command", default="")
    args = ap.parse_args()

    rows = load_samples(args.tegrastats_log)
    if not rows:
        print(f"ERROR: no samples parsed from {args.tegrastats_log}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.csv"
    write_samples_csv(rows, samples_path)

    meta = {
        "label": args.label,
        "model": args.model,
        "size": args.size,
        "git_commit": args.git_commit,
        "git_branch": args.git_branch,
        "nvpmodel": args.nvpmodel,
        "wall_s": args.wall_s,
        "command": args.command,
    }
    summary = summarize(rows, meta, dt_s=args.dt)
    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    append_all_summaries(summary, args.all_summaries)

    print(f"samples:  {samples_path} ({len(rows)} rows)")
    print(f"summary:  {summary_path}")
    print(f"all:      {args.all_summaries}")
    print(
        f"peak={summary['peak_power_mW']} mW  mean={summary['mean_power_mW']:.1f} mW  "
        f"energy={summary['energy_J']} J ({summary['energy_Wh']} Wh)  "
        f"source={summary['power_total_source']}"
    )


if __name__ == "__main__":
    main()
