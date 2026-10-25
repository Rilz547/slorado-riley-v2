#!/usr/bin/env python3
"""RNA median identity on the Jetson for every kept RNA FASTQ, written in the same CSV layout as the PC's
fetch-and-minimap.py (one minimap_accuracy-rna-<stage>.csv per stage) so `analyze.py tables --accuracy` can join it.

Method (preprint, Supplementary Note 1): minimap2 -cx splice -uf -k14 --secondary=no vs GENCODE v40 transcripts,
median of PAF col10/col11 (via accuracy.py; this file only wraps it and is outside the harness version stamp).
Run it AFTER the campaign (it uses all CPU cores for minimap2):

    python3 rna_accuracy_csv.py [--kept ../bench-runs/kept] [--out ../bench-runs/accuracy-rna] [--threads 4]
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import accuracy

HERE = Path(__file__).resolve().parent
FIELDS = ["timestamp", "fastq", "n_reads", "n_alignments", "map_rate", "accuracy_median", "expected_median",
          "delta_vs_expected", "nsys_pulled", "reference"]
NAME = re.compile(r"output_(fast|hac)_(\d+k)-rna-(.+)\.fastq")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kept", type=Path, default=HERE.parent / "bench-runs" / "kept")
    ap.add_argument("--out", type=Path, default=HERE.parent / "bench-runs" / "accuracy-rna")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cfg = accuracy._load_cfg()
    files = sorted(p for p in a.kept.glob("output_*-rna-*.fastq") if NAME.fullmatch(p.name) and "explore" not in p.name)
    if not files:
        print(f"no RNA FASTQs in {a.kept}")
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    by_stage: dict[str, list[dict]] = {}
    for i, fq in enumerate(files, 1):
        stage = NAME.fullmatch(fq.name).group(3)
        r = accuracy.median_identity(fq, "rna", cfg, threads=a.threads)
        print(f"[{i}/{len(files)}] {fq.name}: median identity {r['median']:.6f}  ({r['n_alignments']}/{r['n_reads']} alignments, map rate {100 * r['map_rate']:.1f} %)", flush=True)
        by_stage.setdefault(stage, []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "fastq": fq.name, "n_reads": r["n_reads"],
            "n_alignments": r["n_alignments"], "map_rate": f"{100 * r['map_rate']:.1f}%",
            "accuracy_median": f"{r['median']:.6f}", "expected_median": "-", "delta_vs_expected": "-", "nsys_pulled": "",
            "reference": cfg["references"]["rna"]["fasta"]})
    for stage, rows in by_stage.items():
        path = a.out / f"minimap_accuracy-rna-{stage}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
