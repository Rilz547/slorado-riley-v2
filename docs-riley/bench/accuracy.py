#!/usr/bin/env python3
"""Median read identity on the Jetson, using the Slorado/Openfish preprint's method (Supplementary Note 1):

    DNA:  minimap2 -cx map-ont        --secondary=no  hg38noAlt.fa            reads.fastq | awk '{print $10/$11}' | datamash median
    RNA:  minimap2 -cx splice -uf -k14 --secondary=no gencode.v40.transcripts.fa reads.fastq | awk '{print $10/$11}' | datamash median

i.e. PAF column 10 (matches) over column 11 (alignment block length) for every output line, median over lines.
(The 5 Oct accuracy numbers from the PC script use the same definition.)  RNA mapping is cheap on the Jetson
(0.45 GB transcriptome); DNA uses a cached minimap2 index (`~/data/ref/hg38noAlt.mmi`, built once) so repeated checks
do not re-index the 3 GB genome. On the 8 GB Jetson the hg38 index is NOT built automatically (it needs ~6 GB RAM and swaps);
DNA accuracy is the PC's job, RNA (0.45 GB transcriptome) runs fine here.  This is a convenience for smoke tests and spot checks; the campaign accuracy
tables come from the PC script, which uses the same method.

    accuracy.py FASTQ [--dtype dna|rna] [--threads N]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def _load_cfg() -> dict:
    return json.loads((HERE / "stages.json").read_text())


def _count_reads(fastq: Path) -> int:
    n = 0
    with open(fastq, "rb") as f:
        for i, _ in enumerate(f):
            n = i + 1
    return n // 4


def _target(dtype: str, cfg: dict, threads: int) -> Path:
    ref = cfg["references"][dtype]
    fasta = _expand(ref["fasta"])
    if dtype != "dna":
        return fasta
    mmi = _expand("~/data/ref") / (fasta.stem + ".mmi")
    if not mmi.exists():
        # A map-ont index of hg38 is ~7 GB and needs ~6 GB of RAM to build; the 8 GB Jetson swaps badly (tried 2026-10-22).
        raise RuntimeError("DNA accuracy needs a prebuilt hg38 minimap2 index (~7 GB) that does not fit comfortably on the 8 GB Jetson; "
                           "run the DNA mapping on the PC (fetch-and-minimap.py) or copy a prebuilt index to " + str(mmi))
    return mmi


def median_identity(fastq: Path, dtype: str, cfg: dict | None = None, threads: int = 4) -> dict:
    cfg = cfg or _load_cfg()
    opts = cfg["references"][dtype]["minimap2"]
    cmd = ["minimap2", *opts, "-t", str(threads), str(_target(dtype, cfg, threads)), str(fastq)]
    ids: list[float] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:  # type: ignore[union-attr]
        c = line.split("\t")
        if len(c) >= 11:
            try:
                block = float(c[10])
                if block > 0:
                    ids.append(float(c[9]) / block)
            except ValueError:
                continue
    proc.wait()
    n_reads = _count_reads(fastq)
    return {"median": statistics.median(ids) if ids else float("nan"), "n_alignments": len(ids), "n_reads": n_reads,
            "map_rate": len(ids) / n_reads if n_reads else float("nan"), "mean": statistics.mean(ids) if ids else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fastq", type=Path)
    ap.add_argument("--dtype", choices=("dna", "rna"), default="dna")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    r = median_identity(a.fastq, a.dtype, threads=a.threads)
    print(f"{a.fastq.name}: median identity {r['median']:.6f}  alignments {r['n_alignments']} / reads {r['n_reads']} (map rate {100 * r['map_rate']:.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
