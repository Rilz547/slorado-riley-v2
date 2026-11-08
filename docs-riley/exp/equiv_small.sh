#!/usr/bin/env bash
# V7 output-equivalence check on the 1k and 20k FASTQs only (DNA + RNA), compared read by read with `upstream`.
# `python3 analyze.py equiv` on its own also loads the 4 GB 500k FASTQs into RAM, which does not fit in the Jetson's 8 GB.
#   ~/slorado-riley-v2/docs-riley/exp/equiv_small.sh        (about 1-2 minutes, CPU only)
set -eu
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
K=~/slorado-riley-v2/docs-riley/bench-runs/kept
for f in "$K"/output_*_1k-*.fastq "$K"/output_*_20k-*.fastq; do ln -s "$f" "$TMP/"; done
cd ~/slorado-riley-v2/docs-riley/bench
python3 - "$TMP" <<'PY'
import sys, argparse
from pathlib import Path
import bench, analyze
bench.KEPT_DIR = Path(sys.argv[1])
raise SystemExit(analyze.equiv(argparse.Namespace(ref="upstream")))
PY
