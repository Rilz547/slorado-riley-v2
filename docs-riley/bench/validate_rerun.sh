#!/usr/bin/env bash
# Quiet re-run of the ABAB 1k blocks (V3 repeatability, V4 shoulders, V5 drift/order) after the first validation run
# (validation.jsonl) was disturbed by desktop activity on the Jetson (terminal / GNOME apps / display re-probe during blocks).
# Criteria are unchanged. Writes only bench-runs/validation-rerun.jsonl. Do not touch the Jetson desktop while it runs
# (watch progress over SSH only). ~35-40 min of GPU time. Evaluate with:
#   python3 docs-riley/bench/analyze.py --runs-file docs-riley/bench-runs/validation-rerun.jsonl validate
set -u
cd "$(dirname "$0")"
V=../bench-runs/validation-rerun.jsonl
run() { echo "=== $(date +%T) bench.py run $*"; python3 bench.py run --runs-file "$V" "$@"; }
run --stage spec     --cells dna_fast_1k --modes A,B --reps 5
run --stage upstream --cells dna_fast_1k --modes A,B --reps 5
run --stage spec     --cells dna_fast_1k --modes A,B --reps 5
run --stage upstream --cells dna_fast_1k --modes A,B --reps 5
echo "=== $(date +%T) validation re-run complete"
