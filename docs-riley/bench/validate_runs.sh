#!/usr/bin/env bash
# Harness validation runs (V1-V6). Writes only to bench-runs/validation.jsonl; the campaign file runs.jsonl is untouched.
# ABAB order (spec, upstream, spec, upstream) for the drift/order check; HAC and 20k blocks for V2/V3; 20k with tegrastats for V1.
# Start it yourself (or approve me starting it); ~1-1.5 h of GPU time. Evaluate with:
#   python3 docs-riley/bench/analyze.py --runs-file docs-riley/bench-runs/validation.jsonl validate
set -u
cd "$(dirname "$0")"
V=../bench-runs/validation.jsonl
run() { echo "=== $(date +%T) bench.py run $*"; python3 bench.py run --runs-file "$V" "$@"; }
run --stage spec     --cells dna_fast_1k --modes A,B --reps 5
run --stage upstream --cells dna_fast_1k --modes A,B --reps 5
run --stage spec     --cells dna_fast_1k --modes A,B --reps 5
run --stage upstream --cells dna_fast_1k --modes A,B --reps 5
run --stage spec     --cells dna_hac_1k  --modes A,B --reps 3
run --stage spec     --cells dna_fast_20k --modes A --reps 3 --tegrastats
echo "=== $(date +%T) validation runs complete"
