#!/usr/bin/env bash
# Unattended sequence for the baseline-pairing questions. Start it in tmux (see below); read the log in the morning.
# Order = value first: (1) spec pairing (no build), (2) build Bonson's dev, (3) dev comparison, (4) build the 3 old commits, (5) bundle split.
# A failure in one step is logged and the script moves on; steps that need a binary that did not build are skipped.
# Total roughly 7-10 h (builds are the uncertain part). Nothing else may use the GPU and nobody may use the Jetson desktop.
#   cd ~/slorado-riley-v2/docs-riley/bench && ./overnight_variants.sh 2>&1 | tee ../bench-runs/overnight-variants.log
set -u
cd "$(dirname "$0")"
ROOT="$HOME/bench-stages"
step() { echo; echo "######## $(date '+%F %T')  $*"; }

step "0. pre-flight"
if pgrep -f "bench.py|variant_runs.py|hang_hunt|slorado basecaller" | grep -qv "^$$\$"; then echo "another harness or slorado process is running; aborting"; pgrep -af "bench.py|variant_runs.py|hang_hunt|slorado basecaller" | head -3; exit 2; fi
python3 bench.py check | tail -3

step "1. spec pairing: same 49ac641 binary, speculative decode off vs on (no build)"
python3 variant_runs.py --set pairing --with-b1 || echo "!!! pairing run failed"

step "2. build Bonson's dev (c2ca899)"
./setup_extra_worktrees.sh bonson_dev || echo "!!! dev build script failed"
if [[ -x "$ROOT/bonson_dev/slorado" ]]; then
  step "3. dev comparison (upstream vs dev vs dev --async=yes vs spec)"
  VARS="ref_upstream,bdev,bdev_async,spec_on"
  if ! "$ROOT/bonson_dev/slorado" basecaller --help 2>&1 | grep -q -- "--async"; then
    echo "dev binary has no --async option: dropping bdev_async"; VARS="ref_upstream,bdev,spec_on"
  fi
  python3 variant_runs.py --set dev --variants "$VARS" --with-b1 || echo "!!! dev run failed"
else
  echo "!!! skipping the dev comparison: no dev binary (see $ROOT/bonson_dev/build.log)"
fi

step "4. build the three old commits for the bundle split"
./setup_extra_worktrees.sh b_3403f19 b_e00bada b_9f8db24 || echo "!!! old-commit build script failed"
VARS="ref_upstream"
for pair in "u_timing:b_3403f19" "v10_off:b_e00bada" "v11_off:b_9f8db24"; do
  v="${pair%%:*}"; wt="${pair##*:}"
  if [[ -x "$ROOT/$wt/slorado" ]]; then VARS="$VARS,$v"; else echo "!!! no binary for $v ($wt): left out of the bundle run"; fi
done
VARS="$VARS,ref_ov_off"
step "5. bundle split with: $VARS"
if [[ "$VARS" == "ref_upstream,ref_ov_off" ]]; then echo "no old-commit binaries built; skipping"; else python3 variant_runs.py --set bundle --variants "$VARS" || echo "!!! bundle run failed"; fi

step "ALL DONE. Records: bench-runs/variants-{pairing,dev,bundle}.jsonl; kept FASTQs: bench-runs/kept/output_*-dna-<variant>.fastq; failures above are marked '!!!'"
grep -c '^!!!' ../bench-runs/overnight-variants.log 2>/dev/null | sed 's/^/number of !!! lines: /'
