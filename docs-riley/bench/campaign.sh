#!/usr/bin/env bash
# Final measurement campaign driver. Only calls `bench.py run` (it adds nothing to the measurements), so it is outside
# the harness version stamp. Safe to stop and restart: every call uses --resume (runs already recorded OK are skipped).
# Do NOT use the Jetson desktop while this runs; watch it over SSH:   python3 bench.py status ;  tail -f ../bench-runs/campaign.log
#
# Start (survives an SSH/VS Code disconnect):
#   cd ~/slorado-riley-v2/docs-riley/bench && nohup setsid ./campaign.sh > ../bench-runs/campaign.log 2>&1 &
#
# Order (partial results are usable at every phase boundary):
#   1  core stages x 1k cells            upstream ov_off ov_on lb_best spec   (DNA FAST/HAC; RNA FAST/HAC on the four RNA stages)
#   2  core stages x 20k cells
#   3  500k FAST, mode B, 1 rep          paper_xavier upstream spec            (1.5-3 h per run)
#   4  extras (dead ends / baselines)    lb_baseline cb_off cb_on, 1k then 20k, DNA only
# Phase 4 is optional; stop the script after phase 3 if you do not need it.
set -u
cd "$(dirname "$0")"

CORE_DNA="upstream ov_off ov_on lb_best spec"
RNA_STAGES=" upstream ov_on lb_best spec "
EXTRA="lb_baseline cb_off cb_on"

run() { echo "=== $(date '+%F %T') bench.py run $*"; python3 bench.py run --resume "$@" || echo "!!! exit $? for: $*"; }

cells_for() {  # $1 stage, $2 size (1k|20k)
  local c="dna_fast_$2,dna_hac_$2"
  [[ "$RNA_STAGES" == *" $1 "* ]] && c="$c,rna_fast_$2,rna_hac_$2"
  echo "$c"
}

python3 bench.py check | tail -2

echo "##### phase 1: 1k"
for s in $CORE_DNA; do run --stage "$s" --cells "$(cells_for "$s" 1k)"; done
echo "##### phase 2: 20k"
for s in $CORE_DNA; do run --stage "$s" --cells "$(cells_for "$s" 20k)"; done
echo "##### phase 3: 500k FAST"
for s in paper_xavier upstream spec; do run --stage "$s" --cells dna_fast_500k; done
echo "##### phase 4: extras"
for z in 1k 20k; do for s in $EXTRA; do run --stage "$s" --cells "dna_fast_$z,dna_hac_$z"; done; done
echo "=== $(date '+%F %T') campaign complete"
python3 bench.py status | tail -20
