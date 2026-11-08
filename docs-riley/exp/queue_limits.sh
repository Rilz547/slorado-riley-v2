#!/usr/bin/env bash
# Limits of the GPU batch width -C on the Jetson. Started by Riley:
#     nohup ~/slorado-riley-v2/docs-riley/exp/queue_limits.sh > ~/slorado-riley-v2/docs-riley/exp/queue_limits.log 2>&1 &
# Steps (independent: a failed step does not stop the next):
#   1. maximum: largest -C that fits on the 20k-read set, bisected to 16, for upstream / ov_on / spec x FAST / HAC, and SUP (upstream)   about 1 h
#   2. minimum: -C 8, 16, 24, 48 on upstream FAST 1k (2 timed runs per value), added to the 32..512 values already measured                about 10 min
#   3. minimum: -C 16, 24, 48 on upstream HAC 1k (2 timed runs per value)                                                                 about 20 min
# Do not use the Jetson desktop while it runs. Progress: tail -f queue_limits.log. Stop: kill the script, then pkill -f "slorado basecaller".
set -u
cd "$(dirname "$0")"

busy() {
  pgrep -f "[s]lorado basecaller" > /dev/null && return 0
  pgrep -f "[r]un_fixed_500k.py|[c]_sweep.py|[c]_limits.py|[b]ench.py run|[v]ariant_runs.py|[m]axn_sens.py|[c]locks_ab.py|[h]ang_hunt.py|[s]up_stage_probe.py" > /dev/null
}

echo "[queue] $(date '+%F %T') waiting for the Jetson to be idle"
while busy; do sleep 60; done
echo "[queue] $(date '+%F %T') idle, starting"

step() {
  echo; echo "[queue] $(date '+%F %T') STEP $1"; shift
  "$@"; echo "[queue] $(date '+%F %T') exit $?"
}

step "1 maximum -C on 20k reads" python3 c_limits.py --models fast,hac,sup
step "2 minimum -C, FAST 1k" python3 c_sweep.py --flag C --families upstream --cells dna_fast_1k --values 8,16,24,48 --reps 2
step "3 minimum -C, HAC 1k" python3 c_sweep.py --flag C --families upstream --cells dna_hac_1k --values 16,24,48 --reps 2

echo; echo "[queue] $(date '+%F %T') all steps done"
python3 c_limits.py --summary
python3 c_sweep.py --summary
