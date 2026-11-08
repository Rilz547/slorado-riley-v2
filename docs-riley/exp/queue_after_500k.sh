#!/usr/bin/env bash
# Queue of side experiments, run one after another once the Jetson is idle (the running fixed 500k job has finished). Started by Riley:
#     nohup ~/slorado-riley-v2/docs-riley/exp/queue_after_500k.sh > ~/slorado-riley-v2/docs-riley/exp/queue.log 2>&1 &
# Order (each step is independent: a failed step does not stop the next):
#   1. chunk length -c 6000..16000 on upstream, FAST 1k + HAC 1k (3 timed runs + 1 kept FASTQ per value), then FAST 20k (1 kept run per value)   about 50 min
#   2. GPU batch width -C 32..512 on upstream, ov_on, spec, FAST 1k + HAC 1k (3 reps)                                                             about 1.5 h
#   3. the optimised build (x_barrier, spec) at the Slorado paper's own settings on 500k reads                                                    about 80 min (more if a run hangs)
# Do not use the Jetson desktop while it runs. Progress: tail -f queue.log. Stop: kill the script, then pkill -f "slorado basecaller".
# Optional extra, not queued (about 80 min): HAC 20k for the chunk-length test:
#   python3 c_sweep.py --flag c --families upstream --cells dna_hac_20k --reps 0 --no-warmup --keep
set -u
cd "$(dirname "$0")"

busy() {
  pgrep -f "[s]lorado basecaller" > /dev/null && return 0
  pgrep -f "[r]un_fixed_500k.py|[c]_sweep.py|[b]ench.py run|[v]ariant_runs.py|[m]axn_sens.py|[c]locks_ab.py|[h]ang_hunt.py|[s]up_stage_probe.py" > /dev/null
}

echo "[queue] $(date '+%F %T') waiting for the Jetson to be idle"
while busy; do sleep 60; done
echo "[queue] $(date '+%F %T') idle, starting"

step() {
  echo; echo "[queue] $(date '+%F %T') STEP $1"; shift
  "$@"; echo "[queue] $(date '+%F %T') exit $?"
}

step "1a chunk length, FAST/HAC 1k" python3 c_sweep.py --flag c --families upstream --cells dna_fast_1k,dna_hac_1k --reps 3 --keep
step "1b chunk length, FAST 20k (kept FASTQs)" python3 c_sweep.py --flag c --families upstream --cells dna_fast_20k --reps 0 --no-warmup --keep
step "2 GPU batch width -C" python3 c_sweep.py --flag C
step "3 optimised build at the paper's settings, 500k" python3 run_fixed_500k.py --tag paperflags --extra "-t 8 -B 500M -K 2000 -c 10000 -C 200" \
     --kept-dir "$HOME/slorado-riley-v2/docs-riley/bench-runs/kept-sweeps"

echo; echo "[queue] $(date '+%F %T') all steps done"
python3 c_sweep.py --summary
