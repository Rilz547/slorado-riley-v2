#!/usr/bin/env bash
# MAXN_SUPER + jetson_clocks sensitivity run (upstream / ov_off / spec; FAST 1k, HAC 1k x3, FAST 20k x1). About 35-50 min.
#   run inside tmux, desktop idle:   ~/slorado-riley-v2/docs-riley/exp/run_maxn_sens.sh
# Needs: nvpmodel already MAXN_SUPER, jetson_clocks currently OFF (min < max). Pins clocks, runs, ALWAYS restores clocks on exit.
# It does not touch the power mode: switch back to 25W yourself before any frozen-harness run.
set -u
cd "$(dirname "$0")"
GPU=/sys/class/devfreq/17000000.gpu
OC=/sys/class/hwmon/hwmon3/oc3_event_cnt
LOG="maxn-sens-run.log"
exec > >(tee -a "$LOG") 2>&1

gpu_min() { echo $(( $(cat $GPU/min_freq) / 1000000 )); }
gpu_max() { echo $(( $(cat $GPU/max_freq) / 1000000 )); }
tj()      { python3 -c "import sys;sys.path.insert(0,'../bench');import bench;print(bench.read_tj_C() or 0)"; }

echo "[maxn] $(date '+%F %T')  gpu min/max $(gpu_min)/$(gpu_max) MHz  oc_count $(cat $OC 2>/dev/null)"
sudo -v || { echo "[maxn] sudo failed"; exit 2; }
MODE=$(sudo -n nvpmodel -q 2>&1 | tr '\n' ' ')
echo "[maxn] nvpmodel: $MODE"
echo "$MODE" | grep -q "MAXN_SUPER" || { echo "[maxn] ABORT: power mode is not MAXN_SUPER"; exit 2; }
[ "$(gpu_min)" -lt "$(gpu_max)" ] || { echo "[maxn] ABORT: jetson_clocks already ON (min == max). 'sudo jetson_clocks --restore' first."; exit 2; }
pgrep -f "slorado basecaller|variant_runs|bench.py|clocks_ab" >/dev/null && { echo "[maxn] ABORT: another slorado/bench process is running"; exit 2; }

( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) 2>/dev/null &
KEEPALIVE=$!
restore() {
    trap - EXIT INT TERM
    echo "[maxn] restoring clocks ..."
    sudo -n jetson_clocks --restore 2>&1 | tail -2
    echo "[maxn] gpu min/max after restore: $(gpu_min)/$(gpu_max) MHz  oc_count $(cat $OC 2>/dev/null)"
    kill "$KEEPALIVE" 2>/dev/null
}
trap restore EXIT INT TERM
sudo -n jetson_clocks --store || { echo "[maxn] could not store clock state; refusing"; exit 2; }
sudo -n jetson_clocks || { echo "[maxn] jetson_clocks failed"; exit 2; }
sleep 5
echo "[maxn] pinned: gpu min/max $(gpu_min)/$(gpu_max) MHz"
[ "$(gpu_min)" -ge "$(gpu_max)" ] || echo "[maxn] WARNING: GPU min != max, clocks may not be pinned"

t0=$SECONDS
while :; do
    t=$(tj); (( $(echo "$t < 52" | bc) )) && break
    (( SECONDS - t0 > 600 )) && { echo "[maxn] cool-down timeout at ${t} C, continuing"; break; }
    sleep 10
done
echo "[maxn] tj $(tj) C, starting"
python3 maxn_sens.py --label maxn || echo "[maxn] run reported an error"
echo; python3 maxn_sens.py --summary
echo "[maxn] done $(date '+%F %T')  (log: $(pwd)/$LOG)"
