#!/usr/bin/env bash
# jetson_clocks A/B in one go:  clocks as-is (off) -> jetson_clocks ON -> restored (off2) -> summary.
#   run inside tmux, desktop idle:   ~/slorado-riley-v2/docs-riley/exp/run_clocks_ab.sh [reps]
# Asks for your sudo password once at the start. Always restores the clock state on exit (also on Ctrl-C or error).
# Needs the clocks to be OFF at the start (min_freq < max_freq); aborts otherwise.  About 25-30 min.
set -u
REPS="${1:-5}"
cd "$(dirname "$0")"
GPU=/sys/class/devfreq/17000000.gpu
LOG="clocks-ab-run.log"
exec > >(tee -a "$LOG") 2>&1

gpu_min() { echo $(( $(cat $GPU/min_freq) / 1000000 )); }
gpu_max() { echo $(( $(cat $GPU/max_freq) / 1000000 )); }
tj()      { python3 -c "import sys;sys.path.insert(0,'../bench');import bench;print(bench.read_tj_C() or 0)"; }

cooldown() {   # wait until the GPU/CPU junction is below 52 C (max 10 min)
    local t0=$SECONDS
    while :; do
        local t; t=$(tj)
        (( $(echo "$t < 52" | bc) )) && break
        (( SECONDS - t0 > 600 )) && { echo "[clocks-ab] cool-down timeout at ${t} C, continuing"; break; }
        sleep 10
    done
    echo "[clocks-ab] tj $(tj) C"
}

echo "[clocks-ab] $(date '+%F %T')  reps=$REPS  gpu min/max now $(gpu_min)/$(gpu_max) MHz"
if [ "$(gpu_min)" -ge "$(gpu_max)" ]; then
    echo "[clocks-ab] ABORT: jetson_clocks already ON (min == max). Run 'sudo jetson_clocks --restore' (or reboot) first."; exit 2
fi
pgrep -f "slorado basecaller|variant_runs|bench.py" >/dev/null && { echo "[clocks-ab] ABORT: another slorado/bench process is running"; exit 2; }

sudo -v || { echo "[clocks-ab] sudo failed"; exit 2; }
( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) 2>/dev/null &
KEEPALIVE=$!

restore() {
    trap - EXIT INT TERM
    echo "[clocks-ab] restoring clocks ..."
    sudo -n jetson_clocks --restore 2>&1 | tail -2
    echo "[clocks-ab] gpu min/max after restore: $(gpu_min)/$(gpu_max) MHz"
    kill "$KEEPALIVE" 2>/dev/null
}
trap restore EXIT INT TERM

sudo -n jetson_clocks --store || { echo "[clocks-ab] could not store the clock state; refusing to continue"; exit 2; }
echo "[clocks-ab] stored current clock state"

echo; echo "=================== block 1: off ==================="
cooldown
python3 clocks_ab.py --label off --reps "$REPS" || echo "[clocks-ab] block 'off' reported an error"

echo; echo "=================== block 2: jetson_clocks ON ==================="
sudo -n jetson_clocks || { echo "[clocks-ab] jetson_clocks failed"; exit 2; }
sleep 5
echo "[clocks-ab] gpu min/max now $(gpu_min)/$(gpu_max) MHz"
[ "$(gpu_min)" -ge "$(gpu_max)" ] || echo "[clocks-ab] WARNING: GPU min != max, clocks may not be pinned"
cooldown
python3 clocks_ab.py --label on --reps "$REPS" || echo "[clocks-ab] block 'on' reported an error"

echo; echo "=================== block 3: restored (off2) ==================="
restore
trap - EXIT INT TERM
kill "$KEEPALIVE" 2>/dev/null
[ "$(gpu_min)" -lt "$(gpu_max)" ] || { echo "[clocks-ab] ABORT: clocks still pinned after restore; reboot or run 'sudo jetson_clocks --restore'"; exit 2; }
cooldown
python3 clocks_ab.py --label off2 --reps "$REPS" || echo "[clocks-ab] block 'off2' reported an error"

echo; echo "=================== summary ==================="
python3 clocks_ab.py --summary
echo "[clocks-ab] done $(date '+%F %T')  (log: $(pwd)/$LOG)"
