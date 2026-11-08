#!/usr/bin/env bash
# SUP feasibility probe: does the SUP model (transformer, 18 layers, d_model 512, CRF state length 5, 302 MB) run on the 8 GB Orin Nano,
# at which -C, and how long does a batch take?  100 reads (~460 chunks, ~4 batches of 128), upstream binary, /dev/null output.
# Run inside tmux with the desktop idle, ~10 min worst case:   ~/slorado-riley-v2/docs-riley/exp/sup_probe.sh
# Not a measurement for the report: it only decides whether SUP runs are possible and what -C to use.
set -u
cd ~/slorado-riley-v2
BIN=~/bench-stages/upstream/slorado
MODEL=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
DATA=~/data/sup_probe/reads_1k_0.blow5          # first 100 reads of reads_1k.blow5 (cut with slow5tools split -r 100)
OUT=docs-riley/exp/sup-probe.log
pgrep -f "slorado basecaller|bench.py|variant_runs|hang_hunt|maxn_sens|clocks_ab" >/dev/null && { echo "another slorado/bench job is running"; exit 2; }
nvpmodel -q 2>/dev/null | tr '\n' ' ' | tee -a $OUT; echo | tee -a $OUT
free -m | sed -n 2p | tee -a $OUT
run() {   # label, extra flags
    local label="$1"; shift
    local tag; tag=$(echo "$label" | tr -c 'A-Za-z0-9\n' '_')
    echo "=== $label  $(date +%T)" | tee -a $OUT
    ( while :; do awk '/MemAvailable/{print $2}' /proc/meminfo; sleep 1; done > docs-riley/exp/sup-probe.mem ) &
    local mon=$!
    local t0; t0=$(date +%s)
    timeout 900 $BIN basecaller "$@" -c 12288 -K 4096 -p 150 -o /dev/null $MODEL $DATA > docs-riley/exp/sup-probe-$tag.log 2>&1
    local rc=$?
    local t1; t1=$(date +%s)
    kill $mon 2>/dev/null; wait $mon 2>/dev/null
    grep -E "Real time|reads basecalled|NvMap|out of memory|OOM|CUDA error|Killed|rror" docs-riley/exp/sup-probe-$tag.log | head -6 | tee -a $OUT
    echo "elapsed $((t1 - t0)) s   lowest MemAvailable $(( $(sort -n docs-riley/exp/sup-probe.mem | head -1) / 1024 )) MB   exit code $rc   tj $(python3 -c "import sys;sys.path.insert(0,'docs-riley/bench');import bench;print(bench.read_tj_C())")" | tee -a $OUT
    pkill -9 -f "slorado basecaller" 2>/dev/null; sleep 20
}
run "C=128"            -C 128
run "C=64"             -C 64
run "C=32"             -C 32
run "C=32 flash=yes"   -C 32 --flash=yes
echo "done; summary: $OUT   per-run console: docs-riley/exp/sup-probe-<label>.log"
