#!/usr/bin/env bash
# Follow-up runs after the main campaign (2026-10-25 review). Only calls `bench.py run`; outside the harness version stamp.
#  1. ov_on / dna_fast_20k, both modes, into a SEPARATE file: the campaign runs had one +11 % (mode A rep 3) and one +13 %
#     (mode B rep 1) outlier, so the cell mean was contaminated (CV 6-7 %). The original records stay untouched in runs.jsonl.
#  2. spec / dna_fast_500k: hung twice (no output for 900 s after 1164 s and 2430 s of progress). One more attempt into runs.jsonl.
# Do NOT use the Jetson desktop while this runs. Start (survives disconnects):
#   cd ~/slorado-riley-v2/docs-riley/bench && nohup setsid ./campaign_fixups.sh > ../bench-runs/campaign-fixups.log 2>&1 &
# ~35 min for step 1, ~1.2 h for step 2 if it does not hang (up to ~3.5 h if it hangs again).
set -u
cd "$(dirname "$0")"
run() { echo "=== $(date '+%F %T') bench.py run $*"; python3 bench.py run "$@" || echo "!!! exit $? for: $*"; }

run --stage ov_on --cells dna_fast_20k --runs-file ../bench-runs/rerun-ov_on-fast20k.jsonl
run --stage spec --cells dna_fast_500k --resume
echo "=== $(date '+%F %T') fixups complete"
python3 bench.py status | tail -5
