#!/usr/bin/env bash
# RUN ON THE PC.  Checks that everything on the Jetson that cannot be regenerated cheaply is also on the PC, and that the PC repos are pushed.
# Nothing is copied or deleted: rsync runs with --dry-run.
#   ./backup-check.sh [PC_DIR_FOR_BENCH_RUNS]         default: ~/Desktop/thesisB-data-and-diagrams/final-campaign/jetson-campaign-runs
#   CHECKSUM=1 ./backup-check.sh ...                  compare by content instead of size+time (slow on the 12 GB of FASTQs)
JET=riley@192.168.8.211
SRC=/home/riley/slorado-riley-v2/docs-riley/bench-runs/
DEST="${1:-$HOME/Desktop/thesisB-data-and-diagrams/final-campaign/jetson-campaign-runs}"
KDEST="${KDEST:-$DEST/kept}"
CK=""; [ "${CHECKSUM:-0}" = 1 ] && CK="--checksum"
RSH="ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-bkchk-%p -o ControlPersist=300"   # one ssh connection, password asked once

echo "== 1. run records, power traces, consoles, tables, jsonl (everything except kept FASTQs) -> $DEST"
mkdir -p "$DEST"
rsync -avn $CK --itemize-changes -e "$RSH" --exclude='kept*/' "$JET:$SRC" "$DEST/" | grep -E '^(>f|\*deleting)' > /tmp/bkchk-main.txt
echo "   files missing or different on the PC: $(grep -c '^>f' /tmp/bkchk-main.txt)   (list: /tmp/bkchk-main.txt)"
grep -E '^>f\+' /tmp/bkchk-main.txt | head -8 | sed 's/^/   NEW /'
echo "   key files:"
for f in runs.jsonl runs.csv excluded-runs.json tables-with-accuracy.md variants-bundle.jsonl variants-pairing.jsonl variants-dev.jsonl variants-exp.jsonl clocks-off-20261031T125204.jsonl clocks-on-20261031T131306.jsonl clocks-off2-20261031T132545.jsonl maxn-maxn-20261031T190909.jsonl reducedK-500k.jsonl; do
  [ -f "$DEST/$f" ] && echo "     ok       $f" || echo "     MISSING  $f"; done
echo "   power traces on PC: $(ls "$DEST/power" 2>/dev/null | wc -l)   consoles: $(ls "$DEST/console" 2>/dev/null | wc -l)   (Jetson: see below)"
$RSH "$JET" 'cd /home/riley/slorado-riley-v2/docs-riley/bench-runs && echo "   Jetson power: $(ls power | wc -l)   consoles: $(ls console | wc -l)   kept FASTQs: $(ls kept | wc -l)   hang-hunt dirs: $(ls hang-hunt | wc -l)"' 2>/dev/null

echo; echo "== 2. kept FASTQs (12 GB; the only copy of the campaign outputs) -> $KDEST"
mkdir -p "$KDEST"
rsync -avn $CK --itemize-changes -e "$RSH" --include='output_*.fastq' --exclude='*' "$JET:${SRC}kept/" "$KDEST/" | grep '^>f' > /tmp/bkchk-kept.txt
echo "   FASTQs missing or different on the PC: $(wc -l < /tmp/bkchk-kept.txt)   (list: /tmp/bkchk-kept.txt)"
echo "   to copy them:  rsync -avP --include='output_*.fastq' --exclude='*' $JET:${SRC}kept/ \"$KDEST/\""

echo; echo "== 3. PC repos: anything not pushed?"
for r in "$HOME/Desktop/thesisB-data-and-diagrams" "$HOME/slorado-riley-v2" "$HOME/Desktop/slorado_analysis"; do
  [ -d "$r/.git" ] || continue
  echo "-- $r"
  git -C "$r" status --short | head -15
  echo "   unpushed commits: $(git -C "$r" log --oneline @{u}..HEAD 2>/dev/null | wc -l)  (branch $(git -C "$r" branch --show-current))"
  echo "   large ignored files (>20 MB) that git does NOT back up:"
  git -C "$r" ls-files --others --ignored --exclude-standard -z 2>/dev/null | xargs -0 -r du -m 2>/dev/null | awk '$1>20' | sort -rn | head -5 | sed 's/^/     /'
done
echo; echo "Done. Nothing was changed."
