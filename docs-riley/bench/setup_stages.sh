#!/usr/bin/env bash
# Create one git worktree per stage commit under ~/bench-stages and build slorado in each.
# The main working tree is never switched. Safe to re-run: existing worktrees/binaries are kept.
#
#   docs-riley/bench/setup_stages.sh                # all worktrees in stages.json
#   docs-riley/bench/setup_stages.sh spec upstream  # only these
#
# Builds use all cores: do NOT run measurements while this is running.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$BENCH_DIR/../.." && pwd)"
CONFIG="$BENCH_DIR/stages.json"

ROOT="$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$CONFIG'))['worktree_root']))")"
mkdir -p "$ROOT"

if [[ $# -gt 0 ]]; then
  WTS=("$@")
else
  mapfile -t WTS < <(python3 -c "import json;print('\n'.join(json.load(open('$CONFIG'))['worktrees']))")
fi

# shellcheck disable=SC1090
source ~/slorado_venv/bin/activate
export TORCH_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/torch"

for wt in "${WTS[@]}"; do
  commit="$(python3 -c "import json;print(json.load(open('$CONFIG'))['worktrees']['$wt']['commit'])")"
  dir="$ROOT/$wt"
  echo "=== $wt @ $commit -> $dir"
  if [[ ! -d "$dir" ]]; then
    git -C "$REPO" worktree add --detach "$dir" "$commit"
  fi
  (cd "$dir" && git checkout --detach "$commit" -q && git submodule update --init --force)
  if [[ -x "$dir/slorado" ]]; then
    echo "    binary already built: $dir/slorado"
    continue
  fi
  echo "    building (log: $dir/build.log) ..."
  start=$(date +%s)
  if (cd "$dir" && make -j6 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR="$TORCH_PATH" > build.log 2>&1); then
    echo "    built in $(( $(date +%s) - start )) s"
  else
    echo "    BUILD FAILED for $wt (see $dir/build.log)" >&2
    tail -15 "$dir/build.log" >&2 || true
  fi
done
echo "=== done"
