#!/usr/bin/env bash
# Worktrees + builds for variant_runs.py (outside the frozen stages.json; same recipe as setup_stages.sh).
#   b_3403f19   3403f19  upstream + LSTM layer-timing instrumentation (before the openfish fork)
#   b_e00bada   e00bada  overlap v1.0 (fork openfish pin, stream-aware decode)
#   b_9f8db24   9f8db24  overlap v1.1 (persistent pinned host decode buffers)
#   bonson_dev  c2ca899  Bonson's dev branch, v0.7.0 in progress (needs the bonson/dev refs: `git fetch bonson dev`)
#
#   ./setup_extra_worktrees.sh                  # all four (3 for the bundle split, 1 for the dev comparison)
#   ./setup_extra_worktrees.sh bonson_dev       # only the dev comparison
#
# Builds use all cores and the submodule step needs the network: do NOT run measurements at the same time.
# Safe to re-run: existing worktrees and binaries are kept. Each build takes several minutes; log: <worktree>/build.log

set -euo pipefail
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$BENCH_DIR/../.." && pwd)"
ROOT="$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$BENCH_DIR/stages.json'))['worktree_root']))")"
declare -A COMMIT=( [b_3403f19]=3403f19 [b_e00bada]=e00bada [b_9f8db24]=9f8db24 [bonson_dev]=c2ca899 )
# The old slorado commits do not build with the openfish pin recorded in them (the pins lag the slorado code), so each is paired
# with the openfish commit its code needs: e00bada needs the stream-aware decode API (openfish 59e12e9); 9f8db24 needs persistent
# buffers (openfish cf182a7). 3403f19 keeps the openfish pin recorded in it (2530c2a, the same as upstream) but contains one stray line,
# `core->decoder_opts = openfish_decoder_default_opts();` in src/slorado.cpp, that only exists in newer openfish; it is reverted to
# upstream's `DECODER_INIT` in the build tree (build-only patch, see PATCH_3403 below), so that u_timing = upstream + instrumentation.
declare -A OFISH=( [b_e00bada]=59e12e9 [b_9f8db24]=cf182a7 )
OFISH_FORK_URL=https://github.com/Rilz547/openfish-riley.git
if [[ $# -gt 0 ]]; then WTS=("$@"); else WTS=(b_3403f19 b_e00bada b_9f8db24 bonson_dev); fi

# shellcheck disable=SC1090
source ~/slorado_venv/bin/activate
export TORCH_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/torch"

for wt in "${WTS[@]}"; do
  commit="${COMMIT[$wt]:?unknown worktree $wt}"
  dir="$ROOT/$wt"
  echo "=== $wt @ $commit -> $dir"
  git -C "$REPO" cat-file -e "$commit^{commit}" 2>/dev/null || { echo "commit $commit not in the local repo (for bonson_dev run: git fetch bonson dev)" >&2; continue; }
  [[ -d "$dir" ]] || git -C "$REPO" worktree add --detach "$dir" "$commit"
  (cd "$dir" && git checkout --detach "$commit" -q && git submodule update --init --force)
  if [[ -x "$dir/slorado" ]]; then echo "    binary already built: $dir/slorado"; continue; fi
  if [[ -n "${OFISH[$wt]:-}" ]]; then
    of="${OFISH[$wt]}"
    if ! git -C "$dir/openfish" cat-file -e "$of^{commit}" 2>/dev/null; then
      git -C "$dir/openfish" remote add riley "$OFISH_FORK_URL" 2>/dev/null || true
      git -C "$dir/openfish" fetch -q riley
    fi
    git -C "$dir/openfish" checkout --detach -q "$of"
    echo "    openfish set to $of"
    (cd "$dir" && make clean > /dev/null 2>&1 || true)
  fi
  if [[ "$wt" == b_3403f19 ]]; then
    sed -i 's/core->decoder_opts = openfish_decoder_default_opts();/core->decoder_opts = DECODER_INIT;/' "$dir/src/slorado.cpp"
    echo "    patched src/slorado.cpp: DECODER_INIT (build tree only)"
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
