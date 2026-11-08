#!/usr/bin/env bash
# Build the experiment tree (~/bench-stages/exp_src) and copy the binary to ~/bench-stages/x_<name>/slorado.
#   build_exp.sh <name> [make-vars ...]        e.g.  build_exp.sh sm87 CUDA_ARCH="-gencode arch=compute_87,code=sm_87"
# Always cleans openfish first (its objects do not depend on slorado's flags, so flag changes need a clean). Log: exp_src/build-<name>.log
set -euo pipefail
name="$1"; shift
src=~/bench-stages/exp_src
source ~/slorado_venv/bin/activate
export TORCH_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/torch"
cd "$src"
make -C openfish clean >/dev/null 2>&1 || true
rm -f openfish/lib/libopenfish.a slorado
t0=$(date +%s)
if make -j5 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR="$TORCH_PATH" "$@" > "build-$name.log" 2>&1; then
  mkdir -p ~/bench-stages/x_"$name"
  cp slorado ~/bench-stages/x_"$name"/slorado
  echo "built x_$name in $(( $(date +%s) - t0 )) s"
  cuobjdump --list-elf ~/bench-stages/x_"$name"/slorado 2>&1 | head -4
else
  echo "BUILD FAILED ($name), tail of build-$name.log:"; tail -20 "build-$name.log"; exit 1
fi
