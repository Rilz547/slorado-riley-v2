#!/bin/bash

# A quick safety check: Kill any lingering background nsys daemons that might be locking files
killall nsys-ui nsys 2>/dev/null

# Full trace set kept: cuda (kernels/API), nvtx (ranges if present), osrt (host OS APIs).
# Optional: raise OSRT_THRESHOLD (ns) to drop tiny osrt calls while still tracing real syscalls
# (default nsys is 1000). Example: OSRT_THRESHOLD=100000 ./nsight_runner.sh
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
OSRT_THRESHOLD="${OSRT_THRESHOLD:-1000}"
NSYS_EXTRA=(
  --force-overwrite=true
  --trace="$NSYS_TRACE"
  --sample=none
  --osrt-threshold="$OSRT_THRESHOLD"
)

# echo "Running NSight profile for 1k fast basecaller..."
# rm -f nsys_fast_1k_base.nsys-rep nsys_fast_1k_base.sqlite
# nsys profile "${NSYS_EXTRA[@]}" --output=nsys_fast_1k_base \
#   ./slorado basecaller -C 128 -K 2048 -o output_fast_1k.fastq \
#   models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
#   test/PGXXXX230339/reads_1k.blow5;

# echo "Running NSight profile for 1k fast overlap caller..."
# rm -f nsys_fast_1k_overlap.nsys-rep nsys_fast_1k_overlap.sqlite
# nsys profile "${NSYS_EXTRA[@]}" --output=nsys_fast_1k_overlap \
#   ./slorado basecaller --overlap-decode=yes -C 128 -K 2048 -o output_fast_1k_overlap.fastq \
#   models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
#   test/PGXXXX230339/reads_1k.blow5;

# echo "Running NSight profile for 20k fast basecaller..."
# rm -f nsys_fast_20k_base.nsys-rep nsys_fast_20k_base.sqlite
# nsys profile "${NSYS_EXTRA[@]}" --output=nsys_fast_20k_base \
#   ./slorado basecaller -C 128 -o output_fast_20k.fastq \
#   models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
#   test/PGXXXX230339/reads_20k.blow5;

# echo "Running NSight profile for 20k fast overlap caller..."
# rm -f nsys_fast_20k_overlap.nsys-rep nsys_fast_20k_overlap.sqlite
# nsys profile "${NSYS_EXTRA[@]}" --output=nsys_fast_20k_overlap \
#   ./slorado basecaller --overlap-decode=yes -C 128 -o output_fast_20k_overlap.fastq \
#   models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
#   test/PGXXXX230339/reads_20k.blow5;

echo "Running NSight profile for 1k hac basecaller (trace=$NSYS_TRACE)..."
rm -f nsys_hac_1k_base.nsys-rep nsys_hac_1k_base.sqlite
nsys profile "${NSYS_EXTRA[@]}" --output=nsys_hac_1k_base \
  ./slorado basecaller -C 128 -o output_hac_1k.fastq \
  models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
  test/PGXXXX230339/reads_1k.blow5;

echo "Running NSight profile for 1k hac overlap caller (trace=$NSYS_TRACE)..."
rm -f nsys_hac_1k_overlap.nsys-rep nsys_hac_1k_overlap.sqlite
nsys profile "${NSYS_EXTRA[@]}" --output=nsys_hac_1k_overlap \
  ./slorado basecaller --overlap-decode=yes -C 128 -o output_hac_1k_overlap.fastq \
  models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
  test/PGXXXX230339/reads_1k.blow5;

echo "Running NSight profile for 20k hac basecaller (trace=$NSYS_TRACE)..."
rm -f nsys_hac_20k_base.nsys-rep nsys_hac_20k_base.sqlite
nsys profile "${NSYS_EXTRA[@]}" --output=nsys_hac_20k_base \
  ./slorado basecaller -C 128 -o output_hac_20k.fastq \
  models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
  test/PGXXXX230339/reads_20k.blow5;

echo "Running NSight profile for 20k hac overlap caller (trace=$NSYS_TRACE)..."
rm -f nsys_hac_20k_overlap.nsys-rep nsys_hac_20k_overlap.sqlite
nsys profile "${NSYS_EXTRA[@]}" --output=nsys_hac_20k_overlap \
  ./slorado basecaller --overlap-decode=yes -C 128 -o output_hac_20k_overlap.fastq \
  models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
  test/PGXXXX230339/reads_20k.blow5;

echo "Done. FASTQs / profiles:"
ls -lh output_hac_{1k,20k}{,_overlap}.fastq nsys_hac_{1k,20k}_{base,overlap}.nsys-rep 2>/dev/null
