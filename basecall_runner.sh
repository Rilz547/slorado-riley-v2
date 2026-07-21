#!/bin/bash

echo "Running basecaller..."

echo "Running 1k FAST..."
./slorado basecaller -C 128 -o /tmp/base.fastq \
  models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_1k.blow5

echo "Running 1k FAST OVERLAP..."
./slorado basecaller --overlap-decode=yes -C 128 -o /tmp/overlap.fastq \
  models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_1k.blow5

echo "Running 20k FAST..."
./slorado basecaller -C 128 -o /tmp/base.fastq \
  models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_20k.blow5

echo "Running 20k FAST OVERLAP..."
./slorado basecaller --overlap-decode=yes -C 128 -o /tmp/overlap.fastq \
  models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_20k.blow5