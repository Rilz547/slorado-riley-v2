#!/usr/bin/env python3
"""Are two FASTQ files the same set of reads?  Compares name -> (sequence, quality) so read order does not matter.
usage: fastq_same.py a.fastq b.fastq"""
import sys
def load(p):
    d = {}
    with open(p) as f:
        while True:
            h = f.readline()
            if not h: break
            s, _, q = f.readline().rstrip(), f.readline(), f.readline().rstrip()
            d[h.split()[0]] = (s, q)
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
only_a, only_b = a.keys() - b.keys(), b.keys() - a.keys()
diff = [k for k in a.keys() & b.keys() if a[k] != b[k]]
seq_diff = [k for k in diff if a[k][0] != b[k][0]]
print(f"reads: {len(a)} vs {len(b)}   only in first: {len(only_a)}   only in second: {len(only_b)}   differing: {len(diff)} (sequence differs in {len(seq_diff)})")
print("IDENTICAL" if not (only_a or only_b or diff) else "DIFFERENT")
sys.exit(0 if not (only_a or only_b or diff) else 1)
