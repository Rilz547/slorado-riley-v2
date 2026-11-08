#!/usr/bin/env python3
"""Timeline split: when is the infer side (all streams except the decode stream) busy, the decode stream busy, both, neither.
usage: overlap_nsys.py file.sqlite [decode_stream_id]"""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
rows = db.execute("""select k.streamId, s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k
                     join StringIds s on s.id = coalesce(k.demangledName, k.shortName) order by k.start""").fetchall()
dec = int(sys.argv[2]) if len(sys.argv) > 2 else collections.Counter(r[0] for r in rows if "beam_search" in r[1]).most_common(1)[0][0]
t0 = rows[0][2]; t1 = max(r[3] for r in rows)
def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out
D = merge([(r[2], r[3]) for r in rows if r[0] == dec])
I = merge([(r[2], r[3]) for r in rows if r[0] != dec])
def inter(A, B):
    i = j = 0; tot = 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0]); hi = min(A[i][1], B[j][1])
        if hi > lo: tot += hi - lo
        if A[i][1] < B[j][1]: i += 1
        else: j += 1
    return tot
tD = sum(e - s for s, e in D); tI = sum(e - s for s, e in I); both = inter(D, I)
span = t1 - t0
print(f"decode stream = {dec}; span {span/1e9:.3f} s")
print(f"  decode busy {tD/1e9:.3f} s   infer-side busy {tI/1e9:.3f} s   both {both/1e9:.3f} s")
print(f"  decode only (infer side idle)  {(tD-both)/1e9:.3f} s  = {100*(tD-both)/span:.1f} % of span   <- decode exposed on the critical path")
print(f"  infer only                     {(tI-both)/1e9:.3f} s  = {100*(tI-both)/span:.1f} %")
print(f"  both                           {both/1e9:.3f} s  = {100*both/span:.1f} %")
print(f"  neither (GPU idle)             {(span-(tD+tI-both))/1e9:.3f} s  = {100*(span-(tD+tI-both))/span:.1f} %")
# the same over the steady state only (drop first/last 0.8 s: model load, tail)
