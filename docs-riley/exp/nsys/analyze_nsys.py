#!/usr/bin/env python3
"""Per-kernel and critical-path summary from an nsys sqlite export (kernel table only).  usage: analyze_nsys.py file.sqlite"""
import sqlite3, sys, collections, statistics as st
db = sqlite3.connect(sys.argv[1])
rows = db.execute("""select k.streamId, s.value, k.start, k.end, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
                     from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = coalesce(k.demangledName, k.shortName)
                     order by k.start""").fetchall()
t0, t1 = rows[0][2], max(r[3] for r in rows)
wall = (t1 - t0) / 1e9
print(f"kernels: {len(rows)}   first start -> last end: {wall:.3f} s")
# GPU busy (union of intervals) and by stream
def union(iv):
    iv = sorted(iv); tot = 0; cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s > cur_e: tot += cur_e - cur_s; cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    return (tot + cur_e - cur_s) / 1e9
print(f"GPU busy (any kernel running): {union([(r[2], r[3]) for r in rows]):.3f} s = {100*union([(r[2], r[3]) for r in rows])/wall:.1f} %")
by = collections.defaultdict(list)
for r in rows:
    nm = r[1]
    key = nm.split("(")[0][:60]
    by[(key, r[0])].append(r)
streams = sorted({r[0] for r in rows})
print("streams:", {s: round(union([(r[2], r[3]) for r in rows if r[0] == s]), 3) for s in streams}, "(busy s per stream)")
print(f"\n{'kernel':58} {'strm':>4} {'n':>5} {'tot s':>7} {'mean ms':>8} {'med ms':>7} {'max ms':>7}  grid x block (first)")
tab = []
for (k, s), rs in by.items():
    d = [(r[3] - r[2]) / 1e6 for r in rs]
    tab.append((sum(d), k, s, len(d), st.mean(d), st.median(d), max(d), rs[0]))
for tot, k, s, n, m, md, mx, r in sorted(tab, reverse=True)[:22]:
    print(f"{k:58} {s:>4} {n:>5} {tot/1000:7.3f} {m:8.3f} {md:7.3f} {mx:7.3f}  {r[4]}x{r[5]}x{r[6]} / {r[7]}x{r[8]}x{r[9]}")
# beam_search duration vs grid size
print("\nbeam_search / greedy by grid size (N): count, median ms")
for name in ("beam_search", "greedy_search", "bwd_scan", "fwd_post_scan", "compute_qual_data", "generate_sequence"):
    g = collections.defaultdict(list)
    for r in rows:
        if name in r[1]: g[r[4]].append((r[3] - r[2]) / 1e6)
    print(f"  {name:20}", ", ".join(f"N={n}: {len(v)}x {st.median(v):.2f} ms" for n, v in sorted(g.items())[:8]))
# decode-stream chain vs infer stream: idle gaps on the stream that runs the LSTM/GEMM kernels
gem = collections.Counter(r[0] for r in rows if "gemm" in r[1].lower() or "lstm" in r[1].lower() or "cudnn" in r[1].lower())
infer_stream = gem.most_common(1)[0][0] if gem else None
print("\ninfer-like stream:", infer_stream)
if infer_stream is not None:
    iv = sorted((r[2], r[3]) for r in rows if r[0] == infer_stream)
    gaps = [(iv[i+1][0] - iv[i][1]) / 1e6 for i in range(len(iv) - 1) if iv[i+1][0] > iv[i][1]]
    big = [g for g in gaps if g > 1.0]
    print(f"  infer stream: span {(iv[-1][1]-iv[0][0])/1e9:.3f} s, busy {union(iv):.3f} s, gaps > 1 ms: {len(big)} totalling {sum(big)/1000:.3f} s (max {max(big) if big else 0:.1f} ms)")
