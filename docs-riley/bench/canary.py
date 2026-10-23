#!/usr/bin/env python3
"""System canary: a fixed GPU workload whose time exposes thermal / clock / background drift between stage blocks.

One cuDNN LSTM layer (hidden 384, N=128, T=2048, fp16) -- the kernel that dominates HAC inference.  An 8 s warm loop
lets the GPU clock reach its steady state before 7 timed repeats; prints the median in milliseconds.
Run with the same python that has torch (the slorado venv).
"""
import statistics
import time

import torch

H, N, T = 384, 128, 2048
lstm = torch.nn.LSTM(H, H, batch_first=True).half().cuda().eval()
x = torch.randn(N, T, H, device="cuda", dtype=torch.half)
with torch.inference_mode():
    t0 = time.time()
    while time.time() - t0 < 8.0:
        lstm(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        lstm(x)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
print("CANARY_MS=%.3f" % statistics.median(ts))
print("CANARY_ALL_MS=" + ",".join("%.2f" % t for t in ts))
