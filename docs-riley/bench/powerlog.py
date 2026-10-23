#!/usr/bin/env python3
"""High-rate power / system logger for the Jetson Orin (INA3221 via hwmon) + energy integration helpers.

Why not tegrastats: the INA3221 averages 512 samples per reading and refreshes about 3 times a second, so
1 Hz tegrastats sees roughly a third of the time and its log window does not match slorado's timed window.
This logger polls the hwmon files at ~10-20 Hz, timestamps every poll, flags, per rail, polls where a *new* reading
appeared, and records GPU clock/load, CPU utilisation, memory and temperature alongside.

CLI
    powerlog.py log OUT.csv [--hz 20] [--seconds N]     # runs until SIGTERM/SIGINT (or N seconds)
    powerlog.py summarize TRACE.csv --t0 EPOCH --t1 EPOCH [--method backward|trapezoid]

Energy convention ("backward rectangle", default): an INA3221 reading is the average over the interval since the
previous reading, so reading k (first seen at t_k) describes the interval (t_{k-1}, t_k]. The trace therefore has to extend
past the end of the window (the harness records a post-run shoulder). "trapezoid" is provided for cross-checks.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import signal
import sys
import time
from pathlib import Path

HARNESS_COMPONENT = "powerlog-2"

RAILS = ("in", "cg", "soc")  # VDD_IN, VDD_CPU_GPU_CV, VDD_SOC
RAIL_LABELS = {"VDD_IN": "in", "VDD_CPU_GPU_CV": "cg", "VDD_SOC": "soc"}
GPU_DEVFREQ = "/sys/class/devfreq/17000000.gpu"
GPU_LOAD_GLOBS = (
    "/sys/devices/platform/bus@0/17000000.gpu/load",
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/devices/gpu.0/load",
)

COLUMNS = [
    "t_epoch", "t_mono", "new_in", "new_cg", "new_soc",
    "p_in_mW", "p_cg_mW", "p_soc_mW",
    "v_in_mV", "v_cg_mV", "v_soc_mV", "i_in_mA", "i_cg_mA", "i_soc_mA",
    "gpu_mhz", "gpu_load_permille", "cpu_util_pct",
    "ram_used_mb", "swap_used_mb", "tj_mC", "read_ms",
]


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    v = _read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def find_ina3221() -> tuple[str, dict[str, int]]:
    """Return (hwmon dir, {rail_key: channel_index}) by reading the in<N>_label files."""
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _read(h + "/name") != "ina3221":
            continue
        mapping: dict[str, int] = {}
        for ch in (1, 2, 3):
            label = _read(f"{h}/in{ch}_label")
            if label in RAIL_LABELS:
                mapping[RAIL_LABELS[label]] = ch
        if set(mapping) == set(RAILS):
            return h, mapping
    raise SystemExit("ina3221 hwmon with VDD_IN / VDD_CPU_GPU_CV / VDD_SOC labels not found")


def find_gpu_load() -> str | None:
    for g in GPU_LOAD_GLOBS:
        if os.path.exists(g):
            return g
    return None


def find_tj_zone() -> str | None:
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        if _read(z + "/type") == "tj-thermal":
            return z + "/temp"
    return None


def cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies from the aggregate cpu line of /proc/stat."""
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals[:8]) if len(vals) >= 8 else sum(vals)
    return total - idle, total


def mem_used_mb() -> tuple[int, int]:
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                info[k] = int(rest.split()[0])
    ram = (info["MemTotal"] - info["MemAvailable"]) // 1024
    swap = (info["SwapTotal"] - info["SwapFree"]) // 1024
    return ram, swap


def log_trace(out: Path, hz: float = 20.0, seconds: float | None = None) -> None:
    hwmon, ch = find_ina3221()
    gpu_load = find_gpu_load()
    tj = find_tj_zone()
    period = 1.0 / hz
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    out.parent.mkdir(parents=True, exist_ok=True)
    prev_key = {k: None for k in RAILS}
    prev_busy, prev_total = cpu_times()
    cpu_util = 0.0
    ram = swap = 0
    n = 0
    t_end = time.monotonic() + seconds if seconds else None
    next_t = time.monotonic()
    with open(out, "w", newline="", buffering=1) as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        last_flush = time.monotonic()
        while not stop["flag"]:
            r0 = time.perf_counter()
            t_epoch = time.time()
            t_mono = time.monotonic()
            v = {k: _read_int(f"{hwmon}/in{c}_input") for k, c in ch.items()}
            i = {k: _read_int(f"{hwmon}/curr{c}_input") for k, c in ch.items()}
            if any(x is None for x in v.values()) or any(x is None for x in i.values()):
                time.sleep(period)
                continue
            p = {k: v[k] * i[k] / 1000.0 for k in RAILS}  # mV * mA / 1000 = mW
            # each rail is converted at its own moment, so "a new reading appeared" is tracked per rail
            new = {}
            for k in RAILS:
                key = (v[k], i[k])
                new[k] = 1 if key != prev_key[k] else 0
                prev_key[k] = key

            busy, total = cpu_times()
            if total > prev_total:
                cpu_util = 100.0 * (busy - prev_busy) / (total - prev_total)
                prev_busy, prev_total = busy, total
            if n % 10 == 0:
                ram, swap = mem_used_mb()
            gpu_hz = _read_int(GPU_DEVFREQ + "/cur_freq")
            load = _read_int(gpu_load) if gpu_load else None
            tjv = _read_int(tj) if (tj and n % 5 == 0) else None
            read_ms = (time.perf_counter() - r0) * 1000.0

            w.writerow([
                f"{t_epoch:.4f}", f"{t_mono:.4f}", new["in"], new["cg"], new["soc"],
                f"{p['in']:.1f}", f"{p['cg']:.1f}", f"{p['soc']:.1f}",
                v["in"], v["cg"], v["soc"], i["in"], i["cg"], i["soc"],
                f"{gpu_hz / 1e6:.0f}" if gpu_hz else "",
                load if load is not None else "",
                f"{cpu_util:.1f}", ram, swap,
                tjv if tjv is not None else "", f"{read_ms:.2f}",
            ])
            n += 1
            now = time.monotonic()
            if now - last_flush > 1.0:
                f.flush()
                last_flush = now
            if t_end is not None and now >= t_end:
                break
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind; do not burst


# ---------------------------------------------------------------- reading + integration

def load_trace(path: Path | str) -> list[dict]:
    import gzip
    rows: list[dict] = []
    opener = (lambda p: gzip.open(p, "rt", newline="")) if str(path).endswith(".gz") else (lambda p: open(p, newline=""))
    with opener(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["t_epoch"]),
                    "new": {"in": int(r["new_in"]), "cg": int(r["new_cg"]), "soc": int(r["new_soc"])},
                    "in": float(r["p_in_mW"]), "cg": float(r["p_cg_mW"]), "soc": float(r["p_soc_mW"]),
                    "gpu_mhz": float(r["gpu_mhz"]) if r["gpu_mhz"] else None,
                    "gpu_load": float(r["gpu_load_permille"]) if r["gpu_load_permille"] else None,
                    "cpu": float(r["cpu_util_pct"]) if r["cpu_util_pct"] else None,
                    "ram": float(r["ram_used_mb"]) if r["ram_used_mb"] else None,
                    "swap": float(r["swap_used_mb"]) if r["swap_used_mb"] else None,
                    "tj": float(r["tj_mC"]) / 1000.0 if r["tj_mC"] else None,
                    "read_ms": float(r["read_ms"]) if r["read_ms"] else None,
                })
            except (KeyError, ValueError):
                continue  # truncated final line
    return rows


def updates(rows: list[dict], rail: str) -> list[dict]:
    """Rows where a new reading of `rail` was first observed (its real update instants)."""
    out = [r for r in rows if r["new"][rail]]
    if rows and (not out or out[0] is not rows[0]):
        out.insert(0, rows[0])
    return out


def integrate(rows: list[dict], t0: float, t1: float, method: str = "backward") -> dict:
    """Energy (J), mean power (W) and peak (W) per rail over [t0, t1]; residual = in - cg - soc."""
    res: dict = {"method": method}
    energy: dict[str, float] = {}
    peak: dict[str, float] = {}
    n_upd: dict[str, int] = {}
    gap: dict[str, float] = {}
    cov: dict[str, float] = {}
    if t1 <= t0:
        return {"error": "empty window", **res}
    for k in RAILS:
        ups = updates(rows, k)
        e = pk = cv = mg = 0.0
        n = 0
        for a, b in zip(ups, ups[1:]):
            lo, hi = max(a["t"], t0), min(b["t"], t1)
            if hi <= lo:
                continue
            dt = hi - lo
            if method == "backward":
                pw = b[k]  # reading b describes (a.t, b.t]
                pk = max(pk, b[k])
            else:  # trapezoid between consecutive readings
                f0 = (lo - a["t"]) / (b["t"] - a["t"])
                f1 = (hi - a["t"]) / (b["t"] - a["t"])
                pw = a[k] + (b[k] - a[k]) * 0.5 * (f0 + f1)
                pk = max(pk, a[k], b[k])
            e += pw * dt / 1000.0  # mW * s / 1000 = J
            cv += dt
            n += 1
            mg = max(mg, b["t"] - a["t"])
        energy[k], peak[k], n_upd[k], gap[k], cov[k] = e, pk, n, mg, cv
    span = t1 - t0
    resid = energy["in"] - energy["cg"] - energy["soc"]
    res.update({
        "window_s": span, "covered_s": min(cov.values()),
        "n_updates_in": n_upd["in"], "n_updates_cg": n_upd["cg"], "n_updates_soc": n_upd["soc"],
        "max_gap_s": max(gap.values()),
        "E_in_J": energy["in"], "E_cg_J": energy["cg"], "E_soc_J": energy["soc"], "E_resid_J": resid,
        "Pmean_in_W": energy["in"] / span, "Pmean_cg_W": energy["cg"] / span,
        "Pmean_soc_W": energy["soc"] / span, "Pmean_resid_W": resid / span,
        "Ppeak_in_W": peak["in"] / 1000.0, "Ppeak_cg_W": peak["cg"] / 1000.0, "Ppeak_soc_W": peak["soc"] / 1000.0,
    })
    return res


def window_stats(rows: list[dict], t0: float, t1: float) -> dict:
    """Mean/max of the auxiliary channels inside the window."""
    sel = [r for r in rows if t0 <= r["t"] <= t1]

    def mean(key):
        v = [r[key] for r in sel if r[key] is not None]
        return sum(v) / len(v) if v else None

    def vmax(key):
        v = [r[key] for r in sel if r[key] is not None]
        return max(v) if v else None

    tjs = [r["tj"] for r in rows if r["tj"] is not None]
    return {
        "gpu_mhz_mean": mean("gpu_mhz"), "gpu_load_mean_permille": mean("gpu_load"),
        "cpu_util_mean_pct": mean("cpu"), "ram_used_mb_max": vmax("ram"), "swap_used_mb_max": vmax("swap"),
        "tj_max_C": vmax("tj"), "tj_first_C": tjs[0] if tjs else None, "tj_last_C": tjs[-1] if tjs else None,
        "poll_read_ms_mean": mean("read_ms"),
    }


def emulate_1hz(rows: list[dict], t0: float, t1: float) -> dict:
    """What a tegrastats-style 1 Hz point sampler would conclude for [t0, t1] (validation V2).

    Samples the held value at t0+0.5, t0+1.5, ... (one per whole second inside the window) and returns the energy as
    mean(samples) * window length, so the result isolates *sampling* error (it does not add the extra partial second
    that a "sum of samples x 1 s" over a longer log would).
    """
    import bisect
    if not rows or t1 <= t0:
        return {}
    times = [r["t"] for r in rows]
    n = max(1, int(t1 - t0))
    acc = {k: 0.0 for k in RAILS}
    for j in range(n):
        idx = max(0, bisect.bisect_right(times, t0 + 0.5 + j) - 1)
        for k in RAILS:
            acc[k] += rows[idx][k]
    span = t1 - t0
    return {f"E_{k}_J_1hz": acc[k] / n * span / 1000.0 for k in RAILS} | {"n_samples_1hz": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("out", type=Path)
    lg.add_argument("--hz", type=float, default=20.0)
    lg.add_argument("--seconds", type=float, default=None)
    sm = sub.add_parser("summarize")
    sm.add_argument("trace", type=Path)
    sm.add_argument("--t0", type=float, required=True)
    sm.add_argument("--t1", type=float, required=True)
    sm.add_argument("--method", choices=("backward", "trapezoid"), default="backward")
    args = ap.parse_args()
    if args.cmd == "log":
        log_trace(args.out, args.hz, args.seconds)
    else:
        rows = load_trace(args.trace)
        for k, v in {**integrate(rows, args.t0, args.t1, args.method), **window_stats(rows, args.t0, args.t1)}.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
