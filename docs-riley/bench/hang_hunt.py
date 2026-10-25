#!/usr/bin/env python3
"""Catch a slorado GPU hang and record where every thread is stuck. NOT a measurement (records nothing in runs.jsonl).

Background (campaign-review-2026-10-25.md): overlap-decode stages hang intermittently; just before the kill, power falls to ~6 W,
CPU ~0 %, GPU load reads 99.9 % and there is no output. This script runs one stage/cell repeatedly (as `-o /dev/null`) with slorado launched
as a child of gdb (kernel.yama.ptrace_scope = 1 forbids attaching to a non-child), and watches the output. When no line has appeared
for --silence seconds (after the first progress line) it:
  1. snapshots power, GPU load, temperature, memory and, for every thread, its state / wchan / syscall from /proc;
  2. sends SIGINT to slorado, so gdb stops it and prints `info threads` and `thread apply all bt` (needs ptrace; if gdb cannot
     stop the process within --gdb-timeout it is killed, which itself indicates threads in uninterruptible sleep);
  3. kills everything and writes bench-runs/hang-hunt/<timestamp>-<stage>-<cell>/ {console.log, snapshot.txt, gdb.txt, meta.json}.

    python3 hang_hunt.py [--stage spec] [--cell dna_fast_500k] [--max-hours 3] [--max-hangs 1] [--silence 180]
    python3 hang_hunt.py --selftest         # 1k HAC (~45 s run), force the dump path after 15 s: checks gdb/snapshot work, not a hang

Do not use the Jetson desktop while it runs. A 500k spec run hangs after ~40-50 min on average (it needs ~66 min to finish).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

import bench

OUT_DIR = bench.OUT_ROOT / "hang-hunt"
GPU_LOAD = Path("/sys/devices/platform/bus@0/17000000.gpu/load")


def sh(cmd: str, timeout: float = 20) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.SubprocessError:
        return "(timeout)"


def read(path: str | Path) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        return f"({e.strerror})"


def snapshot(pid: int, last_lines: list[str]) -> str:
    out = [f"time {time.strftime('%F %T')}  pid {pid}"]
    v = read("/sys/class/hwmon/hwmon1/in1_input")
    i = read("/sys/class/hwmon/hwmon1/curr1_input")
    try:
        out.append(f"VDD_IN {float(v) * float(i) / 1e6:.2f} W ({v} mV x {i} mA)")
    except ValueError:
        out.append(f"VDD_IN unreadable ({v}, {i})")
    out.append(f"gpu load node: {read(GPU_LOAD)} (permille)  tj: {bench.read_tj_C()} C")
    out.append("free -m:\n" + sh("free -m"))
    out.append("status:\n" + "\n".join(l for l in read(f"/proc/{pid}/status").splitlines() if l.split(':')[0] in ("State", "Threads", "VmRSS", "VmSwap")))
    out.append("threads (tid state comm | wchan | syscall):")
    try:
        tids = sorted(os.listdir(f"/proc/{pid}/task"), key=int)
    except OSError:
        tids = []
    for t in tids:
        stat = read(f"/proc/{pid}/task/{t}/stat")
        comm = stat[stat.find("(") + 1:stat.rfind(")")] if "(" in stat else "?"
        state = stat[stat.rfind(")") + 2:].split(" ")[0] if ")" in stat else "?"
        out.append(f"  {t:>8} {state} {comm:<16} | {read(f'/proc/{pid}/task/{t}/wchan'):<28} | {read(f'/proc/{pid}/task/{t}/syscall')}")
    out.append("last output lines:\n  " + "\n  ".join(last_lines[-8:]))
    out.append("kernel messages, last 10 min:\n" + sh("journalctl -k --since '-10 min' --no-pager 2>&1 | tail -15"))
    return "\n".join(out)


def one_run(cfg: dict, stage: str, cell: str, args: argparse.Namespace, run_dir: Path) -> dict:
    c = bench.parse_cell(cfg, cell)
    cmd = bench.build_cmd(cfg, stage, c, "/dev/null", None, None)
    gdb = ["gdb", "-q", "-batch", "-nx",
           "-ex", "set pagination off", "-ex", "set confirm off", "-ex", "set startup-with-shell off",
           "-ex", "set debuginfod enabled off", "-ex", "handle SIGPIPE nostop noprint pass",
           "-ex", "run",
           "-ex", "echo \\n===== STOPPED =====\\n", "-ex", "info threads", "-ex", "thread apply all bt 40", "-ex", "echo \\n===== END =====\\n",
           "-ex", "kill", "--args", *cmd]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cmd.txt").write_text(" ".join(shlex.quote(x) for x in cmd) + "\n")
    lines: list[str] = []
    state = {"last": time.time(), "progress": 0, "stopped_at": None}
    logf = open(run_dir / "console.log", "w")

    def reader(p: subprocess.Popen) -> None:
        for ln in p.stdout:  # type: ignore[union-attr]
            now = time.time()
            ln = ln.rstrip("\n")
            lines.append(ln)
            logf.write(f"{now:.1f} {ln}\n")
            logf.flush()
            if "[basecaller_main::" in ln and "processed" in ln:
                state["progress"] += 1
            if "===== STOPPED =====" in ln:
                state["stopped_at"] = now
            if "===== STOPPED =====" not in ln and state["stopped_at"] is None:
                state["last"] = now

    t0 = time.time()
    proc = subprocess.Popen(gdb, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True, cwd=str(bench.REPO))
    threading.Thread(target=reader, args=(proc,), daemon=True).start()
    result = {"hung": False, "completed": False, "forced": False, "elapsed_s": None, "batches": None}
    pid = None
    while proc.poll() is None:
        time.sleep(1)
        now = time.time()
        if pid is None:
            kids = sh(f"pgrep -P {proc.pid}").split()
            pid = int(kids[0]) if kids else None
        silent = now - state["last"] > args.silence and state["progress"] >= 1
        forced = args.selftest and now - t0 > args.selftest_after
        if (silent or forced) and pid and state["stopped_at"] is None:
            result.update(hung=True, forced=bool(forced and not silent), elapsed_s=round(now - t0, 1), batches=state["progress"])
            print(f"[hunt] {'FORCED (selftest)' if result['forced'] else 'HANG detected'}: no output for {now - state['last']:.0f} s after {state['progress']} batches "
                  f"({now - t0:.0f} s into the run). Snapshotting, then stopping under gdb ...", flush=True)
            (run_dir / "snapshot.txt").write_text(snapshot(pid, lines))
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                print("[hunt] the process had already exited; nothing to stop", flush=True)
            t_stop = time.time()
            while proc.poll() is None and time.time() - t_stop < args.gdb_timeout:
                time.sleep(1)
            if proc.poll() is None:
                (run_dir / "gdb-timeout.txt").write_text(f"gdb did not finish within {args.gdb_timeout} s of SIGINT: inferior could not be stopped "
                                                         "(threads likely in uninterruptible sleep). See snapshot.txt for thread states.\n")
                print("[hunt] gdb did not stop the process in time; killing it", flush=True)
            break
    for p_ in (pid,):
        if p_:
            try:
                os.kill(p_, signal.SIGKILL)
            except OSError:
                pass
    try:
        proc.kill()
    except OSError:
        pass
    subprocess.run(["pkill", "-9", "-f", "slorado basecaller"], capture_output=True)
    time.sleep(2)
    logf.close()
    text = (run_dir / "console.log").read_text()
    if "===== STOPPED =====" in text:
        tail = text[text.index("===== STOPPED ====="):].splitlines()
        # console.log lines are "<epoch> <text>"; drop the epoch for readability
        (run_dir / "gdb.txt").write_text("\n".join(l.split(" ", 1)[1] if l[:1].isdigit() and " " in l else l for l in tail) + "\n")
    if not result["hung"]:
        result.update(completed=("Real time" in text), elapsed_s=round(time.time() - t0, 1), batches=state["progress"])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="spec")
    ap.add_argument("--cell", default="dna_fast_500k")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-hangs", type=int, default=1)
    ap.add_argument("--silence", type=float, default=180.0, help="seconds without any output line that count as a hang")
    ap.add_argument("--gdb-timeout", type=float, default=240.0)
    ap.add_argument("--selftest", action="store_true", help="1k HAC, force the dump path early (checks gdb, /proc snapshot, file output)")
    ap.add_argument("--selftest-after", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        args.cell, args.max_hangs = "dna_hac_1k", 1

    cfg = bench.load_config()
    snap, problems = bench.preflight(cfg, [args.stage], True, [args.cell])
    if problems:
        print("PRE-FLIGHT FAILED:\n  " + "\n  ".join(problems))
        return 2
    c = bench.parse_cell(cfg, args.cell)
    print("command under gdb:", " ".join(bench.build_cmd(cfg, args.stage, c, "/dev/null", None, None)))
    if args.dry_run:
        return 0
    bench.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(bench.LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2

    hangs, n, t_start = 0, 0, time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    summary = []
    while hangs < args.max_hangs and (time.time() - t_start) < args.max_hours * 3600:
        n += 1
        run_dir = OUT_DIR / f"{stamp}-{args.stage}-{args.cell}{'-selftest' if args.selftest else ''}-{n}"
        print(f"[hunt] run {n} ({(time.time() - t_start) / 3600:.2f} h used of {args.max_hours} h) -> {run_dir}", flush=True)
        res = one_run(cfg, args.stage, args.cell, args, run_dir)
        res.update(run=n, dir=str(run_dir))
        (run_dir / "meta.json").write_text(json.dumps(res, indent=1))
        summary.append(res)
        print(f"[hunt] run {n}: {res}", flush=True)
        if res["hung"]:
            hangs += 1
    print(f"[hunt] finished: {hangs} hang(s) captured in {n} run(s), {(time.time() - t_start) / 3600:.2f} h")
    print("[hunt] look at: " + ", ".join(f"{r['dir']}/gdb.txt" for r in summary if r["hung"]))
    return 0 if hangs else 1


if __name__ == "__main__":
    raise SystemExit(main())
