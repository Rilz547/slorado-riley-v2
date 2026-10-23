#!/usr/bin/env python3
"""Unified timing + power harness for slorado on the Jetson Orin (stdlib only).

One driver captures, in the *same* runs: slorado's own timings, per-rail power/energy over exactly slorado's
`Real time` window, system state (clock, load, temperature, swap) and, in mode B, the FASTQ for accuracy.

Modes
    A  raw speed          -o /dev/null
    B  end-to-end speed   -o <FASTQ on NVMe>   (the FASTQ is counted, a `sync` is timed, one per cell is kept)

Timed window: slorado prints progress lines `[basecaller_main::<elapsed>*...]` where <elapsed> is seconds since its
internal start (realtime0).  The first such line, timestamped on arrival, gives realtime0's epoch; the window is
[realtime0, realtime0 + Real time].  No change to slorado is needed, so every historical commit works.

Subcommands
    check                          pre-flight report (no runs)
    idle   [--seconds N]           record an idle baseline
    canary                         fixed GPU micro-benchmark under the power logger
    run    --stage S [--cells ...] [--modes A,B] ...    full stage block (idle, canary, cells, canary, idle)
    status                         progress summary from runs.jsonl
    plan   [--stages ...]          time estimate for the whole campaign
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import powerlog

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
CONFIG_PATH = BENCH_DIR / "stages.json"
OUT_ROOT = REPO / "docs-riley" / "bench-runs"
RUNS_JSONL = OUT_ROOT / "runs.jsonl"
POWER_DIR = OUT_ROOT / "power"
CONSOLE_DIR = OUT_ROOT / "console"
LOCK_PATH = OUT_ROOT / ".bench.lock"
BENCH_OUT = Path.home() / "bench-out"          # temporary FASTQs (deleted after counting)
KEPT_DIR = OUT_ROOT / "kept"                    # one FASTQ per cell kept for accuracy: output_<model>_<size>-<dtype>-<stage>.fastq
RUNS_FILE = RUNS_JSONL                          # active records file (smoke tests write elsewhere via --runs-file)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PROGRESS_RE = re.compile(r"\[basecaller_main::([0-9.]+)\*([0-9.]+)\]")
REAL_RE = re.compile(r"\[main\]\s+Real time:\s*([0-9.]+)\s*sec;\s*CPU time:\s*([0-9.]+)\s*sec;\s*Peak RAM:\s*([0-9.]+)\s*GB")
TIMING_RE = re.compile(r"^\[basecaller_main\]\s+(?:-\s+)?([A-Za-z_ ]+?):\s+([0-9.]+) sec")
TIMING_KEYS = {
    "model initialization", "data loading", "data processing", "parse", "preprocess", "runners", "accept",
    "basecall", "inference", "decode", "postprocess", "data output", "conv_stack", "rnns",
}
SPEC_RE = re.compile(r"speculative: drafted (\d+), repaired (\d+)")


def harness_version() -> str:
    h = hashlib.sha256()
    for name in ("bench.py", "powerlog.py", "canary.py", "accuracy.py", "stages.json"):
        h.update((BENCH_DIR / name).read_bytes())
    return h.hexdigest()[:12]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_cell(cfg: dict, cell: str) -> dict:
    """'rna_hac_20k' -> dtype, model, size token, dataset key and dataset dict. A two-part id ('fast_1k') means DNA."""
    parts = cell.split("_")
    if len(parts) == 2:
        parts = ["dna", *parts]
    if len(parts) != 3:
        raise SystemExit(f"bad cell id {cell!r}; expected <dtype>_<model>_<size>, e.g. dna_fast_1k")
    dtype, model, size = parts
    key = f"{dtype}_{size}"
    if key not in cfg["datasets"]:
        raise SystemExit(f"unknown dataset {key!r} for cell {cell!r}; known: {', '.join(cfg['datasets'])}")
    ds = cfg["datasets"][key]
    if model not in cfg["models"][dtype]:
        raise SystemExit(f"unknown model {model!r} for {dtype}")
    return {"cell": f"{dtype}_{model}_{size}", "dtype": dtype, "model": model, "size_token": size, "dataset": key, "ds": ds}


def dataset_path(ds: dict) -> Path:
    p = expand(ds["path"])
    return p if p.is_absolute() else REPO / p


def model_path(cfg: dict, dtype: str, model: str) -> Path:
    return REPO / cfg["models"][dtype][model]


def wall_cap_s(cfg: dict, c: dict) -> float:
    lim = cfg["run_limits"]
    samples = c["ds"].get("samples") or 1e9
    return max(lim["min_cap_s"], lim["cap_factor"] * samples / lim["floor_rate_samples_per_s"][c["model"]])


def default_cells(cfg: dict, stage: str) -> list[str]:
    if "default_cells" in cfg["stages"][stage]:
        return list(cfg["stages"][stage]["default_cells"])
    cells = []
    for key, ds in cfg["datasets"].items():
        if not ds.get("default"):
            continue
        if ds["dtype"] == "rna" and stage not in cfg["rna_stages"]:
            continue
        for model in ds["models"]:
            cells.append(f"{ds['dtype']}_{model}_{key.split('_', 1)[1]}")
    return cells


def stage_flags(cfg: dict, stage: str) -> list[str]:
    return cfg["stages"][stage].get("flags", cfg["shared_flags"])


# ---------------------------------------------------------------- system helpers

def sh(cmd: list[str], timeout: float = 30) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def read_tj_C() -> float | None:
    z = powerlog.find_tj_zone()
    v = powerlog._read_int(z) if z else None
    return v / 1000.0 if v is not None else None


def system_snapshot() -> dict:
    ram, swap = powerlog.mem_used_mb()
    return {
        "nvpmodel": " ".join(sh(["nvpmodel", "-q"]).split()),
        "gpu_min_mhz": (powerlog._read_int(powerlog.GPU_DEVFREQ + "/min_freq") or 0) / 1e6,
        "gpu_max_mhz": (powerlog._read_int(powerlog.GPU_DEVFREQ + "/max_freq") or 0) / 1e6,
        "gpu_cur_mhz": (powerlog._read_int(powerlog.GPU_DEVFREQ + "/cur_freq") or 0) / 1e6,
        "ram_used_mb": ram, "swap_used_mb": swap, "tj_C": read_tj_C(),
    }


def other_gpu_jobs() -> list[str]:
    """Processes that would disturb a measurement, matched on executable name (not on whole command lines)."""
    bad_comm = {"slorado", "nsys", "nsys-launcher", "ncu", "ncu-ui", "tegrastats"}
    bad_scripts = ("rileys-runner.py", "canary.py", "bench.py", "power-record", "spec-fastq-sweep", "lstm_scaling.py")
    me = os.getpid()
    out = []
    for ln in sh(["ps", "-eo", "pid=,comm=,args="]).splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == me:
            continue
        comm = parts[1]
        args = parts[2] if len(parts) > 2 else ""
        if comm in bad_comm or (comm.startswith(("python", "bash", "sh")) and any(f"/{x}" in args or args.endswith(x) or f" {x}" in args for x in bad_scripts)
                                and "-c" not in args.split()[:3]):
            out.append(ln[:100])
    return out


def preflight(cfg: dict, stage_names: list[str], need_stage_bins: bool = True, cells: list[str] | None = None) -> tuple[dict, list[str]]:
    problems: list[str] = []
    snap = system_snapshot()
    if "25W" not in snap["nvpmodel"]:
        problems.append(f"nvpmodel is not 25W: {snap['nvpmodel']!r}")
    if snap["gpu_min_mhz"] and snap["gpu_min_mhz"] == snap["gpu_max_mhz"]:
        problems.append("GPU min==max frequency: jetson_clocks appears to be ON")
    jobs = other_gpu_jobs()
    if jobs:
        problems.append("other GPU-related processes running: " + "; ".join(j[:80] for j in jobs))
    needed_models: set[tuple[str, str]] = set()
    needed_ds: set[str] = set()
    for cell in (cells or []):
        c = parse_cell(cfg, cell)
        needed_models.add((c["dtype"], c["model"]))
        needed_ds.add(c["dataset"])
    for dtype, model in sorted(needed_models):
        if not model_path(cfg, dtype, model).exists():
            problems.append(f"missing model {cfg['models'][dtype][model]}")
    for key in sorted(needed_ds):
        if not dataset_path(cfg["datasets"][key]).exists():
            problems.append(f"missing dataset {key}: {dataset_path(cfg['datasets'][key])}")
        ref = cfg["references"][cfg["datasets"][key]["dtype"]]["fasta"]
        if not expand(ref).exists():
            problems.append(f"missing reference {ref} (needed for accuracy of {key})")
    if shutil.disk_usage(Path.home()).free < 5 * 2**30:
        problems.append("less than 5 GiB free in $HOME")
    if need_stage_bins:
        for s in stage_names:
            b = stage_binary(cfg, s)
            if not b.exists():
                problems.append(f"missing binary for stage {s}: {b}")
    try:
        powerlog.find_ina3221()
    except SystemExit as e:
        problems.append(str(e))
    return snap, problems


def stage_binary(cfg: dict, stage: str, override: Path | None = None) -> Path:
    if override is not None:
        return override / "slorado"
    wt = cfg["stages"][stage]["worktree"]
    return expand(cfg["worktree_root"]) / wt / "slorado"


def stage_commit(cfg: dict, stage: str, override: Path | None = None) -> str:
    d = override if override is not None else expand(cfg["worktree_root"]) / cfg["stages"][stage]["worktree"]
    return sh(["git", "-C", str(d), "rev-parse", "--short", "HEAD"]) or "unknown"


def wait_for_gate(cfg: dict) -> dict:
    gate = cfg["gate"]
    t0 = time.time()
    tj = read_tj_C()
    while tj is not None and tj > gate["tj_C"] and time.time() - t0 < gate["timeout_s"]:
        time.sleep(5)
        tj = read_tj_C()
    waited = time.time() - t0
    return {"gate_waited_s": round(waited, 1), "gate_timed_out": bool(tj is not None and tj > gate["tj_C"]), "tj_at_start_C": tj}


# ---------------------------------------------------------------- power logger process

class Logger:
    def __init__(self, trace_path: Path, hz: float, tegrastats_log: Path | None = None):
        self.path = trace_path
        self.hz = hz
        self.tegra_log = tegrastats_log
        self.proc: subprocess.Popen | None = None
        self.tegra: subprocess.Popen | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen([sys.executable, str(BENCH_DIR / "powerlog.py"), "log", str(self.path), "--hz", str(self.hz)])
        if self.tegra_log is not None:
            subprocess.run(["tegrastats", "--stop"], capture_output=True)
            self.tegra = subprocess.Popen(["tegrastats", "--interval", "1000", "--logfile", str(self.tegra_log)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 5:
            if self.path.exists() and self.path.stat().st_size > 200:
                return
            time.sleep(0.05)
        raise RuntimeError("power logger did not produce output")

    def stop(self) -> Path:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.tegra:
            subprocess.run(["tegrastats", "--stop"], capture_output=True)
            try:
                self.tegra.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tegra.kill()
        gz = self.path.with_suffix(".csv.gz")
        with open(self.path, "rb") as f_in, gzip.open(gz, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
        self.path.unlink()
        return gz


# ---------------------------------------------------------------- one slorado run

def parse_console(lines: list[str]) -> dict:
    out: dict = {"timings": {}}
    for ln in lines:
        s = strip_ansi(ln)
        m = REAL_RE.search(s)
        if m:
            out["real_s"], out["cpu_s"], out["peak_ram_gb"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
        m = TIMING_RE.match(s)
        if m and m.group(1).strip() in TIMING_KEYS and m.group(1).strip() not in out["timings"]:
            out["timings"][m.group(1).strip()] = float(m.group(2))
        m = SPEC_RE.search(s)
        if m:
            out["spec_drafted"], out["spec_repaired"] = int(m.group(1)), int(m.group(2))
    return out


def count_fastq(path: Path) -> tuple[int, int]:
    reads = bases = 0
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                reads += 1
                bases += len(line.rstrip(b"\n"))
    return reads, bases


def build_cmd(cfg: dict, stage: str, c: dict, out_path: str, override: Path | None, extra_args: list[str] | None = None) -> list[str]:
    st = cfg["stages"][stage]
    return [str(stage_binary(cfg, stage, override)), "basecaller", *stage_flags(cfg, stage), *st["args"],
            *st["model_args"].get(c["model"], []), *(extra_args or []), "-o", out_path,
            str(model_path(cfg, c["dtype"], c["model"])), str(dataset_path(c["ds"]))]


def run_slorado_measured(cfg: dict, stage: str, cell: str, mode: str, rep: int, kind: str, block: str,
                         keep: bool, override: Path | None, with_tegrastats: bool,
                         extra_args: list[str] | None = None, keep_as: str | None = None) -> dict:
    c = parse_cell(cfg, cell)
    cell, model, size, dtype = c["cell"], c["model"], c["ds"]["size"], c["dtype"]
    rid = f"{stage}__{cell}__{mode}__{kind}{rep}__{int(time.time() * 1000)}"
    pw = cfg["power"]
    limits = cfg["run_limits"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_OUT / "tmp").mkdir(parents=True, exist_ok=True)
    KEPT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = "/dev/null" if mode == "A" else str(BENCH_OUT / "tmp" / f"{rid}.fastq")
    cmd = build_cmd(cfg, stage, c, out_path, override, extra_args)
    rec: dict = {
        "run_id": rid, "block": block, "harness": harness_version(), "kind": kind, "stage": stage, "cell": cell,
        "dtype": dtype, "config": model, "size": size, "dataset": c["dataset"], "samples": c["ds"].get("samples"),
        "reads_in": c["ds"].get("reads"), "mode": mode, "rep": rep, "commit": stage_commit(cfg, stage, override),
        "cmd": " ".join(cmd), "status": "failed",
    }
    rec.update(wait_for_gate(cfg))
    rec["sys_start"] = system_snapshot()

    tegra_log = (POWER_DIR / f"{rid}.tegrastats.log") if with_tegrastats else None
    trace = POWER_DIR / f"{rid}.csv"
    logger = Logger(trace, pw["hz"], tegra_log)
    logger.start()
    time.sleep(pw["pre_shoulder_s"])

    lines: list[tuple[float, str]] = []
    state = {"last": time.time()}

    def reader(p: subprocess.Popen) -> None:
        for ln in p.stdout:  # type: ignore[union-attr]
            now = time.time()
            lines.append((now, ln.rstrip("\n")))
            state["last"] = now

    t_launch = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True, cwd=str(REPO))
    th = threading.Thread(target=reader, args=(proc,), daemon=True)
    th.start()
    cap = wall_cap_s(cfg, c)
    killed = ""
    while proc.poll() is None:
        time.sleep(0.5)
        now = time.time()
        if now - state["last"] > limits["stall_s"]:
            killed = "stall"
        elif now - t_launch > cap:
            killed = "wall_cap"
        if killed:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
    proc.wait()
    th.join(timeout=5)
    t_exit = time.time()

    sync_s = None
    if mode == "B" and not killed:
        s0 = time.time()
        subprocess.run(["sync"])
        sync_s = time.time() - s0
    time.sleep(pw["post_shoulder_s"])
    gz = logger.stop()

    console = [ln for _, ln in lines]
    (CONSOLE_DIR / f"{rid}.log").write_text("\n".join(strip_ansi(c) for c in console) + "\n")
    parsed = parse_console(console)
    rec.update({"returncode": proc.returncode, "killed": killed, "sync_s": sync_s, "t_launch_epoch": t_launch,
                "t_exit_epoch": t_exit, "wall_process_s": round(t_exit - t_launch, 3), "power_trace": str(gz.relative_to(REPO)), "console": str((CONSOLE_DIR / f"{rid}.log").relative_to(REPO))})
    if tegra_log is not None:
        rec["tegrastats_log"] = str(tegra_log.relative_to(REPO))
    rec.update({k: v for k, v in parsed.items() if k != "timings"})
    rec["timings"] = parsed["timings"]

    if killed or proc.returncode != 0 or "real_s" not in parsed:
        rec["error"] = killed or (f"returncode {proc.returncode}" if proc.returncode else "no Real time line")
        if mode == "B":
            Path(out_path).unlink(missing_ok=True)
        return rec

    # window from slorado's own clock: preferred anchor = progress lines `[basecaller_main::<elapsed>*...]` (elapsed since
    # realtime0); fallback (stream-pipeline builds print none) = arrival time of the `[main] Real time:` line, which marks the window end.
    ests = []
    t_real_line = None
    for t_line, ln in lines:
        s_ln = strip_ansi(ln)
        if len(ests) < 6:
            m = PROGRESS_RE.search(s_ln)
            if m:
                ests.append(t_line - float(m.group(1)))
        if t_real_line is None and REAL_RE.search(s_ln):
            t_real_line = t_line
    if ests:
        w0 = min(ests)
        w1 = w0 + parsed["real_s"]
        rec.update({"anchor_method": "progress", "anchor_spread_ms": round((max(ests) - w0) * 1000, 1)})
        if t_real_line is not None:
            rec["real_line_after_window_end_ms"] = round((t_real_line - w1) * 1000, 1)   # cross-check of the two anchors
    elif t_real_line is not None:
        w1 = t_real_line
        w0 = w1 - parsed["real_s"]
        rec.update({"anchor_method": "real_line", "anchor_spread_ms": None})
    else:
        rec["error"] = "cannot recover slorado start time (no progress lines and no Real time line)"
        if mode == "B":
            Path(out_path).unlink(missing_ok=True)
        return rec
    rec.update({"window_start_epoch": w0, "window_end_epoch": w1, "exit_gap_s": round(t_exit - w1, 3)})
    if rec.get("samples"):
        rec["samples_per_s"] = rec["samples"] / parsed["real_s"]

    rows = powerlog.load_trace(gz)
    en = powerlog.integrate(rows, w0, w1)
    rec.update({k: v for k, v in en.items() if k not in ("method",)})
    rec["E_in_J_proc"] = powerlog.integrate(rows, t_launch, t_exit).get("E_in_J")  # whole-process energy (old-style window)
    rec["E_in_J_1hz_emulated"] = powerlog.emulate_1hz(rows, w0, w1).get("E_in_J_1hz")
    pre = powerlog.integrate(rows, t_launch - pw["pre_shoulder_s"] + 1.0, t_launch - 0.2)
    post = powerlog.integrate(rows, t_exit + 1.0, t_exit + pw["post_shoulder_s"] - 0.2)
    rec["P_pre_in_W"], rec["P_post_in_W"] = pre.get("Pmean_in_W"), post.get("Pmean_in_W")
    rec.update(powerlog.window_stats(rows, w0, w1))

    if mode == "B":
        fq = Path(out_path)
        if fq.exists():
            rec["fastq_bytes"] = fq.stat().st_size
            rec["reads_out"], rec["bases_out"] = count_fastq(fq)
            if keep:
                dest = KEPT_DIR / (keep_as or f"output_{model}_{size}-{dtype}-{stage}.fastq")   # matches fetch-and-minimap.py naming (APPEND = -<dtype>-<stage>)
                shutil.move(str(fq), dest)
                rec["fastq_kept"] = str(dest)
            else:
                fq.unlink()
        else:
            rec["error"] = "mode B but no FASTQ written"
            return rec
    rec["status"] = "ok"
    return rec


def append_record(rec: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RUNS_FILE, "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_records(path: Path | None = None) -> list[dict]:
    path = path or RUNS_FILE
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def done_keys(records: list[dict]) -> set[tuple]:
    return {(r["stage"], r["cell"], r["mode"], r["kind"], r["rep"]) for r in records
            if r.get("status") == "ok" and r.get("kind") in ("timed", "warmup")}


def run_with_retries(*args, **kw) -> dict:
    cfg = args[0]
    attempts = cfg["run_limits"]["attempts"]
    rec = {}
    for a in range(1, attempts + 1):
        rec = run_slorado_measured(*args, **kw)
        rec["attempt"] = a
        append_record(rec)
        if rec["status"] == "ok":
            return rec
        print(f"    attempt {a}/{attempts} failed: {rec.get('error')}", flush=True)
        subprocess.run(["pkill", "-9", "-f", "slorado basecaller"], capture_output=True)
        time.sleep(15)
    return rec


# ---------------------------------------------------------------- idle / canary

def record_idle(cfg: dict, label: str, block: str, seconds: float | None = None) -> dict:
    pw = cfg["power"]
    seconds = seconds or pw["idle_s"]
    rid = f"idle__{label}__{int(time.time() * 1000)}"
    trace = POWER_DIR / f"{rid}.csv"
    logger = Logger(trace, pw["hz"])
    logger.start()
    t0 = time.time()
    time.sleep(seconds)
    t1 = time.time()
    gz = logger.stop()
    rows = powerlog.load_trace(gz)
    en = powerlog.integrate(rows, t0 + 2.0, t1)
    rec = {"run_id": rid, "block": block, "harness": harness_version(), "kind": "idle", "label": label, "stage": label.split(":")[0],
           "cell": "idle", "mode": "-", "rep": 0, "status": "ok", "t_start_epoch": t0, "t_end_epoch": t1,
           "power_trace": str(gz.relative_to(REPO)), "sys": system_snapshot()}
    rec.update({k: v for k, v in en.items() if k != "method"})
    rec.update(powerlog.window_stats(rows, t0 + 2.0, t1))
    append_record(rec)
    return rec


def record_canary(cfg: dict, label: str, block: str) -> dict:
    pw = cfg["power"]
    rid = f"canary__{label}__{int(time.time() * 1000)}"
    trace = POWER_DIR / f"{rid}.csv"
    logger = Logger(trace, pw["hz"])
    logger.start()
    time.sleep(pw["pre_shoulder_s"])
    t0 = time.time()
    p = subprocess.run([str(expand(cfg["canary_python"])), str(BENCH_DIR / "canary.py")], capture_output=True, text=True)
    t1 = time.time()
    time.sleep(pw["post_shoulder_s"])
    gz = logger.stop()
    ms = None
    for ln in p.stdout.splitlines():
        if ln.startswith("CANARY_MS="):
            ms = float(ln.split("=")[1])
    rows = powerlog.load_trace(gz)
    en = powerlog.integrate(rows, t0, t1)
    rec = {"run_id": rid, "block": block, "harness": harness_version(), "kind": "canary", "label": label, "stage": label.split(":")[0],
           "cell": "canary", "mode": "-", "rep": 0, "status": "ok" if ms else "failed", "canary_ms": ms,
           "t_start_epoch": t0, "t_end_epoch": t1, "power_trace": str(gz.relative_to(REPO)), "sys": system_snapshot()}
    if not ms:
        rec["error"] = (p.stderr or "")[-300:]
    rec.update({k: v for k, v in en.items() if k != "method"})
    rec.update(powerlog.window_stats(rows, t0, t1))
    append_record(rec)
    return rec


# ---------------------------------------------------------------- stage block

def resolve_cells(cfg: dict, stage: str, arg: str | None) -> list[str]:
    raw = arg.split(",") if arg else default_cells(cfg, stage)
    return [parse_cell(cfg, c)["cell"] for c in raw]


def run_block(args: argparse.Namespace) -> int:
    global RUNS_FILE
    if args.runs_file:
        RUNS_FILE = Path(args.runs_file)
    cfg = load_config()
    stage = args.stage
    if stage not in cfg["stages"]:
        print(f"unknown stage {stage}; known: {', '.join(cfg['stage_order'])}")
        return 2
    if cfg["stages"][stage].get("invalid") and not args.allow_invalid:
        print(f"stage {stage} is marked INVALID: {cfg['stages'][stage]['invalid']}\n(use --allow-invalid to run it anyway)")
        return 2
    override = expand(args.stage_dir) if args.stage_dir else None
    cells = resolve_cells(cfg, stage, args.cells)
    snap, problems = preflight(cfg, [stage], need_stage_bins=override is None, cells=cells)
    if override is not None and not stage_binary(cfg, stage, override).exists():
        problems.append(f"missing binary {stage_binary(cfg, stage, override)}")
    if problems:
        print("PRE-FLIGHT FAILED:\n  " + "\n  ".join(problems))
        return 2
    block = f"{stage}-{time.strftime('%Y%m%dT%H%M%S')}"
    print(f"[bench] block {block}  harness {harness_version()}  commit {stage_commit(cfg, stage, override)}  cells {len(cells)}", flush=True)
    done = done_keys(load_records()) if args.resume else set()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2

    if not args.smoke:
        print("[bench] idle baseline (before)", flush=True)
        record_idle(cfg, f"{stage}:before", block)
        print("[bench] canary (before)", flush=True)
        record_canary(cfg, f"{stage}:before", block)

    for cell in cells:
        c = parse_cell(cfg, cell)
        ds = c["ds"]
        modes = args.modes.split(",") if args.modes else ds["modes"]
        n = args.reps if args.reps else ds["reps"]
        warm_mode = modes[0]
        if args.warmup != "none" and ds.get("warmup", True) and (stage, cell, warm_mode, "warmup", 0) not in done:
            print(f"[bench] {stage} {cell} warm-up (mode {warm_mode})", flush=True)
            run_with_retries(cfg, stage, cell, warm_mode, 0, "warmup", block, False, override, False)
        for mode in modes:
            if args.warmup == "per-mode" and mode != warm_mode and ds.get("warmup", True) and (stage, cell, mode, "warmup", 0) not in done:
                print(f"[bench] {stage} {cell} warm-up (mode {mode})", flush=True)
                run_with_retries(cfg, stage, cell, mode, 0, "warmup", block, False, override, False)
            for rep in range(1, n + 1):
                if (stage, cell, mode, "timed", rep) in done:
                    continue
                keep = (mode == "B" and rep == 1)
                print(f"[bench] {stage} {cell} mode {mode} rep {rep}/{n}", flush=True)
                rec = run_with_retries(cfg, stage, cell, mode, rep, "timed", block, keep, override, args.tegrastats)
                if rec["status"] == "ok":
                    print(f"        Real {rec['real_s']:.3f}s  proc {rec['wall_process_s']:.1f}s  E_in {rec.get('E_in_J', float('nan')):.1f} J  "
                          f"P {rec.get('Pmean_in_W', float('nan')):.2f} W  gpu {rec.get('gpu_mhz_mean') or 0:.0f} MHz  tj {rec.get('tj_max_C') or 0:.1f} C",
                          flush=True)
                else:
                    print(f"        FAILED: {rec.get('error')}", flush=True)

    if not args.smoke:
        print("[bench] canary (after)", flush=True)
        record_canary(cfg, f"{stage}:after", block)
        print("[bench] idle baseline (after)", flush=True)
        record_idle(cfg, f"{stage}:after", block)
    print("[bench] block complete", flush=True)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """One short run per (stage, cell) in mode B, then an accuracy sanity check on the Jetson. Not a measurement:
    records go to smoke.jsonl, never to runs.jsonl."""
    global RUNS_FILE
    RUNS_FILE = OUT_ROOT / "smoke.jsonl"
    import accuracy
    cfg = load_config()
    stages = args.stages.split(",")
    rows = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2
    block = f"smoke-{time.strftime('%Y%m%dT%H%M%S')}"
    for stage in stages:
        cells = resolve_cells(cfg, stage, args.cells) if args.cells else [c for c in default_cells(cfg, stage) if c.endswith("_1k")]
        _, problems = preflight(cfg, [stage], True, cells)
        if problems:
            print(f"PRE-FLIGHT FAILED for {stage}:\n  " + "\n  ".join(problems))
            return 2
        for cell in cells:
            c = parse_cell(cfg, cell)
            print(f"[smoke] {stage} {cell} ...", flush=True)
            rec = run_slorado_measured(cfg, stage, cell, "B", 1, "smoke", block, True, None, False)
            append_record(rec)
            row = {"stage": stage, "cell": cell, "status": rec["status"], "real_s": rec.get("real_s"), "reads_out": rec.get("reads_out"),
                   "reads_in": rec.get("reads_in"), "error": rec.get("error"), "alpha": None, "identity": None, "map_rate": None}
            if rec.get("spec_drafted"):
                row["alpha"] = 1 - rec["spec_repaired"] / rec["spec_drafted"]
            if rec["status"] == "ok" and rec.get("fastq_kept") and not args.no_accuracy:
                try:
                    acc = accuracy.median_identity(Path(rec["fastq_kept"]), c["dtype"], cfg, threads=args.threads)
                    row.update({"identity": acc["median"], "map_rate": acc["map_rate"]})
                except RuntimeError as e:
                    row["error"] = "accuracy skipped (DNA mapping is done on the PC)"
            rows.append(row)
    print()
    print(f"{'stage':12s} {'cell':14s} {'status':7s} {'Real s':>8s} {'reads out/in':>13s} {'median id':>10s} {'map %':>6s} {'alpha':>6s}  note")
    for r in rows:
        ro = f"{r['reads_out']}/{r['reads_in']}" if r["reads_out"] is not None else "-"
        print(f"{r['stage']:12s} {r['cell']:14s} {r['status']:7s} {(r['real_s'] or 0):8.2f} {ro:>13s} "
              f"{(r['identity'] if r['identity'] is not None else float('nan')):10.6f} {(100*r['map_rate'] if r['map_rate'] is not None else float('nan')):6.1f} "
              f"{(r['alpha'] if r['alpha'] is not None else float('nan')):6.3f}  {r['error'] or ''}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


def cmd_explore(args: argparse.Namespace) -> int:
    """Grid over extra slorado flags for one stage/cell, each run checked for accuracy against a matched non-speculative reference.
    Exploratory (single runs, records in explore.jsonl); not part of the measured campaign."""
    global RUNS_FILE
    import accuracy, itertools as it
    RUNS_FILE = OUT_ROOT / "explore.jsonl"
    cfg = load_config()
    c = parse_cell(cfg, args.cell)
    grid = {}
    for part in args.grid.split(";"):
        k, v = part.split("=")
        grid[k.strip()] = [x.strip() for x in v.split(",")]
    names = list(grid)
    combos = list(it.product(*[grid[n] for n in names]))
    _, problems = preflight(cfg, [args.stage, args.ref_stage], True, [args.cell])
    if problems:
        print("PRE-FLIGHT FAILED:\n  " + "\n  ".join(problems))
        return 2
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another bench.py is running (lock held)")
        return 2
    block = f"explore-{time.strftime('%Y%m%dT%H%M%S')}"
    tag = args.cell.replace("_", "")
    results = []

    def one(stage, extra, label):
        keep_as = f"output_{c['model']}_{c['ds']['size']}-{c['dtype']}-explore-{label}.fastq"
        rec = run_slorado_measured(cfg, stage, c["cell"], "B", 1, "explore", block, True, None, False, extra, keep_as)
        append_record(rec)
        row = {"label": label, "stage": stage, "extra": " ".join(extra), "status": rec["status"], "real_s": rec.get("real_s"), "alpha": None,
               "identity": None, "map_rate": None, "bases": rec.get("bases_out"), "fastq": rec.get("fastq_kept")}
        if rec.get("spec_drafted"):
            row["alpha"] = 1 - rec["spec_repaired"] / rec["spec_drafted"]
        if rec["status"] == "ok" and rec.get("fastq_kept"):
            acc = accuracy.median_identity(Path(rec["fastq_kept"]), c["dtype"], cfg, threads=args.threads)
            row.update({"identity": acc["median"], "map_rate": acc["map_rate"]})
        results.append(row)
        print(f"  {label:22s} {row['status']:6s} Real {row['real_s'] or 0:6.2f}s  alpha {row['alpha'] if row['alpha'] is not None else float('nan'):.3f}  identity {row['identity'] if row['identity'] is not None else float('nan'):.6f}", flush=True)
        return row

    print(f"[explore] reference: {args.ref_stage} (non-speculative)", flush=True)
    ref = one(args.ref_stage, [], f"ref-{args.ref_stage}")
    for combo in combos:
        extra = []
        for n, v in zip(names, combo):
            extra += [f"--{n}", v] if not v.startswith("=") else [f"--{n}{v}"]
        label = "_".join(f"{n.replace('spec-', '').replace('-threshold', '')}{v}" for n, v in zip(names, combo))
        print(f"[explore] {args.stage} {label}", flush=True)
        one(args.stage, extra, label)
    out = OUT_ROOT / f"explore-{tag}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps({"cell": args.cell, "stage": args.stage, "ref": ref, "results": results}, indent=1))
    print(f"\n{'setting':24s} {'alpha':>6s} {'Real s':>7s} {'vs ref':>8s} {'identity':>9s} {'d pp':>7s} {'bases/ref':>9s}")
    for r in results:
        d = (r["identity"] - ref["identity"]) * 100 if r["identity"] is not None and ref["identity"] is not None else float("nan")
        sp = (r["real_s"] - ref["real_s"]) / ref["real_s"] * 100 if r["real_s"] and ref["real_s"] else float("nan")
        br = r["bases"] / ref["bases"] if r["bases"] and ref["bases"] else float("nan")
        print(f"{r['label']:24s} {(r['alpha'] if r['alpha'] is not None else float('nan')):6.3f} {r['real_s'] or 0:7.2f} {sp:+7.1f}% {r['identity'] or float('nan'):9.6f} {d:+7.3f} {br:9.3f}")
    print("wrote", out)
    return 0


# ---------------------------------------------------------------- misc subcommands

def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config()
    snap, problems = preflight(cfg, cfg["stage_order"], need_stage_bins=False)
    print(json.dumps(snap, indent=1))
    print("harness version:", harness_version())
    for s in cfg["stage_order"]:
        b = stage_binary(cfg, s)
        print(f"  stage {s:13s} binary {'OK     ' if b.exists() else 'MISSING'} {b}")
    print("datasets:")
    for key, ds in cfg["datasets"].items():
        pth = dataset_path(ds)
        ok = pth.exists()
        print(f"  {key:10s} {'OK     ' if ok else 'MISSING'} reads {ds['reads']:>7d} samples {ds['samples'] or 0:>14d} reps {ds['reps']} modes {','.join(ds['modes'])} {pth}")
    for dtype, ref in cfg["references"].items():
        print(f"  reference {dtype}: {'OK     ' if expand(ref['fasta']).exists() else 'MISSING'} {ref['fasta']}")
    hwmon, ch = powerlog.find_ina3221()
    print("power sensor:", hwmon, ch, "| gpu load node:", powerlog.find_gpu_load(), "| tj zone:", powerlog.find_tj_zone())
    print("problems:", problems or "none")
    return 1 if [p for p in problems if "missing binary" not in p] else 0


def cmd_idle(args: argparse.Namespace) -> int:
    cfg = load_config()
    rec = record_idle(cfg, args.label, f"idle-{time.strftime('%Y%m%dT%H%M%S')}", args.seconds)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k.startswith(("Pmean", "n_updates", "gpu_mhz", "tj"))})
    return 0


def cmd_canary(args: argparse.Namespace) -> int:
    cfg = load_config()
    rec = record_canary(cfg, args.label, f"canary-{time.strftime('%Y%m%dT%H%M%S')}")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k in ("canary_ms", "E_in_J", "Pmean_in_W", "gpu_mhz_mean", "tj_max_C", "status")})
    return 0 if rec["status"] == "ok" else 1


def cmd_status(args: argparse.Namespace) -> int:
    recs = load_records()
    from collections import Counter
    c = Counter((r["stage"], r["cell"], r["mode"]) for r in recs if r.get("status") == "ok" and r.get("kind") == "timed")
    for k in sorted(c):
        print(f"{k[0]:13s} {k[1]:14s} mode {k[2]}: {c[k]} timed runs ok")
    bad = [r for r in recs if r.get("status") != "ok"]
    print(f"{len(recs)} records, {len(bad)} failed/other")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Time estimate. Throughput (M samples/s) per model per stage comes from measured runs; cells are scaled by sample counts."""
    cfg = load_config()
    rate = {"fast": 6.0e6, "hac": 1.28e6}                    # typical campaign-flag throughput
    slow = {"cb_on": 0.85, "upstream": 0.85, "pre_overlap": 0.9, "paper_xavier": 0.8}   # relative throughput of slower stages
    stages = args.stages.split(",") if args.stages else cfg["stage_order"]
    pw = cfg["power"]
    total = 0.0
    for s in stages:
        t = 0.0
        cells = resolve_cells(cfg, s, args.cells) if args.cells else default_cells(cfg, s)
        for cell in cells:
            c = parse_cell(cfg, cell)
            ds = c["ds"]
            samples = ds.get("samples") or 1e9
            run_s = samples / (rate[c["model"]] * slow.get(s, 1.0))
            overhead = 8 + pw["pre_shoulder_s"] + pw["post_shoulder_s"] + 25
            modes = ds["modes"]
            n = ds["reps"]
            t += (run_s + overhead) * (len(modes) * n + (1 if ds.get("warmup", True) else 0))
        t += 2 * (pw["idle_s"] + 40)
        print(f"{s:13s} {len(cells):2d} cells  ~{t / 3600:5.2f} h")
        total += t
    print(f"{'TOTAL':13s}            ~{total / 3600:5.2f} h  (plus builds, nsys pass and validation)")
    return 0


def main() -> int:
    global RUNS_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("idle")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--label", default="manual")
    p.set_defaults(fn=cmd_idle)
    p = sub.add_parser("canary")
    p.add_argument("--label", default="manual")
    p.set_defaults(fn=cmd_canary)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("plan")
    p.add_argument("--stages", default=None)
    p.add_argument("--cells", default=None)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("smoke", help="functional check (one run per stage/cell + accuracy sanity); not a measurement")
    p.add_argument("--stages", required=True, help="comma list of stages")
    p.add_argument("--cells", default=None, help="default: the 1k cells that exist for the stage")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-accuracy", action="store_true")
    p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("explore", help="grid over extra slorado flags for one stage/cell with an accuracy check (exploratory, single runs)")
    p.add_argument("--stage", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--grid", required=True, help='e.g. "spec-repair-threshold=10,15;spec-margin-threshold=2,3"')
    p.add_argument("--ref-stage", default="lb_best", help="non-speculative stage with the same decode, as the accuracy/time reference")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(fn=cmd_explore)
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True)
    p.add_argument("--cells", default=None, help="comma list, e.g. dna_fast_1k,rna_hac_20k,dna_fast_500k (default: the default datasets)")
    p.add_argument("--modes", default=None, help="A,B (default: per dataset)")
    p.add_argument("--reps", type=int, default=None, help="override repeats for every cell (default: per dataset)")
    p.add_argument("--warmup", choices=("per-cell", "per-mode", "none"), default="per-cell")
    p.add_argument("--smoke", action="store_true", help="skip idle/canary (quick tests)")
    p.add_argument("--stage-dir", default=None, help="use this directory's slorado instead of the stage worktree (testing only)")
    p.add_argument("--tegrastats", action="store_true", help="also run tegrastats 1 Hz alongside (validation V1)")
    p.add_argument("--resume", action="store_true", help="skip runs already recorded OK")
    p.add_argument("--runs-file", default=None, help="write records here instead of runs.jsonl")
    p.add_argument("--allow-invalid", action="store_true", help="run a stage that is marked invalid in stages.json (not for measurements)")
    p.set_defaults(fn=run_block)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
