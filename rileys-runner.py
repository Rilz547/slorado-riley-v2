#!/usr/bin/env python3
"""
Riley's all-in-one Jetson runner: variance timing + optional Nsight + console logs.

Parses:
  [main] Real time: X sec; CPU time: Y sec; Peak RAM: Z GB

---------------------------------------------------------------------------
FASTQ / timing rules
---------------------------------------------------------------------------
  Accuracy pass (NOT included in mean/stdev/min/max/CV%):
    A) NSIGHT == "yes":
         One nsys-wrapped run that writes a real FASTQ + .nsys-rep.
    B) Else if that config's NUM_RUNS == 1:
         One plain run that writes a real FASTQ (accuracy without nsys).

  Timed variance runs (ALWAYS -o /dev/null; these alone feed the stats):
    Exactly NUM_RUNS_* successful-or-failed slots after the accuracy pass.
    Warmup never uses nsys and never keeps a FASTQ.

---------------------------------------------------------------------------
Hang / resume
---------------------------------------------------------------------------
  - Live progress from [basecaller_main::...] lines.
  - Stall timeout (no stdout) AND per-config wall budget (timed vs nsys≈2×):
      kill → killall slorado/nsys → nvidia-smi snapshot → retry once → fail slot.
  - Global lock file so two runners cannot share the GPU.
  - Report + CSV + state flushed after every run.
  - Resume:  python3 rileys-runner.py --resume

---------------------------------------------------------------------------
Output naming (FILENAME_APPEND_FLAG)
---------------------------------------------------------------------------
  mode = "overlap" if EXTRA_ARGS contains overlap-decode=yes, else "base"

  FASTQ (when saved):
    output_{fast|hac}_{1k|20k}[_overlap]{APPEND}.fastq
  Nsight:
    nsys_{fast|hac}_{1k|20k}_{overlap|base}{APPEND}.nsys-rep
  Report / consoles / state / csv / lock:
    riley-runner-output{APPEND}.txt
    riley-runner-timings{APPEND}.csv
    riley-runner-{fast,hac}-console{APPEND}.txt
    riley-runner-state{APPEND}.json
    riley-runner.lock

Tip: prefer APPEND like "-v1.2" (mode already in the filename).
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import re
import select
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

# ===========================================================================
# Top-level knobs (edit these)
# ===========================================================================

# Appended to every output artifact this invocation writes (report, consoles,
# FASTQs, nsys, state). Examples: "" | "-v1.2" | "-baseline-v1.2"
FILENAME_APPEND_FLAG = "-spec-q10m2brave"

# Timed variance runs per config (0 = skip). Stats use ONLY these /dev/null runs.
# 1 + NSIGHT="no" => per config: FASTQ run (acts as the warm-up, excluded from stats), then ONE
# timed /dev/null run (the recorded time). Two runs per config, FASTQ kept for accuracy analysis.
NUM_RUNS_FAST_1K = 1
NUM_RUNS_HAC_1K = 1
NUM_RUNS_FAST_20K = 1
NUM_RUNS_HAC_20K = 1

# One uncounted warmup before accuracy + timed runs (per config).
# "no" here: the FASTQ accuracy run below already warms the GPU/caches before the timed run.
WARMUP_FAST_1K = "no"
WARMUP_HAC_1K = "no"
WARMUP_FAST_20K = "no"
WARMUP_HAC_20K = "no"

# Separate accuracy pass (nsys + FASTQ) before timed runs; excluded from stats.
NSIGHT = "no"  # "yes" | "no"

# Capture full console (decode/RNN dumps) into fast/hac console files.
RECORD_CONSOLE = "yes"  # "yes" | "no"

# Shared by every basecaller invocation.
EXTRA_ARGS = (
    "--overlap-decode=yes --fixed-c-batch=no "
    "--speculative-decode=yes --spec-repair-threshold=10 --spec-margin-threshold=2 "
    "--spec-overlap-repair=yes"
)
# EXTRA_ARGS = "--overlap-decode=yes --fixed-c-batch=no"   # non-speculative "best" baseline
# Per-model flags appended after EXTRA_ARGS (best load-imbalance flush threshold: FAST 64, HAC 128).
FAMILY_ARGS = {
    "fast": "--flush-threshold=64",
    "hac": "--flush-threshold=128",
}
# Batch/chunk flags only — do NOT put -o here; the runner chooses /dev/null vs real FASTQ.
BASE_ARGS = "-C 128 -c 12288 -K 4096 -p 150"

# Nsight (used only when NSIGHT=yes)
NSYS_TRACE = "cuda,nvtx,osrt"
NSYS_OSRT_THRESHOLD = "1000"  # nanoseconds

# Hang recovery: either limit trips → kill; retry once; then fail slot.
# Stall: no new stdout (HAC 20k logs every few minutes — 10 min silence ⇒ hung).
RUN_STALL_TIMEOUT_S = 600
RUN_MAX_ATTEMPTS = 2

# Expected max wall for a plain timed (/dev/null) attempt (seconds).
# Nsight accuracy ≈ basecall + report generation ≈ 2× these values.
EXPECTED_TIMED_WALL_S = {
    ("fast", "1k"): 60,  # observed ~10s; keep headroom
    ("hac", "1k"): 120,  # observed ~45s
    ("fast", "20k"): 350,
    ("hac", "20k"): 20 * 60,  # 1200s
}
WALL_MARGIN = 1.25  # safety factor on expected
NSYS_WALL_FACTOR = 2.0  # run time + ~equal nsys report gen

# Fallback if a config key is missing from the table.
RUN_WALL_TIMEOUT_S = int(EXPECTED_TIMED_WALL_S[("hac", "20k")] * WALL_MARGIN * NSYS_WALL_FACTOR)

# Back-compat alias (older docs / muscle memory).
RUN_TIMEOUT_S = RUN_STALL_TIMEOUT_S

SLORADO = "./slorado"
REPO_ROOT = Path(__file__).resolve().parent
LOCK_PATH = REPO_ROOT / "riley-runner.lock"
_LOCK_HELD = False

# ===========================================================================

YES = {"yes", "y", "true", "1", "on"}

PROGRESS_RE = re.compile(
    r"\[basecaller_main::([0-9.]+)\*([0-9.]+)\]\s+(\d+)\s+Entries\s+\(([^)]+)\)\s+(processed|loaded)",
    re.IGNORECASE,
)

TIMING_RE = re.compile(
    r"\[main\]\s+Real time:\s*([0-9.]+)\s*sec;\s*"
    r"CPU time:\s*([0-9.]+)\s*sec;\s*"
    r"Peak RAM:\s*([0-9.]+)\s*GB",
    re.IGNORECASE,
)


def is_yes(val: str) -> bool:
    return str(val).strip().lower() in YES


def overlap_mode() -> bool:
    return "overlap-decode=yes" in EXTRA_ARGS.replace(" ", "")


def mode_tag_fastq() -> str:
    return "_overlap" if overlap_mode() else ""


def mode_tag_nsys() -> str:
    return "overlap" if overlap_mode() else "base"


@dataclass
class Config:
    key: str
    label: str
    model_family: str  # "fast" | "hac"
    size: str  # "1k" | "20k"
    model: str
    data: str
    num_runs: int
    warmup: str


CONFIGS: list[Config] = [
    Config(
        "FAST_1K",
        "FAST 1k",
        "fast",
        "1k",
        "models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0",
        "test/PGXXXX230339/reads_1k.blow5",
        NUM_RUNS_FAST_1K,
        WARMUP_FAST_1K,
    ),
    Config(
        "HAC_1K",
        "HAC 1k",
        "hac",
        "1k",
        "models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
        "test/PGXXXX230339/reads_1k.blow5",
        NUM_RUNS_HAC_1K,
        WARMUP_HAC_1K,
    ),
    Config(
        "FAST_20K",
        "FAST 20k",
        "fast",
        "20k",
        "models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0",
        "test/PGXXXX230339/reads_20k.blow5",
        NUM_RUNS_FAST_20K,
        WARMUP_FAST_20K,
    ),
    Config(
        "HAC_20K",
        "HAC 20k",
        "hac",
        "20k",
        "models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
        "test/PGXXXX230339/reads_20k.blow5",
        NUM_RUNS_HAC_20K,
        WARMUP_HAC_20K,
    ),
]


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"


def cprint(msg: str, colour: str = "", *, file: TextIO = sys.stdout) -> None:
    if colour and file.isatty():
        print(f"{colour}{msg}{C.RESET}", file=file)
    else:
        print(msg, file=file)


@dataclass
class RunResult:
    real_s: float | None
    cpu_s: float | None
    peak_ram_gb: float | None
    saved_fastq: str | None = None
    nsys_rep: str | None = None
    used_nsight: bool = False
    status: str = "ok"  # ok | failed | timeout
    attempts: int = 1
    note: str = ""
    in_stats: bool = True  # False for accuracy / nsys pass

    def ok(self) -> bool:
        return self.status == "ok" and self.real_s is not None


def append_flag() -> str:
    return FILENAME_APPEND_FLAG or ""


def report_path() -> Path:
    return REPO_ROOT / f"riley-runner-output{append_flag()}.txt"


def console_path(family: str) -> Path:
    return REPO_ROOT / f"riley-runner-{family}-console{append_flag()}.txt"


def state_path() -> Path:
    return REPO_ROOT / f"riley-runner-state{append_flag()}.json"


def csv_path() -> Path:
    return REPO_ROOT / f"riley-runner-timings{append_flag()}.csv"


def fastq_name(cfg: Config) -> str:
    return f"output_{cfg.model_family}_{cfg.size}{mode_tag_fastq()}{append_flag()}.fastq"


def nsys_stem(cfg: Config) -> str:
    return f"nsys_{cfg.model_family}_{cfg.size}_{mode_tag_nsys()}{append_flag()}"


def needs_accuracy_pass(cfg: Config) -> bool:
    if is_yes(NSIGHT):
        return True
    return (not is_yes(NSIGHT)) and cfg.num_runs == 1


def wall_timeout_for(cfg: Config, *, nsys: bool = False) -> float:
    """Absolute wall-clock cap for one attempt of this config."""
    base = EXPECTED_TIMED_WALL_S.get(
        (cfg.model_family, cfg.size),
        EXPECTED_TIMED_WALL_S[("hac", "20k")],
    )
    t = float(base) * WALL_MARGIN
    if nsys:
        t *= NSYS_WALL_FACTOR
    return t


def stall_timeout_for(wall_s: float) -> float:
    """Stall cap: global default, but never longer than the wall budget."""
    return float(min(RUN_STALL_TIMEOUT_S, max(60.0, wall_s)))


def build_slorado_cmd(cfg: Config, out_path: str) -> list[str]:
    parts = [SLORADO, "basecaller"]
    parts.extend(shlex.split(BASE_ARGS))
    parts.extend(["-o", out_path])
    if EXTRA_ARGS.strip():
        parts.extend(shlex.split(EXTRA_ARGS))
    fam = FAMILY_ARGS.get(cfg.model_family, "")
    if fam.strip():
        parts.extend(shlex.split(fam))
    parts.extend([cfg.model, cfg.data])
    return parts


def build_nsys_cmd(cfg: Config, slorado_cmd: list[str]) -> list[str]:
    stem = nsys_stem(cfg)
    return [
        "nsys",
        "profile",
        "--force-overwrite=true",
        f"--trace={NSYS_TRACE}",
        "--sample=none",
        f"--osrt-threshold={NSYS_OSRT_THRESHOLD}",
        f"--output={stem}",
        *slorado_cmd,
    ]


def cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def parse_timing(text: str) -> tuple[float, float, float]:
    matches = TIMING_RE.findall(text)
    if not matches:
        raise RuntimeError("could not find '[main] Real time: ... Peak RAM: ...' in output")
    real_s, cpu_s, ram = matches[-1]
    return float(real_s), float(cpu_s), float(ram)


def sample_stats(xs: list[float]) -> dict[str, float]:
    n = len(xs)
    if n == 0:
        return {
            "n": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
            "cv_pct": 0.0,
        }
    mean = statistics.fmean(xs)
    amin = min(xs)
    amax = max(xs)
    if n >= 2:
        stdev = statistics.stdev(xs)
        var = statistics.variance(xs)
    else:
        stdev = 0.0
        var = 0.0
    cv_pct = (stdev / mean * 100.0) if mean else 0.0
    return {
        "n": float(n),
        "mean": mean,
        "stdev": stdev,
        "variance": var,
        "min": amin,
        "max": amax,
        "range": amax - amin,
        "cv_pct": cv_pct,
    }


def kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.time() + 10.0
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except Exception:
                pass


def cleanup_gpu_children() -> str:
    """Best-effort kill leftover slorado/nsys; return nvidia-smi snapshot text."""
    for name in ("slorado", "nsys", "nsys-ui"):
        subprocess.run(["killall", "-9", name], check=False, capture_output=True)
    snap = nvidia_smi_snapshot()
    note = (
        "\n[rileys-runner] cleanup: killall -9 slorado nsys nsys-ui\n"
        "[rileys-runner] If GPU util stays ~100% after this, reboot the Jetson "
        "(or try: sudo nvidia-smi --gpu-reset) before --resume.\n"
    )
    if snap:
        note += "[rileys-runner] nvidia-smi:\n" + snap + "\n"
    else:
        note += "[rileys-runner] nvidia-smi unavailable\n"
    cprint(note.rstrip(), C.YELLOW, file=sys.stderr)
    return note


def nvidia_smi_snapshot() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        body = (proc.stdout or "").strip()
        if body:
            return "  " + body.replace("\n", "\n  ")
        return (proc.stderr or "").strip()
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock() -> None:
    global _LOCK_HELD
    if LOCK_PATH.is_file():
        try:
            data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            other_pid = int(data.get("pid", -1))
            if pid_alive(other_pid) and other_pid != os.getpid():
                raise RuntimeError(
                    f"another rileys-runner is active "
                    f"(pid={other_pid}, append={data.get('append')!r}, started={data.get('started')}). "
                    f"If stale, delete {LOCK_PATH.name} after confirming that pid is dead."
                )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # treat unreadable lock as stale
    payload = {
        "pid": os.getpid(),
        "append": append_flag(),
        "started": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "extra_args": EXTRA_ARGS,
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _LOCK_HELD = True
    atexit.register(release_lock)


def release_lock() -> None:
    global _LOCK_HELD
    if not _LOCK_HELD:
        return
    try:
        if LOCK_PATH.is_file():
            data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(data.get("pid", -1)) == os.getpid():
                LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    _LOCK_HELD = False


# Active child for Ctrl+C cleanup
_ACTIVE_PROC: subprocess.Popen[str] | None = None


def _progress_line(
    label: str,
    wall_s: float,
    factor: float,
    n: int,
    nbytes: str,
    kind: str,
    *,
    wall_budget_s: float,
) -> str:
    bar_w = 24
    filled = min(bar_w, max(1, int((wall_s / max(wall_budget_s, 1)) * bar_w)))
    bar = "#" * filled + "-" * (bar_w - filled)
    return (
        f"  [{bar}] {label}  t={wall_s:.0f}s*{factor:.2f}  "
        f"{n} entries ({nbytes}) {kind}"
    )


def run_capture(
    cmd: list[str],
    *,
    progress_label: str = "",
    stall_timeout_s: float = RUN_STALL_TIMEOUT_S,
    wall_timeout_s: float = RUN_WALL_TIMEOUT_S,
) -> tuple[int, str, str]:
    """
    Stream child stdout. Returns (returncode, full_text, outcome)
    outcome: ok | timeout | crashed
    Kill if no stdout for stall_timeout_s OR elapsed wall_timeout_s.
    """
    global _ACTIVE_PROC
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    _ACTIVE_PROC = proc
    assert proc.stdout is not None
    chunks: list[str] = []
    t0 = time.time()
    last_output = t0
    last_progress_print = 0.0
    tty = sys.stdout.isatty()

    def _timeout(reason: str) -> tuple[int, str, str]:
        kill_process_tree(proc)
        extra = cleanup_gpu_children()
        if tty and last_progress_print:
            print(file=sys.stdout)
        text = "".join(chunks)
        text += f"\n[rileys-runner] TIMEOUT ({reason}); killed process group.\n"
        text += extra
        return -9, text, "timeout"

    try:
        while True:
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    chunks.append(rest)
                break

            now = time.time()
            if now - t0 >= wall_timeout_s:
                return _timeout(f"wall-clock {wall_timeout_s:.0f}s exceeded")
            if now - last_output >= stall_timeout_s:
                return _timeout(f"no new stdout for {stall_timeout_s:.0f}s")

            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if ready:
                line = proc.stdout.readline()
                if line:
                    chunks.append(line)
                    last_output = time.time()
                    m = PROGRESS_RE.search(line)
                    if m and tty:
                        wall_s = float(m.group(1))
                        factor = float(m.group(2))
                        n = int(m.group(3))
                        nbytes = m.group(4)
                        kind = m.group(5)
                        now = time.time()
                        if now - last_progress_print >= 0.5:
                            msg = _progress_line(
                                progress_label,
                                wall_s,
                                factor,
                                n,
                                nbytes,
                                kind,
                                wall_budget_s=wall_timeout_s,
                            )
                            print("\r" + msg + " " * 8, end="", flush=True)
                            last_progress_print = now

        if tty and last_progress_print:
            print(file=sys.stdout)
        rc = proc.returncode if proc.returncode is not None else -1
        text = "".join(chunks)
        if rc != 0:
            return rc, text, "crashed"
        return rc, text, "ok"
    finally:
        _ACTIVE_PROC = None
        if proc.poll() is None:
            kill_process_tree(proc)


def append_console(family: str, header: str, body: str) -> None:
    if not is_yes(RECORD_CONSOLE):
        return
    path = console_path(family)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 72 + "\n")
        f.write(header + "\n")
        f.write("=" * 72 + "\n")
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def result_from_dict(d: dict) -> RunResult:
    return RunResult(
        real_s=d.get("real_s"),
        cpu_s=d.get("cpu_s"),
        peak_ram_gb=d.get("peak_ram_gb"),
        saved_fastq=d.get("saved_fastq"),
        nsys_rep=d.get("nsys_rep"),
        used_nsight=bool(d.get("used_nsight", False)),
        status=d.get("status", "ok"),
        attempts=int(d.get("attempts", 1)),
        note=d.get("note", ""),
        in_stats=bool(d.get("in_stats", True)),
    )


def summarise_config(
    label: str,
    cmd_example: str,
    timed: list[RunResult],
    accuracy: RunResult | None,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"=== {label} ===")
    lines.append(f"command (typical timed /dev/null run): {cmd_example}")
    if accuracy is not None:
        if accuracy.saved_fastq:
            lines.append(f"saved FASTQ (accuracy pass, excluded from stats): {accuracy.saved_fastq}")
        if accuracy.nsys_rep:
            lines.append(f"nsys report (excluded from stats): {accuracy.nsys_rep}")
        if accuracy.ok():
            lines.append(
                f"accuracy pass: real={accuracy.real_s:.3f}s cpu={accuracy.cpu_s:.3f}s "
                f"peak_ram={accuracy.peak_ram_gb:.3f}GB  status={accuracy.status}"
                + (f"  note={accuracy.note}" if accuracy.note else "")
            )
        else:
            lines.append(
                f"accuracy pass: FAILED status={accuracy.status}"
                + (f"  note={accuracy.note}" if accuracy.note else "")
            )
    lines.append("")

    hdr = (
        f"{'run':>4}  {'real(s)':>10}  {'cpu(s)':>10}  {'peak_ram(GB)':>12}  "
        f"{'status':>8}  {'tries':>5}"
    )
    lines.append("Per-run values (timed /dev/null only — used for stats):")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(timed, 1):
        if r.ok():
            lines.append(
                f"{i:>4}  {r.real_s:>10.3f}  {r.cpu_s:>10.3f}  {r.peak_ram_gb:>12.3f}  "
                f"{r.status:>8}  {r.attempts:>5}"
            )
        else:
            lines.append(
                f"{i:>4}  {'—':>10}  {'—':>10}  {'—':>12}  "
                f"{r.status:>8}  {r.attempts:>5}"
                + (f"  {r.note}" if r.note else "")
            )
    lines.append("")

    ok_runs = [r for r in timed if r.ok()]
    real = sample_stats([r.real_s for r in ok_runs if r.real_s is not None])
    cpu = sample_stats([r.cpu_s for r in ok_runs if r.cpu_s is not None])
    ram = sample_stats([r.peak_ram_gb for r in ok_runs if r.peak_ram_gb is not None])

    shdr = f"{'metric':<10}  {'real(s)':>12}  {'cpu(s)':>12}  {'peak_ram(GB)':>12}"
    lines.append(
        f"Summary (ok timed runs only; n={len(ok_runs)}/{len(timed)}; "
        f"accuracy/nsys excluded; sample stdev when n>=2):"
    )
    lines.append(shdr)
    lines.append("-" * len(shdr))

    def cell(v: float, digits: int = 3) -> str:
        if real["n"] == 0:
            return f"{'—':>12}"
        return f"{v:.{digits}f}".rjust(12)

    def row(name: str, key: str, digits: int = 3) -> str:
        return f"{name:<10}  {cell(real[key], digits)}  {cell(cpu[key], digits)}  {cell(ram[key], digits)}"

    if real["n"] == 0:
        lines.append(f"{'n':<10}  {0:>12}  {0:>12}  {0:>12}")
        lines.append("(no successful timed runs — stats unavailable)")
    else:
        lines.append(f"{'n':<10}  {int(real['n']):>12}  {int(cpu['n']):>12}  {int(ram['n']):>12}")
        lines.append(row("mean", "mean"))
        lines.append(row("stdev", "stdev"))
        lines.append(row("variance", "variance", 6))
        lines.append(row("min", "min"))
        lines.append(row("max", "max"))
        lines.append(row("range", "range"))
        cv_real = f"{real['cv_pct']:.3f}%".rjust(12)
        cv_cpu = f"{cpu['cv_pct']:.3f}%".rjust(12)
        cv_ram = f"{ram['cv_pct']:.3f}%".rjust(12)
        lines.append(f"{'CV%':<10}  {cv_real}  {cv_cpu}  {cv_ram}")
    lines.append("")
    return lines


def comparison_table(all_timed: dict[str, list[RunResult]]) -> list[str]:
    lines: list[str] = []
    lines.append("=== Quick real-time comparison (timed /dev/null ok runs only) ===")
    hdr = (
        f"{'config':<12}  {'mean(s)':>10}  {'stdev(s)':>10}  {'min(s)':>10}  "
        f"{'max(s)':>10}  {'range(s)':>10}  {'CV%':>8}  {'ok':>6}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for label, runs in all_timed.items():
        ok = [r for r in runs if r.ok()]
        s = sample_stats([r.real_s for r in ok if r.real_s is not None])
        if s["n"] == 0:
            lines.append(
                f"{label:<12}  {'—':>10}  {'—':>10}  {'—':>10}  "
                f"{'—':>10}  {'—':>10}  {'—':>8}  {0:>3}/{len(runs)}"
            )
        else:
            lines.append(
                f"{label:<12}  {s['mean']:>10.3f}  {s['stdev']:>10.3f}  {s['min']:>10.3f}  "
                f"{s['max']:>10.3f}  {s['range']:>10.3f}  {s['cv_pct']:>7.3f}%  "
                f"{int(s['n']):>3}/{len(runs)}"
            )
    lines.append("")
    return lines


@dataclass
class ConfigProgress:
    warmup_done: bool = False
    accuracy_done: bool = False
    accuracy: dict | None = None
    timed: list[dict] = field(default_factory=list)
    complete: bool = False


@dataclass
class SuiteState:
    version: int = 2
    append: str = ""
    extra_args: str = ""
    base_args: str = ""
    nsight: str = ""
    started: str = ""
    configs: dict[str, dict] = field(default_factory=dict)

    def save(self) -> None:
        path = state_path()
        payload = {
            "version": self.version,
            "append": self.append,
            "extra_args": self.extra_args,
            "base_args": self.base_args,
            "nsight": self.nsight,
            "started": self.started,
            "configs": self.configs,
            "knobs": {
                "NUM_RUNS_FAST_1K": NUM_RUNS_FAST_1K,
                "NUM_RUNS_HAC_1K": NUM_RUNS_HAC_1K,
                "NUM_RUNS_FAST_20K": NUM_RUNS_FAST_20K,
                "NUM_RUNS_HAC_20K": NUM_RUNS_HAC_20K,
                "RUN_STALL_TIMEOUT_S": RUN_STALL_TIMEOUT_S,
                "WALL_MARGIN": WALL_MARGIN,
                "NSYS_WALL_FACTOR": NSYS_WALL_FACTOR,
                "EXPECTED_TIMED_WALL_S": {
                    f"{fam}_{size}": sec for (fam, size), sec in EXPECTED_TIMED_WALL_S.items()
                },
                "RUN_MAX_ATTEMPTS": RUN_MAX_ATTEMPTS,
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_state() -> SuiteState:
    path = state_path()
    if not path.is_file():
        raise FileNotFoundError(f"no state file to resume: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    st = SuiteState(
        version=int(data.get("version", 1)),
        append=data.get("append", ""),
        extra_args=data.get("extra_args", ""),
        base_args=data.get("base_args", ""),
        nsight=data.get("nsight", ""),
        started=data.get("started", ""),
        configs=data.get("configs", {}),
    )
    return st


def new_state(ts: str) -> SuiteState:
    st = SuiteState(
        append=append_flag(),
        extra_args=EXTRA_ARGS,
        base_args=BASE_ARGS,
        nsight=NSIGHT,
        started=ts,
        configs={
            c.key: asdict(ConfigProgress())
            for c in CONFIGS
            if c.num_runs > 0
        },
    )
    return st


def build_report_header(ts: str, out_file: Path, enabled: list[Config]) -> list[str]:
    report: list[str] = []
    report.append("riley's runner — variance + optional nsight + console capture")
    report.append(f"timestamp:              {ts}")
    report.append(f"repo:                   {REPO_ROOT}")
    report.append(f"FILENAME_APPEND_FLAG:   {FILENAME_APPEND_FLAG!r}")
    report.append(f"NSIGHT:                 {NSIGHT}")
    report.append(f"RECORD_CONSOLE:         {RECORD_CONSOLE}")
    report.append(f"EXTRA_ARGS:             {EXTRA_ARGS!r}")
    report.append(f"FAMILY_ARGS:            {FAMILY_ARGS!r}")
    report.append(f"BASE_ARGS:              {BASE_ARGS!r}")
    report.append(f"overlap_mode:           {overlap_mode()}")
    report.append(f"NSYS_TRACE:             {NSYS_TRACE!r}")
    report.append(f"NSYS_OSRT_THRESHOLD:    {NSYS_OSRT_THRESHOLD!r}")
    report.append(f"RUN_STALL_TIMEOUT_S:    {RUN_STALL_TIMEOUT_S}")
    report.append(f"WALL_MARGIN:            {WALL_MARGIN}")
    report.append(f"NSYS_WALL_FACTOR:       {NSYS_WALL_FACTOR}")
    report.append(f"RUN_MAX_ATTEMPTS:       {RUN_MAX_ATTEMPTS}")
    report.append("EXPECTED wall budgets (timed / nsys-accuracy):")
    for c in enabled:
        tw = wall_timeout_for(c, nsys=False)
        nw = wall_timeout_for(c, nsys=True)
        report.append(f"  {c.key}: timed<={tw:.0f}s  nsys-accuracy<={nw:.0f}s")
    report.append("")
    report.append("NUM_RUNS / WARMUP:")
    for c in CONFIGS:
        report.append(f"  {c.key}: runs={c.num_runs}  warmup={c.warmup}")
    report.append("")
    report.append("FASTQ / timing rules:")
    report.append("  - Accuracy pass (NSIGHT or single-run FASTQ) is EXCLUDED from mean/stdev/CV%.")
    report.append("  - Timed variance runs always use -o /dev/null (NUM_RUNS_* each).")
    report.append("  - Hang: stall OR per-config wall timeout → killall + nvidia-smi, retry once, then fail.")
    report.append("  - Lock: riley-runner.lock prevents two suites sharing the GPU.")
    report.append("  - Resume: python3 rileys-runner.py --resume")
    report.append("")
    report.append("=== Artifact names (this invocation) ===")
    report.append(f"  report:         {out_file.name}")
    report.append(f"  timings csv:    {csv_path().name}")
    report.append(f"  state:          {state_path().name}")
    report.append(f"  lock:           {LOCK_PATH.name}")
    report.append(f"  console fast:   {console_path('fast').name}")
    report.append(f"  console hac:    {console_path('hac').name}")
    for c in enabled:
        report.append(f"  [{c.label}] FASTQ (if saved): {fastq_name(c)}")
        report.append(f"  [{c.label}] nsys stem:        {nsys_stem(c)}.nsys-rep")
    report.append("")
    report.append("=== Commands used ===")
    for c in enabled:
        report.append(f"  [{c.label}] timed/devnull: {cmd_str(build_slorado_cmd(c, '/dev/null'))}")
        if is_yes(NSIGHT):
            fq = fastq_name(c)
            report.append(
                f"  [{c.label}] accuracy+nsys: {cmd_str(build_nsys_cmd(c, build_slorado_cmd(c, fq)))}"
            )
        elif c.num_runs == 1:
            report.append(
                f"  [{c.label}] accuracy+fastq: {cmd_str(build_slorado_cmd(c, fastq_name(c)))}"
            )
    report.append("")
    report.append("How to read this file:")
    report.append("  - Per-config tables: timed runs + summary (accuracy/nsys listed but not in stats).")
    report.append("  - CV% = stdev/mean*100 on real time — headline for run-to-run jitter.")
    report.append("  - File is rewritten after every run so overnight hangs still leave partial results.")
    report.append("")
    return report


def write_timings_csv(state: SuiteState, enabled: list[Config]) -> None:
    """config,phase,run,real_s,cpu_s,peak_ram_gb,status,attempts,nsight,fastq,note"""
    path = csv_path()
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "phase",
                "run",
                "real_s",
                "cpu_s",
                "peak_ram_gb",
                "status",
                "attempts",
                "nsight",
                "fastq",
                "note",
            ]
        )
        for cfg in enabled:
            cp = state.configs.get(cfg.key, {})
            if cp.get("accuracy"):
                r = result_from_dict(cp["accuracy"])
                w.writerow(
                    [
                        cfg.label,
                        "accuracy",
                        "",
                        "" if r.real_s is None else f"{r.real_s:.6f}",
                        "" if r.cpu_s is None else f"{r.cpu_s:.6f}",
                        "" if r.peak_ram_gb is None else f"{r.peak_ram_gb:.6f}",
                        r.status,
                        r.attempts,
                        "yes" if r.used_nsight else "no",
                        r.saved_fastq or "",
                        r.note,
                    ]
                )
            for i, d in enumerate(cp.get("timed", []), 1):
                r = result_from_dict(d)
                w.writerow(
                    [
                        cfg.label,
                        "timed",
                        i,
                        "" if r.real_s is None else f"{r.real_s:.6f}",
                        "" if r.cpu_s is None else f"{r.cpu_s:.6f}",
                        "" if r.peak_ram_gb is None else f"{r.peak_ram_gb:.6f}",
                        r.status,
                        r.attempts,
                        "yes" if r.used_nsight else "no",
                        r.saved_fastq or "",
                        r.note,
                    ]
                )


def flush_outputs(
    header: list[str],
    state: SuiteState,
    enabled: list[Config],
) -> None:
    """Rebuild full report from state and write report + CSV + state JSON."""
    body: list[str] = list(header)
    all_timed: dict[str, list[RunResult]] = {}

    for cfg in enabled:
        cp = state.configs.get(cfg.key, {})
        accuracy = result_from_dict(cp["accuracy"]) if cp.get("accuracy") else None
        timed = [result_from_dict(d) for d in cp.get("timed", [])]
        if timed or accuracy is not None or cp.get("warmup_done"):
            example = cmd_str(build_slorado_cmd(cfg, "/dev/null"))
            body.extend(summarise_config(cfg.label, example, timed, accuracy))
        if timed:
            all_timed[cfg.label] = timed

    if all_timed:
        body.extend(comparison_table(all_timed))

    body.append("Notes:")
    body.append("  - Stats use timed /dev/null runs only (ok status); accuracy/nsys excluded.")
    body.append("  - stdev/variance are sample statistics (n-1) when n>=2.")
    body.append("  - CV% = stdev/mean*100; lower = tighter run-to-run timing.")
    body.append(f"  - Timings CSV: {csv_path().name}")
    body.append(
        f"  - Console logs: RECORD_CONSOLE={RECORD_CONSOLE} → "
        f"riley-runner-{{fast,hac}}-console{append_flag()}.txt"
    )
    body.append(f"  - State file: {state_path().name}  (python3 rileys-runner.py --resume)")
    body.append(f"  - Change EXTRA_ARGS / FILENAME_APPEND_FLAG to label A/B experiments.")
    body.append("")

    report_path().write_text("\n".join(body) + "\n", encoding="utf-8")
    write_timings_csv(state, enabled)
    state.save()


def prepare_nsys(cfg: Config) -> None:
    subprocess.run(["killall", "nsys-ui", "nsys"], check=False, capture_output=True)
    stem = nsys_stem(cfg)
    for p in REPO_ROOT.glob(f"{stem}.*"):
        try:
            p.unlink()
        except OSError:
            pass


def run_with_retries(
    cmd: list[str],
    *,
    label: str,
    family: str,
    header_prefix: str,
    ts: str,
    progress_label: str,
    wall_timeout_s: float,
) -> RunResult:
    last_text = ""
    last_outcome = "crashed"
    last_rc = -1
    stall_s = stall_timeout_for(wall_timeout_s)
    for attempt in range(1, RUN_MAX_ATTEMPTS + 1):
        if attempt > 1:
            cprint(f"  retry {attempt}/{RUN_MAX_ATTEMPTS} after {last_outcome}...", C.YELLOW)
        cprint(
            f"  $ {cmd_str(cmd)}  (wall<={wall_timeout_s:.0f}s stall<={stall_s:.0f}s)",
            C.DIM,
        )
        rc, text, outcome = run_capture(
            cmd,
            progress_label=progress_label,
            stall_timeout_s=stall_s,
            wall_timeout_s=wall_timeout_s,
        )
        last_text, last_outcome, last_rc = text, outcome, rc
        append_console(
            family,
            f"{header_prefix} | attempt {attempt}/{RUN_MAX_ATTEMPTS} | {ts}\ncmd: {cmd_str(cmd)}",
            text,
        )
        if outcome == "ok" and rc == 0:
            try:
                real_s, cpu_s, ram = parse_timing(text)
                return RunResult(
                    real_s=real_s,
                    cpu_s=cpu_s,
                    peak_ram_gb=ram,
                    status="ok",
                    attempts=attempt,
                    in_stats=True,
                )
            except Exception as e:
                last_outcome = "crashed"
                last_text = text + f"\n[rileys-runner] parse error: {e}\n"
                cprint(f"  parse FAILED: {e}", C.RED, file=sys.stderr)
                cleanup_gpu_children()
        else:
            cprint(f"  attempt {attempt} {outcome} (exit {rc})", C.RED, file=sys.stderr)
            if outcome != "timeout":
                # timeout path already ran cleanup_gpu_children inside run_capture
                cleanup_gpu_children()

    status = "timeout" if last_outcome == "timeout" else "failed"
    return RunResult(
        real_s=None,
        cpu_s=None,
        peak_ram_gb=None,
        status=status,
        attempts=RUN_MAX_ATTEMPTS,
        note=f"exit={last_rc} outcome={last_outcome}",
        in_stats=True,
    )


def run_suite(*, resume: bool) -> int:
    global FILENAME_APPEND_FLAG  # noqa: PLW0603 — resume may keep saved append

    enabled = [c for c in CONFIGS if c.num_runs > 0]
    if not enabled:
        cprint("All NUM_RUNS_* are 0 — nothing to do.", C.RED, file=sys.stderr)
        return 2

    if is_yes(NSIGHT) and shutil.which("nsys") is None:
        cprint("NSIGHT=yes but `nsys` not found on PATH", C.RED, file=sys.stderr)
        return 2

    acquire_lock()
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    if resume:
        st = load_state()
        if st.append != append_flag():
            cprint(
                f"WARNING: state append={st.append!r} but FILENAME_APPEND_FLAG={append_flag()!r}. "
                f"Using state append for paths.",
                C.YELLOW,
                file=sys.stderr,
            )
            FILENAME_APPEND_FLAG = st.append
        if st.extra_args != EXTRA_ARGS or st.base_args != BASE_ARGS or st.nsight != NSIGHT:
            cprint(
                "WARNING: knobs in file differ from state "
                f"(EXTRA_ARGS/BASE_ARGS/NSIGHT). Resuming with *current* knobs; "
                f"state was started={st.started}",
                C.YELLOW,
                file=sys.stderr,
            )
        cprint(f"Resuming from {state_path().name} (started {st.started})", C.CYAN)
        # Do not wipe consoles on resume
    else:
        if is_yes(RECORD_CONSOLE):
            for fam in ("fast", "hac"):
                console_path(fam).write_text("", encoding="utf-8")
        st = new_state(ts)
        st.save()

    out_file = report_path()
    header = build_report_header(st.started or ts, out_file, enabled)
    flush_outputs(header, st, enabled)

    cprint(f"{C.BOLD}Riley's runner{C.RESET}  ({ts})", C.CYAN)
    cprint(
        f"APPEND={FILENAME_APPEND_FLAG!r}  NSIGHT={NSIGHT}  RECORD_CONSOLE={RECORD_CONSOLE}  "
        f"EXTRA_ARGS={EXTRA_ARGS!r}  STALL={RUN_STALL_TIMEOUT_S}s",
        C.CYAN,
    )
    for c in enabled:
        cprint(
            f"  wall[{c.key}]: timed<={wall_timeout_for(c):.0f}s  "
            f"nsys<={wall_timeout_for(c, nsys=True):.0f}s",
            C.DIM,
        )
    cprint(f"Output → {out_file}  csv → {csv_path()}  state → {state_path()}", C.DIM)

    for cfg in enabled:
        cp = st.configs.setdefault(cfg.key, asdict(ConfigProgress()))
        if cp.get("complete"):
            cprint(f"\n{C.BOLD}>> {cfg.label}{C.RESET}  (already complete — skip)", C.DIM)
            continue

        cprint(
            f"\n{C.BOLD}>> {cfg.label}{C.RESET}  (runs={cfg.num_runs}, warmup={cfg.warmup})",
            C.MAGENTA,
        )

        # --- warmup ---
        if is_yes(cfg.warmup) and not cp.get("warmup_done"):
            cprint("  warmup (uncounted, /dev/null)...", C.YELLOW)
            wcmd = build_slorado_cmd(cfg, "/dev/null")
            wr = run_with_retries(
                wcmd,
                label=cfg.label,
                family=cfg.model_family,
                header_prefix=f"{cfg.label} | WARMUP",
                ts=ts,
                progress_label=f"{cfg.label} warmup",
                wall_timeout_s=wall_timeout_for(cfg, nsys=False),
            )
            if not wr.ok():
                cprint(f"  warmup FAILED ({wr.status}) — skipping config", C.RED, file=sys.stderr)
                cp["warmup_done"] = True
                cp["complete"] = True
                cp["timed"] = [
                    asdict(
                        RunResult(
                            None,
                            None,
                            None,
                            status="failed",
                            note="warmup failed; config skipped",
                            in_stats=True,
                        )
                    )
                ]
                st.configs[cfg.key] = cp
                flush_outputs(header, st, enabled)
                continue
            cprint(
                f"  warmup ok: real={wr.real_s:.3f}s cpu={wr.cpu_s:.3f}s ram={wr.peak_ram_gb:.3f}GB",
                C.DIM,
            )
            cp["warmup_done"] = True
            st.configs[cfg.key] = cp
            flush_outputs(header, st, enabled)
        elif is_yes(cfg.warmup):
            cprint("  warmup already done", C.DIM)

        # --- accuracy / nsys (excluded from stats) ---
        if needs_accuracy_pass(cfg) and not cp.get("accuracy_done"):
            use_nsight = is_yes(NSIGHT)
            out = fastq_name(cfg)
            slorado_cmd = build_slorado_cmd(cfg, out)
            cmd = build_nsys_cmd(cfg, slorado_cmd) if use_nsight else slorado_cmd
            tag = "accuracy+nsys" if use_nsight else "accuracy+fastq"
            cprint(f"  {tag} (excluded from variance stats)...", C.GREEN)
            if use_nsight:
                prepare_nsys(cfg)
            ar = run_with_retries(
                cmd,
                label=cfg.label,
                family=cfg.model_family,
                header_prefix=f"{cfg.label} | {tag}",
                ts=ts,
                progress_label=f"{cfg.label} accuracy",
                wall_timeout_s=wall_timeout_for(cfg, nsys=use_nsight),
            )
            ar.in_stats = False
            ar.used_nsight = use_nsight
            if ar.ok():
                ar.saved_fastq = out
                if use_nsight:
                    ar.nsys_rep = f"{nsys_stem(cfg)}.nsys-rep"
                cprint(
                    f"  accuracy ok: real={ar.real_s:.3f}s  FASTQ={ar.saved_fastq}"
                    + (f"  NSYS={ar.nsys_rep}" if ar.nsys_rep else "")
                    + "  (not in stats)",
                    C.GREEN,
                )
            else:
                cprint(
                    f"  accuracy FAILED ({ar.status}) — continuing timed runs anyway",
                    C.YELLOW,
                    file=sys.stderr,
                )
            cp["accuracy"] = asdict(ar)
            cp["accuracy_done"] = True
            st.configs[cfg.key] = cp
            flush_outputs(header, st, enabled)
        elif needs_accuracy_pass(cfg):
            cprint("  accuracy pass already done", C.DIM)

        # --- timed variance runs ---
        timed_dicts: list[dict] = list(cp.get("timed", []))
        while len(timed_dicts) < cfg.num_runs:
            i = len(timed_dicts) + 1
            cmd = build_slorado_cmd(cfg, "/dev/null")
            cprint(f"  timed run {i}/{cfg.num_runs} (/dev/null)...", C.GREEN)
            r = run_with_retries(
                cmd,
                label=cfg.label,
                family=cfg.model_family,
                header_prefix=f"{cfg.label} | TIMED {i}/{cfg.num_runs}",
                ts=ts,
                progress_label=f"{cfg.label} run {i}/{cfg.num_runs}",
                wall_timeout_s=wall_timeout_for(cfg, nsys=False),
            )
            r.in_stats = True
            if r.ok():
                cprint(
                    f"  run {i}: real={r.real_s:.3f}s  cpu={r.cpu_s:.3f}s  "
                    f"peak_ram={r.peak_ram_gb:.3f} GB  (attempts={r.attempts})",
                    C.GREEN,
                )
            else:
                cprint(
                    f"  run {i}: MARKED {r.status.upper()} after {r.attempts} attempt(s) — continuing",
                    C.YELLOW,
                    file=sys.stderr,
                )
            timed_dicts.append(asdict(r))
            cp["timed"] = timed_dicts
            st.configs[cfg.key] = cp
            flush_outputs(header, st, enabled)

        cp["complete"] = True
        st.configs[cfg.key] = cp
        flush_outputs(header, st, enabled)

        # Print summary block for this config
        accuracy = result_from_dict(cp["accuracy"]) if cp.get("accuracy") else None
        timed = [result_from_dict(d) for d in cp.get("timed", [])]
        block = summarise_config(cfg.label, cmd_str(build_slorado_cmd(cfg, "/dev/null")), timed, accuracy)
        for line in block:
            cprint(line, C.BLUE if line.startswith("===") else "")

    # Final comparison
    all_timed = {
        c.label: [result_from_dict(d) for d in st.configs.get(c.key, {}).get("timed", [])]
        for c in enabled
        if st.configs.get(c.key, {}).get("timed")
    }
    for line in comparison_table(all_timed):
        cprint(line, C.CYAN if line.startswith("===") else "")

    flush_outputs(header, st, enabled)
    cprint(f"\nWrote {out_file}", C.CYAN)
    cprint(f"Wrote {csv_path()}", C.CYAN)
    cprint(f"Wrote {state_path()}", C.DIM)
    if is_yes(RECORD_CONSOLE):
        cprint(f"Wrote {console_path('fast')}", C.DIM)
        cprint(f"Wrote {console_path('hac')}", C.DIM)

    any_fail = any(
        (not result_from_dict(d).ok())
        for c in enabled
        for d in st.configs.get(c.key, {}).get("timed", [])
    )
    release_lock()
    return 1 if any_fail else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Riley's variance / nsight / resume runner")
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume from riley-runner-state{{APPEND}}.json (APPEND={FILENAME_APPEND_FLAG!r})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_suite(resume=args.resume)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cprint("\nInterrupted — killing active child and flushing state...", C.YELLOW, file=sys.stderr)
        if _ACTIVE_PROC is not None:
            kill_process_tree(_ACTIVE_PROC)
        cleanup_gpu_children()
        release_lock()
        cprint(
            f"Resume with: python3 rileys-runner.py --resume\n"
            f"(state file uses FILENAME_APPEND_FLAG={FILENAME_APPEND_FLAG!r})",
            C.YELLOW,
            file=sys.stderr,
        )
        raise SystemExit(130)
    except FileNotFoundError as e:
        release_lock()
        cprint(str(e), C.RED, file=sys.stderr)
        raise SystemExit(2)
    except RuntimeError as e:
        # lock contention etc.
        cprint(str(e), C.RED, file=sys.stderr)
        raise SystemExit(2)
