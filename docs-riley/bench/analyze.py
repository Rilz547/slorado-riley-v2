#!/usr/bin/env python3
"""Analysis for the unified harness (stdlib only).

    analyze.py csv      [--out runs.csv]                 tidy per-run CSV (adds net energy, per-read and per-Mbase energy)
    analyze.py tables   [--baseline STAGE] [--md FILE]   markdown tables with the conventions of measurement-methodology.md
    analyze.py validate                                  harness validation checks V1-V6 with explicit pass criteria
    analyze.py legacy                                    re-derive the August power-run energies from their saved samples.csv

Conventions (see measurement-methodology.md):
    time change d = (t_new - t_base) / t_base       (negative = faster)
    speed-up    S = t_base / t_new
    "real" difference: |delta| > 2 x pooled sd AND > 1 %
Net energy = gross - P_idle x window, with P_idle the mean of the idle baselines recorded immediately before and
after that stage block (or the median of all idle baselines if the stage has none).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import bench
import powerlog

REPO = bench.REPO
RAILS = ("in", "cg", "soc", "resid")
CELLS: tuple[str, ...] = ()   # filled from the data by set_cells()


def norm_cell(c: str) -> str:
    """Old two-part ids ('fast_1k') are DNA cells."""
    return c if len(c.split("_")) == 3 else f"dna_{c}"


def _cell_sort_key(c: str) -> tuple:
    dtype, model, size = c.split("_")
    mult = {"k": 1e3}.get(size[-1], 1.0)
    num = float("".join(ch for ch in size if ch.isdigit()) or 0) * mult
    return (dtype != "dna", num, model != "fast")


def set_cells(rows: list[dict]) -> None:
    global CELLS
    CELLS = tuple(sorted({r["cell"] for r in rows}, key=_cell_sort_key))


def mean_sd(v: list[float]) -> tuple[float, float]:
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fmt_ms(v: list[float], digits: int = 2) -> str:
    if not v:
        return "—"
    m, s = mean_sd(v)
    return f"{m:.{digits}f} ± {s:.{digits}f} ({len(v)})"


# ---------------------------------------------------------------- data

def idle_power(records: list[dict]) -> tuple[dict[str, dict], dict]:
    """Per-stage idle power per rail (W) from before/after idle records; plus a global fallback."""
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "idle" and r.get("status") == "ok":
            by_stage[r["stage"]].append(r)
    out = {}
    for stage, rs in by_stage.items():
        out[stage] = {k: st.mean(r[f"Pmean_{k}_W"] for r in rs) for k in RAILS}
    allr = [r for rs in by_stage.values() for r in rs]
    glob = {k: st.median(r[f"Pmean_{k}_W"] for r in allr) for k in RAILS} if allr else {}
    return out, glob


SPEC_LINE_RE = re.compile(r"speculative: drafted (\d+), repaired (\d+).*?draft ([\d.]+)s repair ([\d.]+)s")


def spec_columns(rec: dict) -> dict:
    """Speculative-decode counters; draft/repair seconds are read back from the saved console log."""
    out = {"spec_drafted": rec.get("spec_drafted"), "spec_repaired": rec.get("spec_repaired"), "spec_alpha": None,
           "spec_draft_s": None, "spec_repair_s": None}
    if out["spec_drafted"]:
        out["spec_alpha"] = 1 - out["spec_repaired"] / out["spec_drafted"]
        log = bench.REPO / rec["console"] if rec.get("console") else None
        if log and log.exists():
            m = SPEC_LINE_RE.search(log.read_text())
            if m:
                out["spec_draft_s"], out["spec_repair_s"] = float(m.group(3)), float(m.group(4))
    return out


EXCLUDED_FILE = bench.OUT_ROOT / "excluded-runs.json"
OUTLIER_PCT = 3.0  # rule stated in HARNESS-FROZEN.md: a run more than 3 % above its cell median is flagged


def load_excluded() -> dict[str, str]:
    """run_id -> reason, from bench-runs/excluded-runs.json (written by `analyze.py outliers --write`; edit or delete to undo)."""
    try:
        return json.loads(EXCLUDED_FILE.read_text())
    except (OSError, ValueError):
        return {}


def find_outliers(records: list[dict]) -> list[tuple[dict, float]]:
    """(record, % above the cell median) for timed OK runs more than OUTLIER_PCT above the median of their (stage, cell, mode) group (n >= 3)."""
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "timed" and r.get("status") == "ok":
            g[(r["stage"], r["cell"], r["mode"])].append(r)
    out = []
    for v in g.values():
        if len(v) < 3:
            continue
        med = st.median(r["real_s"] for r in v)
        out += [(r, 100 * (r["real_s"] / med - 1)) for r in v if r["real_s"] > med * (1 + OUTLIER_PCT / 100)]
    return out


def outliers(args: argparse.Namespace) -> int:
    recs = bench.load_records()
    found = find_outliers(recs)
    print(f"runs more than {OUTLIER_PCT:.0f} % above their (stage, cell, mode) median (timed, ok, n >= 3):")
    new = {}
    for r, pct in sorted(found, key=lambda x: (x[0]["stage"], x[0]["cell"], x[0]["mode"], x[0]["rep"])):
        print(f"  {r['stage']:11} {r['cell']:14} mode {r['mode']} rep {r['rep']}: {r['real_s']:.2f} s ({pct:+.1f} %)  {r['run_id']}")
        new[r["run_id"]] = f"{pct:+.1f} % above the cell median (rule: > {OUTLIER_PCT:.0f} %); {r['stage']} {r['cell']} mode {r['mode']} rep {r['rep']}"
    if args.write:
        merged = {**load_excluded(), **new}
        EXCLUDED_FILE.write_text(json.dumps(merged, indent=1) + "\n")
        print(f"wrote {EXCLUDED_FILE} ({len(merged)} run(s) excluded from csv/tables; validate and equiv still use every run)")
    return 0


def tidy(records: list[dict]) -> list[dict]:
    per_stage, glob = idle_power(records)
    # reads/bases per (stage, cell) from mode B runs, to normalise mode A rows
    sizes: dict[tuple, tuple[float, float]] = {}
    for r in records:
        if r.get("kind") == "timed" and r.get("status") == "ok" and r.get("mode") == "B" and r.get("reads_out"):
            sizes[(r["stage"], r["cell"])] = (r["reads_out"], r["bases_out"])
    rows = []
    excluded = load_excluded()
    for r in records:
        if r.get("kind") != "timed" or r.get("status") != "ok" or r.get("run_id") in excluded:
            continue
        cell = norm_cell(r["cell"])
        row = {
            "stage": r["stage"], "commit": r["commit"], "dtype": r.get("dtype", "dna"), "cell": cell, "config": r["config"], "size": r["size"],
            "dataset": r.get("dataset"), "mode": r["mode"], "rep": r["rep"],
            "t_real_s": r["real_s"], "wall_process_s": r.get("wall_process_s"), "samples": r.get("samples"),
            "samples_per_s": r.get("samples_per_s"), "cpu_s": r.get("cpu_s"), "peak_ram_gb": r.get("peak_ram_gb"),
            "t_infer_s": r.get("timings", {}).get("inference"), "t_decode_s": r.get("timings", {}).get("decode"),
            "window_start_epoch": r["window_start_epoch"], "window_s": r["real_s"],
        }
        pidle = per_stage.get(r["stage"]) or glob
        row["idle_source"] = "stage" if r["stage"] in per_stage else ("global" if glob else "none")
        for k in RAILS:
            row[f"E_{k}_J"] = r.get(f"E_{k}_J")
            row[f"Pmean_{k}_W"] = r.get(f"Pmean_{k}_W")
            row[f"P_idle_{k}_W"] = pidle.get(k) if pidle else None
            e = r.get(f"E_{k}_J")
            row[f"E_net_{k}_J"] = (e - pidle[k] * r["real_s"]) if (pidle and e is not None) else None
        for k in ("in", "cg", "soc"):
            row[f"Ppeak_{k}_W"] = r.get(f"Ppeak_{k}_W")
        reads, bases = sizes.get((r["stage"], r["cell"]), (r.get("reads_out"), r.get("bases_out")))
        row["reads"], row["bases"] = reads, bases
        row["J_per_1k_reads"] = r["E_in_J"] / reads * 1000 if (reads and r.get("E_in_J")) else None
        row["J_per_Mbase"] = r["E_in_J"] / (bases / 1e6) if (bases and r.get("E_in_J")) else None
        for k in ("gpu_mhz_mean", "gpu_load_mean_permille", "cpu_util_mean_pct", "ram_used_mb_max", "swap_used_mb_max", "tj_max_C"):
            row[k] = r.get(k)
        row["sync_s"] = r.get("sync_s")
        row.update(spec_columns(r))
        row["flags"] = r["cmd"].split(" basecaller ", 1)[-1].split(" -o ")[0]
        rows.append(row)
    return rows


def groups(rows: list[dict], key: str) -> dict[tuple, list[float]]:
    g: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None:
            g[(r["stage"], r["cell"], r["mode"])].append(v)
    return g


def compare(base: list[float], new: list[float]) -> dict:
    mb, sb = mean_sd(base)
    mn, sn = mean_sd(new)
    d = (mn - mb) / mb * 100
    pooled = math.sqrt((sb ** 2 + sn ** 2) / 2)
    real = abs(mn - mb) > 2 * pooled and abs(d) > 1.0 if (len(base) > 1 and len(new) > 1) else abs(d) > 1.0
    return {"d": d, "S": mb / mn if mn else float("nan"), "real": real, "single": len(base) < 2 or len(new) < 2}


def cmp_cell(c: dict) -> str:
    flag = "" if c["real"] else " ~"
    sing = " †" if c["single"] else ""
    return f"{c['d']:+.1f} % ({c['S']:.2f}×){flag}{sing}"


# ---------------------------------------------------------------- tables

def tables(args: argparse.Namespace) -> int:
    cfg = bench.load_config()
    records = bench.load_records()
    rows = tidy(records)
    if not rows:
        print("no timed records")
        return 1
    set_cells(rows)
    order = [s for s in cfg["stage_order"] if any(r["stage"] == s for r in rows)]
    out: list[str] = []

    def table(title: str, header: list[str], body: list[list[str]]) -> None:
        out.append(f"\n### {title}\n")
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for b in body:
            out.append("| " + " | ".join(b) + " |")

    for mode, name in (("A", "H1 raw speed (`-o /dev/null`)"), ("B", "H2 end-to-end speed (FASTQ written)")):
        g = groups(rows, "t_real_s")
        body = [[s] + [fmt_ms(g.get((s, c, mode), []), 3) for c in CELLS] for s in order]
        table(f"{name}: `Real time` s, mean ± sd (n)", ["stage", *CELLS], body)
        gw = groups(rows, "wall_process_s")
        body = [[s] + [fmt_ms(gw.get((s, c, mode), []), 3) for c in CELLS] for s in order]
        table(f"{name}: whole-process wall time s (like GNU `time` elapsed; includes start-up/shutdown), mean ± sd (n)", ["stage", *CELLS], body)
        gs = groups(rows, "samples_per_s")
        body = [[s] + [fmt_ms([x / 1e6 for x in gs.get((s, c, mode), [])], 2) for c in CELLS] for s in order]
        table(f"{name}: throughput, M samples/s (samples / `Real time`), mean ± sd (n)", ["stage", *CELLS], body)
        base = args.baseline or order[0]
        body = []
        for s in order:
            if s == base:
                continue
            body.append([s] + [cmp_cell(compare(g.get((base, c, mode), []), g.get((s, c, mode), []))) if g.get((base, c, mode)) and g.get((s, c, mode)) else "—" for c in CELLS])
        table(f"{name}: time change vs baseline `{base}` (speed-up); `~` = within noise, `†` = n < 2", ["stage", *CELLS], body)
        body = []
        for prev, s in zip(order, order[1:]):
            body.append([f"{prev} → {s}"] + [cmp_cell(compare(g[(prev, c, mode)], g[(s, c, mode)])) if g.get((prev, c, mode)) and g.get((s, c, mode)) else "—" for c in CELLS])
        table(f"{name}: incremental change vs previous stage in `stage_order`", ["step", *CELLS], body)

    # H2 vs H1
    g = groups(rows, "t_real_s")
    body = []
    for s in order:
        body.append([s] + [(lambda a, b: f"{(st.mean(b) - st.mean(a)) / st.mean(a) * 100:+.2f} %" if a and b else "—")(g.get((s, c, "A"), []), g.get((s, c, "B"), [])) for c in CELLS])
    table("Output-mode cost: mode B vs mode A time change", ["stage", *CELLS], body)

    for key, title, d in (("E_in_J", "Gross energy `VDD_IN` (J)", 1), ("E_net_in_J", "Net (idle-subtracted) energy `VDD_IN` (J)", 1),
                          ("Pmean_in_W", "Mean power `VDD_IN` (W)", 2), ("J_per_1k_reads", "Energy per 1000 reads (J)", 1)):
        g = groups(rows, key)
        body = [[s] + [fmt_ms(g.get((s, c, "A"), []), d) for c in CELLS] for s in order]
        table(f"{title}, mode A, mean ± sd (n)", ["stage", *CELLS], body)

    for cell in CELLS:
        body = []
        for s in order:
            line = [s]
            for k in ("cg", "soc", "resid", "in"):
                v = [r[f"E_{k}_J"] for r in rows if r["stage"] == s and r["cell"] == cell and r["mode"] == "A" and r.get(f"E_{k}_J") is not None]
                line.append(fmt_ms(v, 1))
            body.append(line)
        table(f"Energy by rail (J), mode A, {cell}", ["stage", "CPU+GPU+CV", "SoC", "residual", "total"], body)

    if args.accuracy:
        acc = {}
        for p in args.accuracy:
            for r in csv.DictReader(open(p)):
                m = re.fullmatch(r"output_(fast|hac)_(\d+k)-(dna|rna)-(.+)\.fastq", Path(r["fastq"]).name)
                if m:
                    key = [k for k, d in cfg["datasets"].items() if d["dtype"] == m.group(3) and d["size"] == m.group(2)]
                    if key:
                        acc[(m.group(4), f"{m.group(3)}_{m.group(1)}_{key[0].split('_', 1)[1]}")] = float(r["accuracy_median"])
        base = args.baseline or order[0]
        body = []
        for s in order:
            line = [s]
            for c in CELLS:
                v = acc.get((s, c))
                b = acc.get((base, c))
                line.append("—" if v is None else (f"{v:.6f}" + (f" ({(v - b) * 100:+.3f} pp)" if b is not None and s != base else "")))
            body.append(line)
        table(f"Median identity (minimap2), Δ in percentage points vs `{base}`", ["stage", *CELLS], body)

    excl = load_excluded()
    if excl:
        out.append("\n### Excluded runs\n\nExcluded from every table above (outlier rule: more than %.0f %% above the cell median; `bench-runs/excluded-runs.json`):\n" % OUTLIER_PCT)
        out += [f"- `{k}`: {v}" for k, v in excl.items()]
    text = "\n".join(out) + "\n"
    if args.md:
        Path(args.md).write_text(text)
        print(f"wrote {args.md}")
    else:
        print(text)
    return 0


def summary_cmd(args: argparse.Namespace) -> int:
    """One row per (cell, mode, stage): n, mean/sd/min/max of Real time and the change against the baseline stage (default upstream)."""
    rows = tidy(bench.load_records())
    g = groups(rows, "t_real_s")
    base = args.baseline
    out = []
    for (stage, cell, mode), v in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        m, sd = mean_sd(v)
        b = g.get((base, cell, mode))
        bm = mean_sd(b)[0] if b else None
        out.append({"cell": cell, "mode": mode, "stage": stage, "n": len(v), "mean_s": round(m, 3), "sd_s": round(sd, 3),
                    "min_s": round(min(v), 3), "max_s": round(max(v), 3), f"baseline_{base}_mean_s": round(bm, 3) if bm else "",
                    "time_change_pct": round((m - bm) / bm * 100, 2) if bm else "", "speedup_x": round(bm / m, 3) if bm else "",
                    "throughput_gain_pct": round((bm / m - 1) * 100, 2) if bm else ""})
    path = Path(args.out)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")
    return 0


def csv_cmd(args: argparse.Namespace) -> int:
    rows = tidy(bench.load_records())
    if not rows:
        print("no timed records")
        return 1
    set_cells(rows)
    cols = list(rows[0].keys())
    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


# ---------------------------------------------------------------- validation

def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def _read_seqs(path: Path) -> dict[str, str]:
    import itertools
    d: dict[str, str] = {}
    with open(path) as f:
        for h, s_, _, _ in itertools.zip_longest(*[f] * 4):
            d[h.split()[0]] = s_.strip()
    return d


def equiv(args: argparse.Namespace) -> int:
    """V7: every stage's kept FASTQ must reproduce the reference stage's reads (identical, or near-identical where a difference is
    intended). A stage whose output is unrelated is INVALID (this is how the broken `pre_overlap` build was found)."""
    import difflib
    kept = bench.KEPT_DIR
    intended = {"spec": "speculative decode (approximate by design)", "paper_xavier": "different chunk size (-c 10000)"}
    print(f"{'file':44s} {'identical':>10s} {'len ratio':>10s} {'similarity':>10s}  verdict")
    bad = 0
    for ref in sorted(kept.glob(f"output_*-{args.ref}.fastq")):
        stem = ref.name[: -len(f"-{args.ref}.fastq")]
        R = _read_seqs(ref)
        for other in sorted(kept.glob(stem + "-*.fastq")):
            stage = other.name[len(stem) + 1:-6]
            if stage == args.ref:
                continue
            O = _read_seqs(other)
            ids = sorted(set(R) & set(O))
            if not ids:
                print(f"{other.name:44s} no common reads"); bad += 1; continue
            same = sum(R[i] == O[i] for i in ids)
            lr = st.median(len(O[i]) / max(1, len(R[i])) for i in ids)
            sim = st.median(difflib.SequenceMatcher(None, R[i][:400], O[i][:400], autojunk=False).ratio() for i in ids[:60])
            ok = sim >= 0.90 and 0.8 <= lr <= 1.2
            note = "" if ok else "INVALID?"
            if stage.split("-")[-1] in intended or stage in intended:
                note = (note + " ") + "(" + intended.get(stage, "intended difference") + ")"
            print(f"{other.name:44s} {same:>5d}/{len(ids):<4d} {lr:10.3f} {sim:10.3f}  {note}")
            bad += 0 if ok else 1
    print("\nV7:", "all stages reproduce the reference reads" if not bad else f"{bad} stage file(s) FAILED the equivalence check")
    return 0 if not bad else 1


def validate(args: argparse.Namespace) -> int:
    records = bench.load_records()
    timed = [r for r in records if r.get("kind") == "timed" and r.get("status") == "ok"]
    for r in timed:
        r["cell"] = norm_cell(r["cell"])
    set_cells(timed)
    allok = True
    print("V1  logger vs tegrastats (needs runs made with --tegrastats; compare MEAN power over windows >= 60 s)")
    spec = __import__("importlib.util").util.spec_from_file_location("pt", bench.BENCH_DIR.parent / "parse_tegrastats.py")
    pt = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(pt)
    from datetime import datetime
    v1 = []
    for r in timed:
        if "tegrastats_log" not in r or r["real_s"] < 60:
            continue
        rows = pt.load_samples(REPO / r["tegrastats_log"])
        sel = [x["VDD_IN_mW"] for x in rows
               if r["window_start_epoch"] <= datetime.strptime(x["timestamp"], "%m-%d-%Y %H:%M:%S").timestamp() <= r["window_end_epoch"]]
        if len(sel) >= 30:
            v1.append((st.mean(sel) / 1000.0, r["Pmean_in_W"], r["run_id"]))
    if v1:
        worst = max(abs(a - b) / b * 100 for a, b, _ in v1)
        allok &= check("mean-power agreement", worst <= 3.0, f"{len(v1)} runs, worst |diff| {worst:.2f} % (limit 3 %)")
    else:
        print("  (no qualifying runs yet)")

    print("V2  sampling adequacy: 1 Hz sample-and-hold vs full-resolution integration over the same window")
    d = [(r["E_in_J_1hz_emulated"] - r["E_in_J"]) / r["E_in_J"] * 100 for r in timed if r.get("E_in_J_1hz_emulated") and r.get("E_in_J")]
    if d:
        print(f"  n={len(d)} runs: mean {st.mean(d):+.2f} %, worst |.| {max(abs(x) for x in d):.2f} %  (informational: quantifies the old sampler's error)")
    else:
        print("  (no data)")

    print("V3  repeatability per (stage, cell, mode), n >= 3")
    g = defaultdict(list)
    for r in timed:
        g[(r["stage"], r["cell"], r["mode"])].append(r)
    reps = {k: v for k, v in g.items() if len(v) >= 3}
    if reps:
        worst_t = max((mean_sd([r["real_s"] for r in v])[1] / mean_sd([r["real_s"] for r in v])[0] * 100, k) for k, v in reps.items())
        worst_e = max((mean_sd([r["E_in_J"] for r in v])[1] / mean_sd([r["E_in_J"] for r in v])[0] * 100, k) for k, v in reps.items())
        allok &= check("time CV", worst_t[0] < 1.0, f"worst {worst_t[0]:.2f} % at {worst_t[1]} (limit 1 %)")
        allok &= check("energy CV", worst_e[0] < 3.0, f"worst {worst_e[0]:.2f} % at {worst_e[1]} (limit 3 %)")
    else:
        print("  (no cell with n >= 3 yet)")

    print("V4  window alignment and coverage")
    if timed:
        sps = [r["anchor_spread_ms"] for r in timed if r.get("anchor_spread_ms") is not None]
        if sps:
            allok &= check("start-epoch consistency", max(sps) <= 5.0, f"max spread across progress lines {max(sps):.1f} ms (limit 5 ms), {len(sps)} runs with progress-line anchors")
        xs = [r["real_line_after_window_end_ms"] for r in timed if r.get("real_line_after_window_end_ms") is not None]
        if xs:
            allok &= check("anchor cross-check", max(abs(x) for x in xs) <= 25.0, f"`Real time` line arrives {min(xs):.1f}..{max(xs):.1f} ms after the progress-line window end (limit 25 ms)")
        n_rl = sum(1 for r in timed if r.get("anchor_method") == "real_line")
        if n_rl:
            print(f"  note: {n_rl} runs used the `Real time`-line anchor (stream-mode builds print no progress lines)")
        eg = [r["exit_gap_s"] for r in timed]
        allok &= check("exit gap plausible", 0.0 <= min(eg) and max(eg) <= 3.0, f"{min(eg):.2f}..{max(eg):.2f} s (expect 0..3 s)")
        cov = min(r["covered_s"] / r["window_s"] for r in timed)
        allok &= check("rail coverage", cov >= 0.99, f"min covered/window {cov:.4f} (limit 0.99)")
        sh = [r["P_post_in_W"] - r["P_pre_in_W"] for r in timed if r.get("P_pre_in_W") and r.get("P_post_in_W")]
        # One-sided (amended 2026-10-23, see validation-run1-2026-10-22.md): the check guards against leftover load after the run,
        # so only a post-run shoulder ABOVE the pre-run one counts. A lower post-run shoulder is a post-exit transient, reported only.
        allok &= check("idle shoulders (leftover load)", max(sh) <= 0.5, f"max post-minus-pre idle power {max(sh):+.2f} W (limit +0.5 W)")
        print(f"    info: two-sided max |pre-post| {max(abs(x) for x in sh):.2f} W, median |.| {st.median(abs(x) for x in sh):.2f} W, mean signed {st.mean(sh):+.2f} W")

    print("V5  drift: canary and idle before/after each stage block")
    blocks = defaultdict(dict)
    for r in records:
        if r.get("kind") in ("canary", "idle") and r.get("status") == "ok":
            blocks[r["block"]][(r["kind"], r["label"].split(":")[-1])] = r
    for b, d5 in blocks.items():
        if ("canary", "before") in d5 and ("canary", "after") in d5:
            a, c = d5[("canary", "before")]["canary_ms"], d5[("canary", "after")]["canary_ms"]
            allok &= check(f"{b} canary", abs(c - a) / a <= 0.03, f"{a:.1f} -> {c:.1f} ms ({(c - a) / a * 100:+.2f} %, limit 3 %)")
        if ("idle", "before") in d5 and ("idle", "after") in d5:
            a, c = d5[("idle", "before")]["Pmean_in_W"], d5[("idle", "after")]["Pmean_in_W"]
            allok &= check(f"{b} idle power", abs(c - a) <= 0.3, f"{a:.2f} -> {c:.2f} W (limit 0.3 W)")
    if not blocks:
        print("  (no idle/canary blocks yet)")
    print("    order/drift: same stage + cell + mode measured in repeated blocks (limit: block means within 2 %)")
    byb: dict = defaultdict(lambda: defaultdict(list))
    for r in timed:
        byb[(r["stage"], r["cell"], r["mode"])][r["block"]].append(r["real_s"])
    for k, bl in sorted(byb.items()):
        if len(bl) >= 2:
            means = {b: st.mean(v) for b, v in bl.items()}
            spread = (max(means.values()) - min(means.values())) / st.mean(means.values()) * 100
            allok &= check(f"{k[0]} {k[1]} mode {k[2]}", spread <= 2.0, f"{len(bl)} blocks, means " + ", ".join(f"{m:.3f}" for m in means.values()) + f" s (spread {spread:.2f} %)")

    print("V6  mode A vs mode B")
    for s in sorted({r["stage"] for r in timed}):
        for c in CELLS:
            a = [r["real_s"] for r in timed if r["stage"] == s and r["cell"] == c and r["mode"] == "A"]
            b = [r["real_s"] for r in timed if r["stage"] == s and r["cell"] == c and r["mode"] == "B"]
            sy = [r["sync_s"] for r in timed if r["stage"] == s and r["cell"] == c and r["mode"] == "B" and r.get("sync_s") is not None]
            if a and b:
                print(f"  {s} {c}: B vs A {(st.mean(b) - st.mean(a)) / st.mean(a) * 100:+.2f} %  (A {st.mean(a):.3f} s, B {st.mean(b):.3f} s, sync {st.mean(sy) if sy else 0:.3f} s)")
    print("\nOVERALL:", "ALL CHECKED CRITERIA PASS" if allok else "SOME CHECKS FAILED")
    return 0 if allok else 1


# ---------------------------------------------------------------- legacy re-derivation

def legacy(args: argparse.Namespace) -> int:
    root = REPO / "docs-riley" / "power-runs"
    print(f"{'run':44s} {'E_summary':>9s} {'E_recomp':>9s} {'diff%':>6s} {'cg%':>5s} {'soc%':>5s} {'resid%':>6s}")
    worst = 0.0
    for d in sorted(root.glob("2026*")):
        if not (d / "summary.csv").exists():
            continue
        s = next(csv.DictReader(open(d / "summary.csv")))
        rows = list(csv.DictReader(open(d / "samples.csv")))
        tot = [float(r["VDD_IN_mW"]) for r in rows if r.get("VDD_IN_mW")]
        cg = [float(r["VDD_CPU_GPU_CV_mW"]) for r in rows if r.get("VDD_CPU_GPU_CV_mW")]
        so = [float(r["VDD_SOC_mW"]) for r in rows if r.get("VDD_SOC_mW")]
        e = sum(tot) / 1000.0
        es = float(s["energy_J"])
        diff = (e - es) / es * 100
        worst = max(worst, abs(diff))
        print(f"{d.name[:44]:44s} {es:9.1f} {e:9.1f} {diff:6.2f} {100 * sum(cg) / sum(tot):5.1f} {100 * sum(so) / sum(tot):5.1f} {100 * (1 - (sum(cg) + sum(so)) / sum(tot)):6.1f}")
    print(f"worst |diff| between re-derived and stored energy: {worst:.3f} %  ({'OK' if worst < 1 else 'CHECK'})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("csv")
    p.add_argument("--out", default=str(bench.OUT_ROOT / "runs.csv"))
    p.set_defaults(fn=csv_cmd)
    p = sub.add_parser("tables")
    p.add_argument("--baseline", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--accuracy", nargs="*", default=None, help="minimap_accuracy*.csv from fetch-and-minimap.py")
    p.set_defaults(fn=tables)
    p = sub.add_parser("summary", help="one row per cell/mode/stage with mean, sd and the change against a baseline stage")
    p.add_argument("--baseline", default="upstream")
    p.add_argument("--out", default=str(bench.OUT_ROOT / "summary-by-cell.csv"))
    p.set_defaults(fn=summary_cmd)
    p = sub.add_parser("outliers", help="list (and with --write exclude from csv/tables) runs > 3 %% above their cell median")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=outliers)
    sub.add_parser("validate").set_defaults(fn=validate)
    p = sub.add_parser("equiv", help="V7: compare kept FASTQs of every stage with the reference stage")
    p.add_argument("--ref", default="upstream")
    p.set_defaults(fn=equiv)
    sub.add_parser("legacy").set_defaults(fn=legacy)
    ap.add_argument("--runs-file", default=None, help="read records from this file instead of runs.jsonl")
    args = ap.parse_args()
    if args.runs_file:
        bench.RUNS_FILE = Path(args.runs_file)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
