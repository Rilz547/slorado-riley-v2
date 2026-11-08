#!/usr/bin/env python3
"""SUP stage probe: do the campaign stages (upstream, ov_off, ov_on, lb_best, spec) run on the SUP model (transformer, CRF state length 5)?
100 reads, -C 32 (the only batch size that fit in 8 GB), FASTQ written so reads can be compared with `upstream`. ~4-5 min in total.
Not a measurement for the report. Uses the stage binaries and flags from the frozen stages.json, no harness changes.

    python3 sup_stage_probe.py [--flash] [--stages upstream,ov_off,ov_on,lb_best,spec]
Writes docs-riley/exp/sup-stage-probe.log (+ one console log per stage).
"""
from __future__ import annotations
import argparse, difflib, os, re, statistics, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench, analyze

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "data/sup_probe/reads_1k_0.blow5"
MODEL = "models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"
OUTDIR = Path.home() / "data/sup_probe"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="upstream,ov_off,ov_on,lb_best,spec")
    ap.add_argument("--flash", action="store_true")
    ap.add_argument("--C", default="32")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE set for every run (e.g. CUDA_LAUNCH_BLOCKING=1); repeatable")
    a = ap.parse_args()
    cfg = bench.load_config()
    cfg["models"]["dna"]["sup"] = MODEL
    c = bench.parse_cell(cfg, "dna_sup_1k")
    log = open(HERE / "sup-stage-probe.log", "a")
    def say(s: str) -> None:
        print(s, flush=True); log.write(s + "\n"); log.flush()
    say(f"=== SUP stage probe {time.strftime('%F %T')}  -C {a.C}  flash={'yes' if a.flash else 'no'}  env={a.env or '-'}  nvpmodel: {' '.join(bench.sh(['nvpmodel', '-q']).split())}")
    ref: dict | None = None
    prev: dict[str, dict] = {}   # last output of each stage, to compare a repeated stage with itself
    for stage in a.stages.split(","):
        name, _, binary = stage.partition("@")   # `ov_on@x_barrier` = the ov_on flags with ~/bench-stages/x_barrier/slorado
        out = OUTDIR / f"out-{stage.replace('@', '_')}.fastq"
        cmd = bench.build_cmd(cfg, name, c, str(out), None, ["--flash=yes"] if a.flash else None)
        if binary:
            cmd[0] = str(Path.home() / "bench-stages" / binary / "slorado")
        cmd[cmd.index("-C") + 1] = a.C
        cmd[-1] = str(DATA)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=bench.REPO, capture_output=True, text=True, timeout=900, env={**os.environ, **dict(e.split('=', 1) for e in a.env)})
            rc, err = p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, err = -9, (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        (HERE / f"sup-stage-probe-{stage}.log").write_text(err)
        m = re.search(r"Real time: ([\d.]+) sec", err)
        inf = re.search(r"- inference: ([\d.]+) sec", err)
        dec = re.search(r"- decode: ([\d.]+) sec", err)
        problems = [ln.strip()[:110] for ln in err.splitlines() if re.search(r"NvMap|error|Error|assert|terminate|what\(\)", ln)][:2]
        same = ""
        if rc == 0 and out.exists():
            seqs = analyze._read_seqs(out)
            if ref is None:
                ref = seqs
            ids = sorted(set(ref) & set(seqs))
            diff = [i for i in ids if ref[i] != seqs[i]]
            sim = statistics.median(difflib.SequenceMatcher(None, ref[i][:400], seqs[i][:400], autojunk=False).ratio() for i in diff[:40]) if diff else 1.0
            same = f"reads {len(seqs)}  identical to first stage {len(ids) - len(diff)}/{len(ids)} (similarity of the differing reads {sim:.3f})"
            if stage in prev:
                pid = sorted(set(prev[stage]) & set(seqs))
                same += f"  | to its own previous run {sum(prev[stage][i] == seqs[i] for i in pid)}/{len(pid)}"
            prev[stage] = seqs
        spec = [ln.strip() for ln in err.splitlines() if "speculative" in ln.lower() and ("drafted" in ln.lower() or "repair" in ln.lower())][:1]
        say(f"{stage:9} exit {rc:4d}  Real {m.group(1) if m else '-':>7} s  inference {inf.group(1) if inf else '-':>7}  decode {dec.group(1) if dec else '-':>6}  "
            f"(wall {time.time() - t0:.0f} s)  {same} {' | '.join(spec)} {' | '.join(problems)}")
        subprocess.run(["pkill", "-9", "-f", "slorado basecaller"], capture_output=True)
        time.sleep(15)
    say("done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
