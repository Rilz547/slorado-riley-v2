import numpy as np, subprocess, os, sys, tempfile, time, shutil
OF, batch, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
T, C = 1666, 256
base = np.fromfile(os.path.join(os.path.dirname(OF), 'test/data/scores.blob'), dtype=np.float16)[:T*C].reshape(T, C)
tiled = np.ascontiguousarray(np.broadcast_to(base[:, None, :], (T, batch, C)))
blob = os.path.join(os.path.dirname(OF), f'tiled_{batch}.blob'); tiled.tofile(blob)
ref = None; bad_runs = hung = 0; cap_total = 0
t_start = time.time()
for r in range(reps):
    d = tempfile.mkdtemp(dir=os.path.dirname(OF))
    p = subprocess.run(['timeout', '-s', 'KILL', '15', OF, blob, str(batch), '3'], cwd=d, capture_output=True)
    if p.returncode != 0:
        hung += 1; print(f'run {r}: rc={p.returncode} (HUNG/KILLED)', flush=True)
    else:
        m = np.fromfile(os.path.join(d, 'moves.blob'), dtype=np.uint8).reshape(batch, T)
        s = np.fromfile(os.path.join(d, 'sequence.blob'), dtype=np.uint8).reshape(batch, T)
        q = np.fromfile(os.path.join(d, 'qstring.blob'), dtype=np.uint8).reshape(batch, T)
        diff_rows = [i for i in range(1, batch) if not (np.array_equal(m[i], m[0]) and np.array_equal(s[i], s[0]) and np.array_equal(q[i], q[0]))]
        if ref is None: ref = (m[0].copy(), s[0].copy(), q[0].copy())
        vs_ref = not (np.array_equal(m[0], ref[0]) and np.array_equal(s[0], ref[1]) and np.array_equal(q[0], ref[2]))
        err = p.stderr.decode(errors='ignore')
        cap = [l for l in err.splitlines() if l.startswith('DBG')]
        if cap: cap_total += int(cap[0].split('cap_hits=')[1])
        if diff_rows or vs_ref:
            bad_runs += 1; print(f'run {r}: chunks differing from chunk 0: {len(diff_rows)}; chunk0 differs from first run: {vs_ref}', flush=True)
    shutil.rmtree(d)
    if (r + 1) % 50 == 0: print(f'  ... {r+1} runs, {bad_runs} inconsistent, {hung} hung, {time.time()-t_start:.0f}s', flush=True)
print(f'SUMMARY {os.path.basename(os.path.dirname(OF))}: {reps} runs x {batch} chunks: {bad_runs} runs with inconsistent output, {hung} hung/killed; cap-hit events observed ~{cap_total}')
