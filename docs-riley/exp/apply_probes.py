#!/usr/bin/env python3
"""Apply the T-C timing probes to ~/bench-stages/exp_src (openfish submodule).  Idempotent: refuses to run twice.
Probes are selected at run time with OPENFISH_EXP=<comma list>; unset = exactly the original behaviour.
  dup_bwd dup_fwd dup_beam dup_draft dup_qual dup_gen   launch that kernel a second time (outputs unchanged)
  skip_repair_scans                                     repair (non-greedy) call skips bwd_scan + fwd_post_scan  (TIMING ONLY: output wrong)
  pack32                                                greedy/qual/gen use 32-thread blocks, ceil(N/32) blocks (R7; output must be identical)
"""
import re, sys
from pathlib import Path
root = Path.home() / "bench-stages/exp_src/openfish/src"
dc = root / "decode_cuda.c"
s = dc.read_text()
if "OPENFISH_EXP" in s:
    sys.exit("probes already applied")

s = s.replace('''/* Pick event-based timing when overlapping''', '''/* ---- EXPERIMENT PROBES (docs-riley/exp). OPENFISH_EXP=<comma list>; unset = original behaviour. ---- */
typedef struct {
    int dup_bwd, dup_fwd, dup_beam, dup_draft, dup_qual, dup_gen, skip_repair_scans, pack32;
} exp_flags_t;

static exp_flags_t exp_flags(void) {
    static int init = 0;
    static exp_flags_t f;
    if (!init) {
        const char *e = getenv("OPENFISH_EXP");
        if (e != NULL) {
            f.dup_bwd = strstr(e, "dup_bwd") != NULL;
            f.dup_fwd = strstr(e, "dup_fwd") != NULL;
            f.dup_beam = strstr(e, "dup_beam") != NULL;
            f.dup_draft = strstr(e, "dup_draft") != NULL;
            f.dup_qual = strstr(e, "dup_qual") != NULL;
            f.dup_gen = strstr(e, "dup_gen") != NULL;
            f.skip_repair_scans = strstr(e, "skip_repair_scans") != NULL;
            f.pack32 = strstr(e, "pack32") != NULL;
        }
        init = 1;
    }
    return f;
}

/* Pick event-based timing when overlapping''', 1)

s = s.replace("#include <stdint.h>\n", "#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n", 1)

# launch geometry for the one-thread-per-chunk kernels
s = s.replace('''	dim3 grid_size(batch_size, 1, 1);
''', '''	dim3 grid_size(batch_size, 1, 1);
    const exp_flags_t xf = exp_flags();
    dim3 gen_block(xf.pack32 ? 32 : 1, 1, 1);
    dim3 gen_grid(xf.pack32 ? (batch_size + 31) / 32 : batch_size, 1, 1);
''', 1)

# half branch only (production path)
s = s.replace('''            bwd_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
            checkCudaError();
''', '''            if (!(xf.skip_repair_scans && !greedy)) {
                bwd_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
                checkCudaError();
            }
            if (xf.dup_bwd) {
                bwd_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC);
                checkCudaError();
            }
''', 1)
g_old = '''                greedy_search<half><<<grid_size,block_size_gen,0,s>>>(
                    beam_args, scores_NTC, gpubuf->bwd_NTC,
                    (state_t *)gpubuf->states, gpubuf->moves, gpubuf->margins,
                    fixed_stay_score, score_scale
                );
                checkCudaError();
'''
g_new = '''                greedy_search<half><<<gen_grid,gen_block,0,s>>>(
                    beam_args, scores_NTC, gpubuf->bwd_NTC,
                    (state_t *)gpubuf->states, gpubuf->moves, gpubuf->margins,
                    fixed_stay_score, score_scale
                );
                checkCudaError();
                if (xf.dup_draft) {
                    greedy_search<half><<<gen_grid,gen_block,0,s>>>(
                        beam_args, scores_NTC, gpubuf->bwd_NTC,
                        (state_t *)gpubuf->states, gpubuf->moves, gpubuf->margins,
                        fixed_stay_score, score_scale
                    );
                    checkCudaError();
                }
'''
assert g_old in s
s = s.replace(g_old, g_new, 1) if s.count(g_old) == 1 else s.replace(g_old, g_new)  # int8 branch uses <int8_t>, so exactly one match
b_old = '''                beam_search<half><<<grid_size,block_size_beam,num_states*sizeof(float),s>>>(
                    beam_args, scores_NTC, gpubuf->bwd_NTC,
                    (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
                    beam_cut, fixed_stay_score, score_scale
                );
                checkCudaError();
'''
b_new = b_old + '''                if (xf.dup_beam) {
                    beam_search<half><<<grid_size,block_size_beam,num_states*sizeof(float),s>>>(
                        beam_args, scores_NTC, gpubuf->bwd_NTC,
                        (state_t *)gpubuf->states, gpubuf->moves, (beam_element_t *)gpubuf->beam_vector,
                        beam_cut, fixed_stay_score, score_scale
                    );
                    checkCudaError();
                }
'''
assert b_old in s
s = s.replace(b_old, b_new, 1)
f_old = '''            fwd_post_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
            checkCudaError();
'''
f_new = '''            if (!(xf.skip_repair_scans && !greedy)) {
                fwd_post_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
                checkCudaError();
            }
            if (xf.dup_fwd) {
                fwd_post_scan<half><<<grid_size,block_size,0,s>>>(scan_args, scores_NTC, gpubuf->bwd_NTC, gpubuf->post_NTC);
                checkCudaError();
            }
'''
assert f_old in s
s = s.replace(f_old, f_new, 1)
q_old = '''        compute_qual_data<<<grid_size,block_size_gen,0,s>>>(
            beam_args,
            gpubuf->post_NTC,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            1.0f
        );
        checkCudaError();
'''
q_new = '''        if (xf.dup_qual) {
            /* compute_qual_data rewrites states in place (% NUM_BASES): run the extra copy on a scratch copy */
            static state_t *scratch = NULL;
            static size_t scratch_n = 0;
            const size_t need = (size_t)batch_size * n_timesteps;
            if (need > scratch_n) {
                if (scratch != NULL) { cudaFree(scratch); }
                cudaMalloc((void **)&scratch, need * sizeof(state_t));
                checkCudaError();
                scratch_n = need;
            }
            cudaMemcpyAsync(scratch, gpubuf->states, need * sizeof(state_t), cudaMemcpyDeviceToDevice, s);
            checkCudaError();
            compute_qual_data<<<gen_grid,gen_block,0,s>>>(beam_args, gpubuf->post_NTC, scratch, gpubuf->qual_data, 1.0f);
            checkCudaError();
        }
        compute_qual_data<<<gen_grid,gen_block,0,s>>>(
            beam_args,
            gpubuf->post_NTC,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            1.0f
        );
        checkCudaError();
'''
assert q_old in s
s = s.replace(q_old, q_new, 1)
g2_old = '''        generate_sequence<<<grid_size,block_size_gen,0,s>>>(
            beam_args,
            gpubuf->moves,
            (state_t *)gpubuf->states,
            gpubuf->qual_data,
            gpubuf->base_probs,
            gpubuf->total_probs,
            gpubuf->sequence,
            gpubuf->qstring,
            q_shift,
            q_scale
        );
        checkCudaError();
'''
g2_new = g2_old.replace("<<<grid_size,block_size_gen,0,s>>>", "<<<gen_grid,gen_block,0,s>>>") + '''        if (xf.dup_gen) {
''' + g2_old.replace("<<<grid_size,block_size_gen,0,s>>>", "<<<gen_grid,gen_block,0,s>>>").replace("\n        ", "\n            ").replace("        generate_sequence", "            generate_sequence", 1).rstrip() + '''
        }
'''
assert g2_old in s
s = s.replace(g2_old, g2_new, 1)
# int8 branch: keep original geometry (probes are half-only) but the shared names must still compile
dc.write_text(s)

# kernels: chunk index for the 1-thread kernels works for blockDim.x == 1 (unchanged) and blockDim.x == 32 (pack32)
gs = root / "greedy_search_cuda.h"
t = gs.read_text()
t = t.replace('''    const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);
    if (chunk >= args.batch_size || threadIdx.x != 0 || threadIdx.y != 0) {''', '''    const uint64_t chunk = (blockIdx.x + (blockIdx.y * gridDim.x)) * blockDim.x + threadIdx.x;
    if (chunk >= args.batch_size || threadIdx.y != 0) {''', 1)
gs.write_text(t)
bs = root / "beam_search_cuda.h"
t = bs.read_text()
for head in ("static __global__ void generate_sequence(", "static __global__ void compute_qual_data("):
    i = t.index(head)
    j = t.index("const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);", i)
    t = t[:j] + "const uint64_t chunk = (blockIdx.x + (blockIdx.y * gridDim.x)) * blockDim.x + threadIdx.x;" + t[j + len("const uint64_t chunk = blockIdx.x + (blockIdx.y * gridDim.x);"):]
bs.write_text(t)
print("probes applied")
