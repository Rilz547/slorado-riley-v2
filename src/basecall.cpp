/**
 * @file basecall.cpp
 * @brief runs DNA base calling steps
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

/** Riley Updates (Remove at the end)
 * @file basecall.cpp
 * @lastmodified: P5-lite overlap: 2-slot host ring; sync prev decode, launch next decode, then write prev on CPU under next decode.
 * @lastpatched: 2026-07-18

******************************************************************************/

#include <cstdint>
#include <stdlib.h>
#include <vector>

#include "torchbox.h"
#include "basecall.h"
#include "misc.h"
#include "error.h"

#ifdef USE_GPU
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

typedef struct {
    std::vector<basecall_chunk_t *> chunks;
    // CRF scores in openfish layout NTC [batch, time, channels] (no TNC transpose).
    torch::Tensor scores_NTC;
    int slot;       // input / infer_event ping-pong index
    int host_slot;  // openfish pinned-host ring index (OPENFISH_HOST_RING)
    bool active;
    // Set after decode kernels+D2H are queued but before stream sync / write.
    bool decode_launched;
    uint8_t *moves;
    char *sequence;
    char *qstring;
    int T;
} overlap_pending_t;

static torch::Tensor &runner_input_slot(runner_t *runner, int slot) {
    return (slot == 0) ? runner->input_tensor : runner->input_tensor_alt;
}

int64_t resolve_score_index(
    const int64_t hit_sig_abs,
    const int64_t chunk_signal_start,
    const int64_t scores_states,
    const int64_t chunk_size,
    const int64_t context_samples_before,
    const int64_t context_samples_after,
    const int64_t modbase_stride
) {
    if (hit_sig_abs < chunk_signal_start) {
        ERROR("%s", "Modbase hit before chunk start.");
    }

    // Context hit chunk-relative signal index
    const int64_t hit_sig_rel = hit_sig_abs - chunk_signal_start;

    // Skip hits at end of a chunk without enough downstream context
    // It will be processed at the start if the next chunk with complete context
    if (hit_sig_rel > chunk_size - context_samples_after) {
        return -2;
    }

    // Skip hits at the start of a chunk with insufficient context
    // This hit will have been processed in a previous chunk
    // UNLESS it's the start of a read where there's no useful lead-in
    if (hit_sig_abs > context_samples_before && hit_sig_rel < context_samples_before) {
        return -1;
    }

    // We should land on a canonical base
    if (hit_sig_rel % modbase_stride != 0) {
        ERROR("%s", "Modbase score did not align to canonical base.");
    }

    // Convert chunk-relative signal-space score index into sequence-space (/stride)
    // and then into scores-space (*num_states)
    return hit_sig_rel / modbase_stride * scores_states;
}

static void convert_f32_to_f16_impl(c10::Half* const dest, const float* const src, size_t count) {
    auto src_tensor_f32 = at::from_blob(const_cast<float*>(src), {static_cast<int64_t>(count)});
    auto src_tensor_f16 = src_tensor_f32.to(at::ScalarType::Half);
    std::memcpy(dest, src_tensor_f16.data_ptr(), count * sizeof(c10::Half));
}

void copy_tensor_elems(
    at::Tensor& dest_tensor,
    std::size_t dest_offset,
    const at::Tensor& src_tensor,
    std::size_t src_offset,
    std::size_t count
) {
    assert(dest_tensor.is_contiguous());
    assert(src_tensor.is_contiguous());
    assert(dest_offset + count <= size_t(dest_tensor.numel()));
    assert(src_offset + count <= size_t(src_tensor.numel()));

    if (dest_tensor.dtype() == src_tensor.dtype()) {
        // No conversion.
        char* const dest_ptr = reinterpret_cast<char*>(dest_tensor.data_ptr());
        const char* const src_ptr = reinterpret_cast<const char*>(src_tensor.data_ptr());
        const size_t elem_size = dest_tensor.element_size();
        std::memcpy(&dest_ptr[dest_offset * elem_size], &src_ptr[src_offset * elem_size],
                    count * elem_size);
    } else if (dest_tensor.dtype() == at::ScalarType::Half &&
               src_tensor.dtype() == at::ScalarType::Float) {
        // float32 -> float16 conversion.
        auto* const dest_ptr = dest_tensor.data_ptr<c10::Half>();
        const auto* const src_ptr = src_tensor.data_ptr<float>();
        convert_f32_to_f16_impl(&dest_ptr[dest_offset], &src_ptr[src_offset], count);
    } else {
        // Slow fallback path for other conversions.
        using at::indexing::Slice;
        dest_tensor.flatten().index_put_(
                {Slice(dest_offset, dest_offset + count)},
                src_tensor.flatten().index({Slice(src_offset, src_offset + count)}));
    }
}

static void mod_accept_chunk(const int num_chunks, const torch::Tensor& signal, const std::vector<int8_t>& kmers, const core_t* core, const int runner_idx) {
    runner_t* runner = (*core->mod_runners)[runner_idx];

    auto& input_sigs = runner->input_sigs;
    auto& input_seqs = runner->input_seqs;

    // auto& input_sigs = m_input_sigs[model_id];
    // auto& input_seqs = m_input_seqs[model_id];
    if (signal.size(0) != input_sigs.size(2)) {
        ERROR("%s", "ModBaseRunner received signal and sequence chunks with different lengths.");
        exit(1);
    }
    input_sigs = input_sigs.contiguous(); // todo: move these somewhere else
    input_seqs = input_seqs.contiguous();

    const auto sig_len = signal.size(0);
    copy_tensor_elems(input_sigs, num_chunks * sig_len, signal, 0, sig_len);

    const auto kmer_elem_count = input_seqs.size(1) * input_seqs.size(2);
    assert(input_seqs.dtype() == torch::kInt8);
    int8_t* const input_seqs_ptr = input_seqs.data_ptr<int8_t>();
    std::memcpy(&input_seqs_ptr[num_chunks * kmer_elem_count], kmers.data(),  kmer_elem_count * sizeof(int8_t));
}

static void accept_chunk(const int num_chunks, const basecall_chunk_t *chunk, runner_t *runner, int chunk_size, int slot) {
    ASSERT(chunk->read_dat->scaled_signal.size(0) > 0);
    torch::Tensor input_slice = (chunk->read_dat->scaled_signal).index({torch::indexing::Ellipsis, torch::indexing::Slice(chunk->input_offset, chunk->input_offset + chunk_size)});
    input_slice = input_slice.unsqueeze(0);
    auto slice_size = input_slice.size(1);
    ASSERT(slice_size != 0);

    // repeat-pad non-full chunks
    if (slice_size != chunk_size) {
        int64_t quot = chunk_size / slice_size;
        int64_t rem = chunk_size % slice_size;
        input_slice = torch::concat(
            {
                input_slice.repeat({1, quot}),
                input_slice.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, rem)})
            },
            1
        );
    }

    runner_input_slot(runner, slot).index_put_({num_chunks, 0}, {input_slice});
}

static void write_decode_results(
    const std::vector<basecall_chunk_t *> &chunks,
    int T,
    uint8_t *moves,
    char *sequence,
    char *qstring
) {
    for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        size_t idx = chunk * T;
        chunks[chunk]->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move: chunks[chunk]->moves) {
            num_bases += move;
        }
        if (num_bases > (size_t)T) {
            ERROR("num bases %zu greater than number of timesteps %d", num_bases, T);
            exit(EXIT_FAILURE);
        }
        chunks[chunk]->seq = std::string(sequence + idx, num_bases);
        chunks[chunk]->qstring = std::string(qstring + idx, num_bases);

        size_t seq_size = strlen(chunks[chunk]->seq.c_str());
        size_t qstr_size = strlen(chunks[chunk]->qstring.c_str());

        if (seq_size == 0) {
            ERROR("%s", "empty sequence returned by decoder");
            exit(EXIT_FAILURE);
        }

        if (qstr_size == 0) {
            ERROR("%s", "empty qstring returned by decoder");
            exit(EXIT_FAILURE);
        }
        
        if (seq_size != qstr_size) {
            ERROR("mismatch sequence size of %zu with qstring size of %zu", seq_size, qstr_size);
            ERROR("seq: %s", chunks[chunk]->seq.c_str());
            ERROR("qstring: %s", chunks[chunk]->qstring.c_str());
            exit(EXIT_FAILURE);
        }
    }
}

static void decode_scores_to_chunks(
    const core_t* core,
    runner_t* runner,
    runner_stat_t* ts,
    const std::vector<basecall_chunk_t *> &chunks,
    torch::Tensor &scores_NTC,
    void *cuda_stream
) {
    // openfish expects NTC [N,T,C]; model forward already produces that layout.
    const int N = scores_NTC.size(0);
    const int T = scores_NTC.size(1);
    const int C = scores_NTC.size(2);
    const int state_len = core->model_config->state_len;
    int nthreads = core->opt.num_thread / core->runners->size();

    uint8_t *moves;
    char *sequence;
    char *qstring;

    LOG_DEBUG("%s", "decoding scores");

    ts->time_decode -= realtime();
    if (runner->device == "cpu") {
        openfish_decode_cpu(T, N, C, nthreads, scores_NTC.data_ptr(), OPENFISH_SCORE_F16, 1.0f, state_len, &core->decoder_opts, &moves, &sequence, &qstring, &ts->decode_stats);
    } else {
#ifdef USE_GPU
        openfish_decode_gpu(T, N, C, scores_NTC.data_ptr(), OPENFISH_SCORE_F16, 1.0f, state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring, &ts->decode_stats, cuda_stream, 0);
#else
        ERROR("Invalid device: %s. Please compile again for GPU", runner->device.c_str());
        exit(EXIT_FAILURE);
#endif
    }

    LOG_DEBUG("%s", "writing to chunks");
    write_decode_results(chunks, T, moves, sequence, qstring);
    ts->time_decode += realtime();

    LOG_DEBUG("%s", "done writing to chunks");

    if (runner->device == "cpu") {
        free(moves);
        free(sequence);
        free(qstring);
    }
    /* GPU path: host buffers are persistent on openfish_gpubuf_t — do not free per batch. */
}

static void call_chunks(
    const core_t* core,
    const std::vector<basecall_chunk_t *> &chunks,
    const int runner_idx
) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];

#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;
    
    LOG_DEBUG("%s", "basecalling chunks");
    ts->time_infer -= realtime();
    /* Narrow the fixed [C,1,chunk_size] input buffer to the N real chunks before forward, so the
       GPU only infers/decodes N slots instead of the full C (tail/partial batches no longer pad).
       --fixed-c-batch=yes disables this to reproduce the original full-C padded behaviour. */
    const int64_t N = (int64_t)chunks.size();
    torch::Tensor input = (core->opt.flag & SLORADO_FIXED_C_BATCH)
                          ? runner->input_tensor
                          : runner->input_tensor.narrow(0, 0, N);
    auto scores = runner->module->forward(input.to(runner->tensor_opts.device()));
#ifdef USE_GPU
    if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
#endif
    ts->time_infer += realtime();

    // Keep NTC — openfish decode consumes batch-major scores (see openfish/src/main.c).
    // Transposing to TNC made N=1 look fine (layouts coincide) but destroyed accuracy for N>1.
    auto scores_NTC = scores.contiguous();
#ifdef USE_GPU
    if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
#endif

    decode_scores_to_chunks(core, runner, ts, chunks, scores_NTC, nullptr);
}

static void mod_call_chunks(
    const core_t* core,
    const std::vector<mod_chunk_t *> &chunks,
    const int runner_idx
) {
    runner_t* runner = (*core->mod_runners)[runner_idx];
    
#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;

    LOG_DEBUG("%s", "mod calling chunks");
    const int64_t active_chunks = static_cast<int64_t>(chunks.size());
    auto active_input_sigs = runner->input_sigs.narrow(0, 0, active_chunks);
    auto active_input_seqs = runner->input_seqs.narrow(0, 0, active_chunks);

    auto scores = runner->module->forward(
        active_input_sigs.to(runner->tensor_opts.device_opt().value()),
        active_input_seqs.to(runner->tensor_opts.device_opt().value())
    );
#ifdef USE_GPU
    if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
#endif
    
    auto scores_f16 = scores.cpu().contiguous();
    assert(scores_f16.is_contiguous());
    assert(scores_f16.dtype() == at::ScalarType::Half);

    const int64_t row_size = scores_f16.size(1);
    const auto* const scores_f16_ptr = scores_f16.data_ptr<c10::Half>();
    for (size_t i = 0; i < chunks.size(); ++i) {
        mod_chunk_t *chunk = chunks[i];
        read_dat_t *read_dat = chunk->read_dat;
        const int64_t row_offset = static_cast<int64_t>(i) * row_size;

        const std::vector<int64_t>& hits_seq = read_dat->per_base_hits_seq.at(chunk->base_id);
        const std::vector<int64_t>& hits_sig = read_dat->per_base_hits_sig.at(chunk->base_id);

        const auto& cfg = core->modbase_config;
        const char modbase_model_base = cfg->mods.base;
        const int64_t modbase_stride = cfg->general.stride;
        const int64_t chunk_size = cfg->context.chunk_size;
        const int64_t context_samples_before = cfg->context.samples_before;
        const int64_t context_samples_after = cfg->context.samples_after;

        // The number of states predicted by this modbase model `num_mods + 1`
        const int64_t scores_states = chunk->num_states;
        const int64_t scores_size = row_size;
        // const int64_t scores_seq_len = scores_size / scores_states;

        const int64_t base_offset = static_cast<int64_t>(core->modbase_info->base_probs_offsets.at(cfg->mods.base_id));
        const auto num_states = NUM_BASES + cfg->mods.count;

        for (size_t hit = chunk->hit_offset; hit < hits_sig.size(); ++hit) {
            // Context hit sequence index in the chunk sequence
            const int64_t hit_seq = hits_seq.at(hit);

            // const auto& seq = is_template_direction
            //                           ? read.seq[hit_seq]
            //                           : dorado::utils::complement_table[read_dat->seq[hit_seq]];

            char seq = read_dat->seq[hit_seq];

            // The canonical base should be constant for a single model
            if (seq != modbase_model_base) {
                ERROR("Modbase hit base is not correct : %c", seq);
            }

            int64_t hit_score_idx = resolve_score_index(
                    hits_sig.at(hit), chunk->signal_offset, scores_states, chunk_size,
                    context_samples_before, context_samples_after, modbase_stride);

            if (hit_score_idx <= -2) {
                // No more hits in this chunk
                break;
            } else if (hit_score_idx == -1) {
                // This hit is skipped
                continue;
            }

            // Extract the scores for the canonical base and each of the mods in this model
            for (int64_t mod_offset = 0; mod_offset < scores_states; ++mod_offset) {
                const int64_t score_idx = hit_score_idx + mod_offset;
                if (score_idx >= scores_size) {
                    ERROR("%s", "Modbase score index out of bounds.");
                }

                const int64_t row_score_idx = row_offset + score_idx;
                const float score_value = static_cast<float>(scores_f16_ptr[row_score_idx]);
                const uint8_t score = static_cast<uint8_t>(std::min(std::floor(score_value * 256), 255.0f));

                // Index into the probabilities is calculated by
                // sequence_index * num_states := canonical "A" base probs index
                // offset then by the canonical base modification offsets
                const int64_t prob_idx = hit_seq * num_states + base_offset + mod_offset;
                read_dat->base_mod_probs.at(prob_idx) = score;
            }
        }
    }

//     auto scores_TNC = scores;
//     // scores_TNC = scores_TNC.to(torch::kCPU).to(torch::kF32).transpose(0, 1).contiguous();
//     scores_TNC = scores_TNC.transpose(0, 1).contiguous();
// #ifdef USE_GPU
//     if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
// #endif

//     const int T = scores_TNC.size(0);
//     const int N = scores_TNC.size(1);
//     const int C = scores_TNC.size(2);
//     const int state_len = core->model_config->state_len;
//     int nthreads = core->opt.num_thread / core->runners->size();
}

static void basecall_chunks(
    const core_t* core,
    const int runner_idx,
    const std::vector<basecall_chunk_t *> &chunks
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    runner_t* runner = (*core->runners)[runner_idx];
    auto chunk_size = core->chunk_size;

    LOG_DEBUG("%s", "accepting chunks");
    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) {
        accept_chunk(i, chunks[i], runner, chunk_size, 0);
    }
    ts->time_accept += realtime();
    LOG_DEBUG("%s", "done accepting chunks");

    ts->time_basecall -= realtime();
    call_chunks(core, chunks, runner_idx);
    ts->time_basecall += realtime();
}

#ifdef USE_GPU
/* Queue decode kernels + pinned D2H for pending into pending->host_slot; does NOT sync. */
static void overlap_launch_decode(
    const core_t* core,
    runner_t* runner,
    runner_stat_t* ts,
    overlap_pending_t *pending
) {
    if (!pending->active || pending->decode_launched) {
        return;
    }

    c10::cuda::CUDAStreamGuard decode_guard(*runner->decode_stream);
    cudaError_t err = cudaStreamWaitEvent(runner->decode_stream->stream(), runner->infer_event[pending->slot], 0);
    if (err != cudaSuccess) {
        ERROR("cudaStreamWaitEvent failed: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    const int N = pending->scores_NTC.size(0);
    const int T = pending->scores_NTC.size(1);
    const int C = pending->scores_NTC.size(2);
    const int state_len = core->model_config->state_len;

    openfish_decode_gpu(
        T, N, C,
        pending->scores_NTC.data_ptr(),
        OPENFISH_SCORE_F16, 1.0f, state_len,
        &core->decoder_opts,
        runner->gpubuf,
        &pending->moves,
        &pending->sequence,
        &pending->qstring,
        &ts->decode_stats,
        (void *)runner->decode_stream->stream(),
        pending->host_slot
    );

    pending->T = T;
    pending->decode_launched = true;
}

/* Block until queued decode finishes and resolve phase timers. Leaves host ptrs for write. */
static void overlap_sync_decode(
    runner_t* runner,
    runner_stat_t* ts,
    overlap_pending_t *pending
) {
    if (!pending->active || !pending->decode_launched) {
        return;
    }

    cudaError_t err = cudaStreamSynchronize(runner->decode_stream->stream());
    if (err != cudaSuccess) {
        ERROR("cudaStreamSynchronize(decode) failed: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Must finish before the next decode launch (async timing state is one-shot). */
    openfish_decode_stats_finish(&ts->decode_stats);
}

/* Copy synced host results into chunks; host ring slot may be reused after this returns. */
static void overlap_write_decode(overlap_pending_t *pending) {
    if (!pending->active || !pending->decode_launched) {
        return;
    }

    write_decode_results(pending->chunks, pending->T, pending->moves, pending->sequence, pending->qstring);
    pending->moves = nullptr;
    pending->sequence = nullptr;
    pending->qstring = nullptr;
    pending->decode_launched = false;
    pending->chunks.clear();
    pending->scores_NTC = torch::Tensor();
    pending->active = false;
}

/* Sync + write (flush path when there is no following batch to overlap the write with). */
static void overlap_finalize_decode(
    runner_t* runner,
    runner_stat_t* ts,
    overlap_pending_t *pending
) {
    overlap_sync_decode(runner, ts, pending);
    overlap_write_decode(pending);
}

static void basecall_chunks_overlap(
    const core_t* core,
    const int runner_idx,
    const std::vector<basecall_chunk_t *> &chunks,
    overlap_pending_t *pending
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    runner_t* runner = (*core->runners)[runner_idx];
    auto chunk_size = core->chunk_size;
    const int slot = runner->overlap_slot;

    c10::DeviceGuard device_guard(runner->tensor_opts.device());
    torch::InferenceMode inference_guard;

    LOG_DEBUG("%s", "accepting chunks (overlap / P5-lite)");
    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) {
        accept_chunk(i, chunks[i], runner, chunk_size, slot);
    }
    ts->time_accept += realtime();

    ts->time_basecall -= realtime();

    // 1) Ensure previous batch's decode is queued (usually already launched last iteration).
    const bool had_pending = pending->active;
    if (had_pending) {
        ts->time_decode -= realtime();
        overlap_launch_decode(core, runner, ts, pending);
    }

    // 2) Infer current batch; overlaps in-flight decode of previous on decode_stream.
    ts->time_infer -= realtime();
    torch::Tensor scores_NTC;
    {
        c10::cuda::CUDAStreamGuard infer_guard(*runner->infer_stream);
        const int64_t N = (int64_t)chunks.size();
        torch::Tensor slot_input = (core->opt.flag & SLORADO_FIXED_C_BATCH)
                                  ? runner_input_slot(runner, slot)
                                  : runner_input_slot(runner, slot).narrow(0, 0, N);
        auto scores = runner->module->forward(slot_input.to(runner->tensor_opts.device()));
        scores_NTC = scores.contiguous(); // NTC for openfish
        cudaError_t err = cudaEventRecord(runner->infer_event[slot], runner->infer_stream->stream());
        if (err != cudaSuccess) {
            ERROR("cudaEventRecord failed: %s", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    ts->time_infer += realtime();

    // 3) P5-lite: sync previous decode → launch current decode into the other host slot →
    //    write previous results on the CPU while current decode runs on the GPU.
    overlap_pending_t prev{};
    prev.active = false;
    if (had_pending) {
        overlap_sync_decode(runner, ts, pending);
        prev = std::move(*pending);
        /* Scores no longer needed after D2H; drop before launching the next decode. */
        prev.scores_NTC = torch::Tensor();
        pending->active = false;
        pending->decode_launched = false;
        pending->moves = nullptr;
        pending->sequence = nullptr;
        pending->qstring = nullptr;
        pending->scores_NTC = torch::Tensor();
        pending->chunks.clear();
    }

    pending->chunks = chunks;
    pending->scores_NTC = scores_NTC;
    pending->slot = slot;
    pending->host_slot = had_pending ? (prev.host_slot ^ 1) : 0;
    pending->active = true;
    pending->decode_launched = false;
    pending->moves = nullptr;
    pending->sequence = nullptr;
    pending->qstring = nullptr;

    overlap_launch_decode(core, runner, ts, pending);

    if (had_pending) {
        overlap_write_decode(&prev);
        ts->time_decode += realtime();
    }

    runner->overlap_slot = slot ^ 1;

    ts->time_basecall += realtime();
}
#endif

static void* pthread_single_basecall(void* voidargs) {
    model_thread_arg_t* args = (model_thread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    const size_t runner_idx = args->runner;
    const size_t start = args->start;
    const size_t end = args->end;
    opt_t opt = core->opt;
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    const size_t C = (size_t)opt.gpu_batch_size;
    /* streaming-sim: flush at flush_threshold if set and <= C, else pack full C-wide batches. */
    const size_t flush_thr = (opt.flush_threshold > 0 && (size_t)opt.flush_threshold <= C)
                             ? (size_t)opt.flush_threshold : C;

    std::vector<basecall_chunk_t *> chunks;
#ifdef USE_GPU
    overlap_pending_t pending{};
    pending.active = false;
    pending.decode_launched = false;
    pending.host_slot = 0;
    const bool use_overlap = runner->overlap_decode;
#else
    const bool use_overlap = false;
#endif

    /* Dispatch a (possibly partial) GPU batch of N=ch.size() real chunks. The GPU launch is always
       C-wide; slots N..C-1 are padded and their work discarded. Counted as load-imbalance waste. */
    auto dispatch = [&](std::vector<basecall_chunk_t *> &ch) {
        const size_t N = ch.size();
        ts->total_batches++;
        ts->total_chunks_processed += N;
        if (N < C) {
            ts->tail_batches++;
            ts->padded_slots += C - N;
        }
#ifdef USE_GPU
        if (use_overlap) {
            basecall_chunks_overlap(core, runner_idx, ch, &pending);
        } else
#endif
        {
            basecall_chunks(core, runner_idx, ch);
        }
    };

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        auto& db_chunks = (*db->basecall_chunks)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < db_chunks.size(); ++chunk_idx) {
            chunks.push_back(&db_chunks[chunk_idx]);

            if (chunks.size() >= flush_thr) {
                dispatch(chunks);
                chunks.clear();
            }
        }
    }

    // leftover chunks
    if (chunks.size() > 0) {
        dispatch(chunks);
    }

#ifdef USE_GPU
    if (use_overlap && pending.active) {
        runner_stat_t* ts = (*core->runner_stats)[runner_idx];
        c10::DeviceGuard device_guard(runner->tensor_opts.device());
        ts->time_basecall -= realtime();
        ts->time_decode -= realtime();
        // Last batch: no following infer to overlap with — launch + finalize serially.
        overlap_launch_decode(core, runner, ts, &pending);
        overlap_finalize_decode(runner, ts, &pending);
        ts->time_decode += realtime();
        ts->time_basecall += realtime();
    }
#endif

    pthread_exit(0);
}

static void mod_basecall_chunks(
    const core_t* core,
    const int runner_idx,
    const std::vector<mod_chunk_t *> &results
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    const auto chunk_size = core->modbase_config->context.chunk_size;
    const int kmer_size_per_sample = core->modbase_config->context.kmer_len * NUM_BASES;
    auto sequence_stride_ratio = core->modbase_config->general.stride_ratio();

    assert(core->modbase_config->is_chunked_input_model());

    ts->time_modcall -= realtime();
    for (size_t i = 0; i < results.size(); ++i) {
        mod_chunk_t *chunk = results[i];
        read_dat_t *read_dat = chunk->read_dat;

        const int64_t start = chunk->signal_offset;
        const int64_t end = std::min(start + chunk_size, static_cast<int64_t>(read_dat->scaled_signal.size(0)));
        const int64_t len = end - start;
        assert(start <= end);

        auto signal_chunk = read_dat->scaled_signal.index({at::indexing::Slice(start, end)});

        const std::int64_t kmer_start = start / sequence_stride_ratio;
        const std::int64_t kmer_end = end / sequence_stride_ratio;

        auto encoded_kmers_chunk = std::vector<int8_t>(
            read_dat->encoded_kmers.begin() + kmer_start * kmer_size_per_sample,
            read_dat->encoded_kmers.begin() + kmer_end * kmer_size_per_sample
        );

        if (len < chunk_size) {
            // Tile the signal tensor
            auto result = std::div(chunk_size, len);
            int n_tiles = result.quot;
            int n_overhang = result.rem;
            signal_chunk = at::concat({signal_chunk.repeat({n_tiles}),
                                       signal_chunk.index({at::indexing::Slice(0, n_overhang)})},
                                      -1);
            // Tile the kmer vector
            const int64_t original_size = static_cast<int64_t>(encoded_kmers_chunk.size());
            const int64_t extended_size = chunk_size * kmer_size_per_sample;
            encoded_kmers_chunk.resize(extended_size);

            for (int64_t i = original_size; i < extended_size; ++i) {
                encoded_kmers_chunk[i] = encoded_kmers_chunk[i % original_size];
            }
        }

        mod_accept_chunk(i, signal_chunk, encoded_kmers_chunk, core, runner_idx);
    }
    
    mod_call_chunks(core, results, runner_idx);
    ts->time_modcall += realtime();
}

static void* pthread_single_mod_basecall(void* voidargs) {
    model_thread_arg_t* args = (model_thread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    const size_t runner_idx = args->runner;
    const size_t start = args->start;
    const size_t end = args->end;
    opt_t opt = core->opt;

    std::vector<mod_chunk_t *> results;

    LOG_DEBUG("%s", "loading chunks");

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        auto& chunks = (*db->mod_chunks)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            results.push_back(&chunks[chunk_idx]);

            if (results.size() == (size_t)opt.gpu_batch_size) {
                mod_basecall_chunks(core, runner_idx, results);
                results.clear();
            }
        }
    }

    // leftover chunks
    if (results.size() > 0) {
        mod_basecall_chunks(core, runner_idx, results);
    }

    pthread_exit(0);
}

void basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = db->n_rec;
    int32_t num_threads = (*core->runners).size();
    int32_t step = (n_reads + num_threads - 1) / num_threads;

    // create threads
    pthread_t tids[num_threads];
    model_thread_arg_t pt_args[num_threads];
    int32_t t, ret;
    int32_t i = 0;
    // set the data structures
    for (t = 0; t < num_threads; t++) {
        pt_args[t].core = core;
        pt_args[t].db = db;
        pt_args[t].start = i;
        pt_args[t].runner = t;
        i += step;
        if (i > n_reads) {
            pt_args[t].end = n_reads;
        } else {
            pt_args[t].end = i;
        }
    }

    // create threads
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_basecall,
                                (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    double time_sync = 0;

    // pthread joining
    for (t = 0; t < num_threads; t++) {
        int ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
        if (t == 0) {
            time_sync -= realtime();
        }
        if (t == num_threads-1) {
            time_sync += realtime();
        }
    }

    core->time_sync += time_sync;
}


void mod_basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = db->n_rec;
    int32_t num_threads = (*core->mod_runners).size();
    int32_t step = (n_reads + num_threads - 1) / num_threads;

    pthread_t tids[num_threads];
    model_thread_arg_t pt_args[num_threads];
    int32_t t, ret;
    int32_t i = 0;
    // set the data structures
    for (t = 0; t < num_threads; t++) {
        pt_args[t].core = core;
        pt_args[t].db = db;
        pt_args[t].start = i;
        pt_args[t].runner = t;
        i += step;
        if (i > n_reads) {
            pt_args[t].end = n_reads;
        } else {
            pt_args[t].end = i;
        }
    }

    double time_sync = 0;

    LOG_DEBUG("%s", "starting mod basecall");

    // modbase call
    // create threads
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_mod_basecall,
                                (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    // pthread joining
    for (t = 0; t < num_threads; t++) {
        int ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
        if (t == 0) {
            time_sync -= realtime();
        }
        if (t == num_threads-1) {
            time_sync += realtime();
        }
    }

    core->time_sync += time_sync;
}