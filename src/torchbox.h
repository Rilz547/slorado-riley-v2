/** Riley Updates (Remove at the end)
 * @file torchbox.h
 * @lastmodified: Extended runner_t with dual input buffers plus infer/decode CUDA streams and events for depth-1 overlap.
 * @lastpatched: 2026-07-14

******************************************************************************/

#ifndef TORCHBOX_H
#define TORCHBOX_H

#include <torch/torch.h>
#include <cstdint>
#include "slorado.h"

#ifdef USE_GPU
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

// per read data
struct read_dat {
    torch::Tensor scaled_signal;

    // mod data
    const char *seq;

    std::vector<int8_t> encoded_kmers;
    std::array<std::vector<int64_t>, 4> per_base_hits_seq; // sequence indices for hits for each base (i.e. one per model)
    std::array<std::vector<int64_t>, 4> per_base_hits_sig; // signal indices for hits for each base (i.e. one per model)
    int64_t target_start;

    std::vector<uint8_t> base_mod_probs;
    std::vector<bool> base_mod_simplex_motif_hits;
};

struct runner {
    std::string device;
    torch::Tensor input_tensor;
    torch::Tensor input_tensor_alt; // second accept buffer when overlap_decode
    torch::TensorOptions tensor_opts;
    torch::nn::ModuleHolder<torch::nn::AnyModule> module{nullptr};
    bool overlap_decode = false;
    int overlap_slot = 0;
    int overlap_depth = 1; // 1 = v1.2; 2 = dual decode lanes
#ifdef USE_GPU
    int64_t device_idx;
    openfish_gpubuf_t *gpubuf = nullptr;      // decode lane 0
    openfish_gpubuf_t *gpubuf_alt = nullptr;  // decode lane 1 (depth 2 only)
    c10::cuda::CUDAStream *infer_stream = nullptr;
    c10::cuda::CUDAStream *decode_stream = nullptr;      // lane 0
    c10::cuda::CUDAStream *decode_stream_alt = nullptr;  // lane 1 (depth 2)
    cudaEvent_t infer_event[2] = {nullptr, nullptr};
    cudaEvent_t decode_event[2] = {nullptr, nullptr};
#endif

    // modbase stuff
    at::Tensor input_sigs;
    at::Tensor input_seqs;
};

#endif