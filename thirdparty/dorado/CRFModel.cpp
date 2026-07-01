/** Riley Updates (Remove at the end)
 * @file CRFModel.cpp
 * @lastmodified: Added even more model stats.
 * @lastpatched: 2026-07-01

******************************************************************************/

#include <math.h>
#include <string>

#include "CRFModel.h"
#include "error.h"
#include "tensor_chunk_utils.h"
#include "misc.h"

#ifdef USE_GPU
#include <c10/cuda/CUDAStream.h>
#endif

using namespace torch::nn;

ConvStackImpl::ConvStackImpl(const std::vector<ConvParams> &layer_params, lstm_stats_t *stats)
        : model_stats(stats) {
    for (size_t i = 0; i < layer_params.size(); ++i) {
        layers.emplace_back(layer_params[i]);
        auto &layer = layers.back();
        auto opts = Conv1dOptions(layer.params.insize, layer.params.size, layer.params.winlen)
            .stride(layer.params.stride)
            .padding(layer.params.winlen / 2);
        layer.conv = register_module(std::string("conv") + std::to_string(i + 1), Conv1d(opts));
    }
}

torch::Tensor ConvStackImpl::forward(torch::Tensor x) {
    // Input x is [N, C_in, T_in], contiguity optional
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();

    for (size_t i = 0; i < layers.size(); ++i) {
        auto &layer = layers[i];
        double a = realtime();
        x = layer.conv(x);
        if (layer.params.activation == Activation::SWISH) {
            torch::silu_(x);
        } else if (layer.params.activation == Activation::SWISH_CLAMP) {
            torch::silu_(x).clamp_(c10::nullopt, 3.5f);
        } else if (layer.params.activation == Activation::TANH) {
            x.tanh_();
        } else {
            ERROR("%s", "Unrecognised activation function id.");
        }
        if (on_gpu) torch::cuda::synchronize(dev_idx);
        double b = realtime();
        if (model_stats && i < MAX_CONV_LAYERS) {
            if (model_stats->conv_n[i] == 0) {
                model_stats->conv_n[i] = x.size(0);
                model_stats->conv_c[i] = x.size(1);
                model_stats->conv_t[i] = x.size(2);
            }
            model_stats->time_conv[i] += b - a;
            model_stats->time_conv_stack += b - a;
        }
    }

    double a = realtime();
    x = x.transpose(1, 2);
    if (on_gpu) torch::cuda::synchronize(dev_idx);
    double b = realtime();
    if (model_stats) {
        model_stats->time_conv_transpose += b - a;
        model_stats->time_conv_stack += b - a;
    }

    // Output is [N, T_out, C_out], non-contiguous
    return x;
}

ConvStackImpl::ConvLayer::ConvLayer(const ConvParams &conv_params) : params(conv_params) {}

LinearCRFImpl::LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale) : bias(bias_) {
    linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
    if (tanh_and_scale) {
        activation = register_module("activation", Tanh());
    }
};

torch::Tensor LinearCRFImpl::forward(const torch::Tensor &x) {
    // Input x is [N, T, C], contiguity optional
    auto scores = linear(x);
    if (activation) {
        scores = activation(scores) * scale;
    }

    // Output is [N, T, C], contiguous
    return scores;
}

LSTMStackImpl::LSTMStackImpl(int num_layers_, int size, lstm_stats_t *stats)
        : layer_size(size), num_layers(num_layers_), model_stats(stats) {
    // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
    const auto lstm_opts = LSTMOptions(size, size).batch_first(true);
    for (int i = 0; i < num_layers_; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        rnns.emplace_back(register_module(label, LSTM(lstm_opts)));
    }
};

torch::Tensor LSTMStackImpl::forward(torch::Tensor x) {
    // Input is [N, T, C], contiguity optional
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();

    for (size_t i = 0; i < rnns.size(); ++i) {
        double a = realtime();
        auto flipped = x.flip(1);
        if (on_gpu) torch::cuda::synchronize(dev_idx);
        double b = realtime();

        double a2 = realtime();
        x = std::get<0>(rnns[i](flipped));
        if (on_gpu) torch::cuda::synchronize(dev_idx);
        double b2 = realtime();

        if (model_stats && i < MAX_LSTM_LAYERS) {
            if (model_stats->rnn_n[i] == 0) {
                model_stats->rnn_n[i] = x.size(0);
                model_stats->rnn_t[i] = x.size(1);
                model_stats->rnn_c[i] = x.size(2);
            }
            model_stats->time_rnn_flip[i] += b - a;
            model_stats->time_rnn_lstm[i] += b2 - a2;
            model_stats->time_rnn[i] += (b - a) + (b2 - a2);
            model_stats->time_rnns += (b - a) + (b2 - a2);
        }
    }

    // Output is [N, T, C], contiguous
    return (rnns.size() & 1) ? x.flip(1) : x;
}

ClampImpl::ClampImpl(float _min, float _max, bool _active)
        : active(_active), min(_min), max(_max) {}

torch::Tensor ClampImpl::forward(torch::Tensor x) {
    if (active) {
        x.clamp_(min, max);
    }
    return x;
}

CRFModelImpl::CRFModelImpl(const CRFModelConfig &config, lstm_stats_t *stats) : model_stats(stats) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    convs = register_module("convs", ConvStack(cv, model_stats));
    rnns = register_module("rnns", LSTMStack(5, lstm_size, model_stats));

    if (config.has_out_features) {
        // The linear layer is decomposed into 2 matmuls.
        const int decomposition = config.out_features;
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, true, false));
        linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        has_linear2 = true;
        has_clamp = true;
        encoder = Sequential(convs, rnns, linear1, linear2, clamp1);
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x model without linear decomposition
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        has_clamp = true;
        encoder = Sequential(convs, rnns, linear1, clamp1);
    } else {
        // Pre v4 model
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true));
        encoder = Sequential(convs, rnns, linear1);
    }
}

void CRFModelImpl::load_state_dict(const std::vector<torch::Tensor> &weights) {
    module_load_state_dict(*this, weights);
}

// Moved logic into LSTMStackImpl to get individual layer timings.
torch::Tensor CRFModelImpl::forward(const torch::Tensor &x) {
    torch::Tensor h;
    double a, b;

    h = convs->forward(x);

    h = rnns->forward(h);

    a = realtime();
    h = linear1->forward(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    if (model_stats) model_stats->time_crf_1 += b - a;

    if (has_linear2) {
        a = realtime();
        h = linear2->forward(h);
        if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
        b = realtime();
        if (model_stats) model_stats->time_crf_2 += b - a;
    }

    if (has_clamp) {
        a = realtime();
        h = clamp1->forward(h);
        if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
        b = realtime();
        if (model_stats) model_stats->time_clamp += b - a;
    }

    // Output is [N, T, C]
    return h;
}

std::vector<torch::Tensor> load_lstm_model_weights(const std::string &dir,
                                                  bool decomposition,
                                                  bool bias) {
    auto tensors = std::vector<std::string>{
        "0.conv.weight.tensor",      "0.conv.bias.tensor",

        "1.conv.weight.tensor",      "1.conv.bias.tensor",

        "2.conv.weight.tensor",      "2.conv.bias.tensor",

        "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
        "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

        "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
        "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

        "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
        "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

        "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
        "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

        "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
        "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

        "9.linear.weight.tensor"
    };

    if (bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, lstm_stats_t *model_stats) {
    auto model = CRFModel(model_config, model_stats);
    auto state_dict = load_lstm_model_weights(model_config.model_path, model_config.has_out_features, model_config.bias);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}