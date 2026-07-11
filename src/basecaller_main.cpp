/**
 * @file basecaller_main.cpp
 * @brief entry point to basecaller_main
 * @author Hasindu Gamaarachchi (hasindu@unsw.edu.au)
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
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
 * @file basecaller_main.cpp
 * @lastmodified: Added LSRNN layer times in colour, and remembered to update this message lol.
 * @lastpatched: 2026-07-11

******************************************************************************/

#include <getopt.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <unordered_set>

#include <openfish/openfish_error.h>

#include "slorado.h"
#include "misc.h"
#include "error.h"

// add supported modbase models here
static const std::unordered_set<std::string> supported = std::unordered_set<std::string>({
    "5mCG_5hmCG@v3",
});

static inline bool is_modbase_supported(const char *mod) {
    return supported.find(std::string(mod)) != supported.end();
}

enum profile_colour_level {
    PROFILE_COLOUR_SUMMARY = 0,
    PROFILE_COLOUR_PIPELINE,
    PROFILE_COLOUR_RUNNERS,
    PROFILE_COLOUR_BASECALL,
    PROFILE_COLOUR_INFERENCE,
    PROFILE_COLOUR_CONV,
    PROFILE_COLOUR_RNN,
    PROFILE_COLOUR_CRF,
    PROFILE_COLOUR_DECODE,
    PROFILE_COLOUR_DECODE_DETAIL,
    PROFILE_COLOUR_DETAIL,
};

static bool profile_use_colour(void) {
    static int use_colour = -1;
    if (use_colour < 0) {
        use_colour = isatty(STDERR_FILENO) ? 1 : 0;
    }
    return use_colour != 0;
}

static const char *profile_colour_code(profile_colour_level level) {
    if (!profile_use_colour()) {
        return "";
    }
    switch (level) {
        case PROFILE_COLOUR_SUMMARY:       return "\033[1;36m";
        case PROFILE_COLOUR_PIPELINE:      return "\033[1;32m";
        case PROFILE_COLOUR_RUNNERS:       return "\033[1;33m";
        case PROFILE_COLOUR_BASECALL:      return "\033[1;35m";
        case PROFILE_COLOUR_INFERENCE:     return "\033[1;34m";
        case PROFILE_COLOUR_CONV:          return "\033[36m";
        case PROFILE_COLOUR_RNN:           return "\033[32m";
        case PROFILE_COLOUR_CRF:           return "\033[33m";
        case PROFILE_COLOUR_DECODE:        return "\033[1;91m";
        case PROFILE_COLOUR_DECODE_DETAIL: return "\033[95m";
        case PROFILE_COLOUR_DETAIL:        return "\033[2;37m";
        default:                           return "";
    }
}

static void profile_print(profile_colour_level level, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fputs(profile_colour_code(level), stderr);
    vfprintf(stderr, fmt, args);
    if (profile_use_colour()) {
        fputs(NO_COLOUR, stderr);
    }
    va_end(args);
}

static struct option long_options[] = {
    {"threads", required_argument, 0, 't'},         //0 number of threads [8]
    {"batchsize", required_argument, 0, 'K'},       //1 batchsize - number of reads loaded at once [4096]
    {"max-bytes", required_argument, 0, 'B'},       //2 batchsize - number of bytes loaded at once
    {"verbose", required_argument, 0, 'v'},         //3 verbosity level [1]
    {"help", no_argument, 0, 'h'},                  //4
    {"version", no_argument, 0, 'V'},               //5
    {"output", required_argument, 0, 'o'},          //6 output to a file [stdout]
    {"debug-break", required_argument, 0, 0},       //7 break after processing the first batch (used for debugging)
    {"profile-cpu", required_argument, 0, 0},       //8 perform section by section (used for profiling - for CPU only)
    {"accel",required_argument, 0, 0},              //9 accelerator //not used, can be reused for something elese
    {"chunk-size", required_argument, 0, 'c'},      //10 chunk size [12288]
    {"overlap", required_argument, 0, 'p'},         //11 overlap [150]
    {"device", required_argument, 0, 'x'},          //12 device [cpu]
    {"num-runners", required_argument, 0, 'r'},     //13 number of runners [1]
    {"emit-sam", no_argument, 0, 0},                //14 toggles emit sam
    {"gpu_batchsize", required_argument, 0, 'C'},   //15 gpu batchsize - number of chunks loaded at once [512]
    {"flash", required_argument, 0, 0},             //16 toggles flash attention when possible
    {"mod", required_argument, 0, 0},               //17 detect modified bases
    {0, 0, 0, 0}};


static inline void print_help_msg(FILE *fp_help, opt_t opt){
    fprintf(fp_help, "usage: slorado basecaller [model] [data]\n");
    fprintf(fp_help, "positional arguments:\n");
    fprintf(fp_help, "  model FILE                  the basecaller model to run.\n");
    fprintf(fp_help, "  data FILE                   the data directory.\n");
    fprintf(fp_help, "\nbasic options:\n");
    fprintf(fp_help, "  -t INT                      number of processing threads [%d]\n", opt.num_thread);
    fprintf(fp_help, "  -K INT                      batch size (max number of reads loaded at once) [%d]\n", opt.batch_size);
    fprintf(fp_help, "  -C INT                      gpu batch size (max number of chunks loaded at once) [%d]\n", opt.gpu_batch_size);
    fprintf(fp_help, "  -B FLOAT[K/M/G]             max number of bytes loaded at once [%.1fM]\n", opt.batch_size_bytes/(float)(1000*1000));
    fprintf(fp_help, "  -o FILE                     output to file [%s]\n", opt.out_path);
    fprintf(fp_help, "  -c INT                      chunk size [%zu]\n", opt.chunk_size);
    fprintf(fp_help, "  -p INT                      overlap [%d]\n", opt.overlap);
    fprintf(fp_help, "  -x DEVICE                   specify device [%s]\n", opt.device);
    fprintf(fp_help, "  -h                          shows help message and exits\n");
    fprintf(fp_help, "  --flash=yes|no              use flash attention for better performance [%s]\n", (opt.flag & SLORADO_FLASH) ? "yes" : "no");
    fprintf(fp_help, "  --mod STR                   detect modified bases (5mCG_5hmCG@v3) [%s]\n", opt.mod ? opt.mod : "NULL");
    fprintf(fp_help, "  --verbose INT               verbosity level [%d]\n",(int)get_log_level());
    fprintf(fp_help, "  --version                   print version\n");
    fprintf(fp_help, "\ndebug options:\n");
    fprintf(fp_help, "  --debug-break INT           break after processing the specified no. of batches\n");
    fprintf(fp_help, "  --emit-sam                  emits sam output format\n");
    fprintf(fp_help, "  --profile-cpu=yes|no        process section by section (used for profiling on CPU)\n");
}

int basecaller_main(int argc, char* argv[]) {
    double realtime0 = realtime();
    double a, b;

    const char* optstring = "t:B:K:C:v:o:x:r:p:c:hV";

    int longindex = 0;
    int32_t c = -1;

    char *data = NULL;
    char *model = NULL;

    FILE *fp_help = stderr;

    opt_t opt;
    init_opt(&opt); // initialise options to defaults

    // parse the user args
    while ((c = getopt_long(argc, argv, optstring, long_options, &longindex)) >= 0) {
        if (c == 'B') {
            opt.batch_size_bytes = mm_parse_num(optarg);
            if(opt.batch_size_bytes<=0){
                ERROR("%s","Maximum number of bytes should be larger than 0.");
                exit(EXIT_FAILURE);
            }
        } else if (c == 'K') {
            opt.batch_size = atoi(optarg);
            if (opt.batch_size < 1) {
                ERROR("Batch size should larger than 0. You entered %d",opt.batch_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'C') {
            opt.gpu_batch_size = atoi(optarg);
            if (opt.gpu_batch_size < 1) {
                ERROR("Batch size should larger than 0. You entered %d",opt.gpu_batch_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 't') {
            opt.num_thread = atoi(optarg);
            if (opt.num_thread < 1) {
                ERROR("Number of threads should larger than 0. You entered %d", opt.num_thread);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'v') {
            int v = atoi(optarg);
            set_log_level((enum log_level_opt)v);
            set_openfish_log_level((enum openfish_log_level_opt)v);
        } else if (c == 'x') {
            opt.device = optarg;
        } else if (c == 'c') {
            opt.chunk_size = atoi(optarg);
            if (opt.chunk_size < 1) {
                ERROR("Chunk size should larger than 0. You entered %zu", opt.chunk_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'p') {
            opt.overlap = atoi(optarg);
            if (opt.overlap < 1) {
                ERROR("Overlap should larger than 0. You entered %d", opt.overlap);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'o') {
            opt.out_path = optarg;
            opt.out = fopen(opt.out_path, "w");
            if (opt.out == NULL) {
                ERROR("Error in opening output file %s: %s\n", opt.out_path, strerror(errno));
                exit(EXIT_FAILURE);
            }
        }  else if (c == 'V') {
            fprintf(stdout,"slorado %s\n",SLORADO_VERSION);
            exit(EXIT_SUCCESS);
        } else if (c == 'h') {
            fp_help = stdout;
        } else if (c == 0 && longindex == 7) { // debug break
            opt.debug_break = atoi(optarg);
        } else if (c == 0 && longindex == 8) { // sectional benchmark todo : warning for gpu mode
            yes_or_no(&opt.flag, SLORADO_PRF, long_options[longindex].name, optarg, 1);
        } else if (c == 0 && longindex == 14) { // emit fastq
            opt.flag |= SLORADO_SAM;
        } else if (c == 0 && longindex == 16) { // flash attention
            yes_or_no(&opt.flag, SLORADO_FLASH, long_options[longindex].name, optarg, 1);
        } else if (c == 0 && longindex == 17) { // flash attention
            opt.mod = optarg;
        }
    }

    size_t max_input_tensor_len = 10000 * 6000; // 10k chunk len, 6k gpu batch size

    if ((size_t)opt.chunk_size * opt.gpu_batch_size > max_input_tensor_len) {
        ERROR("Your input tensor size: %zu (chunk_size * gpu_batch_size) exceeds maximum allowed size: %zu", (size_t)opt.chunk_size * opt.gpu_batch_size, max_input_tensor_len);
        exit(EXIT_FAILURE);
    }

    if ((size_t)opt.overlap >= opt.chunk_size) {
        ERROR("Your overlap: %d should be lesser than chunk size: %ld", opt.overlap, opt.chunk_size);
        exit(EXIT_FAILURE);
    }

    // Incorrect number of arguments given
    if (argc - optind != 2 || fp_help == stdout) {
        print_help_msg(fp_help, opt);
        if (fp_help == stdout) {
            exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
    }

    model = argv[optind++];

    if (opt.mod != NULL && !is_modbase_supported(opt.mod)) {
        std::string error_msg = "unsupported modbase model \"" + std::string(opt.mod) + "\"curent supported modbase models are: ";
        for (const auto &s : supported) {
            error_msg += s + ", ";
        }
        ERROR("%s", error_msg.c_str());
        exit(EXIT_FAILURE);
    }

    if (model == NULL) {
        print_help_msg(fp_help, opt);
        if (fp_help == stdout) {
            exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
    }

    data = argv[optind];

    if (data == NULL) {
        print_help_msg(fp_help, opt);
        if(fp_help == stdout){
            exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
    }

    // print summary
    fprintf(stderr,"\nslorado base-caller version %s\n", SLORADO_VERSION);
    fprintf(stderr,"model path:         %s\n", model);
    fprintf(stderr,"input path:         %s\n", data);
    fprintf(stderr,"output path:        %s\n", opt.out_path == NULL ? "stdout" : opt.out_path);
    fprintf(stderr,"device:             %s\n", opt.device);
    fprintf(stderr,"chunk size:         %zu\n", opt.chunk_size);
    fprintf(stderr,"read batch size:    %d\n", opt.batch_size);
    fprintf(stderr,"gpu batch size:     %d\n", opt.gpu_batch_size);
    fprintf(stderr,"no. threads:        %d\n", opt.num_thread);
    fprintf(stderr,"overlap:            %d\n", opt.overlap);
    fprintf(stderr, "\n");

/////////////////////////////////////////////////////////////////////////////

    // initialise the core data structure
    core_t* core = init_core(data, opt, model, realtime0);

    int32_t counter = 0;

    // initialise a databatch
    db_t* db = init_db(core);

    ret_status_t status = {core->opt.batch_size, core->opt.batch_size_bytes};
    while (status.num_reads >= core->opt.batch_size || status.num_bytes>=core->opt.batch_size_bytes) {
        // load a databatch
        status = load_db(core, db);

        fprintf(stderr, "[%s::%.3f*%.2f] %d Entries (%.1fM bytes) loaded\n", __func__,
                realtime() - realtime0, cputime() / (realtime() - realtime0),
                status.num_reads,status.num_bytes/(1000.0*1000.0));

        // process a databatch
        process_db(core, db);

        fprintf(stderr, "[%s::%.3f*%.2f] %d Entries (%.1fM bytes) processed\n", __func__,
                realtime() - realtime0, cputime() / (realtime() - realtime0),
                status.num_reads,status.num_bytes/(1000.0*1000.0));

        // output print
        output_db(core, db);

        // free temporary
        a = realtime();
        free_db_tmp(db);
        b = realtime();
        core->time_free_db += b-a;

        if (opt.debug_break == counter) {
            break;
        }
        counter++;
    }

    // free the databatch
    a = realtime();
    free_db(db);
    b = realtime();
    core->time_free_db += b-a;

    profile_print(PROFILE_COLOUR_SUMMARY, "\n[%s] total entries: %ld", __func__, (long)core->total_reads);
    profile_print(PROFILE_COLOUR_SUMMARY, "\n[%s] total bytes: %.1f M", __func__, core->sum_bytes/(float)(1000*1000));

    profile_print(PROFILE_COLOUR_SUMMARY, "\n[%s] model initialization: %.3f sec", __func__, core->time_init_runners);
    profile_print(PROFILE_COLOUR_SUMMARY, "\n[%s] data loading: %.3f sec", __func__, core->time_load_db);
    profile_print(PROFILE_COLOUR_PIPELINE, "\n[%s] data processing: %.3f sec", __func__, core->time_process_db);
    profile_print(PROFILE_COLOUR_PIPELINE, "\n[%s]     - parse: %.3f sec", __func__, core->time_parse);
    profile_print(PROFILE_COLOUR_PIPELINE, "\n[%s]     - preprocess: %.3f sec", __func__, core->time_preproc);
    profile_print(PROFILE_COLOUR_RUNNERS, "\n[%s]     - runners: %.3f sec", __func__, core->time_runners);
    profile_print(PROFILE_COLOUR_RUNNERS, "\n[%s]          - synchronisation: %.3f sec", __func__, core->time_sync);

    auto runner_stats = *core->runner_stats;
    for (size_t i = 0; i < runner_stats.size(); ++i) {
        profile_print(PROFILE_COLOUR_RUNNERS, "\n[%s]          - model runner [%zu]: %.3f sec", __func__, i,
            runner_stats[i]->time_basecall +
            runner_stats[i]->time_accept +
            runner_stats[i]->time_modcall
        );
        profile_print(PROFILE_COLOUR_BASECALL, "\n[%s]             - accept: %.3f sec", __func__, runner_stats[i]->time_accept);
        profile_print(PROFILE_COLOUR_BASECALL, "\n[%s]             - basecall: %.3f sec", __func__, runner_stats[i]->time_basecall);
        profile_print(PROFILE_COLOUR_INFERENCE, "\n[%s]                 - inference: %.3f sec", __func__, runner_stats[i]->time_infer);
        if (core->model_config->tx != NULL) { // tx
            tx_stats_t *model_stats = (tx_stats_t *)runner_stats[i]->model_stats;
            profile_print(PROFILE_COLOUR_CONV, "\n[%s]                     - conv_stack: %.3f sec", __func__, model_stats->time_conv_stack);
            profile_print(PROFILE_COLOUR_RNN, "\n[%s]                     - tx_encoder: %.3f sec", __func__, model_stats->time_tx_encoder);
            profile_print(PROFILE_COLOUR_RNN, "\n[%s]                         - self_attn: %.3f sec", __func__, model_stats->time_self_attn);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             - mm: %.3f sec", __func__, model_stats->time_mm);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             - rotary_emb: %.3f sec", __func__, model_stats->time_rotary_emb);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             - sdp_attn: %.3f sec", __func__, model_stats->time_sdp_attn);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             - out_proj: %.3f sec", __func__, model_stats->time_out_proj);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                         - norm1: %.3f sec", __func__, model_stats->time_norm1);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                         - ff: %.3f sec", __func__, model_stats->time_ff);
            profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                         - norm2: %.3f sec", __func__, model_stats->time_norm2);
            profile_print(PROFILE_COLOUR_CONV, "\n[%s]                     - tx_decoder: %.3f sec", __func__, model_stats->time_tx_decoder);
            profile_print(PROFILE_COLOUR_CRF, "\n[%s]                     - crf: %.3f sec", __func__, model_stats->time_crf);
        } else { // lstm
            lstm_stats_t *model_stats = (lstm_stats_t *)runner_stats[i]->model_stats;
            profile_print(PROFILE_COLOUR_CONV, "\n[%s]                     - conv_stack: %.3f sec", __func__, model_stats->time_conv_stack);
            if (model_stats->conv_input_n > 0) {
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                         - input [N,C,T]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->conv_input_n, (long)model_stats->conv_input_c,
                        (long)model_stats->conv_input_t);
            }
            for (int c = 0; c < MAX_CONV_LAYERS && model_stats->time_conv[c] > 0.0; ++c) {
                profile_print(PROFILE_COLOUR_CONV, "\n[%s]                         - conv[%d]: %.3f sec", __func__, c,
                        model_stats->time_conv[c]);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             in  [N,C,T]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->conv_in_n[c], (long)model_stats->conv_in_c[c],
                        (long)model_stats->conv_in_t[c]);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             out [N,C,T]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->conv_n[c], (long)model_stats->conv_c[c],
                        (long)model_stats->conv_t[c]);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             op: %.3f sec  act: %.3f sec", __func__,
                        model_stats->time_conv_op[c], model_stats->time_conv_act[c]);
            }
            if (model_stats->time_conv_transpose > 0.0) {
                profile_print(PROFILE_COLOUR_CONV, "\n[%s]                         - transpose: %.3f sec", __func__,
                        model_stats->time_conv_transpose);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             in  [N,C,T]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->transpose_in_n, (long)model_stats->transpose_in_c,
                        (long)model_stats->transpose_in_t);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             out [N,T,C]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->transpose_out_n, (long)model_stats->transpose_out_t,
                        (long)model_stats->transpose_out_c);
            }
            profile_print(PROFILE_COLOUR_RNN, "\n[%s]                     - rnns: %.3f sec", __func__, model_stats->time_rnns);
            for (int r = 0; r < MAX_LSTM_LAYERS && model_stats->time_rnn[r] > 0.0; ++r) {
                profile_print(PROFILE_COLOUR_RNN, "\n[%s]                         - rnn[%d] (%s): %.3f sec", __func__, r,
                        (r % 2 == 0) ? "reverse" : "forward", model_stats->time_rnn[r]);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             in  [N,T,C]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->rnn_in_n[r], (long)model_stats->rnn_in_t[r],
                        (long)model_stats->rnn_in_c[r]);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             out [N,T,C]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->rnn_n[r], (long)model_stats->rnn_t[r],
                        (long)model_stats->rnn_c[r]);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             flip: %.3f sec  lstm: %.3f sec", __func__,
                        model_stats->time_rnn_flip[r], model_stats->time_rnn_lstm[r]);
            }
            if (model_stats->time_rnn_out_flip > 0.0) {
                profile_print(PROFILE_COLOUR_RNN, "\n[%s]                         - rnn_out_flip: %.3f sec", __func__,
                        model_stats->time_rnn_out_flip);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                             out [N,T,C]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->rnn_out_flip_n, (long)model_stats->rnn_out_flip_t,
                        (long)model_stats->rnn_out_flip_c);
            }
            profile_print(PROFILE_COLOUR_CRF, "\n[%s]                     - crf_1: %.3f sec", __func__, model_stats->time_crf_1);
            if (model_stats->crf1_in_n > 0) {
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                         in  [N,T,C]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->crf1_in_n, (long)model_stats->crf1_in_t,
                        (long)model_stats->crf1_in_c);
                profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]                         out [N,T,C]: [%ld, %ld, %ld]", __func__,
                        (long)model_stats->crf1_out_n, (long)model_stats->crf1_out_t,
                        (long)model_stats->crf1_out_c);
            }
            profile_print(PROFILE_COLOUR_CRF, "\n[%s]                     - crf_2: %.3f sec", __func__, model_stats->time_crf_2);
            profile_print(PROFILE_COLOUR_CRF, "\n[%s]                     - clamp: %.3f sec", __func__, model_stats->time_clamp);
        }
        profile_print(PROFILE_COLOUR_DECODE, "\n[%s]                 - decode: %.3f sec", __func__, runner_stats[i]->time_decode);
        if (runner_stats[i]->decode_stats.batch_size > 0) {
            openfish_decode_stats_t *ds = &runner_stats[i]->decode_stats;
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - scores [N,T,C]: [%d, %d, %d]", __func__,
                    ds->batch_size, ds->n_timesteps, ds->n_channels);
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - bwd_scan: %.3f sec", __func__, ds->time_bwd_scan);
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - beam_search: %.3f sec", __func__, ds->time_beam_search);
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - fwd_post_scan: %.3f sec", __func__, ds->time_fwd_post_scan);
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - qual_data: %.3f sec", __func__, ds->time_qual_data);
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - gen_sequence: %.3f sec", __func__, ds->time_gen_sequence);
            profile_print(PROFILE_COLOUR_DECODE_DETAIL, "\n[%s]                     - d2h_copy: %.3f sec", __func__, ds->time_d2h_copy);
        }
        profile_print(PROFILE_COLOUR_BASECALL, "\n[%s]             - modcall: %.3f sec", __func__, runner_stats[i]->time_modcall);
        // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]             - total data points copied: %lu", __func__, runner_stats[i]->total_dp);
    }
    profile_print(PROFILE_COLOUR_PIPELINE, "\n[%s]     - postprocess: %.3f sec", __func__, core->time_postproc);
    profile_print(PROFILE_COLOUR_PIPELINE, "\n[%s]     - mod_preprocess: %.3f sec", __func__, core->time_preproc_mod);
    // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]         - seq_to_sig_map: %.3f sec", __func__, core->time_seq_to_sig_map);
    // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]         - seq_to_ints: %.3f sec", __func__, core->time_seq_to_ints);
    // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]         - populate_hits_sig: %.3f sec", __func__, core->time_populate_hits_sig);
    // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]         - populate_signal: %.3f sec", __func__, core->time_populate_signal);
    // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]         - get_minimal_encoding_skips: %.3f sec", __func__, core->time_get_minimal_encoding_skips);
    // profile_print(PROFILE_COLOUR_DETAIL, "\n[%s]         - populate_encoded_kmer: %.3f sec", __func__, core->time_populate_encoded_kmer);
    profile_print(PROFILE_COLOUR_PIPELINE, "\n[%s]     - mod_postprocess: %.3f sec", __func__, core->time_postproc_mod);
    profile_print(PROFILE_COLOUR_SUMMARY, "\n[%s] data output: %.3f sec", __func__, core->time_output);
    profile_print(PROFILE_COLOUR_SUMMARY, "\n[%s] data free: %.3f sec", __func__, core->time_free_db);
    fprintf(stderr,"\n");

    // free the core data structure
    free_core(core, opt);

    if (opt.out != stdout) {
        fclose(opt.out);
    }

    return 0;
}
