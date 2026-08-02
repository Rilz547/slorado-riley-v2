/**
 * @file slorado.c
 * @brief common functions for slorado
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

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slorado.h"
#include "misc.h"
#include "error.h"

#include "basecall.h"
#include "writer.h"

#include <sys/wait.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

void init_runners(core_t* core, opt_t *opt, char *model);
void free_runners(core_t *core);
void init_cascade_hac_runners(core_t *core, char *hac_model);
void free_cascade_hac_runners(core_t *core);
void cascade_park_fast_runners(core_t *core);
void cascade_unpark_fast_runners(core_t *core);
void preprocess_signal(core_t* core, db_t* db, int32_t i);
void stitch_chunks(db_t *basecall_db, size_t i, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, size_t len_raw_signal, int model_stride);
void free_read_dat(read_dat_t *read_dat);
void preprocess_modbase(core_t *core, db_t *db, int32_t i);
void postprocess_modbase(core_t *core, db_t *db, int32_t i);

static float cascade_mean_phred(const std::string &qstring) {
    if (qstring.empty()) {
        return 0.f;
    }
    double sum = 0.0;
    for (unsigned char c : qstring) {
        sum += (double)((int)c - 33);
    }
    return (float)(sum / (double)qstring.size());
}

/* initialise the core data structure */
core_t* init_core(char *slow5file, opt_t opt, char *model, double realtime0) {
    core_t* core = (core_t*)calloc(1, sizeof(core_t));
    MALLOC_CHK(core);
    core->opt = opt;

    core->realtime0 = realtime0;

    core->sp = slow5_open(slow5file, "r");
    if (core->sp == NULL) {
        VERBOSE("Error opening SLOW5 file %s\n", slow5file);
        exit(EXIT_FAILURE);
    }

    // modbase stuff
    if (opt.mod != NULL) {
        INFO("%s", "modification calling detected, output will be in SAM format");
        core->opt.flag |= SLORADO_SAM;
        
        LOG_TRACE("%s", "loading modbase configs...");
        auto model_str = std::string(model);
        if (model_str.back() == '/') {
            model_str.pop_back(); // remove trailing slash if exists
        }
        auto modbase_config_path = model_str + "_" + opt.mod;
        ModBaseModelConfig modbase_config = load_modbase_model_config(modbase_config_path.c_str());
        auto configs = std::vector<ModBaseModelConfig>({modbase_config});
        ModBaseInfo modbase_info = get_modbase_info(configs);
        core->modbase_config = new ModBaseModelConfig(modbase_config);
        core->modbase_info = new ModBaseInfo(modbase_info);
        LOG_TRACE("%s", "modbase config loaded");
    }

    CRFModelConfig model_config;
    if (is_tx_model_config(model)) {
        model_config = load_tx_model_config(model);
    } else {
        model_config = load_lstm_model_config(model);
    }
    model_config.model_path = std::string(model);
    model_config.sample_type = get_sample_type_from_model_name(model_config.model_path);

    core->model_stride = static_cast<size_t>(model_config.stride);
    core->chunk_size = opt.chunk_size - (opt.chunk_size % core->model_stride);

    core->decoder_opts = openfish_decoder_default_opts();
    core->decoder_opts.q_shift = model_config.qbias;
    core->decoder_opts.q_scale = model_config.qscale;

    core->model_config = new CRFModelConfig(model_config);
    LOG_TRACE("%s", "model config loaded");

    core->time_init_runners -= realtime();
    init_runners(core, &opt, model);
    if (opt.flag & SLORADO_CASCADE) {
        if (opt.cascade_hac == NULL) {
            ERROR("%s", "--cascade=yes requires --cascade-hac=PATH");
            exit(EXIT_FAILURE);
        }
        if (opt.mod != NULL) {
            ERROR("%s", "cascade does not support --mod in v1");
            exit(EXIT_FAILURE);
        }
        // Lazy-load HAC per promote pass so FAST+HAC are never both in VRAM.
        core->cascade_fast_path = model;
        core->cascade_hac_path = opt.cascade_hac;
        if (opt.cascade_log != NULL) {
            core->cascade_log_fp = fopen(opt.cascade_log, "w");
            if (core->cascade_log_fp == NULL) {
                ERROR("Error opening cascade log %s: %s", opt.cascade_log, strerror(errno));
                exit(EXIT_FAILURE);
            }
            fprintf(core->cascade_log_fp, "read_id\tmean_q\tmodel\n");
            fflush(core->cascade_log_fp);
        }
        INFO("cascade enabled: FAST scout → HAC worst %.1f%% by mean_q (HAC lazy-load)",
             100.0 * (double)opt.cascade_force_frac);
    }
    core->time_init_runners += realtime();
    LOG_DEBUG("%s", "successfully initialized runners");

    core->sum_bytes=0;
    core->total_reads=0; // total number mapped entries in the bam file (after filtering based on flags, mapq etc)

    return core;
}


/* free the core data structure */
void free_core(core_t* core, opt_t opt) {
    free_cascade_hac_runners(core);
    free_runners(core);

    if (core->cascade_log_fp != nullptr) {
        fclose(core->cascade_log_fp);
        core->cascade_log_fp = nullptr;
    }
    delete core->cascade_read_mask;

    slow5_close(core->sp);
    delete core->runners;
    delete core->mod_runners;
    delete core->runner_stats;
    delete core->model_config;
    delete core->hac_model_config;

    if (core->modbase_config != NULL) {
        delete core->modbase_config;
        delete core->modbase_info;
    }
    free(core);
}

/* initialise a data batch */
db_t* init_db(core_t* core) {
    db_t* db = (db_t*)(malloc(sizeof(db_t)));
    MALLOC_CHK(db);

    db->capacity_rec = core->opt.batch_size;
    db->n_rec = 0;

    db->mem_records = (char **)(calloc(db->capacity_rec, sizeof(char *)));
    MALLOC_CHK(db->mem_records);
    db->mem_bytes = (size_t *)(calloc(db->capacity_rec, sizeof(size_t)));
    MALLOC_CHK(db->mem_bytes);

    db->slow5_rec = (slow5_rec_t**)calloc(db->capacity_rec,sizeof(slow5_rec_t*));
    MALLOC_CHK(db->slow5_rec);

    db->means = (double*)calloc(db->capacity_rec,sizeof(double));
    MALLOC_CHK(db->means);

    db->sequence = new std::vector<std::string>(db->capacity_rec);
    db->qstring = new std::vector<std::string>(db->capacity_rec);
    db->read_dats = new std::vector<read_dat_t *>(db->capacity_rec, NULL);
    db->basecall_chunks = new std::vector<std::vector<basecall_chunk_t>>(db->capacity_rec, std::vector<basecall_chunk_t>());
    db->mod_chunks = new std::vector<std::vector<mod_chunk_t>>(db->capacity_rec, std::vector<mod_chunk_t>());
    db->moves = new std::vector<std::vector<uint8_t>>(db->capacity_rec, std::vector<uint8_t>());

    db->mod_string = new std::vector<std::string>(db->capacity_rec);
    db->mod_prob = new std::vector<std::vector<uint8_t>>(db->capacity_rec, std::vector<uint8_t>());

    db->total_reads = 0;
    db->sum_bytes = 0;

    return db;
}

/* load a data batch from disk */
ret_status_t load_db(core_t* core, db_t* db) {
    double load_start = realtime();

    db->n_rec = 0;
    db->sum_bytes = 0;
    db->total_reads = 0;

    ret_status_t status = {0, 0};
    int32_t i = 0;
    while (db->n_rec < db->capacity_rec && db->sum_bytes<core->opt.batch_size_bytes) {
        i=db->n_rec;

        if (slow5_get_next_bytes(&db->mem_records[i], &db->mem_bytes[i], core->sp) < 0) {
            if (slow5_errno != SLOW5_ERR_EOF) {
                ERROR("Error reading from SLOW5 file %d", slow5_errno);
                exit(EXIT_FAILURE);
            } else {
                break;
            }
        } else {
            db->n_rec++;
            db->total_reads++; // candidate read
            db->sum_bytes += db->mem_bytes[i];
        }
    }

    status.num_reads=db->n_rec;
    status.num_bytes=db->sum_bytes;

    double load_end = realtime();
    core->time_load_db += (load_end-load_start);

    return status;
}

void parse_single(core_t* core,db_t* db, int32_t i) {
    assert(db->mem_bytes[i] > 0);
    assert(db->mem_records[i] != NULL);

    int ret = slow5_decode(&db->mem_records[i], &db->mem_bytes[i], &db->slow5_rec[i], core->sp);
    if (ret < 0) {
        ERROR("Error parsing the record %d", i);
        exit(EXIT_FAILURE);
    }
}

void postprocess_signal(core_t* core, db_t* db, int32_t i) {
    slow5_rec_t* rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    if (len_raw_signal > 0) {
        auto& sequence = (*db->sequence)[i];
        sequence.clear();
        auto& qstring = (*db->qstring)[i];
        qstring.clear();
        auto& moves = (*db->moves)[i];
        moves.clear();

        stitch_chunks(db, i, sequence, qstring, moves, len_raw_signal, core->model_stride);
        
        if (is_rna(core->model_config->sample_type)) {
            std::reverse(sequence.begin(), sequence.end());
            std::reverse(qstring.begin(), qstring.end());
            std::reverse(moves.begin(), moves.end()); // might not need this, no idea
        }

    }
}

void process_db(core_t* core, db_t* db) {
    double proc_start = realtime();
    double a, b;

    a = realtime();
    work_db(core, db, parse_single);
    b = realtime();
    core->time_parse += (b - a);
    LOG_DEBUG("%s", "parsed reads");

    a = realtime();
    work_db(core, db, preprocess_signal);
    b = realtime();
    core->time_preproc += (b-a);
    LOG_DEBUG("%s", "preprocessed reads");

    // Pass A: FAST (or sole model) — no read filter
    core->cascade_filter_active = 0;
    a = realtime();
    basecall_db(core, db);
    b = realtime();
    core->time_runners += (b-a);
    LOG_DEBUG("%s", "basecalled reads");

    a = realtime();
    work_db(core, db, postprocess_signal);
    b = realtime();
    core->time_postproc += (b-a);
    LOG_DEBUG("%s", "postprocessed reads");

    if (core->opt.flag & SLORADO_CASCADE) {
        if (core->cascade_read_mask == nullptr) {
            core->cascade_read_mask = new std::vector<char>();
        }
        core->cascade_read_mask->assign(db->n_rec, 0);

        std::vector<float> mean_qs((size_t)db->n_rec, 0.f);
        for (int32_t i = 0; i < db->n_rec; ++i) {
            mean_qs[(size_t)i] = cascade_mean_phred((*db->qstring)[i]);
        }

        // Promote worst force_frac of reads by FAST mean_q.
        {
            std::vector<int32_t> order((size_t)db->n_rec);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                if (mean_qs[(size_t)a] != mean_qs[(size_t)b]) {
                    return mean_qs[(size_t)a] < mean_qs[(size_t)b];
                }
                return a < b;
            });
            int n_promote = (int)llround((double)db->n_rec * (double)core->opt.cascade_force_frac);
            if (n_promote < 0) {
                n_promote = 0;
            }
            if (n_promote > db->n_rec) {
                n_promote = db->n_rec;
            }
            for (int j = 0; j < n_promote; ++j) {
                (*core->cascade_read_mask)[order[(size_t)j]] = 1;
            }
        }

        int n_hard = 0;
        for (int32_t i = 0; i < db->n_rec; ++i) {
            bool hard = (*core->cascade_read_mask)[i] != 0;
            float mean_q = mean_qs[(size_t)i];
            if (hard) {
                n_hard++;
                core->cascade_n_hac_promoted++;
                (*db->sequence)[i].clear();
                (*db->qstring)[i].clear();
                (*db->moves)[i].clear();
                for (auto &ch : (*db->basecall_chunks)[i]) {
                    ch.seq.clear();
                    ch.qstring.clear();
                    ch.moves.clear();
                }
            } else {
                core->cascade_n_fast_kept++;
            }
            if (core->cascade_log_fp != nullptr) {
                const char *rid = db->slow5_rec[i] ? db->slow5_rec[i]->read_id : "?";
                fprintf(core->cascade_log_fp, "%s\t%.3f\t%s\n", rid, mean_q, hard ? "hac" : "fast");
            }
        }
        if (core->cascade_log_fp != nullptr) {
            fflush(core->cascade_log_fp);
        }

        if (n_hard > 0) {
            LOG_DEBUG("cascade: promoting %d / %d reads to HAC (worst %.1f%% by mean_q)",
                      n_hard, db->n_rec, 100.0 * (double)core->opt.cascade_force_frac);

            auto *fast_cfg = core->model_config;
            openfish_opt_t fast_dec = core->decoder_opts;

            // Unload FAST, load HAC (only one model in VRAM), then swap back.
            cascade_park_fast_runners(core);
            init_cascade_hac_runners(core, (char *)core->cascade_hac_path);

            core->runners = core->hac_runners;
            core->runner_stats = core->hac_runner_stats;
            core->model_config = core->hac_model_config;
            core->decoder_opts = core->hac_decoder_opts;
            core->cascade_filter_active = 1;

            a = realtime();
            basecall_db(core, db);
            b = realtime();
            core->time_runners += (b - a);

            a = realtime();
            for (int32_t i = 0; i < db->n_rec; ++i) {
                if ((*core->cascade_read_mask)[i]) {
                    postprocess_signal(core, db, i);
                }
            }
            b = realtime();
            core->time_postproc += (b - a);

            core->cascade_filter_active = 0;
            core->model_config = fast_cfg;
            core->decoder_opts = fast_dec;

            free_cascade_hac_runners(core);
            cascade_unpark_fast_runners(core);
        }
    }

    if (core->opt.mod != NULL) {
        a = realtime();
        work_db(core, db, preprocess_modbase);
        b = realtime();
        core->time_preproc_mod += (b-a);
        LOG_DEBUG("%s", "mod preprocessed reads");

        a = realtime();
        mod_basecall_db(core, db);
        b = realtime();
        core->time_runners += (b-a);
        LOG_DEBUG("%s", "mod basecalled reads");

        a = realtime();
        work_db(core, db, postprocess_modbase);
        b = realtime();
        core->time_postproc_mod += (b-a);
        LOG_DEBUG("%s", "mod postprocessed reads");
    }

    double proc_end = realtime();
    core->time_process_db += (proc_end-proc_start);
}

/* write the output for a processed data batch */
void output_db(core_t* core, db_t* db) {
    double output_start = realtime();

    int32_t i = 0;
    for (i = 0; i < db->n_rec; i++) {
        if (db->slow5_rec[i]->len_raw_signal > 0) {
            if ((core->opt.flag & SLORADO_SAM) != 0) {
                write_to_file_sam(core->opt.out, (*db->sequence)[i].c_str(), (*db->qstring)[i].c_str(), db->slow5_rec[i]->read_id, (*db->mod_string)[i].c_str(), (*db->mod_prob)[i]);
            } else {
                write_to_file_fastq(core->opt.out, (*db->sequence)[i].c_str(), (*db->qstring)[i].c_str(), db->slow5_rec[i]->read_id);
            }
        }
    }

    core->sum_bytes += db->sum_bytes;
    core->total_reads += db->total_reads;

    double output_end = realtime();
    core->time_output += (output_end-output_start);
}

/* partially free a data batch - only the read dependent allocations are freed */
void free_db_tmp(db_t* db) {
    LOG_DEBUG("%s", "freeing db_tmp");
    int32_t i = 0;
    for (i = 0; i < db->n_rec; ++i) {
        free(db->mem_records[i]);
        db->mem_records[i] = NULL;
    }
}

/* completely free a data batch */
void free_db(db_t* db) {
    LOG_DEBUG("%s", "freeing db");
    int32_t i = 0;
    for (i = 0; i < db->capacity_rec; ++i) {
        free_read_dat((*db->read_dats)[i]);
        slow5_rec_free(db->slow5_rec[i]);
    }
    free(db->slow5_rec);
    free(db->mem_records);
    free(db->mem_bytes);
    free(db->means);
    delete db->sequence;
    delete db->qstring;
    delete db->moves;
    delete db->mod_string;
    delete db->mod_prob;
    delete db->basecall_chunks;
    delete db->mod_chunks;
    delete db->read_dats;
    free(db);
}

/* initialise user specified options */
void init_opt(opt_t* opt) {
    memset(opt, 0, sizeof(opt_t));
    opt->batch_size = 4096;
    opt->gpu_batch_size = 512;
    opt->batch_size_bytes = 512*1000*1000;
    opt->num_thread = 8;

    opt->debug_break = -1;

#ifdef USE_GPU
    opt->device = "cuda:all";
#else
    opt->device = "cpu";
#endif

    opt->chunk_size = 12288;
    opt->overlap = 150;
    opt->overlap_depth = 1;

    opt->out = stdout;

    opt->mod = NULL;
    opt->cascade_hac = NULL;
    opt->cascade_force_frac = 0.0f; // must be set in (0,1] when --cascade=yes
    opt->cascade_log = NULL;

    // opt->flag |= SLORADO_SAM;
}
