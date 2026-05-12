#pragma once

#include "server-slot.h"

#include <cstdint>
#include <vector>

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total = 0;
    uint64_t n_tokens_predicted_total = 0;
    uint64_t t_tokens_generation_total = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing = 0;

    uint64_t n_tokens_predicted = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total = 0;
    uint64_t n_busy_slots_total = 0;

    void init();

    void on_prompt_eval(const server_slot& slot);

    void on_prediction(const server_slot& slot);

    void on_decoded(const std::vector<server_slot>& slots);

    void reset_bucket();
};