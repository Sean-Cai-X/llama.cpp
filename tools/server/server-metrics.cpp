#include "server-metrics.h"

#include "common.h"

#include <algorithm>

void server_metrics::init() {
    t_start = ggml_time_us();
}

void server_metrics::on_prompt_eval(const server_slot& slot) {
    n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
    n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
    t_prompt_processing += slot.t_prompt_processing;
    t_prompt_processing_total += slot.t_prompt_processing;

    n_tokens_max = std::max(
        n_tokens_max,
        static_cast<uint64_t>(slot.prompt.n_tokens()));
}

void server_metrics::on_prediction(const server_slot& slot) {
    n_tokens_predicted_total += slot.n_decoded;
    n_tokens_predicted += slot.n_decoded;
    t_tokens_generation += slot.t_token_generation;
    t_tokens_generation_total += slot.t_token_generation;
}

void server_metrics::on_decoded(const std::vector<server_slot>& slots) {
    n_decode_total++;

    for (const auto& slot : slots) {
        if (slot.is_processing()) {
            n_busy_slots_total++;
        }

        n_tokens_max = std::max(
            n_tokens_max,
            static_cast<uint64_t>(slot.prompt.n_tokens()));
    }
}

void server_metrics::reset_bucket() {
    n_prompt_tokens_processed = 0;
    t_prompt_processing = 0;
    n_tokens_predicted = 0;
    t_tokens_generation = 0;
}