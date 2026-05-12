#pragma once

#include "server-common.h"
#include "server-queue.h"
#include "server-slot.h"

#include "llama.h"

namespace server_token_processor {

    struct runtime {
        server_response* queue_results = nullptr;

        llama_context* ctx = nullptr;
        const llama_vocab* vocab = nullptr;

        const common_params* params = nullptr;

        int slots_debug = 0;
        bool special = false;
    };

    void send_partial_response(
        const runtime& rt,
        server_slot& slot,
        const completion_token_output& tkn,
        bool is_progress);

    void send_final_response(
        const runtime& rt,
        server_slot& slot);

    bool process_token(
        const runtime& rt,
        completion_token_output& result,
        server_slot& slot);

    void populate_token_probs(
        const runtime& rt,
        const server_slot& slot,
        completion_token_output& result,
        bool post_sampling,
        int idx);

} // namespace server_token_processor