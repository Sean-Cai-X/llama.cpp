#pragma once

#include "server-common.h"
#include "server-queue.h"
#include "server-slot.h"

#include "llama.h"
#include "mtmd.h"

#include <functional>
#include <vector>

namespace server_slot_updater {

    struct runtime {
        server_queue* queue_tasks = nullptr;

        common_params* params = nullptr;

        llama_context* ctx = nullptr;
        mtmd_context* mctx = nullptr;

        std::vector<server_slot>* slots = nullptr;

        bool add_bos_token = true;

        std::function<void(server_slot&, const std::string&, error_type)> send_error;
    };

    bool all_slots_idle(const std::vector<server_slot>& slots);

    void post_next_response(server_queue& queue_tasks);

    void apply_context_shift(const runtime& rt);

} // namespace server_slot_updater