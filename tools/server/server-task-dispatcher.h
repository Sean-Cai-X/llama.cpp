#pragma once

#include "server-common.h"
#include "server-lora-manager.h"
#include "server-metrics.h"
#include "server-queue.h"
#include "server-slot.h"
#include "server-task.h"

#include "llama.h"

#include <functional>
#include <vector>

namespace server_task_dispatcher {

    struct runtime {
        server_queue* queue_tasks = nullptr;
        server_response* queue_results = nullptr;

        common_params* params = nullptr;
        server_metrics* metrics = nullptr;

        llama_context* ctx = nullptr;
        const llama_vocab* vocab = nullptr;

        std::vector<server_slot>* slots = nullptr;

        int slots_debug = 0;

        std::function<server_slot* (int)> get_slot_by_id;
        std::function<void(const server_task&, const std::string&, error_type)> send_error;
        std::function<bool(int)> check_no_mtmd;
    };

    void handle_cancel(
        const runtime& rt,
        const server_task& task);

    void handle_next_response(
        const runtime& rt,
        const server_task& task);

    void handle_metrics(
        const runtime& rt,
        const server_task& task);

    void handle_slot_save(
        const runtime& rt,
        server_task&& task);

    void handle_slot_restore(
        const runtime& rt,
        server_task&& task);

    void handle_slot_erase(
        const runtime& rt,
        server_task&& task);

    void handle_get_lora(
        const runtime& rt,
        const server_task& task);

    void handle_set_lora(
        const runtime& rt,
        const server_task& task);

} // namespace server_task_dispatcher