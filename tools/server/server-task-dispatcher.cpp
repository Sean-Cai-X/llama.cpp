#include "server-task-dispatcher.h"

#include "common.h"
#include "log.h"

#include <memory>
#include <utility>

namespace {

    server_slot* resolve_slot_or_send_error(
        const server_task_dispatcher::runtime& rt,
        const server_task& task,
        int id_slot) {
        GGML_ASSERT(rt.get_slot_by_id);
        GGML_ASSERT(rt.send_error);

        server_slot* slot = rt.get_slot_by_id(id_slot);

        if (slot == nullptr) {
            rt.send_error(
                task,
                "Invalid slot ID",
                ERROR_TYPE_INVALID_REQUEST);
            return nullptr;
        }

        return slot;
    }

    bool defer_if_slot_processing(
        const server_task_dispatcher::runtime& rt,
        server_task& task,
        server_slot& slot) {
        GGML_ASSERT(rt.queue_tasks != nullptr);

        if (!slot.is_processing()) {
            return false;
        }

        // if requested slot is unavailable, defer this task for processing later
        SRV_DBG(
            "requested slot is unavailable, defer task, id_task = %d\n",
            task.id);

        rt.queue_tasks->defer(std::move(task));
        return true;
    }

} // namespace

namespace server_task_dispatcher {

    void handle_cancel(
        const runtime& rt,
        const server_task& task) {
        GGML_ASSERT(rt.slots != nullptr);

        // release slot linked with the task id
        for (auto& slot : *rt.slots) {
            if (slot.task && slot.task->id == task.id_target) {
                slot.release();
                break;
            }
        }
    }

    void handle_next_response(
        const runtime&,
        const server_task&) {
        // do nothing
    }

    void handle_metrics(
        const runtime& rt,
        const server_task& task) {
        GGML_ASSERT(rt.queue_tasks != nullptr);
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.metrics != nullptr);
        GGML_ASSERT(rt.slots != nullptr);

        json slots_data = json::array();

        int n_idle_slots = 0;
        int n_processing_slots = 0;

        for (server_slot& slot : *rt.slots) {
            json slot_data = slot.to_json(rt.slots_debug == 0);

            if (slot.is_processing()) {
                n_processing_slots++;
            }
            else {
                n_idle_slots++;
            }

            slots_data.push_back(slot_data);
        }

        SRV_DBG(
            "n_idle_slots = %d, n_processing_slots = %d\n",
            n_idle_slots,
            n_processing_slots);

        auto res = std::make_unique<server_task_result_metrics>();

        res->id = task.id;
        res->slots_data = std::move(slots_data);
        res->n_idle_slots = n_idle_slots;
        res->n_processing_slots = n_processing_slots;
        res->n_tasks_deferred = rt.queue_tasks->queue_tasks_deferred_size();
        res->t_start = rt.metrics->t_start;

        res->n_prompt_tokens_processed_total = rt.metrics->n_prompt_tokens_processed_total;
        res->t_prompt_processing_total = rt.metrics->t_prompt_processing_total;
        res->n_tokens_predicted_total = rt.metrics->n_tokens_predicted_total;
        res->t_tokens_generation_total = rt.metrics->t_tokens_generation_total;

        res->n_tokens_max = rt.metrics->n_tokens_max;

        res->n_prompt_tokens_processed = rt.metrics->n_prompt_tokens_processed;
        res->t_prompt_processing = rt.metrics->t_prompt_processing;
        res->n_tokens_predicted = rt.metrics->n_tokens_predicted;
        res->t_tokens_generation = rt.metrics->t_tokens_generation;

        res->n_decode_total = rt.metrics->n_decode_total;
        res->n_busy_slots_total = rt.metrics->n_busy_slots_total;

        if (task.metrics_reset_bucket) {
            rt.metrics->reset_bucket();
        }

        rt.queue_results->send(std::move(res));
    }

    void handle_slot_save(
        const runtime& rt,
        server_task&& task) {
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.ctx != nullptr);
        GGML_ASSERT(rt.check_no_mtmd);

        if (!rt.check_no_mtmd(task.id)) {
            return;
        }

        const int id_slot = task.slot_action.id_slot;

        server_slot* slot = resolve_slot_or_send_error(rt, task, id_slot);
        if (slot == nullptr) {
            return;
        }

        if (defer_if_slot_processing(rt, task, *slot)) {
            return;
        }

        const size_t token_count = slot->prompt.tokens.size();
        const int64_t t_start = ggml_time_us();

        const std::string filename = task.slot_action.filename;
        const std::string filepath = task.slot_action.filepath;

        const llama_tokens& tokens = slot->prompt.tokens.get_tokens();

        const size_t nwrite =
            llama_state_seq_save_file(
                rt.ctx,
                filepath.c_str(),
                slot->id,
                tokens.data(),
                token_count);

        const int64_t t_end = ggml_time_us();
        const double t_save_ms = (t_end - t_start) / 1000.0;

        auto res = std::make_unique<server_task_result_slot_save_load>();

        res->id = task.id;
        res->id_slot = id_slot;
        res->filename = filename;
        res->is_save = true;
        res->n_tokens = token_count;
        res->n_bytes = nwrite;
        res->t_ms = t_save_ms;

        rt.queue_results->send(std::move(res));
    }

    void handle_slot_restore(
        const runtime& rt,
        server_task&& task) {
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.ctx != nullptr);
        GGML_ASSERT(rt.check_no_mtmd);
        GGML_ASSERT(rt.send_error);

        if (!rt.check_no_mtmd(task.id)) {
            return;
        }

        const int id_slot = task.slot_action.id_slot;

        server_slot* slot = resolve_slot_or_send_error(rt, task, id_slot);
        if (slot == nullptr) {
            return;
        }

        if (defer_if_slot_processing(rt, task, *slot)) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        const std::string filename = task.slot_action.filename;
        const std::string filepath = task.slot_action.filepath;

        llama_tokens tokens;
        tokens.resize(slot->n_ctx);

        size_t token_count = 0;

        const size_t nread =
            llama_state_seq_load_file(
                rt.ctx,
                filepath.c_str(),
                slot->id,
                tokens.data(),
                tokens.size(),
                &token_count);

        if (nread == 0) {
            // KV may already been invalidated?
            slot->prompt.tokens.clear();

            rt.send_error(
                task,
                "Unable to restore slot, no available space in KV cache or invalid slot save file",
                ERROR_TYPE_INVALID_REQUEST);

            return;
        }

        tokens.resize(token_count);

        slot->prompt.tokens.clear();
        slot->prompt.tokens.insert(tokens);

        const int64_t t_end = ggml_time_us();
        const double t_restore_ms = (t_end - t_start) / 1000.0;

        auto res = std::make_unique<server_task_result_slot_save_load>();

        res->id = task.id;
        res->id_slot = id_slot;
        res->filename = filename;
        res->is_save = false;
        res->n_tokens = token_count;
        res->n_bytes = nread;
        res->t_ms = t_restore_ms;

        rt.queue_results->send(std::move(res));
    }

    void handle_slot_erase(
        const runtime& rt,
        server_task&& task) {
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.check_no_mtmd);

        if (!rt.check_no_mtmd(task.id)) {
            return;
        }

        const int id_slot = task.slot_action.id_slot;

        server_slot* slot = resolve_slot_or_send_error(rt, task, id_slot);
        if (slot == nullptr) {
            return;
        }

        if (defer_if_slot_processing(rt, task, *slot)) {
            return;
        }

        // Erase token cache
        const size_t n_erased = slot->prompt.tokens.size();

        slot->prompt_clear(false);

        auto res = std::make_unique<server_task_result_slot_erase>();

        res->id = task.id;
        res->id_slot = id_slot;
        res->n_erased = n_erased;

        rt.queue_results->send(std::move(res));
    }

    void handle_get_lora(
        const runtime& rt,
        const server_task& task) {
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.params != nullptr);
        GGML_ASSERT(rt.vocab != nullptr);

        auto res =
            server_lora_manager::build_get_lora_result(
                task.id,
                rt.params->lora_adapters,
                rt.vocab);

        rt.queue_results->send(std::move(res));
    }

    void handle_set_lora(
        const runtime& rt,
        const server_task& task) {
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.params != nullptr);

        auto res =
            server_lora_manager::build_apply_lora_result(
                task.id,
                task.set_lora,
                rt.params->lora_adapters);

        rt.queue_results->send(std::move(res));
    }

} // namespace server_task_dispatcher