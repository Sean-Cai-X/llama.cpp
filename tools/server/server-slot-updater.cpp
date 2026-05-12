#include "server-slot-updater.h"

#include "common.h"
#include "log.h"

#include <algorithm>

namespace server_slot_updater {

    bool all_slots_idle(const std::vector<server_slot>& slots) {
        for (const auto& slot : slots) {
            if (slot.is_processing()) {
                return false;
            }
        }

        return true;
    }

    void post_next_response(server_queue& queue_tasks) {
        SRV_DBG("%s", "posting NEXT_RESPONSE\n");

        server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);

        task.id = queue_tasks.get_new_id();

        queue_tasks.post(std::move(task));
    }

    void apply_context_shift(const runtime& rt) {
        GGML_ASSERT(rt.params != nullptr);
        GGML_ASSERT(rt.ctx != nullptr);
        GGML_ASSERT(rt.slots != nullptr);
        GGML_ASSERT(rt.send_error);

        auto& params_base = *rt.params;
        auto& slots = *rt.slots;

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot& slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING ||
                slot.prompt.n_tokens() + 1 < slot.n_ctx) {
                continue;
            }

            if (!params_base.ctx_shift) {
                // this check is redundant (for good)
                // we should never get here, because generation should already stopped in process_token()
                rt.send_error(
                    slot,
                    "context shift is disabled",
                    ERROR_TYPE_SERVER);

                slot.release();
                continue;
            }

            if (rt.mctx) {
                // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                // we don't support ctx_shift because an image chunk may contains multiple tokens
                GGML_ABORT("not supported by multimodal");
            }

            if (slot.task->is_parent() || slot.task->is_child()) {
                rt.send_error(
                    slot,
                    "context shift cannot be used for shared prompt",
                    ERROR_TYPE_SERVER);

                slot.release();
                continue;
            }

            // Shift context
            int n_keep =
                slot.task->params.n_keep < 0
                ? slot.task->n_tokens()
                : slot.task->params.n_keep;

            if (rt.add_bos_token) {
                n_keep += 1;
            }

            n_keep = std::min(slot.n_ctx - 4, n_keep);

            const int n_left =
                slot.prompt.n_tokens() - n_keep;

            const int n_discard =
                slot.task->params.n_discard
                ? slot.task->params.n_discard
                : n_left / 2;

            SLT_WRN(
                slot,
                "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n",
                n_keep,
                n_left,
                n_discard);

            llama_memory_seq_rm(
                llama_get_memory(rt.ctx),
                slot.id,
                n_keep,
                n_keep + n_discard);

            llama_memory_seq_add(
                llama_get_memory(rt.ctx),
                slot.id,
                n_keep + n_discard,
                slot.prompt.n_tokens(),
                -n_discard);

            // add generated tokens to cache
            // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
            {
                GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy

                for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                    new_tokens[i - n_discard] = new_tokens[i];
                }

                new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                slot.prompt.tokens.clear();
                slot.prompt.tokens.insert(new_tokens);
            }

            slot.truncated = true;

        }
    }

} // namespace server_slot_updater