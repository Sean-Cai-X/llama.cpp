#include "server-lora-manager.h"

#include "common.h"
#include "log.h"

namespace server_lora_manager {

    std::vector<common_adapter_lora_info> construct_lora_list(
        const std::vector<common_adapter_lora_info>& base_loras,
        const std::map<int, float>& config) {
        std::vector<common_adapter_lora_info> output = base_loras; // copy

        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(static_cast<int>(i));

            if (it != config.end()) {
                output[i].scale = it->second;
            }
            else {
                output[i].scale = 0.0f;
            }
        }

        return output;
    }

    slot_lora_apply_result apply_task_lora_to_slot(
        server_slot& slot,
        const server_task& task,
        const std::vector<common_adapter_lora_info>& base_loras) {
        slot_lora_apply_result result;

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(base_loras, task.params.lora);

            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_INF(
                        slot,
                        "clearing cache for lora change. %zu loras -> %zu loras\n",
                        slot.lora.size(),
                        task.params.lora.size());

                    slot.prompt.tokens.clear();
                }
                else {
                    SLT_INF(
                        slot,
                        "keeping cache for alora. %zu target loras\n",
                        task_loras.size());
                }

                slot.lora = task_loras;
            }
        }
        else {
            slot.lora = base_loras;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();

        if (lora_all_alora(slot.lora)) {
            const auto& enabled_ids = lora_get_enabled_ids(slot.lora);

            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                result.ok = false;
                result.error_message = "Cannot run multiple aLoRAs in a single request";
                result.error = ERROR_TYPE_INVALID_REQUEST;
                return result;
            }

            const auto& lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token* invocation_tokens = llama_adapter_get_alora_invocation_tokens(lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;

            for (int i = static_cast<int>(task.tokens.size()) - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }

                    // otherwise, check the next token in the sequence
                    --match_idx;
                }
                else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(
                    slot,
                    "alora %zu requested, but not found. deactivating\n",
                    enabled_ids[0]);

                slot.lora[enabled_ids[0]].scale = 0.0f;
            }
            else {
                SLT_DBG(
                    slot,
                    "alora %zu activated starting at %zu\n",
                    enabled_ids[0],
                    alora_invocation_start);

                slot.alora_invocation_start = static_cast<int32_t>(alora_invocation_start);
            }
        }

        return result;
    }

    std::unique_ptr<server_task_result_get_lora> build_get_lora_result(
        int id_task,
        const std::vector<common_adapter_lora_info>& loras,
        const llama_vocab* vocab) {
        auto res = std::make_unique<server_task_result_get_lora>();
        res->id = id_task;

        for (size_t i = 0; i < loras.size(); ++i) {
            const auto& lora = loras[i];

            std::string alora_invocation_string;
            llama_tokens alora_invocation_tokens;

            const uint64_t n_alora_tokens =
                llama_adapter_get_alora_n_invocation_tokens(lora.ptr);

            if (n_alora_tokens) {
                const llama_token* alora_tokens =
                    llama_adapter_get_alora_invocation_tokens(lora.ptr);

                for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                    alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                    alora_invocation_tokens.push_back(alora_tokens[j]);
                }
            }

            res->loras.push_back(server_task_result_get_lora::lora{
                lora,
                alora_invocation_string,
                alora_invocation_tokens,
                });
        }

        return res;
    }

    std::unique_ptr<server_task_result_apply_lora> build_apply_lora_result(
        int id_task,
        const std::map<int, float>& config,
        std::vector<common_adapter_lora_info>& base_loras) {
        auto new_loras = construct_lora_list(base_loras, config);

        // logging
        for (size_t i = 0; i < new_loras.size(); ++i) {
            SRV_INF(
                "set lora adapter idx=%zu scale=%f\n",
                i,
                new_loras[i].scale);
        }

        // TODO @ngxson : make lora_adapters a dedicated member of server_context
        base_loras = std::move(new_loras);

        auto res = std::make_unique<server_task_result_apply_lora>();
        res->id = id_task;

        return res;
    }

} // namespace server_lora_manager