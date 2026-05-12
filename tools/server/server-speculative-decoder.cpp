#include "server-speculative-decoder.h"

#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <utility>

namespace {

    bool accept_special_token(
        const server_speculative_decoder::runtime& rt,
        const server_slot& slot,
        llama_token token) {
        return rt.special ||
            slot.task->params.sampling.preserved_tokens.find(token) !=
            slot.task->params.sampling.preserved_tokens.end();
    }

    bool verify_and_accept_draft(server_slot& slot, size_t n_draft) {
        const auto& params_spec = slot.task->params.speculative;

        common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

        GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);

        auto accepted = common_sampler_sample_and_accept_n(
            slot.smpl.get(),
            slot.ctx,
            slot.spec_i_batch,
            slot.spec_draft);

        slot.spec_i_batch.clear();

        SLT_DBG(
            slot,
            "%s: n_draft=%zu, accepted=%zu\n",
            __func__,
            slot.spec_draft.size(),
            accepted.size());

        GGML_ASSERT(accepted.size() >= 1);

        // check for partial draft acceptance
        if (accepted.size() < slot.spec_draft.size() + 1) {
            if (params_spec.use_checkpoints) {
                // partial acceptance is not supported by the context -> truncate the draft and restore the state
                slot.spec_draft = std::move(accepted);

                auto& ckpt = slot.spec_ckpt;

                SLT_DBG(
                    slot,
                    "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n",
                    ckpt.pos_min,
                    ckpt.pos_max,
                    ckpt.size());

                const size_t n = llama_state_seq_set_data_ext(
                    slot.ctx,
                    ckpt.data.data(),
                    ckpt.size(),
                    slot.id,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                if (n != ckpt.size()) {
                    GGML_ABORT(
                        "%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu, get_data_ext->%zu, set_data_ext->%zu",
                        __func__,
                        ckpt.pos_min,
                        ckpt.pos_max,
                        ckpt.size(),
                        ckpt.size(),
                        n);
                }

                llama_memory_seq_rm(
                    llama_get_memory(slot.ctx),
                    slot.id,
                    ckpt.pos_max + 1,
                    -1);

                slot.prompt.tokens.keep_first(ckpt.n_tokens);
                slot.smpl = std::move(smpl_save);

                return false;
            }

            LOG_DBG(
                "%s: partial acceptance: %zu < %zu\n",
                __func__,
                accepted.size(),
                slot.spec_draft.size());
        }

        common_speculative_accept(slot.spec.get(), accepted.size() - 1);

        slot.spec_draft = std::move(accepted);

        return true;
    }

    void finish_slot_after_stop(
        const server_speculative_decoder::runtime& rt,
        server_slot& slot) {
        slot.print_timings();

        server_token_processor::send_final_response(
            rt.token_rt,
            slot);

        GGML_ASSERT(rt.metrics != nullptr);
        rt.metrics->on_prediction(slot);

        slot.release();
    }

} // namespace

namespace server_speculative_decoder {

    void process_accepted_drafts(
        const runtime& rt,
        std::vector<server_slot>& slots) {
        GGML_ASSERT(rt.metrics != nullptr);

        for (auto& slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING ||
                !slot.can_speculate() ||
                slot.spec_draft.empty()) {
                continue;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            if (!verify_and_accept_draft(slot, n_draft)) {
                continue;
            }

            const int64_t t_current = ggml_time_us();

            llama_tokens ids = std::move(slot.spec_draft);

            slot.n_decoded += static_cast<int32_t>(ids.size());
            slot.t_token_generation =
                std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += static_cast<int32_t>(ids.size() - 1);
            slot.n_draft_total += static_cast<int32_t>(n_draft);

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ ids.begin(), ids.end() - 1 });

            slot.sampled = ids.back(); // last accepted token

            SLT_DBG(
                slot,
                "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n",
                slot.sampled,
                ids.size(),
                n_draft);

            llama_memory_seq_rm(
                llama_get_memory(slot.ctx),
                slot.id,
                slot.prompt.n_tokens(),
                -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok = ids[i];
                result.text_to_send =
                    common_token_to_piece(
                        slot.ctx,
                        result.tok,
                        accept_special_token(rt, slot, result.tok));
                result.prob = 1.0f; // set later

                // TODO: set result.probs

                if (!server_token_processor::process_token(
                    rt.token_rt,
                    result,
                    slot)) {
                    finish_slot_after_stop(rt, slot);
                    break;
                }
            }

            SLT_DBG(
                slot,
                "accepted %d/%d draft tokens, new n_tokens = %d\n",
                static_cast<int>(ids.size()) - 1,
                static_cast<int>(n_draft),
                slot.prompt.n_tokens());
        }
    }

} // namespace server_speculative_decoder