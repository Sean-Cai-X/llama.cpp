#include "server-token-processor.h"

#include "common.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace server_token_processor {

    void send_partial_response(
        const runtime& rt,
        server_slot& slot,
        const completion_token_output& tkn,
        bool is_progress) {
        GGML_ASSERT(rt.queue_results != nullptr);

        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress = true;
            res->progress.total = slot.task->n_tokens();
            res->progress.cache = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        else {
            res->content = tkn.text_to_send;
            res->tokens = { tkn.tok };
        }

        res->n_decoded = slot.n_decoded;
        res->n_prompt_tokens = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose = slot.task->params.verbose;
        res->res_type = slot.task->params.res_type;
        res->oaicompat_model = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;
        res->generation_params = slot.task->params;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        rt.queue_results->send(std::move(res));
    }

    void send_final_response(
        const runtime& rt,
        server_slot& slot) {
        GGML_ASSERT(rt.queue_results != nullptr);
        GGML_ASSERT(rt.ctx != nullptr);

        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (rt.slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content = "";
            res->tokens = llama_tokens{};
        }
        else {
            res->content = std::move(slot.generated_text);
            res->tokens = std::move(slot.generated_tokens);
        }

        res->timings = slot.get_timings();
        res->prompt = slot.task->tokens.detokenize(rt.ctx, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated = slot.truncated;
        res->n_decoded = slot.n_decoded;
        res->n_prompt_tokens = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached = slot.prompt.n_tokens();
        res->has_new_line = slot.has_new_line;
        res->stopping_word = slot.stopping_word;
        res->stop = slot.stop;
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose = slot.task->params.verbose;
        res->stream = slot.task->params.stream;
        res->include_usage = slot.task->params.include_usage;
        res->res_type = slot.task->params.res_type;
        res->oaicompat_model = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks =
                    common_tokenize(rt.ctx, slot.stopping_word, false);

                const size_t safe_offset =
                    std::min(slot.generated_token_probs.size(), stop_word_toks.size());

                res->probs_output = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(),
                    slot.generated_token_probs.end() - safe_offset);
            }
            else {
                res->probs_output = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(),
                    slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        rt.queue_results->send(std::move(res));
    }

    bool process_token(
        const runtime& rt,
        completion_token_output& result,
        server_slot& slot) {
        GGML_ASSERT(rt.vocab != nullptr);
        GGML_ASSERT(rt.params != nullptr);

        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;

        slot.sampled = result.tok;

        slot.generated_text += token_str;

        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }

        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(
                str_test,
                token_str.size(),
                true);

            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());

                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            }
            else if (slot.has_next_token &&
                !llama_vocab_is_eog(rt.vocab, result.tok)) {
                stop_pos = slot.find_stopping_strings(
                    str_test,
                    token_str.size(),
                    false);

                send_text = stop_pos == std::string::npos;
            }

            if (send_text) {
                // do not send the stop word in the response
                result.text_to_send =
                    slot.generated_text.substr(pos, std::string::npos);

                slot.n_sent_text += result.text_to_send.size();
            }
            else {
                result.text_to_send = "";
            }

            slot.add_token(result);

            if (slot.task->params.stream) {
                send_partial_response(rt, slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!rt.params->ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated = true;
            slot.stop = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(
                slot,
                "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                slot.prompt.n_tokens(),
                slot.task->n_tokens(),
                slot.n_decoded,
                slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 &&
            slot.has_next_token &&
            !slot.has_budget(*rt.params)) {
            slot.stop = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(
                slot,
                "stopped by limit, n_decoded = %d, n_predict = %d\n",
                slot.n_decoded,
                slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix
            // of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;

                    while (pos < slot.generated_text.size() &&
                        (slot.generated_text[pos] == ' ' ||
                            slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() &&
                        n_indent < slot.task->params.n_indent) {
                        slot.stop = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(
                            slot,
                            "stopped by indentation limit, n_decoded = %d, n_indent = %d\n",
                            slot.n_decoded,
                            n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos =
                        slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit,
            // but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 &&
                (ggml_time_us() - slot.t_start_generation >
                    1000.0f * slot.task->params.t_max_predict_ms)) {
                slot.stop = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(
                    slot,
                    "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n",
                    slot.n_decoded,
                    static_cast<int>(slot.task->params.t_max_predict_ms));
            }
        }

        if (llama_vocab_is_eog(rt.vocab, result.tok)) {
            slot.stop = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(
            slot,
            "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n",
            slot.n_decoded,
            slot.n_remaining,
            result.tok,
            token_str.c_str());

        return slot.has_next_token;
    }

    void populate_token_probs(
        const runtime& rt,
        const server_slot& slot,
        completion_token_output& result,
        bool post_sampling,
        int idx) {
        GGML_ASSERT(rt.ctx != nullptr);

        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto* cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);

            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(rt.ctx, cur_p->data[i].id, rt.special),
                    cur_p->data[i].p
                    });
            }
        }
        else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(rt.ctx, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);

            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(rt.ctx, cur[i].id, rt.special),
                    cur[i].p
                    });
            }
        }
    }

} // namespace server_token_processor