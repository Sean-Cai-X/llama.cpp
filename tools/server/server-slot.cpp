#include "server-slot.h"

#include "log.h"

#include <algorithm>
#include <cinttypes>
server_prompt_checkpoint server_get_checkpoint(
    llama_context* ctx,
    int id,
    int64_t n_tokens,
    llama_pos pos_min,
    llama_pos pos_max) {
    if (pos_min == -1) {
        pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), id);
    }

    if (pos_max == -1) {
        pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), id);
    }

    const size_t checkpoint_size =
        llama_state_seq_get_size_ext(
            ctx,
            id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    auto cur = server_prompt_checkpoint{
        /*.pos_min  = */ pos_min,
        /*.pos_max  = */ pos_max,
        /*.n_tokens = */ n_tokens,
        /*.data     = */ std::vector<uint8_t>(checkpoint_size),
    };

    const size_t n =
        llama_state_seq_get_data_ext(
            ctx,
            cur.data.data(),
            checkpoint_size,
            id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    if (n != checkpoint_size) {
        GGML_ABORT(
            "checkpoint size mismatch: expected %zu, got %zu\n",
            checkpoint_size,
            n);
    }

    return cur;
}

void server_slot::prompt_save(server_prompt_cache& prompt_cache) const {
    GGML_ASSERT(prompt.data.size() == 0);

    const size_t cur_size = llama_state_seq_get_size_ext(ctx, id, 0);

    SRV_WRN(
        " - saving prompt with length %d, total state size = %.3f MiB\n",
        static_cast<int>(prompt.tokens.size()),
        cur_size / (1024.0 * 1024.0));

    auto* cur = prompt_cache.alloc(prompt, cur_size);
    if (cur == nullptr) {
        return;
    }

    llama_state_seq_get_data_ext(ctx, cur->data.data(), cur_size, id, 0);
}

bool server_slot::prompt_load(
    server_prompt_cache& prompt_cache,
    const server_tokens& tokens) {
    bool res = prompt_cache.load(prompt, tokens, ctx, id);

    if (!res) {
        SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
    }

    return res;
}

void server_slot::prompt_clear(bool allow_processing) {
    if (!allow_processing) {
        GGML_ASSERT(!is_processing());
    }

    SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

    llama_memory_seq_rm(llama_get_memory(ctx), id, -1, -1);
    prompt.tokens.clear();
}

void server_slot::reset() {
    SLT_DBG(*this, "%s", "\n");

    n_prompt_tokens_cache = 0;

    last_nl_pos = 0;
    generated_text = "";
    has_new_line = false;
    truncated = false;
    stop = STOP_TYPE_NONE;
    stopping_word = "";
    n_sent_text = 0;

    if (can_speculate()) {
        spec_draft.clear();
        spec_i_batch.clear();
        spec_ckpt.clear();
    }

    generated_tokens.clear();
    generated_token_probs.clear();
    json_schema = json();

    // clear speculative decoding stats
    n_draft_total = 0;
    n_draft_accepted = 0;

    task_prev = std::move(task);
    task.reset();

    llama_set_sampler(ctx, id, nullptr);

    // clear alora start
    alora_invocation_start = -1;
}

void server_slot::init_sampler() const {
    common_sampler_reset(smpl.get());

    if (!task->need_sampling()) {
        return;
    }

    const int64_t t_start = ggml_time_us();

    int n_text = 0;

    for (int i = 0; i < static_cast<int>(prompt.tokens.size()); i++) {
        const llama_token id = prompt.tokens[i];

        if (id != LLAMA_TOKEN_NULL) {
            common_sampler_accept(smpl.get(), id, false);
            n_text++;
        }
    }

    SLT_INF(
        *this,
        "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
        (ggml_time_us() - t_start) / 1000.0,
        n_text,
        static_cast<int>(prompt.tokens.size()));
}

bool server_slot::can_split() const {
    GGML_ASSERT(task);

    return
        !task->need_embd() ||
        (llama_get_memory(ctx) &&
            llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
}

bool server_slot::can_batch_with(server_slot& other_slot) const {
    GGML_ASSERT(task);

    return task->type == other_slot.task->type &&
        are_lora_equal(lora, other_slot.lora);
}

bool server_slot::has_budget(const common_params& global_params) {
    GGML_ASSERT(task);

    if (task->params.n_predict == -1 && global_params.n_predict == -1) {
        return true; // limitless
    }

    n_remaining = -1;

    if (task->params.n_predict != -1) {
        n_remaining = task->params.n_predict - n_decoded;
    }
    else if (global_params.n_predict != -1) {
        n_remaining = global_params.n_predict - n_decoded;
    }

    return n_remaining > 0;
}

bool server_slot::is_processing() const {
    return state != SLOT_STATE_IDLE;
}

bool server_slot::can_speculate() const {
    return !!spec;
}

void server_slot::add_token(const completion_token_output& token) {
    if (!is_processing()) {
        SLT_WRN(*this, "%s", "slot is not processing\n");
        return;
    }

    generated_token_probs.push_back(token);
}

int server_slot::get_n_draft_max() const {
    GGML_ASSERT(task);

    if (!can_speculate()) {
        return 0;
    }

    // determine the max draft that fits the current slot state
    int n_draft_max = task->params.speculative.n_max;

    // note: slot.prompt is not yet expanded with the `id` token sampled above
    //       also, need to leave space for 1 extra token to allow context shifts
    n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

    if (n_remaining > 0) {
        n_draft_max = std::min(n_draft_max, n_remaining - 1);
    }

    SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

    if (n_draft_max < task->params.speculative.n_min) {
        SLT_DBG(
            *this,
            "the max possible draft is too small: %d < %d - skipping speculative decoding\n",
            n_draft_max,
            task->params.speculative.n_min);

        n_draft_max = 0;
    }

    return n_draft_max;
}

void server_slot::update_batch(llama_batch& batch) {
    const int n_draft_max = get_n_draft_max();

    if (n_draft_max > 0) {
        GGML_ASSERT(can_speculate());

        // generate draft tokens in speculative decoding mode
        // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
        //       perform the speculative drafting for all sequences at the same time in a single batch
        const llama_tokens& tokens = prompt.tokens.get_text_tokens();

        const auto& params_spec = task->params.speculative;

        if (!spec_draft.empty()) {
            // we have a previous (partial) draft to reuse
            if (task->params.speculative.use_checkpoints) {
                GGML_ASSERT(!spec_ckpt.empty());
            }
        }
        else {
            GGML_ASSERT(spec_i_batch.empty());

            // generate a new draft
            spec_draft =
                common_speculative_draft(
                    spec.get(),
                    params_spec,
                    tokens,
                    sampled);

            if (spec_draft.size() > static_cast<size_t>(n_draft_max)) {
                SLT_WRN(
                    *this,
                    "draft size %d exceeds max %d, truncating\n",
                    static_cast<int>(spec_draft.size()),
                    n_draft_max);

                spec_draft.resize(n_draft_max);
            }

            if (spec_draft.size() < static_cast<size_t>(params_spec.n_min)) {
                SLT_DBG(
                    *this,
                    "ignoring small draft: %d < %d\n",
                    static_cast<int>(spec_draft.size()),
                    params_spec.n_min);

                spec_draft.clear();
            }

            if (!spec_draft.empty() && params_spec.use_checkpoints) {
                const auto n_tokens = prompt.tokens.size();

                auto& ckpt = spec_ckpt;

                ckpt = server_get_checkpoint(ctx, this->id, n_tokens);

                SLT_DBG(
                    *this,
                    "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %zu, size = %.3f MiB)\n",
                    ckpt.pos_min,
                    ckpt.pos_max,
                    n_tokens,
                    static_cast<float>(ckpt.data.size()) / 1024 / 1024);
            }
        }

        GGML_ASSERT(spec_draft.size() <= static_cast<size_t>(n_draft_max));
    }

    if (spec_draft.empty()) {
        // no speculative decoding
        i_batch = batch.n_tokens;

        common_batch_add(
            batch,
            sampled,
            prompt.tokens.pos_next(),
            { this->id },
            true);

        SLT_DBG(
            *this,
            "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
            sampled,
            n_ctx,
            prompt.n_tokens(),
            truncated);
    }
    else {
        SLT_DBG(
            *this,
            "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
            sampled,
            prompt.tokens.size(),
            spec_draft.size(),
            prompt.tokens.pos_next());

        GGML_ASSERT(spec_i_batch.empty());

        spec_i_batch.push_back(batch.n_tokens);

        for (size_t i = 0; i < spec_draft.size(); i++) {
            spec_i_batch.push_back(batch.n_tokens + i + 1);
        }

        auto pos0 = prompt.tokens.pos_next();

        common_batch_add(batch, sampled, pos0++, { this->id }, true);

        for (auto token : spec_draft) {
            common_batch_add(batch, token, pos0++, { this->id }, true);
        }
    }

    prompt.tokens.push_back(sampled);
    prompt.tokens.insert(spec_draft);
}

void server_slot::release() {
    if (is_processing()) {
        GGML_ASSERT(task);

        SLT_INF(
            *this,
            "stop processing: n_tokens = %d, truncated = %d\n",
            prompt.n_tokens(),
            truncated);

        t_last_used = ggml_time_us();
        t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

        state = SLOT_STATE_IDLE;

        // do not keep context of the child slots - the parent's context is enough
        if (task->is_child()) {
            prompt_clear(false);
        }

        reset();

        callback_on_release(id);
    }
}

result_timings server_slot::get_timings() const {
    result_timings timings;

    timings.cache_n = n_prompt_tokens_cache;

    timings.prompt_n = n_prompt_tokens_processed;
    timings.prompt_ms = t_prompt_processing;
    timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
    timings.prompt_per_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    timings.predicted_n = n_decoded;
    timings.predicted_ms = t_token_generation;
    timings.predicted_per_token_ms = t_token_generation / n_decoded;
    timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

    // Add speculative metrics
    if (n_draft_total > 0) {
        timings.draft_n = n_draft_total;
        timings.draft_n_accepted = n_draft_accepted;
    }

    return timings;
}

size_t server_slot::find_stopping_strings(
    const std::string& text,
    const size_t last_token_size,
    bool is_full_stop) {
    GGML_ASSERT(task);

    size_t stop_pos = std::string::npos;

    for (const std::string& word : task->params.antiprompt) {
        size_t pos;

        if (is_full_stop) {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

            pos = text.find(word, from_pos);
        }
        else {
            // otherwise, partial stop
            pos = string_find_partial_stop(text, word);
        }

        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos)) {
            if (is_full_stop) {
                stop = STOP_TYPE_WORD;
                stopping_word = word;
                has_next_token = false;
            }

            stop_pos = pos;
        }
    }

    return stop_pos;
}

void server_slot::print_timings() const {
    const double t_prompt = t_prompt_processing / n_prompt_tokens_processed;
    const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    const double t_gen = t_token_generation / n_decoded;
    const double n_gen_second = 1e3 / t_token_generation * n_decoded;

    SLT_INF(
        *this,
        "\n"
        "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
        "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
        "      total time = %10.2f ms / %5d tokens\n",
        t_prompt_processing,
        n_prompt_tokens_processed,
        t_prompt,
        n_prompt_second,
        t_token_generation,
        n_decoded,
        t_gen,
        n_gen_second,
        t_prompt_processing + t_token_generation,
        n_prompt_tokens_processed + n_decoded);

    if (n_draft_total > 0) {
        const float draft_ratio =
            static_cast<float>(n_draft_accepted) / n_draft_total;

        SLT_CNT(
            *this,
            "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
            draft_ratio,
            n_draft_accepted,
            n_draft_total);
    }

    common_speculative_print_stats(spec.get());
}

json server_slot::to_json(bool only_metrics) const {
    json res;

    res = {
        {"id",            id},
        {"n_ctx",         n_ctx},
        {"speculative",   can_speculate()},
        {"is_processing", is_processing()},
    };

    const auto& ptask = task ? task : task_prev;

    if (ptask) {
        res["id_task"] = ptask->id;
        res["params"] = ptask->params.to_json(only_metrics);

        res["next_token"] = {
            {
                {"has_next_token", has_next_token},
                {"has_new_line",   has_new_line},
                {"n_remain",       n_remaining},
                {"n_decoded",      n_decoded},
            }
        };

        if (!only_metrics) {
            res["prompt"] =
                ptask->tokens.detokenize(ctx, true);

            res["generated"] =
                generated_text.empty()
                ? debug_generated_text
                : generated_text;
        }
    }

    return res;
}

void server_slot::copy_state_to(server_slot& other) const {
    GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

    llama_memory_seq_rm(llama_get_memory(ctx), other.id, -1, -1);
    llama_memory_seq_cp(llama_get_memory(ctx), id, other.id, -1, -1);

    other.n_decoded = n_decoded;
    other.n_remaining = n_remaining;
    other.i_batch = i_batch;

    other.t_start_process_prompt = t_start_process_prompt;
    other.t_prompt_processing = t_prompt_processing;
    other.n_prompt_tokens_cache = n_prompt_tokens_cache;
    other.n_prompt_tokens_processed = n_prompt_tokens_processed;

    other.prompt = prompt.clone();
    other.init_sampler();
}