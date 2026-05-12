#pragma once

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

server_prompt_checkpoint server_get_checkpoint(
    llama_context* ctx,
    int id,
    int64_t n_tokens,
    llama_pos pos_min = -1,
    llama_pos pos_max = -1);

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot {
    int id = 0;

    // TODO: change to unique_ptrs for consistency:
    llama_context* ctx = nullptr;

    // multimodal
    mtmd_context* mctx = nullptr;

    // speculative decoding
    llama_tokens spec_draft;
    std::vector<int32_t> spec_i_batch;
    server_prompt_checkpoint spec_ckpt;
    common_speculative_ptr spec;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx = 0;  // context size per slot
    int32_t n_keep = 0;
    int32_t n_decoded = 0;
    int32_t n_remaining = -1;
    int32_t i_batch = -1;

    int32_t n_prompt_tokens_cache = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string generated_text;
    std::string debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line = false;
    bool truncated = false;

    stop_type stop = STOP_TYPE_NONE;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled = LLAMA_TOKEN_NULL; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_start_process_prompt = 0;
    int64_t t_start_generation = 0;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0; // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0; // Total draft tokens generated
    int32_t n_draft_accepted = 0; // Draft tokens actually accepted

    void prompt_save(server_prompt_cache& prompt_cache) const;
    bool prompt_load(server_prompt_cache& prompt_cache, const server_tokens& tokens);
    void prompt_clear(bool allow_processing);

    void reset();
    void init_sampler() const;

    bool can_split() const;
    bool can_batch_with(server_slot& other_slot) const;
    bool has_budget(const common_params& global_params);
    bool is_processing() const;
    bool can_speculate() const;

    void add_token(const completion_token_output& token);

    int get_n_draft_max() const;
    void update_batch(llama_batch& batch);

    void release();

    result_timings get_timings() const;

    size_t find_stopping_strings(
        const std::string& text,
        size_t last_token_size,
        bool is_full_stop);

    void print_timings() const;

    json to_json(bool only_metrics = false) const;

    void copy_state_to(server_slot& other) const;
};