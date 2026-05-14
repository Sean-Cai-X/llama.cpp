#pragma once

#include "common.h"
#include "rag_clips_runner.h"
#include "llama-cpp.h"
#include "rag_clips_meta.h"
#include "rag.h"

#include <nlohmann/json_fwd.hpp>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

struct RagTraceContext {
    std::string request_id;
    std::string trace_id;
    std::string query_id;
    std::string route;
    std::string source_request_id;
    int top_k = 0;
};

class RagServerRuntime {
public:
    RagServerRuntime();
    ~RagServerRuntime();

    bool init(common_params & params, llama_model * model, llama_context * ctx_main, const llama_vocab * vocab);
    void shutdown();

    bool enabled() const;
    bool ready() const;

    bool enqueue_documents(
        const std::vector<std::string> & docs,
        const std::vector<std::string> & metadata,
        bool reset_before_add,
        std::string * error_message = nullptr);

    bool enqueue_repo_index(
        const std::string & repo_path,
        bool reset_before_add,
        std::string * error_message = nullptr);

    std::vector<RagSearchResult> search_with_metadata(const std::string & query, int top_k, int timeout_ms) const;

    json build_search_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const;
    json build_explain_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const;
    json build_chat_context_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const;
    json build_clips_meta_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const;
    json build_clips_manifest_payload() const;
    json build_clips_run_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const;
    bool inject_chat_context(json & body, const std::string & query, int top_k, int timeout_ms) const;

    json build_status_payload() const;

    static std::string extract_query_from_body(const json & body);

private:
    struct ingest_job {
        std::vector<std::string> docs;
        std::vector<std::string> metadata;
        bool reset_before_add = false;
        std::string kind = "documents";
    };

    void start_worker();
    void worker_loop();
    RagTraceContext build_trace_context(const json & body, const std::string & route, const std::string & query, int top_k) const;

    common_params params_base_;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;

    common_init_result_ptr llama_init_embed_;
    llama_context_ptr ctx_embed_owned_;
    llama_context * ctx_embed_ = nullptr;

    std::unique_ptr<RagEngine> rag_engine_;

    mutable std::mutex queue_mutex_;
    mutable std::condition_variable queue_cv_;
    mutable std::queue<ingest_job> pending_jobs_;
    std::thread worker_;
    bool running_ = false;

    mutable std::mutex status_mutex_;
    std::string last_error_;
    std::string last_job_kind_ = "none";
    bool last_reset_before_add_ = false;

    std::atomic<int> pending_count_ {0};
    std::atomic<bool> job_active_ {false};
    std::atomic<int> jobs_completed_ {0};
    std::atomic<int> docs_completed_ {0};
    std::atomic<int> metadata_completed_ {0};
};
