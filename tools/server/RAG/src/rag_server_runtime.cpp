#include "rag_server_runtime.h"

#include "log.h"
#include "rag_metadata.h"
#include "repo_scanner.h"
#include "server-trace-registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

std::atomic<uint64_t> g_rag_trace_counter {0};

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string sanitize_rag_preview(std::string text, size_t max_chars = 240) {
    text = trim_copy(text);
    for (char & ch : text) {
        if (ch == '\r' || ch == '\n' || ch == '\t') {
            ch = ' ';
        }
    }
    if (text.size() > max_chars) {
        text.resize(max_chars);
        text += "...";
    }
    return text;
}

std::string format_rag_source_label(const json & metadata) {
    const std::string path = metadata.value("source_uri", metadata.value("path", ""));
    const std::string lines_begin = rag_metadata_value_string(metadata, "start_line");
    const std::string lines_end = rag_metadata_value_string(metadata, "end_line");
    if (!path.empty() && !lines_begin.empty() && !lines_end.empty()) {
        return path + ":" + lines_begin + "-" + lines_end;
    }
    if (!path.empty()) {
        return path;
    }
    return "retrieved-context";
}

std::string get_string_or_empty(const json & body, const char * key) {
    if (!body.contains(key) || !body.at(key).is_string()) {
        return "";
    }
    return body.at(key).get<std::string>();
}

std::string generate_trace_token(const char * prefix) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
    const uint64_t seq = ++g_rag_trace_counter;
    return std::string(prefix) + "-" + std::to_string(now_ms) + "-" + std::to_string(seq);
}

json build_trace_payload(const RagTraceContext & trace, bool rag_enabled, bool clips_enabled) {
    return json{
        {"request_id", trace.request_id},
        {"trace_id", trace.trace_id},
        {"query_id", trace.query_id},
        {"route", trace.route},
        {"source_request_id", trace.source_request_id},
        {"top_k", trace.top_k},
        {"rag_enabled", rag_enabled},
        {"clips_enabled", clips_enabled},
    };
}

std::string extract_text_from_message_content(const json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        return "";
    }

    std::string text;
    for (const auto & item : content) {
        if (!item.is_object()) {
            continue;
        }
        if (item.value("type", "") == "text" && item.contains("text") && item["text"].is_string()) {
            if (!text.empty()) {
                text += '\n';
            }
            text += item["text"].get<std::string>();
        }
    }
    return text;
}

std::string build_approved_context_block(const json & approved_context) {
    if (!approved_context.is_array() || approved_context.empty()) {
        return "";
    }

    std::string block = "CLIPS-approved project context:\n";
    for (size_t i = 0; i < approved_context.size(); ++i) {
        const json & item = approved_context.at(i);
        block += "\n[" + std::to_string(i + 1) + "] " + item.value("source", "approved-context") + "\n";
        block += item.value("preview_text", "") + "\n";
    }
    return block;
}

json build_source_info(const json & metadata) {
    return json{
        {"path", metadata.value("source_uri", metadata.value("path", ""))},
        {"language", metadata.value("language", "")},
        {"start_line", rag_metadata_value_int(metadata, "start_line", 0)},
        {"end_line", rag_metadata_value_int(metadata, "end_line", 0)},
    };
}

std::string map_clips_decision_route(const std::string & decision) {
    if (decision == "allow") {
        return "admit_to_cognitive_layer";
    }
    if (decision == "reject") {
        return "drop_result";
    }
    if (decision == "repair") {
        return "repair_then_retry";
    }
    if (decision == "short_term_only") {
        return "session_only";
    }
    if (decision == "require_human") {
        return "human_review";
    }
    return "no_action";
}

json build_admission_decision_item(const RagClipsAdmissionDecision & decision) {
    return json{
        {"slice_id", decision.slice_id},
        {"decision", decision.decision},
        {"reason", decision.reason},
        {"rule_id", decision.rule_id},
        {"next_action", decision.next_action},
        {"raw_fact", decision.raw_fact},
    };
}

json build_approved_context_items(
    const std::vector<RagSearchResult> & results,
    const RagClipsFactBundle & bundle,
    const std::vector<RagClipsAdmissionDecision> & decisions,
    const RagTraceContext & trace) {
    std::unordered_map<std::string, size_t> result_index_by_slice_id;
    std::unordered_map<std::string, const RagMetaSliceLink *> link_by_slice_id;
    std::unordered_set<std::string> emitted_slice_ids;

    for (size_t i = 0; i < bundle.slice_links.size(); ++i) {
        const auto & link = bundle.slice_links[i];
        link_by_slice_id[link.slice_id] = &link;
        if (i < results.size()) {
            result_index_by_slice_id[link.slice_id] = i;
        }
    }

    json approved = json::array();
    for (const auto & decision : decisions) {
        if (decision.decision != "allow") {
            continue;
        }
        if (!emitted_slice_ids.insert(decision.slice_id).second) {
            continue;
        }

        const auto link_it = link_by_slice_id.find(decision.slice_id);
        if (link_it == link_by_slice_id.end()) {
            continue;
        }
        const auto result_it = result_index_by_slice_id.find(decision.slice_id);
        if (result_it == result_index_by_slice_id.end() || result_it->second >= results.size()) {
            continue;
        }

        const RagMetaSliceLink & link = *link_it->second;
        const RagSearchResult & result = results[result_it->second];
        const json metadata = rag_parse_metadata(result.metadata);
        approved.push_back(json{
            {"context_id", "CTX-" + decision.slice_id},
            {"request_id", trace.request_id},
            {"trace_id", trace.trace_id},
            {"query_id", trace.query_id},
            {"source_type", link.source_type},
            {"source_id", decision.slice_id},
            {"slice_id", decision.slice_id},
            {"link_id", link.link_id},
            {"decision", decision.decision},
            {"reason", decision.reason},
            {"rule_id", decision.rule_id},
            {"next_action", decision.next_action},
            {"priority", link.info_weight},
            {"score", result.score},
            {"approved", true},
            {"projection_incomplete", link.projection_incomplete},
            {"vector_skip_reason", link.vector_skip_reason},
            {"browser_visible_summary", link.browser_visible_summary},
            {"body_quality", link.body_quality},
            {"source", format_rag_source_label(metadata)},
            {"source_info", build_source_info(metadata)},
            {"preview_text", sanitize_rag_preview(result.chunk_text, 480)},
            {"chunk_text", result.chunk_text},
        });
    }

    return approved;
}

} // namespace

RagServerRuntime::RagServerRuntime() = default;

RagServerRuntime::~RagServerRuntime() {
    shutdown();
}

bool RagServerRuntime::init(common_params & params, llama_model * model, llama_context * ctx_main, const llama_vocab * vocab) {
    shutdown();

    params_base_ = params;
    model_ = model;
    vocab_ = vocab;

    if (!params_base_.rag_enable || model_ == nullptr || ctx_main == nullptr || vocab_ == nullptr) {
        return false;
    }

    ctx_embed_ = nullptr;

    if (!params_base_.model_embed_path.empty()) {
        auto embed_params = params_base_;
        embed_params.model.path = params_base_.model_embed_path;
        embed_params.n_gpu_layers = params_base_.rag_embed_n_gpu_layers;
        embed_params.n_ctx = params_base_.rag_embed_ctx_size;
        embed_params.n_batch = params_base_.rag_embed_batch;
        embed_params.n_ubatch = params_base_.rag_embed_ubatch;
        embed_params.embedding = true;
        embed_params.pooling_type = params_base_.rag_embed_pooling_type;
        embed_params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        embed_params.n_parallel = 1;
        embed_params.mmproj.path.clear();
        embed_params.mmproj.url.clear();
        embed_params.no_mmproj = true;

        llama_init_embed_ = common_init_from_params(embed_params);
        if (llama_init_embed_ && llama_init_embed_->model() && llama_init_embed_->context()) {
            ctx_embed_ = llama_init_embed_->context();
            LOG_INF("%s: loaded dedicated embedding model '%s' (ngl=%d, ctx=%d, batch=%d, ubatch=%d, pooling=%d)\n",
                __func__,
                params_base_.model_embed_path.c_str(),
                embed_params.n_gpu_layers,
                embed_params.n_ctx,
                embed_params.n_batch,
                embed_params.n_ubatch,
                static_cast<int>(embed_params.pooling_type));
        } else {
            LOG_WRN("%s: failed to load dedicated embedding model, falling back to the main model when possible\n", __func__);
        }
    }

    if (ctx_embed_ == nullptr) {
        if (llama_pooling_type(ctx_main) == LLAMA_POOLING_TYPE_NONE) {
            LOG_ERR("%s: RAG requested but the main model does not expose embeddings and no dedicated embedding model is configured\n", __func__);
            return false;
        }

        auto embed_params = params_base_;
        embed_params.embedding = true;
        embed_params.pooling_type = LLAMA_POOLING_TYPE_LAST;
        embed_params.n_parallel = 1;

        auto cparams_embed = common_context_params_to_llama(embed_params);
        ctx_embed_owned_.reset(llama_init_from_model(model_, cparams_embed));
        if (!ctx_embed_owned_) {
            LOG_ERR("%s: failed to create a dedicated embedding context from the main model\n", __func__);
            return false;
        }

        ctx_embed_ = ctx_embed_owned_.get();
        LOG_INF("%s: created a dedicated embedding context from the main model\n", __func__);
    }

    RagConfig rag_config;
    rag_config.index_path = params_base_.rag_index_path;
    rag_config.top_k = params_base_.rag_top_k;
    rag_config.chunk_size = params_base_.rag_chunk_size;
    rag_config.chunk_overlap = params_base_.rag_chunk_overlap;
    rag_config.use_hybrid = params_base_.rag_hybrid_search;

    rag_engine_ = std::make_unique<RagEngine>(rag_config);
    const llama_model * rag_model = llama_init_embed_ ? llama_init_embed_->model() : model_;
    if (!rag_engine_->init(ctx_embed_, rag_model, vocab_)) {
        LOG_ERR("%s: failed to initialize RAG engine\n", __func__);
        rag_engine_.reset();
        return false;
    }

    server_trace_registry::configure_storage(params_base_.rag_index_path);

    start_worker();
    params = params_base_;
    return true;
}

void RagServerRuntime::shutdown() {
    if (running_) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_ = false;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    if (rag_engine_) {
        rag_engine_->save();
        rag_engine_.reset();
    }

    llama_init_embed_.reset();
    ctx_embed_owned_.reset();
    ctx_embed_ = nullptr;
    model_ = nullptr;
    vocab_ = nullptr;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<ingest_job> empty;
        std::swap(pending_jobs_, empty);
    }

    pending_count_ = 0;
    job_active_ = false;
}

bool RagServerRuntime::enabled() const {
    return params_base_.rag_enable && rag_engine_ != nullptr;
}

bool RagServerRuntime::ready() const {
    return rag_engine_ && rag_engine_->is_ready();
}

void RagServerRuntime::start_worker() {
    running_ = true;
    worker_ = std::thread([this]() { worker_loop(); });
}

void RagServerRuntime::worker_loop() {
    while (true) {
        ingest_job job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !pending_jobs_.empty() || !running_;
            });

            if (!running_ && pending_jobs_.empty()) {
                break;
            }

            job = std::move(pending_jobs_.front());
            pending_jobs_.pop();
        }

        pending_count_--;
        job_active_ = true;
        try {
            if (rag_engine_) {
                if (job.reset_before_add) {
                    rag_engine_->clear();
                }
                rag_engine_->add_documents(job.docs, job.metadata);
                // Persist ingest results immediately so sidecar stores are available
                // to status checks and external inspection without waiting for shutdown.
                rag_engine_->save();
            }

            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                last_error_.clear();
                last_job_kind_ = job.kind;
                last_reset_before_add_ = job.reset_before_add;
            }

            jobs_completed_++;
            docs_completed_ += (int) job.docs.size();
            metadata_completed_ += (int) job.metadata.size();
        } catch (const std::exception & e) {
            std::lock_guard<std::mutex> lock(status_mutex_);
            last_error_ = e.what();
            last_job_kind_ = job.kind;
            last_reset_before_add_ = job.reset_before_add;
        }
        job_active_ = false;
    }
}

bool RagServerRuntime::enqueue_documents(
    const std::vector<std::string> & docs,
    const std::vector<std::string> & metadata,
    bool reset_before_add,
    std::string * error_message) {
    if (!enabled()) {
        if (error_message) {
            *error_message = "RAG is not enabled";
        }
        return false;
    }

    ingest_job job;
    job.docs = docs;
    job.metadata = metadata;
    job.reset_before_add = reset_before_add;
    job.kind = "documents";

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_jobs_.push(std::move(job));
        pending_count_++;
    }
    queue_cv_.notify_one();
    return true;
}

bool RagServerRuntime::enqueue_repo_index(
    const std::string & repo_path,
    bool reset_before_add,
    std::string * error_message) {
    if (!enabled()) {
        if (error_message) {
            *error_message = "RAG is not enabled";
        }
        return false;
    }

    try {
        RepoScannerResult scan = RepoScanner::scan(repo_path);
        std::vector<std::string> docs;
        std::vector<std::string> metadata;
        docs.reserve(scan.chunks.size());
        metadata.reserve(scan.chunks.size());
        for (const auto & chunk : scan.chunks) {
            docs.push_back(chunk.text);
            metadata.push_back(chunk.metadata);
        }
        return enqueue_documents(docs, metadata, reset_before_add, error_message);
    } catch (const std::exception & e) {
        if (error_message) {
            *error_message = e.what();
        }
        return false;
    }
}

std::vector<RagSearchResult> RagServerRuntime::search_with_metadata(const std::string & query, int top_k, int timeout_ms) const {
    if (!enabled() || query.empty()) {
        return {};
    }
    return rag_engine_->search_with_metadata(query, top_k, timeout_ms);
}

RagTraceContext RagServerRuntime::build_trace_context(const json & body, const std::string & route, const std::string & query, int top_k) const {
    RagTraceContext trace;
    trace.request_id = get_string_or_empty(body, "request_id");
    if (trace.request_id.empty()) {
        trace.request_id = get_string_or_empty(body, "codex_request_id");
    }
    if (trace.request_id.empty()) {
        trace.request_id = generate_trace_token("REQ");
    }

    trace.trace_id = get_string_or_empty(body, "trace_id");
    if (trace.trace_id.empty()) {
        trace.trace_id = trace.request_id + ".trace";
    }

    trace.query_id = get_string_or_empty(body, "query_id");
    if (trace.query_id.empty()) {
        trace.query_id = generate_trace_token("QUERY");
    }

    trace.source_request_id = get_string_or_empty(body, "codex_request_id");
    if (trace.source_request_id.empty()) {
        trace.source_request_id = trace.request_id;
    }

    trace.route = route;
    trace.top_k = top_k;
    (void) query;
    return trace;
}

json RagServerRuntime::build_search_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const {
    const auto results = search_with_metadata(query, top_k, timeout_ms);
    const RagTraceContext trace = build_trace_context(body, "/rag/search", query, top_k);
    json items = json::array();

    for (size_t i = 0; i < results.size(); ++i) {
        const json metadata = rag_parse_metadata(results[i].metadata);
        items.push_back(json{
            {"rank", (int) i + 1},
            {"score", results[i].score},
            {"chunk_text", results[i].chunk_text},
            {"preview_text", sanitize_rag_preview(results[i].chunk_text)},
            {"metadata", metadata},
            {"source", format_rag_source_label(metadata)},
            {"source_info", build_source_info(metadata)},
        });
    }

    return json{
        {"request_id", trace.request_id},
        {"trace_id", trace.trace_id},
        {"query_id", trace.query_id},
        {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
        {"query", query},
        {"retrieval_mode", params_base_.rag_hybrid_search ? "hybrid" : "dense"},
        {"count", (int) items.size()},
        {"results", std::move(items)},
    };
}

json RagServerRuntime::build_explain_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const {
    const auto results = search_with_metadata(query, top_k, timeout_ms);
    const RagTraceContext trace = build_trace_context(body, "/rag/explain", query, top_k);
    json evidence = json::array();

    for (size_t i = 0; i < results.size(); ++i) {
        const json metadata = rag_parse_metadata(results[i].metadata);
        evidence.push_back(json{
            {"rank", (int) i + 1},
            {"score", results[i].score},
            {"source", format_rag_source_label(metadata)},
            {"source_info", build_source_info(metadata)},
            {"preview", sanitize_rag_preview(results[i].chunk_text, 320)},
        });
    }

    return json{
        {"request_id", trace.request_id},
        {"trace_id", trace.trace_id},
        {"query_id", trace.query_id},
        {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
        {"query", query},
        {"retrieval_mode", params_base_.rag_hybrid_search ? "hybrid" : "dense"},
        {"summary", results.empty() ? "No relevant project context was retrieved." : "Retrieved project context is ready for downstream reasoning."},
        {"evidence", std::move(evidence)},
    };
}

json RagServerRuntime::build_chat_context_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const {
    const auto results = search_with_metadata(query, top_k, timeout_ms);
    const std::string retrieval_mode = params_base_.rag_hybrid_search ? "hybrid" : "dense";
    const RagTraceContext trace = build_trace_context(body, "/rag/chat/context", query, top_k);
    const RagClipsFactBundle bundle = BuildRagClipsFactBundle(query, results, retrieval_mode);
    const auto assertions = SerializeRagClipsFacts(bundle);
    const RagClipsRunResult run_result = RunRagClipsRules(params_base_, bundle, assertions);
    const json approved_context = build_approved_context_items(results, bundle, run_result.admission_report, trace);
    const std::string route = map_clips_decision_route(run_result.dominant_decision);

    return json{
        {"request_id", trace.request_id},
        {"trace_id", trace.trace_id},
        {"query_id", trace.query_id},
        {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
        {"query", query},
        {"retrieval_mode", retrieval_mode},
        {"ok", run_result.ok},
        {"message", run_result.message},
        {"raw_result_count", static_cast<int>(results.size())},
        {"approved_context_count", static_cast<int>(approved_context.size())},
        {"approved_context", approved_context},
        {"context_block", build_approved_context_block(approved_context)},
        {"admission_report", [run_result]() {
            json report = json::array();
            for (const auto & decision : run_result.admission_report) {
                report.push_back(build_admission_decision_item(decision));
            }
            return report;
        }()},
        {"admission_summary", {
            {"allow_count", run_result.allow_count},
            {"reject_count", run_result.reject_count},
            {"repair_count", run_result.repair_count},
            {"short_term_only_count", run_result.short_term_only_count},
            {"require_human_count", run_result.require_human_count},
            {"dominant_decision", run_result.dominant_decision},
            {"route", route},
        }},
        {"status", build_status_payload()},
    };
}

json RagServerRuntime::build_clips_meta_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const {
    const auto results = search_with_metadata(query, top_k, timeout_ms);
    const std::string retrieval_mode = params_base_.rag_hybrid_search ? "hybrid" : "dense";
    const RagTraceContext trace = build_trace_context(body, "/rag/clips/meta", query, top_k);
    const RagClipsFactBundle bundle = BuildRagClipsFactBundle(query, results, retrieval_mode);
    const auto assertions = SerializeRagClipsFacts(bundle);

    json nodes = json::array();
    for (const auto & node : bundle.nodes) {
        nodes.push_back(json{
            {"node_id", node.node_id},
            {"parent_id", node.parent_id},
            {"anchor_level", node.anchor_level},
            {"is_immutable", node.is_immutable},
            {"node_type", node.node_type},
            {"lifecycle", node.lifecycle},
            {"domain_name", node.domain_name},
            {"concept_name", node.concept_name},
            {"core_definition", node.core_definition},
        });
    }

    json slice_links = json::array();
    for (const auto & link : bundle.slice_links) {
        slice_links.push_back(json{
            {"link_id", link.link_id},
            {"slice_id", link.slice_id},
            {"bind_node_id", link.bind_node_id},
            {"info_weight", link.info_weight},
            {"truth_status", link.truth_status},
            {"is_reversible", link.is_reversible},
            {"similarity_to_core", link.similarity_to_core},
            {"slice_text", link.slice_text},
            {"embedding_hash", link.embedding_hash},
            {"dedup_hash", link.dedup_hash},
            {"source_type", link.source_type},
            {"provider_id", link.provider_id},
            {"evidence_ref", link.evidence_ref},
            {"body_quality", link.body_quality},
            {"projection_incomplete", link.projection_incomplete},
            {"vector_skip_reason", link.vector_skip_reason},
            {"browser_visible_summary", link.browser_visible_summary},
        });
    }

    json inference_rules = json::array();
    for (const auto & rule : bundle.inference_rules) {
        inference_rules.push_back(json{
            {"rule_id", rule.rule_id},
            {"inference_type", rule.inference_type},
            {"source_node_type", rule.source_node_type},
            {"target_node_type", rule.target_node_type},
            {"max_diffuse_depth", rule.max_diffuse_depth},
            {"max_slice_per_node", rule.max_slice_per_node},
            {"min_similarity_threshold", rule.min_similarity_threshold},
            {"is_enabled", rule.is_enabled},
        });
    }

    json graph_nodes = json::array();
    for (const auto & graph_node : bundle.meta_graph.nodes) {
        graph_nodes.push_back(json{
            {"graph_node_id", graph_node.graph_node_id},
            {"node_kind", graph_node.node_kind},
            {"parent_graph_node_id", graph_node.parent_graph_node_id},
            {"label", graph_node.label},
            {"ref_id", graph_node.ref_id},
            {"depth", graph_node.depth},
        });
    }

    json graph_edges = json::array();
    for (const auto & graph_edge : bundle.meta_graph.edges) {
        graph_edges.push_back(json{
            {"edge_id", graph_edge.edge_id},
            {"edge_type", graph_edge.edge_type},
            {"from_node_id", graph_edge.from_node_id},
            {"to_node_id", graph_edge.to_node_id},
            {"relation", graph_edge.relation},
        });
    }

    int meta_root_count = 0;
    int domain_count = 0;
    int file_count = 0;
    int concept_count = 0;
    int slice_count = 0;
    int inference_rule_count = 0;
    for (const auto & graph_node : bundle.meta_graph.nodes) {
        if (graph_node.node_kind == "meta_root") {
            meta_root_count++;
        } else if (graph_node.node_kind == "domain") {
            domain_count++;
        } else if (graph_node.node_kind == "file") {
            file_count++;
        } else if (graph_node.node_kind == "concept") {
            concept_count++;
        } else if (graph_node.node_kind == "slice") {
            slice_count++;
        } else if (graph_node.node_kind == "inference_rule") {
            inference_rule_count++;
        }
    }

    json serialized_assertions = json::array();
    for (const auto & assertion : assertions) {
        serialized_assertions.push_back(assertion);
    }

    return json{
        {"request_id", trace.request_id},
        {"trace_id", trace.trace_id},
        {"query_id", trace.query_id},
        {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
        {"query", query},
        {"retrieval_mode", retrieval_mode},
        {"result_count", static_cast<int>(results.size())},
        {"fact_bundle", {
            {"nodes", std::move(nodes)},
            {"slice_links", std::move(slice_links)},
            {"inference_rules", std::move(inference_rules)},
        }},
        {"meta_tree_graph", {
            {"nodes", std::move(graph_nodes)},
            {"edges", std::move(graph_edges)},
        }},
        {"graph_summary", {
            {"meta_root_count", meta_root_count},
            {"domain_count", domain_count},
            {"file_count", file_count},
            {"concept_count", concept_count},
            {"slice_count", slice_count},
            {"inference_rule_count", inference_rule_count},
            {"edge_count", static_cast<int>(bundle.meta_graph.edges.size())},
        }},
        {"serialized_assertions", std::move(serialized_assertions)},
    };
}

json RagServerRuntime::build_clips_manifest_payload() const {
    const RagClipsFactBundle empty_bundle;
    const std::vector<std::string> empty_assertions;
    return BuildRagClipsManifestPayload(params_base_, empty_bundle, empty_assertions);
}

json RagServerRuntime::build_clips_run_payload(const json & body, const std::string & query, int top_k, int timeout_ms) const {
    const RagTraceContext trace = build_trace_context(body, "/rag/clips/run", query, top_k);
    try {
        LOG_INF("%s: building CLIPS run payload query='%s' top_k=%d timeout_ms=%d\n",
            __func__, query.c_str(), top_k, timeout_ms);
        const auto results = search_with_metadata(query, top_k, timeout_ms);
        const std::string retrieval_mode = params_base_.rag_hybrid_search ? "hybrid" : "dense";
        const RagClipsFactBundle bundle = BuildRagClipsFactBundle(query, results, retrieval_mode);
        const auto assertions = SerializeRagClipsFacts(bundle);
        const RagClipsRunResult run_result = RunRagClipsRules(params_base_, bundle, assertions);
        LOG_INF("%s: CLIPS run finished ok=%s dominant_decision='%s' result_count=%d\n",
            __func__,
            run_result.ok ? "true" : "false",
            run_result.dominant_decision.c_str(),
            static_cast<int>(results.size()));

        const std::string route = map_clips_decision_route(run_result.dominant_decision);
        const bool has_blocking_issue =
            run_result.reject_count > 0 ||
            run_result.repair_count > 0 ||
            run_result.require_human_count > 0;
        const bool can_admit_directly =
            run_result.allow_count > 0 &&
            !has_blocking_issue &&
            run_result.short_term_only_count == 0;
        const bool short_term_only =
            run_result.short_term_only_count > 0 &&
            !has_blocking_issue;
        json admission_report = json::array();
        json accepted = json::array();
        json rejected = json::array();
        json repair_queue = json::array();
        json short_term_only_queue = json::array();
        json human_review_queue = json::array();
        for (const auto & decision : run_result.admission_report) {
            json item = build_admission_decision_item(decision);
            admission_report.push_back(item);
            if (decision.decision == "allow") {
                accepted.push_back(item);
            } else if (decision.decision == "reject") {
                rejected.push_back(item);
            } else if (decision.decision == "repair") {
                repair_queue.push_back(item);
            } else if (decision.decision == "short_term_only") {
                short_term_only_queue.push_back(item);
            } else if (decision.decision == "require_human") {
                human_review_queue.push_back(item);
            }
        }
        const int accepted_count = static_cast<int>(accepted.size());
        const int rejected_count = static_cast<int>(rejected.size());
        const int repair_count = static_cast<int>(repair_queue.size());
        const int short_term_only_count = static_cast<int>(short_term_only_queue.size());
        const int human_review_count = static_cast<int>(human_review_queue.size());

        LOG_INF("%s: assembling compact CLIPS run payload route='%s'\n", __func__, route.c_str());

        return json{
            {"request_id", trace.request_id},
            {"trace_id", trace.trace_id},
            {"query_id", trace.query_id},
            {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
            {"query", query},
            {"retrieval_mode", retrieval_mode},
            {"result_count", static_cast<int>(results.size())},
            {"backend", run_result.backend},
            {"ok", run_result.ok},
            {"message", run_result.message},
            {"rules_dir", run_result.rules_dir},
            {"manifest_path", run_result.manifest_path},
            {"rule_set_id", run_result.rule_set_id},
            {"engine_config", {
                {"storage_backend", "memory"},
                {"rocksdb_enabled", false},
                {"memory_pool_mb", run_result.memory_pool_mb},
                {"batch_fact_limit", run_result.batch_fact_limit},
            }},
            {"admission_report", std::move(admission_report)},
            {"admission_layer_input", {
                {"version", 1},
                {"query", query},
                {"retrieval_mode", retrieval_mode},
                {"result_count", static_cast<int>(results.size())},
                {"dominant_decision", run_result.dominant_decision},
                {"route", route},
                {"can_admit_directly", can_admit_directly},
                {"requires_repair", run_result.repair_count > 0},
                {"requires_human_review", run_result.require_human_count > 0},
                {"allow_short_term_memory_only", short_term_only},
                {"drop_from_admission", run_result.reject_count > 0},
                {"has_unresolved_slices", false},
                {"recommended_next_action", route},
                {"queue_counts", {
                    {"accepted", accepted_count},
                    {"rejected", rejected_count},
                    {"repair", repair_count},
                    {"short_term_only", short_term_only_count},
                    {"human_review", human_review_count},
                    {"undecided", 0},
                }},
                {"accepted", std::move(accepted)},
                {"rejected", std::move(rejected)},
                {"repair_queue", std::move(repair_queue)},
                {"short_term_only_queue", std::move(short_term_only_queue)},
                {"human_review_queue", std::move(human_review_queue)},
                {"undecided", json::array()},
            }},
            {"facts_asserted", run_result.facts_asserted},
            {"facts_after_run_count", run_result.facts_after_run_count},
            {"activations_before_run_count", run_result.activations_before_run_count},
            {"rules_fired", run_result.rules_fired},
            {"admission_summary", {
                {"allow_count", run_result.allow_count},
                {"reject_count", run_result.reject_count},
                {"repair_count", run_result.repair_count},
                {"short_term_only_count", run_result.short_term_only_count},
                {"require_human_count", run_result.require_human_count},
                {"dominant_decision", run_result.dominant_decision},
            }},
            {"runner_diagnostics", {
                {"loaded_rule_file_count", static_cast<int>(run_result.loaded_rule_files.size())},
                {"activation_count", run_result.activations_before_run_count},
                {"fact_count_after_run", run_result.facts_after_run_count},
                {"rules_fired", run_result.rules_fired},
                {"facts_rejected_by_limit", run_result.facts_rejected_by_limit},
            }},
        };
    } catch (const std::exception & e) {
        LOG_ERR("%s: CLIPS run threw exception: %s\n", __func__, e.what());
        return json{
            {"request_id", trace.request_id},
            {"trace_id", trace.trace_id},
            {"query_id", trace.query_id},
            {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
            {"query", query},
            {"retrieval_mode", params_base_.rag_hybrid_search ? "hybrid" : "dense"},
            {"result_count", 0},
            {"backend", "clips-core"},
            {"ok", false},
            {"message", std::string("CLIPS run failed: ") + e.what()},
            {"admission_decisions", json::array()},
            {"admission_report", json::array()},
            {"admission_layer_input", {
                {"version", 1},
                {"query", query},
                {"retrieval_mode", params_base_.rag_hybrid_search ? "hybrid" : "dense"},
                {"result_count", 0},
                {"dominant_decision", "require_human"},
                {"route", "human_review"},
                {"can_admit_directly", false},
                {"requires_repair", false},
                {"requires_human_review", true},
                {"allow_short_term_memory_only", false},
                {"drop_from_admission", false},
                {"has_unresolved_slices", true},
                {"recommended_next_action", "human_review"},
                {"queue_counts", {
                    {"accepted", 0},
                    {"rejected", 0},
                    {"repair", 0},
                    {"short_term_only", 0},
                    {"human_review", 1},
                    {"undecided", 0},
                }},
                {"accepted", json::array()},
                {"rejected", json::array()},
                {"repair_queue", json::array()},
                {"short_term_only_queue", json::array()},
                {"human_review_queue", json::array({
                    json{
                        {"slice_id", "query-context"},
                        {"decision", "require_human"},
                        {"reason", "clips-run-exception"},
                        {"rule_id", "runtime.clips.exception"},
                        {"next_action", "collect_more_context"},
                        {"raw_fact", e.what()},
                    }
                })},
                {"undecided", json::array()},
            }},
            {"facts_asserted", 0},
            {"facts_after_run_count", 0},
            {"activations_before_run_count", 0},
            {"rules_fired", 0},
            {"admission_summary", {
                {"allow_count", 0},
                {"reject_count", 0},
                {"repair_count", 0},
                {"short_term_only_count", 0},
                {"require_human_count", 1},
                {"dominant_decision", "require_human"},
            }},
            {"manifest_snapshot", json::object()},
        };
    } catch (...) {
        LOG_ERR("%s: CLIPS run threw non-standard exception\n", __func__);
        return json{
            {"request_id", trace.request_id},
            {"trace_id", trace.trace_id},
            {"query_id", trace.query_id},
            {"request_context", build_trace_payload(trace, enabled(), params_base_.rag_clips_enable)},
            {"query", query},
            {"retrieval_mode", params_base_.rag_hybrid_search ? "hybrid" : "dense"},
            {"result_count", 0},
            {"backend", "clips-core"},
            {"ok", false},
            {"message", "CLIPS run failed: non-standard exception"},
            {"admission_decisions", json::array()},
            {"admission_report", json::array()},
            {"admission_layer_input", {
                {"version", 1},
                {"query", query},
                {"retrieval_mode", params_base_.rag_hybrid_search ? "hybrid" : "dense"},
                {"result_count", 0},
                {"dominant_decision", "require_human"},
                {"route", "human_review"},
                {"can_admit_directly", false},
                {"requires_repair", false},
                {"requires_human_review", true},
                {"allow_short_term_memory_only", false},
                {"drop_from_admission", false},
                {"has_unresolved_slices", true},
                {"recommended_next_action", "human_review"},
                {"queue_counts", {
                    {"accepted", 0},
                    {"rejected", 0},
                    {"repair", 0},
                    {"short_term_only", 0},
                    {"human_review", 1},
                    {"undecided", 0},
                }},
                {"accepted", json::array()},
                {"rejected", json::array()},
                {"repair_queue", json::array()},
                {"short_term_only_queue", json::array()},
                {"human_review_queue", json::array({
                    json{
                        {"slice_id", "query-context"},
                        {"decision", "require_human"},
                        {"reason", "clips-run-non-standard-exception"},
                        {"rule_id", "runtime.clips.non_std_exception"},
                        {"next_action", "collect_more_context"},
                        {"raw_fact", "non-standard exception"},
                    }
                })},
                {"undecided", json::array()},
            }},
            {"facts_asserted", 0},
            {"facts_after_run_count", 0},
            {"activations_before_run_count", 0},
            {"rules_fired", 0},
            {"admission_summary", {
                {"allow_count", 0},
                {"reject_count", 0},
                {"repair_count", 0},
                {"short_term_only_count", 0},
                {"require_human_count", 1},
                {"dominant_decision", "require_human"},
            }},
            {"manifest_snapshot", json::object()},
        };
    }
}

bool RagServerRuntime::inject_chat_context(json & body, const std::string & query, int top_k, int timeout_ms) const {
    if (!enabled()) {
        return false;
    }

    const json payload = build_chat_context_payload(body, query, top_k, timeout_ms);
    const std::string context = payload.value("context_block", "");
    if (context.empty()) {
        return false;
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        return false;
    }

    body["messages"].insert(body["messages"].begin(), json{
        {"role", "system"},
        {"content", context},
    });
    body["rag"] = true;
    body["rag_request_id"] = payload.value("request_id", "");
    body["rag_trace_id"] = payload.value("trace_id", "");
    body["rag_query_id"] = payload.value("query_id", "");
    body["approved_context"] = payload.value("approved_context", json::array());
    body["admission_summary"] = payload.value("admission_summary", json::object());
    return true;
}

json RagServerRuntime::build_status_payload() const {
    std::string last_error;
    std::string last_job_kind;
    bool last_reset_before_add = false;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        last_error = last_error_;
        last_job_kind = last_job_kind_;
        last_reset_before_add = last_reset_before_add_;
    }

    int chunk_count = 0;
    int vector_map_count = 0;
    int raw_slice_count = 0;
    if (rag_engine_) {
        chunk_count = (int) rag_engine_->get_chunk_count();
        vector_map_count = (int) rag_engine_->get_vector_map_count();
        raw_slice_count = (int) rag_engine_->get_raw_slice_count();
    }

    return json{
        {"enabled", enabled()},
        {"ready", ready()},
        {"status", job_active_.load() ? "indexing" : (pending_count_.load() > 0 ? "queued" : "idle")},
        {"pending", pending_count_.load()},
        {"active", job_active_.load()},
        {"jobs_completed", jobs_completed_.load()},
        {"docs_completed", docs_completed_.load()},
        {"metadata_completed", metadata_completed_.load()},
        {"chunk_count", chunk_count},
        {"vector_map_count", vector_map_count},
        {"raw_slice_count", raw_slice_count},
        {"last_job_kind", last_job_kind},
        {"last_reset_before_add", last_reset_before_add},
        {"last_error", last_error},
    };
}

std::string RagServerRuntime::extract_query_from_body(const json & body) {
    const std::string direct_query = body.value("query", "");
    if (!direct_query.empty()) {
        return direct_query;
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        return "";
    }

    for (auto it = body["messages"].rbegin(); it != body["messages"].rend(); ++it) {
        if (!it->is_object()) {
            continue;
        }
        if (it->value("role", "") != "user") {
            continue;
        }
        if (!it->contains("content")) {
            continue;
        }
        const std::string text = extract_text_from_message_content((*it)["content"]);
        if (!text.empty()) {
            return text;
        }
    }

    return "";
}
