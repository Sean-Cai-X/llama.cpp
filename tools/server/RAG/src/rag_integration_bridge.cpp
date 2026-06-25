#include "rag_integration_bridge.h"

#include <atomic>
#include <chrono>
#include <vector>

namespace {

std::atomic<uint64_t> g_rag_bridge_trace_counter {0};

std::string get_trace_string(const json & body, const char * key) {
    if (!body.contains(key) || !body.at(key).is_string()) {
        return "";
    }
    return body.at(key).get<std::string>();
}

std::string generate_bridge_trace_token(const char * prefix) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
    const uint64_t seq = ++g_rag_bridge_trace_counter;
    return std::string(prefix) + "-" + std::to_string(now_ms) + "-" + std::to_string(seq);
}

json build_bridge_request_context(const json & body, const char * route, bool rag_enabled, bool clips_enabled) {
    std::string request_id = get_trace_string(body, "request_id");
    if (request_id.empty()) {
        request_id = get_trace_string(body, "codex_request_id");
    }
    if (request_id.empty()) {
        request_id = generate_bridge_trace_token("REQ");
    }

    std::string trace_id = get_trace_string(body, "trace_id");
    if (trace_id.empty()) {
        trace_id = request_id + ".trace";
    }

    return json{
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"route", route},
        {"source_request_id", get_trace_string(body, "codex_request_id").empty() ? request_id : get_trace_string(body, "codex_request_id")},
        {"rag_enabled", rag_enabled},
        {"clips_enabled", clips_enabled},
    };
}

} // namespace

RagIntegrationBridge::RagIntegrationBridge(const common_params & params, RagServerRuntime * runtime)
    : params_(params), runtime_(runtime) {
}

RagBridgeResult RagIntegrationBridge::runtime_unavailable() const {
    RagBridgeResult result;
    result.ok = false;
    result.error = RagBridgeError::not_supported;
    result.message = "RAG is not enabled";
    return result;
}

int RagIntegrationBridge::resolve_top_k(const json & body) const {
    return body.value("top_k", params_.rag_top_k);
}

RagBridgeResult RagIntegrationBridge::build_status_response() const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_status_payload();
    return result;
}

RagBridgeResult RagIntegrationBridge::build_index_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string repo_path = body.value("repo_path", body.value("path", ""));
    if (repo_path.empty()) {
        return {false, RagBridgeError::invalid_request, "\"repo_path\" must be provided", json::object()};
    }

    std::string error_message;
    const bool reset_before_add = body.value("reset", false);
    if (!runtime_->enqueue_repo_index(repo_path, reset_before_add, &error_message)) {
        return {false, RagBridgeError::internal_error, error_message.empty() ? "Failed to enqueue RAG index job" : error_message, json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    const json request_context = build_bridge_request_context(body, "/rag/index", true, params_.rag_clips_enable);
    result.payload = json{
        {"request_id", request_context.value("request_id", "")},
        {"trace_id", request_context.value("trace_id", "")},
        {"status", "queued"},
        {"kind", "index"},
        {"repo_path", repo_path},
        {"pending", runtime_->build_status_payload().value("pending", 0)},
        {"request_context", request_context},
    };
    return result;
}

RagBridgeResult RagIntegrationBridge::build_add_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const bool reset_before_add = body.value("reset", false);
    std::vector<std::string> docs = body.value("documents", std::vector<std::string> {});
    if (docs.empty() && body.contains("content") && body["content"].is_string()) {
        docs.push_back(body["content"].get<std::string>());
    }
    if (docs.empty()) {
        return {false, RagBridgeError::invalid_request, "\"documents\" or \"content\" must be provided", json::object()};
    }

    std::vector<std::string> metadata = body.value("metadata", std::vector<std::string> {});
    std::string error_message;
    if (!runtime_->enqueue_documents(docs, metadata, reset_before_add, &error_message)) {
        return {false, RagBridgeError::internal_error, error_message.empty() ? "Failed to enqueue RAG add job" : error_message, json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    const json request_context = build_bridge_request_context(body, "/rag/add", true, params_.rag_clips_enable);
    result.payload = json{
        {"request_id", request_context.value("request_id", "")},
        {"trace_id", request_context.value("trace_id", "")},
        {"status", "queued"},
        {"kind", "documents"},
        {"documents", (int) docs.size()},
        {"pending", runtime_->build_status_payload().value("pending", 0)},
        {"request_context", request_context},
    };
    return result;
}

RagBridgeResult RagIntegrationBridge::build_search_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string query = body.value("query", "");
    if (query.empty()) {
        return {false, RagBridgeError::invalid_request, "\"query\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_search_payload(body, query, resolve_top_k(body), params_.rag_search_timeout_ms);
    return result;
}

RagBridgeResult RagIntegrationBridge::build_explain_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string query = body.value("query", "");
    if (query.empty()) {
        return {false, RagBridgeError::invalid_request, "\"query\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_explain_payload(body, query, resolve_top_k(body), params_.rag_search_timeout_ms);
    return result;
}

RagBridgeResult RagIntegrationBridge::build_chat_context_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string query = body.value("query", RagServerRuntime::extract_query_from_body(body));
    if (query.empty()) {
        return {false, RagBridgeError::invalid_request, "\"query\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_chat_context_payload(body, query, resolve_top_k(body), params_.rag_search_timeout_ms);
    return result;
}

RagBridgeResult RagIntegrationBridge::build_clips_meta_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string query = body.value("query", RagServerRuntime::extract_query_from_body(body));
    if (query.empty()) {
        return {false, RagBridgeError::invalid_request, "\"query\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_clips_meta_payload(body, query, resolve_top_k(body), params_.rag_search_timeout_ms);
    return result;
}

RagBridgeResult RagIntegrationBridge::build_clips_manifest_response() const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_clips_manifest_payload();
    return result;
}

RagBridgeResult RagIntegrationBridge::build_clips_run_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string query = body.value("query", RagServerRuntime::extract_query_from_body(body));
    const std::string baseline_id = body.value("baseline_id", body.value("baseline", ""));
    if (query.empty() && baseline_id.empty()) {
        return {false, RagBridgeError::invalid_request, "\"query\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_clips_run_payload(
        body,
        query.empty() ? ("baseline:" + baseline_id) : query,
        resolve_top_k(body),
        params_.rag_search_timeout_ms);
    return result;
}

RagBridgeResult RagIntegrationBridge::build_review_observation_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_review_observation_payload(body);
    if (!result.payload.value("ok", false)) {
        result.ok = false;
        result.error = RagBridgeError::invalid_request;
        result.message = result.payload.value("message", "invalid review observation request");
        result.payload = json::object();
    }
    return result;
}

RagBridgeResult RagIntegrationBridge::build_storage_lookup_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string kind = body.value("kind", "");
    if (kind.empty()) {
        return {false, RagBridgeError::invalid_request, "\"kind\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_storage_lookup_payload(body);
    return result;
}

RagBridgeResult RagIntegrationBridge::build_storage_page_response(const json & body) const {
    if (!runtime_ || !runtime_->enabled()) {
        return runtime_unavailable();
    }

    const std::string kind = body.value("kind", "");
    if (kind.empty()) {
        return {false, RagBridgeError::invalid_request, "\"kind\" must be provided", json::object()};
    }

    RagBridgeResult result;
    result.ok = true;
    result.payload = runtime_->build_storage_page_payload(body);
    return result;
}

bool RagIntegrationBridge::maybe_inject_chat_context(json & body, std::string * error_message) const {
    if (!runtime_ || !runtime_->enabled()) {
        if (error_message) {
            *error_message = "RAG is not enabled";
        }
        return false;
    }

    const bool use_rag = body.value("rag", params_.rag_auto_search);
    if (!use_rag) {
        return false;
    }

    const std::string query = RagServerRuntime::extract_query_from_body(body);
    if (query.empty()) {
        if (error_message) {
            *error_message = "RAG query could not be extracted";
        }
        return false;
    }

    return runtime_->inject_chat_context(body, query, params_.rag_top_k, params_.rag_search_timeout_ms);
}
