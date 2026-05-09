#pragma once

#include "common.h"
#include "rag_server_runtime.h"

#include <nlohmann/json.hpp>

#include <string>

using json = nlohmann::ordered_json;

enum class RagBridgeError {
    none,
    invalid_request,
    not_supported,
    internal_error,
};

struct RagBridgeResult {
    bool ok = false;
    RagBridgeError error = RagBridgeError::none;
    std::string message;
    json payload = json::object();
};

class RagIntegrationBridge {
public:
    RagIntegrationBridge(const common_params & params, RagServerRuntime * runtime);

    RagBridgeResult build_status_response() const;
    RagBridgeResult build_index_response(const json & body) const;
    RagBridgeResult build_add_response(const json & body) const;
    RagBridgeResult build_search_response(const json & body) const;
    RagBridgeResult build_explain_response(const json & body) const;
    RagBridgeResult build_chat_context_response(const json & body) const;
    RagBridgeResult build_clips_meta_response(const json & body) const;
    RagBridgeResult build_clips_manifest_response() const;
    RagBridgeResult build_clips_run_response(const json & body) const;

    bool maybe_inject_chat_context(json & body, std::string * error_message = nullptr) const;

private:
    RagBridgeResult runtime_unavailable() const;
    int resolve_top_k(const json & body) const;

    const common_params & params_;
    RagServerRuntime * runtime_ = nullptr;
};
