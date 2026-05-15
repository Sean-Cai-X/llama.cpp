#include "server-rag-routes.h"

#include "log.h"
#include "server-trace-registry.h"
#include "RAG/src/rag_integration_bridge.h"
#include "RAG/src/rag_server_runtime.h"

namespace {

error_type to_server_error_type(RagBridgeError error) {
    switch (error) {
    case RagBridgeError::invalid_request:
        return ERROR_TYPE_INVALID_REQUEST;
    case RagBridgeError::not_supported:
        return ERROR_TYPE_NOT_SUPPORTED;
    case RagBridgeError::internal_error:
        return ERROR_TYPE_SERVER;
    case RagBridgeError::none:
    default:
        return ERROR_TYPE_SERVER;
    }
}

std::unique_ptr<server_res_generator> finish_rag_response(
        std::unique_ptr<server_res_generator> res,
        const RagBridgeResult & result) {
    if (!result.ok) {
        res->error(format_error_response(
            result.message,
            to_server_error_type(result.error)));
        return res;
    }

    res->ok(result.payload);
    return res;
}

} // namespace

namespace server_rag_routes {

bool maybe_inject_chat_context(
        const common_params & params,
        RagServerRuntime * rag_runtime,
        json & body,
        std::string * error) {
    RagIntegrationBridge rag_bridge(params, rag_runtime);
    return rag_bridge.maybe_inject_chat_context(body, error);
}

std::unique_ptr<server_res_generator> handle_index(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_index_response(json::parse(req.body));

    if (result.ok) {
        server_trace_registry::record_stage(
            result.payload.value("trace_id", ""),
            "rag_index",
            result.payload);
    }

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_index_status(
        const server_http_req &,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result = rag_bridge.build_status_response();

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_add(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_add_response(json::parse(req.body));

    if (result.ok) {
        server_trace_registry::record_stage(
            result.payload.value("trace_id", ""),
            "rag_add",
            result.payload);
    }

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_search(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_search_response(json::parse(req.body));

    if (result.ok) {
        server_trace_registry::record_stage(
            result.payload.value("trace_id", ""),
            "rag_search",
            result.payload);
    }

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_explain(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_explain_response(json::parse(req.body));

    if (result.ok) {
        server_trace_registry::record_stage(
            result.payload.value("trace_id", ""),
            "rag_explain",
            result.payload);
    }

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_chat_context(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_chat_context_response(json::parse(req.body));

    if (result.ok) {
        server_trace_registry::record_stage(
            result.payload.value("trace_id", ""),
            "rag_chat_context",
            result.payload);
    }

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_clips_meta(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_clips_meta_response(json::parse(req.body));

    if (result.ok) {
        server_trace_registry::record_stage(
            result.payload.value("trace_id", ""),
            "rag_clips_meta",
            result.payload);
    }

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_clips_manifest(
        const server_http_req &,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);
    const RagBridgeResult result =
        rag_bridge.build_clips_manifest_response();

    return finish_rag_response(std::move(res), result);
}

std::unique_ptr<server_res_generator> handle_clips_run(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response) {
    auto res = make_response();

    RagIntegrationBridge rag_bridge(params, rag_runtime);

    LOG_INF("%s: handling /rag/clips/run request\n", __func__);

    const RagBridgeResult result =
        rag_bridge.build_clips_run_response(json::parse(req.body));

    LOG_INF(
        "%s: /rag/clips/run bridge result ok=%s\n",
        __func__,
        result.ok ? "true" : "false");

    if (!result.ok) {
        return finish_rag_response(std::move(res), result);
    }

    server_trace_registry::record_stage(
        result.payload.value("trace_id", ""),
        "rag_clips_run",
        result.payload);

    LOG_INF("%s: serializing /rag/clips/run response\n", __func__);

    res->ok(result.payload);

    LOG_INF("%s: /rag/clips/run response serialized\n", __func__);

    return res;
}

} // namespace server_rag_routes
