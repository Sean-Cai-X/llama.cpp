#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-response-generator.h"

#include <functional>
#include <memory>
#include <string>

class RagServerRuntime;

namespace server_rag_routes {

using response_factory = std::function<std::unique_ptr<server_res_generator>()>;

bool maybe_inject_chat_context(
        const common_params & params,
        RagServerRuntime * rag_runtime,
        json & body,
        std::string * error);

std::unique_ptr<server_res_generator> handle_index(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_index_status(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_add(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_search(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_explain(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_chat_context(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_clips_meta(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_clips_manifest(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_clips_run(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_review_observe(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_storage_lookup(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

std::unique_ptr<server_res_generator> handle_storage_page(
        const server_http_req & req,
        const common_params & params,
        RagServerRuntime * rag_runtime,
        const response_factory & make_response);

} // namespace server_rag_routes
