#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-response-generator.h"

#include "llama.h"
#include "mtmd.h"

#include <functional>
#include <memory>
#include <string>

namespace server_embedding_routes {

    using response_factory = std::function<std::unique_ptr<server_res_generator>()>;

    struct route_context {
        const common_params& params;

        uint32_t n_ubatch = 0;
        llama_model* model = nullptr;
        const llama_vocab* vocab = nullptr;
        mtmd_context* mctx = nullptr;

        std::string model_name;
        enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_NONE;
    };

    std::unique_ptr<server_res_generator> handle_embeddings(
        const server_http_req& req,
        const route_context& ctx,
        task_response_type res_type,
        const response_factory& make_response);

    std::unique_ptr<server_res_generator> handle_rerank(
        const server_http_req& req,
        const route_context& ctx,
        const response_factory& make_response);

} // namespace server_embedding_routes
