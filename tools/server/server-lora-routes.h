#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-response-generator.h"

#include <functional>
#include <memory>

namespace server_lora_routes {

    using response_factory = std::function<std::unique_ptr<server_res_generator>()>;

    std::unique_ptr<server_res_generator> handle_get_lora_adapters(
        const server_http_req& req,
        const response_factory& make_response);

    std::unique_ptr<server_res_generator> handle_post_lora_adapters(
        const server_http_req& req,
        const response_factory& make_response);

} // namespace server_lora_routes