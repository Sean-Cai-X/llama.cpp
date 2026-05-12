#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-response-generator.h"

#include <functional>
#include <memory>

namespace server_slot_action_routes {

    using response_factory = std::function<std::unique_ptr<server_res_generator>()>;

    std::unique_ptr<server_res_generator> handle_slots_action(
        const server_http_req& req,
        const common_params& params,
        const response_factory& make_response);

} // namespace server_slot_action_routes