#include "server-lora-routes.h"

#include "server-task.h"

namespace server_lora_routes {

    std::unique_ptr<server_res_generator> handle_get_lora_adapters(
        const server_http_req& req,
        const response_factory& make_response) {
        auto res = make_response();

        auto& rd = res->rd;

        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);

            task.id = rd.get_new_id();

            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);

        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);

        res->ok(result->to_json());
        return res;
    }

    std::unique_ptr<server_res_generator> handle_post_lora_adapters(
        const server_http_req& req,
        const response_factory& make_response) {
        auto res = make_response();

        const json body = json::parse(req.body);

        if (!body.is_array()) {
            res->error(format_error_response(
                "Request body must be an array",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto& rd = res->rd;

        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);

            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);

            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);

        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);

        res->ok(result->to_json());
        return res;
    }

} // namespace server_lora_routes