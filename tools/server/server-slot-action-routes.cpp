#include "server-slot-action-routes.h"

#include "server-task.h"

#include "common.h"

#include <exception>
#include <string>

namespace {

    std::unique_ptr<server_res_generator> handle_slots_save_impl(
        std::unique_ptr<server_res_generator> res,
        const server_http_req& req,
        const common_params& params,
        int id_slot) {
        const json request_data = json::parse(req.body);

        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res->error(format_error_response(
                "Invalid filename",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string filepath = params.slot_save_path + filename;

        auto& rd = res->rd;

        {
            server_task task(SERVER_TASK_TYPE_SLOT_SAVE);

            task.id = rd.get_new_id();
            task.slot_action.id_slot = id_slot;
            task.slot_action.filename = filename;
            task.slot_action.filepath = filepath;

            rd.post_task(std::move(task));
        }

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

        res->ok(result->to_json());
        return res;
    }

    std::unique_ptr<server_res_generator> handle_slots_restore_impl(
        std::unique_ptr<server_res_generator> res,
        const server_http_req& req,
        const common_params& params,
        int id_slot) {
        const json request_data = json::parse(req.body);

        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res->error(format_error_response(
                "Invalid filename",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string filepath = params.slot_save_path + filename;

        auto& rd = res->rd;

        {
            server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);

            task.id = rd.get_new_id();
            task.slot_action.id_slot = id_slot;
            task.slot_action.filename = filename;
            task.slot_action.filepath = filepath;

            rd.post_task(std::move(task));
        }

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

        GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);

        res->ok(result->to_json());
        return res;
    }

    std::unique_ptr<server_res_generator> handle_slots_erase_impl(
        std::unique_ptr<server_res_generator> res,
        const server_http_req& req,
        int id_slot) {
        auto& rd = res->rd;

        {
            server_task task(SERVER_TASK_TYPE_SLOT_ERASE);

            task.id = rd.get_new_id();
            task.slot_action.id_slot = id_slot;

            rd.post_task(std::move(task));
        }

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

        GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);

        res->ok(result->to_json());
        return res;
    }

} // namespace

namespace server_slot_action_routes {

    std::unique_ptr<server_res_generator> handle_slots_action(
        const server_http_req& req,
        const common_params& params,
        const response_factory& make_response) {
        auto res = make_response();

        if (params.slot_save_path.empty()) {
            res->error(format_error_response(
                "This server does not support slots action. Start it with `--slot-save-path`",
                ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const std::string id_slot_str = req.get_param("id_slot");

        int id_slot = -1;
        try {
            id_slot = std::stoi(id_slot_str);
        }
        catch (const std::exception&) {
            res->error(format_error_response(
                "Invalid slot ID",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save_impl(
                std::move(res),
                req,
                params,
                id_slot);
        }

        if (action == "restore") {
            return handle_slots_restore_impl(
                std::move(res),
                req,
                params,
                id_slot);
        }

        if (action == "erase") {
            return handle_slots_erase_impl(
                std::move(res),
                req,
                id_slot);
        }

        res->error(format_error_response(
            "Invalid action",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

} // namespace server_slot_action_routes