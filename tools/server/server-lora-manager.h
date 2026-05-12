#pragma once

#include "server-common.h"
#include "server-slot.h"
#include "server-task.h"

#include "llama.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace server_lora_manager {

    struct slot_lora_apply_result {
        bool ok = true;
        std::string error_message;
        error_type error = ERROR_TYPE_INVALID_REQUEST;
    };

    std::vector<common_adapter_lora_info> construct_lora_list(
        const std::vector<common_adapter_lora_info>& base_loras,
        const std::map<int, float>& config);

    slot_lora_apply_result apply_task_lora_to_slot(
        server_slot& slot,
        const server_task& task,
        const std::vector<common_adapter_lora_info>& base_loras);

    std::unique_ptr<server_task_result_get_lora> build_get_lora_result(
        int id_task,
        const std::vector<common_adapter_lora_info>& loras,
        const llama_vocab* vocab);

    std::unique_ptr<server_task_result_apply_lora> build_apply_lora_result(
        int id_task,
        const std::map<int, float>& config,
        std::vector<common_adapter_lora_info>& base_loras);

} // namespace server_lora_manager