#pragma once

#include "server-common.h"

#include <string>

struct server_tool_call_envelope {
    std::string tool_call_id;
    std::string goal_id;
    std::string request_id;
    std::string trace_id;
    std::string tool_name;
    json params = json::object();
    std::string safety_class = "READ_ONLY";
    std::string delivery_status = "PENDING";
    std::string result_hash;
};

std::string build_server_result_hash(const json & result);
json to_json(const server_tool_call_envelope & envelope);
