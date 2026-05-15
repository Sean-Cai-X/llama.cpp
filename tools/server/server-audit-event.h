#pragma once

#include "server-common.h"

#include <string>

struct server_audit_event {
    std::string trace_id;
    std::string stage;
    std::string goal_id;
    std::string tool_call_id;
    json payload = json::object();
};

json to_json(const server_audit_event & event);
