#pragma once

#include "server-goal.h"
#include "server-tool-envelope.h"

#include <string>

struct server_goal_route_decision {
    std::string decision_type;

    // Machine-readable route status.  In Phase 3 this is still produced by the
    // server-side CLIPS-compatible router stub; the shape is intentionally the
    // same shape expected from the real CLIPS bridge later.
    bool result_valid = true;
    bool allowed = true;
    bool verified = true;
    bool goal_complete = false;

    std::string reason;
    std::string failure_mode;
    std::string final_status = "IN_PROGRESS";

    // This is the response gate consumed by the server path.  It must remain
    // false while a goal is incomplete and still has safe next_actions.
    bool assistant_response_allowed = false;

    server_goal_progress progress;
    json next_actions = json::array();
    json alarms = json::array();
    json route_metadata = json::object();
};

class server_clips_goal_router {
public:
    server_goal_route_decision pre_guard(
        const server_goal_envelope & goal,
        const server_tool_call_envelope & action,
        bool permission_write) const;

    server_goal_route_decision post_guard(
        const server_goal_envelope & goal,
        const server_tool_call_envelope & action,
        const json & result) const;

    server_goal_route_decision acceptance_check(
        const server_goal_envelope & goal,
        const json & state_snapshot) const;
};

json to_json(const server_goal_route_decision & decision);