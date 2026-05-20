#pragma once

#include "server-clips-goal-router.h"
#include "server-goal.h"

#include <cstddef>
#include <functional>
#include <string>

class server_goal_execution_controller {
public:
    struct loop_policy {
        int max_steps = 50;
        int max_same_action_repeat = 2;
        int max_read_files = 500;
        int max_read_bytes = 104857600;
        int max_pending_actions_per_round = 20;
        int max_runtime_ms = 300000;
    };

    using invoke_tool_fn = std::function<json(const std::string &, const json &)>;
    using is_write_tool_fn = std::function<bool(const std::string &)>;

    server_goal_execution_controller(
        invoke_tool_fn invoke_tool,
        is_write_tool_fn is_write_tool);

    json execute_goal_until_closed(
        const server_goal_envelope & goal,
        const loop_policy & policy = {}) const;

private:
    invoke_tool_fn invoke_tool_;
    is_write_tool_fn is_write_tool_;
};