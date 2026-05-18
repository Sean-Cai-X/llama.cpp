#include "server-goal.h"

namespace {

std::vector<std::string> read_string_list(
        const json & body,
        const char * key,
        const std::vector<std::string> & fallback) {
    if (!body.contains(key)) {
        return fallback;
    }

    const json & value = body.at(key);
    if (value.is_string()) {
        return {value.get<std::string>()};
    }
    if (!value.is_array()) {
        return fallback;
    }

    std::vector<std::string> result;
    for (const auto & item : value) {
        if (item.is_string()) {
            result.push_back(item.get<std::string>());
        }
    }
    return result.empty() ? fallback : result;
}

} // namespace

server_goal_envelope parse_server_goal_envelope(const json & body) {
    server_goal_envelope goal;

    const json & goal_body = body.contains("goal") && body.at("goal").is_object()
        ? body.at("goal")
        : body;

    goal.goal_id = goal_body.value("goal_id", "goal-" + gen_tool_call_id());
    goal.request_id = goal_body.value("request_id", "req-" + gen_tool_call_id());
    goal.trace_id = goal_body.value("trace_id", "trace-" + gen_tool_call_id());
    goal.goal_type = goal_body.value("goal_type", "");
    goal.root_path = goal_body.value("root_path", goal_body.value("path", ""));
    goal.status = goal_body.value("status", "RUNNING");
    goal.include_patterns = read_string_list(goal_body, "include", {"**/*.cpp", "**/*.h", "**/*.hpp", "**/*.c", "**/*.cc"});
    goal.exclude_patterns = read_string_list(goal_body, "exclude", {"**/build/**", "**/.git/**", "**/vendor/**"});
    goal.read_chunk_lines = std::max(1, goal_body.value("read_chunk_lines", 500));
    if (goal_body.contains("test_acceptance_snapshot") && goal_body.at("test_acceptance_snapshot").is_object()) {
        goal.test_acceptance_snapshot = goal_body.at("test_acceptance_snapshot");
    }

    return goal;
}

json to_json(const server_goal_envelope & goal) {
    return json{
        {"goal_id", goal.goal_id},
        {"request_id", goal.request_id},
        {"trace_id", goal.trace_id},
        {"goal_type", goal.goal_type},
        {"root_path", goal.root_path},
        {"status", goal.status},
        {"include", goal.include_patterns},
        {"exclude", goal.exclude_patterns},
        {"read_chunk_lines", goal.read_chunk_lines},
    };
}

json to_json(const server_goal_progress & progress) {
    return json{
        {"target_count", progress.target_count},
        {"completed_count", progress.completed_count},
        {"failed_count", progress.failed_count},
        {"pending_count", progress.pending_count},
        {"skipped_count", progress.skipped_count},
    };
}

json to_json(const server_goal_acceptance & acceptance) {
    return json{
        {"goal_id", acceptance.goal_id},
        {"acceptance_status", acceptance.acceptance_status},
        {"final_status", acceptance.final_status},
        {"failure_mode", acceptance.failure_mode},
        {"supervision", to_json(acceptance.supervision)},
        {"supervision_state", acceptance.supervision.supervision_status},
        {"execution_disposition", acceptance.supervision.execution_disposition},
        {"assistant_response_allowed", acceptance.supervision.response_allowed},
        {"alarm_code", acceptance.supervision.alarm_code},
        {"alarm_message", acceptance.supervision.alarm_message},
        {"progress", to_json(acceptance.progress)},
        {"next_actions", acceptance.next_actions},
        {"alarms", acceptance.alarms},
    };
}
