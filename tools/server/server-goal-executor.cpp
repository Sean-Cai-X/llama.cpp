#include "server-goal-executor.h"

#include "server-audit-event.h"
#include "server-supervision.h"
#include "server-tool-envelope.h"
#include "server-trace-registry.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace {

struct file_read_state {
    std::string path;
    int line_count = -1;
    int next_start_line = 1;
    int last_end_line = 0;
    bool completed = false;
    bool failed = false;
    bool skipped = false;
    std::vector<std::pair<int, int>> ranges_completed;
};

struct goal_runtime_state {
    bool target_files_discovered = false;
    bool file_discovery_failed = false;
    std::map<std::string, file_read_state> files;
    std::vector<std::string> ordered_paths;
    json alarms = json::array();
    std::string failure_mode;
};

json make_next_action(
        const std::string & tool_name,
        const json & params,
        const std::string & safety_class) {
    return json{
        {"action_id", tool_name},
        {"tool_name", tool_name},
        {"params", params},
        {"safety_class", safety_class},
    };
}

std::string compute_progress_signature(const goal_runtime_state & state) {
    std::ostringstream oss;
    oss << state.target_files_discovered << '|'
        << state.file_discovery_failed << '|'
        << state.files.size() << '|';
    for (const auto & path : state.ordered_paths) {
        const auto it = state.files.find(path);
        if (it == state.files.end()) {
            continue;
        }
        const auto & file = it->second;
        oss << path << ':' << file.completed << ':' << file.failed << ':'
            << file.skipped << ':' << file.next_start_line << ':'
            << file.ranges_completed.size() << '|';
    }
    return oss.str();
}

json build_state_snapshot(
        const server_goal_envelope & goal,
        goal_runtime_state & state) {
    int completed_count = 0;
    int failed_count = 0;
    int skipped_count = 0;
    int pending_count = 0;
    json pending_actions = json::array();

    if (!state.target_files_discovered && !state.file_discovery_failed) {
        pending_actions.push_back(make_next_action("file_glob_search", json{
            {"path", goal.root_path},
            {"include_patterns", goal.include_patterns},
            {"exclude_patterns", goal.exclude_patterns},
            {"max_results", 5000},
        }, "READ_ONLY"));
        pending_count = 1;
    } else {
        for (const auto & path : state.ordered_paths) {
            const auto & file = state.files.at(path);
            if (file.completed) {
                completed_count++;
                continue;
            }
            if (file.failed) {
                failed_count++;
                continue;
            }
            if (file.skipped) {
                skipped_count++;
                continue;
            }

            if (file.line_count < 0) {
                pending_actions.push_back(make_next_action("file_info", json{
                    {"path", path},
                }, "READ_ONLY"));
                pending_count++;
                continue;
            }

            const int end_line = std::min(file.line_count, file.next_start_line + goal.read_chunk_lines - 1);
            pending_actions.push_back(make_next_action("read_file", json{
                {"path", path},
                {"start_line", file.next_start_line},
                {"end_line", end_line},
                {"append_loc", false},
            }, "READ_ONLY"));
            pending_count++;
        }
    }

    if (state.target_files_discovered && state.files.empty() && state.failure_mode.empty()) {
        state.failure_mode = "NO_TARGET_FILES_DISCOVERED";
        state.alarms.push_back(build_alarm_event(
            "ERROR",
            "NO_TARGET_FILES_DISCOVERED",
            "goal discovery completed but no target files matched the include/exclude filters"));
    }

    const bool goal_complete = state.target_files_discovered &&
        !state.file_discovery_failed &&
        !state.files.empty() &&
        pending_count == 0 &&
        failed_count == 0 &&
        skipped_count == 0;
    const bool goal_closed = state.target_files_discovered &&
        !state.file_discovery_failed &&
        !state.files.empty() &&
        pending_count == 0;

    server_goal_progress progress;
    progress.target_count = static_cast<int>(state.files.size());
    progress.completed_count = completed_count;
    progress.failed_count = failed_count;
    progress.skipped_count = skipped_count;
    progress.pending_count = pending_count;

    return json{
        {"goal_complete", goal_complete},
        {"target_count", progress.target_count},
        {"completed_count", progress.completed_count},
        {"failed_count", progress.failed_count},
        {"skipped_count", progress.skipped_count},
        {"pending_count", progress.pending_count},
        {"next_actions", pending_actions},
        {"alarms", state.alarms},
        {"failure_mode", state.failure_mode},
        {"goal_closed", goal_closed},
        {"final_status_hint", goal_complete ? "COMPLETE" : (goal_closed ? "PARTIAL_COMPLETE_WITH_SKIPS" : "IN_PROGRESS")},
    };
}

std::vector<std::string> read_string_array(const json & value) {
    std::vector<std::string> result;
    if (!value.is_array()) {
        return result;
    }
    for (const auto & item : value) {
        if (item.is_string()) {
            result.push_back(item.get<std::string>());
        }
    }
    return result;
}

void apply_tool_result(
        goal_runtime_state & state,
        const server_tool_call_envelope & action,
        const json & result) {
    if (action.tool_name == "file_glob_search") {
        if (result.contains("matches") && result.at("matches").is_array()) {
            state.target_files_discovered = true;
            for (const auto & item : result.at("matches")) {
                if (!item.is_string()) {
                    continue;
                }
                const std::string path = item.get<std::string>();
                if (state.files.count(path) > 0) {
                    continue;
                }
                state.files.emplace(path, file_read_state{
                    path,
                    -1,
                    1,
                    0,
                    false,
                    false,
                    false,
                    {}
                });
                state.ordered_paths.push_back(path);
            }
        } else {
            state.file_discovery_failed = true;
            state.failure_mode = "FILE_DISCOVERY_FAILED";
        }
        return;
    }

    const std::string path = action.params.value("path", "");
    if (path.empty()) {
        return;
    }

    auto it = state.files.find(path);
    if (it == state.files.end()) {
        return;
    }

    file_read_state & file = it->second;
    if (action.tool_name == "file_info") {
        if (result.contains("line_count")) {
            file.line_count = result.value("line_count", -1);
            if (file.line_count == 0) {
                file.completed = true;
            }
        } else {
            file.failed = true;
        }
        return;
    }

    if (action.tool_name == "read_file") {
        const int start_line = action.params.value("start_line", 1);
        const int end_line = action.params.value("end_line", start_line);
        file.ranges_completed.push_back({start_line, end_line});
        file.last_end_line = std::max(file.last_end_line, end_line);
        file.next_start_line = end_line + 1;

        const bool reached_eof = result.value("reached_eof", false);
        const int line_count = result.value("line_count", file.line_count);
        if (line_count > 0) {
            file.line_count = line_count;
        }

        if (reached_eof || (file.line_count > 0 && file.next_start_line > file.line_count)) {
            file.completed = true;
        }
    }
}

server_tool_call_envelope make_action_envelope(
        const server_goal_envelope & goal,
        const json & action) {
    server_tool_call_envelope envelope;
    envelope.tool_call_id = gen_tool_call_id();
    envelope.goal_id = goal.goal_id;
    envelope.request_id = goal.request_id;
    envelope.trace_id = goal.trace_id;
    envelope.tool_name = action.value("tool_name", "");
    envelope.params = action.value("params", json::object());
    envelope.safety_class = action.value("safety_class", "READ_ONLY");
    return envelope;
}

bool normalize_next_action_json(
        json & action,
        std::string & failure_mode,
        json & alarm) {
    if (!action.is_object()) {
        failure_mode = "NEXT_ACTION_DESCRIPTOR_INVALID";
        alarm = build_alarm_event("ERROR", failure_mode, "next_action entry must be an object");
        return false;
    }

    if (action.value("tool_name", "").empty()) {
        failure_mode = "NEXT_ACTION_TOOL_NAME_MISSING";
        alarm = build_alarm_event("ERROR", failure_mode, "next_action is missing tool_name");
        return false;
    }

    if (action.value("safety_class", "").empty()) {
        failure_mode = "NEXT_ACTION_SAFETY_CLASS_MISSING";
        alarm = build_alarm_event("ERROR", failure_mode, "next_action is missing safety_class");
        return false;
    }

    if (action.contains("params") && action.at("params").is_object()) {
        return true;
    }

    const std::string next_call_json = action.value("next_call_json", "");
    if (next_call_json.empty()) {
        failure_mode = "NEXT_CALL_JSON_MISSING";
        alarm = build_alarm_event("ERROR", failure_mode, "next_action is missing both params and next_call_json");
        return false;
    }

    try {
        json parsed = json::parse(next_call_json);
        if (!parsed.is_object()) {
            failure_mode = "NEXT_CALL_JSON_INVALID";
            alarm = build_alarm_event("ERROR", failure_mode, "next_call_json must decode to a JSON object");
            return false;
        }
        action["params"] = parsed;
        return true;
    } catch (const std::exception &) {
        failure_mode = "NEXT_CALL_JSON_INVALID";
        alarm = build_alarm_event("ERROR", failure_mode, "next_call_json is not valid JSON");
        return false;
    }
}

json build_goal_result(
        const server_goal_envelope & goal,
        const server_goal_acceptance & acceptance) {
    return json{
        {"record_model", "goal_execution_result_v1"},
        {"status", acceptance.final_status},
        {"supervision", to_json(acceptance.supervision)},
        {"goal", to_json(goal)},
        {"acceptance", to_json(acceptance)},
    };
}

void finalize_acceptance(
        const server_goal_envelope & goal,
        server_goal_acceptance & acceptance) {
    if (acceptance.final_status == "COMPLETE") {
        acceptance.supervision.supervision_status = "COMPLETE";
        acceptance.supervision.execution_disposition = "FINALIZE_ANSWER";
        acceptance.supervision.response_allowed = true;
    } else if (acceptance.final_status == "PARTIAL_COMPLETE_WITH_SKIPS") {
        acceptance.supervision.supervision_status = "PARTIAL_COMPLETE_WITH_SKIPS";
        acceptance.supervision.execution_disposition = "FINALIZE_ANSWER";
        acceptance.supervision.response_allowed = true;
    } else if (acceptance.final_status == "IN_PROGRESS") {
        acceptance.supervision.supervision_status = "IN_PROGRESS";
        acceptance.supervision.execution_disposition = "CONTINUE_EXECUTION";
        acceptance.supervision.response_allowed = false;
    } else if (acceptance.final_status == "NEEDS_HUMAN_APPROVAL") {
        acceptance.supervision.supervision_status = "NEEDS_HUMAN_APPROVAL";
        acceptance.supervision.execution_disposition = "RETURN_FAILURE_REPORT";
        acceptance.supervision.response_allowed = true;
    } else if (acceptance.final_status == "BLOCKED_BY_POLICY") {
        acceptance.supervision.supervision_status = "BLOCKED_BY_POLICY";
        acceptance.supervision.execution_disposition = "RETURN_FAILURE_REPORT";
        acceptance.supervision.response_allowed = true;
    } else if (acceptance.final_status == "DEADLOCK_DETECTED") {
        acceptance.supervision.supervision_status = "DEADLOCK_DETECTED";
        acceptance.supervision.execution_disposition = "RETURN_FAILURE_REPORT";
        acceptance.supervision.response_allowed = true;
    } else if (acceptance.final_status == "TIMEOUT") {
        acceptance.supervision.supervision_status = "TIMEOUT";
        acceptance.supervision.execution_disposition = "RETURN_FAILURE_REPORT";
        acceptance.supervision.response_allowed = true;
    } else {
        acceptance.supervision.supervision_status = acceptance.final_status.empty() ? "FAILED" : acceptance.final_status;
        acceptance.supervision.execution_disposition = "RETURN_FAILURE_REPORT";
        acceptance.supervision.response_allowed = true;
    }

    acceptance.supervision.failure_mode = acceptance.failure_mode;
    for (const auto & alarm : acceptance.alarms) {
        if (!alarm.is_object()) {
            continue;
        }
        acceptance.supervision.alarm_code = alarm.value("alarm_code", alarm.value("reason", ""));
        acceptance.supervision.alarm_message = alarm.value("alarm_message", "");
        if (!acceptance.supervision.alarm_code.empty() || !acceptance.supervision.alarm_message.empty()) {
            break;
        }
    }

    server_trace_registry::record_stage(goal.trace_id, "goal_final", json{
        {"goal_id", goal.goal_id},
        {"goal_type", goal.goal_type},
        {"request_id", goal.request_id},
        {"trace_id", goal.trace_id},
        {"acceptance", to_json(acceptance)},
    });
}

} // namespace

server_goal_execution_controller::server_goal_execution_controller(
        invoke_tool_fn invoke_tool,
        is_write_tool_fn is_write_tool)
    : invoke_tool_(std::move(invoke_tool)),
      is_write_tool_(std::move(is_write_tool)) {
}

json server_goal_execution_controller::execute_goal_until_closed(
        const server_goal_envelope & goal,
        const loop_policy & policy) const {
    if (goal.goal_type != "READ_ALL_CODE_FILES") {
        server_goal_acceptance acceptance;
        acceptance.goal_id = goal.goal_id;
        acceptance.acceptance_status = "NOT_COMPLETE";
        acceptance.final_status = "BLOCKED_BY_POLICY";
        acceptance.failure_mode = "UNSUPPORTED_GOAL_TYPE";
        acceptance.alarms.push_back(build_alarm_event(
            "ERROR",
            "UNSUPPORTED_GOAL_TYPE",
            "only READ_ALL_CODE_FILES is supported by the current goal executor phase"));
        finalize_acceptance(goal, acceptance);
        return build_goal_result(goal, acceptance);
    }

    server_clips_goal_router router;
    goal_runtime_state state;

    std::string last_progress_signature;
    int same_progress_rounds = 0;
    std::map<std::string, int> action_repeat_counts;
    json test_acceptance_snapshot = goal.test_acceptance_snapshot;

    server_trace_registry::record_stage(goal.trace_id, "goal_start", to_json(goal));

    for (int step = 0; step < policy.max_steps; ++step) {
        const json state_snapshot = test_acceptance_snapshot.is_object()
            ? test_acceptance_snapshot
            : build_state_snapshot(goal, state);
        test_acceptance_snapshot = nullptr;
        const server_goal_route_decision acceptance_route = router.acceptance_check(goal, state_snapshot);
        server_trace_registry::append_event(goal.trace_id, "acceptance_check", json{
            {"goal_id", goal.goal_id},
            {"step", step},
            {"decision", to_json(acceptance_route)},
        });

        if (acceptance_route.goal_complete) {
            server_goal_acceptance acceptance;
            acceptance.goal_id = goal.goal_id;
            acceptance.acceptance_status = "COMPLETE";
            acceptance.final_status = "COMPLETE";
            acceptance.progress = acceptance_route.progress;
            acceptance.alarms = acceptance_route.alarms;
            finalize_acceptance(goal, acceptance);
            return build_goal_result(goal, acceptance);
        }

        if (!acceptance_route.failure_mode.empty() && acceptance_route.next_actions.empty()) {
            server_goal_acceptance acceptance;
            acceptance.goal_id = goal.goal_id;
            acceptance.acceptance_status = acceptance_route.failure_mode == "PARTIAL_COMPLETE_WITH_SKIPS"
                ? "PARTIAL"
                : "NOT_COMPLETE";
            acceptance.final_status = acceptance_route.failure_mode == "PARTIAL_COMPLETE_WITH_SKIPS"
                ? "PARTIAL_COMPLETE_WITH_SKIPS"
                : "FAILED";
            acceptance.failure_mode = acceptance_route.failure_mode == "PARTIAL_COMPLETE_WITH_SKIPS"
                ? ""
                : acceptance_route.failure_mode;
            acceptance.progress = acceptance_route.progress;
            acceptance.alarms = acceptance_route.alarms;
            finalize_acceptance(goal, acceptance);
            return build_goal_result(goal, acceptance);
        }

        if (acceptance_route.next_actions.empty()) {
            server_goal_acceptance acceptance;
            acceptance.goal_id = goal.goal_id;
            acceptance.acceptance_status = "NOT_COMPLETE";
            acceptance.final_status = "FAILED";
            acceptance.failure_mode = "NO_NEXT_ACTION_FOR_INCOMPLETE_GOAL";
            acceptance.progress = acceptance_route.progress;
            acceptance.alarms = acceptance_route.alarms;
            finalize_acceptance(goal, acceptance);
            return build_goal_result(goal, acceptance);
        }

        const int action_limit = std::min<int>(
            static_cast<int>(acceptance_route.next_actions.size()),
            policy.max_pending_actions_per_round);

        for (int i = 0; i < action_limit; ++i) {
            json action_json = acceptance_route.next_actions.at(i);
            std::string action_shape_failure_mode;
            json action_shape_alarm;
            if (!normalize_next_action_json(action_json, action_shape_failure_mode, action_shape_alarm)) {
                server_goal_acceptance acceptance;
                acceptance.goal_id = goal.goal_id;
                acceptance.acceptance_status = "NOT_COMPLETE";
                acceptance.final_status = "FAILED";
                acceptance.failure_mode = action_shape_failure_mode;
                acceptance.progress = acceptance_route.progress;
                acceptance.alarms.push_back(action_shape_alarm);
                finalize_acceptance(goal, acceptance);
                return build_goal_result(goal, acceptance);
            }

            server_tool_call_envelope action = make_action_envelope(goal, action_json);
            const bool permission_write = is_write_tool_(action.tool_name);

            const std::string action_signature = action.tool_name + "|" + action.params.dump();
            action_repeat_counts[action_signature]++;
            if (action_repeat_counts[action_signature] > policy.max_same_action_repeat + 1) {
                server_goal_acceptance acceptance;
                acceptance.goal_id = goal.goal_id;
                acceptance.acceptance_status = "NOT_COMPLETE";
                acceptance.final_status = "DEADLOCK_DETECTED";
                acceptance.failure_mode = "DUPLICATE_ACTION_LOOP";
                acceptance.progress = acceptance_route.progress;
                acceptance.alarms.push_back(build_alarm_event(
                    "ERROR",
                    "DUPLICATE_ACTION_LOOP",
                    "same action repeated too many times"));
                finalize_acceptance(goal, acceptance);
                return build_goal_result(goal, acceptance);
            }

            const server_goal_route_decision pre = router.pre_guard(goal, action, permission_write);
            server_trace_registry::append_event(goal.trace_id, "clips_pre_guard", json{
                {"goal_id", goal.goal_id},
                {"tool_call", to_json(action)},
                {"decision", to_json(pre)},
            });
            if (!pre.allowed) {
                server_goal_acceptance acceptance;
                acceptance.goal_id = goal.goal_id;
                acceptance.acceptance_status = "NOT_COMPLETE";
                acceptance.final_status = "NEEDS_HUMAN_APPROVAL";
                acceptance.failure_mode = pre.failure_mode;
                acceptance.progress = acceptance_route.progress;
                acceptance.alarms = pre.alarms;
                finalize_acceptance(goal, acceptance);
                return build_goal_result(goal, acceptance);
            }

            json result = invoke_tool_(action.tool_name, action.params);
            action.delivery_status = result.contains("error") ? "FAILED" : "SUCCESS";
            action.result_hash = build_server_result_hash(result);
            server_trace_registry::append_event(goal.trace_id, "mcp_tool_result", json{
                {"goal_id", goal.goal_id},
                {"tool_call", to_json(action)},
                {"result", result},
            });

            const server_goal_route_decision post = router.post_guard(goal, action, result);
            server_trace_registry::append_event(goal.trace_id, "clips_post_guard", json{
                {"goal_id", goal.goal_id},
                {"tool_call", to_json(action)},
                {"decision", to_json(post)},
            });
            if (!post.verified) {
                server_goal_acceptance acceptance;
                acceptance.goal_id = goal.goal_id;
                acceptance.acceptance_status = "NOT_COMPLETE";
                acceptance.final_status = "FAILED";
                acceptance.failure_mode = post.failure_mode;
                acceptance.progress = acceptance_route.progress;
                acceptance.alarms = post.alarms;
                finalize_acceptance(goal, acceptance);
                return build_goal_result(goal, acceptance);
            }

            apply_tool_result(state, action, result);
        }

        if (static_cast<int>(state.files.size()) > policy.max_read_files) {
            server_goal_acceptance acceptance;
            acceptance.goal_id = goal.goal_id;
            acceptance.acceptance_status = "NOT_COMPLETE";
            acceptance.final_status = "FAILED";
            acceptance.failure_mode = "EXPANDING_WORKSET_LIMIT_EXCEEDED";
            acceptance.alarms.push_back(build_alarm_event(
                "ERROR",
                "EXPANDING_WORKSET_LIMIT_EXCEEDED",
                "too many files discovered for current goal policy"));
            acceptance.progress = router.acceptance_check(goal, build_state_snapshot(goal, state)).progress;
            finalize_acceptance(goal, acceptance);
            return build_goal_result(goal, acceptance);
        }

        const std::string progress_signature = compute_progress_signature(state);
        if (progress_signature == last_progress_signature) {
            same_progress_rounds++;
        } else {
            same_progress_rounds = 0;
        }
        if (same_progress_rounds >= 2) {
            server_goal_acceptance acceptance;
            acceptance.goal_id = goal.goal_id;
            acceptance.acceptance_status = "NOT_COMPLETE";
            acceptance.final_status = "DEADLOCK_DETECTED";
            acceptance.failure_mode = "NO_PROGRESS_DETECTED";
            acceptance.alarms.push_back(build_alarm_event(
                "ERROR",
                "NO_PROGRESS_DETECTED",
                "no progress detected across repeated acceptance rounds"));
            acceptance.progress = router.acceptance_check(goal, build_state_snapshot(goal, state)).progress;
            finalize_acceptance(goal, acceptance);
            return build_goal_result(goal, acceptance);
        }
        last_progress_signature = progress_signature;
    }

    server_goal_acceptance acceptance;
    acceptance.goal_id = goal.goal_id;
    acceptance.acceptance_status = "NOT_COMPLETE";
    acceptance.final_status = "TIMEOUT";
    acceptance.failure_mode = "MAX_TRACEBACK_STEPS_EXCEEDED";
    acceptance.alarms.push_back(build_alarm_event(
        "ERROR",
        "MAX_TRACEBACK_STEPS_EXCEEDED",
        "goal executor reached max_steps"));
    acceptance.progress = router.acceptance_check(goal, build_state_snapshot(goal, state)).progress;
    finalize_acceptance(goal, acceptance);
    return build_goal_result(goal, acceptance);
}
