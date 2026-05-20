#include "server-goal-executor.h"
#include "server-audit-event.h"
#include "server-supervision.h"
#include "server-tool-envelope.h"
#include "server-trace-registry.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct file_read_state {
    std::string path;
    int line_count = -1;
    int next_start_line = 1;
    int last_end_line = 0;
    int current_chunk_lines = 0;
    bool completed = false;
    bool failed = false;
    bool skipped = false;
    bool metadata_loaded = false;
    size_t size_bytes = 0;
    size_t delivered_bytes = 0;
    std::string content_hash;
    std::string terminal_reason;
    std::vector<std::pair<int, int>> ranges_completed;
};

struct goal_runtime_state {
    bool target_files_discovered = false;
    bool file_discovery_failed = false;
    std::map<std::string, file_read_state> files;
    std::vector<std::string> ordered_paths;
    json alarms = json::array();
    std::string failure_mode;
    size_t total_target_bytes = 0;
    size_t total_delivered_bytes = 0;
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

int json_int_value(const json & value, const char * key, int fallback) {
    if (!value.is_object() || !value.contains(key)) {
        return fallback;
    }
    const json & item = value.at(key);
    if (item.is_number_integer() || item.is_number_unsigned()) {
        return item.get<int>();
    }
    return fallback;
}

size_t json_size_value(const json & value, const char * key, size_t fallback) {
    if (!value.is_object() || !value.contains(key)) {
        return fallback;
    }
    const json & item = value.at(key);
    if (item.is_number_unsigned()) {
        return item.get<size_t>();
    }
    if (item.is_number_integer()) {
        const int v = item.get<int>();
        return v < 0 ? fallback : static_cast<size_t>(v);
    }
    return fallback;
}

std::string json_error_message(const json & result) {
    if (!result.is_object() || !result.contains("error") || result.at("error").is_null()) {
        return "";
    }
    const json & err = result.at("error");
    if (err.is_string()) {
        return err.get<std::string>();
    }
    if (err.is_object() && err.contains("message") && err.at("message").is_string()) {
        return err.at("message").get<std::string>();
    }
    return safe_json_to_str(err);
}

void add_alarm(goal_runtime_state & state,
               const std::string & level,
               const std::string & code,
               const std::string & message) {
    state.alarms.push_back(build_alarm_event(level, code, message));
}

void mark_file_failed(goal_runtime_state & state,
                      file_read_state & file,
                      const std::string & code,
                      const std::string & message) {
    if (file.completed || file.skipped) {
        return;
    }
    file.failed = true;
    file.terminal_reason = code;
    add_alarm(state, "ERROR", code, file.path + ": " + message);
}

bool ranges_cover_full_file(const file_read_state & file) {
    if (file.line_count < 0) {
        return false;
    }
    if (file.line_count == 0) {
        return true;
    }

    int expected_start = 1;
    for (const auto & range : file.ranges_completed) {
        if (range.first != expected_start || range.second < range.first) {
            return false;
        }
        expected_start = range.second + 1;
    }
    return expected_start > file.line_count;
}

void refresh_file_completion(file_read_state & file) {
    if (file.failed || file.skipped) {
        return;
    }
    if (ranges_cover_full_file(file)) {
        file.completed = true;
        file.terminal_reason = "COVERAGE_FULL";
    }
}

std::string compute_progress_signature(const goal_runtime_state & state) {
    std::ostringstream oss;
    oss << state.target_files_discovered << '|'
        << state.file_discovery_failed << '|'
        << state.files.size() << '|'
        << state.total_target_bytes << '|'
        << state.total_delivered_bytes << '|'
        << state.failure_mode << '|';

    for (const auto & path : state.ordered_paths) {
        const auto it = state.files.find(path);
        if (it == state.files.end()) {
            continue;
        }
        const auto & file = it->second;
        oss << path << ':'
            << file.completed << ':'
            << file.failed << ':'
            << file.skipped << ':'
            << file.line_count << ':'
            << file.next_start_line << ':'
            << file.current_chunk_lines << ':'
            << file.ranges_completed.size() << ':'
            << file.terminal_reason << '|';
        if (!file.ranges_completed.empty()) {
            const auto & last = file.ranges_completed.back();
            oss << last.first << '-' << last.second << '|';
        }
    }
    return oss.str();
}

json build_file_coverage_snapshot(const goal_runtime_state & state) {
    json files = json::array();
    for (const auto & path : state.ordered_paths) {
        const auto it = state.files.find(path);
        if (it == state.files.end()) {
            continue;
        }
        const auto & file = it->second;
        json ranges = json::array();
        for (const auto & r : file.ranges_completed) {
            ranges.push_back(json::array({r.first, r.second}));
        }
        files.push_back(json{
            {"path", file.path},
            {"line_count", file.line_count},
            {"next_start_line", file.next_start_line},
            {"completed", file.completed},
            {"failed", file.failed},
            {"skipped", file.skipped},
            {"coverage_status", ranges_cover_full_file(file) ? "FULL" : "PARTIAL"},
            {"ranges_completed", ranges},
            {"size_bytes", file.size_bytes},
            {"delivered_bytes", file.delivered_bytes},
            {"content_hash", file.content_hash},
            {"terminal_reason", file.terminal_reason},
        });
    }
    return files;
}

json build_state_snapshot(
    const server_goal_envelope & goal,
    goal_runtime_state & state,
    const server_goal_execution_controller::loop_policy & policy) {
    int completed_count = 0;
    int failed_count = 0;
    int skipped_count = 0;
    int pending_count = 0;
    json pending_actions = json::array();

    for (auto & kv : state.files) {
        refresh_file_completion(kv.second);
    }

    if (state.file_discovery_failed) {
        if (state.failure_mode.empty()) {
            state.failure_mode = "FILE_DISCOVERY_FAILED";
        }
    } else if (!state.target_files_discovered) {
        pending_actions.push_back(make_next_action("file_glob_search", json{
            {"path", goal.root_path},
            {"include_patterns", goal.include_patterns},
            {"exclude_patterns", goal.exclude_patterns},
            {"max_results", policy.max_read_files + 1},
        }, "READ_ONLY"));
        pending_count = 1;
    } else {
        if (state.files.empty() && state.failure_mode.empty()) {
            state.failure_mode = "NO_TARGET_FILES_DISCOVERED";
            add_alarm(
                state,
                "ERROR",
                "NO_TARGET_FILES_DISCOVERED",
                "goal discovery completed but no target files matched the include/exclude filters");
        }

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

            pending_count++;
            if (file.line_count < 0) {
                pending_actions.push_back(make_next_action("file_info", json{
                    {"path", path},
                }, "READ_ONLY"));
                continue;
            }

            if (file.next_start_line > file.line_count) {
                continue;
            }

            const int chunk_lines = std::max(1, file.current_chunk_lines > 0 ? file.current_chunk_lines : goal.read_chunk_lines);
            const int end_line = std::min(file.line_count, file.next_start_line + chunk_lines - 1);
            pending_actions.push_back(make_next_action("read_file", json{
                {"path", path},
                {"start_line", file.next_start_line},
                {"end_line", end_line},
                {"append_loc", false},
            }, "READ_ONLY"));
        }
    }

    const bool discovered_nonempty = state.target_files_discovered && !state.files.empty();
    const bool goal_closed = discovered_nonempty && pending_count == 0;
    const bool goal_complete = goal_closed && failed_count == 0 && skipped_count == 0 && state.failure_mode.empty();

    std::string final_status_hint = "IN_PROGRESS";
    std::string failure_mode = state.failure_mode;
    if (goal_complete) {
        final_status_hint = "COMPLETE";
    } else if (goal_closed && (failed_count > 0 || skipped_count > 0)) {
        final_status_hint = "PARTIAL_COMPLETE_WITH_SKIPS";
        failure_mode = "PARTIAL_COMPLETE_WITH_SKIPS";
    } else if (!failure_mode.empty() && pending_actions.empty()) {
        final_status_hint = "FAILED";
    }

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
        {"failure_mode", failure_mode},
        {"goal_closed", goal_closed},
        {"final_status_hint", final_status_hint},
        {"coverage", build_file_coverage_snapshot(state)},
        {"total_target_bytes", state.total_target_bytes},
        {"total_delivered_bytes", state.total_delivered_bytes},
    };
}

void apply_file_glob_result(
    goal_runtime_state & state,
    const server_goal_execution_controller::loop_policy & policy,
    const json & result) {
    const std::string error = json_error_message(result);
    if (!error.empty()) {
        state.file_discovery_failed = true;
        state.failure_mode = "FILE_DISCOVERY_FAILED";
        add_alarm(state, "ERROR", "FILE_DISCOVERY_FAILED", error);
        return;
    }

    if (!result.contains("matches") || !result.at("matches").is_array()) {
        state.file_discovery_failed = true;
        state.failure_mode = "FILE_DISCOVERY_RESULT_INVALID";
        add_alarm(state, "ERROR", "FILE_DISCOVERY_RESULT_INVALID", "file_glob_search result did not contain matches[]");
        return;
    }

    const json & matches = result.at("matches");
    const int returned_count = json_int_value(result, "returned_count", static_cast<int>(matches.size()));
    const int total_matches = json_int_value(result, "total_matches", returned_count);

    state.target_files_discovered = true;

    if (total_matches > policy.max_read_files || returned_count > policy.max_read_files || static_cast<int>(matches.size()) > policy.max_read_files) {
        state.file_discovery_failed = true;
        state.failure_mode = "MAX_READ_FILES_EXCEEDED";
        add_alarm(state, "ERROR", "MAX_READ_FILES_EXCEEDED", "too many target files discovered for current Phase 1 policy");
        return;
    }

    if (total_matches > returned_count) {
        state.file_discovery_failed = true;
        state.failure_mode = "FILE_GLOB_RESULT_TRUNCATED";
        add_alarm(state, "ERROR", "FILE_GLOB_RESULT_TRUNCATED", "file_glob_search returned fewer files than total_matches; refusing incomplete target set");
        return;
    }

    for (const auto & item : matches) {
        if (!item.is_string()) {
            continue;
        }
        const std::string path = item.get<std::string>();
        if (path.empty() || state.files.count(path) > 0) {
            continue;
        }
        file_read_state file;
        file.path = path;
        state.files.emplace(path, file);
        state.ordered_paths.push_back(path);
    }
}

void apply_file_info_result(
    goal_runtime_state & state,
    file_read_state & file,
    const server_goal_execution_controller::loop_policy & policy,
    const json & result) {
    const std::string error = json_error_message(result);
    if (!error.empty()) {
        mark_file_failed(state, file, "FILE_INFO_FAILED", error);
        return;
    }

    if (!result.contains("line_count")) {
        mark_file_failed(state, file, "FILE_INFO_MISSING_LINE_COUNT", "file_info did not return line_count");
        return;
    }

    file.line_count = std::max(0, json_int_value(result, "line_count", -1));
    file.size_bytes = json_size_value(result, "size_bytes", 0);
    file.content_hash = result.value("content_hash", "");
    file.metadata_loaded = true;
    state.total_target_bytes += file.size_bytes;

    if (state.total_target_bytes > static_cast<size_t>(policy.max_read_bytes)) {
        mark_file_failed(state, file, "MAX_READ_BYTES_EXCEEDED", "target corpus exceeds max_read_bytes policy");
        state.failure_mode = "MAX_READ_BYTES_EXCEEDED";
        return;
    }

    if (file.line_count == 0) {
        file.completed = true;
        file.terminal_reason = "EMPTY_FILE";
    }
}

void apply_read_file_result(
    goal_runtime_state & state,
    file_read_state & file,
    const server_goal_envelope & goal,
    const server_tool_call_envelope & action,
    const json & result) {
    const std::string error = json_error_message(result);
    if (!error.empty()) {
        mark_file_failed(state, file, "READ_FILE_FAILED", error);
        return;
    }

    const int requested_start_line = action.params.value("start_line", 1);
    const int requested_end_line = action.params.value("end_line", requested_start_line);
    const int lines_read = json_int_value(result, "line_count_read", 0);
    const bool output_truncated = result.value("output_truncated", false);
    const bool reached_eof = result.value("reached_eof", false);
    const std::string response_text = result.value("plain_text_response", "");

    file.delivered_bytes += response_text.size();
    state.total_delivered_bytes += response_text.size();

    if (requested_start_line != file.next_start_line) {
        mark_file_failed(state, file, "READ_RANGE_OUT_OF_ORDER", "read_file did not match expected next_start_line");
        return;
    }

    if (lines_read <= 0) {
        if (reached_eof && file.line_count >= 0 && requested_start_line > file.line_count) {
            refresh_file_completion(file);
            return;
        }
        mark_file_failed(state, file, output_truncated ? "READ_CHUNK_TRUNCATED_WITH_ZERO_LINES" : "READ_CHUNK_EMPTY", "read_file made no line progress");
        return;
    }

    const int actual_end_line = requested_start_line + lines_read - 1;
    if (actual_end_line > requested_end_line) {
        mark_file_failed(state, file, "READ_FILE_RESULT_RANGE_INVALID", "line_count_read exceeds requested range");
        return;
    }

    file.ranges_completed.push_back({requested_start_line, actual_end_line});
    file.last_end_line = std::max(file.last_end_line, actual_end_line);
    file.next_start_line = actual_end_line + 1;

    if (output_truncated) {
        const int previous_chunk = file.current_chunk_lines > 0 ? file.current_chunk_lines : goal.read_chunk_lines;
        file.current_chunk_lines = std::max(1, std::min(lines_read, std::max(1, previous_chunk / 2)));
        add_alarm(
            state,
            "WARN",
            "READ_CHUNK_TRUNCATED_RETRY_SMALLER_RANGE",
            file.path + ": read_file output was truncated; continuing from actual delivered line with smaller chunks");
    } else {
        file.current_chunk_lines = std::max(1, goal.read_chunk_lines);
    }

    if (file.line_count < 0 && reached_eof) {
        file.line_count = actual_end_line;
    }

    refresh_file_completion(file);
}

void apply_tool_result(
    goal_runtime_state & state,
    const server_goal_envelope & goal,
    const server_tool_call_envelope & action,
    const json & result,
    const server_goal_execution_controller::loop_policy & policy) {
    if (action.tool_name == "file_glob_search") {
        apply_file_glob_result(state, policy, result);
        return;
    }

    const std::string path = action.params.value("path", "");
    if (path.empty()) {
        state.failure_mode = "ACTION_PATH_MISSING";
        add_alarm(state, "ERROR", "ACTION_PATH_MISSING", "tool action did not contain params.path");
        return;
    }

    auto it = state.files.find(path);
    if (it == state.files.end()) {
        state.failure_mode = "ACTION_TARGET_FILE_UNKNOWN";
        add_alarm(state, "ERROR", "ACTION_TARGET_FILE_UNKNOWN", "tool action referenced a file outside target set: " + path);
        return;
    }

    file_read_state & file = it->second;
    if (action.tool_name == "file_info") {
        apply_file_info_result(state, file, policy, result);
        return;
    }
    if (action.tool_name == "read_file") {
        apply_read_file_result(state, file, goal, action, result);
        return;
    }

    state.failure_mode = "UNEXPECTED_PHASE1_TOOL_RESULT";
    add_alarm(state, "ERROR", "UNEXPECTED_PHASE1_TOOL_RESULT", "unexpected Phase 1 tool result: " + action.tool_name);
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

json summarize_tool_result(const std::string & tool_name, const json & result) {
    if (!result.is_object()) {
        return result;
    }

    json summary = json::object();
    if (result.contains("error")) {
        summary["error"] = result.at("error");
    }

    if (tool_name == "read_file") {
        for (const char * key : {"path", "start_line", "end_line", "line_count_read", "reached_eof", "output_truncated"}) {
            if (result.contains(key)) {
                summary[key] = result.at(key);
            }
        }
        if (result.contains("plain_text_response") && result.at("plain_text_response").is_string()) {
            summary["plain_text_response_bytes"] = result.at("plain_text_response").get<std::string>().size();
        }
        return summary;
    }

    if (tool_name == "file_glob_search") {
        for (const char * key : {"path", "returned_count", "total_matches"}) {
            if (result.contains(key)) {
                summary[key] = result.at(key);
            }
        }
        return summary;
    }

    return result;
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

json terminal_result(
    const server_goal_envelope & goal,
    const std::string & acceptance_status,
    const std::string & final_status,
    const std::string & failure_mode,
    const server_goal_progress & progress,
    const json & alarms) {
    server_goal_acceptance acceptance;
    acceptance.goal_id = goal.goal_id;
    acceptance.acceptance_status = acceptance_status;
    acceptance.final_status = final_status;
    acceptance.failure_mode = failure_mode;
    acceptance.progress = progress;
    acceptance.alarms = alarms;
    finalize_acceptance(goal, acceptance);
    return build_goal_result(goal, acceptance);
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
    const auto started_at = std::chrono::steady_clock::now();

    server_trace_registry::record_stage(goal.trace_id, "goal_start", to_json(goal));

    for (int step = 0; step < policy.max_steps; ++step) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - started_at).count();
        if (elapsed_ms > policy.max_runtime_ms) {
            server_goal_progress progress = router.acceptance_check(goal, build_state_snapshot(goal, state, policy)).progress;
            json alarms = state.alarms;
            alarms.push_back(build_alarm_event("ERROR", "MAX_RUNTIME_EXCEEDED", "goal executor exceeded max_runtime_ms"));
            return terminal_result(goal, "NOT_COMPLETE", "TIMEOUT", "MAX_RUNTIME_EXCEEDED", progress, alarms);
        }

        const json state_snapshot = test_acceptance_snapshot.is_object() ? test_acceptance_snapshot : build_state_snapshot(goal, state, policy);
        test_acceptance_snapshot = nullptr;

        const server_goal_route_decision acceptance_route = router.acceptance_check(goal, state_snapshot);
        server_trace_registry::append_event(goal.trace_id, "acceptance_check", json{
            {"goal_id", goal.goal_id},
            {"step", step},
            {"decision", to_json(acceptance_route)},
            {"coverage", state_snapshot.value("coverage", json::array())},
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
            const bool partial = acceptance_route.failure_mode == "PARTIAL_COMPLETE_WITH_SKIPS";
            server_goal_acceptance acceptance;
            acceptance.goal_id = goal.goal_id;
            acceptance.acceptance_status = partial ? "PARTIAL" : "NOT_COMPLETE";
            acceptance.final_status = partial ? "PARTIAL_COMPLETE_WITH_SKIPS" : "FAILED";
            acceptance.failure_mode = partial ? "" : acceptance_route.failure_mode;
            acceptance.progress = acceptance_route.progress;
            acceptance.alarms = acceptance_route.alarms;
            finalize_acceptance(goal, acceptance);
            return build_goal_result(goal, acceptance);
        }

        if (acceptance_route.next_actions.empty()) {
            json alarms = acceptance_route.alarms;
            alarms.push_back(build_alarm_event(
                "ERROR",
                "NO_NEXT_ACTION_FOR_INCOMPLETE_GOAL",
                "acceptance route is incomplete but produced no next_action"));
            return terminal_result(
                goal,
                "NOT_COMPLETE",
                "FAILED",
                "NO_NEXT_ACTION_FOR_INCOMPLETE_GOAL",
                acceptance_route.progress,
                alarms);
        }

        const int action_limit = std::min(
            static_cast<int>(acceptance_route.next_actions.size()),
            policy.max_pending_actions_per_round);

        for (int i = 0; i < action_limit; ++i) {
            json action_json = acceptance_route.next_actions.at(i);
            std::string action_shape_failure_mode;
            json action_shape_alarm;
            if (!normalize_next_action_json(action_json, action_shape_failure_mode, action_shape_alarm)) {
                return terminal_result(
                    goal,
                    "NOT_COMPLETE",
                    "FAILED",
                    action_shape_failure_mode,
                    acceptance_route.progress,
                    json::array({action_shape_alarm}));
            }

            server_tool_call_envelope action = make_action_envelope(goal, action_json);
            const bool permission_write = is_write_tool_(action.tool_name);
            const std::string action_signature = action.tool_name + "|" + action.params.dump();
            action_repeat_counts[action_signature]++;
            if (action_repeat_counts[action_signature] > policy.max_same_action_repeat + 1) {
                return terminal_result(
                    goal,
                    "NOT_COMPLETE",
                    "DEADLOCK_DETECTED",
                    "DUPLICATE_ACTION_LOOP",
                    acceptance_route.progress,
                    json::array({build_alarm_event("ERROR", "DUPLICATE_ACTION_LOOP", "same action repeated too many times")}));
            }

            const server_goal_route_decision pre = router.pre_guard(goal, action, permission_write);
            server_trace_registry::append_event(goal.trace_id, "clips_pre_guard", json{
                {"goal_id", goal.goal_id},
                {"tool_call", to_json(action)},
                {"decision", to_json(pre)},
            });

            if (!pre.allowed) {
                return terminal_result(
                    goal,
                    "NOT_COMPLETE",
                    "NEEDS_HUMAN_APPROVAL",
                    pre.failure_mode,
                    acceptance_route.progress,
                    pre.alarms);
            }

            json result = invoke_tool_(action.tool_name, action.params);
            action.delivery_status = json_error_message(result).empty() ? "SUCCESS" : "FAILED";
            action.result_hash = build_server_result_hash(result);

            server_trace_registry::append_event(goal.trace_id, "mcp_tool_result", json{
                {"goal_id", goal.goal_id},
                {"tool_call", to_json(action)},
                {"result_hash", action.result_hash},
                {"result", summarize_tool_result(action.tool_name, result)},
            });

            const server_goal_route_decision post = router.post_guard(goal, action, result);
            server_trace_registry::append_event(goal.trace_id, "clips_post_guard", json{
                {"goal_id", goal.goal_id},
                {"tool_call", to_json(action)},
                {"decision", to_json(post)},
            });

            if (!post.verified) {
                apply_tool_result(state, goal, action, result, policy);
                break;
            }

            apply_tool_result(state, goal, action, result, policy);

            if (state.total_delivered_bytes > static_cast<size_t>(policy.max_read_bytes)) {
                json alarms = state.alarms;
                alarms.push_back(build_alarm_event("ERROR", "MAX_READ_BYTES_EXCEEDED", "read chain exceeded max_read_bytes policy"));
                return terminal_result(
                    goal,
                    "NOT_COMPLETE",
                    "FAILED",
                    "MAX_READ_BYTES_EXCEEDED",
                    router.acceptance_check(goal, build_state_snapshot(goal, state, policy)).progress,
                    alarms);
            }
        }

        const std::string progress_signature = compute_progress_signature(state);
        if (progress_signature == last_progress_signature) {
            same_progress_rounds++;
        } else {
            same_progress_rounds = 0;
        }

        if (same_progress_rounds >= 2) {
            json alarms = state.alarms;
            alarms.push_back(build_alarm_event(
                "ERROR",
                "NO_PROGRESS_DETECTED",
                "no progress detected across repeated acceptance rounds"));
            return terminal_result(
                goal,
                "NOT_COMPLETE",
                "DEADLOCK_DETECTED",
                "NO_PROGRESS_DETECTED",
                router.acceptance_check(goal, build_state_snapshot(goal, state, policy)).progress,
                alarms);
        }
        last_progress_signature = progress_signature;
    }

    json alarms = state.alarms;
    alarms.push_back(build_alarm_event(
        "ERROR",
        "MAX_TRACEBACK_STEPS_EXCEEDED",
        "goal executor reached max_steps"));
    return terminal_result(
        goal,
        "NOT_COMPLETE",
        "TIMEOUT",
        "MAX_TRACEBACK_STEPS_EXCEEDED",
        router.acceptance_check(goal, build_state_snapshot(goal, state, policy)).progress,
        alarms);
}