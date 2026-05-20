#include "server-clips-goal-router.h"
#include "server-supervision.h"

#include <set>
#include <string>

namespace {

bool result_has_error(const json & result) {
    return result.is_object() && result.contains("error") && !result.at("error").is_null();
}

bool phase1_read_only_tool_allowed(const std::string & tool_name) {
    static const std::set<std::string> allowed = {
        "read_file",
        "file_info",
        "stat_file",
        "hash_file",
        "file_glob_search",
        "grep_search",
        "list_directory",
        "read_directory_files",
    };
    return allowed.count(tool_name) != 0;
}

} // namespace

server_goal_route_decision server_clips_goal_router::pre_guard(
    const server_goal_envelope &,
    const server_tool_call_envelope & action,
    bool permission_write) const {
    server_goal_route_decision decision;
    decision.decision_type = "PRE_GUARD";
    decision.assistant_response_allowed = false;

    if (permission_write || action.safety_class != "READ_ONLY") {
        decision.allowed = false;
        decision.verified = false;
        decision.reason = "SIDE_EFFECT_ACTION_REQUIRES_HUMAN_APPROVAL";
        decision.failure_mode = decision.reason;
        decision.alarms.push_back(build_alarm_event(
            "ERROR",
            decision.reason,
            "side-effect action requires human approval"));
        return decision;
    }

    if (!phase1_read_only_tool_allowed(action.tool_name)) {
        decision.allowed = false;
        decision.verified = false;
        decision.reason = "TOOL_NOT_ALLOWED_FOR_PHASE1_READ_CHAIN";
        decision.failure_mode = decision.reason;
        decision.alarms.push_back(build_alarm_event(
            "ERROR",
            decision.reason,
            "tool is not in the Phase 1 read-only allowlist"));
        return decision;
    }

    decision.allowed = true;
    decision.verified = true;
    return decision;
}

server_goal_route_decision server_clips_goal_router::post_guard(
    const server_goal_envelope &,
    const server_tool_call_envelope &,
    const json & result) const {
    server_goal_route_decision decision;
    decision.decision_type = "POST_GUARD";
    decision.allowed = true;
    decision.verified = !result_has_error(result);
    decision.assistant_response_allowed = false;

    if (!decision.verified) {
        decision.reason = "TOOL_RESULT_NOT_VERIFIED";
        decision.failure_mode = decision.reason;
        decision.alarms.push_back(build_alarm_event(
            "ERROR",
            decision.reason,
            "tool execution failed or returned an error payload"));
    }

    return decision;
}

server_goal_route_decision server_clips_goal_router::acceptance_check(
    const server_goal_envelope &,
    const json & state_snapshot) const {
    server_goal_route_decision decision;
    decision.decision_type = "ACCEPTANCE_ROUTE";
    decision.goal_complete = state_snapshot.value("goal_complete", false);
    decision.progress.target_count = state_snapshot.value("target_count", 0);
    decision.progress.completed_count = state_snapshot.value("completed_count", 0);
    decision.progress.failed_count = state_snapshot.value("failed_count", 0);
    decision.progress.pending_count = state_snapshot.value("pending_count", 0);
    decision.progress.skipped_count = state_snapshot.value("skipped_count", 0);
    decision.next_actions = state_snapshot.value("next_actions", json::array());
    decision.alarms = state_snapshot.value("alarms", json::array());
    decision.allowed = true;
    decision.verified = true;
    decision.assistant_response_allowed = decision.goal_complete;

    if (!decision.goal_complete && decision.next_actions.empty()) {
        if (state_snapshot.value("goal_closed", false)) {
            decision.failure_mode = state_snapshot.value("final_status_hint", "PARTIAL_COMPLETE_WITH_SKIPS");
            decision.assistant_response_allowed = true;
        } else {
            decision.failure_mode = state_snapshot.value("failure_mode", "NO_NEXT_ACTION_FOR_INCOMPLETE_GOAL");
            decision.assistant_response_allowed = false;
        }
    }

    return decision;
}

json to_json(const server_goal_route_decision & decision) {
    return json{
        {"decision_type", decision.decision_type},
        {"allowed", decision.allowed},
        {"verified", decision.verified},
        {"goal_complete", decision.goal_complete},
        {"reason", decision.reason},
        {"failure_mode", decision.failure_mode},
        {"assistant_response_allowed", decision.assistant_response_allowed},
        {"progress", to_json(decision.progress)},
        {"next_actions", decision.next_actions},
        {"alarms", decision.alarms},
    };
}