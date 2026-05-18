
#include "server-context.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-remote-session.h"
#include "server-json-utils.h"
#include "server-remote-session-turn.h"
#include "server-response-generator.h"
#include "server-rag-routes.h"
#include "server-supervision.h"
#include "server-tools.h"
#include "server-trace-registry.h"
#include "server-embedding-routes.h"
#include "server-slot-action-routes.h"
#include "server-slot.h"
#include "server-metrics.h"
#include "server-lora-manager.h"
#include "server-lora-routes.h"
#include "server-token-processor.h"
#include "server-task-dispatcher.h"
#include "server-slot-updater.h"
#include "server-speculative-decoder.h"

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "RAG/src/rag_server_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <filesystem>
#include <sstream>
#include <utility>
#include <vector>
#include <unordered_set>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

namespace {
remote_session_store g_remote_session_store;

using server_json::flatten_message_content;
using server_json::get_array_or_empty;
using server_json::get_bool_or_default;
using server_json::get_string_or_empty;
using server_json::parse_direct_payload;

using server_remote_session_turn::build_messages_from_session;
using server_remote_session_turn::build_ventriloquy_result;
using server_remote_session_turn::default_tool_availability_snapshot;

json build_rag_supporting_slice_ids(const json & approved_context) {
    json slice_ids = json::array();
    std::unordered_set<std::string> seen;
    if (!approved_context.is_array()) {
        return slice_ids;
    }

    for (const auto & item : approved_context) {
        if (!item.is_object()) {
            continue;
        }
        const std::string slice_id = item.value("slice_id", item.value("source_id", ""));
        if (slice_id.empty() || !seen.insert(slice_id).second) {
            continue;
        }
        slice_ids.push_back(slice_id);
    }

    return slice_ids;
}

struct chat_supervision_verdict {
    bool supervised = false;
    bool model_response_allowed = true;
    bool service_report_allowed = false;
    std::string supervision_state = "UNSUPERVISED";
    std::string execution_disposition = "FINALIZE_ANSWER";
    std::string failure_mode;
    std::string reason;
    std::string route;
    std::string dominant_decision;
    int approved_context_count = 0;
    std::string clips_first_decision;
    std::string next_action_0_tool_name;
    std::string next_action_0_safety_class;
    json next_action_0_params = json::object();
};

server_supervision_envelope to_supervision_envelope(const chat_supervision_verdict & verdict) {
    server_supervision_envelope envelope;
    envelope.supervision_status = verdict.supervision_state;
    envelope.execution_disposition = verdict.execution_disposition;
    envelope.response_allowed = verdict.model_response_allowed;
    envelope.failure_mode = verdict.failure_mode;
    envelope.reason = verdict.reason;
    envelope.route = verdict.route;
    envelope.dominant_decision = verdict.dominant_decision;
    envelope.approved_context_count = verdict.approved_context_count;
    if (!verdict.failure_mode.empty()) {
        envelope.alarm_code = verdict.failure_mode;
        envelope.alarm_message = verdict.reason;
    }
    return envelope;
}

bool should_force_supervision_override(const json & body) {
    return body.value("supervision_override_mode", "") == "force";
}

bool should_skip_rag_injection(const json & body) {
    return body.value("skip_rag_injection", false) || should_force_supervision_override(body);
}

std::string get_clips_first_decision(const json & body) {
    if (!body.contains("clips_first_decision")) {
        return "";
    }
    const json & value = body.at("clips_first_decision");
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (!value.is_object()) {
        return "";
    }
    return value.value("decision", value.value("status", value.value("route", "")));
}

json find_tool_continuation_payload(const json & body) {
    if (!body.contains("messages") || !body.at("messages").is_array()) {
        return json::object();
    }

    const json & messages = body.at("messages");
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (!it->is_object()) {
            continue;
        }

        const std::string content = flatten_message_content(it->value("content", json()));
        if (content.empty()) {
            continue;
        }

        json parsed = parse_direct_payload(content);
        if (!parsed.is_object() || parsed.empty()) {
            continue;
        }

        if (parsed.contains("status") ||
            parsed.contains("next_call_json") ||
            parsed.contains("continue_required") ||
            parsed.contains("clips_first_decision") ||
            parsed.contains("assistant_response_allowed") ||
            parsed.contains("final_answer_allowed") ||
            parsed.contains("required_tool_arguments_json") ||
            parsed.contains("next_action_0_params_json")) {
            return parsed;
        }
    }

    return json::object();
}

void populate_next_action_from_continuation_payload(
        chat_supervision_verdict & verdict,
        const json & payload) {
    if (payload.contains("next_action_0_tool_name") && payload.at("next_action_0_tool_name").is_string()) {
        verdict.next_action_0_tool_name = payload.at("next_action_0_tool_name").get<std::string>();
    }
    if (payload.contains("next_action_0_safety_class") && payload.at("next_action_0_safety_class").is_string()) {
        verdict.next_action_0_safety_class = payload.at("next_action_0_safety_class").get<std::string>();
    }
    if (payload.contains("next_action_0_params_json") && payload.at("next_action_0_params_json").is_string()) {
        try {
            json parsed = json::parse(payload.at("next_action_0_params_json").get<std::string>());
            if (parsed.is_object()) {
                verdict.next_action_0_params = parsed;
            }
        } catch (...) {
        }
    }

    auto load_from_call = [&](const json & call) {
        if (!call.is_object()) {
            return;
        }
        if (verdict.next_action_0_tool_name.empty()) {
            verdict.next_action_0_tool_name = call.value("tool", call.value("name", ""));
        }
        if (verdict.next_action_0_safety_class.empty()) {
            verdict.next_action_0_safety_class = "READ_ONLY";
        }
        if ((!verdict.next_action_0_params.is_object() || verdict.next_action_0_params.empty()) &&
            call.contains("params") && call.at("params").is_object()) {
            verdict.next_action_0_params = call.at("params");
        } else if ((!verdict.next_action_0_params.is_object() || verdict.next_action_0_params.empty()) &&
                   call.contains("arguments") && call.at("arguments").is_object()) {
            verdict.next_action_0_params = call.at("arguments");
        }
    };

    if (payload.contains("next_call_json")) {
        const json & next_call = payload.at("next_call_json");
        if (next_call.is_object()) {
            load_from_call(next_call);
        } else if (next_call.is_string()) {
            try {
                load_from_call(json::parse(next_call.get<std::string>()));
            } catch (...) {
            }
        }
    }

    if (payload.contains("required_tool_arguments_json") && payload.at("required_tool_arguments_json").is_string()) {
        try {
            load_from_call(json::parse(payload.at("required_tool_arguments_json").get<std::string>()));
        } catch (...) {
        }
    }
}

void populate_first_next_action(chat_supervision_verdict & verdict, const json & body) {
    verdict.next_action_0_tool_name = body.value("next_action_0_tool_name", "");
    verdict.next_action_0_safety_class = body.value("next_action_0_safety_class", "");

    if (body.contains("next_action_0_params") && body.at("next_action_0_params").is_object()) {
        verdict.next_action_0_params = body.at("next_action_0_params");
        return;
    }

    const std::string params_json = body.value("next_action_0_params_json", "");
    if (params_json.empty()) {
        return;
    }

    try {
        json parsed = json::parse(params_json);
        if (parsed.is_object()) {
            verdict.next_action_0_params = parsed;
        }
    } catch (const std::exception &) {
        verdict.next_action_0_params = json::object();
    }
}

chat_supervision_verdict evaluate_request_supervision(const json & body) {
    chat_supervision_verdict verdict;
    verdict.clips_first_decision = get_clips_first_decision(body);
    const json continuation_payload = find_tool_continuation_payload(body);
    const json admission_summary = body.value("admission_summary", json::object());
    const json approved_context = body.value("approved_context", json::array());

    verdict.supervised =
        !verdict.clips_first_decision.empty() ||
        !continuation_payload.empty() ||
        body.contains("rag_request_id") ||
        body.contains("rag_trace_id") ||
        !admission_summary.empty() ||
        (approved_context.is_array() && !approved_context.empty());
    verdict.approved_context_count = approved_context.is_array() ? static_cast<int>(approved_context.size()) : 0;

    if (!verdict.supervised) {
        return verdict;
    }

    populate_first_next_action(verdict, body);
    if (!continuation_payload.empty()) {
        if (verdict.clips_first_decision.empty()) {
            verdict.clips_first_decision = continuation_payload.value("clips_first_decision", "");
        }
        populate_next_action_from_continuation_payload(verdict, continuation_payload);
    }

    if (!verdict.clips_first_decision.empty()) {
        verdict.dominant_decision = verdict.clips_first_decision;
        if (verdict.clips_first_decision == "complete") {
            verdict.model_response_allowed = true;
            verdict.supervision_state = "APPROVED";
            verdict.execution_disposition = "FINALIZE_ANSWER";
            verdict.reason = "CLIPS_FIRST_DECISION_COMPLETE";
            verdict.route = "closed_loop_complete";
            return verdict;
        }

        verdict.model_response_allowed = false;
        verdict.service_report_allowed = true;

        if (verdict.clips_first_decision == "continue") {
            verdict.supervision_state = "IN_PROGRESS";
            verdict.execution_disposition = "CONTINUE_EXECUTION";
            verdict.failure_mode = "CLIPS_FIRST_DECISION_CONTINUE";
            verdict.reason = "CLIPS_FIRST_DECISION_REQUIRES_NEXT_ACTION";
            verdict.route = "closed_loop_continue";
            return verdict;
        }

        if (verdict.clips_first_decision == "alarm") {
            const json clips_first = body.value("clips_first_decision", json::object());
            verdict.supervision_state = "BLOCKED_BY_POLICY";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = clips_first.value("alarm_code", clips_first.value("failure_mode", "CLIPS_FIRST_DECISION_ALARM"));
            verdict.reason = clips_first.value("alarm_message", clips_first.value("reason", "CLIPS_FIRST_DECISION_RAISED_ALARM"));
            verdict.route = "alarm";
            return verdict;
        }
    }

    if (!continuation_payload.empty()) {
        const std::string continuation_status = continuation_payload.value("status", "");
        const std::string acceptance_status = continuation_payload.value("acceptance_status", "");
        const bool continue_required = get_bool_or_default(continuation_payload, "continue_required", false) ||
            get_bool_or_default(continuation_payload, "auto_continue_required", false);
        const bool assistant_response_allowed = get_bool_or_default(continuation_payload, "assistant_response_allowed", true);
        const bool final_answer_allowed = get_bool_or_default(continuation_payload, "final_answer_allowed", true);
        const bool supervision_alarm = get_bool_or_default(continuation_payload, "supervision_alarm", false);

        verdict.route = continuation_payload.value("route_target", verdict.route);
        verdict.dominant_decision = continuation_payload.value("clips_first_decision", verdict.dominant_decision);

        if (supervision_alarm || continuation_status == "failed") {
            verdict.model_response_allowed = false;
            verdict.service_report_allowed = true;
            verdict.supervision_state = "BLOCKED_BY_POLICY";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = continuation_payload.value("supervision_alarm_code",
                continuation_payload.value("error_code",
                continuation_payload.value("status_code", "MCP_TOOL_RESULT_FAILED")));
            verdict.reason = continuation_payload.value("supervision_alarm_message",
                continuation_payload.value("error_message",
                continuation_payload.value("error", "MCP tool result failed")));
            return verdict;
        }

        if (continuation_status == "needs_continue" ||
            acceptance_status == "continue" ||
            continue_required ||
            !assistant_response_allowed ||
            !final_answer_allowed) {
            verdict.model_response_allowed = false;
            verdict.service_report_allowed = true;
            verdict.supervision_state = "IN_PROGRESS";
            verdict.execution_disposition = "CONTINUE_EXECUTION";
            verdict.failure_mode = continuation_payload.value("status_code", "MCP_TOOL_RESULT_CONTINUE_REQUIRED");
            verdict.reason = continuation_payload.value("clips_first_reason",
                continuation_payload.value("acceptance_reason",
                continuation_payload.value("next_action", "MCP tool result requires continuation")));
            verdict.route = "closed_loop_continue";
            if (verdict.dominant_decision.empty()) {
                verdict.dominant_decision = "continue";
            }
            if (verdict.clips_first_decision.empty()) {
                verdict.clips_first_decision = "continue";
            }
            return verdict;
        }

        if (continuation_status == "success" && final_answer_allowed) {
            verdict.model_response_allowed = true;
            verdict.supervision_state = "APPROVED";
            verdict.execution_disposition = "FINALIZE_ANSWER";
            verdict.reason = continuation_payload.value("summary", "MCP_TOOL_RESULT_COMPLETE");
            verdict.route = "closed_loop_complete";
            if (verdict.dominant_decision.empty()) {
                verdict.dominant_decision = "complete";
            }
            if (verdict.clips_first_decision.empty()) {
                verdict.clips_first_decision = "complete";
            }
            return verdict;
        }
    }

    verdict.route = admission_summary.value("route", "");
    verdict.dominant_decision = admission_summary.value("dominant_decision", "");

    if (verdict.route == "admit_to_cognitive_layer" && verdict.approved_context_count > 0) {
        verdict.model_response_allowed = true;
        verdict.supervision_state = "APPROVED";
        verdict.execution_disposition = "FINALIZE_ANSWER";
        verdict.reason = "APPROVED_CONTEXT_READY";
        return verdict;
    }

    verdict.model_response_allowed = false;
    verdict.service_report_allowed = true;

    if (verdict.route == "repair_then_retry") {
        verdict.supervision_state = "IN_PROGRESS";
        verdict.execution_disposition = "CONTINUE_EXECUTION";
        verdict.failure_mode = "SUPERVISION_REPAIR_REQUIRED";
        verdict.reason = "CLIPS_REQUIRES_REPAIR_BEFORE_MODEL_RESPONSE";
    } else if (verdict.route == "human_review") {
        verdict.supervision_state = "NEEDS_HUMAN_REVIEW";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "SUPERVISION_REQUIRES_HUMAN_REVIEW";
        verdict.reason = "CLIPS_BLOCKED_DIRECT_RESPONSE";
    } else if (verdict.route == "session_only") {
        verdict.supervision_state = "SESSION_ONLY";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "SUPERVISION_SESSION_ONLY";
        verdict.reason = "CLIPS_ALLOWED_SHORT_TERM_ONLY";
    } else if (verdict.route == "drop_result") {
        verdict.supervision_state = "BLOCKED_BY_POLICY";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "SUPERVISION_DROP_RESULT";
        verdict.reason = "CLIPS_REJECTED_CONTEXT_ADMISSION";
    } else if (verdict.approved_context_count == 0) {
        verdict.supervision_state = "BLOCKED_BY_POLICY";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "NO_APPROVED_CONTEXT";
        verdict.reason = "SUPERVISION_CONTEXT_NOT_APPROVED";
    } else {
        verdict.supervision_state = "BLOCKED_BY_POLICY";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "SUPERVISION_ROUTE_UNRESOLVED";
        verdict.reason = "UNRESOLVED_SUPERVISION_ROUTE";
    }

    return verdict;
}

chat_supervision_verdict evaluate_output_supervision(
        const json & request_body,
        const json & output_validation) {
    chat_supervision_verdict verdict = evaluate_request_supervision(request_body);
    if (!verdict.supervised || !verdict.model_response_allowed) {
        return verdict;
    }

    if (verdict.clips_first_decision == "complete") {
        verdict.supervision_state = "APPROVED";
        verdict.execution_disposition = "FINALIZE_ANSWER";
        verdict.reason = "CLIPS_FIRST_DECISION_COMPLETE";
        verdict.route = "closed_loop_complete";
        verdict.dominant_decision = "complete";
        return verdict;
    }

    const std::string decision = output_validation.value("decision", "");
    const std::string status = output_validation.value("status", "");
    const std::string reason = output_validation.value("reason", "");

    if (decision == "PASS" && status == "APPROVED") {
        verdict.supervision_state = "APPROVED";
        verdict.execution_disposition = "FINALIZE_ANSWER";
        verdict.reason = reason.empty() ? "ALL_CLAIMS_SUPPORTED_BY_APPROVED_CONTEXT" : reason;
        return verdict;
    }

    verdict.model_response_allowed = false;
    verdict.service_report_allowed = true;

    if (decision == "REJECT" || status == "RETRY_REQUIRED") {
        verdict.supervision_state = "FAILED_VALIDATION";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "OUTPUT_VALIDATION_REJECTED";
        verdict.reason = reason.empty() ? "OUTPUT_VALIDATION_REJECTED" : reason;
    } else {
        verdict.supervision_state = "NEEDS_HUMAN_REVIEW";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = "OUTPUT_VALIDATION_REVIEW";
        verdict.reason = reason.empty() ? "OUTPUT_VALIDATION_REVIEW" : reason;
    }

    return verdict;
}

json build_supervision_json(const chat_supervision_verdict & verdict) {
    json result = to_json(to_supervision_envelope(verdict));
    result["service_report_allowed"] = verdict.service_report_allowed;
    result["clips_first_decision"] = verdict.clips_first_decision;
    if (!verdict.next_action_0_tool_name.empty()) {
        result["next_action_0_tool_name"] = verdict.next_action_0_tool_name;
    }
    if (!verdict.next_action_0_safety_class.empty()) {
        result["next_action_0_safety_class"] = verdict.next_action_0_safety_class;
    }
    if (verdict.next_action_0_params.is_object() && !verdict.next_action_0_params.empty()) {
        result["next_action_0_params"] = verdict.next_action_0_params;
    }
    return result;
}

std::string build_supervision_status_message(const chat_supervision_verdict & verdict) {
    if (verdict.clips_first_decision == "continue") {
        if (!verdict.next_action_0_tool_name.empty()) {
            return "Supervision blocked releasing a model answer. The service is continuing the closed loop by executing next_action_0.";
        }
        return "Supervision blocked releasing a model answer because clips_first_decision requested continue, but no executable next_action_0 was available.";
    }
    if (verdict.execution_disposition == "CONTINUE_EXECUTION") {
        return "Supervision blocked releasing a model answer because the request still requires repair or further execution before completion.";
    }
    return "Supervision blocked releasing the model answer. The service is returning a status report instead of model-generated natural language.";
}

json extract_continuation_payload_from_tool_result(const json & result) {
    if (!result.is_object()) {
        return json::object();
    }

    if (result.contains("status") ||
        result.contains("next_call_json") ||
        result.contains("continue_required") ||
        result.contains("clips_first_decision") ||
        result.contains("assistant_response_allowed") ||
        result.contains("final_answer_allowed") ||
        result.contains("required_tool_arguments_json") ||
        result.contains("next_action_0_params_json")) {
        return result;
    }

    if (result.contains("plain_text_response") && result.at("plain_text_response").is_string()) {
        json parsed = parse_direct_payload(result.at("plain_text_response").get<std::string>());
        if (parsed.is_object() && !parsed.empty()) {
            return parsed;
        }
    }

    return json::object();
}

bool continuation_payload_requires_continue(const json & payload) {
    if (!payload.is_object() || payload.empty()) {
        return false;
    }
    const std::string continuation_status = payload.value("status", "");
    const std::string acceptance_status = payload.value("acceptance_status", "");
    const bool continue_required = get_bool_or_default(payload, "continue_required", false) ||
        get_bool_or_default(payload, "auto_continue_required", false);
    const bool assistant_response_allowed = get_bool_or_default(payload, "assistant_response_allowed", true);
    const bool final_answer_allowed = get_bool_or_default(payload, "final_answer_allowed", true);
    return continuation_status == "needs_continue" ||
        acceptance_status == "continue" ||
        continue_required ||
        !assistant_response_allowed ||
        !final_answer_allowed;
}

bool continuation_payload_failed(const json & payload) {
    if (!payload.is_object() || payload.empty()) {
        return false;
    }
    return get_bool_or_default(payload, "supervision_alarm", false) ||
        payload.value("status", "") == "failed" ||
        (payload.contains("error") && !payload.at("error").is_null() && payload.at("error") != "");
}

void apply_continuation_payload_to_verdict(
        chat_supervision_verdict & verdict,
        const json & payload) {
    if (!payload.is_object() || payload.empty()) {
        return;
    }

    if (payload.contains("clips_first_decision") && payload.at("clips_first_decision").is_string()) {
        verdict.clips_first_decision = payload.at("clips_first_decision").get<std::string>();
    }

    if (continuation_payload_failed(payload)) {
        verdict.model_response_allowed = false;
        verdict.service_report_allowed = true;
        verdict.supervision_state = "BLOCKED_BY_POLICY";
        verdict.execution_disposition = "RETURN_FAILURE_REPORT";
        verdict.failure_mode = payload.value("supervision_alarm_code",
            payload.value("error_code",
            payload.value("status_code", "MCP_TOOL_RESULT_FAILED")));
        verdict.reason = payload.value("supervision_alarm_message",
            payload.value("error_message",
            payload.value("error", "MCP tool result failed")));
        return;
    }

    if (continuation_payload_requires_continue(payload)) {
        verdict.model_response_allowed = false;
        verdict.service_report_allowed = true;
        verdict.supervision_state = "IN_PROGRESS";
        verdict.execution_disposition = "CONTINUE_EXECUTION";
        verdict.failure_mode = payload.value("status_code", "MCP_TOOL_RESULT_CONTINUE_REQUIRED");
        verdict.reason = payload.value("clips_first_reason",
            payload.value("acceptance_reason",
            payload.value("next_action", "MCP tool result requires continuation")));
        verdict.route = "closed_loop_continue";
        if (verdict.dominant_decision.empty()) {
            verdict.dominant_decision = "continue";
        }
        if (verdict.clips_first_decision.empty()) {
            verdict.clips_first_decision = "continue";
        }
        populate_next_action_from_continuation_payload(verdict, payload);
        return;
    }

    const std::string continuation_status = payload.value("status", "");
    const bool final_answer_allowed = get_bool_or_default(payload, "final_answer_allowed", true);
    if (continuation_status == "success" && final_answer_allowed) {
        verdict.model_response_allowed = true;
        verdict.service_report_allowed = false;
        verdict.supervision_state = "APPROVED";
        verdict.execution_disposition = "FINALIZE_ANSWER";
        verdict.failure_mode.clear();
        verdict.reason = payload.value("summary", "MCP_TOOL_RESULT_COMPLETE");
        verdict.route = "closed_loop_complete";
        verdict.dominant_decision = "complete";
        verdict.clips_first_decision = "complete";
    }
}

json maybe_execute_first_continue_action(
        const common_params & params,
        chat_supervision_verdict & verdict) {
    const bool should_continue =
        verdict.clips_first_decision == "continue" ||
        (verdict.execution_disposition == "CONTINUE_EXECUTION" && !verdict.next_action_0_tool_name.empty());
    if (!should_continue) {
        return json();
    }

    json execution_chain = json::array();
    for (int step = 0; step < 1024; ++step) {
        if (verdict.next_action_0_tool_name.empty()) {
            verdict.supervision_state = "FAILED";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = "NEXT_ACTION_0_TOOL_NAME_MISSING";
            verdict.reason = "continuation was requested but next_action_0_tool_name is missing";
            return json{
                {"error", verdict.reason},
                {"failure_mode", verdict.failure_mode},
                {"continuation_execution_chain", execution_chain},
            };
        }

        const std::string tool_name = verdict.next_action_0_tool_name;
        const json tool_params = verdict.next_action_0_params;
        const std::string safety_class = verdict.next_action_0_safety_class.empty()
            ? "READ_ONLY"
            : verdict.next_action_0_safety_class;
        if (safety_class != "READ_ONLY") {
            verdict.supervision_state = "NEEDS_HUMAN_REVIEW";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = "NEXT_ACTION_0_REQUIRES_HUMAN_APPROVAL";
            verdict.reason = "next_action_0 safety_class is not READ_ONLY";
            return json{
                {"tool_name", tool_name},
                {"safety_class", safety_class},
                {"error", verdict.reason},
                {"failure_mode", verdict.failure_mode},
                {"continuation_execution_chain", execution_chain},
            };
        }

        if (!tool_params.is_object() || tool_params.empty()) {
            verdict.supervision_state = "FAILED";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = "NEXT_ACTION_0_PARAMS_MISSING";
            verdict.reason = "continuation was requested but next_action_0 params are missing";
            return json{
                {"tool_name", tool_name},
                {"safety_class", safety_class},
                {"error", verdict.reason},
                {"failure_mode", verdict.failure_mode},
                {"continuation_execution_chain", execution_chain},
            };
        }

        if (params.server_tools.empty()) {
            verdict.supervision_state = "FAILED";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = "SERVER_TOOLS_NOT_ENABLED";
            verdict.reason = "cannot execute next_action_0 because built-in tools are not enabled";
            return json{
                {"tool_name", tool_name},
                {"safety_class", safety_class},
                {"params", tool_params},
                {"error", verdict.reason},
                {"failure_mode", verdict.failure_mode},
                {"continuation_execution_chain", execution_chain},
            };
        }

        server_tools tools;
        tools.setup(params.server_tools);
        SRV_INF("continuation_execute: step=%d tool='%s'\n", step, tool_name.c_str());
        json result = tools.invoke(tool_name, tool_params);
        execution_chain.push_back(json{
            {"step", step},
            {"tool_name", tool_name},
            {"safety_class", safety_class},
            {"params", tool_params},
            {"result", result},
        });

        if (result.is_object() && result.contains("error") && !result.at("error").is_null()) {
            verdict.supervision_state = "FAILED";
            verdict.execution_disposition = "RETURN_FAILURE_REPORT";
            verdict.failure_mode = "NEXT_ACTION_0_EXECUTION_FAILED";
            verdict.reason = "next_action_0 execution failed";
            return json{
                {"tool_name", tool_name},
                {"safety_class", safety_class},
                {"params", tool_params},
                {"result", result},
                {"continuation_execution_chain", execution_chain},
            };
        }

        const json continuation_payload = extract_continuation_payload_from_tool_result(result);
        if (!continuation_payload.empty()) {
            apply_continuation_payload_to_verdict(verdict, continuation_payload);
            if (continuation_payload_requires_continue(continuation_payload) &&
                !verdict.next_action_0_tool_name.empty()) {
                continue;
            }
        }

        return json{
            {"tool_name", tool_name},
            {"safety_class", safety_class},
            {"params", tool_params},
            {"result", result},
            {"continuation_execution_chain", execution_chain},
        };
    }

    verdict.supervision_state = "FAILED";
    verdict.execution_disposition = "RETURN_FAILURE_REPORT";
    verdict.failure_mode = "NEXT_ACTION_0_CONTINUATION_TOO_DEEP";
    verdict.reason = "continuation execution exceeded max steps";
    return json{
        {"error", verdict.reason},
        {"failure_mode", verdict.failure_mode},
        {"continuation_execution_chain", execution_chain},
    };
}

json build_supervision_blocked_chat_response(
        const std::string & model_name,
        const json & body,
        const chat_supervision_verdict & verdict,
        const json & next_action_execution = json()) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto created = std::chrono::duration_cast<std::chrono::seconds>(now).count();
    json response = json{
        {"id", gen_chatcmplid()},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model_name},
        {"choices", json::array({
            json{
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", build_supervision_status_message(verdict)},
                }},
                {"finish_reason", "stop"},
            }
        })},
        {"request_id", body.value("rag_request_id", body.value("request_id", ""))},
        {"trace_id", body.value("rag_trace_id", body.value("trace_id", ""))},
        {"query_id", body.value("rag_query_id", body.value("query_id", ""))},
        {"approved_context", body.value("approved_context", json::array())},
        {"admission_summary", body.value("admission_summary", json::object())},
        {"supervision", build_supervision_json(verdict)},
    };
    if (next_action_execution.is_object() && !next_action_execution.empty()) {
        response["next_action_0_execution"] = next_action_execution;
    }
    return response;
}

std::string extract_completion_text(const json & payload) {
    if (payload.contains("choices") && payload["choices"].is_array() && !payload["choices"].empty()) {
        const json & choice = payload["choices"][0];
        if (choice.contains("message") && choice["message"].is_object()) {
            return flatten_message_content(choice["message"].value("content", json()));
        }
        if (choice.contains("text") && choice["text"].is_string()) {
            return choice["text"].get<std::string>();
        }
    }
    return "";
}

json build_llama_output_validation(
        const std::string & request_id,
        const std::string & trace_id,
        const std::string & model_task_id,
        const std::string & content,
        const json & approved_context,
        const json & supporting_slice_ids) {
    const bool has_content = !content.empty();
    const bool has_approved_context = approved_context.is_array() && !approved_context.empty();
    const bool has_supporting_slice_ids = supporting_slice_ids.is_array() && !supporting_slice_ids.empty();

    std::string decision = "PASS";
    std::string reason = "ALL_CLAIMS_SUPPORTED_BY_APPROVED_CONTEXT";
    std::string status = "APPROVED";

    if (!has_content) {
        decision = "REJECT";
        reason = "EMPTY_OUTPUT";
        status = "RETRY_REQUIRED";
    } else if (!has_approved_context) {
        decision = "REVIEW";
        reason = "NO_APPROVED_CONTEXT";
        status = "HUMAN_REVIEW";
    } else if (!has_supporting_slice_ids) {
        decision = "REVIEW";
        reason = "MISSING_SUPPORTING_SLICE_IDS";
        status = "HUMAN_REVIEW";
    }

    return json{
        {"model_task_id", model_task_id},
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"decision", decision},
        {"reason", reason},
        {"status", status},
    };
}

void attach_rag_completion_metadata(json & payload, const json & data, const std::string & model_task_id) {
    const json approved_context = data.value("approved_context", json::array());
    const json supporting_slice_ids = data.contains("supporting_slice_ids")
        ? data["supporting_slice_ids"]
        : build_rag_supporting_slice_ids(approved_context);
    const std::string request_id = data.value("rag_request_id", data.value("request_id", ""));
    const std::string trace_id = data.value("rag_trace_id", data.value("trace_id", ""));
    const std::string query_id = data.value("rag_query_id", data.value("query_id", ""));
    const std::string content = extract_completion_text(payload);
    const std::string content_hash = content.empty()
        ? ""
        : "HASH-" + std::to_string(std::hash<std::string>{}(content));
    const bool approved_by_clips = approved_context.is_array() && !approved_context.empty();
    const json output_validation = build_llama_output_validation(
        request_id,
        trace_id,
        model_task_id,
        content,
        approved_context,
        supporting_slice_ids);
    const chat_supervision_verdict supervision_verdict = evaluate_output_supervision(data, output_validation);

    if (!request_id.empty()) {
        payload["request_id"] = request_id;
    }
    if (!trace_id.empty()) {
        payload["trace_id"] = trace_id;
    }
    if (!query_id.empty()) {
        payload["query_id"] = query_id;
    }

    payload["approved_context"] = approved_context;
    payload["supporting_slice_ids"] = supporting_slice_ids;
    if (data.contains("admission_summary")) {
        payload["admission_summary"] = data["admission_summary"];
    }

    payload["llama_output_candidate"] = json{
        {"model_task_id", model_task_id},
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"query_id", query_id},
        {"output_hash", content_hash},
        {"status", "CANDIDATE"},
    };

    payload["final_output"] = json{
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"content_hash", content_hash},
        {"approved_by_clips", approved_by_clips},
        {"supporting_slice_ids", supporting_slice_ids},
        {"status", supervision_verdict.model_response_allowed ? (approved_by_clips ? "APPROVED" : "UNCONSTRAINED") : "SUPPRESSED"},
    };
    payload["llama_output_validation"] = output_validation;
    payload["supervision"] = build_supervision_json(supervision_verdict);

    if (!supervision_verdict.model_response_allowed) {
        const std::string status_message = build_supervision_status_message(supervision_verdict);
        if (payload.contains("choices") && payload["choices"].is_array() && !payload["choices"].empty()) {
            json & first_choice = payload["choices"][0];
            if (first_choice.contains("message") && first_choice["message"].is_object()) {
                first_choice["message"]["content"] = status_message;
                first_choice["message"].erase("reasoning_content");
            } else if (first_choice.contains("text")) {
                first_choice["text"] = status_message;
            }
            first_choice["finish_reason"] = "stop";
        }
    }

    if (payload.contains("structured_conclusion") && payload["structured_conclusion"].is_object()) {
        payload["structured_conclusion"]["request_id"] = request_id;
        payload["structured_conclusion"]["trace_id"] = trace_id;
        payload["structured_conclusion"]["rag_request_id"] = request_id;
        payload["structured_conclusion"]["rag_trace_id"] = trace_id;
        payload["structured_conclusion"]["rag_query_id"] = query_id;
        payload["structured_conclusion"]["approved_context"] = approved_context;
        payload["structured_conclusion"]["supporting_slice_ids"] = supporting_slice_ids;
        payload["structured_conclusion"]["llama_output_validation"] = output_validation;
        payload["structured_conclusion"]["supervision"] = payload["supervision"];
        if (data.contains("admission_summary")) {
            payload["structured_conclusion"]["admission_summary"] = data["admission_summary"];
        }
    }

    server_trace_registry::record_stage(trace_id, "completion", json{
        {"trace_id", trace_id},
        {"request_id", request_id},
        {"query_id", query_id},
        {"model_task_id", model_task_id},
        {"approved_context", approved_context},
        {"supporting_slice_ids", supporting_slice_ids},
        {"admission_summary", data.value("admission_summary", json::object())},
        {"llama_output_candidate", payload["llama_output_candidate"]},
        {"llama_output_validation", payload["llama_output_validation"]},
        {"supervision", payload["supervision"]},
        {"final_output", payload["final_output"]},
        {"structured_conclusion", payload.value("structured_conclusion", json::object())},
    });
}

void propagate_rag_request_fields(const json & source_body, json & parsed_body) {
    for (const auto & key : {
            "request_id",
            "trace_id",
            "query_id",
            "rag_request_id",
            "rag_trace_id",
            "rag_query_id",
            "approved_context",
            "admission_summary",
            "supporting_slice_ids"
        }) {
        if (source_body.contains(key)) {
            parsed_body[key] = source_body.at(key);
        }
    }
}

void record_injected_rag_stage(const json & body) {
    const std::string trace_id = body.value("rag_trace_id", body.value("trace_id", ""));
    if (trace_id.empty()) {
        return;
    }

    server_trace_registry::record_stage(trace_id, "rag_chat_context", json{
        {"trace_id", trace_id},
        {"request_id", body.value("rag_request_id", body.value("request_id", ""))},
        {"query_id", body.value("rag_query_id", body.value("query_id", ""))},
        {"approved_context", body.value("approved_context", json::array())},
        {"supporting_slice_ids", build_rag_supporting_slice_ids(body.value("approved_context", json::array()))},
        {"admission_summary", body.value("admission_summary", json::object())},
    });
}

}  // namespace


enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};






//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;
    std::unique_ptr<RagServerRuntime> rag_runtime;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;

    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    server_token_processor::runtime make_token_processor_runtime();

    server_task_dispatcher::runtime make_task_dispatcher_runtime();

    server_slot_updater::runtime make_slot_updater_runtime();

    server_speculative_decoder::runtime make_speculative_decoder_runtime();

    void destroy() {
        if (rag_runtime) {
            rag_runtime->shutdown();
            rag_runtime.reset();
        }

        llama_init.reset();

        ctx = nullptr;
        model = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        for (server_slot & slot : slots) {
            if (slot.can_speculate()) {
                slot.spec.reset();
            }
        }

        llama_batch_free(batch);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CLEAR_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;

        llama_init = common_init_from_params(params_base);

        model = llama_init->model();
        ctx   = llama_init->context();

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            SRV_INF("loading draft model '%s'\n", params_base.speculative.mparams_dft.path.c_str());

            const auto & params_spec = params_base.speculative;

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams_dft;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model);

        int n_ctx_slot = llama_n_ctx_seq(ctx);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        const auto spec_type = common_speculative_is_compat(ctx);
        if (spec_type == COMMON_SPECULATIVE_COMPAT_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (spec_type == COMMON_SPECULATIVE_COMPAT_TYPE_CKPT) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
            params_base.speculative.use_checkpoints = true;
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id    = i;
            slot.ctx   = ctx;
            slot.n_ctx = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            // try speculative decoding
            if (spec_type != COMMON_SPECULATIVE_COMPAT_TYPE_NO) {
                slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx));

                if (slot.spec) {
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                }
            }

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("slots debug = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_WRN("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_WRN("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        if (params_base.rag_enable) {
            rag_runtime = std::make_unique<RagServerRuntime>();
            if (!rag_runtime->init(params_base, model, ctx, vocab)) {
                SRV_WRN("%s", "failed to initialize isolated RAG runtime, disabling built-in RAG\n");
                rag_runtime.reset();
                params_base.rag_enable = false;
            } else {
                SRV_INF("%s", "isolated RAG runtime initialized\n");
            }
        }

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(model != nullptr);
        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.clear_idle) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s: --clear-idle requires --kv-unified, disabling\n", __func__);
                params_base.clear_idle = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s: --clear-idle requires --cache-ram, disabling\n", __func__);
                params_base.clear_idle = false;
            } else {
                SRV_INF("%s: idle slots will be saved to prompt cache and cleared upon starting a new task\n", __func__);
                SRV_DBG("%s", "__TEST_TAG_CLEAR_IDLE_ENABLED__\n");
            }
        }

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            SRV_INF("%s: chat defaults: use_jinja=%d, prefill_assistant=%d, enable_reasoning=%d, template_supports_thinking=%d, reasoning_format=%d, reasoning_budget=%d, force_pure_content=%d\n",
                __func__,
                params_base.use_jinja,
                params_base.prefill_assistant,
                params_base.enable_reasoning,
                template_supports_thinking,
                static_cast<int>(params_base.reasoning_format),
                params_base.reasoning_budget,
                params_base.force_pure_content_parser);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.reasoning_budget,
                /* reasoning_budget_msg  */ params_base.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }


    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        const auto lora_apply_result =
            server_lora_manager::apply_task_lora_to_slot(
                slot,
                task,
                params_base.lora_adapters);

        if (!lora_apply_result.ok) {
            send_error(
                task,
                lora_apply_result.error_message,
                lora_apply_result.error);
            return false;
        }

        if (!task.tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_logits = task.params.sampling.n_probs > 0;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate() && task.params.speculative.n_max > 0);

            // TODO: getting post/pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx, slot.id, nullptr);
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(ctx, i);
            } else {
                embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        const auto & cur = slot.prompt.checkpoints.emplace_back(server_get_checkpoint(ctx, slot.id, slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max));

        SLT_WRN(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.clear_idle) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
            {
                server_task_dispatcher::handle_cancel(
                    make_task_dispatcher_runtime(),
                    task);
            } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
            {
                server_task_dispatcher::handle_next_response(
                    make_task_dispatcher_runtime(),
                    task);
            } break;
            case SERVER_TASK_TYPE_METRICS:
            {
                server_task_dispatcher::handle_metrics(
                    make_task_dispatcher_runtime(),
                    task);
            } break;
            
            case SERVER_TASK_TYPE_SLOT_SAVE:
            {
                server_task_dispatcher::handle_slot_save(
                    make_task_dispatcher_runtime(),
                    std::move(task));
            } break;
           
            case SERVER_TASK_TYPE_SLOT_RESTORE:
            {
                server_task_dispatcher::handle_slot_restore(
                    make_task_dispatcher_runtime(),
                    std::move(task));
            } break;
           
            case SERVER_TASK_TYPE_SLOT_ERASE:
            {
                server_task_dispatcher::handle_slot_erase(
                    make_task_dispatcher_runtime(),
                    std::move(task));
            } break;

            case SERVER_TASK_TYPE_GET_LORA:
            {
                server_task_dispatcher::handle_get_lora(
                    make_task_dispatcher_runtime(),
                    task);
            } break;

            case SERVER_TASK_TYPE_SET_LORA:
            {
                server_task_dispatcher::handle_set_lora(
                    make_task_dispatcher_runtime(),
                    task);
            } break;

        }
    }

  



    void update_slots() { 
            if (server_slot_updater::all_slots_idle(slots)) {
                SRV_INF("%s", "all slots are idle\n");
                return;
            }

            server_slot_updater::post_next_response(queue_tasks);

            server_slot_updater::apply_context_shift(
                make_slot_updater_runtime());

        // prompt eval / decode / sampling 主体暂时保留在这里
        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        const bool memoryless_context = llama_get_memory(ctx) == nullptr;

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            if (memoryless_context && slot_batched) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = memoryless_context ? llama_n_ubatch(ctx) : llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                if (memoryless_context && slot_batched && slot_batched != &slot) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            server_token_processor::send_final_response(
                                make_token_processor_runtime(),
                                slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        const bool requires_single_ubatch = memoryless_context || !slot.can_split();

                        if (requires_single_ubatch) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process with the current encoder/runtime limits. "
                                               "split or shorten the input, or increase the physical batch size (current n_ubatch: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
                                            llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // note: when n_swa == 0, the model does not use SWA
                            const auto n_swa = std::max(0, llama_model_n_swa(model));

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa);

                            if (n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min, n_swa);

                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        const size_t checkpoint_size = it->data.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                                        } else {
                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.data.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            server_token_processor::send_partial_response(
                                make_token_processor_runtime(),
                                slot,
                                completion_token_output{},
                                true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_INF(slot, "n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, p0, -1)) {
                        SLT_WRN(slot, "failed to truncate tokens with position >= %d - clearing the memory\n", p0);

                        slot.prompt_clear(true);

                        // there is no common part left
                        slot.n_prompt_tokens_cache = 0;
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model uses SWA and we are not using `swa_full`
                    // - the model architecture is marked as recurrent or hybrid
                    //
                    // TODO: try to make this conditional on the context or the memory module, instead of the model type
                    do_checkpoint = do_checkpoint && (
                            llama_model_is_recurrent(model) ||
                            llama_model_is_hybrid(model) ||
                            (llama_model_n_swa(model) > 0 && !params_base.swa_full)
                            );

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                        SLT_INF(slot, "prompt processing done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);
                    } else {
                        if (slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch) {
                            // near the end of the prompt
                            do_checkpoint = do_checkpoint && true;
                        } else {
                            // only do non-end checkpoints if the "checkpoint every n tokens" option is set
                            do_checkpoint = do_checkpoint && params_base.checkpoint_every_nt > 0;

                            if (do_checkpoint) {
                                llama_pos last_checkpoint = 0;
                                if (!slot.prompt.checkpoints.empty()) {
                                    last_checkpoint = slot.prompt.checkpoints.back().n_tokens;
                                }

                                do_checkpoint = do_checkpoint && slot.prompt.n_tokens() - batch.n_tokens - last_checkpoint >= params_base.checkpoint_every_nt;

                                if (do_checkpoint) {
                                    SLT_INF(slot, "%d tokens since last checkpoint at %d, creating new checkpoint during processing at position %d\n", params_base.checkpoint_every_nt, last_checkpoint, slot.prompt.n_tokens());
                                }
                            }
                        }

                        SLT_INF(slot, "prompt processing progress, n_tokens = %d, batch.n_tokens = %d, progress = %f\n", slot.prompt.n_tokens(), batch.n_tokens, (float) slot.prompt.n_tokens() / slot.task->n_tokens());
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                    // no need for empty or small checkpoints
                    do_checkpoint = do_checkpoint && (pos_min >= 0 && slot.prompt.n_tokens() >= 64);

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || slot.prompt.n_tokens() - n_tokens_cur > slot.prompt.checkpoints.back().n_tokens + 64);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx, slot_batched->task->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = memoryless_context ? llama_n_ubatch(ctx) : llama_n_batch(ctx);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        server_token_processor::send_partial_response(
                            make_token_processor_runtime(),
                            slot,
                            completion_token_output{},
                            true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec.get(), slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    server_token_processor::populate_token_probs(
                        make_token_processor_runtime(),
                        slot,
                        result,
                        slot.task->params.post_sampling_probs,
                        tok_idx);
                }

                if (!server_token_processor::process_token(
                    make_token_processor_runtime(),
                    result,
                    slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    server_token_processor::send_final_response(
                        make_token_processor_runtime(),
                        slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
            }

            server_speculative_decoder::process_accepted_drafts(
                make_speculative_decoder_runtime(),
                slots);
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }




};



server_token_processor::runtime server_context_impl::make_token_processor_runtime() {
    server_token_processor::runtime rt;

    rt.queue_results = &queue_results;
    rt.ctx = ctx;
    rt.vocab = vocab;
    rt.params = &params_base;
    rt.slots_debug = slots_debug;
    rt.special = params_base.special;

    return rt;
}

server_task_dispatcher::runtime server_context_impl::make_task_dispatcher_runtime() {
    server_task_dispatcher::runtime rt;

    rt.queue_tasks = &queue_tasks;
    rt.queue_results = &queue_results;

    rt.params = &params_base;
    rt.metrics = &metrics;

    rt.ctx = ctx;
    rt.vocab = vocab;

    rt.slots = &slots;

    rt.slots_debug = slots_debug;

    rt.get_slot_by_id = [this](int id_slot) -> server_slot* {
        return get_slot_by_id(id_slot);
        };

    rt.send_error = [this](
        const server_task& task,
        const std::string& message,
        error_type type) {
            send_error(task, message, type);
        };

    rt.check_no_mtmd = [this](int id_task) -> bool {
        return check_no_mtmd(id_task);
        };

    return rt;
}

server_slot_updater::runtime server_context_impl::make_slot_updater_runtime() {
    server_slot_updater::runtime rt;

    rt.queue_tasks = &queue_tasks;

    rt.params = &params_base;

    rt.ctx = ctx;
    rt.mctx = mctx;

    rt.slots = &slots;

    rt.add_bos_token = add_bos_token;

    rt.send_error = [this](
        server_slot& slot,
        const std::string& message,
        error_type type) {
            send_error(slot, message, type);
        };

    return rt;
}

server_speculative_decoder::runtime server_context_impl::make_speculative_decoder_runtime() {
    server_speculative_decoder::runtime rt;

    rt.token_rt = make_token_processor_runtime();
    rt.metrics = &metrics;
    rt.special = params_base.special;

    return rt;
}

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* n_ubatch               */ llama_n_ubatch(impl->ctx),
        /* pooling_type           */ llama_pooling_type(impl->ctx),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model),
        /* model_n_params         */ llama_model_n_params(impl->model),
        /* model_size             */ llama_model_size(impl->model),
    };
}


void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                attach_rag_completion_metadata(arr[0], data, completion_id);
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                attach_rag_completion_metadata(arr[0], data, completion_id);
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_chat_with_builtin_tool_loop(
            const server_http_req & req,
            json body,
            task_response_type res_type) {
    if ((res_type != TASK_RESPONSE_TYPE_OAI_CHAT && res_type != TASK_RESPONSE_TYPE_OAI_RESP) ||
        json_value(body, "stream", false) ||
        params.server_tools.empty() ||
        !body.contains("tools") ||
        !body.at("tools").is_array() ||
        body.at("tools").empty()) {
        std::vector<raw_buffer> files;
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        propagate_rag_request_fields(body, body_parsed);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            res_type);
    }

    if (!body.contains("messages") || !body.at("messages").is_array()) {
        auto res = create_response();
        res->error(format_error_response("'messages' must be an array for native tool loop", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    server_tools tools;
    tools.setup(params.server_tools);

    constexpr int SERVER_NATIVE_TOOL_LOOP_MAX_STEPS = 128;
    for (int step = 0; step < SERVER_NATIVE_TOOL_LOOP_MAX_STEPS; ++step) {
        std::vector<raw_buffer> files;
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        propagate_rag_request_fields(body, body_parsed);

        auto model_res = handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            res_type);
        if (!model_res || model_res->status != 200 || model_res->is_stream()) {
            return model_res;
        }

        json response_json = json::parse(model_res->data);
        json message = json::object();
        json tool_calls = json::array();
        if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            const json output_items = response_json.value("output", json::array());
            if (!output_items.is_array()) {
                return model_res;
            }

            json content_parts = json::array();
            for (const auto & item : output_items) {
                if (!item.is_object()) {
                    continue;
                }
                const std::string item_type = item.value("type", "");
                if (item_type == "message" && item.value("role", "") == "assistant") {
                    const json content = item.value("content", json::array());
                    if (content.is_array()) {
                        for (const auto & part : content) {
                            if (!part.is_object()) {
                                continue;
                            }
                            if (part.value("type", "") == "output_text" && part.contains("text")) {
                                content_parts.push_back(json{
                                    {"type", "text"},
                                    {"text", part.value("text", "")},
                                });
                            }
                        }
                    }
                } else if (item_type == "function_call") {
                    tool_calls.push_back(json{
                        {"id", item.value("call_id", "")},
                        {"type", "function"},
                        {"function", {
                            {"name", item.value("name", "")},
                            {"arguments", item.value("arguments", "")},
                        }},
                    });
                }
            }
            message = json{
                {"role", "assistant"},
                {"content", content_parts.empty() ? json("") : content_parts},
            };
            if (!tool_calls.empty()) {
                message["tool_calls"] = tool_calls;
            }
        } else {
            const json choices = response_json.value("choices", json::array());
            if (!choices.is_array() || choices.empty() || !choices[0].is_object()) {
                return model_res;
            }
            message = choices[0].value("message", json::object());
            tool_calls = message.value("tool_calls", json::array());
        }

        if (!tool_calls.is_array() || tool_calls.empty()) {
            return model_res;
        }

        SRV_INF("native_tool_loop: model_step=%d tool_calls=%zu\n", step, tool_calls.size());

        json assistant_message = {
            {"role", "assistant"},
            {"content", message.contains("content") ? message.at("content") : json("")},
        };
        if (message.contains("tool_calls")) {
            assistant_message["tool_calls"] = message.at("tool_calls");
        }
        if (message.contains("reasoning_content")) {
            assistant_message["reasoning_content"] = message.at("reasoning_content");
        }
        body["messages"].push_back(assistant_message);

        for (const auto & tool_call : tool_calls) {
            if (!tool_call.is_object()) {
                continue;
            }

            const std::string tool_call_id = tool_call.value("id", gen_tool_call_id());
            const json function = tool_call.value("function", json::object());
            const std::string tool_name = function.value("name", "");
            json tool_params = json::object();

            if (function.contains("arguments") && function.at("arguments").is_string()) {
                try {
                    json parsed = json::parse(function.at("arguments").get<std::string>());
                    if (parsed.is_object()) {
                        tool_params = std::move(parsed);
                    }
                } catch (const std::exception & e) {
                    tool_params = json{
                        {"_argument_parse_error", e.what()},
                    };
                }
            }

            SRV_INF("native_tool_loop: executing tool_call_id='%s' tool='%s'\n",
                tool_call_id.c_str(),
                tool_name.c_str());
            json tool_result = tools.invoke(tool_name, tool_params);
            body["messages"].push_back(json{
                {"role", "tool"},
                {"tool_call_id", tool_call_id},
                {"content", safe_json_to_str(tool_result)},
            });
        }
    }

    auto res = create_response();
    res->error(format_error_response("native tool loop exceeded max steps", ERROR_TYPE_SERVER));
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_remote_session_turn(
            const server_http_req & req,
            const json & request_body,
            const std::string & requested_session_id,
            bool append_mode) {
    auto res = create_response();

    try {
        const auto t_start_total = std::chrono::steady_clock::now();
        json body = request_body;

        std::string session_id = requested_session_id.empty()
            ? get_string_or_empty(body, "session_id")
            : requested_session_id;
        if (session_id.empty()) {
            session_id = g_remote_session_store.create_session_id();
        }

        std::optional<json> existing_session = g_remote_session_store.get_session(session_id);
        if (append_mode && !existing_session.has_value()) {
            res->error(format_error_response("Unknown session_id for append_turn", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string write_mode = append_mode ? "append" : "new";
        const std::string turn_id = get_string_or_empty(body, "turn_id").empty()
            ? g_remote_session_store.create_turn_id()
            : get_string_or_empty(body, "turn_id");

        const auto t_start_build_messages = std::chrono::steady_clock::now();
        body["session_id"] = session_id;
        body["turn_id"] = turn_id;
        body["stream"] = false;
        body["messages"] = build_messages_from_session(existing_session, body, append_mode);

        if (!body.contains("response_format")) {
            body["response_format"] = json{
                {"type", "json_schema"},
                {"json_schema", {
                    {"name", "ventriloquy_reply"},
                    {"schema", {
                        {"type", "object"},
                        {"properties", {
                            {"direct_answer", {{"type", "string"}}},
                            {"evidence", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                            {"next_action", {{"type", "string"}}},
                            {"confidence", {{"type", "string"}, {"enum", json::array({"confirmed", "likely", "unclear", "blocked"})}}}
                        }},
                        {"required", json::array({"direct_answer", "evidence", "next_action", "confidence"})},
                        {"additionalProperties", false}
                    }}
                }}
            };
        }

        if (!body.contains("max_tokens") && !body.contains("n_predict")) {
            body["max_tokens"] = 512;
        }
        const auto build_messages_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start_build_messages).count();

        const chat_supervision_verdict supervision_verdict = evaluate_request_supervision(body);
        if (supervision_verdict.supervised && !supervision_verdict.model_response_allowed) {
            chat_supervision_verdict mutable_verdict = supervision_verdict;
            const json next_action_execution = maybe_execute_first_continue_action(params, mutable_verdict);
            const std::string result_ref = get_string_or_empty(body, "result_ref").empty()
                ? ("session:" + session_id + "/turn:" + turn_id)
                : get_string_or_empty(body, "result_ref");
            const std::string evidence_ref = get_string_or_empty(body, "evidence_ref").empty()
                ? result_ref
                : get_string_or_empty(body, "evidence_ref");
            const json response_json = build_supervision_blocked_chat_response(meta->model_name, body, mutable_verdict, next_action_execution);
            json normalized = build_ventriloquy_result(response_json, session_id, turn_id, write_mode);
            const json messages = body["messages"];
            const std::string user_text = messages.empty()
                ? ""
                : flatten_message_content(messages.back().value("content", json()));
            const json tool_availability_snapshot = body.contains("tool_availability_snapshot")
                && !body.at("tool_availability_snapshot").is_null()
                    ? body.at("tool_availability_snapshot")
                    : default_tool_availability_snapshot();

            json metadata = {
                {"turn_id", turn_id},
                {"write_mode", write_mode},
                {"task_id", get_string_or_empty(body, "task_id")},
                {"task_group_id", get_string_or_empty(body, "task_group_id")},
                {"source_type", get_string_or_empty(body, "source_type")},
                {"source_label", get_string_or_empty(body, "source_label")},
                {"handoff_from", get_string_or_empty(body, "handoff_from")},
                {"handoff_to", get_string_or_empty(body, "handoff_to")},
                {"takeover_relation", get_string_or_empty(body, "takeover_relation")},
                {"speaker_mode", get_string_or_empty(body, "speaker_mode")},
                {"reasoning_level", get_string_or_empty(body, "reasoning_level")},
                {"prompt_purpose", get_string_or_empty(body, "prompt_purpose")},
                {"response_mode", get_string_or_empty(body, "response_mode")},
                {"context_refs", get_array_or_empty(body, "context_refs")},
                {"tool_availability_snapshot", tool_availability_snapshot},
                {"user_text", user_text},
                {"assistant_text", build_supervision_status_message(mutable_verdict)},
                {"summary", build_supervision_status_message(mutable_verdict)},
                {"direct_answer", normalized.value("direct_answer", "")},
                {"next_action", normalized.value("next_action", "")},
                {"confidence", normalized.value("confidence", "blocked")},
                {"result_ref", result_ref},
                {"evidence_ref", evidence_ref},
                {"admission_summary", body.value("admission_summary", json::object())},
                {"supervision", build_supervision_json(mutable_verdict)},
                {"timings", json{
                    {"build_messages_ms", build_messages_ms},
                    {"model_completion_ms", 0},
                    {"normalize_result_ms", 0},
                    {"persist_session_ms", 0}
                }}
            };

            const auto t_start_persist = std::chrono::steady_clock::now();
            const json persisted_turn = g_remote_session_store.upsert_turn(session_id, metadata, body, response_json);
            const auto persist_session_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start_persist).count();
            const auto remote_session_turn_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start_total).count();

            metadata["timings"]["persist_session_ms"] = persist_session_ms;
            normalized["result_ref"] = result_ref;
            normalized["evidence_ref"] = evidence_ref;
            normalized["admission_summary"] = body.value("admission_summary", json::object());
            normalized["supervision"] = build_supervision_json(mutable_verdict);
            if (next_action_execution.is_object() && !next_action_execution.empty()) {
                normalized["next_action_0_execution"] = next_action_execution;
            }
            normalized["provider_id"] = get_string_or_empty(persisted_turn, "provider_id");
            normalized["capability_id"] = get_string_or_empty(persisted_turn, "capability_id");
            normalized["slice_id"] = get_string_or_empty(persisted_turn, "slice_id");
            normalized["slice_path"] = get_string_or_empty(persisted_turn, "slice_path");
            normalized["timings"] = json{
                {"build_messages_ms", build_messages_ms},
                {"model_completion_ms", 0},
                {"normalize_result_ms", 0},
                {"persist_session_ms", persist_session_ms},
                {"remote_session_turn_total_ms", remote_session_turn_total_ms}
            };

            const std::string trace_id = body.value("rag_trace_id", body.value("trace_id", ""));
            server_trace_registry::record_stage(trace_id, "supervision_gate_pre_model", json{
                {"trace_id", trace_id},
                {"request_id", body.value("rag_request_id", body.value("request_id", ""))},
                {"query_id", body.value("rag_query_id", body.value("query_id", ""))},
                {"admission_summary", body.value("admission_summary", json::object())},
                {"approved_context", body.value("approved_context", json::array())},
                {"supervision", build_supervision_json(mutable_verdict)},
                {"next_action_0_execution", next_action_execution},
            });

            res->ok(normalized);
            return res;
        }

        const auto t_start_model_completion = std::chrono::steady_clock::now();
        std::vector<raw_buffer> files;
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        auto model_res = handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);

        if (!model_res || model_res->status != 200) {
            return model_res;
        }
        if (model_res->is_stream()) {
            model_res->error(format_error_response("remote session turns do not support stream mode", ERROR_TYPE_INVALID_REQUEST));
            return model_res;
        }
        const auto model_completion_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start_model_completion).count();

        const auto t_start_normalize = std::chrono::steady_clock::now();
        json response_json = json::parse(model_res->data);
        json normalized = build_ventriloquy_result(response_json, session_id, turn_id, write_mode);
        const auto normalize_result_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start_normalize).count();

        const json structured = response_json.value("structured_conclusion", json::object());
        const json message = response_json.value("choices", json::array()).empty()
            ? json::object()
            : response_json["choices"][0].value("message", json::object());
        const std::string assistant_text = flatten_message_content(message.value("content", json()));
        const json messages = body["messages"];
        const std::string user_text = messages.empty()
            ? ""
            : flatten_message_content(messages.back().value("content", json()));

        const std::string result_ref = get_string_or_empty(body, "result_ref").empty()
            ? ("session:" + session_id + "/turn:" + turn_id)
            : get_string_or_empty(body, "result_ref");
        const std::string evidence_ref = get_string_or_empty(body, "evidence_ref").empty()
            ? result_ref
            : get_string_or_empty(body, "evidence_ref");
        const json tool_availability_snapshot = body.contains("tool_availability_snapshot")
            && !body.at("tool_availability_snapshot").is_null()
                ? body.at("tool_availability_snapshot")
                : default_tool_availability_snapshot();

        json metadata = {
            {"turn_id", turn_id},
            {"write_mode", write_mode},
            {"codex_request_id", get_string_or_empty(body, "codex_request_id")},
            {"agent_dispatch_id", get_string_or_empty(body, "agent_dispatch_id")},
            {"task_id", get_string_or_empty(body, "task_id")},
            {"task_group_id", get_string_or_empty(body, "task_group_id")},
            {"source_type", get_string_or_empty(body, "source_type")},
            {"source_label", get_string_or_empty(body, "source_label")},
            {"source_detail", get_string_or_empty(body, "source_detail")},
            {"handoff_from", get_string_or_empty(body, "handoff_from")},
            {"handoff_to", get_string_or_empty(body, "handoff_to")},
            {"takeover_relation", get_string_or_empty(body, "takeover_relation")},
            {"authorization_default", get_string_or_empty(body, "authorization_default")},
            {"auto_authorize", get_bool_or_default(body, "auto_authorize", false)},
            {"speaker_mode", get_string_or_empty(body, "speaker_mode")},
            {"reasoning_level", get_string_or_empty(body, "reasoning_level")},
            {"primary_intent", get_string_or_empty(body, "primary_intent")},
            {"prompt_purpose", get_string_or_empty(body, "prompt_purpose")},
            {"response_mode", get_string_or_empty(body, "response_mode")},
            {"context_refs", get_array_or_empty(body, "context_refs")},
            {"tool_availability_snapshot", tool_availability_snapshot},
            {"tool_first_policy", get_string_or_empty(body, "tool_first_policy")},
            {"multi_file_read_policy", get_string_or_empty(body, "multi_file_read_policy")},
            {"long_session_policy", get_string_or_empty(body, "long_session_policy")},
            {"permission_policy", get_string_or_empty(body, "permission_policy")},
            {"async_reply_policy", get_string_or_empty(body, "async_reply_policy")},
            {"raw_tool_error", get_string_or_empty(body, "raw_tool_error")},
            {"async_task_id", get_string_or_empty(body, "async_task_id")},
            {"config_source", get_string_or_empty(body, "config_source")},
            {"model", get_string_or_empty(body, "model")},
            {"system_message", get_string_or_empty(body, "systemMessage")},
            {"disable_reasoning_parsing", get_bool_or_default(body, "disableReasoningParsing", false)},
            {"exclude_reasoning_from_context", get_bool_or_default(body, "excludeReasoningFromContext", false)},
            {"reasoning_strength_level", get_string_or_empty(body, "reasoningStrengthLevel")},
            {"show_tool_call_in_progress", get_bool_or_default(body, "showToolCallInProgress", false)},
            {"always_show_agentic_turns", get_bool_or_default(body, "alwaysShowAgenticTurns", false)},
            {"py_interpreter_enabled", get_bool_or_default(body, "pyInterpreterEnabled", false)},
            {"pdf_as_image", get_bool_or_default(body, "pdfAsImage", false)},
            {"stream", get_bool_or_default(body, "stream", false)},
            {"timings_per_token", get_bool_or_default(body, "timings_per_token", false)},
            {"backend_sampling", get_bool_or_default(body, "backend_sampling", false)},
            {"samplers", get_string_or_empty(body, "samplers")},
            {"temperature", body.contains("temperature") ? body["temperature"] : json()},
            {"max_tokens", body.contains("max_tokens") ? body["max_tokens"] : json()},
            {"top_k", body.contains("top_k") ? body["top_k"] : json()},
            {"top_p", body.contains("top_p") ? body["top_p"] : json()},
            {"min_p", body.contains("min_p") ? body["min_p"] : json()},
            {"repeat_penalty", body.contains("repeat_penalty") ? body["repeat_penalty"] : json()},
            {"inherited_model_config", body.contains("inherited_model_config") ? body["inherited_model_config"] : json::object()},
            {"user_text", user_text},
            {"assistant_text", assistant_text},
            {"summary", get_string_or_empty(structured, "summary").empty() ? normalized.value("direct_answer", "") : get_string_or_empty(structured, "summary")},
            {"direct_answer", normalized.value("direct_answer", "")},
            {"next_action", normalized.value("next_action", "")},
            {"confidence", normalized.value("confidence", "unclear")},
            {"result_ref", result_ref},
            {"evidence_ref", evidence_ref},
            {"timings", json{
                {"build_messages_ms", build_messages_ms},
                {"model_completion_ms", model_completion_ms},
                {"normalize_result_ms", normalize_result_ms},
                {"persist_session_ms", 0}
            }}
        };

        const auto t_start_persist = std::chrono::steady_clock::now();
        const json persisted_turn = g_remote_session_store.upsert_turn(session_id, metadata, body, response_json);
        const auto persist_session_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start_persist).count();
        const auto remote_session_turn_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start_total).count();

        metadata["timings"]["persist_session_ms"] = persist_session_ms;

        normalized["result_ref"] = result_ref;
        normalized["evidence_ref"] = evidence_ref;
        normalized["codex_request_id"] = get_string_or_empty(body, "codex_request_id");
        normalized["agent_dispatch_id"] = get_string_or_empty(body, "agent_dispatch_id");
        normalized["tool_availability_snapshot"] = tool_availability_snapshot;
        normalized["tool_first_policy"] = get_string_or_empty(body, "tool_first_policy");
        normalized["multi_file_read_policy"] = get_string_or_empty(body, "multi_file_read_policy");
        normalized["long_session_policy"] = get_string_or_empty(body, "long_session_policy");
        normalized["permission_policy"] = get_string_or_empty(body, "permission_policy");
        normalized["async_reply_policy"] = get_string_or_empty(body, "async_reply_policy");
        normalized["raw_tool_error"] = get_string_or_empty(body, "raw_tool_error");
        normalized["async_task_id"] = get_string_or_empty(body, "async_task_id");
        normalized["provider_id"] = get_string_or_empty(persisted_turn, "provider_id");
        normalized["capability_id"] = get_string_or_empty(persisted_turn, "capability_id");
        normalized["slice_id"] = get_string_or_empty(persisted_turn, "slice_id");
        normalized["slice_path"] = get_string_or_empty(persisted_turn, "slice_path");
        normalized["dedup_key"] = get_string_or_empty(persisted_turn, "dedup_key");
        normalized["dedup_hash"] = get_string_or_empty(persisted_turn, "dedup_hash");
        normalized["canonical_slice_id"] = get_string_or_empty(persisted_turn, "canonical_slice_id");
        normalized["canonical_status"] = get_string_or_empty(persisted_turn, "canonical_status");
        normalized["error_signature"] = get_string_or_empty(persisted_turn, "error_signature");
        normalized["solution_summary"] = get_string_or_empty(persisted_turn, "solution_summary");
        normalized["strategy_family"] = get_string_or_empty(persisted_turn, "strategy_family");
        normalized["similarity_score"] = persisted_turn.contains("similarity_score")
            ? persisted_turn["similarity_score"]
            : 0.0;
        normalized["vector_ready"] = persisted_turn.value("vector_ready", false);
        normalized["vector_skip_reason"] = get_string_or_empty(persisted_turn, "vector_skip_reason");
        normalized["slice_refs"] = persisted_turn.contains("slice_refs")
            ? persisted_turn["slice_refs"]
            : json::array();
        normalized["storage_refs"] = persisted_turn.contains("storage_refs")
            ? persisted_turn["storage_refs"]
            : json::array();
        normalized["timings"] = json{
            {"build_messages_ms", build_messages_ms},
            {"model_completion_ms", model_completion_ms},
            {"normalize_result_ms", normalize_result_ms},
            {"persist_session_ms", persist_session_ms},
            {"remote_session_turn_total_ms", remote_session_turn_total_ms}
        };
        res->ok(normalized);
        return res;
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(
        queue_tasks,
        queue_results,
        HTTP_POLLING_SECONDS,
        params.sleep_idle_seconds,
        bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    auto make_embedding_route_context = [this]() {
        return server_embedding_routes::route_context{
            /*.params       = */ params,
            /*.n_ubatch     = */ meta->n_ubatch,
            /*.model        = */ ctx_server.model,
            /*.vocab        = */ ctx_server.vocab,
            /*.mctx         = */ ctx_server.mctx,
            /*.model_name   = */ meta->model_name,
            /*.pooling_type = */ meta->pooling_type,
        };
    };

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req& req) {
        return server_slot_action_routes::handle_slots_action(
            req,
            params,
            [this]() {
                return create_response();
            });
        };


    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->get_api_show = [this](const server_http_req &) {
        auto res = create_response();
        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        json data = {
            {
                "model_info", {
                    { "llama.context_length", meta->slot_n_ctx },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", tmpl_default},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})}
        };

        res->ok(data);
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

this->post_chat_completions = [this](const server_http_req & req) {
    auto res = create_response();
    json body = json::parse(req.body);

    std::string rag_error;
    const bool skip_rag_injection = should_skip_rag_injection(body);
    if (!skip_rag_injection) {
        if (!server_rag_routes::maybe_inject_chat_context(
                params,
                ctx_server.rag_runtime.get(),
                body,
                &rag_error) &&
            !rag_error.empty()) {
            SRV_WRN("RAG injection skipped: %s\n", rag_error.c_str());
        }
    } else {
        SRV_INF("chat supervision override active: skipping RAG injection (mode=%s, skip_rag_injection=%d)\n",
            body.value("supervision_override_mode", "").c_str(),
            body.value("skip_rag_injection", false) ? 1 : 0);
    }
    record_injected_rag_stage(body);
    const chat_supervision_verdict supervision_verdict = evaluate_request_supervision(body);
    if (supervision_verdict.supervised && !supervision_verdict.model_response_allowed) {
        chat_supervision_verdict mutable_verdict = supervision_verdict;
        const json next_action_execution = maybe_execute_first_continue_action(params, mutable_verdict);
        const std::string trace_id = body.value("rag_trace_id", body.value("trace_id", ""));
        server_trace_registry::record_stage(trace_id, "supervision_gate_pre_model", json{
            {"trace_id", trace_id},
            {"request_id", body.value("rag_request_id", body.value("request_id", ""))},
            {"query_id", body.value("rag_query_id", body.value("query_id", ""))},
            {"skip_rag_injection", skip_rag_injection},
            {"supervision_override_mode", body.value("supervision_override_mode", "")},
            {"admission_summary", body.value("admission_summary", json::object())},
            {"approved_context", body.value("approved_context", json::array())},
            {"supervision", build_supervision_json(mutable_verdict)},
            {"next_action_0_execution", next_action_execution},
        });
        res->ok(build_supervision_blocked_chat_response(meta->model_name, body, mutable_verdict, next_action_execution));
        return res;
    }

    return handle_chat_with_builtin_tool_loop(
        req,
        body,
        TASK_RESPONSE_TYPE_OAI_CHAT);
};

    this->post_remote_session_new_turn = [this](const server_http_req & req) {
        auto res = create_response();
        GGML_UNUSED(res);
        json body = json::parse(req.body);
        return handle_remote_session_turn(req, body, "", false);
    };

    this->post_remote_session_append_turn = [this](const server_http_req & req) {
        auto res = create_response();
        GGML_UNUSED(res);
        json body = json::parse(req.body);
        const std::string session_id = req.get_param("session_id", get_string_or_empty(body, "session_id"));
        return handle_remote_session_turn(req, body, session_id, true);
    };

    this->get_remote_sessions = [this](const server_http_req & req) {
        auto res = create_response(true);
        int limit = 50;
        try {
            const std::string limit_str = req.get_param("limit");
            if (!limit_str.empty()) {
                limit = std::max(1, std::stoi(limit_str));
            }
        } catch (const std::exception &) {
            limit = 50;
        }

        res->ok(json{
            {"record_model", "remote_session_list_v1"},
            {"items", g_remote_session_store.list_sessions(limit)}
        });
        return res;
    };

    this->get_remote_session = [this](const server_http_req & req) {
        auto res = create_response(true);
        const std::string session_id = req.get_param("session_id");
        if (session_id.empty()) {
            res->error(format_error_response("session_id is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const auto session = g_remote_session_store.get_session(session_id);
        if (!session.has_value()) {
            res->error(format_error_response("session not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        res->ok(*session);
        return res;
    };

this->post_responses_oai = [this](const server_http_req & req) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = convert_responses_to_chatcmpl(json::parse(req.body));

    std::string rag_error;
    const bool skip_rag_injection = should_skip_rag_injection(body);
    if (!skip_rag_injection) {
        if (!server_rag_routes::maybe_inject_chat_context(
                params,
                ctx_server.rag_runtime.get(),
                body,
                &rag_error) &&
            !rag_error.empty()) {
            SRV_WRN("RAG injection skipped: %s\n", rag_error.c_str());
        }
    } else {
        SRV_INF("responses supervision override active: skipping RAG injection (mode=%s, skip_rag_injection=%d)\n",
            body.value("supervision_override_mode", "").c_str(),
            body.value("skip_rag_injection", false) ? 1 : 0);
    }
    record_injected_rag_stage(body);
    const chat_supervision_verdict supervision_verdict = evaluate_request_supervision(body);
    if (supervision_verdict.supervised && !supervision_verdict.model_response_allowed) {
        chat_supervision_verdict mutable_verdict = supervision_verdict;
        const json next_action_execution = maybe_execute_first_continue_action(params, mutable_verdict);
        const std::string trace_id = body.value("rag_trace_id", body.value("trace_id", ""));
        server_trace_registry::record_stage(trace_id, "supervision_gate_pre_model", json{
            {"trace_id", trace_id},
            {"request_id", body.value("rag_request_id", body.value("request_id", ""))},
            {"query_id", body.value("rag_query_id", body.value("query_id", ""))},
            {"skip_rag_injection", skip_rag_injection},
            {"supervision_override_mode", body.value("supervision_override_mode", "")},
            {"admission_summary", body.value("admission_summary", json::object())},
            {"approved_context", body.value("approved_context", json::array())},
            {"supervision", build_supervision_json(mutable_verdict)},
            {"next_action_0_execution", next_action_execution},
        });
        res->ok(build_supervision_blocked_chat_response(meta->model_name, body, mutable_verdict, next_action_execution));
        return res;
    }

    SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", body.dump().c_str());

    json body_parsed = oaicompat_chat_params_parse(
        body,
        meta->chat_params,
        files);
    propagate_rag_request_fields(body, body_parsed);

    return handle_chat_with_builtin_tool_loop(
        req,
        body,
        TASK_RESPONSE_TYPE_OAI_RESP);
};

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       meta->model_name},
                    {"aliases",  meta->model_aliases},
                    {"tags",     meta->model_tags},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     {
                        {"vocab_type",  meta->model_vocab_type},
                        {"n_vocab",     meta->model_vocab_n_tokens},
                        {"n_ctx_train", meta->model_n_ctx_train},
                        {"n_embd",      meta->model_n_embd_inp},
                        {"n_params",    meta->model_n_params},
                        {"size",        meta->model_size},
                    }},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this, make_embedding_route_context](const server_http_req& req) {
        return server_embedding_routes::handle_embeddings(
            req,
            make_embedding_route_context(),
            TASK_RESPONSE_TYPE_NONE,
            [this]() {
                return create_response();
            });
        };

    this->post_embeddings_oai = [this, make_embedding_route_context](const server_http_req& req) {
        return server_embedding_routes::handle_embeddings(
            req,
            make_embedding_route_context(),
            TASK_RESPONSE_TYPE_OAI_EMBD,
            [this]() {
                return create_response();
            });
        };

    this->post_rerank = [this, make_embedding_route_context](const server_http_req& req) {
        return server_embedding_routes::handle_rerank(
            req,
            make_embedding_route_context(),
            [this]() {
                return create_response();
            });
        };

this->post_rag_index = [this](const server_http_req & req) {
    return server_rag_routes::handle_index(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->get_rag_index_status = [this](const server_http_req & req) {
    return server_rag_routes::handle_index_status(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->post_rag_add = [this](const server_http_req & req) {
    return server_rag_routes::handle_add(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->post_rag_search = [this](const server_http_req & req) {
    return server_rag_routes::handle_search(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->post_rag_explain = [this](const server_http_req & req) {
    return server_rag_routes::handle_explain(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->post_rag_chat_context = [this](const server_http_req & req) {
    return server_rag_routes::handle_chat_context(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->post_rag_clips_meta = [this](const server_http_req & req) {
    return server_rag_routes::handle_clips_meta(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->get_rag_clips_manifest = [this](const server_http_req & req) {
    return server_rag_routes::handle_clips_manifest(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->post_rag_clips_run = [this](const server_http_req & req) {
    return server_rag_routes::handle_clips_run(
        req,
        params,
        ctx_server.rag_runtime.get(),
        [this]() {
            return create_response();
        });
};

this->get_debug_trace = [this](const server_http_req & req) {
    auto res = create_response(true);
    const std::string trace_id = req.get_param("trace_id");
    if (trace_id.empty()) {
        res->error(format_error_response("trace_id is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    json payload;
    if (!server_trace_registry::get_trace(trace_id, payload)) {
        res->error(format_error_response("trace_id not found", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    res->ok(payload);
    return res;
};

this->get_debug_request = [this](const server_http_req & req) {
    auto res = create_response(true);
    const std::string request_id = req.get_param("request_id");
    if (request_id.empty()) {
        res->error(format_error_response("request_id is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    json payload;
    if (!server_trace_registry::get_request(request_id, payload)) {
        res->error(format_error_response("request_id not found", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    res->ok(payload);
    return res;
};

this->get_debug_goal = [this](const server_http_req & req) {
    auto res = create_response(true);
    const std::string goal_id = req.get_param("goal_id");
    if (goal_id.empty()) {
        res->error(format_error_response("goal_id is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    json payload;
    if (!server_trace_registry::get_goal(goal_id, payload)) {
        res->error(format_error_response("goal_id not found", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    res->ok(payload);
    return res;
};

this->get_debug_goal_events = [this](const server_http_req & req) {
    auto res = create_response(true);
    const std::string goal_id = req.get_param("goal_id");
    if (goal_id.empty()) {
        res->error(format_error_response("goal_id is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    json payload;
    if (!server_trace_registry::get_goal_events(goal_id, payload)) {
        res->error(format_error_response("goal_id events not found", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    res->ok(payload);
    return res;
};

this->get_debug_evidence = [this](const server_http_req & req) {
    auto res = create_response(true);
    const std::string slice_id = req.get_param("slice_id");
    if (slice_id.empty()) {
        res->error(format_error_response("slice_id is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    json payload;
    if (!server_trace_registry::get_evidence(slice_id, payload)) {
        res->error(format_error_response("slice_id not found", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    res->ok(payload);
    return res;
};

this->get_lora_adapters = [this](const server_http_req& req) {
    return server_lora_routes::handle_get_lora_adapters(
        req,
        [this]() {
            return create_response();
        });
    };

this->post_lora_adapters = [this](const server_http_req& req) {
    return server_lora_routes::handle_post_lora_adapters(
        req,
        [this]() {
            return create_response();
        });
    };
 
}
