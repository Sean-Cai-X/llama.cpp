#include "server-context-support.h"

#include "server-json-utils.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace server_context_support {

using server_json::flatten_message_content;
using server_json::parse_direct_payload;

namespace {

std::mutex g_closed_loop_completed_ids_mutex;
std::unordered_set<std::string> g_closed_loop_completed_ids;

std::vector<std::string> closed_loop_ids_from_payload(const json & payload) {
    std::vector<std::string> ids;
    if (!payload.is_object()) {
        return ids;
    }

    for (const char * key : {"trace_id", "goal_id", "request_id", "tool_call_id"}) {
        if (!payload.contains(key) || !payload.at(key).is_string()) {
            continue;
        }
        const std::string value = payload.at(key).get<std::string>();
        if (!value.empty()) {
            ids.push_back(value);
        }
    }
    return ids;
}

bool closed_loop_was_completed(const json & payload) {
    const std::vector<std::string> ids = closed_loop_ids_from_payload(payload);
    if (ids.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_closed_loop_completed_ids_mutex);
    return std::any_of(ids.begin(), ids.end(), [](const std::string & id) {
        return g_closed_loop_completed_ids.find(id) != g_closed_loop_completed_ids.end();
    });
}

void attach_closed_loop_cache_fact(json & payload) {
    if (!payload.is_object() || !closed_loop_was_completed(payload)) {
        return;
    }

    payload["closed_loop_completion_cache_hit"] = true;
    payload["closed_loop_completion_cache_status"] = "complete";
    payload["closed_loop_completion_cache_fact_schema_id"] = "closed_loop_completion_cache_fact_v1";

    std::ostringstream fact;
    fact << "(closed_loop_completion_cache";
    for (const char * key : {"trace_id", "goal_id", "request_id", "tool_call_id"}) {
        const std::string value = payload.value(key, "");
        if (!value.empty()) {
            fact << " (" << key << " \"" << value << "\")";
        }
    }
    fact << " (status \"complete\")";
    fact << ")";
    payload["closed_loop_completion_cache_fact"] = fact.str();
}

} // namespace

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

bool remove_trailing_empty_assistant_prefill(json & body) {
    if (!body.contains("messages") || !body.at("messages").is_array() || body.at("messages").empty()) {
        return false;
    }

    json & messages = body["messages"];
    json & last = messages.back();
    if (!last.is_object() || last.value("role", "") != "assistant") {
        return false;
    }

    const std::string content = flatten_message_content(last.value("content", json()));
    const json tool_calls = last.value("tool_calls", json::array());
    if (!content.empty() || (tool_calls.is_array() && !tool_calls.empty())) {
        return false;
    }

    messages.erase(messages.end() - 1);
    return true;
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
            attach_closed_loop_cache_fact(parsed);
            return parsed;
        }
    }

    return json::object();
}

void mark_closed_loop_complete(const json & payload) {
    const std::vector<std::string> ids = closed_loop_ids_from_payload(payload);
    if (ids.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_closed_loop_completed_ids_mutex);
    for (const std::string & id : ids) {
        g_closed_loop_completed_ids.insert(id);
    }
}

} // namespace server_context_support
