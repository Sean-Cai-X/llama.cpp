#include "server-remote-session-turn.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace {

using json = server_remote_session_turn::json;

static std::string build_text_from_refs(const json & refs) {
    if (!refs.is_array() || refs.empty()) {
        return "";
    }

    std::ostringstream oss;

    for (const auto & ref : refs) {
        if (ref.is_string()) {
            oss << "- " << ref.get<std::string>() << "\n";
        }
    }

    return oss.str();
}

static std::string build_compressed_task_state(const json & turns, std::size_t retained_turn_count) {
    if (!turns.is_array() || turns.size() <= retained_turn_count) {
        return "";
    }

    const std::size_t compressed_count = turns.size() - retained_turn_count;
    const std::size_t preview_start = compressed_count > 6 ? compressed_count - 6 : 0;

    std::ostringstream oss;
    oss
        << "Compressed task state from " << compressed_count
        << " older turns. Old safety/noise text is intentionally not replayed.\n";

    for (std::size_t i = preview_start; i < compressed_count; ++i) {
        const json & turn = turns.at(i);

        const std::string turn_id = server_json::get_string_or_empty(turn, "turn_id");

        std::string summary = server_json::get_string_or_empty(turn, "summary");
        if (summary.empty()) {
            summary = server_json::get_string_or_empty(turn, "direct_answer");
        }
        if (summary.empty()) {
            summary = server_json::get_string_or_empty(turn, "next_action");
        }

        if (summary.size() > 240) {
            summary = summary.substr(0, 240);
        }

        if (!turn_id.empty() || !summary.empty()) {
            oss << "- " << (turn_id.empty() ? "turn" : turn_id) << ": " << summary << "\n";
        }
    }

    return oss.str();
}

static std::string normalize_direct_answer_candidate(const std::string & text) {
    auto trim = [](const std::string & value) -> std::string {
        const size_t start = value.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            return "";
        }

        const size_t end = value.find_last_not_of(" \t\r\n");
        return value.substr(start, end - start + 1);
    };

    auto lower_ascii = [](std::string value) -> std::string {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });

        return value;
    };

    auto is_metadata_echo = [&](const std::string & value) -> bool {
        const std::string lower = lower_ascii(value);

        return lower.rfind("reasoning_level=", 0) == 0 ||
            lower.rfind("prompt_purpose=", 0) == 0 ||
            lower.rfind("primary_intent=", 0) == 0 ||
            lower.rfind("task_id=", 0) == 0 ||
            lower.rfind("task_group_id=", 0) == 0 ||
            lower.rfind("context_refs:", 0) == 0 ||
            lower.rfind("session metadata:", 0) == 0 ||
            lower.rfind("- reasoning_level:", 0) == 0 ||
            lower.rfind("- prompt_purpose:", 0) == 0 ||
            lower.rfind("- primary_intent:", 0) == 0 ||
            lower.rfind("- task_id:", 0) == 0 ||
            lower.rfind("- task_group_id:", 0) == 0 ||
            lower.rfind("- context_refs:", 0) == 0 ||
            lower.find("authorization default") != std::string::npos;
    };

    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        std::string candidate = trim(line);

        if (candidate.empty()) {
            continue;
        }

        const std::string lower = lower_ascii(candidate);

        if (lower == "thinking process:" ||
            lower == "thinking process" ||
            lower == "thought process:" ||
            lower == "thought process" ||
            candidate == "思考过程：" ||
            candidate == "思考过程" ||
            candidate == "推理过程：" ||
            candidate == "推理过程") {
            continue;
        }

        if (is_metadata_echo(candidate)) {
            continue;
        }

        return candidate;
    }

    return "";
}

} // namespace

namespace server_remote_session_turn {

json default_tool_availability_snapshot() {
    return json{
        {"snapshot_source", "llama_cpp_b8851_remote_session"},
        {"tool_first_required", true},
        {"path_permission_verification", "requires_raw_tool_error"},
        {"multi_file_read_batch_size", "3-5"},
        {"async_final_reply_requires_task_id", true},
        {"available_tool_classes", json::array({
            "remote_session_turn",
            "remote_session_read",
            "tool_error_passthrough",
            "async_task_id_passthrough"
        })}
    };
}

json build_messages_from_session(
        const std::optional<json> & existing_session,
        const json & body,
        bool append_mode) {
    json messages = json::array();

    const std::string approval_mode = server_json::get_string_or_empty(body, "authorization_default");
    const bool auto_authorize = server_json::get_bool_or_default(body, "auto_authorize", false);

    std::string system_prompt =
        "You are a controlled remote-session assistant. "
        "Always produce a usable final answer for the user. "
        "Do not output chain-of-thought, hidden reasoning, or 'Thinking Process'. "
        "Reply in the user's language when possible. "
        "If context is insufficient, still return a concise direct_answer that explicitly states what is missing. "
        "Tool-first rule: verify path permissions and tool availability with explicit tool results before making permission, access, or unavailable-tool claims. "
        "If no raw tool error exists, do not conclude a system permission restriction; state that verification evidence is missing. "
        "Read multiple files in batches of 3-5 files per round, then explicitly continue the next batch. "
        "Never use 'please wait' or '请稍后' as a final answer unless an async task_id was actually created. "
        "When a JSON schema is requested, output only the final schema-compatible JSON object.";

    if (!approval_mode.empty() || body.contains("auto_authorize")) {
        system_prompt += auto_authorize
            ? " Authorization default is allow."
            : " Authorization default is prompt; do not assume hidden approvals or unavailable tool results.";
    }

    messages.push_back({{"role", "system"}, {"content", system_prompt}});

    const std::string prompt_purpose = server_json::get_string_or_empty(body, "prompt_purpose");
    const std::string primary_intent = server_json::get_string_or_empty(body, "primary_intent");
    const std::string reasoning_level = server_json::get_string_or_empty(body, "reasoning_level");
    const std::string task_id = server_json::get_string_or_empty(body, "task_id");
    const std::string task_group_id = server_json::get_string_or_empty(body, "task_group_id");
    const json context_refs = server_json::get_array_or_empty(body, "context_refs");

    std::string metadata_content;

    if (!reasoning_level.empty()) {
        metadata_content += "- reasoning_level: " + reasoning_level + "\n";
    }
    if (!prompt_purpose.empty()) {
        metadata_content += "- prompt_purpose: " + prompt_purpose + "\n";
    }
    if (!primary_intent.empty()) {
        metadata_content += "- primary_intent: " + primary_intent + "\n";
    }
    if (!task_id.empty()) {
        metadata_content += "- task_id: " + task_id + "\n";
    }
    if (!task_group_id.empty()) {
        metadata_content += "- task_group_id: " + task_group_id + "\n";
    }

    const json tool_availability_snapshot = body.contains("tool_availability_snapshot")
        && !body.at("tool_availability_snapshot").is_null()
            ? body.at("tool_availability_snapshot")
            : default_tool_availability_snapshot();

    metadata_content += "- tool_availability_snapshot: " + tool_availability_snapshot.dump() + "\n";

    const std::vector<std::string> policy_keys = {
        "tool_first_policy",
        "multi_file_read_policy",
        "long_session_policy",
        "permission_policy",
        "async_reply_policy",
        "raw_tool_error",
        "async_task_id"
    };

    for (const std::string & policy_key : policy_keys) {
        const std::string policy_value = server_json::get_string_or_empty(body, policy_key);
        if (!policy_value.empty()) {
            metadata_content += "- " + policy_key + ": " + policy_value + "\n";
        }
    }

    if (!context_refs.empty()) {
        metadata_content += "- context_refs:\n" + build_text_from_refs(context_refs);
    }

    if (!metadata_content.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content", std::string("Session metadata:\n") + metadata_content}
        });
    }

    if (body.contains("messages") && body.at("messages").is_array() && !body.at("messages").empty()) {
        for (const auto & message : body.at("messages")) {
            messages.push_back(message);
        }

        return messages;
    }

    if (append_mode && existing_session.has_value()) {
        const json turns = existing_session->value("turns", json::array());
        const std::size_t retained_turn_count = 8;

        const std::string compressed_state = build_compressed_task_state(turns, retained_turn_count);
        if (!compressed_state.empty()) {
            messages.push_back({
                {"role", "system"},
                {"content", std::string("Compressed prior task state:\n") + compressed_state}
            });
        }

        const std::size_t start_index = turns.is_array() && turns.size() > retained_turn_count
            ? turns.size() - retained_turn_count
            : 0;

        for (std::size_t i = start_index; turns.is_array() && i < turns.size(); ++i) {
            const auto & turn = turns.at(i);

            const std::string user_text = server_json::get_string_or_empty(turn, "user_text");
            const std::string assistant_text = server_json::get_string_or_empty(turn, "assistant_text");

            if (!user_text.empty()) {
                messages.push_back({{"role", "user"}, {"content", user_text}});
            }

            if (!assistant_text.empty()) {
                messages.push_back({{"role", "assistant"}, {"content", assistant_text}});
            }
        }
    }

    const std::string user_content = !server_json::get_string_or_empty(body, "prompt_text").empty()
        ? server_json::get_string_or_empty(body, "prompt_text")
        : !server_json::get_string_or_empty(body, "prompt").empty()
            ? server_json::get_string_or_empty(body, "prompt")
            : !server_json::get_string_or_empty(body, "question").empty()
                ? server_json::get_string_or_empty(body, "question")
                : !server_json::get_string_or_empty(body, "user_text").empty()
                    ? server_json::get_string_or_empty(body, "user_text")
                    : server_json::get_string_or_empty(body, "query");

    if (messages.empty() || !user_content.empty()) {
        messages.push_back({{"role", "user"}, {"content", user_content}});
    }

    return messages;
}

json build_ventriloquy_result(
        const json & response_json,
        const std::string & session_id,
        const std::string & turn_id,
        const std::string & write_mode) {
    const json structured = response_json.value("structured_conclusion", json::object());
    const json message = response_json["choices"][0]["message"];

    const std::string content = server_json::flatten_message_content(message.value("content", json()));
    const std::string reasoning_content = server_json::flatten_message_content(message.value("reasoning_content", json()));
    const std::string refusal = server_json::get_string_or_empty(message, "refusal");
    const std::string choice_text = response_json["choices"][0].value("text", std::string());

    const json parsed_content = server_json::parse_direct_payload(content);

    const std::string structured_summary = server_json::get_string_or_empty(structured, "summary");
    const std::string structured_next_action = server_json::get_string_or_empty(structured, "next_action");

    const bool insufficient_context =
        structured.is_object() && structured.value("insufficient_context", false);

    json evidence = json::array();

    if (parsed_content.is_object() &&
        parsed_content.contains("evidence") &&
        parsed_content["evidence"].is_array()) {
        evidence = parsed_content["evidence"];
    } else if (structured.contains("evidence_refs") && structured["evidence_refs"].is_array()) {
        evidence = structured["evidence_refs"];
    }

    std::string direct_answer = parsed_content.value("direct_answer", "");

    if (direct_answer.empty()) {
        direct_answer = normalize_direct_answer_candidate(content);
    }
    if (direct_answer.empty()) {
        direct_answer = normalize_direct_answer_candidate(choice_text);
    }
    if (direct_answer.empty()) {
        direct_answer = refusal;
    }
    if (direct_answer.empty()) {
        direct_answer = structured_summary;
    }

    if (direct_answer.empty() && evidence.is_array() && !evidence.empty() && evidence[0].is_string()) {
        direct_answer = evidence[0].get<std::string>();
    }

    if (direct_answer.empty()) {
        direct_answer = insufficient_context
            ? "Insufficient context to confirm; provide relevant status, logs, or tool results."
            : "No usable final answer was generated; provide more evidence and retry.";
    }

    std::string next_action = parsed_content.value("next_action", "");

    if (next_action.empty()) {
        next_action = structured_next_action;
    }

    if (next_action.empty()) {
        next_action = insufficient_context ? "collect_more_evidence" : "continue_session";
    }

    std::string confidence = parsed_content.value("confidence", std::string("unclear"));

    if (confidence.empty() || confidence == "unclear") {
        if (insufficient_context) {
            confidence = "unclear";
        } else if (!direct_answer.empty()) {
            confidence = "likely";
        }
    }

    if (confidence != "confirmed" &&
        confidence != "likely" &&
        confidence != "unclear" &&
        confidence != "blocked") {
        confidence = "unclear";
    }

    return json{
        {"session_id", session_id},
        {"turn_id", turn_id},
        {"write_mode", write_mode},
        {"direct_answer", direct_answer},
        {"evidence", evidence},
        {"next_action", next_action},
        {"confidence", confidence},
        {"ventriloquy_debug_version", "ventriloquy-fallback-v3"},
        {"raw_response", response_json}
    };
}

} // namespace server_remote_session_turn