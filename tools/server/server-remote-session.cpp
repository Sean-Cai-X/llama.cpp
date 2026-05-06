#include "server-remote-session.h"

#include "common.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace {
static int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static std::string get_string(const json & value, const std::string & key) {
    if (!value.is_object() || !value.contains(key) || value.at(key).is_null()) {
        return "";
    }
    if (value.at(key).is_string()) {
        return value.at(key).get<std::string>();
    }
    return value.at(key).dump();
}

static std::string trim_copy(std::string text) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

static bool looks_dirty_text(const std::string & text) {
    const std::string trimmed = trim_copy(text);
    return trimmed.empty() || trimmed == "{" || trimmed == "}";
}

static std::string extract_preferred_answer_text(const std::string & assistant_text, const std::string & direct_answer) {
    const std::string direct_trimmed = trim_copy(direct_answer);
    if (!direct_trimmed.empty()) {
        return direct_trimmed;
    }
    const std::string assistant_trimmed = trim_copy(assistant_text);
    if (looks_dirty_text(assistant_trimmed)) {
        return "";
    }
    if (!assistant_trimmed.empty() && assistant_trimmed.front() == '{') {
        try {
            const json parsed = json::parse(assistant_trimmed);
            if (parsed.is_object() && parsed.contains("direct_answer") && parsed["direct_answer"].is_string()) {
                return trim_copy(parsed["direct_answer"].get<std::string>());
            }
            if (parsed.is_object() && parsed.contains("summary") && parsed["summary"].is_string()) {
                return trim_copy(parsed["summary"].get<std::string>());
            }
        } catch (...) {
        }
    }
    return assistant_trimmed;
}

static std::string fnv1a64_hex(const std::string & text) {
    std::uint64_t hash = 14695981039346656037ull;
    for (unsigned char ch : text) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::nouppercase << hash;
    return out.str();
}

static json get_array(const json & value, const std::string & key) {
    if (!value.is_object() || !value.contains(key) || !value.at(key).is_array()) {
        return json::array();
    }
    return value.at(key);
}

static std::string random_string(size_t len = 8) {
    static constexpr char alphabet[] =
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz";

    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, sizeof(alphabet) - 2);

    std::string out;
    out.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(alphabet[dist(rng)]);
    }
    return out;
}
}

remote_session_store::remote_session_store() {
    root_dir = (fs::current_path() / "remote_sessions").string();
    slices_dir = (fs::current_path() / "remote_session_slices").string();
    fs::create_directories(root_dir);
    fs::create_directories(slices_dir);
}

std::string remote_session_store::create_session_id() const {
    return random_string();
}

std::string remote_session_store::create_turn_id() const {
    return "turn-" + std::to_string(now_ms()) + "-" + random_string();
}

std::string remote_session_store::session_path(const std::string & session_id) const {
    return (fs::path(root_dir) / (session_id + ".json")).string();
}

std::string remote_session_store::slice_path(const std::string & session_id, const std::string & turn_id) const {
    return (fs::path(slices_dir) / session_id / (turn_id + ".json")).string();
}

std::string remote_session_store::canonical_index_path() const {
    return (fs::path(slices_dir) / "canonical_index.json").string();
}

json remote_session_store::build_empty_session(const std::string & session_id) const {
    const int64_t ts = now_ms();
    return json{
        {"record_model", "remote_session_v1"},
        {"session_id", session_id},
        {"title", ""},
        {"created_at", ts},
        {"updated_at", ts},
        {"last_turn_id", ""},
        {"source_type", ""},
        {"task_group_id", ""},
        {"handoff_from", ""},
        {"handoff_to", ""},
        {"takeover_relation", ""},
        {"turns", json::array()}
    };
}

json remote_session_store::load_session_unlocked(const std::string & session_id) const {
    const std::string path = session_path(session_id);
    if (!fs::exists(path)) {
        return build_empty_session(session_id);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return build_empty_session(session_id);
    }

    try {
        json data = json::parse(in);
        if (!data.is_object()) {
            return build_empty_session(session_id);
        }
        if (!data.contains("turns") || !data["turns"].is_array()) {
            data["turns"] = json::array();
        }
        return data;
    } catch (...) {
        return build_empty_session(session_id);
    }
}

void remote_session_store::save_session_unlocked(const std::string & session_id, const json & session) const {
    const std::string path = session_path(session_id);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << session.dump(2);
}

json remote_session_store::load_canonical_index_unlocked() const {
    const std::string path = canonical_index_path();
    if (!fs::exists(path)) {
        return json::object();
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return json::object();
    }
    try {
        const json parsed = json::parse(in);
        return parsed.is_object() ? parsed : json::object();
    } catch (...) {
        return json::object();
    }
}

void remote_session_store::save_canonical_index_unlocked(const json & index) const {
    std::ofstream out(canonical_index_path(), std::ios::binary | std::ios::trunc);
    out << index.dump(2);
}

json remote_session_store::build_slice_unlocked(const std::string & session_id, const json & turn) const {
    const std::string turn_id = get_string(turn, "turn_id");
    const std::string source_type = get_string(turn, "source_type");
    const std::string user_text = trim_copy(get_string(turn, "user_text"));
    const std::string assistant_text = get_string(turn, "assistant_text");
    const std::string direct_answer = get_string(turn, "direct_answer");
    const std::string preferred_assistant_text = extract_preferred_answer_text(assistant_text, direct_answer);
    const std::string preferred_summary = !preferred_assistant_text.empty()
        ? preferred_assistant_text
        : trim_copy(get_string(turn, "summary"));
    const std::string write_mode = get_string(turn, "write_mode");
    const std::string strategy_family = get_string(turn, "prompt_purpose");
    const bool dirty_slice = looks_dirty_text(user_text) || looks_dirty_text(preferred_summary);
    const std::string error_signature =
        get_string(turn, "confidence") == "unclear" ? preferred_summary : "";
    const std::string solution_summary =
        (get_string(turn, "confidence") == "confirmed" || get_string(turn, "confidence") == "likely")
            ? preferred_summary
            : get_string(turn, "next_action");
    const std::string dedup_source =
        user_text + "\n" + preferred_summary + "\n" + strategy_family + "\n" + source_type + "\n" + write_mode;
    const std::string dedup_key = "dedup:" + fnv1a64_hex(dedup_source);
    const std::string slice_type = source_type == "webui" ? "manual_webui" : "remote_session";
    const std::string slice_id = "slice:" + session_id + ":" + turn_id;

    return json{
        {"record_model", "rag_memory_slice_v1"},
        {"slice_id", slice_id},
        {"slice_type", slice_type},
        {"created_at", turn.value("timestamp", now_ms())},
        {"provider_id", "llama_cpp_b8851_remote_session"},
        {"capability_id", "remote_session_turn"},
        {"source_provider", "llama.cpp-b8851"},
        {"task_id", get_string(turn, "task_id")},
        {"session_id", session_id},
        {"turn_id", turn_id},
        {"task_group_id", get_string(turn, "task_group_id")},
        {"strategy_key", get_string(turn, "prompt_purpose")},
        {"strategy_family", strategy_family},
        {"user_text", user_text},
        {"assistant_text", preferred_assistant_text},
        {"slice_summary", preferred_summary},
        {"error_signature", error_signature},
        {"solution_summary", solution_summary},
        {"expression_keys", json::array()},
        {"reasoning_level", get_string(turn, "reasoning_level")},
        {"primary_intent", ""},
        {"secondary_intents", json::array()},
        {"confidence", get_string(turn, "confidence")},
        {"similarity_score", 0.0},
        {"result_ref", get_string(turn, "result_ref")},
        {"evidence_ref", get_string(turn, "evidence_ref")},
        {"slice_refs", json::array()},
        {"storage_refs", json::array({"remote_session_store:" + session_id, "remote_session_slice:" + slice_id})},
        {"source_type", source_type},
        {"write_mode", write_mode},
        {"vector_payload", dirty_slice ? "" : (user_text + "\n" + preferred_summary)},
        {"vector_ready", !dirty_slice},
        {"vector_skip_reason", dirty_slice ? "dirty_content_only_brace" : ""},
        {"canonical_slice_id", ""},
        {"canonical_status", "pending"},
        {"metadata_json", turn},
        {"dedup_hash", dedup_key}
    };
}

void remote_session_store::save_slice_unlocked(const std::string & session_id, const std::string & turn_id, const json & slice) const {
    const fs::path path = slice_path(session_id, turn_id);
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << slice.dump(2);
}

std::optional<json> remote_session_store::get_session(const std::string & session_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    const std::string path = session_path(session_id);
    if (!fs::exists(path)) {
        return std::nullopt;
    }
    return load_session_unlocked(session_id);
}

json remote_session_store::list_sessions(int limit) const {
    std::lock_guard<std::mutex> lock(mutex);

    struct session_stub {
        int64_t updated_at = 0;
        json value;
    };

    std::vector<session_stub> sessions;
    if (!fs::exists(root_dir)) {
        return json::array();
    }

    for (const auto & entry : fs::directory_iterator(root_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }

        std::ifstream in(entry.path(), std::ios::binary);
        if (!in.good()) {
            continue;
        }

        try {
            json session = json::parse(in);
            json turns = get_array(session, "turns");
            const json last_turn = turns.empty() ? json::object() : turns.back();
            sessions.push_back({
                session.value("updated_at", 0LL),
                json{
                    {"record_model", session.value("record_model", "remote_session_v1")},
                    {"session_id", session.value("session_id", "")},
                    {"title", session.value("title", "")},
                    {"updated_at", session.value("updated_at", 0LL)},
                    {"last_turn_id", session.value("last_turn_id", "")},
                    {"source_type", session.value("source_type", "")},
                    {"task_group_id", session.value("task_group_id", "")},
                    {"handoff_from", session.value("handoff_from", "")},
                    {"handoff_to", session.value("handoff_to", "")},
                    {"takeover_relation", session.value("takeover_relation", "")},
                    {"turn_count", turns.size()},
                    {"current_summary", get_string(last_turn, "summary")},
                    {"last_task_id", get_string(last_turn, "task_id")},
                    {"last_result_ref", get_string(last_turn, "result_ref")},
                    {"last_evidence_ref", get_string(last_turn, "evidence_ref")}
                }
            });
        } catch (...) {
            continue;
        }
    }

    std::sort(sessions.begin(), sessions.end(), [](const session_stub & a, const session_stub & b) {
        return a.updated_at > b.updated_at;
    });

    json result = json::array();
    const size_t n = limit > 0 ? std::min<size_t>((size_t) limit, sessions.size()) : sessions.size();
    for (size_t i = 0; i < n; ++i) {
        result.push_back(std::move(sessions[i].value));
    }
    return result;
}

json remote_session_store::upsert_turn(
    const std::string & session_id,
    const json & metadata,
    const json & request_payload,
    const json & response_payload) {
    std::lock_guard<std::mutex> lock(mutex);

    json session = load_session_unlocked(session_id);
    const int64_t ts = now_ms();
    const std::string turn_id = get_string(metadata, "turn_id");
    const std::string user_text = get_string(metadata, "user_text");
    const std::string assistant_text = get_string(metadata, "assistant_text");
    const std::string summary = get_string(metadata, "summary");
    const std::string task_id = get_string(metadata, "task_id");
    const std::string task_group_id = get_string(metadata, "task_group_id");
    const std::string source_type = get_string(metadata, "source_type");
    const std::string handoff_from = get_string(metadata, "handoff_from");
    const std::string handoff_to = get_string(metadata, "handoff_to");
    const std::string takeover_relation = get_string(metadata, "takeover_relation");

    json turn = {
        {"turn_id", turn_id},
        {"timestamp", ts},
        {"write_mode", get_string(metadata, "write_mode")},
        {"codex_request_id", get_string(metadata, "codex_request_id")},
        {"agent_dispatch_id", get_string(metadata, "agent_dispatch_id")},
        {"task_id", task_id},
        {"task_group_id", task_group_id},
        {"source_type", source_type},
        {"source_label", get_string(metadata, "source_label")},
        {"source_detail", get_string(metadata, "source_detail")},
        {"handoff_from", handoff_from},
        {"handoff_to", handoff_to},
        {"takeover_relation", takeover_relation},
        {"speaker_mode", get_string(metadata, "speaker_mode")},
        {"reasoning_level", get_string(metadata, "reasoning_level")},
        {"prompt_purpose", get_string(metadata, "prompt_purpose")},
        {"response_mode", get_string(metadata, "response_mode")},
        {"context_refs", get_array(metadata, "context_refs")},
        {"user_text", user_text},
        {"assistant_text", assistant_text},
        {"summary", summary},
        {"direct_answer", get_string(metadata, "direct_answer")},
        {"next_action", get_string(metadata, "next_action")},
        {"confidence", get_string(metadata, "confidence")},
        {"result_ref", get_string(metadata, "result_ref")},
        {"evidence_ref", get_string(metadata, "evidence_ref")},
        {"timings", metadata.contains("timings") ? metadata["timings"] : json::object()},
        {"request_payload", request_payload},
        {"response_payload", response_payload}
    };

    json slice = build_slice_unlocked(session_id, turn);
    const std::string slice_id = get_string(slice, "slice_id");
    const std::string dedup_key = get_string(slice, "dedup_hash");
    const std::string persisted_slice_path = slice_path(session_id, turn_id);
    json canonical_index = load_canonical_index_unlocked();
    std::string canonical_slice_id = get_string(canonical_index, dedup_key);
    const bool canonical_exists = !canonical_slice_id.empty();
    if (!canonical_exists) {
        canonical_slice_id = slice_id;
        canonical_index[dedup_key] = canonical_slice_id;
        save_canonical_index_unlocked(canonical_index);
    }
    slice["canonical_slice_id"] = canonical_slice_id;
    slice["canonical_status"] = canonical_exists ? "duplicate" : "canonical";
    if (canonical_exists) {
        slice["vector_ready"] = false;
        if (get_string(slice, "vector_skip_reason").empty()) {
            slice["vector_skip_reason"] = "duplicate_of_canonical";
        }
    }
    turn["slice_id"] = slice_id;
    turn["slice_refs"] = json::array({slice_id});
    turn["storage_refs"] = json::array({"remote_session_store:" + session_id, "remote_session_slice:" + slice_id});
    turn["slice_path"] = persisted_slice_path;
    turn["dedup_key"] = dedup_key;
    turn["dedup_hash"] = dedup_key;
    turn["canonical_slice_id"] = canonical_slice_id;
    turn["canonical_status"] = get_string(slice, "canonical_status");
    turn["provider_id"] = get_string(slice, "provider_id");
    turn["capability_id"] = get_string(slice, "capability_id");
    turn["error_signature"] = get_string(slice, "error_signature");
    turn["solution_summary"] = get_string(slice, "solution_summary");
    turn["strategy_family"] = get_string(slice, "strategy_family");
    turn["similarity_score"] = slice.contains("similarity_score") ? slice["similarity_score"] : 0.0;
    turn["vector_ready"] = slice.value("vector_ready", false);
    turn["vector_skip_reason"] = get_string(slice, "vector_skip_reason");

    session["updated_at"] = ts;
    session["last_turn_id"] = turn_id;
    if (!source_type.empty()) session["source_type"] = source_type;
    if (!task_group_id.empty()) session["task_group_id"] = task_group_id;
    if (!handoff_from.empty()) session["handoff_from"] = handoff_from;
    if (!handoff_to.empty()) session["handoff_to"] = handoff_to;
    if (!takeover_relation.empty()) session["takeover_relation"] = takeover_relation;
    if (!session.contains("turns") || !session["turns"].is_array()) {
        session["turns"] = json::array();
    }
    session["turns"].push_back(turn);

    if (session.value("title", std::string()).empty()) {
        const std::string title_source = !user_text.empty() ? user_text : assistant_text;
        session["title"] = title_source.size() > 72 ? title_source.substr(0, 72) : title_source;
    }

    save_session_unlocked(session_id, session);
    save_slice_unlocked(session_id, turn_id, slice);
    return turn;
}
