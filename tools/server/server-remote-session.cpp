#include "server-remote-session.h"

#include "common.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>

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
    fs::create_directories(root_dir);
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
    return turn;
}
