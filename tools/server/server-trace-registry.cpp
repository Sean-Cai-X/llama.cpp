#include "server-trace-registry.h"

#include "common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct trace_record {
    json payload = json::object();
};

std::mutex g_trace_registry_mutex;
std::unordered_map<std::string, trace_record> g_trace_registry;
std::list<std::string> g_trace_order;
constexpr size_t TRACE_REGISTRY_LIMIT = 128;
std::string g_trace_storage_base_path;
std::atomic<uint64_t> g_trace_event_counter {0};

std::string sanitize_trace_token(const std::string & value) {
    std::string sanitized;
    sanitized.reserve(value.size());
    for (char ch : value) {
        if ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '.' || ch == '-' || ch == '_') {
            sanitized.push_back(ch);
        } else {
            sanitized.push_back('_');
        }
    }
    return sanitized;
}

std::string audit_log_path() {
    if (g_trace_storage_base_path.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".audit.jsonl";
}

std::string trace_snapshot_path(const std::string & trace_id) {
    if (g_trace_storage_base_path.empty() || trace_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".trace." + sanitize_trace_token(trace_id) + ".json";
}

std::string request_index_path(const std::string & request_id) {
    if (g_trace_storage_base_path.empty() || request_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".request." + sanitize_trace_token(request_id) + ".jsonl";
}

std::string audit_request_index_path(const std::string & request_id) {
    if (g_trace_storage_base_path.empty() || request_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".audit_by_request." + sanitize_trace_token(request_id) + ".jsonl";
}

std::string goal_index_path(const std::string & goal_id) {
    if (g_trace_storage_base_path.empty() || goal_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".goal." + sanitize_trace_token(goal_id) + ".jsonl";
}

std::string slice_index_path(const std::string & slice_id) {
    if (g_trace_storage_base_path.empty() || slice_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".slice." + sanitize_trace_token(slice_id) + ".jsonl";
}

std::string audit_slice_index_path(const std::string & slice_id) {
    if (g_trace_storage_base_path.empty() || slice_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".audit_by_slice." + sanitize_trace_token(slice_id) + ".jsonl";
}

std::string audit_node_index_path(const std::string & node_id) {
    if (g_trace_storage_base_path.empty() || node_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".audit_by_node." + sanitize_trace_token(node_id) + ".jsonl";
}

void append_jsonl_record(
        const std::string & path,
        const json & record) {
    if (path.empty()) {
        return;
    }

    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::app);
        if (!out.is_open()) {
            return;
        }
        out << record.dump() << '\n';
    } catch (...) {
    }
}

json read_json_file_or_object(const std::string & path) {
    if (path.empty()) {
        return json::object();
    }
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return json::object();
        }
        json value;
        in >> value;
        return value.is_object() ? value : json::object();
    } catch (...) {
        return json::object();
    }
}

void write_json_file(const std::string & path, const json & value) {
    if (path.empty()) {
        return;
    }
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (out.is_open()) {
            out << value.dump();
        }
    } catch (...) {
    }
}

std::string viewpoint_store_path() {
    return g_trace_storage_base_path.empty() ? "" : g_trace_storage_base_path + ".viewpoints.json";
}

std::string viewpoint_index_path(const std::string & viewpoint_id) {
    if (g_trace_storage_base_path.empty() || viewpoint_id.empty()) {
        return "";
    }
    return g_trace_storage_base_path + ".viewpoint." + sanitize_trace_token(viewpoint_id) + ".jsonl";
}

std::string kv_journal_path() {
    return g_trace_storage_base_path.empty() ? "" : g_trace_storage_base_path + ".kv_journal.jsonl";
}

std::string kv_snapshot_path() {
    return g_trace_storage_base_path.empty() ? "" : g_trace_storage_base_path + ".kv_snapshot.json";
}

json make_backend_put(
        const std::string & column_family,
        const std::string & key,
        const json & value,
        const std::string & record_model) {
    return json{
        {"op", "put"},
        {"column_family", column_family},
        {"key", key},
        {"record_model", record_model},
        {"value", value},
    };
}

void append_backend_write_batch(
        const std::string & batch_id,
        const json & operations) {
    if (g_trace_storage_base_path.empty() || batch_id.empty() ||
        !operations.is_array() || operations.empty()) {
        return;
    }

    const json batch = {
        {"schema_version", 1},
        {"record_model", "rag_storage_backend_write_batch_v1"},
        {"index_name", g_trace_storage_base_path},
        {"batch_id", batch_id},
        {"active_backend", "file_sidecar"},
        {"target_backends", json::array({"file_sidecar", "rocksdb", "sqlite"})},
        {"operation_count", static_cast<int>(operations.size())},
        {"operations", operations},
    };
    append_jsonl_record(kv_journal_path(), batch);

    json snapshot = read_json_file_or_object(kv_snapshot_path());
    if (snapshot.empty()) {
        snapshot = {
            {"schema_version", 1},
            {"record_model", "rag_storage_backend_kv_snapshot_v1"},
            {"index_name", g_trace_storage_base_path},
            {"active_backend", "file_sidecar"},
            {"column_families", json::object()},
            {"records", json::object()},
        };
    }
    if (!snapshot.contains("column_families") || !snapshot["column_families"].is_object()) {
        snapshot["column_families"] = json::object();
    }
    if (!snapshot.contains("records") || !snapshot["records"].is_object()) {
        snapshot["records"] = json::object();
    }

    for (const auto & op : operations) {
        if (!op.is_object() || op.value("op", "") != "put") {
            continue;
        }
        const std::string column_family = op.value("column_family", "");
        const std::string key = op.value("key", "");
        if (column_family.empty() || key.empty()) {
            continue;
        }
        if (!snapshot["column_families"].contains(column_family) ||
            !snapshot["column_families"][column_family].is_object()) {
            snapshot["column_families"][column_family] = json::object();
        }
        snapshot["column_families"][column_family][key] = op.value("value", json());
        snapshot["records"][column_family + ":" + key] = op;
    }
    snapshot["last_batch_id"] = batch_id;
    snapshot["record_count"] = snapshot["records"].size();
    write_json_file(kv_snapshot_path(), snapshot);
}

std::unordered_set<std::string> collect_slice_ids(const json & payload) {
    std::unordered_set<std::string> slice_ids;

    const json approved_context = payload.value("approved_context", json::array());
    if (approved_context.is_array()) {
        for (const auto & item : approved_context) {
            if (!item.is_object()) {
                continue;
            }
            const std::string slice_id = item.value("slice_id", item.value("source_id", ""));
            if (!slice_id.empty()) {
                slice_ids.insert(slice_id);
            }
        }
    }

    const json supporting_slice_ids = payload.value("supporting_slice_ids", json::array());
    if (supporting_slice_ids.is_array()) {
        for (const auto & item : supporting_slice_ids) {
            if (item.is_string()) {
                const std::string slice_id = item.get<std::string>();
                if (!slice_id.empty()) {
                    slice_ids.insert(slice_id);
                }
            }
        }
    }

    return slice_ids;
}

void collect_node_ids_from_value(const json & value, std::unordered_set<std::string> & node_ids) {
    if (value.is_object()) {
        for (const auto & key : {"node_id", "bind_node_id", "from_node_id", "to_node_id"}) {
            if (value.contains(key) && value.at(key).is_string()) {
                const std::string node_id = value.at(key).get<std::string>();
                if (!node_id.empty()) {
                    node_ids.insert(node_id);
                }
            }
        }
        for (const auto & item : value.items()) {
            collect_node_ids_from_value(item.value(), node_ids);
        }
        return;
    }
    if (value.is_array()) {
        for (const auto & item : value) {
            collect_node_ids_from_value(item, node_ids);
        }
    }
}

std::unordered_set<std::string> collect_node_ids(const json & payload) {
    std::unordered_set<std::string> node_ids;
    collect_node_ids_from_value(payload, node_ids);
    return node_ids;
}

std::string collect_goal_id(const json & payload) {
    if (!payload.is_object()) {
        return "";
    }
    if (payload.contains("goal_id") && payload.at("goal_id").is_string()) {
        return payload.at("goal_id").get<std::string>();
    }
    if (payload.contains("goal") && payload.at("goal").is_object()) {
        const json & goal = payload.at("goal");
        if (goal.contains("goal_id") && goal.at("goal_id").is_string()) {
            return goal.at("goal_id").get<std::string>();
        }
    }
    return "";
}

uint64_t persist_audit_event_unlocked(
        const std::string & trace_id,
        const std::string & stage,
        const json & payload) {
    const uint64_t seq = ++g_trace_event_counter;

    try {
        json event = {
            {"record_model", "rag_audit_event_v1"},
            {"seq", seq},
            {"trace_id", trace_id},
            {"stage", stage},
            {"request_id", payload.value("request_id", "")},
            {"query_id", payload.value("query_id", "")},
            {"payload", payload},
        };
        append_jsonl_record(audit_log_path(), event);

        const std::string request_id = payload.value("request_id", "");
        if (!request_id.empty()) {
            const json request_ref = {
                {"record_model", "rag_audit_request_ref_v1"},
                {"seq", seq},
                {"request_id", request_id},
                {"trace_id", trace_id},
                {"query_id", payload.value("query_id", "")},
                {"stage", stage},
            };
            append_jsonl_record(request_index_path(request_id), request_ref);
            append_jsonl_record(audit_request_index_path(request_id), request_ref);
        }

        const std::string goal_id = collect_goal_id(payload);
        if (!goal_id.empty()) {
            append_jsonl_record(goal_index_path(goal_id), json{
                {"record_model", "rag_audit_goal_ref_v1"},
                {"seq", seq},
                {"goal_id", goal_id},
                {"trace_id", trace_id},
                {"request_id", request_id},
                {"query_id", payload.value("query_id", "")},
                {"stage", stage},
            });
        }

        for (const auto & slice_id : collect_slice_ids(payload)) {
            const json slice_ref = {
                {"record_model", "rag_audit_slice_ref_v1"},
                {"seq", seq},
                {"slice_id", slice_id},
                {"trace_id", trace_id},
                {"request_id", request_id},
                {"query_id", payload.value("query_id", "")},
                {"stage", stage},
            };
            append_jsonl_record(slice_index_path(slice_id), slice_ref);
            append_jsonl_record(audit_slice_index_path(slice_id), slice_ref);
        }

        for (const auto & node_id : collect_node_ids(payload)) {
            append_jsonl_record(audit_node_index_path(node_id), json{
                {"record_model", "rag_audit_node_ref_v1"},
                {"seq", seq},
                {"node_id", node_id},
                {"trace_id", trace_id},
                {"request_id", request_id},
                {"query_id", payload.value("query_id", "")},
                {"stage", stage},
            });
        }
    } catch (...) {
    }

    return seq;
}

void persist_trace_snapshot_unlocked(
        const std::string & trace_id,
        const json & payload) {
    const std::string path = trace_snapshot_path(trace_id);
    if (path.empty()) {
        return;
    }

    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return;
        }
        out << payload.dump();
    } catch (...) {
    }
}

void persist_live_viewpoint_unlocked(
        const std::string & trace_id,
        const std::string & stage,
        const json & payload) {
    if (g_trace_storage_base_path.empty() || stage != "completion" || !payload.is_object()) {
        return;
    }

    const json approved_context = payload.value("approved_context", json::array());
    if (!approved_context.is_array() || approved_context.empty()) {
        return;
    }

    const std::string viewpoint_id = "VP-LIVE-" + sanitize_trace_token(trace_id);
    const std::string request_id = payload.value("request_id", "");
    const std::string query_id = payload.value("query_id", "");
    const json structured = payload.value("structured_conclusion", json::object());
    const json validation = payload.value("llama_output_validation", json::object());
    const json final_output = payload.value("final_output", json::object());
    const std::string content_hash = final_output.value("content_hash", "");

    std::string claim_text = structured.value("summary", "");
    if (claim_text.empty()) {
        claim_text = structured.value("slice_summary", "");
    }
    if (claim_text.empty()) {
        claim_text = content_hash;
    }

    std::string source_slice_id;
    if (payload.contains("supporting_slice_ids") && payload["supporting_slice_ids"].is_array() &&
        !payload["supporting_slice_ids"].empty() && payload["supporting_slice_ids"].front().is_string()) {
        source_slice_id = payload["supporting_slice_ids"].front().get<std::string>();
    }
    if (source_slice_id.empty() && approved_context.front().is_object()) {
        source_slice_id = approved_context.front().value("slice_id", approved_context.front().value("source_id", ""));
    }

    json store = read_json_file_or_object(viewpoint_store_path());
    if (!store.is_object()) {
        store = json::object();
    }
    store["schema_version"] = 1;
    store["record_model"] = "rag_viewpoint_store_v1";
    store["index_name"] = g_trace_storage_base_path;
    store["status"] = "live_chat_updated";
    for (const auto & key : {"viewpoint_candidates", "evidence_bindings", "viewpoint_decisions", "viewpoint_validations", "live_runs"}) {
        if (!store.contains(key) || !store[key].is_object()) {
            store[key] = json::object();
        }
    }

    const json candidate = {
        {"record_model", "rag_viewpoint_candidate_v1"},
        {"source", "live_chat_completion"},
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"query_id", query_id},
        {"viewpoint_id", viewpoint_id},
        {"model_task_id", payload.value("model_task_id", "")},
        {"source_slice_id", source_slice_id},
        {"claim_text", claim_text},
        {"claim_type", "MODEL_OUTPUT_SUMMARY"},
        {"subject", source_slice_id},
        {"predicate", "SUMMARIZES_APPROVED_CONTEXT"},
        {"object", content_hash},
        {"confidence", validation.value("decision", "") == "PASS" ? 1.0 : 0.5},
        {"status", "CANDIDATE"},
    };
    store["viewpoint_candidates"][viewpoint_id] = candidate;
    append_jsonl_record(viewpoint_index_path(viewpoint_id), candidate);

    json operations = json::array({
        make_backend_put("viewpoint", "viewpoint:" + viewpoint_id, candidate, "rag_viewpoint_candidate_v1"),
        make_backend_put("viewpoint", "viewpoint_by_trace:" + trace_id + ":" + viewpoint_id, candidate, "rag_viewpoint_candidate_v1"),
    });
    if (!source_slice_id.empty()) {
        operations.push_back(make_backend_put(
            "viewpoint",
            "viewpoint_by_slice:" + source_slice_id + ":" + viewpoint_id,
            candidate,
            "rag_viewpoint_candidate_v1"));
    }

    store["evidence_bindings"][viewpoint_id] = json::array();
    for (const auto & item : approved_context) {
        if (!item.is_object()) {
            continue;
        }
        const std::string slice_id = item.value("slice_id", item.value("source_id", ""));
        if (slice_id.empty()) {
            continue;
        }
        const json evidence = {
            {"record_model", "rag_viewpoint_evidence_binding_v1"},
            {"source", "live_chat_completion"},
            {"request_id", request_id},
            {"trace_id", trace_id},
            {"query_id", query_id},
            {"viewpoint_id", viewpoint_id},
            {"source_type", item.value("source_type", "SLICE")},
            {"source_id", slice_id},
            {"evidence_ref", item.value("source", "")},
            {"evidence_hash", ""},
            {"match_status", "MATCHED"},
            {"status", item.value("approved", false) ? "VERIFIED" : "CANDIDATE"},
        };
        store["evidence_bindings"][viewpoint_id].push_back(evidence);
        append_jsonl_record(viewpoint_index_path(viewpoint_id), evidence);
        operations.push_back(make_backend_put(
            "viewpoint",
            "viewpoint:evidence:" + viewpoint_id + ":" + slice_id,
            evidence,
            "rag_viewpoint_evidence_binding_v1"));
    }

    const json decision = {
        {"record_model", "rag_viewpoint_decision_v1"},
        {"source", "live_chat_completion"},
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"query_id", query_id},
        {"viewpoint_id", viewpoint_id},
        {"decision", validation.value("decision", "") == "PASS" ? "APPROVE" : validation.value("decision", "REVIEW")},
        {"reason", validation.value("reason", "")},
        {"final_confidence", validation.value("decision", "") == "PASS" ? 1.0 : 0.5},
        {"action", validation.value("status", "") == "APPROVED" ? "admit_live_chat_viewpoint" : "require_review"},
        {"status", validation.value("status", "")},
    };
    store["viewpoint_decisions"][viewpoint_id] = decision;
    append_jsonl_record(viewpoint_index_path(viewpoint_id), decision);
    operations.push_back(make_backend_put(
        "viewpoint",
        "viewpoint:decision:" + viewpoint_id,
        decision,
        "rag_viewpoint_decision_v1"));

    store["viewpoint_count"] = store["viewpoint_candidates"].size();
    store["live_runs"][trace_id] = {
        {"record_model", "rag_live_viewpoint_run_ref_v1"},
        {"request_id", request_id},
        {"trace_id", trace_id},
        {"query_id", query_id},
        {"viewpoint_id", viewpoint_id},
        {"content_hash", content_hash},
        {"status", decision.value("status", "")},
    };
    write_json_file(viewpoint_store_path(), store);
    append_backend_write_batch("live_viewpoint_" + sanitize_trace_token(trace_id), operations);
}

void append_event_unlocked(
        json & root,
        const std::string & trace_id,
        const std::string & stage,
        const json & payload) {
    if (!root.contains("events") || !root["events"].is_array()) {
        root["events"] = json::array();
    }

    const uint64_t seq = persist_audit_event_unlocked(trace_id, stage, payload);
    root["events"].push_back(json{
        {"record_model", "rag_trace_event_v1"},
        {"seq", seq},
        {"trace_id", trace_id},
        {"stage", stage},
        {"payload", payload},
    });
    root["event_count"] = root["events"].size();
}

} // namespace

namespace server_trace_registry {

void configure_storage(
        const std::string & base_path) {
    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
    g_trace_storage_base_path = base_path;
}

void record_stage(
        const std::string & trace_id,
        const std::string & stage,
        const json & payload) {
    if (trace_id.empty() || stage.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);

    auto it = g_trace_registry.find(trace_id);
    if (it == g_trace_registry.end()) {
        g_trace_order.push_back(trace_id);
        it = g_trace_registry.emplace(trace_id, trace_record {}).first;
        it->second.payload["trace_id"] = trace_id;
        it->second.payload["record_model"] = "rag_trace_debug_v1";
        it->second.payload["stages"] = json::object();
        it->second.payload["events"] = json::array();
        it->second.payload["event_count"] = 0;
    } else {
        g_trace_order.remove(trace_id);
        g_trace_order.push_back(trace_id);
    }

    json & root = it->second.payload;
    root["latest_stage"] = stage;
    root["stages"][stage] = payload;

    for (const auto & key : {"request_id", "query_id", "model_task_id"}) {
        if (payload.contains(key) && !payload.at(key).is_null()) {
            root[key] = payload.at(key);
        }
    }
    for (const auto & key : {"approved_context", "supporting_slice_ids", "admission_summary", "llama_output_candidate", "llama_output_validation", "final_output", "structured_conclusion"}) {
        if (payload.contains(key) && !payload.at(key).is_null()) {
            root[key] = payload.at(key);
        }
    }

    append_event_unlocked(root, trace_id, stage, payload);
    persist_live_viewpoint_unlocked(trace_id, stage, payload);
    persist_trace_snapshot_unlocked(trace_id, root);

    while (g_trace_order.size() > TRACE_REGISTRY_LIMIT) {
        const std::string evict_id = g_trace_order.front();
        g_trace_order.pop_front();
        g_trace_registry.erase(evict_id);
    }
}

void append_event(
        const std::string & trace_id,
        const std::string & stage,
        const json & payload) {
    if (trace_id.empty() || stage.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);

    auto it = g_trace_registry.find(trace_id);
    if (it == g_trace_registry.end()) {
        g_trace_order.push_back(trace_id);
        it = g_trace_registry.emplace(trace_id, trace_record {}).first;
        it->second.payload["trace_id"] = trace_id;
        it->second.payload["record_model"] = "rag_trace_debug_v1";
        it->second.payload["stages"] = json::object();
        it->second.payload["events"] = json::array();
        it->second.payload["event_count"] = 0;
    } else {
        g_trace_order.remove(trace_id);
        g_trace_order.push_back(trace_id);
    }

    json & root = it->second.payload;
    root["latest_stage"] = stage;
    append_event_unlocked(root, trace_id, stage, payload);
    persist_trace_snapshot_unlocked(trace_id, root);

    while (g_trace_order.size() > TRACE_REGISTRY_LIMIT) {
        const std::string evict_id = g_trace_order.front();
        g_trace_order.pop_front();
        g_trace_registry.erase(evict_id);
    }
}

bool get_trace(
        const std::string & trace_id,
        json & payload) {
    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
    const auto it = g_trace_registry.find(trace_id);
    if (it != g_trace_registry.end()) {
        payload = it->second.payload;
        return true;
    }

    const std::string path = trace_snapshot_path(trace_id);
    if (path.empty()) {
        return false;
    }

    try {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }
        in >> payload;
        return payload.is_object();
    } catch (...) {
        return false;
    }
}

bool get_request(
        const std::string & request_id,
        json & payload) {
    if (request_id.empty()) {
        return false;
    }

    json trace_refs = json::array();
    std::unordered_set<std::string> seen_trace_ids;

    auto collect_from_root = [&](const json & root) {
        if (!root.is_object()) {
            return;
        }
        if (root.value("request_id", "") != request_id) {
            return;
        }

        const std::string trace_id = root.value("trace_id", "");
        if (trace_id.empty() || !seen_trace_ids.insert(trace_id).second) {
            return;
        }

        trace_refs.push_back(json{
            {"trace_id", trace_id},
            {"request_id", root.value("request_id", "")},
            {"query_id", root.value("query_id", "")},
            {"latest_stage", root.value("latest_stage", "")},
        });
    };

    {
        std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
        for (auto it = g_trace_order.rbegin(); it != g_trace_order.rend(); ++it) {
            const auto record_it = g_trace_registry.find(*it);
            if (record_it == g_trace_registry.end()) {
                continue;
            }
            collect_from_root(record_it->second.payload);
        }
    }

    if (g_trace_storage_base_path.empty()) {
        if (trace_refs.empty()) {
            return false;
        }
    } else {
        try {
            const std::string index_path = request_index_path(request_id);
            std::ifstream index_in(index_path, std::ios::binary);
            if (index_in.is_open()) {
                std::string line;
                while (std::getline(index_in, line)) {
                    if (line.empty()) {
                        continue;
                    }

                    json ref = json::parse(line, nullptr, false);
                    if (!ref.is_object()) {
                        continue;
                    }

                    const std::string trace_id = ref.value("trace_id", "");
                    if (trace_id.empty() || !seen_trace_ids.insert(trace_id).second) {
                        continue;
                    }

                    const std::string path = trace_snapshot_path(trace_id);
                    if (path.empty()) {
                        continue;
                    }

                    std::ifstream in(path, std::ios::binary);
                    if (!in.is_open()) {
                        continue;
                    }

                    json root;
                    in >> root;
                    collect_from_root(root);
                }
            }
        } catch (...) {
        }
    }

    if (trace_refs.empty()) {
        return false;
    }

    payload = json{
        {"record_model", "rag_request_debug_v1"},
        {"request_id", request_id},
        {"trace_refs", trace_refs},
    };
    return true;
}

bool get_goal(
        const std::string & goal_id,
        json & payload) {
    if (goal_id.empty()) {
        return false;
    }

    json trace_refs = json::array();
    json latest_goal = json::object();
    json latest_acceptance = json::object();
    std::unordered_set<std::string> seen_trace_ids;

    auto collect_from_root = [&](const json & root) {
        if (!root.is_object()) {
            return;
        }

        const std::string root_goal_id = collect_goal_id(root);
        if (root_goal_id != goal_id) {
            return;
        }

        const std::string trace_id = root.value("trace_id", "");
        if (trace_id.empty()) {
            return;
        }

        if (seen_trace_ids.insert(trace_id).second) {
            trace_refs.push_back(json{
                {"trace_id", trace_id},
                {"goal_id", goal_id},
                {"request_id", root.value("request_id", "")},
                {"latest_stage", root.value("latest_stage", "")},
                {"event_count", root.value("event_count", 0)},
            });
        }

        const json stages = root.value("stages", json::object());
        if (stages.contains("goal_start") && stages.at("goal_start").is_object()) {
            latest_goal = stages.at("goal_start");
        }
        if (stages.contains("goal_final") && stages.at("goal_final").is_object()) {
            const json goal_final = stages.at("goal_final");
            if (latest_goal.empty()) {
                latest_goal = json{
                    {"goal_id", goal_final.value("goal_id", goal_id)},
                    {"goal_type", goal_final.value("goal_type", "")},
                    {"request_id", goal_final.value("request_id", "")},
                    {"trace_id", goal_final.value("trace_id", trace_id)},
                };
            }
            if (goal_final.contains("acceptance") && goal_final.at("acceptance").is_object()) {
                latest_acceptance = goal_final.at("acceptance");
            }
        }
        if (root.contains("events") && root.at("events").is_array()) {
            for (const auto & event : root.at("events")) {
                if (!event.is_object()) {
                    continue;
                }
                if (event.value("stage", "") != "acceptance_check") {
                    continue;
                }
                const json event_payload = event.value("payload", json::object());
                if (event_payload.contains("decision")) {
                    latest_acceptance = event_payload.at("decision");
                }
            }
        }
    };

    {
        std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
        for (auto it = g_trace_order.rbegin(); it != g_trace_order.rend(); ++it) {
            const auto record_it = g_trace_registry.find(*it);
            if (record_it == g_trace_registry.end()) {
                continue;
            }
            collect_from_root(record_it->second.payload);
        }
    }

    if (!g_trace_storage_base_path.empty()) {
        try {
            const std::string index_path = goal_index_path(goal_id);
            std::ifstream index_in(index_path, std::ios::binary);
            if (index_in.is_open()) {
                std::string line;
                while (std::getline(index_in, line)) {
                    if (line.empty()) {
                        continue;
                    }

                    json ref = json::parse(line, nullptr, false);
                    if (!ref.is_object()) {
                        continue;
                    }

                    const std::string trace_id = ref.value("trace_id", "");
                    if (trace_id.empty() || seen_trace_ids.count(trace_id) > 0) {
                        continue;
                    }

                    const std::string path = trace_snapshot_path(trace_id);
                    if (path.empty()) {
                        continue;
                    }

                    std::ifstream in(path, std::ios::binary);
                    if (!in.is_open()) {
                        continue;
                    }

                    json root;
                    in >> root;
                    collect_from_root(root);
                }
            }
        } catch (...) {
        }
    }

    if (trace_refs.empty()) {
        return false;
    }

    payload = json{
        {"record_model", "rag_goal_debug_v1"},
        {"goal_id", goal_id},
        {"goal", latest_goal},
        {"latest_acceptance", latest_acceptance},
        {"trace_refs", trace_refs},
    };
    return true;
}

bool get_goal_events(
        const std::string & goal_id,
        json & payload) {
    if (goal_id.empty()) {
        return false;
    }

    json goal_payload;
    if (!get_goal(goal_id, goal_payload)) {
        return false;
    }

    json events = json::array();
    std::unordered_set<uint64_t> seen_seq;

    const json trace_refs = goal_payload.value("trace_refs", json::array());
    for (const auto & trace_ref : trace_refs) {
        const std::string trace_id = trace_ref.value("trace_id", "");
        if (trace_id.empty()) {
            continue;
        }

        json trace_payload;
        if (!get_trace(trace_id, trace_payload)) {
            continue;
        }

        const json trace_events = trace_payload.value("events", json::array());
        for (const auto & event : trace_events) {
            if (!event.is_object()) {
                continue;
            }
            const uint64_t seq = event.value("seq", static_cast<uint64_t>(0));
            if (seq != 0 && !seen_seq.insert(seq).second) {
                continue;
            }
            if (collect_goal_id(event.value("payload", json::object())) != goal_id &&
                collect_goal_id(trace_payload) != goal_id) {
                continue;
            }
            events.push_back(event);
        }
    }

    payload = json{
        {"record_model", "rag_goal_events_v1"},
        {"goal_id", goal_id},
        {"event_count", events.size()},
        {"events", events},
    };
    return !events.empty();
}

bool get_evidence(
        const std::string & slice_id,
        json & payload) {
    if (slice_id.empty()) {
        return false;
    }

    json traces = json::array();
    json approved_context_item = json::object();
    json final_output = json::object();
    json output_validation = json::object();
    json admission_summary = json::object();

    std::unordered_set<std::string> trace_ref_ids;

    auto collect_from_root = [&](const json & root) {
        if (!root.is_object()) {
            return;
        }

        const std::string root_trace_id = root.value("trace_id", "");
        const json approved_context = root.value("approved_context", json::array());
        bool matched = false;

        if (approved_context.is_array()) {
            for (const auto & item : approved_context) {
                if (!item.is_object()) {
                    continue;
                }
                const std::string item_slice_id = item.value("slice_id", item.value("source_id", ""));
                if (item_slice_id != slice_id) {
                    continue;
                }

                if (approved_context_item.empty()) {
                    approved_context_item = item;
                    final_output = root.value("final_output", json::object());
                    output_validation = root.value("llama_output_validation", json::object());
                    admission_summary = root.value("admission_summary", json::object());
                }
                matched = true;
                break;
            }
        }

        if (matched) {
            if (trace_ref_ids.insert(root_trace_id).second) {
                traces.push_back(json{
                    {"trace_id", root_trace_id},
                    {"request_id", root.value("request_id", "")},
                    {"query_id", root.value("query_id", "")},
                    {"latest_stage", root.value("latest_stage", "")},
                });
            }
        }
    };

    {
        std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
        for (auto it = g_trace_order.rbegin(); it != g_trace_order.rend(); ++it) {
            const auto record_it = g_trace_registry.find(*it);
            if (record_it == g_trace_registry.end()) {
                continue;
            }
            collect_from_root(record_it->second.payload);
        }
    }

    if (approved_context_item.empty() && !g_trace_storage_base_path.empty()) {
        try {
            const std::string index_path = slice_index_path(slice_id);
            std::vector<std::string> indexed_trace_ids;
            std::unordered_set<std::string> scheduled_trace_ids;

            std::ifstream index_in(index_path, std::ios::binary);
            if (index_in.is_open()) {
                std::string line;
                while (std::getline(index_in, line)) {
                    if (line.empty()) {
                        continue;
                    }

                    json ref = json::parse(line, nullptr, false);
                    if (!ref.is_object()) {
                        continue;
                    }

                    const std::string trace_id = ref.value("trace_id", "");
                    if (trace_id.empty()) {
                        continue;
                    }
                    if (scheduled_trace_ids.insert(trace_id).second) {
                        indexed_trace_ids.push_back(trace_id);
                    }
                }
            }

            for (const auto & trace_id : indexed_trace_ids) {
                const std::string path = trace_snapshot_path(trace_id);
                if (path.empty()) {
                    continue;
                }

                std::ifstream in(path, std::ios::binary);
                if (!in.is_open()) {
                    continue;
                }

                json root;
                in >> root;
                collect_from_root(root);
            }

            if (approved_context_item.empty()) {
                const std::filesystem::path base(g_trace_storage_base_path);
                const std::filesystem::path dir = base.parent_path();
                const std::string prefix = base.filename().string() + ".trace.";

                if (std::filesystem::exists(dir)) {
                    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
                        if (!entry.is_regular_file()) {
                            continue;
                        }
                        const std::string filename = entry.path().filename().string();
                        if (filename.rfind(prefix, 0) != 0 || entry.path().extension() != ".json") {
                            continue;
                        }

                        std::ifstream in(entry.path(), std::ios::binary);
                        if (!in.is_open()) {
                            continue;
                        }

                        json root;
                        in >> root;
                        collect_from_root(root);
                    }
                }
            }
        } catch (...) {
        }
    }

    if (approved_context_item.empty()) {
        return false;
    }

    payload = json{
        {"record_model", "rag_evidence_debug_v1"},
        {"slice_id", slice_id},
        {"approved_context", approved_context_item},
        {"admission_summary", admission_summary},
        {"llama_output_validation", output_validation},
        {"final_output", final_output},
        {"trace_refs", traces},
    };
    return true;
}

} // namespace server_trace_registry
