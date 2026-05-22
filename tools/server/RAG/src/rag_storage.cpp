#include "rag_storage.h"

#include <filesystem>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

void ensure_parent_directory(const std::string & path) {
    if (path.empty()) {
        return;
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

std::string rag_storage_sanitize_token(const std::string & value) {
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

std::string rag_storage_vector_map_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".vectors.json";
}

std::string rag_storage_raw_slices_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".raw_slices.json";
}

std::string rag_storage_manifest_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".storage_manifest.json";
}

std::string rag_storage_clips_runs_log_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".clips_runs.jsonl";
}

std::string rag_storage_clips_run_snapshot_path(
        const std::string & base_path,
        const std::string & run_kind,
        const std::string & trace_id) {
    if (base_path.empty() || run_kind.empty() || trace_id.empty()) {
        return "";
    }
    return base_path + ".clips_run." + rag_storage_sanitize_token(run_kind) + "." +
        rag_storage_sanitize_token(trace_id) + ".json";
}

std::string rag_storage_clips_query_index_path(
        const std::string & base_path,
        const std::string & query_id) {
    if (base_path.empty() || query_id.empty()) {
        return "";
    }
    return base_path + ".clips_query." + rag_storage_sanitize_token(query_id) + ".jsonl";
}

std::string rag_storage_clips_slice_index_path(
        const std::string & base_path,
        const std::string & slice_id) {
    if (base_path.empty() || slice_id.empty()) {
        return "";
    }
    return base_path + ".clips_slice." + rag_storage_sanitize_token(slice_id) + ".jsonl";
}

void rag_storage_append_jsonl_record(const std::string & path, const rag_storage_json & record) {
    if (path.empty()) {
        return;
    }

    try {
        ensure_parent_directory(path);
        std::ofstream out(path, std::ios::binary | std::ios::app);
        if (out.is_open()) {
            out << record.dump() << '\n';
        }
    } catch (...) {
    }
}

void rag_storage_write_json_file(const std::string & path, const rag_storage_json & value) {
    if (path.empty()) {
        return;
    }

    try {
        ensure_parent_directory(path);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (out.is_open()) {
            out << value.dump();
        }
    } catch (...) {
    }
}

RagFileStorage::RagFileStorage(std::string base_path)
    : base_path_(std::move(base_path)) {
}

const std::string & RagFileStorage::base_path() const {
    return base_path_;
}

void RagFileStorage::save_manifest() const {
    if (base_path_.empty()) {
        return;
    }

    const json manifest = {
        {"schema_version", 1},
        {"store_model", "rag_storage_contract_v1"},
        {"backend", "file_sidecar"},
        {"index_name", base_path_},
        {"stores", {
            {"raw_slice", {
                {"path", rag_storage_raw_slices_path(base_path_)},
                {"record_model", "rag_raw_slice_store_v1"},
                {"keys", json::array({
                    "slice:{slice_id}:raw",
                    "slice:{slice_id}:meta",
                    "slice:{slice_id}:hash",
                    "slice:{slice_id}:vector_id",
                })},
            }},
            {"vector_slice_map", {
                {"path", rag_storage_vector_map_path(base_path_)},
                {"record_model", "rag_vector_slice_map_v1"},
                {"keys", json::array({
                    "vector:{index_name}:{faiss_id}",
                    "slice:{slice_id}:vector_id",
                })},
            }},
            {"clips_run", {
                {"path", rag_storage_clips_runs_log_path(base_path_)},
                {"record_model", "rag_clips_run_store_v1"},
                {"snapshot_pattern", base_path_ + ".clips_run.{run_kind}.{trace_id}.json"},
                {"keys", json::array({
                    "clips:run:{run_id}:input_facts",
                    "clips:run:{run_id}:output_facts",
                    "clips:decision:{run_id}",
                })},
            }},
            {"clips_query_index", {
                {"path_pattern", base_path_ + ".clips_query.{query_id}.jsonl"},
                {"record_model", "rag_clips_query_ref_v1"},
                {"keys", json::array({
                    "clips:query:{query_id}:run_ref",
                })},
            }},
            {"clips_slice_index", {
                {"path_pattern", base_path_ + ".clips_slice.{slice_id}.jsonl"},
                {"record_model", "rag_clips_slice_ref_v1"},
                {"keys", json::array({
                    "clips:slice:{slice_id}:decision_ref",
                })},
            }},
        }},
        {"planned_stores", {
            {"clips_fact_page", {
                {"record_model", "rag_clips_fact_page_v1"},
                {"keys", json::array({
                    "clips:fact:{trace_id}:{fact_id}",
                    "clips:run:{run_id}:input_facts",
                    "clips:run:{run_id}:output_facts",
                })},
            }},
            {"knowledge_node", {
                {"record_model", "rag_knowledge_node_v1"},
                {"keys", json::array({
                    "node:{node_id}",
                    "node_by_slice:{slice_id}:{node_id}",
                    "relation:{relation_id}",
                    "risk:{risk_id}",
                })},
            }},
            {"audit_event", {
                {"record_model", "rag_audit_event_v1"},
                {"keys", json::array({
                    "audit:{trace_id}:{seq}",
                    "audit_by_request:{request_id}:{seq}",
                    "audit_by_slice:{slice_id}:{seq}",
                    "audit_by_node:{node_id}:{seq}",
                })},
            }},
        }},
    };

    rag_storage_write_json_file(rag_storage_manifest_path(base_path_), manifest);
}

void RagFileStorage::save_vector_slice_map(const RagVectorSliceMap & map) const {
    if (base_path_.empty()) {
        return;
    }

    json vector_map;
    vector_map["schema_version"] = 1;
    vector_map["index_name"] = base_path_;
    vector_map["store_model"] = "rag_vector_slice_map_v1";
    vector_map["vector_to_slice"] = json::object();
    vector_map["slice_to_vector"] = json::object();

    for (size_t i = 0; i < map.vector_to_slice.size(); ++i) {
        const std::string & slice_id = map.vector_to_slice[i];
        if (!slice_id.empty()) {
            vector_map["vector_to_slice"][std::to_string(i)] = slice_id;
        }
    }
    for (const auto & item : map.slice_to_vector) {
        vector_map["slice_to_vector"][item.first] = item.second;
    }

    rag_storage_write_json_file(rag_storage_vector_map_path(base_path_), vector_map);
}

bool RagFileStorage::load_vector_slice_map(RagVectorSliceMap & map) const {
    map.vector_to_slice.clear();
    map.slice_to_vector.clear();

    std::ifstream in(rag_storage_vector_map_path(base_path_), std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    json j;
    in >> j;
    const json vector_to_slice = j.value("vector_to_slice", json::object());
    for (auto it = vector_to_slice.begin(); it != vector_to_slice.end(); ++it) {
        try {
            const size_t vector_id = static_cast<size_t>(std::stoull(it.key()));
            if (vector_id >= map.vector_to_slice.size()) {
                map.vector_to_slice.resize(vector_id + 1);
            }
            if (it.value().is_string()) {
                map.vector_to_slice[vector_id] = it.value().get<std::string>();
            }
        } catch (...) {
        }
    }

    const json slice_to_vector = j.value("slice_to_vector", json::object());
    for (auto it = slice_to_vector.begin(); it != slice_to_vector.end(); ++it) {
        if (it.value().is_number_integer()) {
            map.slice_to_vector[it.key()] = it.value().get<int>();
        }
    }

    return true;
}

void RagFileStorage::save_raw_slices(const RagRawSliceStore & store) const {
    if (base_path_.empty()) {
        return;
    }

    json raw_store;
    raw_store["schema_version"] = 1;
    raw_store["store_model"] = "rag_raw_slice_store_v1";
    raw_store["index_name"] = base_path_;
    raw_store["slices"] = json::object();
    raw_store["records"] = json::object();

    for (const auto & item : store.slices) {
        const RagRawSliceRecord & record = item.second;
        if (record.slice_id.empty()) {
            continue;
        }

        json metadata = json::object();
        if (!record.metadata_json.empty()) {
            try {
                metadata = json::parse(record.metadata_json);
            } catch (...) {
                metadata = record.metadata_json;
            }
        }

        raw_store["slices"][record.slice_id] = json{
            {"slice_id", record.slice_id},
            {"raw", record.raw_text},
            {"meta", metadata},
            {"hash", record.text_hash},
            {"vector_id", record.vector_id},
        };

        raw_store["records"]["slice:" + record.slice_id + ":raw"] = record.raw_text;
        raw_store["records"]["slice:" + record.slice_id + ":meta"] = metadata;
        raw_store["records"]["slice:" + record.slice_id + ":hash"] = record.text_hash;
        raw_store["records"]["slice:" + record.slice_id + ":vector_id"] = record.vector_id;
    }

    rag_storage_write_json_file(rag_storage_raw_slices_path(base_path_), raw_store);
}

bool RagFileStorage::load_raw_slices(RagRawSliceStore & store) const {
    store.slices.clear();

    std::ifstream in(rag_storage_raw_slices_path(base_path_), std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    json j;
    in >> j;
    const json slices = j.value("slices", json::object());
    for (auto it = slices.begin(); it != slices.end(); ++it) {
        if (!it.value().is_object()) {
            continue;
        }

        RagRawSliceRecord record;
        record.slice_id = it.value().value("slice_id", it.key());
        if (record.slice_id.empty()) {
            continue;
        }
        record.raw_text = it.value().value("raw", "");
        record.metadata_json = it.value().contains("meta") ? it.value()["meta"].dump() : "";
        record.text_hash = it.value().value("hash", "");
        record.vector_id = it.value().value("vector_id", -1);
        store.slices[record.slice_id] = std::move(record);
    }

    return true;
}
