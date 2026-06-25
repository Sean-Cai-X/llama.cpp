#include "rag_storage.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>
#include <unordered_map>

#include <nlohmann/json.hpp>

#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#endif

using json = nlohmann::json;

namespace {

std::string rocksdb_database_path_for_base(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".rocksdb";
}

std::vector<std::string> rocksdb_column_families() {
    return {
        "default",
        "slice",
        "vector",
        "clips",
        "knowledge",
        "review",
        "audit",
        "trace",
        "coupling",
        "viewpoint",
    };
}

void ensure_parent_directory(const std::string & path) {
    if (path.empty()) {
        return;
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
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

json parse_backend_json_value(const std::string & raw_value) {
    if (raw_value.empty()) {
        return json();
    }

    json parsed = json::parse(raw_value, nullptr, false);
    if (parsed.is_discarded()) {
        return json{{"raw_value", raw_value}};
    }
    return parsed;
}

#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
rocksdb::Status open_rocksdb_read_only(
        const std::string & db_path,
        std::vector<rocksdb::ColumnFamilyDescriptor> & descriptors,
        std::vector<rocksdb::ColumnFamilyHandle *> & handles,
        std::unique_ptr<rocksdb::DB> & db) {
    rocksdb::DBOptions db_options;
    db_options.create_if_missing = false;
    return rocksdb::DB::OpenForReadOnly(
        db_options,
        db_path,
        descriptors,
        &handles,
        &db,
        false);
}
#endif

void write_json_file_if_missing(const std::string & path, const json & value) {
    if (path.empty() || std::filesystem::exists(path)) {
        return;
    }
    rag_storage_write_json_file(path, value);
}

json read_backend_value_from_snapshot(
        const std::string & base_path,
        const std::string & column_family,
        const std::string & key) {
    const json snapshot = read_json_file_or_object(rag_storage_kv_snapshot_path(base_path));
    const json families = snapshot.value("column_families", json::object());
    if (!families.is_object() || !families.contains(column_family) || !families.at(column_family).is_object()) {
        return json();
    }
    const json family = families.at(column_family);
    if (!family.contains(key)) {
        return json();
    }
    return family.at(key);
}

json scan_backend_prefix_from_snapshot(
        const std::string & base_path,
        const std::string & column_family,
        const std::string & key_prefix,
        int limit) {
    json records = json::array();
    const json snapshot = read_json_file_or_object(rag_storage_kv_snapshot_path(base_path));
    const json families = snapshot.value("column_families", json::object());
    if (!families.is_object() || !families.contains(column_family) || !families.at(column_family).is_object()) {
        return records;
    }

    const json family = families.at(column_family);
    for (auto it = family.begin(); it != family.end(); ++it) {
        if (limit > 0 && static_cast<int>(records.size()) >= limit) {
            break;
        }
        if (it.key().rfind(key_prefix, 0) != 0) {
            continue;
        }
        records.push_back(json{
            {"key", it.key()},
            {"value", it.value()},
        });
    }
    return records;
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

json apply_rocksdb_write_batch(
        const std::string & base_path,
        const std::string & batch_id,
        const json & operations) {
    json status = {
        {"record_model", "rag_rocksdb_mirror_status_v1"},
        {"enabled", false},
        {"database_path", rocksdb_database_path_for_base(base_path)},
        {"batch_id", batch_id},
        {"attempted", false},
        {"applied", false},
        {"operation_count", operations.is_array() ? static_cast<int>(operations.size()) : 0},
    };

#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
    status["enabled"] = true;
    status["attempted"] = true;

    if (base_path.empty() || !operations.is_array() || operations.empty()) {
        status["error"] = "empty_base_or_operations";
        return status;
    }

    const std::string db_path = rocksdb_database_path_for_base(base_path);
    ensure_parent_directory(db_path);

    rocksdb::DBOptions db_options;
    db_options.create_if_missing = true;
    db_options.create_missing_column_families = true;

    rocksdb::ColumnFamilyOptions cf_options;
    std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
    for (const std::string & column_family : rocksdb_column_families()) {
        descriptors.emplace_back(column_family, cf_options);
    }

    std::unique_ptr<rocksdb::DB> db;
    std::vector<rocksdb::ColumnFamilyHandle *> handles;
    rocksdb::Status open_status = rocksdb::DB::Open(db_options, db_path, descriptors, &handles, &db);
    if (!open_status.ok()) {
        status["error"] = open_status.ToString();
        status["stage"] = "open";
        return status;
    }

    std::unordered_map<std::string, rocksdb::ColumnFamilyHandle *> handle_by_name;
    for (size_t i = 0; i < descriptors.size() && i < handles.size(); ++i) {
        handle_by_name[descriptors[i].name] = handles[i];
    }

    rocksdb::WriteBatch batch;
    int put_count = 0;
    for (const auto & op : operations) {
        if (!op.is_object() || op.value("op", "") != "put") {
            continue;
        }
        std::string column_family = op.value("column_family", "default");
        if (column_family.empty()) {
            column_family = "default";
        }
        const std::string key = op.value("key", "");
        if (key.empty()) {
            continue;
        }

        auto handle_it = handle_by_name.find(column_family);
        if (handle_it == handle_by_name.end()) {
            handle_it = handle_by_name.find("default");
        }
        if (handle_it == handle_by_name.end()) {
            continue;
        }

        const json value = op.contains("value") ? op["value"] : json();
        const rocksdb::Status put_status = batch.Put(handle_it->second, key, value.dump());
        if (!put_status.ok()) {
            status["error"] = put_status.ToString();
            status["stage"] = "batch_put";
            for (auto * handle : handles) {
                db->DestroyColumnFamilyHandle(handle);
            }
            return status;
        }
        ++put_count;
    }

    const json batch_ref = {
        {"schema_version", 1},
        {"record_model", "rag_rocksdb_batch_ref_v1"},
        {"index_name", base_path},
        {"batch_id", batch_id},
        {"put_count", put_count},
    };
    auto default_handle = handle_by_name.find("default");
    if (default_handle != handle_by_name.end()) {
        batch.Put(default_handle->second, "meta:last_batch", batch_ref.dump());
    }

    rocksdb::WriteOptions write_options;
    const rocksdb::Status write_status = db->Write(write_options, &batch);

    for (auto * handle : handles) {
        db->DestroyColumnFamilyHandle(handle);
    }

    if (!write_status.ok()) {
        status["error"] = write_status.ToString();
        status["stage"] = "write";
        return status;
    }

    status["applied"] = true;
    status["put_count"] = put_count;
    status["column_families"] = rocksdb_column_families();
    status["stage"] = "complete";
#else
    (void) base_path;
    (void) batch_id;
    (void) operations;
    status["reason"] = "LLAMA_SERVER_RAG_ROCKSDB_BACKEND not enabled";
#endif

    return status;
}

void append_backend_write_batch(
        const std::string & base_path,
        const std::string & batch_id,
        const json & operations) {
    if (base_path.empty() || !operations.is_array() || operations.empty()) {
        return;
    }

    json batch = {
        {"schema_version", 1},
        {"record_model", "rag_storage_backend_write_batch_v1"},
        {"index_name", base_path},
        {"batch_id", batch_id},
        {"active_backend", "file_sidecar"},
        {"target_backends", json::array({"file_sidecar", "rocksdb", "sqlite"})},
        {"operation_count", static_cast<int>(operations.size())},
        {"operations", operations},
    };
    const json rocksdb_status = apply_rocksdb_write_batch(base_path, batch_id, operations);
    batch["rocksdb_mirror"] = rocksdb_status;
    rag_storage_append_jsonl_record(rag_storage_kv_journal_path(base_path), batch);
    rag_storage_write_json_file(
        rag_storage_rocksdb_status_path(base_path),
        json{
            {"schema_version", 1},
            {"record_model", "rag_rocksdb_backend_status_v1"},
            {"index_name", base_path},
            {"database_path", rocksdb_database_path_for_base(base_path)},
            {"last_batch_id", batch_id},
            {"last_mirror", rocksdb_status},
        });

    json snapshot = read_json_file_or_object(rag_storage_kv_snapshot_path(base_path));
    if (snapshot.empty()) {
        snapshot = {
            {"schema_version", 1},
            {"record_model", "rag_storage_backend_kv_snapshot_v1"},
            {"index_name", base_path},
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
    snapshot["last_rocksdb_mirror"] = rocksdb_status;
    snapshot["record_count"] = snapshot["records"].size();
    rag_storage_write_json_file(rag_storage_kv_snapshot_path(base_path), snapshot);
}

json build_database_contract(const std::string & base_path) {
    const json logical_keyspaces = {
        {"slice", json::array({
            "slice:{slice_id}",
            "slice:{slice_id}:raw",
            "slice:{slice_id}:meta",
            "slice:{slice_id}:hash",
            "slice:{slice_id}:vector_id",
            "slice:{slice_id}:source",
        })},
        {"vector", json::array({
            "vector:{index_name}:{faiss_id}",
            "slice:{slice_id}:vector_id",
        })},
        {"clips", json::array({
            "clips:fact:{trace_id}:{fact_id}",
            "clips:run:{run_id}:input_facts",
            "clips:run:{run_id}:output_facts",
            "clips:decision:{run_id}",
            "clips:query:{query_id}:run_ref",
            "clips:slice:{slice_id}:decision_ref",
        })},
        {"knowledge", json::array({
            "node:{node_id}",
            "node_by_slice:{slice_id}:{node_id}",
            "relation:{relation_id}",
            "risk:{risk_id}",
        })},
        {"coupling", json::array({
            "semantic_slice:{slice_id}",
            "slice_edge:{edge_id}",
            "coupling_candidate:{candidate_id}",
            "coupling_decision:{candidate_id}",
            "coupling_by_slice:{slice_id}:{edge_id}",
        })},
        {"viewpoint", json::array({
            "viewpoint:{viewpoint_id}",
            "viewpoint:evidence:{viewpoint_id}:{source_id}",
            "viewpoint:decision:{viewpoint_id}",
            "viewpoint_by_slice:{slice_id}:{viewpoint_id}",
            "viewpoint_by_trace:{trace_id}:{viewpoint_id}",
        })},
        {"review", json::array({
            "review:{observation_id}",
            "review_by_trace:{trace_id}:{observation_id}",
            "review_by_query:{query_id}:{observation_id}",
            "review_by_bucket:{test_bucket}:{observation_id}",
            "review_by_gap:{coverage_gap}:{observation_id}",
        })},
        {"audit", json::array({
            "audit:{trace_id}:{seq}",
            "audit_by_request:{request_id}:{seq}",
            "audit_by_slice:{slice_id}:{seq}",
            "audit_by_node:{node_id}:{seq}",
            "trace:{trace_id}",
        })},
    };

    const json record_models = {
        {"slice", "rag_slice_record_v1"},
        {"slice_db", "rag_slice_db_v1"},
        {"vector_slice_map", "rag_vector_slice_map_v1"},
        {"clips_run", "rag_clips_run_store_v1"},
        {"clips_fact_page", "rag_clips_fact_page_v1"},
        {"knowledge_node", "rag_knowledge_node_v1"},
        {"semantic_slice", "rag_semantic_slice_v1"},
        {"slice_edge", "rag_slice_edge_v1"},
        {"coupling_candidate", "rag_coupling_candidate_v1"},
        {"coupling_decision", "rag_coupling_decision_v1"},
        {"viewpoint_candidate", "rag_viewpoint_candidate_v1"},
        {"evidence_binding", "rag_evidence_binding_v1"},
        {"viewpoint_decision", "rag_viewpoint_decision_v1"},
        {"review_observation", "rag_review_observation_v1"},
        {"audit_event", "rag_audit_event_v1"},
        {"trace_debug", "rag_trace_debug_v1"},
    };

    json rocksdb_backend = {
        {"status", "disabled_at_compile_time"},
        {"model", "rocksdb_column_family_kv"},
        {"source_hint", "analysis_workspace/rocksdb-11.0.4"},
        {"database_path", rocksdb_database_path_for_base(base_path)},
        {"write_batch_journal_path", rag_storage_kv_journal_path(base_path)},
        {"write_batch_snapshot_path", rag_storage_kv_snapshot_path(base_path)},
        {"column_families", rocksdb_column_families()},
        {"column_family_keyspaces", {
            {"slice", logical_keyspaces["slice"]},
            {"vector", logical_keyspaces["vector"]},
            {"clips", logical_keyspaces["clips"]},
            {"knowledge", logical_keyspaces["knowledge"]},
            {"review", logical_keyspaces["review"]},
            {"coupling", logical_keyspaces["coupling"]},
            {"viewpoint", logical_keyspaces["viewpoint"]},
            {"audit", logical_keyspaces["audit"]},
            {"trace", json::array({"trace:{trace_id}"})},
        }},
        {"required_write_semantics", json::array({
            "write raw slice, metadata, hash, and vector refs in one logical batch",
            "append audit, CLIPS, and knowledge refs without rewriting historical events",
            "keep slice_id stable across vector rebuilds and CLIPS replay",
        })},
    };
#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
    rocksdb_backend["status"] = "linked_mirror_backend";
    rocksdb_backend["mirror_write_batch"] = true;
#else
    rocksdb_backend["mirror_write_batch"] = false;
#endif

    return json{
        {"schema_version", 1},
        {"record_model", "rag_storage_backend_contract_v1"},
        {"index_name", base_path},
        {"active_backend", "file_sidecar"},
        {"contract_status", "backend_contract_ready"},
        {"logical_keyspaces", logical_keyspaces},
        {"record_models", record_models},
        {"backends", {
            {"file_sidecar", {
                {"status", "active"},
                {"model", "json_jsonl_sidecar"},
                {"root_path", base_path},
                {"paths", {
                    {"manifest", rag_storage_manifest_path(base_path)},
                    {"vector_map", rag_storage_vector_map_path(base_path)},
                    {"raw_slices", rag_storage_raw_slices_path(base_path)},
                    {"slice_db", rag_storage_slice_db_path(base_path)},
                    {"slice_record_pattern", base_path + ".slice.{slice_id}.json"},
                    {"kv_journal", rag_storage_kv_journal_path(base_path)},
                    {"kv_snapshot", rag_storage_kv_snapshot_path(base_path)},
                    {"viewpoint_store", rag_storage_viewpoint_store_path(base_path)},
                    {"viewpoint_index_pattern", base_path + ".viewpoint.{viewpoint_id}.jsonl"},
                    {"viewpoint_baseline", rag_storage_viewpoint_baseline_path(base_path)},
                    {"review_store", rag_storage_review_store_path(base_path)},
                    {"review_index_pattern", base_path + ".review_observation.{observation_id}.jsonl"},
                    {"review_trace_index_pattern", base_path + ".review_trace.{trace_id}.jsonl"},
                    {"review_bucket_index_pattern", base_path + ".review_bucket.{test_bucket}.jsonl"},
                    {"coupling_graph", rag_storage_coupling_graph_path(base_path)},
                    {"coupling_slice_index_pattern", base_path + ".coupling_slice.{slice_id}.jsonl"},
                    {"coupling_baseline", rag_storage_coupling_baseline_path(base_path)},
                    {"clips_fact_page_pattern", base_path + ".clips_fact_page.{run_kind}.{trace_id}.json"},
                    {"knowledge_nodes_pattern", base_path + ".knowledge_nodes.{run_kind}.{trace_id}.json"},
                    {"audit_log", base_path + ".audit.jsonl"},
                    {"trace_snapshot_pattern", base_path + ".trace.{trace_id}.json"},
                }},
            }},
            {"rocksdb", rocksdb_backend},
            {"sqlite", {
                {"status", "contract_only"},
                {"model", "sqlite_relational_debug_backend"},
                {"database_path", rag_storage_sqlite_path(base_path)},
                {"write_batch_journal_path", rag_storage_kv_journal_path(base_path)},
                {"write_batch_snapshot_path", rag_storage_kv_snapshot_path(base_path)},
                {"tables", {
                    {"rag_slice", json::array({
                        "slice_id TEXT PRIMARY KEY",
                        "source_uri TEXT",
                        "start_line INTEGER",
                        "end_line INTEGER",
                        "language TEXT",
                        "text_hash TEXT",
                        "vector_id INTEGER",
                        "metadata_json TEXT",
                        "raw_text TEXT",
                    })},
                    {"rag_vector_map", json::array({
                        "index_name TEXT",
                        "faiss_id INTEGER",
                        "slice_id TEXT",
                        "PRIMARY KEY(index_name, faiss_id)",
                    })},
                    {"rag_clips_fact", json::array({
                        "trace_id TEXT",
                        "run_id TEXT",
                        "fact_id TEXT",
                        "fact_kind TEXT",
                        "fact_json TEXT",
                    })},
                    {"rag_knowledge_node", json::array({
                        "node_id TEXT PRIMARY KEY",
                        "trace_id TEXT",
                        "node_json TEXT",
                    })},
                    {"rag_slice_edge", json::array({
                        "edge_id TEXT PRIMARY KEY",
                        "from_slice TEXT",
                        "to_slice TEXT",
                        "edge_type TEXT",
                        "confidence REAL",
                        "status TEXT",
                        "edge_json TEXT",
                    })},
                    {"rag_viewpoint", json::array({
                        "viewpoint_id TEXT PRIMARY KEY",
                        "trace_id TEXT",
                        "source_slice_id TEXT",
                        "claim_type TEXT",
                        "decision TEXT",
                        "viewpoint_json TEXT",
                    })},
                    {"rag_review_observation", json::array({
                        "observation_id TEXT PRIMARY KEY",
                        "trace_id TEXT",
                        "query_id TEXT",
                        "test_bucket TEXT",
                        "result_stage TEXT",
                        "coverage_gap TEXT",
                        "observation_json TEXT",
                    })},
                    {"rag_audit_event", json::array({
                        "trace_id TEXT",
                        "seq INTEGER",
                        "request_id TEXT",
                        "query_id TEXT",
                        "stage TEXT",
                        "payload_json TEXT",
                        "PRIMARY KEY(trace_id, seq)",
                    })},
                    {"rag_trace_index", json::array({
                        "trace_id TEXT PRIMARY KEY",
                        "request_id TEXT",
                        "query_id TEXT",
                        "latest_stage TEXT",
                        "snapshot_json TEXT",
                    })},
                }},
            }},
        }},
        {"migration_rules", json::array({
            "file_sidecar remains the export and compatibility backend",
            "rocksdb stores the canonical key/value working set for L2 memory",
            "sqlite stores optional local debug and queryable validation views",
            "all backends must preserve schema_version and record_model fields",
        })},
    };
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

rag_storage_json rag_storage_make_backend_put(
        const std::string & column_family,
        const std::string & key,
        const rag_storage_json & value,
        const std::string & record_model) {
    return make_backend_put(column_family, key, value, record_model);
}

rag_storage_json rag_storage_read_backend_value(
        const std::string & base_path,
        const std::string & column_family,
        const std::string & key) {
    json result = {
        {"record_model", "rag_storage_backend_lookup_v1"},
        {"base_path", base_path},
        {"column_family", column_family},
        {"key", key},
        {"backend", "not_found"},
        {"found", false},
    };

    if (base_path.empty() || column_family.empty() || key.empty()) {
        result["error"] = "empty_base_column_family_or_key";
        return result;
    }

#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
    try {
        const std::string db_path = rocksdb_database_path_for_base(base_path);
        std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
        rocksdb::ColumnFamilyOptions cf_options;
        for (const std::string & family_name : rocksdb_column_families()) {
            descriptors.emplace_back(family_name, cf_options);
        }

        std::unique_ptr<rocksdb::DB> db;
        std::vector<rocksdb::ColumnFamilyHandle *> handles;
        rocksdb::Status open_status = open_rocksdb_read_only(db_path, descriptors, handles, db);
        if (open_status.ok()) {
            rocksdb::ColumnFamilyHandle * selected_handle = nullptr;
            for (size_t i = 0; i < descriptors.size() && i < handles.size(); ++i) {
                if (descriptors[i].name == column_family) {
                    selected_handle = handles[i];
                    break;
                }
            }

            if (selected_handle != nullptr) {
                std::string raw_value;
                const rocksdb::Status get_status = db->Get(rocksdb::ReadOptions(), selected_handle, key, &raw_value);
                if (get_status.ok()) {
                    result["backend"] = "rocksdb";
                    result["found"] = true;
                    result["value"] = parse_backend_json_value(raw_value);
                } else if (!get_status.IsNotFound()) {
                    result["rocksdb_error"] = get_status.ToString();
                }
            }

            for (auto * handle : handles) {
                db->DestroyColumnFamilyHandle(handle);
            }
        } else {
            result["rocksdb_error"] = open_status.ToString();
        }
    } catch (...) {
        result["rocksdb_error"] = "rocksdb_exception";
    }
#else
    result["rocksdb_error"] = "LLAMA_SERVER_RAG_ROCKSDB_BACKEND not enabled";
#endif

    if (!result.value("found", false)) {
        const json snapshot_value = read_backend_value_from_snapshot(base_path, column_family, key);
        if (!snapshot_value.is_null() && !(snapshot_value.is_object() && snapshot_value.empty())) {
            result["backend"] = "kv_snapshot";
            result["found"] = true;
            result["value"] = snapshot_value;
        }
    }

    return result;
}

rag_storage_json rag_storage_scan_backend_prefix(
        const std::string & base_path,
        const std::string & column_family,
        const std::string & key_prefix,
        int limit) {
    json result = {
        {"record_model", "rag_storage_backend_scan_v1"},
        {"base_path", base_path},
        {"column_family", column_family},
        {"key_prefix", key_prefix},
        {"limit", limit},
        {"backend", "not_found"},
        {"record_count", 0},
        {"records", json::array()},
    };

    if (base_path.empty() || column_family.empty() || key_prefix.empty()) {
        result["error"] = "empty_base_column_family_or_key_prefix";
        return result;
    }

#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
    try {
        const std::string db_path = rocksdb_database_path_for_base(base_path);
        std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
        rocksdb::ColumnFamilyOptions cf_options;
        for (const std::string & family_name : rocksdb_column_families()) {
            descriptors.emplace_back(family_name, cf_options);
        }

        std::unique_ptr<rocksdb::DB> db;
        std::vector<rocksdb::ColumnFamilyHandle *> handles;
        rocksdb::Status open_status = open_rocksdb_read_only(db_path, descriptors, handles, db);
        if (open_status.ok()) {
            rocksdb::ColumnFamilyHandle * selected_handle = nullptr;
            for (size_t i = 0; i < descriptors.size() && i < handles.size(); ++i) {
                if (descriptors[i].name == column_family) {
                    selected_handle = handles[i];
                    break;
                }
            }

            if (selected_handle != nullptr) {
                std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(rocksdb::ReadOptions(), selected_handle));
                for (it->Seek(key_prefix);
                     it->Valid() && (limit <= 0 || static_cast<int>(result["records"].size()) < limit);
                     it->Next()) {
                    const std::string scanned_key = it->key().ToString();
                    if (scanned_key.rfind(key_prefix, 0) != 0) {
                        break;
                    }
                    result["records"].push_back(json{
                        {"key", scanned_key},
                        {"value", parse_backend_json_value(it->value().ToString())},
                    });
                }
                if (!it->status().ok()) {
                    result["rocksdb_error"] = it->status().ToString();
                }
            }

            for (auto * handle : handles) {
                db->DestroyColumnFamilyHandle(handle);
            }
        } else {
            result["rocksdb_error"] = open_status.ToString();
        }
    } catch (...) {
        result["rocksdb_error"] = "rocksdb_exception";
    }
#else
    result["rocksdb_error"] = "LLAMA_SERVER_RAG_ROCKSDB_BACKEND not enabled";
#endif

    if (result["records"].empty()) {
        result["records"] = scan_backend_prefix_from_snapshot(base_path, column_family, key_prefix, limit);
        if (!result["records"].empty()) {
            result["backend"] = "kv_snapshot";
        }
    } else {
        result["backend"] = "rocksdb";
    }

    result["record_count"] = result["records"].size();
    return result;
}

void rag_storage_append_backend_write_batch(
        const std::string & base_path,
        const std::string & batch_id,
        const rag_storage_json & operations) {
    append_backend_write_batch(base_path, batch_id, operations);
}

std::string rag_storage_vector_map_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".vectors.json";
}

std::string rag_storage_raw_slices_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".raw_slices.json";
}

std::string rag_storage_slice_db_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".slice_db.json";
}

std::string rag_storage_slice_record_path(const std::string & base_path, const std::string & slice_id) {
    if (base_path.empty() || slice_id.empty()) {
        return "";
    }
    return base_path + ".slice." + rag_storage_sanitize_token(slice_id) + ".json";
}

std::string rag_storage_db_contract_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".db_contract.json";
}

std::string rag_storage_rocksdb_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".rocksdb";
}

std::string rag_storage_rocksdb_status_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".rocksdb_status.json";
}

std::string rag_storage_sqlite_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".sqlite3";
}

std::string rag_storage_kv_journal_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".kv_journal.jsonl";
}

std::string rag_storage_kv_snapshot_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".kv_snapshot.json";
}

std::string rag_storage_viewpoint_store_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".viewpoints.json";
}

std::string rag_storage_viewpoint_index_path(const std::string & base_path, const std::string & viewpoint_id) {
    if (base_path.empty() || viewpoint_id.empty()) {
        return "";
    }
    return base_path + ".viewpoint." + rag_storage_sanitize_token(viewpoint_id) + ".jsonl";
}

std::string rag_storage_viewpoint_baseline_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".baseline.viewpoint_validation.json";
}

std::string rag_storage_review_store_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".review_observations.json";
}

std::string rag_storage_review_index_path(const std::string & base_path, const std::string & observation_id) {
    if (base_path.empty() || observation_id.empty()) {
        return "";
    }
    return base_path + ".review_observation." + rag_storage_sanitize_token(observation_id) + ".jsonl";
}

std::string rag_storage_review_trace_index_path(const std::string & base_path, const std::string & trace_id) {
    if (base_path.empty() || trace_id.empty()) {
        return "";
    }
    return base_path + ".review_trace." + rag_storage_sanitize_token(trace_id) + ".jsonl";
}

std::string rag_storage_review_bucket_index_path(const std::string & base_path, const std::string & test_bucket) {
    if (base_path.empty() || test_bucket.empty()) {
        return "";
    }
    return base_path + ".review_bucket." + rag_storage_sanitize_token(test_bucket) + ".jsonl";
}

std::string rag_storage_coupling_graph_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".coupling_graph.json";
}

std::string rag_storage_coupling_slice_index_path(const std::string & base_path, const std::string & slice_id) {
    if (base_path.empty() || slice_id.empty()) {
        return "";
    }
    return base_path + ".coupling_slice." + rag_storage_sanitize_token(slice_id) + ".jsonl";
}

std::string rag_storage_coupling_baseline_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".baseline.slice_coupling.json";
}

std::string rag_storage_manifest_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".storage_manifest.json";
}

std::string rag_storage_manual_test_plan_path(const std::string & base_path) {
    return base_path.empty() ? "" : base_path + ".manual_test_plan.json";
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

std::string rag_storage_clips_fact_page_path(
        const std::string & base_path,
        const std::string & run_kind,
        const std::string & trace_id) {
    if (base_path.empty() || run_kind.empty() || trace_id.empty()) {
        return "";
    }
    return base_path + ".clips_fact_page." + rag_storage_sanitize_token(run_kind) + "." +
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

std::string rag_storage_knowledge_nodes_path(
        const std::string & base_path,
        const std::string & run_kind,
        const std::string & trace_id) {
    if (base_path.empty() || run_kind.empty() || trace_id.empty()) {
        return "";
    }
    return base_path + ".knowledge_nodes." + rag_storage_sanitize_token(run_kind) + "." +
        rag_storage_sanitize_token(trace_id) + ".json";
}

std::string rag_storage_knowledge_node_index_path(
        const std::string & base_path,
        const std::string & node_id) {
    if (base_path.empty() || node_id.empty()) {
        return "";
    }
    return base_path + ".knowledge_node." + rag_storage_sanitize_token(node_id) + ".jsonl";
}

std::string rag_storage_knowledge_slice_index_path(
        const std::string & base_path,
        const std::string & slice_id) {
    if (base_path.empty() || slice_id.empty()) {
        return "";
    }
    return base_path + ".knowledge_slice." + rag_storage_sanitize_token(slice_id) + ".jsonl";
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

    const json manual_test_plan = {
        {"schema_version", 1},
        {"record_model", "rag_manual_validation_plan_v1"},
        {"index_name", base_path_},
        {"purpose", "Manual validation backup for semantic slices, file slices, semantic indexes, CLIPS meta links, and storage replay."},
        {"fixture", {
            {"documents", json::array({
                "RAG runtime is isolated under tools/server/RAG and exposes /rag/search and /rag/explain.",
                "The embedding side is handled by the RAG main thread, while MCP semantic integration stays in the integration thread.",
            })},
            {"metadata", json::array({
                "path=docs/rag_runtime.txt;language=text;start_line=1;end_line=1;prechunked=1",
                "path=docs/thread_split.txt;language=text;start_line=1;end_line=1;prechunked=1",
            })},
            {"query", "Where is the RAG runtime located? Answer briefly."},
        }},
        {"capability_baselines", {
            {"viewpoint_validation", {
                {"purpose", "Measure whether model claims are gated by evidence, type, confidence, and CLIPS decision before entering reports or knowledge nodes."},
                {"input_facts", json::array({
                    "(viewpoint-candidate (viewpoint-id \"VP-FACT-NO-EVIDENCE\") (request-id \"REQ-baseline\") (trace-id \"TRACE-baseline\") (model-task-id \"MODEL-baseline\") (source-slice-id \"SLICE-14419879304649273516\") (claim-text \"RAG runtime is under tools/server/RAG\") (claim-type FACT_CLAIM) (subject \"SLICE-14419879304649273516\") (predicate \"HAS_LOCATION\") (object \"tools/server/RAG\") (confidence 0.90) (status CANDIDATE))",
                    "(viewpoint-candidate (viewpoint-id \"VP-FACT-WITH-EVIDENCE\") (request-id \"REQ-baseline\") (trace-id \"TRACE-baseline\") (model-task-id \"MODEL-baseline\") (source-slice-id \"SLICE-14419879304649273516\") (claim-text \"RAG runtime is under tools/server/RAG\") (claim-type FACT_CLAIM) (subject \"SLICE-14419879304649273516\") (predicate \"HAS_LOCATION\") (object \"tools/server/RAG\") (confidence 0.90) (status CANDIDATE))",
                    "(evidence-binding (viewpoint-id \"VP-FACT-WITH-EVIDENCE\") (source-type \"SLICE\") (source-id \"SLICE-14419879304649273516\") (evidence-ref \"docs/rag_runtime.txt:1-1\") (evidence-hash \"sha256:8727233990702678416\") (match-status MATCHED) (status VERIFIED))",
                    "(viewpoint-candidate (viewpoint-id \"VP-SPECULATION\") (request-id \"REQ-baseline\") (trace-id \"TRACE-baseline\") (model-task-id \"MODEL-baseline\") (source-slice-id \"SLICE-14419879304649273516\") (claim-text \"RAG runtime may have hidden dependency risk\") (claim-type SPECULATION) (subject \"SLICE-14419879304649273516\") (predicate \"MAY_HAVE_RISK\") (object \"UNKNOWN\") (confidence 0.55) (status CANDIDATE))",
                })},
                {"expected_decisions", json::array({
                    "VP-FACT-NO-EVIDENCE -> REJECT / FACT_CLAIM_WITHOUT_VERIFIED_EVIDENCE",
                    "VP-FACT-WITH-EVIDENCE -> APPROVE / FACT_CLAIM_SUPPORTED_BY_VERIFIED_EVIDENCE",
                    "VP-SPECULATION -> DOWNGRADE_TO_HYPOTHESIS / SPECULATION_CANNOT_BE_TREATED_AS_FACT",
                })},
                {"pass_criteria", json::array({
                    "CLIPS output captures viewpoint-decision facts",
                    "viewpoints store contains candidate, evidence, and decision records when the controller persists them",
                    "audit or trace output can connect viewpoint_id to source_slice_id and trace_id",
                })},
                {"fail_criteria", json::array({
                    "FACT_CLAIM without evidence is approved",
                    "SPECULATION is emitted as verified fact",
                    "viewpoint decision is missing before report or knowledge-node admission",
                })},
            }},
            {"slice_coupling", {
                {"purpose", "Measure whether candidate slice relations are converted into verified edges only when policy, confidence, and slice existence constraints pass."},
                {"input_facts", json::array({
                    "(semantic-slice (slice-id \"SLICE-14419879304649273516\") (file-id \"FILE-docs-rag-runtime\") (source-uri \"docs/rag_runtime.txt\") (start-line 1) (end-line 1) (text-hash \"sha256:8727233990702678416\") (language \"text\") (symbol-scope \"\") (status VERIFIED))",
                    "(semantic-slice (slice-id \"SLICE-10177065769148280947\") (file-id \"FILE-docs-thread-split\") (source-uri \"docs/thread_split.txt\") (start-line 1) (end-line 1) (text-hash \"sha256:5133134130241714942\") (language \"text\") (symbol-scope \"\") (status VERIFIED))",
                    "(coupling-policy (task-type CODE_RISK_ANALYSIS) (allowed-edge-types SAME_FILE ADJACENT SAME_RISK_PATTERN SUPPORTS EXPLAINS) (max-depth 2) (max-context-slices 6) (min-confidence 0.75) (token-budget 1800))",
                    "(coupling-candidate (candidate-id \"CC-SELF\") (from-slice \"SLICE-14419879304649273516\") (to-slice \"SLICE-14419879304649273516\") (candidate-type SUPPORTS) (score 0.95) (source \"baseline\") (status CANDIDATE))",
                    "(coupling-candidate (candidate-id \"CC-LOW\") (from-slice \"SLICE-14419879304649273516\") (to-slice \"SLICE-10177065769148280947\") (candidate-type SUPPORTS) (score 0.20) (source \"baseline\") (status CANDIDATE))",
                    "(coupling-candidate (candidate-id \"CC-ALLOW\") (from-slice \"SLICE-14419879304649273516\") (to-slice \"SLICE-10177065769148280947\") (candidate-type SUPPORTS) (score 0.88) (source \"baseline\") (status CANDIDATE))",
                })},
                {"expected_decisions", json::array({
                    "CC-SELF -> REJECT / SELF_COUPLING_NOT_ALLOWED",
                    "CC-LOW -> REJECT / COUPLING_SCORE_BELOW_POLICY_THRESHOLD",
                    "CC-ALLOW -> APPROVE / COUPLING_ALLOWED_BY_POLICY and slice-edge VERIFIED",
                })},
                {"pass_criteria", json::array({
                    "CLIPS output captures coupling-decision and slice-edge facts",
                    "coupling_graph contains semantic_slices seeded from raw slices",
                    "KV snapshot contains coupling:semantic_slice:{slice_id} entries",
                    "future graph expansion uses only VERIFIED slice-edge records",
                })},
                {"fail_criteria", json::array({
                    "self-coupling becomes a verified edge",
                    "low-confidence candidate becomes a verified edge",
                    "approved_context expands across a coupling edge without coupling-decision APPROVE",
                })},
            }},
        }},
        {"steps", json::array({
            {
                {"step_id", "01_index_fixture"},
                {"route", "POST /rag/add"},
                {"request_body_hint", "Use fixture.documents, fixture.metadata, and reset=true."},
                {"expected", json::array({
                    "status is queued",
                    "request_context.trace_id is present",
                    "after completion /rag/index/status reports chunk_count=2",
                })},
            },
            {
                {"step_id", "02_context_retrieval"},
                {"route", "POST /rag/chat/context"},
                {"request_body_hint", "Use fixture.query with top_k=2 and explicit request_id/trace_id."},
                {"expected", json::array({
                    "approved_context contains slice_id values",
                    "admission_summary.dominant_decision is allow",
                    "context_block cites source path and line range",
                })},
            },
            {
                {"step_id", "03_model_completion"},
                {"route", "POST /v1/chat/completions"},
                {"request_body_hint", "Use rag=true, stream=false, and a user question matching fixture.query."},
                {"expected", json::array({
                    "llama_output_validation.status is APPROVED",
                    "final_output.approved_by_clips is true",
                    "supporting_slice_ids is non-empty",
                })},
            },
            {
                {"step_id", "04_storage_artifacts"},
                {"route", "file_sidecar"},
                {"request_body_hint", "Read paths from /rag/index/status.storage and the trace_id returned by step 03."},
                {"expected", json::array({
                    "vectors file maps FAISS vector ids to slice_id",
                    "raw_slices file stores raw/meta/hash/vector_id by slice_id",
                    "slice_db file stores one database record per slice_id",
                    "slice record file stores raw/meta/hash/vector/source refs for one slice_id",
                    "db_contract file declares the shared file_sidecar, RocksDB, and SQLite backend contract",
                    "coupling_graph declares semantic_slice, coupling_candidate, coupling_decision, and slice_edge stores",
                    "viewpoints store declares viewpoint_candidate, evidence_binding, and viewpoint_decision stores",
                    "clips_fact_page stores input and output CLIPS facts for the trace",
                    "knowledge_nodes stores nodes, slice_links, relations, and risks",
                    "audit_by_request, audit_by_slice, and audit_by_node contain replay refs",
                    "trace snapshot event chain includes knowledge_node_persist, rag_chat_context, completion",
                })},
            },
            {
                {"step_id", "05_debug_replay"},
                {"route", "GET /v1/debug/trace/:trace_id and GET /v1/debug/evidence/:slice_id"},
                {"request_body_hint", "Use trace_id and one slice_id from approved_context."},
                {"expected", json::array({
                    "trace response latest_stage is completion",
                    "evidence response includes trace_refs and approved_context",
                    "slice evidence can be linked back to the final_output support set",
                })},
            },
            {
                {"step_id", "06_viewpoint_validation_baseline"},
                {"route", "POST /rag/clips/run"},
                {"request_body_hint", "{\"baseline_id\":\"viewpoint_validation\",\"request_id\":\"REQ-baseline-viewpoint\",\"trace_id\":\"TRACE-baseline-viewpoint\"}"},
                {"expected", json::array({
                    "VP-FACT-NO-EVIDENCE is rejected",
                    "VP-FACT-WITH-EVIDENCE is approved",
                    "VP-SPECULATION is downgraded to hypothesis",
                    "viewpoint-decision facts are visible in CLIPS captured output",
                    "baseline_evaluation.pass is true",
                })},
            },
            {
                {"step_id", "07_slice_coupling_baseline"},
                {"route", "POST /rag/clips/run"},
                {"request_body_hint", "{\"baseline_id\":\"slice_coupling\",\"request_id\":\"REQ-baseline-coupling\",\"trace_id\":\"TRACE-baseline-coupling\"}"},
                {"expected", json::array({
                    "CC-SELF is rejected",
                    "CC-LOW is rejected by threshold",
                    "CC-ALLOW is approved",
                    "a VERIFIED slice-edge is asserted only for CC-ALLOW",
                    "baseline_evaluation.pass is true",
                })},
            },
            {
                {"step_id", "08_rocksdb_backend_mirror"},
                {"route", "GET /rag/index/status and file_sidecar"},
                {"request_body_hint", "Build with LLAMA_SERVER_RAG_ENABLE_ROCKSDB=ON and configured RocksDB include/library, then rerun steps 01, 06, and 07."},
                {"expected", json::array({
                    "storage.database_backends.rocksdb.enabled is true",
                    "storage.database_backends.rocksdb.applied is true after a write batch",
                    "runtime.rocksdb_status.json last_mirror.applied is true",
                    "runtime.kv_snapshot.json last_rocksdb_mirror.applied is true",
                    "runtime.rocksdb directory exists and is non-empty",
                    "file_sidecar artifacts remain available as debug/export copies",
                })},
                {"fail_criteria", json::array({
                    "RocksDB enabled but last_mirror.applied is false",
                    "RocksDB mirror failure prevents file_sidecar artifacts from being written",
                    "KV journal/snapshot and RocksDB mirror report different last_batch_id values",
                })},
            },
        })},
        {"artifact_checks", {
            {"manifest_path", rag_storage_manifest_path(base_path_)},
            {"vector_map_path", rag_storage_vector_map_path(base_path_)},
            {"raw_slices_path", rag_storage_raw_slices_path(base_path_)},
            {"slice_db_path", rag_storage_slice_db_path(base_path_)},
            {"slice_record_pattern", base_path_ + ".slice.{slice_id}.json"},
            {"db_contract_path", rag_storage_db_contract_path(base_path_)},
            {"rocksdb_path", rag_storage_rocksdb_path(base_path_)},
            {"rocksdb_status_path", rag_storage_rocksdb_status_path(base_path_)},
            {"sqlite_path", rag_storage_sqlite_path(base_path_)},
            {"kv_journal_path", rag_storage_kv_journal_path(base_path_)},
            {"kv_snapshot_path", rag_storage_kv_snapshot_path(base_path_)},
            {"viewpoint_store_path", rag_storage_viewpoint_store_path(base_path_)},
            {"viewpoint_index_pattern", base_path_ + ".viewpoint.{viewpoint_id}.jsonl"},
            {"viewpoint_baseline_path", rag_storage_viewpoint_baseline_path(base_path_)},
            {"review_store_path", rag_storage_review_store_path(base_path_)},
            {"review_index_pattern", base_path_ + ".review_observation.{observation_id}.jsonl"},
            {"review_trace_index_pattern", base_path_ + ".review_trace.{trace_id}.jsonl"},
            {"review_bucket_index_pattern", base_path_ + ".review_bucket.{test_bucket}.jsonl"},
            {"coupling_graph_path", rag_storage_coupling_graph_path(base_path_)},
            {"coupling_slice_index_pattern", base_path_ + ".coupling_slice.{slice_id}.jsonl"},
            {"coupling_baseline_path", rag_storage_coupling_baseline_path(base_path_)},
            {"clips_fact_page_pattern", base_path_ + ".clips_fact_page.{run_kind}.{trace_id}.json"},
            {"knowledge_nodes_pattern", base_path_ + ".knowledge_nodes.{run_kind}.{trace_id}.json"},
            {"audit_log_path", base_path_ + ".audit.jsonl"},
            {"audit_by_request_pattern", base_path_ + ".audit_by_request.{request_id}.jsonl"},
            {"audit_by_slice_pattern", base_path_ + ".audit_by_slice.{slice_id}.jsonl"},
            {"audit_by_node_pattern", base_path_ + ".audit_by_node.{node_id}.jsonl"},
            {"trace_snapshot_pattern", base_path_ + ".trace.{trace_id}.json"},
        }},
    };

    const json manifest = {
        {"schema_version", 1},
        {"store_model", "rag_storage_contract_v1"},
        {"backend", "file_sidecar"},
        {"backend_contract_path", rag_storage_db_contract_path(base_path_)},
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
            {"slice_db", {
                {"path", rag_storage_slice_db_path(base_path_)},
                {"record_pattern", base_path_ + ".slice.{slice_id}.json"},
                {"record_model", "rag_slice_db_v1"},
                {"slice_record_model", "rag_slice_record_v1"},
                {"keys", json::array({
                    "slice:{slice_id}",
                    "slice:{slice_id}:raw",
                    "slice:{slice_id}:meta",
                    "slice:{slice_id}:hash",
                    "slice:{slice_id}:vector_id",
                    "slice:{slice_id}:source",
                })},
            }},
            {"database_backend_contract", {
                {"path", rag_storage_db_contract_path(base_path_)},
                {"record_model", "rag_storage_backend_contract_v1"},
                {"active_backend", "file_sidecar"},
                {"rocksdb_path", rag_storage_rocksdb_path(base_path_)},
                {"rocksdb_status_path", rag_storage_rocksdb_status_path(base_path_)},
                {"sqlite_path", rag_storage_sqlite_path(base_path_)},
                {"kv_journal_path", rag_storage_kv_journal_path(base_path_)},
                {"kv_snapshot_path", rag_storage_kv_snapshot_path(base_path_)},
                {"rocksdb_source_hint", "analysis_workspace/rocksdb-11.0.4"},
                {"status", "contract_ready"},
#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
                {"rocksdb_mirror_write_batch", true},
#else
                {"rocksdb_mirror_write_batch", false},
#endif
            }},
            {"backend_write_batch", {
                {"journal_path", rag_storage_kv_journal_path(base_path_)},
                {"snapshot_path", rag_storage_kv_snapshot_path(base_path_)},
                {"record_model", "rag_storage_backend_write_batch_v1"},
                {"snapshot_model", "rag_storage_backend_kv_snapshot_v1"},
                {"active_backend", "file_sidecar"},
                {"target_backends", json::array({"file_sidecar", "rocksdb", "sqlite"})},
#if defined(LLAMA_SERVER_RAG_ROCKSDB_BACKEND)
                {"rocksdb_mirror_write_batch", true},
#else
                {"rocksdb_mirror_write_batch", false},
#endif
            }},
            {"slice_coupling_graph", {
                {"path", rag_storage_coupling_graph_path(base_path_)},
                {"slice_index_pattern", base_path_ + ".coupling_slice.{slice_id}.jsonl"},
                {"baseline_path", rag_storage_coupling_baseline_path(base_path_)},
                {"record_model", "rag_slice_coupling_graph_v1"},
                {"edge_model", "rag_slice_edge_v1"},
                {"candidate_model", "rag_coupling_candidate_v1"},
                {"decision_model", "rag_coupling_decision_v1"},
                {"keys", json::array({
                    "semantic_slice:{slice_id}",
                    "slice_edge:{edge_id}",
                    "coupling_candidate:{candidate_id}",
                    "coupling_decision:{candidate_id}",
                    "coupling_by_slice:{slice_id}:{edge_id}",
                })},
            }},
            {"viewpoint_validation", {
                {"path", rag_storage_viewpoint_store_path(base_path_)},
                {"viewpoint_index_pattern", base_path_ + ".viewpoint.{viewpoint_id}.jsonl"},
                {"baseline_path", rag_storage_viewpoint_baseline_path(base_path_)},
                {"record_model", "rag_viewpoint_store_v1"},
                {"candidate_model", "rag_viewpoint_candidate_v1"},
                {"evidence_model", "rag_evidence_binding_v1"},
                {"decision_model", "rag_viewpoint_decision_v1"},
                {"keys", json::array({
                    "viewpoint:{viewpoint_id}",
                    "viewpoint:evidence:{viewpoint_id}:{source_id}",
                    "viewpoint:decision:{viewpoint_id}",
                    "viewpoint_by_slice:{slice_id}:{viewpoint_id}",
                    "viewpoint_by_trace:{trace_id}:{viewpoint_id}",
                })},
            }},
            {"review_observation", {
                {"path", rag_storage_review_store_path(base_path_)},
                {"review_index_pattern", base_path_ + ".review_observation.{observation_id}.jsonl"},
                {"trace_index_pattern", base_path_ + ".review_trace.{trace_id}.jsonl"},
                {"bucket_index_pattern", base_path_ + ".review_bucket.{test_bucket}.jsonl"},
                {"record_model", "rag_review_observation_v1"},
                {"keys", json::array({
                    "review:{observation_id}",
                    "review_by_trace:{trace_id}:{observation_id}",
                    "review_by_query:{query_id}:{observation_id}",
                    "review_by_bucket:{test_bucket}:{observation_id}",
                    "review_by_gap:{coverage_gap}:{observation_id}",
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
            {"clips_fact_page", {
                {"path_pattern", base_path_ + ".clips_fact_page.{run_kind}.{trace_id}.json"},
                {"record_model", "rag_clips_fact_page_v1"},
                {"keys", json::array({
                    "clips:fact:{trace_id}:{fact_id}",
                    "clips:run:{run_id}:input_facts",
                    "clips:run:{run_id}:output_facts",
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
            {"knowledge_node", {
                {"path_pattern", base_path_ + ".knowledge_nodes.{run_kind}.{trace_id}.json"},
                {"node_index_pattern", base_path_ + ".knowledge_node.{node_id}.jsonl"},
                {"slice_index_pattern", base_path_ + ".knowledge_slice.{slice_id}.jsonl"},
                {"record_model", "rag_knowledge_node_v1"},
                {"keys", json::array({
                    "node:{node_id}",
                    "node_by_slice:{slice_id}:{node_id}",
                    "relation:{relation_id}",
                    "risk:{risk_id}",
                })},
            }},
            {"audit_event", {
                {"path", base_path_ + ".audit.jsonl"},
                {"request_index_pattern", base_path_ + ".audit_by_request.{request_id}.jsonl"},
                {"slice_index_pattern", base_path_ + ".audit_by_slice.{slice_id}.jsonl"},
                {"node_index_pattern", base_path_ + ".audit_by_node.{node_id}.jsonl"},
                {"trace_snapshot_pattern", base_path_ + ".trace.{trace_id}.json"},
                {"record_model", "rag_audit_event_v1"},
                {"keys", json::array({
                    "audit:{trace_id}:{seq}",
                    "audit_by_request:{request_id}:{seq}",
                    "audit_by_slice:{slice_id}:{seq}",
                    "audit_by_node:{node_id}:{seq}",
                })},
            }},
            {"manual_validation_plan", {
                {"path", rag_storage_manual_test_plan_path(base_path_)},
                {"record_model", "rag_manual_validation_plan_v1"},
                {"purpose", "human-readable validation backup for RAG storage and CLIPS replay"},
            }},
        }},
        {"planned_stores", json::object()},
    };

    rag_storage_write_json_file(rag_storage_manifest_path(base_path_), manifest);
    rag_storage_write_json_file(rag_storage_manual_test_plan_path(base_path_), manual_test_plan);
    rag_storage_write_json_file(rag_storage_db_contract_path(base_path_), build_database_contract(base_path_));
    rag_storage_write_json_file(
        rag_storage_viewpoint_baseline_path(base_path_),
        json{
            {"schema_version", 1},
            {"record_model", "rag_viewpoint_validation_baseline_v1"},
            {"index_name", base_path_},
            {"rule_set_id", "rag-meta-core"},
            {"baseline", manual_test_plan["capability_baselines"]["viewpoint_validation"]},
        });
    rag_storage_write_json_file(
        rag_storage_coupling_baseline_path(base_path_),
        json{
            {"schema_version", 1},
            {"record_model", "rag_slice_coupling_baseline_v1"},
            {"index_name", base_path_},
            {"rule_set_id", "rag-meta-core"},
            {"baseline", manual_test_plan["capability_baselines"]["slice_coupling"]},
        });
    write_json_file_if_missing(
        rag_storage_coupling_graph_path(base_path_),
        json{
            {"schema_version", 1},
            {"record_model", "rag_slice_coupling_graph_v1"},
            {"index_name", base_path_},
            {"status", "contract_ready"},
            {"semantic_slices", json::object()},
            {"coupling_candidates", json::object()},
            {"coupling_decisions", json::object()},
            {"slice_edges", json::object()},
            {"edge_count", 0},
            {"candidate_count", 0},
        });
    write_json_file_if_missing(
        rag_storage_viewpoint_store_path(base_path_),
        json{
            {"schema_version", 1},
            {"record_model", "rag_viewpoint_store_v1"},
            {"index_name", base_path_},
            {"status", "contract_ready"},
            {"viewpoint_candidates", json::object()},
            {"evidence_bindings", json::object()},
            {"viewpoint_validations", json::object()},
            {"viewpoint_decisions", json::object()},
            {"viewpoint_count", 0},
        });
    write_json_file_if_missing(
        rag_storage_review_store_path(base_path_),
        json{
            {"schema_version", 1},
            {"record_model", "rag_review_observation_store_v1"},
            {"index_name", base_path_},
            {"status", "contract_ready"},
            {"review_observations", json::object()},
            {"by_test_bucket", json::object()},
            {"by_trace", json::object()},
            {"observation_count", 0},
        });
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

    json operations = json::array();
    for (size_t i = 0; i < map.vector_to_slice.size(); ++i) {
        const std::string & slice_id = map.vector_to_slice[i];
        if (slice_id.empty()) {
            continue;
        }
        operations.push_back(make_backend_put(
            "vector",
            "vector:" + base_path_ + ":" + std::to_string(i),
            slice_id,
            "rag_vector_slice_ref_v1"));
    }
    for (const auto & item : map.slice_to_vector) {
        operations.push_back(make_backend_put(
            "vector",
            "slice:" + item.first + ":vector_id",
            item.second,
            "rag_vector_slice_ref_v1"));
    }
    append_backend_write_batch(base_path_, "vector_slice_map", operations);
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

    json slice_db;
    slice_db["schema_version"] = 1;
    slice_db["store_model"] = "rag_slice_db_v1";
    slice_db["index_name"] = base_path_;
    slice_db["record_count"] = 0;
    slice_db["records"] = json::object();
    slice_db["slice_ids"] = json::array();

    json coupling_graph;
    coupling_graph["schema_version"] = 1;
    coupling_graph["record_model"] = "rag_slice_coupling_graph_v1";
    coupling_graph["index_name"] = base_path_;
    coupling_graph["status"] = "seeded_from_raw_slices";
    coupling_graph["semantic_slices"] = json::object();
    coupling_graph["coupling_candidates"] = json::object();
    coupling_graph["coupling_decisions"] = json::object();
    coupling_graph["slice_edges"] = json::object();
    coupling_graph["edge_count"] = 0;
    coupling_graph["candidate_count"] = 0;

    json operations = json::array();
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

        const std::string source_uri =
            metadata.is_object() ? metadata.value("source_uri", metadata.value("path", "")) : "";
        json slice_record = {
            {"schema_version", 1},
            {"record_model", "rag_slice_record_v1"},
            {"slice_id", record.slice_id},
            {"raw", record.raw_text},
            {"meta", metadata},
            {"hash", record.text_hash},
            {"vector_id", record.vector_id},
            {"source_uri", source_uri},
            {"keys", {
                {"slice", "slice:" + record.slice_id},
                {"raw", "slice:" + record.slice_id + ":raw"},
                {"meta", "slice:" + record.slice_id + ":meta"},
                {"hash", "slice:" + record.slice_id + ":hash"},
                {"vector_id", "slice:" + record.slice_id + ":vector_id"},
                {"source", "slice:" + record.slice_id + ":source"},
            }},
            {"index_refs", {
                {"vector_map_path", rag_storage_vector_map_path(base_path_)},
                {"raw_slices_path", rag_storage_raw_slices_path(base_path_)},
                {"slice_db_path", rag_storage_slice_db_path(base_path_)},
                {"knowledge_slice_index_path", rag_storage_knowledge_slice_index_path(base_path_, record.slice_id)},
                {"clips_slice_index_path", rag_storage_clips_slice_index_path(base_path_, record.slice_id)},
                {"audit_slice_index_path", base_path_ + ".audit_by_slice." +
                    rag_storage_sanitize_token(record.slice_id) + ".jsonl"},
            }},
        };

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

        slice_db["records"][record.slice_id] = slice_record;
        slice_db["slice_ids"].push_back(record.slice_id);
        slice_db["record_count"] = slice_db["slice_ids"].size();
        rag_storage_write_json_file(rag_storage_slice_record_path(base_path_, record.slice_id), slice_record);

        coupling_graph["semantic_slices"][record.slice_id] = json{
            {"record_model", "rag_semantic_slice_v1"},
            {"slice_id", record.slice_id},
            {"source_uri", source_uri},
            {"start_line", metadata.is_object() ? metadata.value("start_line", 0) : 0},
            {"end_line", metadata.is_object() ? metadata.value("end_line", 0) : 0},
            {"text_hash", record.text_hash},
            {"language", metadata.is_object() ? metadata.value("language", "") : ""},
            {"symbol_scope", metadata.is_object() ? metadata.value("symbol_scope", "") : ""},
            {"status", "VERIFIED"},
        };

        operations.push_back(make_backend_put(
            "slice",
            "slice:" + record.slice_id,
            slice_record,
            "rag_slice_record_v1"));
        operations.push_back(make_backend_put(
            "slice",
            "slice:" + record.slice_id + ":raw",
            record.raw_text,
            "rag_slice_raw_v1"));
        operations.push_back(make_backend_put(
            "slice",
            "slice:" + record.slice_id + ":meta",
            metadata,
            "rag_slice_meta_v1"));
        operations.push_back(make_backend_put(
            "slice",
            "slice:" + record.slice_id + ":hash",
            record.text_hash,
            "rag_slice_hash_v1"));
        operations.push_back(make_backend_put(
            "slice",
            "slice:" + record.slice_id + ":vector_id",
            record.vector_id,
            "rag_slice_vector_ref_v1"));
        operations.push_back(make_backend_put(
            "slice",
            "slice:" + record.slice_id + ":source",
            source_uri,
            "rag_slice_source_ref_v1"));
        operations.push_back(make_backend_put(
            "coupling",
            "semantic_slice:" + record.slice_id,
            json{
                {"record_model", "rag_semantic_slice_v1"},
                {"slice_id", record.slice_id},
                {"source_uri", source_uri},
                {"start_line", metadata.is_object() ? metadata.value("start_line", 0) : 0},
                {"end_line", metadata.is_object() ? metadata.value("end_line", 0) : 0},
                {"text_hash", record.text_hash},
                {"language", metadata.is_object() ? metadata.value("language", "") : ""},
                {"symbol_scope", metadata.is_object() ? metadata.value("symbol_scope", "") : ""},
                {"status", "VERIFIED"},
            },
            "rag_semantic_slice_v1"));
    }

    rag_storage_write_json_file(rag_storage_raw_slices_path(base_path_), raw_store);
    rag_storage_write_json_file(rag_storage_slice_db_path(base_path_), slice_db);
    rag_storage_write_json_file(rag_storage_coupling_graph_path(base_path_), coupling_graph);
    append_backend_write_batch(base_path_, "raw_slice_store", operations);
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
