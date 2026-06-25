#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>
#include <unordered_map>
#include <vector>

using rag_storage_json = nlohmann::json;

struct RagVectorSliceMap {
    std::vector<std::string> vector_to_slice;
    std::unordered_map<std::string, int> slice_to_vector;
};

struct RagRawSliceRecord {
    std::string slice_id;
    std::string raw_text;
    std::string metadata_json;
    std::string text_hash;
    int vector_id = -1;
};

struct RagRawSliceStore {
    std::unordered_map<std::string, RagRawSliceRecord> slices;
};

std::string rag_storage_sanitize_token(const std::string & value);

std::string rag_storage_vector_map_path(const std::string & base_path);
std::string rag_storage_raw_slices_path(const std::string & base_path);
std::string rag_storage_slice_db_path(const std::string & base_path);
std::string rag_storage_slice_record_path(const std::string & base_path, const std::string & slice_id);
std::string rag_storage_db_contract_path(const std::string & base_path);
std::string rag_storage_rocksdb_path(const std::string & base_path);
std::string rag_storage_rocksdb_status_path(const std::string & base_path);
std::string rag_storage_sqlite_path(const std::string & base_path);
std::string rag_storage_kv_journal_path(const std::string & base_path);
std::string rag_storage_kv_snapshot_path(const std::string & base_path);
std::string rag_storage_viewpoint_store_path(const std::string & base_path);
std::string rag_storage_viewpoint_index_path(const std::string & base_path, const std::string & viewpoint_id);
std::string rag_storage_viewpoint_baseline_path(const std::string & base_path);
std::string rag_storage_review_store_path(const std::string & base_path);
std::string rag_storage_review_index_path(const std::string & base_path, const std::string & observation_id);
std::string rag_storage_review_trace_index_path(const std::string & base_path, const std::string & trace_id);
std::string rag_storage_review_bucket_index_path(const std::string & base_path, const std::string & test_bucket);
std::string rag_storage_coupling_graph_path(const std::string & base_path);
std::string rag_storage_coupling_slice_index_path(const std::string & base_path, const std::string & slice_id);
std::string rag_storage_coupling_baseline_path(const std::string & base_path);
std::string rag_storage_manifest_path(const std::string & base_path);
std::string rag_storage_manual_test_plan_path(const std::string & base_path);

std::string rag_storage_clips_runs_log_path(const std::string & base_path);
std::string rag_storage_clips_run_snapshot_path(
    const std::string & base_path,
    const std::string & run_kind,
    const std::string & trace_id);
std::string rag_storage_clips_fact_page_path(
    const std::string & base_path,
    const std::string & run_kind,
    const std::string & trace_id);
std::string rag_storage_clips_query_index_path(
    const std::string & base_path,
    const std::string & query_id);
std::string rag_storage_clips_slice_index_path(
    const std::string & base_path,
    const std::string & slice_id);
std::string rag_storage_knowledge_nodes_path(
    const std::string & base_path,
    const std::string & run_kind,
    const std::string & trace_id);
std::string rag_storage_knowledge_node_index_path(
    const std::string & base_path,
    const std::string & node_id);
std::string rag_storage_knowledge_slice_index_path(
    const std::string & base_path,
    const std::string & slice_id);

void rag_storage_append_jsonl_record(const std::string & path, const rag_storage_json & record);
void rag_storage_write_json_file(const std::string & path, const rag_storage_json & value);
rag_storage_json rag_storage_make_backend_put(
    const std::string & column_family,
    const std::string & key,
    const rag_storage_json & value,
    const std::string & record_model);
rag_storage_json rag_storage_read_backend_value(
    const std::string & base_path,
    const std::string & column_family,
    const std::string & key);
rag_storage_json rag_storage_scan_backend_prefix(
    const std::string & base_path,
    const std::string & column_family,
    const std::string & key_prefix,
    int limit);
void rag_storage_append_backend_write_batch(
    const std::string & base_path,
    const std::string & batch_id,
    const rag_storage_json & operations);

class RagFileStorage {
public:
    explicit RagFileStorage(std::string base_path);

    const std::string & base_path() const;

    void save_manifest() const;

    void save_vector_slice_map(const RagVectorSliceMap & map) const;
    bool load_vector_slice_map(RagVectorSliceMap & map) const;

    void save_raw_slices(const RagRawSliceStore & store) const;
    bool load_raw_slices(RagRawSliceStore & store) const;

private:
    std::string base_path_;
};
