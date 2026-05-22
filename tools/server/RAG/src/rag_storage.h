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
std::string rag_storage_manifest_path(const std::string & base_path);

std::string rag_storage_clips_runs_log_path(const std::string & base_path);
std::string rag_storage_clips_run_snapshot_path(
    const std::string & base_path,
    const std::string & run_kind,
    const std::string & trace_id);
std::string rag_storage_clips_query_index_path(
    const std::string & base_path,
    const std::string & query_id);
std::string rag_storage_clips_slice_index_path(
    const std::string & base_path,
    const std::string & slice_id);

void rag_storage_append_jsonl_record(const std::string & path, const rag_storage_json & record);
void rag_storage_write_json_file(const std::string & path, const rag_storage_json & value);

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
