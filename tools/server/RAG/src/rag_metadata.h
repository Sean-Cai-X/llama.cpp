#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>

using rag_json = nlohmann::ordered_json;

std::string rag_make_stable_id(const std::string & prefix, const std::string & seed);

rag_json rag_parse_metadata(const std::string & metadata);
std::string rag_metadata_to_string(const rag_json & metadata);

std::string rag_metadata_value_string(const rag_json & metadata, const std::string & key);
int rag_metadata_value_int(const rag_json & metadata, const std::string & key, int fallback = 0);
bool rag_metadata_value_bool(const rag_json & metadata, const std::string & key, bool fallback = false);
bool rag_metadata_contains_flag(const std::string & metadata, const std::string & flag);

rag_json rag_build_file_metadata(
    const std::string & source_uri,
    const std::string & language,
    size_t file_size);

rag_json rag_build_chunk_metadata(
    const rag_json & base_metadata,
    int start_line,
    int end_line,
    int chunk_index,
    int chunk_count,
    const std::string & text);

rag_json rag_normalize_ingest_metadata(
    const std::string & metadata,
    const std::string & chunk_text,
    int fallback_doc_id,
    int chunk_index,
    int chunk_count);
