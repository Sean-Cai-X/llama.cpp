#pragma once

#include "vendor/nlohmann/json.hpp"

#include <string>

using json = nlohmann::ordered_json;

json build_rag_source_info(const json & metadata);
json build_rag_execution_target(
    const std::string & query,
    const json & metadata,
    int rank,
    const std::string & preview_text);
json build_rag_evidence_stub(const std::string & query, const json & metadata, int rank);
