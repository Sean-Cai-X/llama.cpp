#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace server_memory_slice_contract {

using json = nlohmann::ordered_json;

constexpr const char * k_slice_version = "rag_memory_slice_v1";

std::string default_audit_ref(const std::string & session_id, const std::string & turn_id);

void append_contract_fields(
    json & target,
    const json & source,
    const std::string & fallback_audit_ref);

} // namespace server_memory_slice_contract
