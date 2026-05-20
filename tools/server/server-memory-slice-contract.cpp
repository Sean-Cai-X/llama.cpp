#include "server-memory-slice-contract.h"

#include "server-json-utils.h"

namespace server_memory_slice_contract {

using server_json::get_string_or_empty;

std::string default_audit_ref(const std::string & session_id, const std::string & turn_id) {
    return "session:" + session_id + "/turn:" + turn_id;
}

void append_contract_fields(
    json & target,
    const json & source,
    const std::string & fallback_audit_ref) {
    const std::string slice_version = get_string_or_empty(source, "slice_version").empty()
        ? get_string_or_empty(source, "record_model")
        : get_string_or_empty(source, "slice_version");
    const std::string audit_ref = get_string_or_empty(source, "audit_ref").empty()
        ? fallback_audit_ref
        : get_string_or_empty(source, "audit_ref");
    const std::string dedup_status = get_string_or_empty(source, "dedup_status").empty()
        ? get_string_or_empty(source, "canonical_status")
        : get_string_or_empty(source, "dedup_status");

    target["provider_id"] = get_string_or_empty(source, "provider_id");
    target["capability_id"] = get_string_or_empty(source, "capability_id");
    target["slice_id"] = get_string_or_empty(source, "slice_id");
    target["slice_type"] = get_string_or_empty(source, "slice_type");
    target["slice_version"] = slice_version;
    target["audit_ref"] = audit_ref;
    target["slice_path"] = get_string_or_empty(source, "slice_path");
    target["slice_summary"] = get_string_or_empty(source, "slice_summary");
    target["dedup_key"] = get_string_or_empty(source, "dedup_key");
    target["dedup_hash"] = get_string_or_empty(source, "dedup_hash");
    target["canonical_slice_id"] = get_string_or_empty(source, "canonical_slice_id");
    target["dedup_status"] = dedup_status;
    target["dedup_reason"] = get_string_or_empty(source, "dedup_reason");
    target["dup_of"] = get_string_or_empty(source, "dup_of");
    target["canonical_status"] = get_string_or_empty(source, "canonical_status");
    target["slice_refs"] = source.contains("slice_refs") ? source["slice_refs"] : json::array();
    target["storage_refs"] = source.contains("storage_refs") ? source["storage_refs"] : json::array();
}

} // namespace server_memory_slice_contract
