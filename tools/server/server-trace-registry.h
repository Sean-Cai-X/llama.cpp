#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>

using json = nlohmann::ordered_json;

namespace server_trace_registry {

void record_stage(
        const std::string & trace_id,
        const std::string & stage,
        const json & payload);

bool get_trace(
        const std::string & trace_id,
        json & payload);

bool get_evidence(
        const std::string & slice_id,
        json & payload);

} // namespace server_trace_registry
