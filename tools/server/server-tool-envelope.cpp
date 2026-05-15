#include "server-tool-envelope.h"

#include <sstream>

namespace {

uint64_t fnv1a64(const std::string & text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

std::string build_server_result_hash(const json & result) {
    std::ostringstream oss;
    oss << std::hex << fnv1a64(result.dump());
    return "fnv1a64:" + oss.str();
}

json to_json(const server_tool_call_envelope & envelope) {
    return json{
        {"tool_call_id", envelope.tool_call_id},
        {"goal_id", envelope.goal_id},
        {"request_id", envelope.request_id},
        {"trace_id", envelope.trace_id},
        {"tool_name", envelope.tool_name},
        {"params", envelope.params},
        {"safety_class", envelope.safety_class},
        {"delivery_status", envelope.delivery_status},
        {"result_hash", envelope.result_hash},
    };
}
