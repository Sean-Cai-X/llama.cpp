#pragma once

#include <nlohmann/json.hpp>

namespace server_context_support {

using json = nlohmann::ordered_json;

json build_rag_supporting_slice_ids(const json & approved_context);
bool remove_trailing_empty_assistant_prefill(json & body);
json find_tool_continuation_payload(const json & body);
void mark_closed_loop_complete(const json & payload);

} // namespace server_context_support
