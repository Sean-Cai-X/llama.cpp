#pragma once

#include "server-json-utils.h"

#include <optional>
#include <string>

namespace server_remote_session_turn {

using json = server_json::json;

json default_tool_availability_snapshot();

json build_messages_from_session(
        const std::optional<json> & existing_session,
        const json & body,
        bool append_mode);

json build_ventriloquy_result(
        const json & response_json,
        const std::string & session_id,
        const std::string & turn_id,
        const std::string & write_mode);

} // namespace server_remote_session_turn