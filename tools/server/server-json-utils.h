#pragma once

#include "server-common.h"

#include <string>

namespace server_json {

using json = nlohmann::ordered_json;

std::string get_string_or_empty(const json & value, const std::string & key);

json get_array_or_empty(const json & value, const std::string & key);

bool get_bool_or_default(
        const json & value,
        const std::string & key,
        bool fallback = false);

std::string flatten_message_content(const json & content);

json parse_direct_payload(const std::string & content);

} // namespace server_json