#include "server-json-utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace {

static std::string trim_ascii(const std::string & value) {
    const size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

} // namespace

namespace server_json {

std::string get_string_or_empty(const json & value, const std::string & key) {
    if (!value.is_object() || !value.contains(key) || value.at(key).is_null()) {
        return "";
    }

    if (value.at(key).is_string()) {
        return value.at(key).get<std::string>();
    }

    return value.at(key).dump();
}

json get_array_or_empty(const json & value, const std::string & key) {
    if (!value.is_object() || !value.contains(key) || !value.at(key).is_array()) {
        return json::array();
    }

    return value.at(key);
}

bool get_bool_or_default(const json & value, const std::string & key, bool fallback) {
    if (!value.is_object() || !value.contains(key) || value.at(key).is_null()) {
        return fallback;
    }

    if (value.at(key).is_boolean()) {
        return value.at(key).get<bool>();
    }

    if (value.at(key).is_string()) {
        const std::string lowered = lower_ascii(trim_ascii(value.at(key).get<std::string>()));

        if (lowered == "true" ||
            lowered == "1" ||
            lowered == "yes" ||
            lowered == "on" ||
            lowered == "allow") {
            return true;
        }

        if (lowered == "false" ||
            lowered == "0" ||
            lowered == "no" ||
            lowered == "off" ||
            lowered == "prompt") {
            return false;
        }
    }

    return fallback;
}

std::string flatten_message_content(const json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }

    if (content.is_null()) {
        return "";
    }

    if (!content.is_array()) {
        return content.dump();
    }

    std::ostringstream oss;
    bool first = true;

    for (const auto & item : content) {
        std::string piece;

        if (item.is_string()) {
            piece = item.get<std::string>();
        } else if (item.is_object()) {
            piece = get_string_or_empty(item, "text");

            if (piece.empty()) {
                piece = get_string_or_empty(item, "content");
            }

            if (piece.empty()) {
                piece = item.dump();
            }
        } else {
            piece = item.dump();
        }

        if (piece.empty()) {
            continue;
        }

        if (!first) {
            oss << "\n";
        }

        first = false;
        oss << piece;
    }

    return oss.str();
}

json parse_direct_payload(const std::string & content) {
    try {
        return json::parse(content);
    } catch (...) {
    }

    const size_t fence_start = content.find("```");
    if (fence_start != std::string::npos) {
        const size_t line_end = content.find('\n', fence_start);
        const size_t fence_end = line_end == std::string::npos
            ? std::string::npos
            : content.find("```", line_end + 1);

        if (line_end != std::string::npos &&
            fence_end != std::string::npos &&
            fence_end > line_end) {
            const std::string fenced = content.substr(line_end + 1, fence_end - line_end - 1);

            try {
                return json::parse(fenced);
            } catch (...) {
            }
        }
    }

    const size_t brace_start = content.find('{');
    const size_t brace_end = content.rfind('}');

    if (brace_start != std::string::npos &&
        brace_end != std::string::npos &&
        brace_end > brace_start) {
        const std::string candidate = content.substr(brace_start, brace_end - brace_start + 1);

        try {
            return json::parse(candidate);
        } catch (...) {
        }
    }

    return json::object();
}

} // namespace server_json