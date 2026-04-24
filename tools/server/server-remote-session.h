#pragma once

#include <nlohmann/json.hpp>

#include <mutex>
#include <optional>
#include <string>

using json = nlohmann::ordered_json;

class remote_session_store {
public:
    remote_session_store();

    std::string create_session_id() const;
    std::string create_turn_id() const;

    json list_sessions(int limit = 50) const;
    std::optional<json> get_session(const std::string & session_id) const;

    json upsert_turn(
        const std::string & session_id,
        const json & metadata,
        const json & request_payload,
        const json & response_payload);

private:
    std::string root_dir;
    mutable std::mutex mutex;

    std::string session_path(const std::string & session_id) const;
    json build_empty_session(const std::string & session_id) const;
    json load_session_unlocked(const std::string & session_id) const;
    void save_session_unlocked(const std::string & session_id, const json & session) const;
};
