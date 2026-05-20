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

    json enrich_session_projection(const json & session) const;

private:
    std::string root_dir;
    std::string slices_dir;
    mutable std::mutex mutex;

    std::string session_path(const std::string & session_id) const;
    std::string slice_path(const std::string & session_id, const std::string & turn_id) const;
    std::string canonical_index_path() const;
    json build_empty_session(const std::string & session_id) const;
    json load_session_unlocked(const std::string & session_id) const;
    void save_session_unlocked(const std::string & session_id, const json & session) const;
    json load_canonical_index_unlocked() const;
    void save_canonical_index_unlocked(const json & index) const;
    json build_slice_unlocked(const std::string & session_id, const json & turn) const;
    void save_slice_unlocked(const std::string & session_id, const std::string & turn_id, const json & slice) const;
};
