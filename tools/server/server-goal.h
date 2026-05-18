#pragma once

#include "server-common.h"
#include "server-supervision.h"

#include <string>
#include <vector>

struct server_goal_envelope {
    std::string goal_id;
    std::string request_id;
    std::string trace_id;
    std::string goal_type;
    std::string root_path;
    std::string status = "RUNNING";
    std::vector<std::string> include_patterns;
    std::vector<std::string> exclude_patterns;
    int read_chunk_lines = 500;
    json test_acceptance_snapshot = nullptr;
};

struct server_goal_progress {
    int target_count = 0;
    int completed_count = 0;
    int failed_count = 0;
    int pending_count = 0;
    int skipped_count = 0;
};

struct server_goal_acceptance {
    std::string goal_id;
    std::string acceptance_status = "NOT_COMPLETE";
    std::string final_status = "IN_PROGRESS";
    std::string failure_mode;
    server_supervision_envelope supervision;
    server_goal_progress progress;
    json next_actions = json::array();
    json alarms = json::array();
};

server_goal_envelope parse_server_goal_envelope(const json & body);
json to_json(const server_goal_envelope & goal);
json to_json(const server_goal_progress & progress);
json to_json(const server_goal_acceptance & acceptance);
