#pragma once

#include "server-common.h"

#include <string>

struct server_supervision_envelope {
    std::string supervision_status = "UNSUPERVISED";
    std::string execution_disposition = "FINALIZE_ANSWER";
    bool response_allowed = true;
    std::string failure_mode;
    std::string alarm_code;
    std::string alarm_message;
    std::string reason;
    std::string route;
    std::string dominant_decision;
    int approved_context_count = 0;
};

json build_alarm_event(
    const std::string & level,
    const std::string & alarm_code,
    const std::string & alarm_message);

json to_json(const server_supervision_envelope & envelope);
