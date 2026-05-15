#include "server-supervision.h"

json build_alarm_event(
        const std::string & level,
        const std::string & alarm_code,
        const std::string & alarm_message) {
    return json{
        {"level", level},
        {"reason", alarm_code},
        {"alarm_code", alarm_code},
        {"alarm_message", alarm_message},
    };
}

json to_json(const server_supervision_envelope & envelope) {
    return json{
        {"supervision_status", envelope.supervision_status},
        {"supervision_state", envelope.supervision_status},
        {"execution_disposition", envelope.execution_disposition},
        {"response_allowed", envelope.response_allowed},
        {"assistant_response_allowed", envelope.response_allowed},
        {"model_response_allowed", envelope.response_allowed},
        {"failure_mode", envelope.failure_mode},
        {"alarm_code", envelope.alarm_code},
        {"alarm_message", envelope.alarm_message},
        {"reason", envelope.reason},
        {"route", envelope.route},
        {"dominant_decision", envelope.dominant_decision},
        {"approved_context_count", envelope.approved_context_count},
    };
}
