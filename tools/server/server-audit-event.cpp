#include "server-audit-event.h"

json to_json(const server_audit_event & event) {
    return json{
        {"trace_id", event.trace_id},
        {"stage", event.stage},
        {"goal_id", event.goal_id},
        {"tool_call_id", event.tool_call_id},
        {"payload", event.payload},
    };
}
