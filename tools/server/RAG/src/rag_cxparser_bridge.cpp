#include "rag_cxparser_bridge.h"

#include "cxparser_ext/pipeline/parser_task_types.h"
#include "cxparser_ext/pipeline/parser_unified_entry.h"

namespace {

std::string task_status_name(cxparser_ext::ParserTaskStatus status) {
    switch (status) {
    case cxparser_ext::pts_created:   return "created";
    case cxparser_ext::pts_prepared:  return "prepared";
    case cxparser_ext::pts_running:   return "running";
    case cxparser_ext::pts_executed:  return "executed";
    case cxparser_ext::pts_validated: return "validated";
    case cxparser_ext::pts_failed:    return "failed";
    }
    return "unknown";
}

json to_json(const cxparser_ext::ParserMainThreadTick & tick) {
    return {
        {"thread_name", tick.thread_name},
        {"thread_role", tick.thread_role},
        {"cycle_index", tick.cycle_index},
        {"accepted_task_count", tick.accepted_task_count},
        {"executed_task_count", tick.executed_task_count},
        {"success", tick.success},
    };
}

json to_json(const cxparser_ext::TaskChainRecord & record) {
    return {
        {"task_id", record.task_id},
        {"task_type", record.task_type},
        {"task_subtype", record.task_subtype},
        {"route", record.route},
        {"execution_mode", record.execution_mode},
        {"modules", record.modules},
    };
}

json to_json(const cxparser_ext::ParserEvidenceBundle & evidence) {
    json trace_entries = json::array();
    for (const auto & item : evidence.trace_entries) {
        trace_entries.push_back({
            {"sequence", item.sequence},
            {"trace_id", item.trace_id},
            {"stage", item.stage},
            {"action", item.action},
            {"status", item.status},
            {"detail", item.detail},
        });
    }

    json log_entries = json::array();
    for (const auto & item : evidence.log_entries) {
        log_entries.push_back({
            {"trace_id", item.trace_id},
            {"level", item.level},
            {"stage", item.stage},
            {"code", item.code},
            {"message", item.message},
        });
    }

    json events = json::array();
    for (const auto & item : evidence.events) {
        events.push_back({
            {"stage", item.stage},
            {"code", item.code},
            {"message", item.message},
            {"expected", item.expected},
            {"actual", item.actual},
        });
    }

    return {
        {"task_id", evidence.task_id},
        {"trace_id", evidence.trace_id},
        {"route_key", evidence.route_key},
        {"route_lane", evidence.route_lane},
        {"protocol_name", evidence.protocol_name},
        {"task_type", evidence.task_type},
        {"task_subtype", evidence.task_subtype},
        {"execution_mode", evidence.execution_mode},
        {"module_chain", evidence.module_chain},
        {"events", events},
        {"trace_entries", trace_entries},
        {"log_entries", log_entries},
        {"notes", evidence.notes},
    };
}

json to_json(const cxparser_ext::ParserValidationReport & report) {
    json issues = json::array();
    for (const auto & item : report.issues) {
        issues.push_back({
            {"code", item.code},
            {"message", item.message},
            {"failure_stage", item.failure_stage},
            {"expected", item.expected},
            {"actual", item.actual},
        });
    }

    return {
        {"passed", report.passed},
        {"issues", issues},
        {"matched_points", report.matched_points},
        {"mismatched_points", report.mismatched_points},
    };
}

json build_trace_summary(const json & evidence_json) {
    const json trace_entries = evidence_json.value("trace_entries", json::array());
    const json log_entries = evidence_json.value("log_entries", json::array());
    const json events = evidence_json.value("events", json::array());

    std::string final_stage;
    std::string final_status;
    if (!trace_entries.empty()) {
        const json & last = trace_entries.back();
        final_stage = last.value("stage", "");
        final_status = last.value("status", "");
    }

    return {
        {"trace_count", (int) trace_entries.size()},
        {"log_count", (int) log_entries.size()},
        {"event_count", (int) events.size()},
        {"final_stage", final_stage},
        {"final_status", final_status},
    };
}

json build_validation_summary(const json & validation_json) {
    const json issues = validation_json.value("issues", json::array());
    int error_count = 0;
    int warning_count = 0;
    for (const auto & issue : issues) {
        const std::string code = issue.value("code", "");
        if (code.find("failed") != std::string::npos || code == "execution_failed") {
            ++error_count;
        } else {
            ++warning_count;
        }
    }

    return {
        {"passed", validation_json.value("passed", false)},
        {"issue_count", (int) issues.size()},
        {"error_count", error_count},
        {"warning_count", warning_count},
        {"matched_count", (int) validation_json.value("matched_points", json::array()).size()},
        {"mismatched_count", (int) validation_json.value("mismatched_points", json::array()).size()},
    };
}

std::string build_safe_script_text(int rank) {
    return std::to_string(rank);
}

cxparser_ext::CxTaskEnvelope build_envelope(
    const std::string & query,
    const json & metadata,
    int rank,
    const std::string & preview_text) {
    const json target = build_rag_execution_target(query, metadata, rank, preview_text);
    const json source_info = build_rag_source_info(metadata);

    cxparser_ext::CxTaskEnvelope envelope;
    envelope.task_id = target.value("task_id", "rag_explain_" + std::to_string(rank));
    envelope.task_name = target.value("task_name", envelope.task_id);
    envelope.trace_id = target.value("trace_id", envelope.task_id + ".trace");
    envelope.task_type = "rag";
    envelope.task_subtype = "explain";
    envelope.route = target.value("route_hint", "default");
    envelope.execution_mode = "dry_run";
    envelope.caller_module = "llama.server.rag";
    envelope.callee_module = target.value("module_name", "cxparser");
    envelope.target_class = target.value("target_class", "script");
    envelope.target_method = target.value("target_method", "eval");
    envelope.script_text = build_safe_script_text(rank);
    envelope.tags = {
        "rag",
        "preview:" + preview_text,
        "source:" + source_info.value("label", ""),
        "query:" + query,
    };
    return envelope;
}

} // namespace

json build_rag_cxparser_dry_run(
    const std::string & query,
    const json & metadata,
    int rank,
    const std::string & preview_text) {
    cxparser_ext::ParserUnifiedEntry entry;
    const cxparser_ext::CxTaskEnvelope envelope = build_envelope(query, metadata, rank, preview_text);

    const bool submitted = entry.SubmitEnvelope(envelope);
    const bool executed = submitted && entry.ExecuteMainThreadCycle();
    const cxparser_ext::ParserMainThreadTick tick = entry.GetLastTick();
    const cxparser_ext::ParserTaskUnit * task = entry.FindTask(envelope.task_id);
    const cxparser_ext::ParserEvidenceBundle * evidence = entry.FindTaskEvidence(envelope.task_id);

    json chain_records = json::array();
    for (const auto & record : entry.GetChainRecords()) {
        chain_records.push_back(to_json(record));
    }

    json result = {
        {"submitted", submitted},
        {"executed", executed},
        {"tick", to_json(tick)},
        {"chain_records", chain_records},
        {"envelope", {
            {"task_id", envelope.task_id},
            {"task_name", envelope.task_name},
            {"trace_id", envelope.trace_id},
            {"task_type", envelope.task_type},
            {"task_subtype", envelope.task_subtype},
            {"route", envelope.route},
            {"execution_mode", envelope.execution_mode},
            {"caller_module", envelope.caller_module},
            {"callee_module", envelope.callee_module},
            {"target_class", envelope.target_class},
            {"target_method", envelope.target_method},
            {"script_text", envelope.script_text},
            {"tags", envelope.tags},
        }},
    };

    if (task) {
        json validation_json = to_json(task->report);
        result["task"] = {
            {"task_id", task->task_id},
            {"status", task_status_name(task->status)},
            {"failure_reason", task->failure_reason},
            {"result", {
                {"success", task->result.success},
                {"scalar_result", task->result.scalar_result},
                {"text_result", task->result.text_result},
                {"error_message", task->result.error_message},
                {"warnings", task->result.warnings},
            }},
            {"validation", validation_json},
            {"validation_summary", build_validation_summary(validation_json)},
        };
    }

    if (evidence) {
        json evidence_json = to_json(*evidence);
        if (evidence_json.value("task_type", "").empty()) {
            evidence_json["task_type"] = envelope.task_type;
        }
        if (evidence_json.value("task_subtype", "").empty()) {
            evidence_json["task_subtype"] = envelope.task_subtype;
        }
        if (evidence_json.value("execution_mode", "").empty()) {
            evidence_json["execution_mode"] = envelope.execution_mode;
        }
        if (evidence_json.value("module_chain", json::array()).empty()) {
            evidence_json["module_chain"] = json::array({envelope.caller_module, envelope.callee_module});
        }
        result["evidence"] = std::move(evidence_json);
        result["trace_summary"] = build_trace_summary(result["evidence"]);
    }

    result["summary"] = {
        {"task_id", envelope.task_id},
        {"submitted", submitted},
        {"executed", executed},
        {"route", envelope.route},
        {"execution_mode", envelope.execution_mode},
        {"status", task ? task_status_name(task->status) : "missing"},
        {"success", task ? task->result.success : false},
    };

    return result;
}
