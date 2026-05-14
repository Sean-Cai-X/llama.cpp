#include "server-trace-registry.h"

#include "common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <list>
#include <mutex>
#include <unordered_map>

namespace {

struct trace_record {
    json payload = json::object();
};

std::mutex g_trace_registry_mutex;
std::unordered_map<std::string, trace_record> g_trace_registry;
std::list<std::string> g_trace_order;
constexpr size_t TRACE_REGISTRY_LIMIT = 128;

} // namespace

namespace server_trace_registry {

void record_stage(
        const std::string & trace_id,
        const std::string & stage,
        const json & payload) {
    if (trace_id.empty() || stage.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);

    auto it = g_trace_registry.find(trace_id);
    if (it == g_trace_registry.end()) {
        g_trace_order.push_back(trace_id);
        it = g_trace_registry.emplace(trace_id, trace_record {}).first;
        it->second.payload["trace_id"] = trace_id;
        it->second.payload["record_model"] = "rag_trace_debug_v1";
        it->second.payload["stages"] = json::object();
    } else {
        g_trace_order.remove(trace_id);
        g_trace_order.push_back(trace_id);
    }

    json & root = it->second.payload;
    root["latest_stage"] = stage;
    root["stages"][stage] = payload;

    for (const auto & key : {"request_id", "query_id", "model_task_id"}) {
        if (payload.contains(key) && !payload.at(key).is_null()) {
            root[key] = payload.at(key);
        }
    }
    for (const auto & key : {"approved_context", "supporting_slice_ids", "admission_summary", "llama_output_candidate", "llama_output_validation", "final_output", "structured_conclusion"}) {
        if (payload.contains(key) && !payload.at(key).is_null()) {
            root[key] = payload.at(key);
        }
    }

    while (g_trace_order.size() > TRACE_REGISTRY_LIMIT) {
        const std::string evict_id = g_trace_order.front();
        g_trace_order.pop_front();
        g_trace_registry.erase(evict_id);
    }
}

bool get_trace(
        const std::string & trace_id,
        json & payload) {
    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
    const auto it = g_trace_registry.find(trace_id);
    if (it == g_trace_registry.end()) {
        return false;
    }

    payload = it->second.payload;
    return true;
}

bool get_evidence(
        const std::string & slice_id,
        json & payload) {
    std::lock_guard<std::mutex> lock(g_trace_registry_mutex);
    if (slice_id.empty()) {
        return false;
    }

    json traces = json::array();
    json approved_context_item = json::object();
    json final_output = json::object();
    json output_validation = json::object();
    json admission_summary = json::object();

    for (auto it = g_trace_order.rbegin(); it != g_trace_order.rend(); ++it) {
        const auto record_it = g_trace_registry.find(*it);
        if (record_it == g_trace_registry.end()) {
            continue;
        }

        const json & root = record_it->second.payload;
        const json approved_context = root.value("approved_context", json::array());
        bool matched = false;

        if (approved_context.is_array()) {
            for (const auto & item : approved_context) {
                if (!item.is_object()) {
                    continue;
                }
                const std::string item_slice_id = item.value("slice_id", item.value("source_id", ""));
                if (item_slice_id != slice_id) {
                    continue;
                }

                if (approved_context_item.empty()) {
                    approved_context_item = item;
                    final_output = root.value("final_output", json::object());
                    output_validation = root.value("llama_output_validation", json::object());
                    admission_summary = root.value("admission_summary", json::object());
                }
                matched = true;
                break;
            }
        }

        if (matched) {
            traces.push_back(json{
                {"trace_id", root.value("trace_id", "")},
                {"request_id", root.value("request_id", "")},
                {"query_id", root.value("query_id", "")},
                {"latest_stage", root.value("latest_stage", "")},
            });
        }
    }

    if (approved_context_item.empty()) {
        return false;
    }

    payload = json{
        {"record_model", "rag_evidence_debug_v1"},
        {"slice_id", slice_id},
        {"approved_context", approved_context_item},
        {"admission_summary", admission_summary},
        {"llama_output_validation", output_validation},
        {"final_output", final_output},
        {"trace_refs", traces},
    };
    return true;
}

} // namespace server_trace_registry
