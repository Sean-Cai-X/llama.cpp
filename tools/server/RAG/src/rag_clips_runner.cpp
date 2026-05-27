#include "rag_clips_runner.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#if defined(LLAMA_SERVER_RAG_CLIPS_CORE_LINKED)
extern "C" {
#include "clips.h"
#include "agenda.h"
#include "constrct.h"
#include "cstrcpsr.h"
#include "engine.h"
#include "envrnbld.h"
#include "factfun.h"
#include "factmngr.h"
#include "memalloc.h"
#include "tmpltdef.h"
#include "utility.h"
}
#endif

namespace fs = std::filesystem;

namespace {

json load_manifest_json(const fs::path & manifest_path) {
    std::ifstream input(manifest_path, std::ios::in | std::ios::binary);
    if (!input) {
        return json::object();
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (buffer.str().empty()) {
        return json::object();
    }

    try {
        return json::parse(buffer.str());
    } catch (...) {
        return json::object();
    }
}

std::vector<RagClipsRuleManifestEntry> parse_rule_sets(const json & manifest) {
    std::vector<RagClipsRuleManifestEntry> entries;
    if (!manifest.is_object() || !manifest.contains("rule_sets") || !manifest["rule_sets"].is_array()) {
        return entries;
    }

    for (const auto & item : manifest["rule_sets"]) {
        if (!item.is_object()) {
            continue;
        }

        RagClipsRuleManifestEntry entry;
        entry.rule_set_id = item.value("id", "");
        entry.description = item.value("description", "");
        entry.mcp_visible = item.value("mcp_visible", true);
        if (item.contains("rule_files") && item["rule_files"].is_array()) {
            for (const auto & file_item : item["rule_files"]) {
                if (file_item.is_string()) {
                    entry.rule_files.push_back(file_item.get<std::string>());
                }
            }
        }

        if (!entry.rule_set_id.empty()) {
            entries.push_back(std::move(entry));
        }
    }

    return entries;
}

json rule_sets_to_json(const std::vector<RagClipsRuleManifestEntry> & entries) {
    json result = json::array();
    for (const auto & entry : entries) {
        json files = json::array();
        for (const auto & file : entry.rule_files) {
            files.push_back(file);
        }
        result.push_back(json{
            {"id", entry.rule_set_id},
            {"description", entry.description},
            {"rule_files", std::move(files)},
            {"mcp_visible", entry.mcp_visible},
        });
    }
    return result;
}

const RagClipsRuleManifestEntry * find_rule_set(
    const std::vector<RagClipsRuleManifestEntry> & entries,
    const std::string & rule_set_id) {
    for (const auto & entry : entries) {
        if (entry.rule_set_id == rule_set_id) {
            return &entry;
        }
    }
    return nullptr;
}

std::string extract_slot_value(const std::string & fact_text, const std::string & slot_name) {
    const std::regex pattern("\\(" + slot_name + " \"?([^\\)\"\\n]+)\"?\\)");
    std::smatch match;
    if (std::regex_search(fact_text, match, pattern) && match.size() >= 2) {
        return match[1].str();
    }
    return "";
}

void append_admission_summary(RagClipsRunResult & result, const std::string & fact_text) {
    if (fact_text.find("(admission-decision") == std::string::npos) {
        return;
    }

    result.admission_decisions.push_back(fact_text);
    RagClipsAdmissionDecision item;
    item.slice_id = extract_slot_value(fact_text, "slice-id");
    item.decision = extract_slot_value(fact_text, "decision");
    item.reason = extract_slot_value(fact_text, "reason");
    item.rule_id = extract_slot_value(fact_text, "rule-id");
    item.next_action = extract_slot_value(fact_text, "next-action");
    item.raw_fact = fact_text;
    const std::string decision = item.decision;
    result.admission_report.push_back(std::move(item));
    if (decision == "allow") {
        result.allow_count++;
    } else if (decision == "reject") {
        result.reject_count++;
    } else if (decision == "repair") {
        result.repair_count++;
    } else if (decision == "short_term_only") {
        result.short_term_only_count++;
    } else if (decision == "require_human") {
        result.require_human_count++;
    }
}

void append_runtime_admission_decision(
    RagClipsRunResult & result,
    const std::string & slice_id,
    const std::string & decision,
    const std::string & reason,
    const std::string & rule_id,
    const std::string & next_action,
    const std::string & raw_fact) {
    RagClipsAdmissionDecision item;
    item.slice_id = slice_id;
    item.decision = decision;
    item.reason = reason;
    item.rule_id = rule_id;
    item.next_action = next_action;
    item.raw_fact = raw_fact;
    result.admission_report.push_back(item);
    result.admission_decisions.push_back(raw_fact);

    if (decision == "allow") {
        result.allow_count++;
    } else if (decision == "reject") {
        result.reject_count++;
    } else if (decision == "repair") {
        result.repair_count++;
    } else if (decision == "short_term_only") {
        result.short_term_only_count++;
    } else if (decision == "require_human") {
        result.require_human_count++;
    }
}

void finalize_admission_summary(RagClipsRunResult & result) {
    if (result.reject_count > 0) {
        result.dominant_decision = "reject";
    } else if (result.require_human_count > 0) {
        result.dominant_decision = "require_human";
    } else if (result.repair_count > 0) {
        result.dominant_decision = "repair";
    } else if (result.short_term_only_count > 0) {
        result.dominant_decision = "short_term_only";
    } else if (result.allow_count > 0) {
        result.dominant_decision = "allow";
    } else {
        result.dominant_decision = "none";
    }
}

#if defined(LLAMA_SERVER_RAG_CLIPS_CORE_LINKED)
std::string fact_to_string(Environment * env, Fact * fact) {
    StringBuilder * builder = CreateStringBuilder(env, 256);
    if (builder == nullptr) {
        return "";
    }

    FactPPForm(fact, builder, true);
    std::string out = builder->contents ? builder->contents : "";
    SBDispose(builder);
    return out;
}

bool should_capture_fact_text(Fact * fact) {
    if (fact == nullptr) {
        return false;
    }

    Deftemplate * deftemplate = FactDeftemplate(fact);
    if (deftemplate == nullptr) {
        return false;
    }

    const char * template_name = DeftemplateName(deftemplate);
    if (template_name == nullptr) {
        return false;
    }

    return
        (std::strcmp(template_name, "admission-decision") == 0) ||
        (std::strcmp(template_name, "slice-requires-repair") == 0) ||
        (std::strcmp(template_name, "node-has-slice") == 0) ||
        (std::strcmp(template_name, "graph-governed") == 0) ||
        (std::strcmp(template_name, "semantic-slice") == 0) ||
        (std::strcmp(template_name, "slice-edge") == 0) ||
        (std::strcmp(template_name, "coupling-decision") == 0) ||
        (std::strcmp(template_name, "viewpoint-decision") == 0) ||
        (std::strcmp(template_name, "viewpoint-validation") == 0) ||
        (std::strcmp(template_name, "query-context") == 0);
}
#endif

} // namespace

std::string ResolveRagClipsRulesDir(const common_params & params) {
    if (!params.rag_clips_rules_dir.empty()) {
        return params.rag_clips_rules_dir;
    }

    if (!params.public_path.empty()) {
        fs::path public_path(params.public_path);
        return (public_path.parent_path() / "RAG" / "rules").string();
    }

    return "tools/server/RAG/rules";
}

json BuildRagClipsManifestPayload(
    const common_params & params,
    const RagClipsFactBundle & bundle,
    const std::vector<std::string> & assertions) {
    const std::string rules_dir = ResolveRagClipsRulesDir(params);
    const fs::path manifest_path = fs::path(rules_dir) / "manifest.json";
    const json manifest = load_manifest_json(manifest_path);
    const auto entries = parse_rule_sets(manifest);

    return json{
        {"backend", "manifest-only"},
        {"clips_enabled", params.rag_clips_enable},
        {"rules_dir", rules_dir},
        {"manifest_path", manifest_path.string()},
        {"rule_set_id", params.rag_clips_rule_set},
        {"available_rule_sets", rule_sets_to_json(entries)},
        {"engine_config", {
            {"storage_backend", "memory"},
            {"rocksdb_enabled", false},
            {"memory_pool_mb", params.rag_clips_memory_pool_mb},
            {"batch_fact_limit", params.rag_clips_batch_fact_limit},
        }},
        {"fact_counts", {
            {"nodes", static_cast<int>(bundle.nodes.size())},
            {"slice_links", static_cast<int>(bundle.slice_links.size())},
            {"inference_rules", static_cast<int>(bundle.inference_rules.size())},
            {"serialized_assertions", static_cast<int>(assertions.size())},
        }},
        {"manifest", manifest.is_object() ? manifest : json::object()},
    };
}

RagClipsRunResult RunRagClipsRules(
    const common_params & params,
    const RagClipsFactBundle & bundle,
    const std::vector<std::string> & assertions) {
    RagClipsRunResult result;
    result.rules_dir = ResolveRagClipsRulesDir(params);
    result.manifest_path = (fs::path(result.rules_dir) / "manifest.json").string();
    result.rule_set_id = params.rag_clips_rule_set;
    result.memory_pool_mb = params.rag_clips_memory_pool_mb;
    result.batch_fact_limit = params.rag_clips_batch_fact_limit;

    const json manifest = load_manifest_json(result.manifest_path);
    const auto entries = parse_rule_sets(manifest);

    if (!params.rag_clips_enable) {
        result.ok = false;
        result.backend = "disabled";
        result.message = "CLIPS runner is disabled";
        return result;
    }

    const RagClipsRuleManifestEntry * rule_set = find_rule_set(entries, params.rag_clips_rule_set);
    if (rule_set == nullptr) {
        result.ok = false;
        result.backend = "manifest-error";
        result.message = "CLIPS rule set not found in manifest";
        return result;
    }

    if (params.rag_clips_batch_fact_limit > 0 && assertions.size() > static_cast<size_t>(params.rag_clips_batch_fact_limit)) {
        result.ok = false;
        result.backend = "memory";
        result.message = "CLIPS batch fact limit exceeded; storage-backed paging is required";
        result.facts_rejected_by_limit = static_cast<int>(assertions.size() - params.rag_clips_batch_fact_limit);
        append_runtime_admission_decision(
            result,
            "query-context",
            "require_human",
            "batch-fact-limit-exceeded",
            "runtime.clips.batch_limit",
            "reduce_top_k_or_enable_storage",
            "(admission-decision (slice-id \"query-context\") (decision require_human) (reason \"batch-fact-limit-exceeded\") (rule-id \"runtime.clips.batch_limit\") (next-action \"reduce_top_k_or_enable_storage\"))");
        finalize_admission_summary(result);
        return result;
    }

    for (const auto & rule_file : rule_set->rule_files) {
        result.loaded_rule_files.push_back((fs::path(result.rules_dir) / rule_file).string());
    }

#if !defined(LLAMA_SERVER_RAG_CLIPS_CORE_LINKED)
    result.ok = false;
    result.backend = "not-linked";
    result.message = "CLIPS core is not linked in this build";
    return result;
#else
    LOG_INF("%s: creating CLIPS environment for rule set '%s'\n", __func__, params.rag_clips_rule_set.c_str());
    Environment * env = CreateEnvironment();
    if (env == nullptr) {
        result.ok = false;
        result.backend = "clips-core";
        result.message = "Failed to create CLIPS environment";
        return result;
    }

    bool success = true;
    std::string error_message;
    GCBlock gcb;
    GCBlockStart(env, &gcb);

    for (const auto & absolute_path : result.loaded_rule_files) {
        LOG_INF("%s: loading CLIPS rule file '%s'\n", __func__, absolute_path.c_str());
        if (Load(env, absolute_path.c_str()) != LE_NO_ERROR) {
            success = false;
            error_message = "Failed to load CLIPS rule file: " + absolute_path;
            break;
        }
    }

    if (success) {
        LOG_INF("%s: resetting CLIPS environment and asserting %d facts\n", __func__, static_cast<int>(assertions.size()));
        Reset(env);
        for (const auto & assertion : assertions) {
            if (AssertString(env, assertion.c_str()) == nullptr) {
                success = false;
                error_message = "Failed to assert CLIPS fact: " + assertion;
                append_runtime_admission_decision(
                    result,
                    "query-context",
                    "repair",
                    "assert-fact-failed",
                    "runtime.clips.assert_fact",
                    "inspect_fact_template",
                    assertion);
                break;
            }
            result.facts_asserted++;
        }
    }

    if (success) {
        LOG_INF("%s: collecting activations before run\n", __func__);
        result.activations_before_run_count = static_cast<int>(GetNumberOfActivations(env));
        for (Activation * activation = GetNextActivation(env, nullptr);
             activation != nullptr;
             activation = GetNextActivation(env, activation)) {
            const char * rule_name = ActivationRuleName(activation);
            result.activations_before_run.push_back(rule_name ? rule_name : "");
        }

        LOG_INF("%s: running CLIPS rules\n", __func__);
        result.rules_fired = static_cast<int>(Run(env, -1));
        LOG_INF("%s: collecting facts after run\n", __func__);
        result.facts_after_run_count = static_cast<int>(GetNumberOfFacts(env));
        for (Fact * fact = GetNextFact(env, nullptr); fact != nullptr; fact = GetNextFact(env, fact)) {
            if (!should_capture_fact_text(fact)) {
                continue;
            }
            const std::string fact_text = fact_to_string(env, fact);
            result.facts_after_run.push_back(fact_text);
            append_admission_summary(result, fact_text);
        }
    }

    GCBlockEnd(env, &gcb);
    LOG_INF("%s: clearing CLIPS environment before destroy\n", __func__);
    Clear(env);
    LOG_INF("%s: destroying CLIPS environment\n", __func__);
    const bool destroyed = DestroyEnvironment(env);
    LOG_INF("%s: CLIPS environment destroy result=%s\n", __func__, destroyed ? "true" : "false");

    result.ok = success;
    result.backend = "clips-core";
    result.message = success ? "CLIPS rules executed" : error_message;
    LOG_INF("%s: finalizing admission summary\n", __func__);
    finalize_admission_summary(result);
    LOG_INF("%s: returning run result ok=%s dominant_decision='%s'\n",
        __func__,
        result.ok ? "true" : "false",
        result.dominant_decision.c_str());
    return result;
#endif
}
