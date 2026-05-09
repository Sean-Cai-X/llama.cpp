#pragma once

#include "common.h"
#include "rag_clips_meta.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct RagClipsRuleManifestEntry {
    std::string rule_set_id;
    std::string description;
    std::vector<std::string> rule_files;
    bool mcp_visible = true;
};

struct RagClipsAdmissionDecision {
    std::string slice_id;
    std::string decision;
    std::string reason;
    std::string rule_id;
    std::string next_action;
    std::string raw_fact;
};

struct RagClipsRunResult {
    bool ok = false;
    std::string backend = "unavailable";
    std::string message;
    std::string rules_dir;
    std::string manifest_path;
    std::string rule_set_id;
    std::vector<std::string> loaded_rule_files;
    std::vector<std::string> activations_before_run;
    std::vector<std::string> facts_after_run;
    std::vector<std::string> admission_decisions;
    std::vector<RagClipsAdmissionDecision> admission_report;
    int facts_asserted = 0;
    int facts_after_run_count = 0;
    int activations_before_run_count = 0;
    int rules_fired = 0;
    int memory_pool_mb = 0;
    int batch_fact_limit = 0;
    int facts_rejected_by_limit = 0;
    int allow_count = 0;
    int reject_count = 0;
    int repair_count = 0;
    int short_term_only_count = 0;
    int require_human_count = 0;
    std::string dominant_decision = "none";
};

std::string ResolveRagClipsRulesDir(const common_params & params);

json BuildRagClipsManifestPayload(
    const common_params & params,
    const RagClipsFactBundle & bundle,
    const std::vector<std::string> & assertions);

RagClipsRunResult RunRagClipsRules(
    const common_params & params,
    const RagClipsFactBundle & bundle,
    const std::vector<std::string> & assertions);
