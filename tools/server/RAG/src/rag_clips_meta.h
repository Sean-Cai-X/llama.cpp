#pragma once

#include "rag.h"

#include <string>
#include <vector>

struct RagMetaNode {
    std::string node_id;
    std::string parent_id = "META-ROOT";
    int anchor_level = 3;
    bool is_immutable = false;
    std::string node_type = "CONCEPT";
    std::string lifecycle = "PENDING";
    std::string domain_name;
    std::string concept_name;
    std::string core_definition;
};

struct RagMetaSliceLink {
    std::string link_id;
    std::string slice_id;
    std::string bind_node_id;
    float info_weight = 0.0f;
    std::string truth_status = "PENDING";
    bool is_reversible = true;
    float similarity_to_core = 0.0f;
    std::string slice_text;
    std::string embedding_hash;
    std::string dedup_hash;
    std::string source_type = "retrieved";
    std::string provider_id;
    std::string evidence_ref;
    std::string body_quality = "clean";
    bool projection_incomplete = false;
    std::string vector_skip_reason;
    bool browser_visible_summary = false;
};

struct RagMetaInferenceRule {
    std::string rule_id;
    std::string inference_type = "FORWARD_CONVERGE";
    std::string source_node_type = "CONCEPT";
    std::string target_node_type = "SLICE-LINK";
    int max_diffuse_depth = 3;
    int max_slice_per_node = 20;
    float min_similarity_threshold = 0.7f;
    bool is_enabled = true;
};

struct RagMetaQueryContext {
    std::string query;
    std::string retrieval_mode;
    int result_count = 0;
};

struct RagMetaGraphNode {
    std::string graph_node_id;
    std::string node_kind;
    std::string parent_graph_node_id;
    std::string label;
    std::string ref_id;
    int depth = 0;
};

struct RagMetaGraphEdge {
    std::string edge_id;
    std::string edge_type;
    std::string from_node_id;
    std::string to_node_id;
    std::string relation;
};

struct RagMetaGraph {
    std::vector<RagMetaGraphNode> nodes;
    std::vector<RagMetaGraphEdge> edges;
};

struct RagClipsFactBundle {
    RagMetaQueryContext query_context;
    std::vector<RagMetaNode> nodes;
    std::vector<RagMetaSliceLink> slice_links;
    std::vector<RagMetaInferenceRule> inference_rules;
    RagMetaGraph meta_graph;
};

RagClipsFactBundle BuildRagClipsFactBundle(
    const std::string & query,
    const std::vector<RagSearchResult> & results,
    const std::string & retrieval_mode);

std::vector<std::string> SerializeRagClipsFacts(const RagClipsFactBundle & bundle);
