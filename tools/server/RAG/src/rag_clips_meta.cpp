#include "rag_clips_meta.h"
#include "rag_metadata.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>

namespace {

std::string EscapeClipsString(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        if (ch == '\r' || ch == '\n' || ch == '\t') {
            escaped.push_back(' ');
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

std::string GuessNodeType(const std::string & retrieval_mode) {
    if (retrieval_mode == "hybrid") {
        return "CONCEPT";
    }
    return "INSTANCE";
}

std::string BuildNodeName(const std::string & source_path, int rank) {
    if (!source_path.empty()) {
        return source_path;
    }
    return "retrieved-slice-" + std::to_string(rank);
}

std::string classify_body_quality(const std::string & text) {
    std::string trimmed;
    trimmed.reserve(text.size());
    for (char ch : text) {
        if (ch != '\r' && ch != '\n' && ch != '\t') {
            trimmed.push_back(ch);
        }
    }

    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), trimmed.end());

    if (trimmed.empty()) {
        return "empty";
    }
    if (trimmed == "{") {
        return "projection_only_brace";
    }
    if (trimmed.find("```") != std::string::npos) {
        return "polluted";
    }
    return "clean";
}

void AppendEdge(
    RagMetaGraph & graph,
    const std::string & edge_type,
    const std::string & from_node_id,
    const std::string & to_node_id,
    const std::string & relation) {
    RagMetaGraphEdge edge;
    edge.edge_id = rag_make_stable_id("EDGE", edge_type + "|" + from_node_id + "|" + to_node_id + "|" + relation);
    edge.edge_type = edge_type;
    edge.from_node_id = from_node_id;
    edge.to_node_id = to_node_id;
    edge.relation = relation;
    graph.edges.push_back(std::move(edge));
}

} // namespace

RagClipsFactBundle BuildRagClipsFactBundle(
    const std::string & query,
    const std::vector<RagSearchResult> & results,
    const std::string & retrieval_mode) {
    RagClipsFactBundle bundle;
    bundle.query_context.query = query;
    bundle.query_context.retrieval_mode = retrieval_mode;
    bundle.query_context.result_count = static_cast<int>(results.size());
    auto & graph = bundle.meta_graph;

    const std::string root_id = "GRAPH-META-ROOT";
    graph.nodes.push_back({
        root_id,
        "meta_root",
        "",
        "META-ROOT",
        "META-ROOT",
        0,
    });

    const std::string domain_id = rag_make_stable_id("GRAPH-DOMAIN", retrieval_mode.empty() ? "runtime" : retrieval_mode);
    graph.nodes.push_back({
        domain_id,
        "domain",
        root_id,
        retrieval_mode.empty() ? "runtime" : retrieval_mode,
        retrieval_mode.empty() ? "runtime" : retrieval_mode,
        1,
    });
    AppendEdge(graph, "contains", root_id, domain_id, "domain");

    RagMetaInferenceRule rule;
    rule.rule_id = "rule.rag." + retrieval_mode;
    rule.inference_type = "FORWARD_CONVERGE";
    rule.source_node_type = GuessNodeType(retrieval_mode);
    rule.target_node_type = "SLICE-LINK";
    rule.min_similarity_threshold = 0.5f;
    bundle.inference_rules.push_back(rule);

    for (size_t i = 0; i < results.size(); ++i) {
        const rag_json metadata = rag_parse_metadata(results[i].metadata);
        const std::string source_path = rag_metadata_value_string(metadata, "source_uri");
        const std::string source_language = rag_metadata_value_string(metadata, "language");
        const std::string source_start = rag_metadata_value_string(metadata, "start_line");
        const std::string source_end = rag_metadata_value_string(metadata, "end_line");

        RagMetaNode node;
        node.node_id = rag_make_stable_id("NODE", query + "|" + source_path + "|" + std::to_string(i));
        node.parent_id = "META-ROOT";
        node.anchor_level = 3;
        node.is_immutable = false;
        node.node_type = GuessNodeType(retrieval_mode);
        node.lifecycle = "VERIFIED";
        node.domain_name = retrieval_mode;
        node.concept_name = BuildNodeName(source_path, (int) i + 1);
        node.core_definition = results[i].chunk_text;
        bundle.nodes.push_back(node);

        const std::string file_ref = source_path.empty() ? ("runtime:retrieved-file-" + std::to_string(i + 1)) : source_path;
        const std::string file_node_id = rag_make_stable_id("GRAPH-FILE", file_ref);
        const auto file_it = std::find_if(
            graph.nodes.begin(),
            graph.nodes.end(),
            [&](const RagMetaGraphNode & item) { return item.graph_node_id == file_node_id; });
        if (file_it == graph.nodes.end()) {
            graph.nodes.push_back({
                file_node_id,
                "file",
                domain_id,
                file_ref,
                file_ref,
                2,
            });
            AppendEdge(graph, "contains", domain_id, file_node_id, "file");
        }

        const std::string concept_graph_id = rag_make_stable_id("GRAPH-CONCEPT", node.node_id);
        graph.nodes.push_back({
            concept_graph_id,
            "concept",
            file_node_id,
            node.concept_name,
            node.node_id,
            3,
        });
        AppendEdge(graph, "describes", file_node_id, concept_graph_id, "concept");

        RagMetaSliceLink link;
        link.link_id = rag_make_stable_id("LINK", node.node_id);
        link.slice_id = rag_metadata_value_string(metadata, "slice_id");
        if (link.slice_id.empty()) {
            link.slice_id = rag_make_stable_id("SLICE", source_path + "|" + source_start + "|" + source_end + "|" + results[i].chunk_text);
        }
        link.bind_node_id = node.node_id;
        link.info_weight = std::max(0.0f, std::min(1.0f, results[i].score));
        link.truth_status = "TRUE";
        link.is_reversible = true;
        link.similarity_to_core = std::max(0.0f, std::min(1.0f, results[i].score));
        link.slice_text = results[i].chunk_text;
        link.embedding_hash = rag_make_stable_id("EMBD", results[i].chunk_text);
        link.dedup_hash = rag_make_stable_id("DEDUP", results[i].chunk_text);
        link.source_type = source_language.empty() ? "retrieved" : source_language;
        link.provider_id = "rag-main-thread";
        link.evidence_ref = source_path.empty() ? "runtime:retrieved-context" : source_path;
        link.body_quality = classify_body_quality(results[i].chunk_text);
        link.projection_incomplete = rag_metadata_value_bool(metadata, "projection_incomplete", false);
        link.vector_skip_reason = rag_metadata_value_string(metadata, "vector_skip_reason");
        link.browser_visible_summary =
            rag_metadata_value_bool(metadata, "browser_visible_summary", false) ||
            rag_metadata_value_bool(metadata, "browser_visible_summary_turn", false);
        bundle.slice_links.push_back(link);

        const std::string slice_label =
            (source_start.empty() || source_end.empty())
                ? ("slice-" + std::to_string(i + 1))
                : ("slice:" + source_start + "-" + source_end);
        const std::string slice_graph_id = rag_make_stable_id("GRAPH-SLICE", link.slice_id);
        graph.nodes.push_back({
            slice_graph_id,
            "slice",
            concept_graph_id,
            slice_label,
            link.slice_id,
            4,
        });
        AppendEdge(graph, "grounds", concept_graph_id, slice_graph_id, "slice_link");
    }

    for (const auto & rule_item : bundle.inference_rules) {
        const std::string rule_graph_id = rag_make_stable_id("GRAPH-RULE", rule_item.rule_id);
        graph.nodes.push_back({
            rule_graph_id,
            "inference_rule",
            domain_id,
            rule_item.rule_id,
            rule_item.rule_id,
            2,
        });
        AppendEdge(graph, "governs", domain_id, rule_graph_id, "inference_rule");
    }

    return bundle;
}

std::vector<std::string> SerializeRagClipsFacts(const RagClipsFactBundle & bundle) {
    std::vector<std::string> facts;

    {
        std::ostringstream out;
        out << "(query-context "
            << "(query \"" << EscapeClipsString(bundle.query_context.query) << "\") "
            << "(retrieval-mode \"" << EscapeClipsString(bundle.query_context.retrieval_mode) << "\") "
            << "(result-count " << bundle.query_context.result_count << "))";
        facts.push_back(out.str());
    }

    for (const auto & node : bundle.nodes) {
        std::ostringstream out;
        out << "(meta-node "
            << "(node-id \"" << EscapeClipsString(node.node_id) << "\") "
            << "(parent-id \"" << EscapeClipsString(node.parent_id) << "\") "
            << "(anchor-level " << node.anchor_level << ") "
            << "(is-immutable " << (node.is_immutable ? "TRUE" : "FALSE") << ") "
            << "(node-type " << node.node_type << ") "
            << "(lifecycle " << node.lifecycle << ") "
            << "(domain-name \"" << EscapeClipsString(node.domain_name) << "\"))";
        facts.push_back(out.str());

        std::ostringstream concept_fact;
        concept_fact << "(concept-node "
                     << "(node-id \"" << EscapeClipsString(node.node_id) << "\") "
                     << "(parent-id \"" << EscapeClipsString(node.parent_id) << "\") "
                     << "(anchor-level " << node.anchor_level << ") "
                     << "(concept-name \"" << EscapeClipsString(node.concept_name) << "\") "
                     << "(core-definition \"" << EscapeClipsString(node.core_definition) << "\"))";
        facts.push_back(concept_fact.str());
    }

    for (const auto & link : bundle.slice_links) {
        std::ostringstream out;
        out << "(slice-instance "
            << "(link-id \"" << EscapeClipsString(link.link_id) << "\") "
            << "(slice-id \"" << EscapeClipsString(link.slice_id) << "\") "
            << "(bind-node-id \"" << EscapeClipsString(link.bind_node_id) << "\") "
            << "(info-weight " << link.info_weight << ") "
            << "(truth-status " << link.truth_status << ") "
            << "(is-reversible " << (link.is_reversible ? "TRUE" : "FALSE") << ") "
            << "(similarity-to-core " << link.similarity_to_core << ") "
            << "(slice-text \"" << EscapeClipsString(link.slice_text) << "\") "
            << "(embedding-hash \"" << EscapeClipsString(link.embedding_hash) << "\") "
            << "(dedup-hash \"" << EscapeClipsString(link.dedup_hash) << "\") "
            << "(source-type \"" << EscapeClipsString(link.source_type) << "\") "
            << "(provider-id \"" << EscapeClipsString(link.provider_id) << "\") "
            << "(evidence-ref \"" << EscapeClipsString(link.evidence_ref) << "\") "
            << "(body-quality \"" << EscapeClipsString(link.body_quality) << "\") "
            << "(projection-incomplete " << (link.projection_incomplete ? "TRUE" : "FALSE") << ") "
            << "(vector-skip-reason \"" << EscapeClipsString(link.vector_skip_reason) << "\") "
            << "(browser-visible-summary " << (link.browser_visible_summary ? "TRUE" : "FALSE") << "))";
        facts.push_back(out.str());
    }

    for (const auto & rule : bundle.inference_rules) {
        std::ostringstream out;
        out << "(meta-inference-rule "
            << "(rule-id \"" << EscapeClipsString(rule.rule_id) << "\") "
            << "(inference-type " << rule.inference_type << ") "
            << "(source-node-type " << rule.source_node_type << ") "
            << "(target-node-type " << rule.target_node_type << ") "
            << "(max-diffuse-depth " << rule.max_diffuse_depth << ") "
            << "(max-slice-per-node " << rule.max_slice_per_node << ") "
            << "(min-similarity-threshold " << rule.min_similarity_threshold << ") "
            << "(is-enabled " << (rule.is_enabled ? "TRUE" : "FALSE") << "))";
        facts.push_back(out.str());
    }

    return facts;
}
