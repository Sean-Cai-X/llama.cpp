#include "rag_chat_minimal.h"

#include "rag_cxparser_bridge.h"
#include "rag_explain.h"

#include <sstream>

namespace {

json parse_rag_metadata(const std::string & metadata) {
    json result = json::object();
    if (metadata.empty()) {
        return result;
    }

    std::stringstream stream(metadata);
    std::string token;
    while (std::getline(stream, token, ';')) {
        const size_t sep = token.find('=');
        if (sep == std::string::npos || sep == 0 || sep + 1 >= token.size()) {
            continue;
        }

        const std::string key = token.substr(0, sep);
        const std::string value = token.substr(sep + 1);
        if (value.empty()) {
            continue;
        }

        try {
            size_t parsed = 0;
            const long long numeric = std::stoll(value, &parsed);
            if (parsed == value.size()) {
                result[key] = numeric;
                continue;
            }
        } catch (...) {
        }

        result[key] = value;
    }

    return result;
}

std::string sanitize_rag_preview(std::string text, size_t max_chars) {
    for (char & ch : text) {
        if (ch == '\r' || ch == '\n' || ch == '\t') {
            ch = ' ';
        }
    }
    if (text.size() > max_chars) {
        text.resize(max_chars);
        text += "...";
    }
    return text;
}

std::string format_rag_source_label(const json & metadata) {
    const std::string path = metadata.value("path", "");
    const bool has_start = metadata.contains("start_line");
    const bool has_end = metadata.contains("end_line");

    if (path.empty()) {
        return "";
    }
    if (has_start && has_end) {
        return path + ":" + std::to_string(metadata["start_line"].get<int>()) + "-" + std::to_string(metadata["end_line"].get<int>());
    }
    return path;
}

std::string build_rag_context_block(const std::vector<RagSearchResult> & results, size_t max_preview_chars) {
    if (results.empty()) {
        return "";
    }

    constexpr size_t max_context_chars = 6000;
    std::string context = "Relevant context:\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const json metadata = parse_rag_metadata(results[i].metadata);
        const std::string source = format_rag_source_label(metadata);
        std::string block = "[" + std::to_string(i + 1) + "]";
        if (!source.empty()) {
            block += " Source: " + source;
        }
        block += "\n" + sanitize_rag_preview(results[i].chunk_text, max_preview_chars) + "\n";

        if (context.size() + block.size() > max_context_chars) {
            context += "[truncated] Additional retrieved context omitted to stay within the prompt budget.\n";
            break;
        }
        context += block;
    }
    return context;
}

json build_recall_item(
    const std::string & query,
    const RagSearchResult & result,
    int rank,
    const std::string & retrieval_mode,
    const RagChatMinimalOptions & options) {
    const json metadata = parse_rag_metadata(result.metadata);
    const std::string preview = sanitize_rag_preview(result.chunk_text, options.max_preview_chars);

    json item = {
        {"rank", rank},
        {"score", result.score},
        {"source", format_rag_source_label(metadata)},
        {"source_info", build_rag_source_info(metadata)},
        {"preview_text", preview},
        {"chunk_text", result.chunk_text},
        {"metadata", metadata},
        {"raw_metadata", result.metadata},
    };

    if (options.include_explanations) {
        item["explain"] = {
            {"retrieval_mode", retrieval_mode},
            {"execution_target", build_rag_execution_target(query, metadata, rank, preview)},
            {"evidence_stub", build_rag_evidence_stub(query, metadata, rank)},
        };
        if (options.include_cxparser_dry_run) {
            item["explain"]["cxparser_dry_run"] = build_rag_cxparser_dry_run(query, metadata, rank, preview);
        }
    }

    return item;
}

} // namespace

json build_rag_chat_minimal_response(
    const std::string & query,
    int top_k,
    const std::string & retrieval_mode,
    const std::vector<RagSearchResult> & results,
    const RagChatMinimalOptions & options) {
    json citations = json::array();
    for (size_t i = 0; i < results.size(); ++i) {
        citations.push_back(build_recall_item(query, results[i], (int) i + 1, retrieval_mode, options));
    }

    json response = {
        {"query", query},
        {"top_k", top_k},
        {"retrieval_mode", retrieval_mode},
        {"summary", {
            {"result_count", (int) citations.size()},
            {"ready_for_chat", !citations.empty()},
            {"include_prompt_context", options.include_prompt_context},
            {"include_explanations", options.include_explanations},
            {"include_cxparser_dry_run", options.include_cxparser_dry_run},
        }},
        {"citations", citations},
    };

    if (options.include_prompt_context) {
        response["prompt_context"] = build_rag_context_block(results, options.max_preview_chars);
    }

    return response;
}
