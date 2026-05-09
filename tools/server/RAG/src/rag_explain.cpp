#include "rag_explain.h"

#include <cctype>

namespace {

std::string sanitize_identifier(const std::string & text, const std::string & fallback) {
    std::string out;
    for (char ch : text) {
        const unsigned char uch = (unsigned char) ch;
        if (std::isalnum(uch)) {
            out += (char) std::tolower(uch);
        } else if ((ch == '_' || ch == '-' || ch == ':' || ch == ' ' || ch == '.') && !out.empty() && out.back() != '_') {
            out += '_';
        }
        if (out.size() >= 48) {
            break;
        }
    }

    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }

    return out.empty() ? fallback : out;
}

std::string metadata_path(const json & metadata) {
    return metadata.value("path", "");
}

std::string metadata_language(const json & metadata) {
    return metadata.value("language", "");
}

std::string guess_module_name_from_metadata(const json & metadata) {
    const std::string path = metadata_path(metadata);
    if (path.empty()) {
        return "cxparser";
    }

    if (path.find("cxparser_ext/") != std::string::npos) {
        return "cxparser_ext";
    }
    if (path.find("cxparser/") != std::string::npos) {
        return "cxparser";
    }
    if (path.find("tools/server/") != std::string::npos) {
        return "llama_server";
    }
    if (path.find("docs/") == 0 || metadata_language(metadata) == "markdown" || metadata_language(metadata) == "text") {
        return "documentation";
    }

    const size_t slash = path.find('/');
    const std::string first = slash == std::string::npos ? path : path.substr(0, slash);
    if (first == "src" || first == "include") {
        return "codebase";
    }
    return sanitize_identifier(first, "cxparser");
}

std::string guess_target_class_from_metadata(const json & metadata) {
    const std::string symbol_name = metadata.value("symbol_name", "");
    if (!symbol_name.empty()) {
        return sanitize_identifier(symbol_name, "symbol");
    }

    std::string filename = metadata_path(metadata);
    const size_t slash = filename.find_last_of('/');
    if (slash != std::string::npos) {
        filename = filename.substr(slash + 1);
    }
    const size_t dot = filename.find('.');
    if (dot != std::string::npos) {
        filename = filename.substr(0, dot);
    }
    return sanitize_identifier(filename, "script");
}

std::string guess_target_method_from_query(const std::string & query) {
    const std::string lowered = sanitize_identifier(query, "eval");
    if (lowered.find("explain") != std::string::npos || lowered.find("why") != std::string::npos || lowered.find("how") != std::string::npos) {
        return "explain";
    }
    if (lowered.find("search") != std::string::npos || lowered.find("retrieve") != std::string::npos) {
        return "search";
    }
    if (lowered.find("index") != std::string::npos || lowered.find("ingest") != std::string::npos) {
        return "index";
    }
    if (lowered.find("parse") != std::string::npos || lowered.find("chunk") != std::string::npos) {
        return "parse";
    }
    return lowered;
}

std::string guess_route_hint(const std::string & query, const json & metadata) {
    const std::string language = metadata_language(metadata);
    const std::string path = metadata_path(metadata);
    const std::string lowered = sanitize_identifier(query, "eval");

    if (language == "markdown" || language == "text" || path.find("docs/") == 0) {
        return "batch";
    }
    if (lowered.find("explain") != std::string::npos || lowered.find("analy") != std::string::npos) {
        return "batch";
    }
    return "default";
}

std::string guess_priority_hint(const std::string & route_hint) {
    return route_hint == "batch" ? "background" : "normal";
}

std::string guess_capability_name(const std::string & query, const json & metadata) {
    const std::string language = metadata_language(metadata);
    const std::string lowered = sanitize_identifier(query, "eval");

    if (language == "markdown" || language == "text") {
        return "documentation_lookup";
    }
    if (lowered.find("explain") != std::string::npos || lowered.find("analy") != std::string::npos) {
        return "semantic_explain";
    }
    return "code_lookup";
}

std::string format_rag_source_label(const json & metadata) {
    const std::string path = metadata.value("path", "");
    if (path.empty()) {
        return "";
    }

    if (metadata.contains("start_line") && metadata.contains("end_line")) {
        return path + ":" + std::to_string(metadata["start_line"].get<int>()) + "-" + std::to_string(metadata["end_line"].get<int>());
    }
    return path;
}

} // namespace

json build_rag_source_info(const json & metadata) {
    json source = json::object();

    const std::string path = metadata_path(metadata);
    if (!path.empty()) {
        source["path"] = path;
    }

    const std::string language = metadata_language(metadata);
    if (!language.empty()) {
        source["language"] = language;
    }

    if (metadata.contains("start_line")) {
        source["start_line"] = metadata["start_line"];
    }
    if (metadata.contains("end_line")) {
        source["end_line"] = metadata["end_line"];
    }
    if (metadata.contains("chunk_id")) {
        source["chunk_id"] = metadata["chunk_id"];
    }

    if (metadata.contains("start_line") || metadata.contains("end_line")) {
        json line_range = json::object();
        if (metadata.contains("start_line")) {
            line_range["start"] = metadata["start_line"];
        }
        if (metadata.contains("end_line")) {
            line_range["end"] = metadata["end_line"];
        }
        source["line_range"] = std::move(line_range);
    }

    source["label"] = format_rag_source_label(metadata);
    return source;
}

json build_rag_execution_target(
    const std::string & query,
    const json & metadata,
    int rank,
    const std::string & preview_text) {
    const std::string module_name = guess_module_name_from_metadata(metadata);
    const std::string target_class = guess_target_class_from_metadata(metadata);
    const std::string target_method = guess_target_method_from_query(query);
    const std::string route_hint = guess_route_hint(query, metadata);
    const std::string priority_hint = guess_priority_hint(route_hint);
    const std::string capability_name = guess_capability_name(query, metadata);
    const std::string route_key = module_name + "::" + target_class + "." + target_method;

    json route = {
        {"route_key", "rag:" + route_key},
        {"lane_name", route_hint == "batch" ? "batch" : "rag_explain"},
        {"deadline_ms", route_hint == "batch" ? 1000 : 500},
        {"timeout_ms", route_hint == "batch" ? 10000 : 3000},
        {"allow_degraded_result", true},
    };

    json module_call = {
        {"caller_module", "llama.server.rag"},
        {"callee_module", module_name},
        {"protocol_name", "cxparser.module.call"},
        {"capability_name", capability_name},
        {"object_name", build_rag_source_info(metadata).value("label", "")},
        {"class_name", target_class},
        {"method_name", target_method},
    };

    return {
        {"task_id", "rag_explain_" + std::to_string(rank)},
        {"task_name", route_key},
        {"trace_id", "rag_explain_" + std::to_string(rank) + ".trace"},
        {"route_hint", route_hint},
        {"priority_hint", priority_hint},
        {"module_name", module_name},
        {"target_class", target_class},
        {"target_method", target_method},
        {"route", route},
        {"module_call", module_call},
        {"script_text", preview_text},
        {"tags", json::array({"rag", "explain", module_name, capability_name})},
    };
}

json build_rag_evidence_stub(const std::string & query, const json & metadata, int rank) {
    const json target = build_rag_execution_target(query, metadata, rank, "");
    return {
        {"trace_id", target.value("trace_id", "")},
        {"route_key", target["route"].value("route_key", "")},
        {"route_lane", target["route"].value("lane_name", "")},
        {"protocol_name", target["module_call"].value("protocol_name", "")},
        {"notes", json::array({
            "Generated from /rag/search results",
            "Prototype mapping for cxparser_ext integration",
            "No parser runtime executed in this release",
        })},
    };
}
