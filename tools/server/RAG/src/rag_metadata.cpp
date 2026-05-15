#include "rag_metadata.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>

namespace {

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string metadata_source_uri(const rag_json & metadata) {
    const std::string source_uri = rag_metadata_value_string(metadata, "source_uri");
    if (!source_uri.empty()) {
        return source_uri;
    }
    return rag_metadata_value_string(metadata, "path");
}

std::string metadata_text_hash(const std::string & text) {
    return "sha256:" + std::to_string(std::hash<std::string>{}(text));
}

} // namespace

std::string rag_make_stable_id(const std::string & prefix, const std::string & seed) {
    return prefix + "-" + std::to_string(std::hash<std::string>{}(seed));
}

rag_json rag_parse_metadata(const std::string & metadata) {
    rag_json out = rag_json::object();
    if (metadata.empty()) {
        return out;
    }

    try {
        const rag_json parsed = rag_json::parse(metadata);
        if (parsed.is_object()) {
            out = parsed;
        }
    } catch (...) {
    }

    if (out.empty()) {
        std::stringstream stream(metadata);
        std::string token;
        while (std::getline(stream, token, ';')) {
            const size_t sep = token.find('=');
            if (sep == std::string::npos || sep == 0 || sep + 1 >= token.size()) {
                continue;
            }

            const std::string key = trim_copy(token.substr(0, sep));
            const std::string value = trim_copy(token.substr(sep + 1));
            if (key.empty() || value.empty()) {
                continue;
            }

            try {
                size_t parsed = 0;
                const long long numeric = std::stoll(value, &parsed);
                if (parsed == value.size()) {
                    out[key] = numeric;
                    continue;
                }
            } catch (...) {
            }

            std::string lowered = value;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            if (lowered == "true" || lowered == "false") {
                out[key] = lowered == "true";
                continue;
            }

            out[key] = value;
        }
    }

    if (out.contains("source_uri") && !out.contains("path")) {
        out["path"] = out["source_uri"];
    }
    if (out.contains("path") && !out.contains("source_uri")) {
        out["source_uri"] = out["path"];
    }
    if (!out.contains("schema_version")) {
        out["schema_version"] = 1;
    }

    return out;
}

std::string rag_metadata_to_string(const rag_json & metadata) {
    return metadata.dump();
}

std::string rag_metadata_value_string(const rag_json & metadata, const std::string & key) {
    if (!metadata.contains(key)) {
        return "";
    }

    const auto & value = metadata.at(key);
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number_unsigned()) {
        return std::to_string(value.get<unsigned long long>());
    }
    return "";
}

int rag_metadata_value_int(const rag_json & metadata, const std::string & key, int fallback) {
    if (!metadata.contains(key)) {
        return fallback;
    }

    const auto & value = metadata.at(key);
    if (value.is_number_integer()) {
        return value.get<int>();
    }
    if (value.is_number_unsigned()) {
        return static_cast<int>(value.get<unsigned int>());
    }
    if (value.is_string()) {
        try {
            return std::stoi(value.get<std::string>());
        } catch (...) {
        }
    }
    return fallback;
}

bool rag_metadata_value_bool(const rag_json & metadata, const std::string & key, bool fallback) {
    if (!metadata.contains(key)) {
        return fallback;
    }

    const auto & value = metadata.at(key);
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int>() != 0;
    }
    if (value.is_string()) {
        std::string lowered = value.get<std::string>();
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
    }
    return fallback;
}

bool rag_metadata_contains_flag(const std::string & metadata, const std::string & flag) {
    return rag_metadata_value_bool(rag_parse_metadata(metadata), flag, false);
}

rag_json rag_build_file_metadata(
    const std::string & source_uri,
    const std::string & language,
    size_t file_size) {
    return rag_json{
        {"schema_version", 1},
        {"source_uri", source_uri},
        {"path", source_uri},
        {"language", language},
        {"file_size", file_size},
    };
}

rag_json rag_build_chunk_metadata(
    const rag_json & base_metadata,
    int start_line,
    int end_line,
    int chunk_index,
    int chunk_count,
    const std::string & text) {
    rag_json metadata = base_metadata;
    metadata["schema_version"] = 1;

    const std::string source_uri = metadata_source_uri(metadata);
    if (!source_uri.empty()) {
        metadata["source_uri"] = source_uri;
        metadata["path"] = source_uri;
    }

    metadata["start_line"] = start_line;
    metadata["end_line"] = end_line;
    metadata["chunk_index"] = chunk_index;
    metadata["chunk_count"] = chunk_count;
    metadata["text_hash"] = metadata_text_hash(text);
    metadata["prechunked"] = true;

    const std::string slice_seed = source_uri + "|" + std::to_string(start_line) + "|" + std::to_string(end_line) + "|" + metadata["text_hash"].get<std::string>();
    metadata["slice_id"] = rag_make_stable_id("SLICE", slice_seed);
    metadata["chunk_id"] = source_uri.empty()
        ? metadata["slice_id"]
        : source_uri + ":" + std::to_string(start_line) + "-" + std::to_string(end_line);

    return metadata;
}

rag_json rag_normalize_ingest_metadata(
    const std::string & metadata_text,
    const std::string & chunk_text,
    int fallback_doc_id,
    int chunk_index,
    int chunk_count) {
    rag_json metadata = rag_parse_metadata(metadata_text);
    metadata["schema_version"] = 1;

    const std::string source_uri = metadata_source_uri(metadata);
    if (!source_uri.empty()) {
        metadata["source_uri"] = source_uri;
        metadata["path"] = source_uri;
    }

    if (!metadata.contains("chunk_index")) {
        metadata["chunk_index"] = chunk_index;
    }
    if (!metadata.contains("chunk_count")) {
        metadata["chunk_count"] = chunk_count;
    }
    if (!metadata.contains("text_hash")) {
        metadata["text_hash"] = metadata_text_hash(chunk_text);
    }

    const int start_line = rag_metadata_value_int(metadata, "start_line", 0);
    const int end_line = rag_metadata_value_int(metadata, "end_line", 0);
    if (!metadata.contains("slice_id")) {
        std::string seed = source_uri;
        if (start_line > 0 || end_line > 0) {
            seed += "|" + std::to_string(start_line) + "|" + std::to_string(end_line);
        } else {
            seed += "|doc:" + std::to_string(fallback_doc_id) + "|chunk:" + std::to_string(chunk_index);
        }
        seed += "|" + rag_metadata_value_string(metadata, "text_hash");
        metadata["slice_id"] = rag_make_stable_id("SLICE", seed);
    }

    if (!metadata.contains("chunk_id")) {
        if (!source_uri.empty() && start_line > 0 && end_line > 0) {
            metadata["chunk_id"] = source_uri + ":" + std::to_string(start_line) + "-" + std::to_string(end_line);
        } else {
            metadata["chunk_id"] = metadata["slice_id"];
        }
    }

    return metadata;
}
