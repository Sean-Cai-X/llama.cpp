#include "repo_scanner.h"
#include "rag_metadata.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

bool should_skip_directory_name(const std::string & name) {
    static const std::unordered_set<std::string> skipped = {
        ".git",
        "build",
        "builds",
        "dist",
        "out",
        "node_modules",
        "third_party",
        "vendor",
        "__pycache__",
    };
    return skipped.find(name) != skipped.end();
}

std::string detect_language(const fs::path & path) {
    const std::string filename = path.filename().string();
    const std::string ext = path.extension().string();

    if (filename == "CMakeLists.txt") {
        return "cmake";
    }
    if (filename == "Makefile") {
        return "make";
    }
    if (ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".c") {
        return "cpp";
    }
    if (ext == ".h" || ext == ".hpp" || ext == ".hh" || ext == ".hxx") {
        return "cpp";
    }
    if (ext == ".py") {
        return "python";
    }
    if (ext == ".js") {
        return "javascript";
    }
    if (ext == ".ts") {
        return "typescript";
    }
    if (ext == ".java") {
        return "java";
    }
    if (ext == ".go") {
        return "go";
    }
    if (ext == ".rs") {
        return "rust";
    }
    if (ext == ".md") {
        return "markdown";
    }
    if (ext == ".txt") {
        return "text";
    }
    if (ext == ".html" || ext == ".htm") {
        return "html";
    }

    return "";
}

bool is_binary_text(const std::string & text) {
    for (unsigned char ch : text) {
        if (ch == 0) {
            return true;
        }
    }
    return false;
}

std::string read_text_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open file: " + path.string());
    }

    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

std::string decode_html_entity(const std::string & entity) {
    if (entity == "&nbsp;") {
        return " ";
    }
    if (entity == "&amp;") {
        return "&";
    }
    if (entity == "&lt;") {
        return "<";
    }
    if (entity == "&gt;") {
        return ">";
    }
    if (entity == "&quot;") {
        return "\"";
    }
    if (entity == "&#39;") {
        return "'";
    }
    return " ";
}

std::string strip_html(const std::string & input) {
    std::string output;
    output.reserve(input.size());

    bool in_tag = false;
    bool in_script = false;
    bool in_style = false;
    std::string tag_buffer;
    std::string entity_buffer;

    auto flush_tag_state = [&](const std::string & tag_text) {
        std::string lowered;
        lowered.reserve(tag_text.size());
        for (char ch : tag_text) {
            lowered.push_back((char) std::tolower((unsigned char) ch));
        }

        if (lowered.find("script") != std::string::npos) {
            in_script = lowered.find("/script") == std::string::npos;
        } else if (lowered.find("style") != std::string::npos) {
            in_style = lowered.find("/style") == std::string::npos;
        } else if (!in_script && !in_style) {
            if (lowered == "br" || lowered == "br/" || lowered == "/p" || lowered == "p" ||
                lowered == "/div" || lowered == "div" || lowered == "/li" || lowered == "li" ||
                lowered == "/tr" || lowered == "tr" || lowered == "/h1" || lowered == "h1" ||
                lowered == "/h2" || lowered == "h2" || lowered == "/h3" || lowered == "h3") {
                output.push_back('\n');
            }
        }
    };

    for (size_t i = 0; i < input.size(); ++i) {
        const char ch = input[i];

        if (in_tag) {
            if (ch == '>') {
                in_tag = false;
                flush_tag_state(tag_buffer);
                tag_buffer.clear();
            } else {
                tag_buffer.push_back(ch);
            }
            continue;
        }

        if (ch == '<') {
            in_tag = true;
            tag_buffer.clear();
            continue;
        }

        if (in_script || in_style) {
            continue;
        }

        if (ch == '&') {
            entity_buffer.clear();
            entity_buffer.push_back(ch);
            size_t j = i + 1;
            while (j < input.size() && entity_buffer.size() < 10) {
                entity_buffer.push_back(input[j]);
                if (input[j] == ';') {
                    break;
                }
                ++j;
            }
            if (!entity_buffer.empty() && entity_buffer.back() == ';') {
                output += decode_html_entity(entity_buffer);
                i = j;
                continue;
            }
        }

        output.push_back(ch);
    }

    std::string normalized;
    normalized.reserve(output.size());
    bool prev_space = false;
    bool prev_newline = false;
    for (char ch : output) {
        if (ch == '\r') {
            continue;
        }
        if (ch == '\n') {
            if (!prev_newline) {
                normalized.push_back('\n');
            }
            prev_newline = true;
            prev_space = false;
            continue;
        }
        if (std::isspace((unsigned char) ch)) {
            if (!prev_space) {
                normalized.push_back(' ');
            }
            prev_space = true;
            continue;
        }
        normalized.push_back(ch);
        prev_space = false;
        prev_newline = false;
    }

    return normalized;
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::stringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    if (lines.empty()) {
        lines.push_back(text);
    }
    return lines;
}

std::string join_lines(const std::vector<std::string> & lines, size_t begin, size_t end) {
    std::ostringstream out;
    for (size_t i = begin; i < end; ++i) {
        if (i > begin) {
            out << '\n';
        }
        out << lines[i];
    }
    return out.str();
}

std::vector<RepoScannerChunk> build_chunks(
    const RepoScannerFile & file,
    int max_lines = 80,
    int overlap_lines = 12) {
    std::vector<RepoScannerChunk> chunks;
    const std::vector<std::string> lines = split_lines(file.text);
    const size_t step = (size_t) std::max(1, max_lines - std::max(0, overlap_lines));

    for (size_t begin = 0; begin < lines.size(); begin += step) {
        const size_t end = std::min(lines.size(), begin + (size_t) max_lines);
        const std::string chunk_text = join_lines(lines, begin, end);
        if (chunk_text.empty()) {
            continue;
        }

        RepoScannerChunk chunk;
        chunk.relative_path = file.relative_path;
        chunk.language = file.language;
        chunk.start_line = (int) begin + 1;
        chunk.end_line = (int) end;
        chunk.text = chunk_text;
        const int chunk_index = static_cast<int>(chunks.size());
        const int chunk_count = static_cast<int>((lines.size() + step - 1) / step);
        const rag_json chunk_metadata = rag_build_chunk_metadata(
            rag_parse_metadata(file.metadata),
            chunk.start_line,
            chunk.end_line,
            chunk_index,
            chunk_count,
            chunk.text);
        chunk.slice_id = rag_metadata_value_string(chunk_metadata, "slice_id");
        chunk.metadata = rag_metadata_to_string(chunk_metadata);

        chunks.push_back(std::move(chunk));
        if (end >= lines.size()) {
            break;
        }
    }

    return chunks;
}

} // namespace

RepoScannerResult RepoScanner::scan(const std::string & repo_path) {
    const fs::path root = fs::weakly_canonical(fs::path(repo_path));
    if (!fs::exists(root)) {
        throw std::invalid_argument("repo_path does not exist");
    }
    if (!fs::is_directory(root)) {
        throw std::invalid_argument("repo_path must be a directory");
    }

    RepoScannerResult result;
    fs::directory_options options = fs::directory_options::skip_permission_denied;

    for (fs::recursive_directory_iterator it(root, options), end; it != end; ++it) {
        const fs::directory_entry & entry = *it;
        const fs::path path = entry.path();

        if (entry.is_directory()) {
            if (should_skip_directory_name(path.filename().string())) {
                it.disable_recursion_pending();
            }
            continue;
        }

        if (!entry.is_regular_file()) {
            result.skipped_files++;
            continue;
        }

        const std::string language = detect_language(path);
        if (language.empty()) {
            result.skipped_files++;
            continue;
        }

        std::string text = read_text_file(path);
        if (is_binary_text(text)) {
            result.skipped_files++;
            continue;
        }
        if (language == "html") {
            text = strip_html(text);
        }
        if (text.empty()) {
            result.skipped_files++;
            continue;
        }

        const std::string relative_path = fs::relative(path, root).generic_string();
        RepoScannerFile file;
        file.relative_path = relative_path;
        file.language = language;
        file.file_size = text.size();
        file.text = text;
        file.metadata = rag_metadata_to_string(rag_build_file_metadata(relative_path, language, file.file_size));

        result.scanned_files++;
        auto chunks = build_chunks(file);
        result.files.push_back(std::move(file));
        result.chunks.insert(result.chunks.end(), chunks.begin(), chunks.end());
    }

    return result;
}
