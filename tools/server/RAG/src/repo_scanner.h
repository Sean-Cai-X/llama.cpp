#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct RepoScannerFile {
    std::string relative_path;
    std::string language;
    size_t file_size = 0;
    std::string text;
    std::string metadata;
};

struct RepoScannerChunk {
    std::string slice_id;
    std::string relative_path;
    std::string language;
    int start_line = 0;
    int end_line = 0;
    std::string text;
    std::string metadata;
};

struct RepoScannerResult {
    std::vector<RepoScannerFile> files;
    std::vector<RepoScannerChunk> chunks;
    size_t scanned_files = 0;
    size_t skipped_files = 0;
};

class RepoScanner {
public:
    static RepoScannerResult scan(const std::string & repo_path);
};
