#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class BM25Index {
public:
    struct DocEntry {
        int doc_id;
        int freq;
    };

    explicit BM25Index(const std::string & path = "");

    void add_document(int doc_id, const std::string & text);
    void clear();
    std::vector<std::pair<float, int>> search(const std::string & query, int top_k = 50) const;

    void save() const;
    bool load();

private:
    std::vector<std::string> tokenize(const std::string & text) const;
    void build_stats();

    std::string index_path_;

    float k1_ = 1.5f;
    float b_  = 0.75f;

    std::unordered_map<std::string, std::vector<DocEntry>> inverted_index_;
    std::vector<int> doc_lengths_;
    float avgdl_ = 0.0f;
    int   n_docs_ = 0;
};
