#include "BM25Index.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

BM25Index::BM25Index(const std::string & path) : index_path_(path) {
}

std::vector<std::string> BM25Index::tokenize(const std::string & text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '_' || c == ':' || c == '.' || c == '-' || c == '>' ||
            c == '(' || c == ')' || c == '{' || c == '}' || c == ';' || c == ',' ||
            c == '<' || c == '=' || c == '+' || c == '*' || c == '/' || c == '&' || c == '|' ||
            c == '!') {
            current.push_back((char) std::tolower(c));
        } else if (!current.empty()) {
            if (current.size() > 1 || std::isdigit((unsigned char) current[0]) ||
                (current.size() == 1 && !std::isalpha((unsigned char) current[0]))) {
                tokens.push_back(current);
            }
            current.clear();
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

void BM25Index::add_document(int doc_id, const std::string & text) {
    auto tokens = tokenize(text);
    if (doc_id >= (int) doc_lengths_.size()) {
        doc_lengths_.resize(doc_id + 1, 0);
    }
    doc_lengths_[doc_id] = (int) tokens.size();

    std::unordered_map<std::string, int> freqs;
    for (const auto & token : tokens) {
        freqs[token]++;
    }

    for (const auto & [term, freq] : freqs) {
        inverted_index_[term].push_back({ doc_id, freq });
    }

    n_docs_++;
    build_stats();
}

void BM25Index::clear() {
    inverted_index_.clear();
    doc_lengths_.clear();
    avgdl_ = 0.0f;
    n_docs_ = 0;
}

void BM25Index::build_stats() {
    if (n_docs_ == 0) {
        avgdl_ = 0.0f;
        return;
    }

    long long total_len = 0;
    for (int len : doc_lengths_) {
        total_len += len;
    }
    avgdl_ = (float) total_len / n_docs_;
}

std::vector<std::pair<float, int>> BM25Index::search(const std::string & query, int top_k) const {
    auto q_tokens = tokenize(query);
    std::unordered_map<int, float> scores;

    for (const auto & term : q_tokens) {
        auto it = inverted_index_.find(term);
        if (it == inverted_index_.end()) {
            continue;
        }

        const auto & doc_list = it->second;
        const int docs_with_term = (int) doc_list.size();
        float idf = std::log(1.0f + (n_docs_ - docs_with_term + 0.5f) / (docs_with_term + 0.5f));
        if (idf < 0.0f) {
            idf = 0.0f;
        }

        for (const auto & entry : doc_list) {
            const int doc_len = entry.doc_id < (int) doc_lengths_.size() ? doc_lengths_[entry.doc_id] : 0;
            const float denom = entry.freq + k1_ * (1.0f - b_ + b_ * (avgdl_ > 0.0f ? doc_len / avgdl_ : 0.0f));
            if (denom > 0.0f) {
                scores[entry.doc_id] += idf * (entry.freq * (k1_ + 1.0f) / denom);
            }
        }
    }

    std::vector<std::pair<float, int>> result;
    result.reserve(scores.size());
    for (const auto & [doc_id, score] : scores) {
        result.push_back({ score, doc_id });
    }

    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    if ((int) result.size() > top_k) {
        result.resize(top_k);
    }

    return result;
}

void BM25Index::save() const {
    if (index_path_.empty()) {
        return;
    }

    json j;
    j["n_docs"] = n_docs_;
    j["avgdl"] = avgdl_;
    j["doc_lengths"] = doc_lengths_;

    json inverted = json::object();
    for (const auto & [term, entries] : inverted_index_) {
        std::vector<int> flat;
        flat.reserve(entries.size() * 2);
        for (const auto & entry : entries) {
            flat.push_back(entry.doc_id);
            flat.push_back(entry.freq);
        }
        inverted[term] = flat;
    }
    j["inverted"] = std::move(inverted);

    std::ofstream out(index_path_, std::ios::binary);
    out << j.dump();
}

bool BM25Index::load() {
    if (index_path_.empty() || !fs::exists(index_path_)) {
        return false;
    }

    std::ifstream in(index_path_, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    try {
        json j;
        in >> j;
        n_docs_ = j.value("n_docs", 0);
        avgdl_ = j.value("avgdl", 0.0f);
        doc_lengths_ = j.value("doc_lengths", std::vector<int> {});

        inverted_index_.clear();
        if (j.contains("inverted") && j["inverted"].is_object()) {
            for (auto & item : j["inverted"].items()) {
                std::vector<int> flat = item.value().get<std::vector<int>>();
                std::vector<DocEntry> entries;
                for (size_t i = 0; i + 1 < flat.size(); i += 2) {
                    entries.push_back({ flat[i], flat[i + 1] });
                }
                inverted_index_[item.key()] = std::move(entries);
            }
        }
        return true;
    } catch (...) {
        inverted_index_.clear();
        doc_lengths_.clear();
        avgdl_ = 0.0f;
        n_docs_ = 0;
        return false;
    }
}
