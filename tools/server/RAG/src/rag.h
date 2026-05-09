#pragma once

#include "BM25Index.h"

#include "common.h"
#include "llama.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace faiss {
class Index;
}

struct RagConfig {
    std::string index_path;
    int         top_k         = 3;
    int         chunk_size    = 512;
    int         chunk_overlap = 64;
    bool        use_hybrid    = true;
    bool        use_faiss     = true;
    int         embedding_dim = 0;
    std::function<std::vector<float>(const std::string &)> embedding_provider;
};

struct RagSearchResult {
    std::string chunk_text;
    std::string metadata;
    float score = 0.0f;
};

class RagEngine {
public:
    explicit RagEngine(const RagConfig & config);
    ~RagEngine();

    bool init(llama_context * ctx_embed, const llama_model * model_embed, const llama_vocab * vocab);

    void add_documents(const std::vector<std::string> & docs, const std::vector<std::string> & metadata = {});
    void clear();
    std::vector<RagSearchResult> search_with_metadata(const std::string & query, int top_k, int timeout_ms = -1);
    std::vector<std::string> search(const std::string & query, int top_k, int timeout_ms = -1);

    void save();
    bool load();

    bool is_ready() const;
    size_t get_chunk_count() const;

    static std::vector<std::string> split_text_by_words(
        const std::string & text,
        int chunk_size,
        int chunk_overlap);
    static std::vector<std::string> analyze_chunks(
        const std::string & text,
        const llama_vocab * vocab,
        int chunk_size,
        int chunk_overlap);

private:
    void recreate_index_locked();
    std::vector<std::string> split_text_smart(const std::string & text) const;
    std::vector<float> get_embedding_locked(const std::string & text);
    std::vector<std::pair<float, int>> rrf_fusion(
        const std::vector<std::pair<float, int>> & dense_results,
        const std::vector<std::pair<float, int>> & sparse_results,
        int final_top_k) const;

    RagConfig config_;
    llama_context * ctx_embed_ = nullptr;
    const llama_model * model_embed_ = nullptr;
    const llama_vocab * vocab_ = nullptr;

    std::unique_ptr<faiss::Index> index_;
    BM25Index bm25_index_;
    std::vector<std::vector<float>> test_vectors_;

    std::vector<std::string> chunks_;
    std::vector<std::string> metadata_;
    std::unordered_map<std::string, std::string> file_hashes_;
    std::unordered_set<std::string> chunk_hashes_;

    mutable std::timed_mutex mutex_;
    bool ready_ = false;
};
