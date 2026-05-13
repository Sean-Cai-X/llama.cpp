#include "rag.h"

#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

bool metadata_contains_flag(const std::string & metadata, const std::string & flag) {
    if (metadata.empty() || flag.empty()) {
        return false;
    }

    if (metadata == flag) {
        return true;
    }

    if (metadata.find(flag + ";") == 0) {
        return true;
    }

    if (metadata.rfind(";" + flag) != std::string::npos) {
        return true;
    }

    return false;
}

} // namespace

RagEngine::RagEngine(const RagConfig & config)
    : config_(config),
      bm25_index_(config.index_path.empty() ? "" : config.index_path + ".bm25.json") {
}

RagEngine::~RagEngine() {
    save();
}

void RagEngine::recreate_index_locked() {
    if (!config_.use_faiss) {
        index_.reset();
        return;
    }

    const int embedding_dim = config_.embedding_provider ? config_.embedding_dim :
        (model_embed_ ? llama_model_n_embd(model_embed_) : 0);
    if (embedding_dim > 0) {
        index_ = std::make_unique<faiss::IndexFlatIP>(embedding_dim);
    } else {
        index_.reset();
    }
}

bool RagEngine::init(llama_context * ctx_embed, const llama_model * model_embed, const llama_vocab * vocab) {
    std::lock_guard<std::timed_mutex> lock(mutex_);

    if (config_.embedding_provider) {
        if (config_.embedding_dim <= 0) {
            LOG_ERR("%s: embedding_dim must be > 0 when using embedding_provider\n", __func__);
            return false;
        }

        ctx_embed_ = nullptr;
        model_embed_ = nullptr;
        vocab_ = vocab;

        if (load()) {
            ready_ = true;
            return true;
        }

        recreate_index_locked();
        ready_ = true;
        return true;
    }

    ctx_embed_ = ctx_embed;
    model_embed_ = model_embed;
    vocab_ = vocab;

    if (!ctx_embed_ || !model_embed_ || !vocab_) {
        LOG_ERR("%s: invalid embedding context\n", __func__);
        return false;
    }

    if (load()) {
        ready_ = true;
        return true;
    }

    recreate_index_locked();
    ready_ = true;
    return true;
}

bool RagEngine::is_ready() const {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    return ready_;
}

size_t RagEngine::get_chunk_count() const {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    return chunks_.size();
}

void RagEngine::clear() {
    std::lock_guard<std::timed_mutex> lock(mutex_);

    chunks_.clear();
    metadata_.clear();
    file_hashes_.clear();
    chunk_hashes_.clear();
    test_vectors_.clear();
    bm25_index_.clear();
    recreate_index_locked();
}

std::vector<std::string> RagEngine::split_text_by_words(
    const std::string & text,
    int chunk_size,
    int chunk_overlap) {
    std::vector<std::string> chunks;
    if (text.empty() || chunk_size <= 0) {
        return chunks;
    }

    std::istringstream stream(text);
    std::vector<std::string> words;
    for (std::string word; stream >> word;) {
        words.push_back(std::move(word));
    }

    if (words.empty()) {
        return chunks;
    }

    const size_t step = (size_t) std::max(1, chunk_size - std::max(0, chunk_overlap));
    for (size_t i = 0; i < words.size(); i += step) {
        const size_t end = std::min(words.size(), i + (size_t) chunk_size);
        std::string chunk;
        for (size_t j = i; j < end; ++j) {
            if (!chunk.empty()) {
                chunk += ' ';
            }
            chunk += words[j];
        }
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        if (end >= words.size()) {
            break;
        }
    }

    return chunks;
}

std::vector<std::string> RagEngine::analyze_chunks(
    const std::string & text,
    const llama_vocab * vocab,
    int chunk_size,
    int chunk_overlap) {
    if (text.empty()) {
        return {};
    }

    if (!vocab) {
        return split_text_by_words(text, chunk_size, chunk_overlap);
    }

    const auto tokens = common_tokenize(vocab, text, false, false);
    if (tokens.empty()) {
        return {};
    }

    if ((int) tokens.size() <= chunk_size) {
        return { text };
    }

    std::vector<std::string> chunks;
    const size_t step = (size_t) std::max(1, chunk_size - chunk_overlap);
    for (size_t i = 0; i < tokens.size(); i += step) {
        const size_t end = std::min(tokens.size(), i + (size_t) chunk_size);
        llama_tokens chunk_tokens(tokens.begin() + i, tokens.begin() + end);
        std::string chunk_text = common_detokenize(vocab, chunk_tokens, false);
        if (!chunk_text.empty()) {
            chunks.push_back(std::move(chunk_text));
        }
        if (end >= tokens.size()) {
            break;
        }
    }

    return chunks;
}

std::vector<std::string> RagEngine::split_text_smart(const std::string & text) const {
    return analyze_chunks(text, vocab_, config_.chunk_size, config_.chunk_overlap);
}

std::vector<float> RagEngine::get_embedding_locked(const std::string & text) {
    if (config_.embedding_provider) {
        auto output = config_.embedding_provider(text);
        float norm = 0.0f;
        for (float v : output) {
            norm += v * v;
        }
        if (norm > 0.0f) {
            const float inv = 1.0f / std::sqrt(norm);
            for (float & v : output) {
                v *= inv;
            }
        }
        return output;
    }

    if (!ctx_embed_ || !model_embed_ || !vocab_) {
        return {};
    }

    auto tokens = common_tokenize(vocab_, text, false, false);
    if (tokens.empty()) {
        return {};
    }

    if (llama_model_has_encoder(model_embed_)) {
        const uint32_t n_ubatch = llama_n_ubatch(ctx_embed_);
        if (n_ubatch > 0 && tokens.size() > n_ubatch) {
            LOG_WRN(
                "%s: skipping embedding because token_count=%zu exceeds encoder n_ubatch=%u\n",
                __func__,
                tokens.size(),
                n_ubatch);
            return {};
        }
    }

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (int32_t) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }

    if (auto * memory = llama_get_memory(ctx_embed_)) {
        llama_memory_seq_rm(memory, 0, -1, -1);
    }

    if (llama_encode(ctx_embed_, batch) < 0) {
        llama_batch_free(batch);
        return {};
    }

    const int n_embd = llama_model_n_embd(model_embed_);
    std::vector<float> output(n_embd, 0.0f);

    const float * embd = llama_get_embeddings_seq(ctx_embed_, 0);
    if (!embd) {
        embd = llama_get_embeddings_ith(ctx_embed_, (int32_t) tokens.size() - 1);
    }
    if (!embd) {
        embd = llama_get_embeddings(ctx_embed_);
    }

    if (embd) {
        std::memcpy(output.data(), embd, n_embd * sizeof(float));

        float norm = 0.0f;
        for (float v : output) {
            norm += v * v;
        }
        if (norm > 0.0f) {
            const float inv = 1.0f / std::sqrt(norm);
            for (float & v : output) {
                v *= inv;
            }
        }
    }

    if (auto * memory = llama_get_memory(ctx_embed_)) {
        llama_memory_seq_rm(memory, 0, -1, -1);
    }

    llama_batch_free(batch);
    return output;
}

void RagEngine::add_documents(const std::vector<std::string> & docs, const std::vector<std::string> & metadata) {
    std::lock_guard<std::timed_mutex> lock(mutex_);

    if (!ready_) {
        return;
    }

    const int n_embd = config_.embedding_provider ? config_.embedding_dim : llama_model_n_embd(model_embed_);
    std::vector<float> all_vectors;

    for (size_t doc_idx = 0; doc_idx < docs.size(); ++doc_idx) {
        const auto & doc = docs[doc_idx];
        const bool is_prechunked = doc_idx < metadata.size() && metadata_contains_flag(metadata[doc_idx], "prechunked=1");
        auto split_chunks = is_prechunked ? std::vector<std::string> { doc } : split_text_smart(doc);
        const size_t chunk_count = split_chunks.size();

        for (size_t chunk_idx = 0; chunk_idx < split_chunks.size(); ++chunk_idx) {
            const auto & chunk = split_chunks[chunk_idx];
            const std::string chunk_hash = std::to_string(std::hash<std::string>{}(chunk));
            if (!chunk_hashes_.insert(chunk_hash).second) {
                continue;
            }

            auto vec = get_embedding_locked(chunk);
            if ((int) vec.size() != n_embd) {
                continue;
            }

            if (config_.use_faiss) {
                all_vectors.insert(all_vectors.end(), vec.begin(), vec.end());
            } else {
                test_vectors_.push_back(vec);
            }
            chunks_.push_back(chunk);
            bm25_index_.add_document((int) chunks_.size() - 1, chunk);

            if (doc_idx < metadata.size()) {
                std::ostringstream meta_stream;
                meta_stream << metadata[doc_idx]
                            << ";chunk_index=" << chunk_idx
                            << ";chunk_count=" << chunk_count;
                metadata_.push_back(meta_stream.str());
            } else {
                json meta = {
                    {"type", "document"},
                    {"source_doc", (int) doc_idx},
                    {"chunk_id", (int) chunks_.size() - 1},
                    {"chunk_index", (int) chunk_idx},
                    {"chunk_count", (int) chunk_count},
                };
                metadata_.push_back(meta.dump());
            }
        }

        file_hashes_["doc_" + std::to_string(doc_idx)] = std::to_string(std::hash<std::string>{}(doc));
    }

    if (config_.use_faiss && index_ && !all_vectors.empty()) {
        index_->add((faiss::idx_t) (all_vectors.size() / n_embd), all_vectors.data());
    }
}

std::vector<std::pair<float, int>> RagEngine::rrf_fusion(
    const std::vector<std::pair<float, int>> & dense_results,
    const std::vector<std::pair<float, int>> & sparse_results,
    int final_top_k) const {
    std::unordered_map<int, float> scores;
    const int k = 60;

    for (size_t i = 0; i < dense_results.size(); ++i) {
        scores[dense_results[i].second] += 1.0f / (k + (int) i + 1);
    }
    for (size_t i = 0; i < sparse_results.size(); ++i) {
        scores[sparse_results[i].second] += 1.0f / (k + (int) i + 1);
    }

    std::vector<std::pair<float, int>> fused;
    fused.reserve(scores.size());
    for (const auto & [doc_id, score] : scores) {
        fused.push_back({ score, doc_id });
    }

    std::sort(fused.begin(), fused.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    if ((int) fused.size() > final_top_k) {
        fused.resize(final_top_k);
    }

    return fused;
}

std::vector<RagSearchResult> RagEngine::search_with_metadata(const std::string & query, int top_k, int timeout_ms) {
    if (top_k <= 0) {
        throw std::invalid_argument("top_k must be greater than 0");
    }

    std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
    if (timeout_ms >= 0) {
        if (!lock.try_lock_for(std::chrono::milliseconds(timeout_ms))) {
            throw std::runtime_error("RAG search timed out while waiting for the index");
        }
    } else {
        lock.lock();
    }

    if (!ready_) {
        return {};
    }

    auto query_vec = get_embedding_locked(query);
    const size_t expected_dim = config_.use_faiss && index_ ? (size_t) index_->d : test_vectors_.empty() ? 0 : test_vectors_.front().size();
    if (query_vec.empty() || query_vec.size() != expected_dim) {
        return {};
    }

    std::vector<std::pair<float, int>> dense_results;
    if (config_.use_faiss && index_) {
        std::vector<faiss::idx_t> ids(top_k);
        std::vector<float> distances(top_k);
        index_->search(1, query_vec.data(), top_k, distances.data(), ids.data());

        for (int i = 0; i < top_k; ++i) {
            if (ids[i] >= 0 && ids[i] < (faiss::idx_t) chunks_.size()) {
                dense_results.push_back({ distances[i], (int) ids[i] });
            }
        }
    } else {
        dense_results.reserve(test_vectors_.size());
        for (size_t i = 0; i < test_vectors_.size(); ++i) {
            float score = 0.0f;
            for (size_t j = 0; j < query_vec.size(); ++j) {
                score += query_vec[j] * test_vectors_[i][j];
            }
            dense_results.push_back({ score, (int) i });
        }

        std::sort(dense_results.begin(), dense_results.end(), [](const auto & a, const auto & b) {
            return a.first > b.first;
        });
        if ((int) dense_results.size() > top_k) {
            dense_results.resize(top_k);
        }
    }

    std::vector<RagSearchResult> results;
    if (config_.use_hybrid) {
        auto sparse_results = bm25_index_.search(query, top_k * 2);
        auto fused = rrf_fusion(dense_results, sparse_results, top_k);
        for (const auto & [score, doc_id] : fused) {
            if (doc_id >= 0 && doc_id < (int) chunks_.size()) {
                results.push_back({
                    chunks_[doc_id],
                    doc_id < (int) metadata_.size() ? metadata_[doc_id] : "",
                    score,
                });
            }
        }
    } else {
        for (const auto & [score, doc_id] : dense_results) {
            if (doc_id >= 0 && doc_id < (int) chunks_.size()) {
                results.push_back({
                    chunks_[doc_id],
                    doc_id < (int) metadata_.size() ? metadata_[doc_id] : "",
                    score,
                });
            }
        }
    }

    return results;
}

std::vector<std::string> RagEngine::search(const std::string & query, int top_k, int timeout_ms) {
    auto detailed = search_with_metadata(query, top_k, timeout_ms);
    std::vector<std::string> results;
    results.reserve(detailed.size());
    for (const auto & item : detailed) {
        results.push_back(item.chunk_text);
    }
    return results;
}

void RagEngine::save() {
    std::lock_guard<std::timed_mutex> lock(mutex_);

    if (!ready_ || !index_ || config_.index_path.empty()) {
        return;
    }

    try {
        faiss::write_index(index_.get(), config_.index_path.c_str());
        bm25_index_.save();

        {
            std::ofstream out(config_.index_path + ".chunks.json", std::ios::binary);
            out << json(chunks_).dump();
        }
        {
            std::ofstream out(config_.index_path + ".meta.json", std::ios::binary);
            out << json(metadata_).dump();
        }
        {
            json hashes;
            hashes["file_hashes"] = file_hashes_;
            hashes["chunk_hashes"] = std::vector<std::string>(chunk_hashes_.begin(), chunk_hashes_.end());
            std::ofstream out(config_.index_path + ".hashes.json", std::ios::binary);
            out << hashes.dump();
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to save RAG index: %s\n", __func__, e.what());
    }
}

bool RagEngine::load() {
    if (config_.index_path.empty() || !fs::exists(config_.index_path)) {
        return false;
    }

    try {
        faiss::Index * raw_index = faiss::read_index(config_.index_path.c_str());
        if (!raw_index) {
            return false;
        }
        index_.reset(raw_index);

        bm25_index_.load();

        {
            std::ifstream in(config_.index_path + ".chunks.json", std::ios::binary);
            if (in.is_open()) {
                json j;
                in >> j;
                chunks_ = j.get<std::vector<std::string>>();
            }
        }
        {
            std::ifstream in(config_.index_path + ".meta.json", std::ios::binary);
            if (in.is_open()) {
                json j;
                in >> j;
                metadata_ = j.get<std::vector<std::string>>();
            }
        }
        {
            std::ifstream in(config_.index_path + ".hashes.json", std::ios::binary);
            if (in.is_open()) {
                json j;
                in >> j;
                file_hashes_ = j.value("file_hashes", std::unordered_map<std::string, std::string> {});
                auto chunk_hashes = j.value("chunk_hashes", std::vector<std::string> {});
                chunk_hashes_.clear();
                chunk_hashes_.insert(chunk_hashes.begin(), chunk_hashes.end());
            }
        }

        return true;
    } catch (...) {
        index_.reset();
        return false;
    }
}
