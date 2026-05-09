#pragma once

#include "rag.h"
#include "vendor/nlohmann/json.hpp"

#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct RagChatMinimalOptions {
    bool include_prompt_context = true;
    bool include_explanations = true;
    bool include_cxparser_dry_run = false;
    size_t max_preview_chars = 240;
};

json build_rag_chat_minimal_response(
    const std::string & query,
    int top_k,
    const std::string & retrieval_mode,
    const std::vector<RagSearchResult> & results,
    const RagChatMinimalOptions & options = {});
