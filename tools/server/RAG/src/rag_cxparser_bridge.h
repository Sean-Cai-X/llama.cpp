#pragma once

#include "rag_explain.h"

json build_rag_cxparser_dry_run(
    const std::string & query,
    const json & metadata,
    int rank,
    const std::string & preview_text);
