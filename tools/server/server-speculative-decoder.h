#pragma once

#include "server-metrics.h"
#include "server-slot.h"
#include "server-token-processor.h"

#include <vector>

namespace server_speculative_decoder {

    struct runtime {
        server_token_processor::runtime token_rt;

        server_metrics* metrics = nullptr;

        bool special = false;
    };

    void process_accepted_drafts(
        const runtime& rt,
        std::vector<server_slot>& slots);

} // namespace server_speculative_decoder