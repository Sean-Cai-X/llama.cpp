#include "ServerMetrics.h"
#include <sstream>
#include <iomanip>
#include <iostream> // 仅为示例，实际项目中需替换为日志库

ServerMetrics::ServerMetrics() = default;

void ServerMetrics::Initialize() {
    // 在实际项目中，这里应使用系统高精度计时器初始化 t_start_us
    t_start_us = 0; 
}

// --- 计数器和计时器更新实现 ---

void ServerMetrics::IncrementPromptTokensProcessed(uint64_t count) {
    n_prompt_tokens_processed_total += count;
}

void ServerMetrics::SetPromptProcessingTime(int64_t duration_us) {
    t_prompt_processing_total_us = duration_us;
}

void ServerMetrics::IncrementTokensPredicted(uint64_t count) {
    n_tokens_predicted_total += count;
}

void ServerMetrics::SetTokenGenerationTime(int64_t duration_us) {
    t_tokens_generation_total_us = duration_us;
}

void ServerMetrics::IncrementDecodeTotal(uint64_t count) {
    n_decode_total += count;
}

void ServerMetrics::IncrementBusySlotsTotal(uint64_t count) {
    n_busy_slots_total += count;
}

void ServerMetrics::SetDraftAcceptanceRate(float rate, uint64_t accepted, uint64_t total) {
    // 这是一个统计值，通常在报告时计算，这里只是存储原始数据
    // 为了简洁，我们只存储最终率
    // 实际代码中可能需要更复杂的统计逻辑
}

std::string ServerMetrics::GenerateStatsReport() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);

    float prompt_time_s = static_cast<float>(t_prompt_processing_total_us) / 1000.0f;
    float generation_time_s = static_cast<float>(t_tokens_generation_total_us) / 1000.0f;

    ss << "=== Server Performance Metrics ===\n";
    ss << "Prompt Processing Time: " << prompt_time_s << "s (Tokens: " << n_prompt_tokens_processed_total << ")\n";
    ss << "Token Generation Time: " << generation_time_s << "s (Tokens: " << n_tokens_predicted_total << ")\n";
    ss << "Total Decoded Tokens: " << n_decode_total << "\n";
    ss << "Busy Slots: " << n_busy_slots_total << "\n";
    ss << "=================================\n";

    return ss.str();
}