#ifndef SERVER_METRICS_H
#define SERVER_METRICS_H

#include <cstdint>
#include <string>

/**
 * @brief 负责收集和记录推理过程中的性能指标。
 *        这是一个数据收集器，不参与流程控制。
 */
class ServerMetrics {
public:
    // 构造函数和初始化
    ServerMetrics();
    void Initialize();

    // --- 计数器和计时器更新接口 ---

    // 提示词处理 (Prompt Processing)
    void IncrementPromptTokensProcessed(uint64_t count);
    void SetPromptProcessingTime(int64_t duration_us);

    // Token 生成 (Token Generation)
    void IncrementTokensPredicted(uint64_t count);
    void SetTokenGenerationTime(int64_t duration_us);
    
    // 内存管理/批次
    void IncrementDecodeTotal(uint64_t count);
    void IncrementBusySlotsTotal(uint64_t count);

    // 推测解码相关
    void SetDraftAcceptanceRate(float rate, uint64_t accepted, uint64_t total);

    // --- 报告接口 ---
    std::string GenerateStatsReport() const;

private:
    // 内部存储状态 (对应原始结构体)
    int64_t t_start_us = 0;

    // 提示词相关
    uint64_t n_prompt_tokens_processed_total = 0;
    int64_t t_prompt_processing_total_us = 0;

    // Token 生成相关
    uint64_t n_tokens_predicted_total = 0;
    int64_t t_tokens_generation_total_us = 0;

    // 其他
    uint64_t n_tokens_max = 0; 
    uint64_t n_decode_total = 0;
    uint64_t n_busy_slots_total = 0;
};

#endif // SERVER_METRICS_H