#ifndef __INFINI_INFER_H__
#define __INFINI_INFER_H__

#include "export.h"

typedef enum
{
    DATA_TYPE_F32,
    DATA_TYPE_F16,
} DataType;

typedef enum
{
    DEVICE_CPU,
    DEVICE_NVIDIA,
    DEVICE_CAMBRICON,
} DeviceType;

////////////////// Models //////////////////
typedef struct
{
    DataType dt_norm, dt_mat;
    unsigned int nlayer, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
} LlamaMeta;

typedef struct
{
    unsigned int nlayer;
    // clang-format off
    void const
        *input_embd,
        *ouput_norm,
        *output_embd,
        *const *attn_norm,
        *const *attn_qkv,
        *const *attn_o,
        *const *ffn_norm,
        *const *ffn_gate_up,
        *const *ffn_down;
    // clang-format on
} LlamaWeights;

//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Model *
create_model(LlamaMeta const *,
             LlamaWeights const *,
             DeviceType device,
             unsigned int ndev,
             unsigned int const *dev_ids);

/// @brief 初始化模型，表示所有权重已经传入完毕，将检查模型完整性并做其他准备工作
__C __export void
init(struct Model *);

/// @brief 创建 KV Cache
__C __export struct KVCache *
create_kv_cache(struct Model const *);

/// @brief 复制 KV Cache
__C __export struct KVCache *
duplicate_kv_cache(struct Model const *,
                   struct KVCache const *, unsigned int seq_len);

/// @brief 销毁 KV Cache
__C __export void
drop_kv_cache(struct Model const *,
              struct KVCache *);

/// @brief 推理
/// @param ntok 输入 token 总数
/// @param tokens 输入 token
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param ans 每个请求的输出 token
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
__C __export void
infer(struct Model const *,
      unsigned int ntok, unsigned int const *tokens,
      unsigned int nreq,
      unsigned int const *req_lens,
      unsigned int const *req_pos,
      struct KVCache *kv_caches,
      unsigned int *ans,
      float temperature, unsigned int topk, float topp);

/// @brief 销毁模型
__C __export void
destroy_model(struct Model *);

#endif // __INFINI_INFER_H__
