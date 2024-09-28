#include "causal-lm.h"

__C __export struct Model *
create_model(LlamaMeta const *model,
             LlamaWeights const *weights,
             DeviceType device,
             unsigned int ndev,
             unsigned int const *dev_ids)
{
      return 0;
}

__C __export void
init(struct Model *model) {}

__C __export struct KVCache *
create_kv_cache(struct Model const *model)
{
      return 0;
}

__C __export struct KVCache *
duplicate_kv_cache(struct Model const *model,
                   struct KVCache const *cache, unsigned int seq_len)
{
      return 0;
}

__C __export void
drop_kv_cache(struct Model const *model,
              struct KVCache *cache) {}

__C __export void
infer(struct Model const *model,
      unsigned int ntok, unsigned int const *tokens,
      unsigned int nreq,
      unsigned int const *req_lens,
      unsigned int const *req_pos,
      struct KVCache *kv_caches,
      unsigned int *ans,
      float temperature, unsigned int topk, float topp) {}

__C __export void
destroy_model(struct Model *model) {}
