template <typename T>
__global__ void next_kernel(
    // 采样分布和状态
    T *logits,             // 概率分布
    unsigned int *records, // 每个 token 的出现次数
    // 词表信息
    unsigned int const n,   // 词表长度
    unsigned int const eos, // 结束符
    // 采样参数
    float const temperature, // 温度
    float const penalty,     // 重复惩罚
    unsigned int const *tok  // 上一次采样结果
) {
    unsigned int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    // 更新出现次数
    if (!tok) {
        records[i] = 0;
    } else if (i == *tok) {
        ++records[i];
    }
    // 调整分布
    if (!tok && i == eos) {
        // 第一轮解码绝不产生 eos
        ((unsigned int *)logits)[i] = 0xFF800000; // float -∞
    } else {
        T scale = temperature * powf(penalty, records[i]);
        if (((float)logits[i]) > .0) {
            logits[i] /= scale;
        } else {
            logits[i] *= scale;
        }
    }
}
