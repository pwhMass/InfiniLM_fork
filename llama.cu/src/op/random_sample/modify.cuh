template <typename T>
__global__ void next_kernel(
    T *logits,
    T *scale_,
    unsigned int const n,
    float const temperature,
    float const penalty,
    unsigned int const *tok,
    unsigned int const eos) {
    unsigned int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float scale;
    if (!tok) {
        // 初始化惩罚权重
        scale = i == eos ? 0 : 1;
        scale_[i] = 1;
    } else {
        // 更新惩罚权重
        scale = (float)scale_[i];
        if (i == *tok) {
            scale *= penalty;
            scale_[i] = scale;
        }
    }
    if (((float)logits[i]) > .0) {
        logits[i] *= (T)(temperature * scale);
    } else {
        logits[i] /= (T)(temperature * scale);
    }
}
