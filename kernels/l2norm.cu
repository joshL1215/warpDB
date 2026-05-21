__global__ void l2_normalize(float *d_vectors, int N, int D) {
    int vec_id = blockIdx.x;
    int tid = threadIdx.x;
    int vec_idx = vec_id * D;

    float partial = 0.0f;
    for (int t = tid; t < D; t += blockDim.x) {
        float v = d_vectors[vec_idx + t];
        partial += v * v;
    }

    __shared__ float sdata[256];
    sdata[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 16; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[s + tid];
        }
        __syncthreads();
    }

    __shared__ float s_inv_norm;
    if (tid < 32) {
        float res = sdata[tid];
        res += __shfl_down_sync(0xffffffff, res, 16);
        res += __shfl_down_sync(0xffffffff, res, 8);
        res += __shfl_down_sync(0xffffffff, res, 4);
        res += __shfl_down_sync(0xffffffff, res, 2);
        res += __shfl_down_sync(0xffffffff, res, 1);

        if (tid == 0) s_inv_norm = rsqrtf(res);
    }
    __syncthreads();

    float inv_norm = s_inv_norm;
    for (int t = tid; t < D; t += blockDim.x) {
        d_vectors[vec_idx + t] *= inv_norm;
    }
}

void launch_l2_normalize(float *d_vectors, int N, int D) {
    if (N <= 0) return;
    if (D <= 0) return;
    constexpr int t_per_block = 256;
    l2_normalize<<<N, t_per_block>>>(d_vectors, N, D);
}
