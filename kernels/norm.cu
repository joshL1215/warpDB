#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void compute_norms(const float *d_vectors, float *d_norms, int N, int D) {

    int vec_id = blockIdx.x;
    int tid = threadIdx.x;

    float partial = 0.0f;
    for (int t = tid; t < D; t += blockDim.x) {
        float v = d_vectors[vec_id * D + t];
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

    if (tid < 32) {
        float res = sdata[tid];
        res += __shfl_down_sync(0xffffffff, res, 16);
        res += __shfl_down_sync(0xffffffff, res, 8);
        res += __shfl_down_sync(0xffffffff, res, 4);
        res += __shfl_down_sync(0xffffffff, res, 2);
        res += __shfl_down_sync(0xffffffff, res, 1);

        if (tid == 0) d_norms[vec_id] = res;
    }
}

void launch_compute_norms(const float *d_vectors, float *d_norms, int N, int D) {
    if (N <= 0) return;
    if (D <= 0) return;
    constexpr int t_per_block = 256;
    compute_norms<<N, t_per_block>>(d_vectors, d_norms, N, D);
}
