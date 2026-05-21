#include <kernels.h>

__global__ void compute_dot_product(
    const float *d_queries,
    const float *d_vectors,
    const uint8_t *d_tombstones,
    float *d_similarities,
    float *d_top_k_values,
    uint64_t *d_top_k_indices,
    int n_queries,
    int N,
    int D,
    int k
) {

    int tid = threadIdx.x;
    int vec_id = blockIdx.x;
    int query_id = blockIdx.y;

    if (d_tombstones[vec_id]) {
        if (tid == 0) {
            d_similarities[query_id * N + vec_id] = -INFINITY;
        }
        return;
    }

    int v_idx = vec_id * D;
    int q_idx = query_id * D;

    float partial = 0.0f;
    for (int t = tid; t < D; t += blockDim.x) {
        partial += d_queries[q_idx + t] * d_vectors[v_idx + t];
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

        if (tid == 0) d_similarities[query_id * N + vec_id] = res;
    }
}

void launch_compute_dot_product(
    const float *d_queries,
    const float *d_vectors,
    const uint8_t *d_tombstones,
    float *d_similarities,
    float *d_top_k_values,
    uint64_t *d_top_k_indices,
    int n_queries,
    int N,
    int D,
    int k
) {
    if (n_queries <= 0 || N <= 0 || D <= 0) return;
    dim3 grid(N, n_queries);
    compute_dot_product<<<grid, 256, 0>>>(
        d_queries, d_vectors,
        d_tombstones, d_similarities,
        d_top_k_values, d_top_k_indices,
        n_queries, N, D, k
    );
}
