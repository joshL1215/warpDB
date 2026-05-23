#include <kernels.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int K_MAX = 32;

__device__ inline float block_reduce_sum(float val, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
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
        return res;
    }
    return 0.0f;
}

__global__ void cosine_search(
    const float *d_queries,
    const float *d_vectors,
    const uint8_t *d_tombstones,
    uint64_t *d_out_indices,
    float *d_out_values,
    int N,
    int D,
    int k
) {
    int query_id = blockIdx.x;
    int tid = threadIdx.x;

    int lane = tid & 31; // equivalent to tid % 32
    int warp = tid >> 5; // on 256 threads, there are 8 warps, so >> 5 AKA div 32 marks what warp we're in

    // NOTE: this makes this kernel only compatible with 768 dim embeddings, though D is still passed in case we generalize later
    // if any dimensionality allowed we need to change shared mem allocation to dynamic
    __shared__ float s_query[768];
    __shared__ float s_reduce[256];
    __shared__ float s_sim;

    for (int t = tid; t < D; t += blockDim.x) {
        s_query[t] = d_queries[query_id * D + t];
    }
    __syncthreads();

    float t_key = -INFINITY;
    uint64_t t_val = (uint64_t)-1;
    float thresh = -INFINITY;

    for (int i = 0; i < N; i++) {

        bool dead = d_tombstones[i];
        float sim;
        if (!dead) {

            float partial = 0.0f;
            for (int t = tid; t < D; t += blockDim.x) {
                partial += s_query[t] * d_vectors[i * D + t];
            }
            float block_sum = block_reduce_sum(partial, s_reduce);

            // Sharing the computed similarity across each thread
            if (tid == 0) {
                s_sim = block_sum;
            }
            __syncthreads();
            sim = s_sim;

        } else {
            sim = -INFINITY;
        }

        // WarpSelect
        if (warp == 0 && sim > thresh) {
            unsigned int ballot = __ballot_sync(0xffffffff, sim > t_key);
            if (ballot) {
                int insert_pos = __ffs(ballot) - 1;
                int prev_pos = max(0, lane - 1);

                float prev_key = __shfl_sync(0xffffffff, t_key, prev_pos);
                uint64_t prev_val = __shfl_sync(0xffffffff, t_val, prev_pos);

                if (lane > insert_pos) {
                    t_key = prev_key;
                    t_val = prev_val;
                } else if (lane == insert_pos) {
                    t_key = sim;
                    t_val = (uint8_t)i;
                }
            }
            thresh = __shfl_sync(0xffffffff, t_key, 31);
        }
        __syncthreads();
    }

    if (warp == 0 && lane < k) {
        d_out_indices[query_id * k + lane] = t_val;
        d_out_values[query_id * k + lane] = t_key;
    }
}

void launch_cosine_search(
    const float *d_queries,
    const float *d_vectors,
    const uint8_t *d_tombstones,
    uint64_t *d_out_indices,
    float *d_out_values,
    int n_queries,
    int N,
    int D,
    int k
) {
    if (n_queries <= 0 || N <= 0 || D <= 0 || k <= 0) return;
    if (k > K_MAX) k = K_MAX;

    dim3 grid(n_queries);
    cosine_search<<<grid, THREADS_PER_BLOCK>>>(
        d_queries, d_vectors, d_tombstones,
        d_out_indices, d_out_values,
        N, D, k
    );
}
