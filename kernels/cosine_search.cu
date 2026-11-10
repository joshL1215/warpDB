#include <kernels.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE; // 8
constexpr int K_MAX = 32;
constexpr unsigned FULL_MASK = 0xffffffff;

// Batch-aware layout:
//   - each block handles QUERIES_PER_BLOCK queries (one per warp)
//   - every stored vector is read from global memory ONCE per block and scored
//     against all of the block's queries, so a batch of B queries reads the
//     dataset ceil(B / QUERIES_PER_BLOCK) times instead of B times
//   - stored vectors are processed VECS_PER_TILE at a time to amortize the
//     block-wide reduction and __syncthreads() cost
//   - warp w owns the WarpSelect top-k list for query w of the block
constexpr int QUERIES_PER_BLOCK = NUM_WARPS;
constexpr int VECS_PER_TILE = 4;

// NOTE: queries live in static shared memory, so the kernel only supports D <= D_MAX.
// 8 queries x 768 floats = 24KB; if any dimensionality is allowed this must become dynamic.
constexpr int D_MAX = 768;

static_assert(VECS_PER_TILE <= WARP_SIZE, "one lane per tile vector is used when gathering sims");
static_assert(QUERIES_PER_BLOCK == NUM_WARPS, "each warp owns exactly one query's top-k list");

__device__ inline float warp_reduce_sum(float val) {
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

// WarpSelect insert: the warp holds a descending top-32 list, one entry per lane.
// sim must be warp-uniform (every lane passes the same value).
__device__ inline void warp_select_insert(
    float sim,
    uint64_t idx,
    float &t_key,
    uint64_t &t_val,
    float &thresh,
    int lane
) {
    if (!(sim > thresh)) return;

    unsigned int ballot = __ballot_sync(FULL_MASK, sim > t_key);
    if (ballot) {
        int insert_pos = __ffs(ballot) - 1;
        int prev_pos = max(0, lane - 1);

        float prev_key = __shfl_sync(FULL_MASK, t_key, prev_pos);
        uint64_t prev_val = __shfl_sync(FULL_MASK, t_val, prev_pos);

        if (lane > insert_pos) {
            t_key = prev_key;
            t_val = prev_val;
        } else if (lane == insert_pos) {
            t_key = sim;
            t_val = idx;
        }
    }
    thresh = __shfl_sync(FULL_MASK, t_key, WARP_SIZE - 1);
}

__global__ void cosine_search(
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
    int tid = threadIdx.x;
    int lane = tid & 31; // equivalent to tid % 32
    int warp = tid >> 5; // equivalent to tid / 32

    int query_base = blockIdx.x * QUERIES_PER_BLOCK;
    int n_local = min(QUERIES_PER_BLOCK, n_queries - query_base);

    __shared__ float s_queries[QUERIES_PER_BLOCK][D_MAX];
    // per-warp partial dot products: [warp][query][tile vector]
    __shared__ float s_partial[NUM_WARPS][QUERIES_PER_BLOCK][VECS_PER_TILE];

    // load this block's queries; pad missing queries (last block) with zeros
    for (int q = 0; q < QUERIES_PER_BLOCK; q++) {
        for (int t = tid; t < D; t += blockDim.x) {
            s_queries[q][t] = (q < n_local)
                ? d_queries[(size_t)(query_base + q) * D + t]
                : 0.0f;
        }
    }
    __syncthreads();

    // top-k state for query `warp` of this block
    float t_key = -INFINITY;
    uint64_t t_val = (uint64_t)-1;
    float thresh = -INFINITY;

    for (int tile_base = 0; tile_base < N; tile_base += VECS_PER_TILE) {

        // tombstones / tail are checked once per tile so dead vectors are never loaded
        bool alive[VECS_PER_TILE];
        #pragma unroll
        for (int j = 0; j < VECS_PER_TILE; j++) {
            int i = tile_base + j;
            alive[j] = (i < N) && !d_tombstones[i];
        }

        float acc[QUERIES_PER_BLOCK][VECS_PER_TILE];
        #pragma unroll
        for (int q = 0; q < QUERIES_PER_BLOCK; q++) {
            #pragma unroll
            for (int j = 0; j < VECS_PER_TILE; j++) acc[q][j] = 0.0f;
        }

        // each stored element is loaded once (coalesced across the block)
        // and reused for every query in the block
        for (int t = tid; t < D; t += blockDim.x) {
            float v[VECS_PER_TILE];
            #pragma unroll
            for (int j = 0; j < VECS_PER_TILE; j++) {
                v[j] = alive[j] ? d_vectors[(size_t)(tile_base + j) * D + t] : 0.0f;
            }

            #pragma unroll
            for (int q = 0; q < QUERIES_PER_BLOCK; q++) {
                float qv = s_queries[q][t];
                #pragma unroll
                for (int j = 0; j < VECS_PER_TILE; j++) {
                    acc[q][j] += qv * v[j];
                }
            }
        }

        // reduce within each warp, stash one partial per warp
        #pragma unroll
        for (int q = 0; q < QUERIES_PER_BLOCK; q++) {
            #pragma unroll
            for (int j = 0; j < VECS_PER_TILE; j++) {
                float sum = warp_reduce_sum(acc[q][j]);
                if (lane == 0) s_partial[warp][q][j] = sum;
            }
        }
        __syncthreads();

        // warp `warp` finishes the reduction for its own query and runs WarpSelect.
        // warp-uniform branch, so the full-mask shuffles inside are safe.
        if (warp < n_local) {
            // lane j (< VECS_PER_TILE) sums the cross-warp partials for tile vector j
            float my_sim = 0.0f;
            if (lane < VECS_PER_TILE) {
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) {
                    my_sim += s_partial[w][warp][lane];
                }
            }

            #pragma unroll
            for (int j = 0; j < VECS_PER_TILE; j++) {
                float sim = __shfl_sync(FULL_MASK, my_sim, j);
                if (!alive[j]) sim = -INFINITY;
                warp_select_insert(sim, (uint64_t)(tile_base + j), t_key, t_val, thresh, lane);
            }
        }
        __syncthreads(); // s_partial is overwritten next tile
    }

    if (warp < n_local && lane < k) {
        size_t out = (size_t)(query_base + warp) * k + lane;
        d_out_indices[out] = t_val;
        d_out_values[out] = t_key;
    }
}

void launch_cosine_search(
    const float *d_queries,
    const float *d_vectors,
    const uint8_t *d_tombstones,
    float *d_out_values,
    uint64_t *d_out_indices,
    int n_queries,
    int N,
    int D,
    int k
) {
    if (n_queries <= 0 || N <= 0 || D <= 0 || k <= 0) return;
    if (D > D_MAX) return;
    if (k > K_MAX) k = K_MAX;

    dim3 grid((n_queries + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK);
    cosine_search<<<grid, THREADS_PER_BLOCK>>>(
        d_queries, d_vectors, d_tombstones,
        d_out_indices, d_out_values,
        n_queries, N, D, k
    );
}
