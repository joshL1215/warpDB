#include <kernels.h>
#include <algorithm>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s line %d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA FAIL: %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

static const uint64_t EMPTY_INDEX = (uint64_t)-1;

struct SearchOutput {
    std::vector<uint64_t> indices; // [n_queries x k]
    std::vector<float> values;     // [n_queries x k]
};

// uploads inputs, runs the kernel, copies back [n_queries x k] results
static SearchOutput run_search(
    const std::vector<float> &queries,
    const std::vector<float> &vectors,
    const std::vector<uint8_t> &tombstones,
    int n_queries,
    int N,
    int D,
    int k
) {
    float *d_queries;
    float *d_vectors;
    uint8_t *d_tombstones;
    uint64_t *d_out_indices;
    float *d_out_values;

    size_t out_count = (size_t)n_queries * k;

    CUDA_CHECK(cudaMalloc(&d_queries, queries.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vectors, vectors.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tombstones, tombstones.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_out_indices, out_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_values, out_count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, vectors.data(), vectors.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tombstones, tombstones.data(), tombstones.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    launch_cosine_search(d_queries, d_vectors, d_tombstones, d_out_values, d_out_indices,
                         n_queries, N, D, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    SearchOutput out;
    out.indices.resize(out_count);
    out.values.resize(out_count);
    CUDA_CHECK(cudaMemcpy(out.indices.data(), d_out_indices, out_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.values.data(), d_out_values, out_count * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_tombstones));
    CUDA_CHECK(cudaFree(d_out_indices));
    CUDA_CHECK(cudaFree(d_out_values));

    return out;
}

static void normalize_rows(std::vector<float> &m, int rows, int D) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int t = 0; t < D; t++) sum += m[(size_t)r * D + t] * m[(size_t)r * D + t];
        float norm = sqrtf(sum);
        if (norm == 0.0f) continue;
        for (int t = 0; t < D; t++) m[(size_t)r * D + t] /= norm;
    }
}

static float cpu_dot(const std::vector<float> &a, int ai, const std::vector<float> &b, int bi, int D) {
    float sum = 0.0f;
    for (int t = 0; t < D; t++) sum += a[(size_t)ai * D + t] * b[(size_t)bi * D + t];
    return sum;
}

static void test_top_k_and_tombstones() {
    const int n_queries = 1, N = 5, D = 3, k = 3;

    std::vector<float> queries = {1.0f, 0.0f, 0.0f};
    std::vector<float> vectors = {
        1.0f, 0.0f, 0.0f,
        0.6f, 0.8f, 0.0f,
        0.0f, 1.0f, 0.0f,
        -1.0f, 0.0f, 0.0f,
        0.8f, 0.6f, 0.0f,
    };
    std::vector<uint8_t> tombstones = {0, 0, 0, 0, 1};

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    CHECK(out.indices[0] == 0);
    CHECK(fabsf(out.values[0] - 1.0f) < 1e-5f);
    CHECK(out.indices[1] == 1);
    CHECK(fabsf(out.values[1] - 0.6f) < 1e-5f);
    CHECK(out.indices[2] == 2);
    CHECK(fabsf(out.values[2] - 0.0f) < 1e-5f);

    printf("test_top_k_and_tombstones PASSED\n");
}

static void test_multiple_queries() {
    const int n_queries = 2, N = 4, D = 3, k = 2;

    std::vector<float> queries = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    std::vector<float> vectors = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.70710677f, 0.70710677f, 0.0f,
    };
    std::vector<uint8_t> tombstones = {0, 0, 0, 0};

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    CHECK(out.indices[0] == 0);
    CHECK(fabsf(out.values[0] - 1.0f) < 1e-5f);
    CHECK(out.indices[1] == 3);
    CHECK(fabsf(out.values[1] - 0.70710677f) < 1e-5f);

    CHECK(out.indices[2] == 1);
    CHECK(fabsf(out.values[2] - 1.0f) < 1e-5f);
    CHECK(out.indices[3] == 3);
    CHECK(fabsf(out.values[3] - 0.70710677f) < 1e-5f);

    printf("test_multiple_queries PASSED\n");
}

// 19 queries -> 3 blocks of 8, last block partially filled.
// query q is a one-hot on axis (q % D), vector i is one-hot on axis i, so query q's best match is q % N.
static void test_batch_spans_multiple_blocks_with_partial_last_block() {
    const int n_queries = 19, N = 7, D = 16, k = 1;

    std::vector<float> queries((size_t)n_queries * D, 0.0f);
    for (int q = 0; q < n_queries; q++) queries[(size_t)q * D + (q % N)] = 1.0f;

    std::vector<float> vectors((size_t)N * D, 0.0f);
    for (int i = 0; i < N; i++) vectors[(size_t)i * D + i] = 1.0f;

    std::vector<uint8_t> tombstones(N, 0);

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    for (int q = 0; q < n_queries; q++) {
        CHECK(out.indices[q] == (uint64_t)(q % N));
        CHECK(fabsf(out.values[q] - 1.0f) < 1e-5f);
    }

    printf("test_batch_spans_multiple_blocks_with_partial_last_block PASSED\n");
}

// N not a multiple of the vector tile; the best match sits in the final partial tile
static void test_best_match_in_tail_tile() {
    const int n_queries = 3, N = 10, D = 8, k = 2;

    std::vector<float> vectors((size_t)N * D, 0.0f);
    for (int i = 0; i < N; i++) vectors[(size_t)i * D + (i % D)] = 1.0f;
    // vector 9 is the exact match for every query, vector 1 the runner-up
    for (int t = 0; t < D; t++) vectors[(size_t)9 * D + t] = 0.0f;
    vectors[(size_t)9 * D + 0] = 0.6f;
    vectors[(size_t)9 * D + 1] = 0.8f;

    std::vector<float> queries((size_t)n_queries * D, 0.0f);
    for (int q = 0; q < n_queries; q++) {
        queries[(size_t)q * D + 0] = 0.6f;
        queries[(size_t)q * D + 1] = 0.8f;
    }

    std::vector<uint8_t> tombstones(N, 0);

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    for (int q = 0; q < n_queries; q++) {
        CHECK(out.indices[q * k + 0] == 9);
        CHECK(fabsf(out.values[q * k + 0] - 1.0f) < 1e-5f);
        CHECK(out.indices[q * k + 1] == 1);
        CHECK(fabsf(out.values[q * k + 1] - 0.8f) < 1e-5f);
    }

    printf("test_best_match_in_tail_tile PASSED\n");
}

// fewer live vectors than k: remaining slots keep the sentinel index and -inf score
static void test_k_larger_than_live_vectors_leaves_empty_slots() {
    const int n_queries = 2, N = 3, D = 4, k = 5;

    std::vector<float> vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    std::vector<float> queries = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    std::vector<uint8_t> tombstones = {0, 1, 0};

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    for (int q = 0; q < n_queries; q++) {
        int live = 0;
        for (int i = 0; i < k; i++) {
            uint64_t idx = out.indices[q * k + i];
            float val = out.values[q * k + i];
            if (i < 2) {
                CHECK(idx == 0 || idx == 2); // tombstoned vector 1 never appears
                CHECK(!isinf(val));
                live++;
            } else {
                CHECK(idx == EMPTY_INDEX);
                CHECK(isinf(val) && val < 0.0f);
            }
        }
        CHECK(live == 2);
    }
    CHECK(out.indices[0] == 0); // query 0 matches vector 0 exactly

    printf("test_k_larger_than_live_vectors_leaves_empty_slots PASSED\n");
}

// k above the kernel's 32-slot cap is clamped, and only the first 32 per query are written
static void test_k_above_max_is_clamped() {
    const int n_queries = 1, N = 40, D = 4, k = 40, K_MAX = 32;

    std::vector<float> vectors((size_t)N * D, 0.0f);
    for (int i = 0; i < N; i++) {
        vectors[(size_t)i * D + 0] = (float)(N - i); // descending similarity to the query
        vectors[(size_t)i * D + 1] = 1.0f;
    }
    normalize_rows(vectors, N, D);

    std::vector<float> queries = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<uint8_t> tombstones(N, 0);

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    for (int i = 0; i < K_MAX; i++) {
        CHECK(out.indices[i] == (uint64_t)i);
        if (i > 0) CHECK(out.values[i] <= out.values[i - 1]);
    }

    printf("test_k_above_max_is_clamped PASSED\n");
}

// full-size (768-d) random batch against a CPU brute-force reference.
// N > 256 also guards against index truncation, and a scattered set of tombstones is applied.
static void test_random_batch_matches_cpu_reference() {
    const int n_queries = 21, N = 1000, D = 768, k = 10;

    srand(1234);
    std::vector<float> queries((size_t)n_queries * D);
    std::vector<float> vectors((size_t)N * D);
    for (float &x : queries) x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (float &x : vectors) x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    normalize_rows(queries, n_queries, D);
    normalize_rows(vectors, N, D);

    std::vector<uint8_t> tombstones(N, 0);
    for (int i = 0; i < N; i += 7) tombstones[i] = 1;

    SearchOutput out = run_search(queries, vectors, tombstones, n_queries, N, D, k);

    for (int q = 0; q < n_queries; q++) {
        // CPU reference: k-th best live score for this query
        std::vector<float> scores;
        for (int i = 0; i < N; i++) {
            if (!tombstones[i]) scores.push_back(cpu_dot(queries, q, vectors, i, D));
        }
        std::vector<float> sorted = scores;
        std::sort(sorted.begin(), sorted.end(), [](float a, float b) { return a > b; });

        for (int i = 0; i < k; i++) {
            uint64_t idx = out.indices[q * k + i];
            float val = out.values[q * k + i];

            CHECK(idx < (uint64_t)N);
            CHECK(!tombstones[idx]);
            // reported score is the true score of the reported vector
            CHECK(fabsf(val - cpu_dot(queries, q, vectors, (int)idx, D)) < 1e-4f);
            // and matches the CPU's i-th best score
            CHECK(fabsf(val - sorted[i]) < 1e-4f);
            if (i > 0) CHECK(val <= out.values[q * k + i - 1]);
        }
    }

    printf("test_random_batch_matches_cpu_reference PASSED\n");
}

int main() {
    test_top_k_and_tombstones();
    test_multiple_queries();
    test_batch_spans_multiple_blocks_with_partial_last_block();
    test_best_match_in_tail_tile();
    test_k_larger_than_live_vectors_leaves_empty_slots();
    test_k_above_max_is_clamped();
    test_random_batch_matches_cpu_reference();
    printf("All cosine_search tests PASSED\n");
}
