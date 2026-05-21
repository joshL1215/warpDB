#include <kernels.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s line %d\n", __FILE__, __LINE__); exit(1); } } while (0)

static void setup(
    int n_queries, int N, int D, int k,
    float *h_queries, float *h_vecs, uint8_t *h_tomb,
    float **d_queries, float **d_vecs, uint8_t **d_tomb,
    float **d_sims, float **d_top_k_values, uint64_t **d_top_k_indices
) {
    cudaMalloc(d_queries,       n_queries * D * sizeof(float));
    cudaMalloc(d_vecs,          N * D * sizeof(float));
    cudaMalloc(d_tomb,          N * sizeof(uint8_t));
    cudaMalloc(d_sims,          n_queries * N * sizeof(float));
    cudaMalloc(d_top_k_values,  n_queries * k * sizeof(float));
    cudaMalloc(d_top_k_indices, n_queries * k * sizeof(uint64_t));
    cudaMemcpy(*d_queries, h_queries, n_queries * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_vecs,    h_vecs,    N * D * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(*d_tomb,    h_tomb,    N * sizeof(uint8_t),           cudaMemcpyHostToDevice);
}

static void teardown(
    float *d_queries, float *d_vecs, uint8_t *d_tomb,
    float *d_sims, float *d_top_k_values, uint64_t *d_top_k_indices
) {
    cudaFree(d_queries);
    cudaFree(d_vecs);
    cudaFree(d_tomb);
    cudaFree(d_sims);
    cudaFree(d_top_k_values);
    cudaFree(d_top_k_indices);
}

// Unit vectors pointing same direction: dot product = 1
static void test_parallel_unit_vectors() {
    int n_queries = 1, N = 1, D = 3, k = 1;
    float h_q[]    = {1.0f, 0.0f, 0.0f};
    float h_vecs[] = {1.0f, 0.0f, 0.0f};
    uint8_t h_tomb[] = {0};

    float *d_q, *d_vecs, *d_sims, *d_topv; uint8_t *d_tomb; uint64_t *d_topi;
    setup(n_queries, N, D, k, h_q, h_vecs, h_tomb, &d_q, &d_vecs, &d_tomb, &d_sims, &d_topv, &d_topi);
    launch_compute_dot_product(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi, n_queries, N, D, k);
    cudaDeviceSynchronize();

    float h_sims[1];
    cudaMemcpy(h_sims, d_sims, sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(fabsf(h_sims[0] - 1.0f) < 1e-5f);

    teardown(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi);
    printf("test_parallel_unit_vectors PASSED\n");
}

// Orthogonal vectors: dot product = 0
static void test_orthogonal_vectors() {
    int n_queries = 1, N = 1, D = 3, k = 1;
    float h_q[]    = {1.0f, 0.0f, 0.0f};
    float h_vecs[] = {0.0f, 1.0f, 0.0f};
    uint8_t h_tomb[] = {0};

    float *d_q, *d_vecs, *d_sims, *d_topv; uint8_t *d_tomb; uint64_t *d_topi;
    setup(n_queries, N, D, k, h_q, h_vecs, h_tomb, &d_q, &d_vecs, &d_tomb, &d_sims, &d_topv, &d_topi);
    launch_compute_dot_product(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi, n_queries, N, D, k);
    cudaDeviceSynchronize();

    float h_sims[1];
    cudaMemcpy(h_sims, d_sims, sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(fabsf(h_sims[0]) < 1e-5f);

    teardown(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi);
    printf("test_orthogonal_vectors PASSED\n");
}

// Tombstoned vector: similarity must be -INFINITY
static void test_tombstone() {
    int n_queries = 1, N = 2, D = 2, k = 1;
    float h_q[]    = {1.0f, 0.0f};
    float h_vecs[] = {1.0f, 0.0f,
                      1.0f, 0.0f};
    uint8_t h_tomb[] = {0, 1};

    float *d_q, *d_vecs, *d_sims, *d_topv; uint8_t *d_tomb; uint64_t *d_topi;
    setup(n_queries, N, D, k, h_q, h_vecs, h_tomb, &d_q, &d_vecs, &d_tomb, &d_sims, &d_topv, &d_topi);
    launch_compute_dot_product(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi, n_queries, N, D, k);
    cudaDeviceSynchronize();

    float h_sims[2];
    cudaMemcpy(h_sims, d_sims, N * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(fabsf(h_sims[0] - 1.0f) < 1e-5f);
    CHECK(isinf(h_sims[1]) && h_sims[1] < 0.0f);

    teardown(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi);
    printf("test_tombstone PASSED\n");
}

// Multiple queries: check output layout is [query_id * N + vec_id]
static void test_multiple_queries() {
    int n_queries = 2, N = 2, D = 2, k = 1;
    float h_q[]    = {1.0f, 0.0f,
                      0.0f, 1.0f};
    float h_vecs[] = {1.0f, 0.0f,
                      0.0f, 1.0f};
    uint8_t h_tomb[] = {0, 0};

    float *d_q, *d_vecs, *d_sims, *d_topv; uint8_t *d_tomb; uint64_t *d_topi;
    setup(n_queries, N, D, k, h_q, h_vecs, h_tomb, &d_q, &d_vecs, &d_tomb, &d_sims, &d_topv, &d_topi);
    launch_compute_dot_product(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi, n_queries, N, D, k);
    cudaDeviceSynchronize();

    float h_sims[4];
    cudaMemcpy(h_sims, d_sims, n_queries * N * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(fabsf(h_sims[0 * N + 0] - 1.0f) < 1e-5f);
    CHECK(fabsf(h_sims[0 * N + 1] - 0.0f) < 1e-5f);
    CHECK(fabsf(h_sims[1 * N + 0] - 0.0f) < 1e-5f);
    CHECK(fabsf(h_sims[1 * N + 1] - 1.0f) < 1e-5f);

    teardown(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi);
    printf("test_multiple_queries PASSED\n");
}

// D > 256 to exercise the multi-block reduction path
static void test_wide_vector() {
    int n_queries = 1, N = 1, D = 512, k = 1;
    float h_q[512], h_vecs[512];
    float val = 1.0f / sqrtf((float)D);
    for (int i = 0; i < D; i++) h_q[i] = h_vecs[i] = val;
    uint8_t h_tomb[] = {0};

    float *d_q, *d_vecs, *d_sims, *d_topv; uint8_t *d_tomb; uint64_t *d_topi;
    setup(n_queries, N, D, k, h_q, h_vecs, h_tomb, &d_q, &d_vecs, &d_tomb, &d_sims, &d_topv, &d_topi);
    launch_compute_dot_product(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi, n_queries, N, D, k);
    cudaDeviceSynchronize();

    float h_sims[1];
    cudaMemcpy(h_sims, d_sims, sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(fabsf(h_sims[0] - 1.0f) < 1e-4f);

    teardown(d_q, d_vecs, d_tomb, d_sims, d_topv, d_topi);
    printf("test_wide_vector PASSED\n");
}

int main() {
    test_parallel_unit_vectors();
    test_orthogonal_vectors();
    test_tombstone();
    test_multiple_queries();
    test_wide_vector();
    printf("All dotprod tests PASSED\n");
}
