#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s line %d\n", __FILE__, __LINE__); exit(1); } } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA FAIL: %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

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
);

static void test_top_k_and_tombstones() {
    const int n_queries = 1;
    const int N = 5;
    const int D = 3;
    const int k = 3;

    const float h_queries[n_queries * D] = {
        1.0f, 0.0f, 0.0f,
    };
    const float h_vectors[N * D] = {
        1.0f, 0.0f, 0.0f,
        0.6f, 0.8f, 0.0f,
        0.0f, 1.0f, 0.0f,
        -1.0f, 0.0f, 0.0f,
        0.8f, 0.6f, 0.0f,
    };
    const uint8_t h_tombstones[N] = {0, 0, 0, 0, 1};

    float *d_queries;
    float *d_vectors;
    uint8_t *d_tombstones;
    uint64_t *d_out_indices;
    float *d_out_values;

    CUDA_CHECK(cudaMalloc(&d_queries, sizeof(h_queries)));
    CUDA_CHECK(cudaMalloc(&d_vectors, sizeof(h_vectors)));
    CUDA_CHECK(cudaMalloc(&d_tombstones, sizeof(h_tombstones)));
    CUDA_CHECK(cudaMalloc(&d_out_indices, n_queries * k * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_values, n_queries * k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, sizeof(h_queries), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, sizeof(h_vectors), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tombstones, h_tombstones, sizeof(h_tombstones), cudaMemcpyHostToDevice));

    launch_cosine_search(d_queries, d_vectors, d_tombstones, d_out_indices, d_out_values,
                         n_queries, N, D, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t h_out_indices[n_queries * k];
    float h_out_values[n_queries * k];
    CUDA_CHECK(cudaMemcpy(h_out_indices, d_out_indices, sizeof(h_out_indices), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_values, d_out_values, sizeof(h_out_values), cudaMemcpyDeviceToHost));

    CHECK(h_out_indices[0] == 0);
    CHECK(fabsf(h_out_values[0] - 1.0f) < 1e-5f);
    CHECK(h_out_indices[1] == 1);
    CHECK(fabsf(h_out_values[1] - 0.6f) < 1e-5f);
    CHECK(h_out_indices[2] == 2);
    CHECK(fabsf(h_out_values[2] - 0.0f) < 1e-5f);

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_tombstones));
    CUDA_CHECK(cudaFree(d_out_indices));
    CUDA_CHECK(cudaFree(d_out_values));

    printf("test_top_k_and_tombstones PASSED\n");
}

static void test_multiple_queries() {
    const int n_queries = 2;
    const int N = 4;
    const int D = 3;
    const int k = 2;

    const float h_queries[n_queries * D] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const float h_vectors[N * D] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.70710677f, 0.70710677f, 0.0f,
    };
    const uint8_t h_tombstones[N] = {0, 0, 0, 0};

    float *d_queries;
    float *d_vectors;
    uint8_t *d_tombstones;
    uint64_t *d_out_indices;
    float *d_out_values;

    CUDA_CHECK(cudaMalloc(&d_queries, sizeof(h_queries)));
    CUDA_CHECK(cudaMalloc(&d_vectors, sizeof(h_vectors)));
    CUDA_CHECK(cudaMalloc(&d_tombstones, sizeof(h_tombstones)));
    CUDA_CHECK(cudaMalloc(&d_out_indices, n_queries * k * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_values, n_queries * k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, sizeof(h_queries), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, sizeof(h_vectors), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tombstones, h_tombstones, sizeof(h_tombstones), cudaMemcpyHostToDevice));

    launch_cosine_search(d_queries, d_vectors, d_tombstones, d_out_indices, d_out_values,
                         n_queries, N, D, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t h_out_indices[n_queries * k];
    float h_out_values[n_queries * k];
    CUDA_CHECK(cudaMemcpy(h_out_indices, d_out_indices, sizeof(h_out_indices), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_values, d_out_values, sizeof(h_out_values), cudaMemcpyDeviceToHost));

    CHECK(h_out_indices[0] == 0);
    CHECK(fabsf(h_out_values[0] - 1.0f) < 1e-5f);
    CHECK(h_out_indices[1] == 3);
    CHECK(fabsf(h_out_values[1] - 0.70710677f) < 1e-5f);

    CHECK(h_out_indices[2] == 1);
    CHECK(fabsf(h_out_values[2] - 1.0f) < 1e-5f);
    CHECK(h_out_indices[3] == 3);
    CHECK(fabsf(h_out_values[3] - 0.70710677f) < 1e-5f);

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_tombstones));
    CUDA_CHECK(cudaFree(d_out_indices));
    CUDA_CHECK(cudaFree(d_out_values));

    printf("test_multiple_queries PASSED\n");
}

int main() {
    test_top_k_and_tombstones();
    test_multiple_queries();
    printf("All cosine_search tests PASSED\n");
}
