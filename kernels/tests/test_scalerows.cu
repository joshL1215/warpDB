#include <kernels.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s line %d\n", __FILE__, __LINE__); exit(1); } } while (0)

// Scale [3, 4, 0] by inv_norm 0.2 → [0.6, 0.8, 0.0]
static void test_basic() {
    int N = 2, D = 3;
    float h_vecs[] = {3.0f, 4.0f, 0.0f,
                      2.0f, 0.0f, 0.0f};
    float h_inv[]  = {0.2f, 0.5f};

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv,  h_inv,  N * sizeof(float),     cudaMemcpyHostToDevice);

    launch_scale_rows(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();

    float h_out[6];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(fabsf(h_out[0] - 0.6f) < 1e-5f);
    CHECK(fabsf(h_out[1] - 0.8f) < 1e-5f);
    CHECK(fabsf(h_out[2] - 0.0f) < 1e-5f);
    CHECK(fabsf(h_out[3] - 1.0f) < 1e-5f);
    CHECK(fabsf(h_out[4] - 0.0f) < 1e-5f);
    CHECK(fabsf(h_out[5] - 0.0f) < 1e-5f);

    cudaFree(d_vecs);
    cudaFree(d_inv);
    printf("test_basic PASSED\n");
}

// Scaling by inv_norm 1.0 should leave the vector unchanged
static void test_unit_scale() {
    int N = 1, D = 4;
    float h_vecs[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float h_inv[]  = {1.0f};

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv,  h_inv,  N * sizeof(float),     cudaMemcpyHostToDevice);

    launch_scale_rows(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();

    float h_out[4];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < D; i++)
        CHECK(fabsf(h_out[i] - 0.5f) < 1e-5f);

    cudaFree(d_vecs);
    cudaFree(d_inv);
    printf("test_unit_scale PASSED\n");
}

// Combine invnorm + scalerows: result should be a unit vector
static void test_normalized_result() {
    int N = 1, D = 3;
    float h_vecs[] = {1.0f, 2.0f, 2.0f};  // norm = 3

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_compute_inv_norms(d_vecs, d_inv, N, D);
    launch_scale_rows(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();

    float h_out[3];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(h_out[0]*h_out[0] + h_out[1]*h_out[1] + h_out[2]*h_out[2]);
    CHECK(fabsf(norm - 1.0f) < 1e-5f);

    cudaFree(d_vecs);
    cudaFree(d_inv);
    printf("test_normalized_result PASSED\n");
}

int main() {
    test_basic();
    test_unit_scale();
    test_normalized_result();
    printf("All scalerows tests PASSED\n");
}
