#include <kernels.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s line %d\n", __FILE__, __LINE__); exit(1); } } while (0)

// [3, 4, 0] has norm 5, so inv_norm = 0.2
// [1, 0, 0] has norm 1, so inv_norm = 1.0
static void test_basic() {
    int N = 2, D = 3;
    float h_vecs[] = {3.0f, 4.0f, 0.0f,
                      1.0f, 0.0f, 0.0f};

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_compute_inv_norms(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();

    float h_inv[2];
    cudaMemcpy(h_inv, d_inv, N * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(fabsf(h_inv[0] - 0.2f) < 1e-5f);
    CHECK(fabsf(h_inv[1] - 1.0f) < 1e-5f);

    cudaFree(d_vecs);
    cudaFree(d_inv);
    printf("test_basic PASSED\n");
}

// All equal components: [1,1,1,1] has norm 2, inv_norm = 0.5
static void test_equal_components() {
    int N = 1, D = 4;
    float h_vecs[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_compute_inv_norms(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();

    float h_inv[1];
    cudaMemcpy(h_inv, d_inv, N * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(fabsf(h_inv[0] - 0.5f) < 1e-5f);

    cudaFree(d_vecs);
    cudaFree(d_inv);
    printf("test_equal_components PASSED\n");
}

// D larger than one thread block (256) to exercise the reduction loop
static void test_wide_vector() {
    int N = 1, D = 512;
    float h_vecs[512];
    // fill with 1/sqrt(512) so norm = 1 and inv_norm = 1
    float val = 1.0f / sqrtf((float)D);
    for (int i = 0; i < D; i++) h_vecs[i] = val;

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_compute_inv_norms(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();

    float h_inv[1];
    cudaMemcpy(h_inv, d_inv, N * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(fabsf(h_inv[0] - 1.0f) < 1e-4f);

    cudaFree(d_vecs);
    cudaFree(d_inv);
    printf("test_wide_vector PASSED\n");
}

int main() {
    test_basic();
    test_equal_components();
    test_wide_vector();
    printf("All invnorm tests PASSED\n");
}
