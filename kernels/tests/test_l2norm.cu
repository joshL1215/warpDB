#include <kernels.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL: %s line %d\n", __FILE__, __LINE__); exit(1); } } while (0)

static float vec_norm(const float *v, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) sum += v[i] * v[i];
    return sqrtf(sum);
}

// [3, 4, 0] has norm 5; after normalization each element is divided by 5
static void test_basic() {
    int N = 1, D = 3;
    float h_vecs[] = {3.0f, 4.0f, 0.0f};

    float *d_vecs;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_l2_normalize(d_vecs, N, D);
    cudaDeviceSynchronize();

    float h_out[3];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(fabsf(h_out[0] - 0.6f) < 1e-5f);
    CHECK(fabsf(h_out[1] - 0.8f) < 1e-5f);
    CHECK(fabsf(h_out[2] - 0.0f) < 1e-5f);

    cudaFree(d_vecs);
    printf("test_basic PASSED\n");
}

// Result of normalizing any non-zero vector must have unit norm
static void test_unit_norm_after_normalize() {
    int N = 3, D = 4;
    float h_vecs[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };

    float *d_vecs;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_l2_normalize(d_vecs, N, D);
    cudaDeviceSynchronize();

    float h_out[12];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        CHECK(fabsf(vec_norm(h_out + i * D, D) - 1.0f) < 1e-5f);

    cudaFree(d_vecs);
    printf("test_unit_norm_after_normalize PASSED\n");
}

// Already-unit vectors must be unchanged
static void test_idempotent() {
    int N = 1, D = 3;
    float inv = 1.0f / sqrtf(3.0f);
    float h_vecs[] = {inv, inv, inv};

    float *d_vecs;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_l2_normalize(d_vecs, N, D);
    cudaDeviceSynchronize();

    float h_out[3];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < D; i++)
        CHECK(fabsf(h_out[i] - inv) < 1e-5f);

    cudaFree(d_vecs);
    printf("test_idempotent PASSED\n");
}

// D > 256 to exercise the multi-pass reduction
static void test_wide_vector() {
    int N = 1, D = 512;
    float h_vecs[512];
    for (int i = 0; i < D; i++) h_vecs[i] = 1.0f;

    float *d_vecs;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    launch_l2_normalize(d_vecs, N, D);
    cudaDeviceSynchronize();

    float h_out[512];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK(fabsf(vec_norm(h_out, D) - 1.0f) < 1e-4f);

    cudaFree(d_vecs);
    printf("test_wide_vector PASSED\n");
}

// Result must match a CPU reference: divide each element by its row's L2 norm
static void test_matches_cpu_reference() {
    int N = 4, D = 8;
    float h_vecs[32], h_ref[32];
    for (int i = 0; i < N * D; i++) h_vecs[i] = h_ref[i] = (float)(i + 1);

    // CPU reference
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < D; j++) sum += h_ref[i*D+j] * h_ref[i*D+j];
        float inv = 1.0f / sqrtf(sum);
        for (int j = 0; j < D; j++) h_ref[i*D+j] *= inv;
    }

    // GPU
    float *d_vecs;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    launch_l2_normalize(d_vecs, N, D);
    cudaDeviceSynchronize();
    float h_out[32];
    cudaMemcpy(h_out, d_vecs, N * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vecs);

    for (int i = 0; i < N * D; i++)
        CHECK(fabsf(h_out[i] - h_ref[i]) < 1e-5f);

    printf("test_matches_cpu_reference PASSED\n");
}

int main() {
    test_basic();
    test_unit_norm_after_normalize();
    test_idempotent();
    test_wide_vector();
    test_matches_cpu_reference();
    printf("All l2norm tests PASSED\n");
}
