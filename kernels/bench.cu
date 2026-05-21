#include <kernels.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static const int N = 100000;
static const int D = 128;
static const int N_QUERIES = 10;
static const int K = 10;

static double g_peak_bw_gbs = 0.0;

static double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static float gpu_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static void print_stats(double cpu_ms, double gpu_ms, double bytes, double flops) {
    double bw_gbs   = (bytes / 1e9) / (gpu_ms / 1e3);
    double pct_peak = bw_gbs / g_peak_bw_gbs * 100.0;
    double gflops   = (flops / 1e9) / (gpu_ms / 1e3);
    printf("  CPU:       %7.2f ms\n", cpu_ms);
    printf("  GPU:       %7.2f ms   |  %.1f GFLOP/s\n", gpu_ms, gflops);
    printf("  Speedup:   %.1fx\n", cpu_ms / gpu_ms);
    printf("  Bandwidth: %.1f GB/s  (peak: %.0f GB/s, %.0f%% utilization)\n\n",
           bw_gbs, g_peak_bw_gbs, pct_peak);
}

static void rand_fill(float *buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// ── CPU references ────────────────────────────────────────────────────────────

static void cpu_l2_normalize(float *vecs, int n, int d) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += vecs[i*d+j] * vecs[i*d+j];
        float inv = 1.0f / sqrtf(sum);
        for (int j = 0; j < d; j++) vecs[i*d+j] *= inv;
    }
}

static void cpu_dot_product(
    const float *queries, const float *vecs, const uint8_t *tombstones,
    float *sims, int n_queries, int n, int d
) {
    for (int q = 0; q < n_queries; q++) {
        for (int i = 0; i < n; i++) {
            if (tombstones[i]) { sims[q*n+i] = -INFINITY; continue; }
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += queries[q*d+j] * vecs[i*d+j];
            sims[q*n+i] = dot;
        }
    }
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

static void bench_l2norm() {
    printf("=== l2norm  (N=%d, D=%d) ===\n", N, D);

    float *h_vecs = (float *)malloc(N * D * sizeof(float));
    rand_fill(h_vecs, N * D);

    float *h_copy = (float *)malloc(N * D * sizeof(float));
    for (int i = 0; i < N * D; i++) h_copy[i] = h_vecs[i];
    double t0 = now_ms();
    cpu_l2_normalize(h_copy, N, D);
    double cpu_ms = now_ms() - t0;

    float *d_vecs;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch_l2_normalize(d_vecs, N, D);
    cudaDeviceSynchronize();
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    launch_l2_normalize(d_vecs, N, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = gpu_elapsed_ms(start, stop);

    // reads N*D floats, writes N*D floats
    double bytes = (double)(2 * N * D) * sizeof(float);
    // D multiply-adds for norm + D multiplies for scale = 3*D FLOPs per vector
    double flops = (double)N * D * 3.0;
    print_stats(cpu_ms, gpu_ms, bytes, flops);

    cudaFree(d_vecs);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_vecs); free(h_copy);
}

static void bench_dotprod() {
    printf("=== dotprod  (n_queries=%d, N=%d, D=%d, k=%d) ===\n", N_QUERIES, N, D, K);

    float   *h_queries = (float *)malloc(N_QUERIES * D * sizeof(float));
    float   *h_vecs    = (float *)malloc(N * D * sizeof(float));
    float   *h_sims    = (float *)malloc(N_QUERIES * N * sizeof(float));
    uint8_t *h_tomb    = (uint8_t *)calloc(N, sizeof(uint8_t));
    rand_fill(h_queries, N_QUERIES * D);
    rand_fill(h_vecs, N * D);

    double t0 = now_ms();
    cpu_dot_product(h_queries, h_vecs, h_tomb, h_sims, N_QUERIES, N, D);
    double cpu_ms = now_ms() - t0;

    float    *d_queries, *d_vecs, *d_sims, *d_topv;
    uint8_t  *d_tomb;
    uint64_t *d_topi;
    cudaMalloc(&d_queries, N_QUERIES * D * sizeof(float));
    cudaMalloc(&d_vecs,    N * D * sizeof(float));
    cudaMalloc(&d_sims,    N_QUERIES * N * sizeof(float));
    cudaMalloc(&d_topv,    N_QUERIES * K * sizeof(float));
    cudaMalloc(&d_topi,    N_QUERIES * K * sizeof(uint64_t));
    cudaMalloc(&d_tomb,    N * sizeof(uint8_t));
    cudaMemcpy(d_queries, h_queries, N_QUERIES * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecs,    h_vecs,    N * D * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(d_tomb,    h_tomb,    N * sizeof(uint8_t),           cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch_compute_dot_product(d_queries, d_vecs, d_tomb, d_sims, d_topv, d_topi, N_QUERIES, N, D, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    launch_compute_dot_product(d_queries, d_vecs, d_tomb, d_sims, d_topv, d_topi, N_QUERIES, N, D, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = gpu_elapsed_ms(start, stop);

    // reads n_queries*D + N*D floats + N tombstone bytes, writes n_queries*N floats
    double bytes = (double)(N_QUERIES * D + N * D + N_QUERIES * N) * sizeof(float)
                 + (double)N * sizeof(uint8_t);
    // 2 FLOPs (mul+add) per element per query
    double flops = (double)N_QUERIES * N * D * 2.0;
    print_stats(cpu_ms, gpu_ms, bytes, flops);

    cudaFree(d_queries); cudaFree(d_vecs); cudaFree(d_sims);
    cudaFree(d_topv); cudaFree(d_topi); cudaFree(d_tomb);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_queries); free(h_vecs); free(h_sims); free(h_tomb);
}

int main() {
    srand(42);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int mem_clock_khz, bus_width_bits;
    cudaDeviceGetAttribute(&mem_clock_khz,  cudaDevAttrMemoryClockRate,      0);
    cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    g_peak_bw_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;
    printf("GPU: %s  |  Peak bandwidth: %.0f GB/s\n\n", prop.name, g_peak_bw_gbs);

    bench_l2norm();
    bench_dotprod();
}
