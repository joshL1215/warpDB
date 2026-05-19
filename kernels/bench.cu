#include <kernels.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static const int N = 100000;
static const int D = 128;
static const int N_QUERIES = 10;

static double g_peak_bw_gbs = 0.0;  // set in main from device props

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

static void print_stats(
    double cpu_ms, double gpu_ms,
    double bytes_moved,   // bytes the GPU kernel reads+writes
    double flops          // total floating-point operations
) {
    double speedup    = cpu_ms / gpu_ms;
    double bw_gbs     = (bytes_moved / 1e9) / (gpu_ms / 1e3);
    double pct_peak   = bw_gbs / g_peak_bw_gbs * 100.0;
    double gflops     = (flops / 1e9) / (gpu_ms / 1e3);

    printf("  CPU:       %7.2f ms\n", cpu_ms);
    printf("  GPU:       %7.2f ms   |  %.1f GFLOP/s\n", gpu_ms, gflops);
    printf("  Speedup:   %.1fx\n", speedup);
    printf("  Bandwidth: %.1f GB/s  (peak: %.0f GB/s, %.0f%% utilization)\n\n",
           bw_gbs, g_peak_bw_gbs, pct_peak);
}

// ── CPU reference implementations ────────────────────────────────────────────

static void cpu_inv_norms(const float *vecs, float *inv_norms, int n, int d) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) { float v = vecs[i*d+j]; sum += v*v; }
        inv_norms[i] = 1.0f / sqrtf(sum);
    }
}

static void cpu_scale_rows(float *vecs, const float *inv_norms, int n, int d) {
    for (int i = 0; i < n; i++) {
        float s = inv_norms[i];
        for (int j = 0; j < d; j++) vecs[i*d+j] *= s;
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

static void rand_fill(float *buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

static void bench_invnorm() {
    printf("=== invnorm  (N=%d, D=%d) ===\n", N, D);

    float *h_vecs = (float *)malloc(N * D * sizeof(float));
    float *h_inv  = (float *)malloc(N * sizeof(float));
    rand_fill(h_vecs, N * D);

    double t0 = now_ms();
    cpu_inv_norms(h_vecs, h_inv, N, D);
    double cpu_ms = now_ms() - t0;

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch_compute_inv_norms(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    launch_compute_inv_norms(d_vecs, d_inv, N, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = gpu_elapsed_ms(start, stop);

    // reads N*D floats, writes N floats
    double bytes = (double)(N * D + N) * sizeof(float);
    // D multiply-adds per vector = 2*D FLOPs, plus 1 rsqrt ≈ negligible
    double flops = (double)N * D * 2.0;
    print_stats(cpu_ms, gpu_ms, bytes, flops);

    cudaFree(d_vecs); cudaFree(d_inv);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_vecs); free(h_inv);
}

static void bench_scalerows() {
    printf("=== scalerows  (N=%d, D=%d) ===\n", N, D);

    float *h_vecs = (float *)malloc(N * D * sizeof(float));
    float *h_inv  = (float *)malloc(N * sizeof(float));
    rand_fill(h_vecs, N * D);
    cpu_inv_norms(h_vecs, h_inv, N, D);

    float *h_copy = (float *)malloc(N * D * sizeof(float));
    for (int i = 0; i < N * D; i++) h_copy[i] = h_vecs[i];
    double t0 = now_ms();
    cpu_scale_rows(h_copy, h_inv, N, D);
    double cpu_ms = now_ms() - t0;

    float *d_vecs, *d_inv;
    cudaMalloc(&d_vecs, N * D * sizeof(float));
    cudaMalloc(&d_inv,  N * sizeof(float));
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv,  h_inv,  N * sizeof(float),     cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch_scale_rows(d_vecs, d_inv, N, D);
    cudaDeviceSynchronize();
    cudaMemcpy(d_vecs, h_vecs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    launch_scale_rows(d_vecs, d_inv, N, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = gpu_elapsed_ms(start, stop);

    // reads N*D + N floats, writes N*D floats
    double bytes = (double)(2 * N * D + N) * sizeof(float);
    // 1 multiply per element
    double flops = (double)N * D;
    print_stats(cpu_ms, gpu_ms, bytes, flops);

    cudaFree(d_vecs); cudaFree(d_inv);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_vecs); free(h_copy); free(h_inv);
}

static void bench_dotprod() {
    printf("=== dotprod  (n_queries=%d, N=%d, D=%d) ===\n", N_QUERIES, N, D);

    float   *h_queries = (float *)malloc(N_QUERIES * D * sizeof(float));
    float   *h_vecs    = (float *)malloc(N * D * sizeof(float));
    float   *h_sims    = (float *)malloc(N_QUERIES * N * sizeof(float));
    uint8_t *h_tomb    = (uint8_t *)calloc(N, sizeof(uint8_t));
    rand_fill(h_queries, N_QUERIES * D);
    rand_fill(h_vecs, N * D);

    double t0 = now_ms();
    cpu_dot_product(h_queries, h_vecs, h_tomb, h_sims, N_QUERIES, N, D);
    double cpu_ms = now_ms() - t0;

    float *d_queries, *d_vecs, *d_sims; uint8_t *d_tomb;
    cudaMalloc(&d_queries, N_QUERIES * D * sizeof(float));
    cudaMalloc(&d_vecs,    N * D * sizeof(float));
    cudaMalloc(&d_sims,    N_QUERIES * N * sizeof(float));
    cudaMalloc(&d_tomb,    N * sizeof(uint8_t));
    cudaMemcpy(d_queries, h_queries, N_QUERIES * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecs,    h_vecs,    N * D * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(d_tomb,    h_tomb,    N * sizeof(uint8_t),           cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch_compute_dot_product(d_queries, d_vecs, d_tomb, d_sims, N_QUERIES, N, D);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    launch_compute_dot_product(d_queries, d_vecs, d_tomb, d_sims, N_QUERIES, N, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = gpu_elapsed_ms(start, stop);

    // reads n_queries*D + N*D floats + N tombstone bytes, writes n_queries*N floats
    double bytes = (double)(N_QUERIES * D + N * D + N_QUERIES * N) * sizeof(float)
                 + (double)N * sizeof(uint8_t);
    // 2 FLOPs (mul+add) per element per query
    double flops = (double)N_QUERIES * N * D * 2.0;
    print_stats(cpu_ms, gpu_ms, bytes, flops);

    cudaFree(d_queries); cudaFree(d_vecs); cudaFree(d_sims); cudaFree(d_tomb);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_queries); free(h_vecs); free(h_sims); free(h_tomb);
}

int main() {
    srand(42);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int mem_clock_khz, bus_width_bits;
    cudaDeviceGetAttribute(&mem_clock_khz,   cudaDevAttrMemoryClockRate,      0);
    cudaDeviceGetAttribute(&bus_width_bits,  cudaDevAttrGlobalMemoryBusWidth, 0);
    // peak bandwidth: clock (kHz) * bus width (bits) * 2 (DDR) / 8 bits per byte
    g_peak_bw_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;
    printf("GPU: %s  |  Peak bandwidth: %.0f GB/s\n\n", prop.name, g_peak_bw_gbs);

    bench_invnorm();
    bench_scalerows();
    bench_dotprod();
}
