#include <kernels.h>

__global__ void scale_rows(float *d_vectors, const float *d_inverse_norms, int N, int D) {
    int tid = threadIdx.x;
    int vec_id = blockIdx.x;
    float inv_norm = d_inverse_norms[vec_id];
    for (int t = tid; t < D; t += blockDim.x) {
        d_vectors[vec_id * D + t] *= inv_norm;
    }
}

void launch_scale_rows(float *d_vectors, const float *d_inverse_norms, int N, int D) {
    if (N <= 0) return;
    if (D <= 0) return;
    constexpr int t_per_block = 256;
    scale_rows<<<N, t_per_block>>>(d_vectors, d_inverse_norms, N, D);
}
