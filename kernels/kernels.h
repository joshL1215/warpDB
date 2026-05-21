#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#endif

// Fused version of the two above
void launch_l2_normalize(
    float *d_vectors,
    int N,
    int D
);

// Compute dot product between normalized vectors, resulting in cosine similarity
// Tombstones tracks which vectors are marked deleted, to ignore in search
// Top k is fused into this computation with WarpSelect style algorithm
void launch_compute_dot_product(
    const float *d_queries,
    const float *d_vectors,
    const uint8_t *d_tombstones,
    float *d_similarities,
    float *d_top_k_values,
    uint64_t *d_top_k_indices,
    int n_queries,
    int N,
    int D,
    int k
);
