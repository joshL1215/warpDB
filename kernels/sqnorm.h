#pragma once

// normalize N vectors of length D by Euclidean (L2) Norm Squared and store in d_norms
void launch_compute_sq_norms(const float *d_vectors, float *d_norms, int N, int D);
