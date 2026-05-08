#pragma once

// normalize N vectors of length D by Euclidean (L2) norm and store in d_norms
void launch_compute_norms(const float *d_vectors, float *d_norms, int N, int D);
