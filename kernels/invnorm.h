#pragma once

// compute inverse L2 norms
void launch_compute_inv_norms(const float *d_vectors, float *d_norms, int N, int D);
