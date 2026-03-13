// include/msm.cuh
// GPU-accelerated Multi-Scalar Multiplication using Pippenger's bucket method.
// Computes: result = sum_{i=0}^{n-1} scalar[i] * base[i]
// Bases: G1 affine points (in Montgomery form)
// Scalars: 8 x uint32_t little-endian (Fr elements in standard form)

#pragma once
#include "ec_g1.cuh"

// Compute MSM over G1.
// d_bases: device pointer to n G1Affine points
// d_scalars: device pointer to n * 8 uint32_t scalars (contiguous)
// n: number of points
// result: host pointer to output G1Affine point
// stream: CUDA stream (default 0)
void msm_g1(G1Affine* result,
            const G1Affine* d_bases,
            const uint32_t* d_scalars,
            size_t n,
            cudaStream_t stream = 0);

// Optimal window size for Pippenger's method.
// c ~ sqrt(log2(n)) for n points with 255-bit scalars.
inline int msm_optimal_window(size_t n) {
    if (n <= 1) return 1;
    int log_n = 0;
    size_t tmp = n;
    while (tmp > 1) { tmp >>= 1; ++log_n; }
    // c = max(1, sqrt(log_n))
    int c = 1;
    while ((c + 1) * (c + 1) <= log_n) ++c;
    // Clamp to reasonable range
    if (c < 4) c = 4;
    if (c > 16) c = 16;
    return c;
}
