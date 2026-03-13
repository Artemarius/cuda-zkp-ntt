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

// Optimal window size for Pippenger's method with signed-digit recoding.
//
// GPU cost model: total ≈ W * [sort(n,c) + n/B + O(c)] + W*c
//   W = ceil(255/c) windows, B = 2^(c-1) signed-digit buckets
//   Dominant cost is W * sort(n) — larger c → fewer windows → fewer sorts
//   Accumulation cost n/B decreases with larger c (more parallel buckets)
//   Parallel reduction cost O(log B) ≈ O(c) is small
//
// Heuristic: c = floor(log2(n)/2) + 1, clamped to [4, 11].
// Upper cap = 11 ensures num_buckets-1 = 2^(c-1) ≤ 1024, which is the
// maximum thread count for the parallel bucket reduction kernel.
// Lower cap = 4 (9 buckets) prevents degenerate windows.
inline int msm_optimal_window(size_t n) {
    if (n <= 1) return 1;
    int log_n = 0;
    size_t tmp = n;
    while (tmp > 1) { tmp >>= 1; ++log_n; }
    int c = log_n / 2 + 1;
    if (c < 4) c = 4;
    if (c > 11) c = 11;  // parallel reduction limit: 2^(c-1) <= 1024
    return c;
}
