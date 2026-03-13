// src/ec_kernels.cu
// GPU kernels for testing elliptic curve point arithmetic.

#include "ec_g1.cuh"
#include "ec_g2.cuh"

// ─── G1 Test Kernels ────────────────────────────────────────────────────────

__global__ void g1_double_kernel(const G1Jacobian* __restrict__ in,
                                  G1Jacobian* __restrict__ out,
                                  uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_double(in[idx]);
}

__global__ void g1_add_kernel(const G1Jacobian* __restrict__ a,
                               const G1Jacobian* __restrict__ b,
                               G1Jacobian* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_add(a[idx], b[idx]);
}

__global__ void g1_add_mixed_kernel(const G1Jacobian* __restrict__ a,
                                     const G1Affine* __restrict__ b,
                                     G1Jacobian* __restrict__ out,
                                     uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_add_mixed(a[idx], b[idx]);
}

__global__ void g1_scalar_mul_kernel(const G1Jacobian* __restrict__ bases,
                                      const uint32_t* __restrict__ scalars,
                                      G1Jacobian* __restrict__ out,
                                      uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_scalar_mul(bases[idx], &scalars[idx * 8]);
}

// ─── G2 Test Kernels ────────────────────────────────────────────────────────

__global__ void g2_double_kernel(const G2Jacobian* __restrict__ in,
                                  G2Jacobian* __restrict__ out,
                                  uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g2_double(in[idx]);
}

__global__ void g2_add_kernel(const G2Jacobian* __restrict__ a,
                               const G2Jacobian* __restrict__ b,
                               G2Jacobian* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g2_add(a[idx], b[idx]);
}

__global__ void g2_add_mixed_kernel(const G2Jacobian* __restrict__ a,
                                     const G2Affine* __restrict__ b,
                                     G2Jacobian* __restrict__ out,
                                     uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g2_add_mixed(a[idx], b[idx]);
}

__global__ void g2_scalar_mul_kernel(const G2Jacobian* __restrict__ bases,
                                      const uint32_t* __restrict__ scalars,
                                      G2Jacobian* __restrict__ out,
                                      uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g2_scalar_mul(bases[idx], &scalars[idx * 8]);
}

// ─── On-curve check kernels (for batch verification) ───────────────────────

__global__ void g1_is_on_curve_kernel(const G1Affine* __restrict__ pts,
                                       bool* __restrict__ results,
                                       uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) results[idx] = g1_is_on_curve(pts[idx]);
}

__global__ void g2_is_on_curve_kernel(const G2Affine* __restrict__ pts,
                                       bool* __restrict__ results,
                                       uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) results[idx] = g2_is_on_curve(pts[idx]);
}

// ─── Affine conversion kernels ─────────────────────────────────────────────

__global__ void g1_to_affine_kernel(const G1Jacobian* __restrict__ in,
                                     G1Affine* __restrict__ out,
                                     uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_to_affine(in[idx]);
}

__global__ void g2_to_affine_kernel(const G2Jacobian* __restrict__ in,
                                     G2Affine* __restrict__ out,
                                     uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g2_to_affine(in[idx]);
}
