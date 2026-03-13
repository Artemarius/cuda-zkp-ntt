// src/poly_ops.cu
// Polynomial operations: coset NTT, pointwise arithmetic.
// All inputs/outputs in standard (non-Montgomery) form, matching NTT interface.

#include "poly_ops.cuh"
#include "cuda_utils.cuh"
#include "ff_reference.h"  // CPU-side Montgomery conversions

// ─── Device helper: binary exponentiation for uint32_t exponent ─────────────
// base must be in Montgomery form. Returns base^exp in Montgomery form.

__device__ __forceinline__
FpElement fp_pow_u32_mont(const FpElement& base, uint32_t exp) {
    // Montgomery(1)
    FpElement result;
    for (int i = 0; i < 8; ++i) result.limbs[i] = BLS12_381_R_MOD[i];

    if (exp == 0) return result;

    FpElement b = base;
    while (exp > 0) {
        if (exp & 1u) result = ff_mul(result, b);
        b = ff_mul(b, b);
        exp >>= 1;
    }
    return result;
}

// ─── Kernel: coset scale ────────────────────────────────────────────────────
// d_data[i] *= g^i  (standard form in/out)
// g_mont: coset generator in Montgomery form

__global__ void coset_scale_kernel(
    FpElement* __restrict__ d_data,
    FpElement g_mont,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    FpElement g_pow = fp_pow_u32_mont(g_mont, idx);
    FpElement val = ff_to_montgomery(d_data[idx]);
    val = ff_mul(val, g_pow);
    d_data[idx] = ff_from_montgomery(val);
}

// ─── Kernel: coset unscale ──────────────────────────────────────────────────
// d_data[i] *= g^{-i}  (standard form in/out)

__global__ void coset_unscale_kernel(
    FpElement* __restrict__ d_data,
    FpElement g_inv_mont,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    FpElement g_inv_pow = fp_pow_u32_mont(g_inv_mont, idx);
    FpElement val = ff_to_montgomery(d_data[idx]);
    val = ff_mul(val, g_inv_pow);
    d_data[idx] = ff_from_montgomery(val);
}

// ─── Kernel: pointwise multiply ─────────────────────────────────────────────

__global__ void pointwise_mul_kernel(
    FpElement* __restrict__ d_c,
    const FpElement* __restrict__ d_a,
    const FpElement* __restrict__ d_b,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    FpElement a_mont = ff_to_montgomery(d_a[idx]);
    FpElement b_mont = ff_to_montgomery(d_b[idx]);
    FpElement prod = ff_mul(a_mont, b_mont);
    d_c[idx] = ff_from_montgomery(prod);
}

// ─── Kernel: pointwise multiply-subtract ────────────────────────────────────

__global__ void pointwise_mul_sub_kernel(
    FpElement* __restrict__ d_out,
    const FpElement* __restrict__ d_a,
    const FpElement* __restrict__ d_b,
    const FpElement* __restrict__ d_c,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    FpElement a_mont = ff_to_montgomery(d_a[idx]);
    FpElement b_mont = ff_to_montgomery(d_b[idx]);
    FpElement c_mont = ff_to_montgomery(d_c[idx]);
    FpElement prod = ff_mul(a_mont, b_mont);
    FpElement diff = ff_sub(prod, c_mont);
    d_out[idx] = ff_from_montgomery(diff);
}

// ─── Kernel: scale by constant ──────────────────────────────────────────────

__global__ void poly_scale_kernel(
    FpElement* __restrict__ d_data,
    FpElement scalar_mont,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    FpElement val = ff_to_montgomery(d_data[idx]);
    val = ff_mul(val, scalar_mont);
    d_data[idx] = ff_from_montgomery(val);
}

// ─── CPU helper: convert standard-form FpElement to Montgomery FpElement ────

static FpElement cpu_to_montgomery(const FpElement& a) {
    ff_ref::FpRef ref = ff_ref::FpRef::from_u32(a.limbs);
    ff_ref::FpRef mont = ff_ref::to_montgomery(ref);
    FpElement result;
    mont.to_u32(result.limbs);
    return result;
}

static FpElement cpu_fp_inv_mont(const FpElement& a_mont) {
    ff_ref::FpRef ref = ff_ref::FpRef::from_u32(a_mont.limbs);
    ff_ref::FpRef inv = ff_ref::fp_inv(ref);  // input/output Montgomery
    FpElement result;
    inv.to_u32(result.limbs);
    return result;
}

// ─── Host functions ─────────────────────────────────────────────────────────

static const int POLY_BLOCK = 256;

void poly_coset_ntt_forward(FpElement* d_data, size_t n,
                            const FpElement& coset_gen,
                            NTTMode mode, cudaStream_t stream)
{
    if (n == 0) return;

    // Convert coset_gen (standard form) to Montgomery on CPU
    FpElement g_mont = cpu_to_montgomery(coset_gen);

    // Scale by coset powers: d_data[i] *= g^i
    int grid = ((int)n + POLY_BLOCK - 1) / POLY_BLOCK;
    coset_scale_kernel<<<grid, POLY_BLOCK, 0, stream>>>(d_data, g_mont, (uint32_t)n);

    // Forward NTT
    ntt_forward(d_data, n, mode, stream);
}

void poly_coset_ntt_inverse(FpElement* d_data, size_t n,
                            const FpElement& coset_gen,
                            NTTMode mode, cudaStream_t stream)
{
    if (n == 0) return;

    // Inverse NTT first
    ntt_inverse(d_data, n, mode, stream);

    // Compute g^{-1} in Montgomery form on CPU
    FpElement g_mont = cpu_to_montgomery(coset_gen);
    FpElement g_inv_mont = cpu_fp_inv_mont(g_mont);

    // Unscale: d_data[i] *= g^{-i}
    int grid = ((int)n + POLY_BLOCK - 1) / POLY_BLOCK;
    coset_unscale_kernel<<<grid, POLY_BLOCK, 0, stream>>>(d_data, g_inv_mont, (uint32_t)n);
}

void poly_pointwise_mul(FpElement* d_c,
                        const FpElement* d_a, const FpElement* d_b,
                        size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    int grid = ((int)n + POLY_BLOCK - 1) / POLY_BLOCK;
    pointwise_mul_kernel<<<grid, POLY_BLOCK, 0, stream>>>(d_c, d_a, d_b, (uint32_t)n);
}

void poly_pointwise_mul_sub(FpElement* d_out,
                            const FpElement* d_a, const FpElement* d_b,
                            const FpElement* d_c,
                            size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    int grid = ((int)n + POLY_BLOCK - 1) / POLY_BLOCK;
    pointwise_mul_sub_kernel<<<grid, POLY_BLOCK, 0, stream>>>(d_out, d_a, d_b, d_c, (uint32_t)n);
}

void poly_scale(FpElement* d_data, const FpElement& scalar,
                size_t n, cudaStream_t stream)
{
    if (n == 0) return;

    FpElement scalar_mont = cpu_to_montgomery(scalar);

    int grid = ((int)n + POLY_BLOCK - 1) / POLY_BLOCK;
    poly_scale_kernel<<<grid, POLY_BLOCK, 0, stream>>>(d_data, scalar_mont, (uint32_t)n);
}
