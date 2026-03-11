// src/ff_mul.cu
// Montgomery multiplication GPU kernels for BLS12-381 scalar field
// Phase 2: real kernel implementations wrapping ff_arithmetic.cuh functions

#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// Isolated FF_mul throughput measurement kernel
__global__ void ff_mul_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_mul(a[tid], b[tid]);
}

// Isolated FF_add throughput measurement kernel
__global__ void ff_add_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_add(a[tid], b[tid]);
}

// Isolated FF_sub throughput measurement kernel
__global__ void ff_sub_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_sub(a[tid], b[tid]);
}

// Isolated FF_sqr throughput measurement kernel
// Takes a single input array; each thread squares its element.
__global__ void ff_sqr_throughput_kernel(
    const FpElement* __restrict__ a,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_sqr(a[tid]);
}

// ─── v2 Kernels: branchless reduction (ff_mul_ptx, ff_add_v2, ff_sub_v2) ───

__global__ void ff_mul_v2_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_mul_ptx(a[tid], b[tid]);
}

__global__ void ff_add_v2_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_add_v2(a[tid], b[tid]);
}

__global__ void ff_sub_v2_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_sub_v2(a[tid], b[tid]);
}

__global__ void ff_sqr_v2_kernel(
    const FpElement* __restrict__ a,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    FpElement val = a[tid];
    out[tid] = ff_mul_ptx(val, val);
}

// ─── Barrett Kernels ─────────────────────────────────────────────────────────
// Standard-form in/out — no Montgomery conversion needed.

#include "ff_barrett.cuh"

__global__ void ff_mul_barrett_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_mul_barrett(a[tid], b[tid]);
}

__global__ void ff_sqr_barrett_kernel(
    const FpElement* __restrict__ a,
    FpElement* __restrict__ out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = ff_sqr_barrett(a[tid]);
}

// ─── SoA (Structure-of-Arrays) Kernel Variants ─────────────────────────────
// Memory layout: limbs[limb_idx * n + element_idx]
// This gives perfectly coalesced 4-byte accesses across a warp.

__device__ __forceinline__
FpElement load_fp_soa(const uint32_t* __restrict__ limbs, uint32_t idx, uint32_t n) {
    FpElement e;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        e.limbs[j] = limbs[j * n + idx];
    }
    return e;
}

__device__ __forceinline__
void store_fp_soa(uint32_t* __restrict__ limbs, uint32_t idx, uint32_t n,
                  const FpElement& e) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        limbs[j * n + idx] = e.limbs[j];
    }
}

__global__ void ff_mul_soa_kernel(
    const uint32_t* __restrict__ a_limbs,
    const uint32_t* __restrict__ b_limbs,
    uint32_t* __restrict__ out_limbs,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    FpElement a = load_fp_soa(a_limbs, tid, n);
    FpElement b = load_fp_soa(b_limbs, tid, n);
    store_fp_soa(out_limbs, tid, n, ff_mul(a, b));
}

__global__ void ff_add_soa_kernel(
    const uint32_t* __restrict__ a_limbs,
    const uint32_t* __restrict__ b_limbs,
    uint32_t* __restrict__ out_limbs,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    FpElement a = load_fp_soa(a_limbs, tid, n);
    FpElement b = load_fp_soa(b_limbs, tid, n);
    store_fp_soa(out_limbs, tid, n, ff_add(a, b));
}

__global__ void ff_sub_soa_kernel(
    const uint32_t* __restrict__ a_limbs,
    const uint32_t* __restrict__ b_limbs,
    uint32_t* __restrict__ out_limbs,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    FpElement a = load_fp_soa(a_limbs, tid, n);
    FpElement b = load_fp_soa(b_limbs, tid, n);
    store_fp_soa(out_limbs, tid, n, ff_sub(a, b));
}

__global__ void ff_sqr_soa_kernel(
    const uint32_t* __restrict__ a_limbs,
    uint32_t* __restrict__ out_limbs,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    FpElement a = load_fp_soa(a_limbs, tid, n);
    store_fp_soa(out_limbs, tid, n, ff_sqr(a));
}
