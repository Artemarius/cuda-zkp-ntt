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
