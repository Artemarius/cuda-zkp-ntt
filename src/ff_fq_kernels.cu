// src/ff_fq_kernels.cu
// GPU throughput kernels for Fq and Fq2 field arithmetic.
// Used by correctness tests and microbenchmarks.

#include "ff_fq.cuh"
#include "ff_fq2.cuh"

// ─── Fq Throughput Kernels ──────────────────────────────────────────────────

__global__ void fq_add_kernel(const FqElement* __restrict__ a,
                              const FqElement* __restrict__ b,
                              FqElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq_add(a[idx], b[idx]);
}

__global__ void fq_sub_kernel(const FqElement* __restrict__ a,
                              const FqElement* __restrict__ b,
                              FqElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq_sub(a[idx], b[idx]);
}

__global__ void fq_mul_kernel(const FqElement* __restrict__ a,
                              const FqElement* __restrict__ b,
                              FqElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq_mul(a[idx], b[idx]);
}

__global__ void fq_sqr_kernel(const FqElement* __restrict__ a,
                              FqElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq_sqr(a[idx]);
}

// ─── Fq2 Throughput Kernels ─────────────────────────────────────────────────

__global__ void fq2_add_kernel(const Fq2Element* __restrict__ a,
                               const Fq2Element* __restrict__ b,
                               Fq2Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq2_add(a[idx], b[idx]);
}

__global__ void fq2_sub_kernel(const Fq2Element* __restrict__ a,
                               const Fq2Element* __restrict__ b,
                               Fq2Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq2_sub(a[idx], b[idx]);
}

__global__ void fq2_mul_kernel(const Fq2Element* __restrict__ a,
                               const Fq2Element* __restrict__ b,
                               Fq2Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq2_mul(a[idx], b[idx]);
}

__global__ void fq2_sqr_kernel(const Fq2Element* __restrict__ a,
                               Fq2Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq2_sqr(a[idx]);
}
