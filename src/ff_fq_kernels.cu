// src/ff_fq_kernels.cu
// GPU throughput kernels for Fq, Fq2, Fq6, and Fq12 field arithmetic.
// Used by correctness tests and microbenchmarks.

#include "ff_fq.cuh"
#include "ff_fq2.cuh"
#include "ff_fq6.cuh"
#include "ff_fq12.cuh"

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

// ─── Fq6 Throughput Kernels ─────────────────────────────────────────────────

__global__ void fq6_add_kernel(const Fq6Element* __restrict__ a,
                               const Fq6Element* __restrict__ b,
                               Fq6Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq6_add(a[idx], b[idx]);
}

__global__ void fq6_sub_kernel(const Fq6Element* __restrict__ a,
                               const Fq6Element* __restrict__ b,
                               Fq6Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq6_sub(a[idx], b[idx]);
}

__global__ void fq6_mul_kernel(const Fq6Element* __restrict__ a,
                               const Fq6Element* __restrict__ b,
                               Fq6Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq6_mul(a[idx], b[idx]);
}

__global__ void fq6_sqr_kernel(const Fq6Element* __restrict__ a,
                               Fq6Element* __restrict__ out,
                               uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq6_sqr(a[idx]);
}

// ─── Fq12 Throughput Kernels ────────────────────────────────────────────────

__global__ void fq12_add_kernel(const Fq12Element* __restrict__ a,
                                const Fq12Element* __restrict__ b,
                                Fq12Element* __restrict__ out,
                                uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq12_add(a[idx], b[idx]);
}

__global__ void fq12_sub_kernel(const Fq12Element* __restrict__ a,
                                const Fq12Element* __restrict__ b,
                                Fq12Element* __restrict__ out,
                                uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq12_sub(a[idx], b[idx]);
}

__global__ void fq12_mul_kernel(const Fq12Element* __restrict__ a,
                                const Fq12Element* __restrict__ b,
                                Fq12Element* __restrict__ out,
                                uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq12_mul(a[idx], b[idx]);
}

__global__ void fq12_sqr_kernel(const Fq12Element* __restrict__ a,
                                Fq12Element* __restrict__ out,
                                uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fq12_sqr(a[idx]);
}
