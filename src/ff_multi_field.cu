// src/ff_multi_field.cu
// GPU throughput kernels for Goldilocks and BabyBear field arithmetic.
// Used by microbenchmarks and correctness tests.

#include "ff_goldilocks.cuh"
#include "ff_babybear.cuh"

// ─── Goldilocks Throughput Kernels ──────────────────────────────────────────

__global__ void gl_add_kernel(const GoldilocksElement* __restrict__ a,
                              const GoldilocksElement* __restrict__ b,
                              GoldilocksElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = gl_add(a[idx], b[idx]);
}

__global__ void gl_sub_kernel(const GoldilocksElement* __restrict__ a,
                              const GoldilocksElement* __restrict__ b,
                              GoldilocksElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = gl_sub(a[idx], b[idx]);
}

__global__ void gl_mul_kernel(const GoldilocksElement* __restrict__ a,
                              const GoldilocksElement* __restrict__ b,
                              GoldilocksElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = gl_mul(a[idx], b[idx]);
}

__global__ void gl_sqr_kernel(const GoldilocksElement* __restrict__ a,
                              GoldilocksElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = gl_sqr(a[idx]);
}

// ─── BabyBear Throughput Kernels ────────────────────────────────────────────

__global__ void bb_add_kernel(const BabyBearElement* __restrict__ a,
                              const BabyBearElement* __restrict__ b,
                              BabyBearElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = bb_add(a[idx], b[idx]);
}

__global__ void bb_sub_kernel(const BabyBearElement* __restrict__ a,
                              const BabyBearElement* __restrict__ b,
                              BabyBearElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = bb_sub(a[idx], b[idx]);
}

__global__ void bb_mul_kernel(const BabyBearElement* __restrict__ a,
                              const BabyBearElement* __restrict__ b,
                              BabyBearElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = bb_mul(a[idx], b[idx]);
}

__global__ void bb_sqr_kernel(const BabyBearElement* __restrict__ a,
                              BabyBearElement* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = bb_sqr(a[idx]);
}
