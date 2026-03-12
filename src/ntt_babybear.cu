// src/ntt_babybear.cu
// NTT implementation for the BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001
// Single uint32_t elements (4 bytes). Standard form throughout.
//
// Architecture:
//   - Fused inner kernel: K=8/9/10/11 (warp shuffle stages 0-4 + shmem 5..K-1)
//   - Cooperative outer stages: radix-8 → radix-4 → radix-2 fallback
//   - K=11: 2048 elements × 4B = 8 KB shmem, 1024 threads
//   - For n=2^22 with K=11: 11 outer stages
//
// BabyBear mul is ~3-5 instructions (32×32→64 product + modulo),
// so register pressure is trivial and radix-8 outer stages work perfectly.

#include "ntt_babybear.cuh"
#include "ff_babybear.cuh"
#include "cuda_utils.cuh"
#include "ff_reference.h"

#include <cooperative_groups.h>
#include <cstdio>
#include <cassert>
#include <vector>

// ─── Twiddle Cache ───────────────────────────────────────────────────────────

static BabyBearElement* s_bb_fwd_twiddles = nullptr;
static BabyBearElement* s_bb_inv_twiddles = nullptr;
static BabyBearElement  s_bb_n_inv;
static size_t           s_bb_cached_n = 0;

static void free_bb_twiddles() {
    if (s_bb_fwd_twiddles) { cudaFree(s_bb_fwd_twiddles); s_bb_fwd_twiddles = nullptr; }
    if (s_bb_inv_twiddles) { cudaFree(s_bb_inv_twiddles); s_bb_inv_twiddles = nullptr; }
    s_bb_cached_n = 0;
}

static void ensure_bb_twiddles(size_t n) {
    if (s_bb_cached_n == n) return;
    free_bb_twiddles();

    size_t half = n / 2;

    ff_ref::BbRef omega     = ff_ref::bb_get_root_of_unity(n);
    ff_ref::BbRef omega_inv = ff_ref::bb_inv(omega);

    std::vector<BabyBearElement> h_fwd(half), h_inv(half);

    ff_ref::BbRef wk     = ff_ref::BbRef::one();
    ff_ref::BbRef wk_inv = ff_ref::BbRef::one();
    for (size_t k = 0; k < half; ++k) {
        h_fwd[k] = {wk.val};
        h_inv[k] = {wk_inv.val};
        wk     = ff_ref::bb_mul(wk, omega);
        wk_inv = ff_ref::bb_mul(wk_inv, omega_inv);
    }

    CUDA_CHECK(cudaMalloc(&s_bb_fwd_twiddles, half * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&s_bb_inv_twiddles, half * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(s_bb_fwd_twiddles, h_fwd.data(),
                          half * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_bb_inv_twiddles, h_inv.data(),
                          half * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    ff_ref::BbRef n_inv = ff_ref::bb_inv(ff_ref::BbRef::from_u32(static_cast<uint32_t>(n)));
    s_bb_n_inv = {n_inv.val};

    s_bb_cached_n = n;
}

// ─── Helper ──────────────────────────────────────────────────────────────────

static int bb_log2(size_t n) {
    int r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// ─── Warp-Shuffle Helper ─────────────────────────────────────────────────────
// BabyBear elements are single uint32_t — direct shuffle.

__device__ __forceinline__
BabyBearElement bb_shfl_xor(const BabyBearElement& val, int xor_mask) {
    return {__shfl_xor_sync(0xFFFFFFFF, val.val, xor_mask)};
}

__device__ __forceinline__
BabyBearElement bb_butterfly_shfl(const BabyBearElement& my_elem,
                                   const BabyBearElement& twiddle,
                                   int xor_mask, bool is_top) {
    BabyBearElement partner = bb_shfl_xor(my_elem, xor_mask);
    BabyBearElement A = is_top ? my_elem : partner;
    BabyBearElement B = is_top ? partner : my_elem;
    BabyBearElement v = bb_mul(B, twiddle);
    return is_top ? bb_add(A, v) : bb_sub(A, v);
}

// ─── Fused Inner Kernel ──────────────────────────────────────────────────────

template <int K>
__global__ void bb_ntt_fused_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n
) {
    constexpr int ELEMS   = 1 << K;
    constexpr int THREADS = ELEMS >> 1;
    constexpr int SHFL_STAGES = (K < 5) ? K : 5;

    __shared__ BabyBearElement sdata[ELEMS];

    const uint32_t boff = blockIdx.x * ELEMS;
    const uint32_t t = threadIdx.x;

    BabyBearElement reg_lo = data[boff + t];
    BabyBearElement reg_hi = data[boff + t + THREADS];

    // Stages 0-4: warp shuffle
    #pragma unroll
    for (int s = 0; s < SHFL_STAGES; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n >> (s + 1);
        const uint32_t j      = t & (half - 1);
        const bool is_top     = (t & half) == 0;
        const int xor_mask    = static_cast<int>(half);

        BabyBearElement w = twiddles[j * stride];
        reg_lo = bb_butterfly_shfl(reg_lo, w, xor_mask, is_top);
        reg_hi = bb_butterfly_shfl(reg_hi, w, xor_mask, is_top);
    }

    // Stages 5..K-1: shared memory
    sdata[t]           = reg_lo;
    sdata[t + THREADS] = reg_hi;
    __syncthreads();

    #pragma unroll
    for (int s = SHFL_STAGES; s < K; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n >> (s + 1);

        const uint32_t group   = t >> s;
        const uint32_t j       = t & (half - 1);
        const uint32_t idx_top = (group << (s + 1)) + j;
        const uint32_t idx_bot = idx_top + half;

        BabyBearElement w = twiddles[j * stride];
        BabyBearElement u = sdata[idx_top];
        BabyBearElement v = bb_mul(sdata[idx_bot], w);
        sdata[idx_top] = bb_add(u, v);
        sdata[idx_bot] = bb_sub(u, v);
        __syncthreads();
    }

    data[boff + t]           = sdata[t];
    data[boff + t + THREADS] = sdata[t + THREADS];
}

// Explicit template instantiations
template __global__ void bb_ntt_fused_kernel<8>(
    BabyBearElement* __restrict__, const BabyBearElement* __restrict__, uint32_t);
template __global__ void bb_ntt_fused_kernel<9>(
    BabyBearElement* __restrict__, const BabyBearElement* __restrict__, uint32_t);
template __global__ void bb_ntt_fused_kernel<10>(
    BabyBearElement* __restrict__, const BabyBearElement* __restrict__, uint32_t);
template __global__ void bb_ntt_fused_kernel<11>(
    BabyBearElement* __restrict__, const BabyBearElement* __restrict__, uint32_t);

// ─── Bit-Reverse Kernel ──────────────────────────────────────────────────────

__global__ void bb_ntt_bit_reverse_kernel(BabyBearElement* data, uint32_t n, uint32_t log_n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = __brev(i) >> (32 - log_n);
    if (i < j) {
        BabyBearElement temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

__global__ void bb_ntt_bit_reverse_batch_kernel(
    BabyBearElement* data, uint32_t n, uint32_t log_n, uint32_t total_elements
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;
    uint32_t batch_id = tid / n;
    uint32_t i = tid - batch_id * n;
    uint32_t j = __brev(i) >> (32 - log_n);
    if (i < j) {
        uint32_t base = batch_id * n;
        BabyBearElement temp = data[base + i];
        data[base + i] = data[base + j];
        data[base + j] = temp;
    }
}

// ─── Scale Kernel ────────────────────────────────────────────────────────────

__global__ void bb_ntt_scale_kernel(BabyBearElement* data, BabyBearElement scalar, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = bb_mul(data[i], scalar);
}

// ─── Per-Stage Butterfly Kernel ──────────────────────────────────────────────

__global__ void bb_ntt_butterfly_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, uint32_t half, uint32_t stride
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    uint32_t group = tid / half;
    uint32_t j     = tid % half;
    uint32_t idx_top = group * (2 * half) + j;
    uint32_t idx_bot = idx_top + half;
    BabyBearElement u = data[idx_top];
    BabyBearElement v = bb_mul(data[idx_bot], twiddles[j * stride]);
    data[idx_top] = bb_add(u, v);
    data[idx_bot] = bb_sub(u, v);
}

// ─── Cooperative Outer Kernels ───────────────────────────────────────────────

__global__ void bb_ntt_outer_r2_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, int start_stage, int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_butterflies = n >> 1;

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n / (2u * half);
        for (uint32_t tid = global_tid; tid < num_butterflies; tid += total_threads) {
            uint32_t group   = tid / half;
            uint32_t j       = tid % half;
            uint32_t idx_top = group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;
            BabyBearElement u = data[idx_top];
            BabyBearElement v = bb_mul(data[idx_bot], twiddles[j * stride]);
            data[idx_top] = bb_add(u, v);
            data[idx_bot] = bb_sub(u, v);
        }
        if (s + 1 < end_stage) grid.sync();
    }
}

__global__ void bb_ntt_outer_r4_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, int start_stage, int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r4 = n >> 2;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;

        for (uint32_t tid = global_tid; tid < num_r4; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            BabyBearElement a0 = data[base];
            BabyBearElement a1 = data[base + half];
            BabyBearElement w1 = twiddles[j * stride];
            BabyBearElement t1 = bb_mul(a1, w1);
            BabyBearElement a0p = bb_add(a0, t1);
            BabyBearElement a1p = bb_sub(a0, t1);

            BabyBearElement a2 = data[base + (half << 1)];
            BabyBearElement a3 = data[base + (half << 1) + half];
            BabyBearElement t2 = bb_mul(a3, w1);
            BabyBearElement a2p = bb_add(a2, t2);
            BabyBearElement a3p = bb_sub(a2, t2);

            BabyBearElement w2 = twiddles[j * stride2];
            BabyBearElement t3 = bb_mul(a2p, w2);
            data[base]              = bb_add(a0p, t3);
            data[base + (half << 1)] = bb_sub(a0p, t3);

            BabyBearElement w3 = twiddles[(j + half) * stride2];
            BabyBearElement t4 = bb_mul(a3p, w3);
            data[base + half]              = bb_add(a1p, t4);
            data[base + (half << 1) + half] = bb_sub(a1p, t4);
        }
        if (pass + 1 < num_r4_passes) grid.sync();
    }
}

__global__ void bb_ntt_outer_r8_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, int start_stage, int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r8 = n >> 3;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;
        const uint32_t stride4 = stride >> 2;

        for (uint32_t tid = global_tid; tid < num_r8; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            BabyBearElement a0 = data[base];
            BabyBearElement a1 = data[base + half];
            BabyBearElement a2 = data[base + 2*half];
            BabyBearElement a3 = data[base + 3*half];
            BabyBearElement a4 = data[base + 4*half];
            BabyBearElement a5 = data[base + 5*half];
            BabyBearElement a6 = data[base + 6*half];
            BabyBearElement a7 = data[base + 7*half];

            // Stage s
            BabyBearElement ws = twiddles[j * stride];
            BabyBearElement v0 = bb_mul(a1, ws);
            BabyBearElement b0 = bb_add(a0, v0); BabyBearElement b1 = bb_sub(a0, v0);
            BabyBearElement v1 = bb_mul(a3, ws);
            BabyBearElement b2 = bb_add(a2, v1); BabyBearElement b3 = bb_sub(a2, v1);
            BabyBearElement v2 = bb_mul(a5, ws);
            BabyBearElement b4 = bb_add(a4, v2); BabyBearElement b5 = bb_sub(a4, v2);
            BabyBearElement v3 = bb_mul(a7, ws);
            BabyBearElement b6 = bb_add(a6, v3); BabyBearElement b7 = bb_sub(a6, v3);

            // Stage s+1
            BabyBearElement w1a = twiddles[j * stride2];
            BabyBearElement w1b = twiddles[(j + half) * stride2];
            BabyBearElement v4 = bb_mul(b2, w1a);
            BabyBearElement c0 = bb_add(b0, v4); BabyBearElement c2 = bb_sub(b0, v4);
            BabyBearElement v5 = bb_mul(b3, w1b);
            BabyBearElement c1 = bb_add(b1, v5); BabyBearElement c3 = bb_sub(b1, v5);
            BabyBearElement v6 = bb_mul(b6, w1a);
            BabyBearElement c4 = bb_add(b4, v6); BabyBearElement c6 = bb_sub(b4, v6);
            BabyBearElement v7 = bb_mul(b7, w1b);
            BabyBearElement c5 = bb_add(b5, v7); BabyBearElement c7 = bb_sub(b5, v7);

            // Stage s+2
            BabyBearElement w2_0 = twiddles[j * stride4];
            BabyBearElement w2_1 = twiddles[(j + half) * stride4];
            BabyBearElement w2_2 = twiddles[(j + 2*half) * stride4];
            BabyBearElement w2_3 = twiddles[(j + 3*half) * stride4];

            BabyBearElement v8  = bb_mul(c4, w2_0);
            data[base]          = bb_add(c0, v8);  data[base + 4*half] = bb_sub(c0, v8);
            BabyBearElement v9  = bb_mul(c5, w2_1);
            data[base + half]   = bb_add(c1, v9);  data[base + 5*half] = bb_sub(c1, v9);
            BabyBearElement v10 = bb_mul(c6, w2_2);
            data[base + 2*half] = bb_add(c2, v10); data[base + 6*half] = bb_sub(c2, v10);
            BabyBearElement v11 = bb_mul(c7, w2_3);
            data[base + 3*half] = bb_add(c3, v11); data[base + 7*half] = bb_sub(c3, v11);
        }
        if (pass + 1 < num_r8_passes) grid.sync();
    }
}

// ─── Batched Cooperative Outer Kernels ───────────────────────────────────────

__global__ void bb_ntt_outer_r2_batch_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, int batch_size, int start_stage, int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t total_butterflies = static_cast<uint32_t>(batch_size) * (n >> 1);
    const uint32_t half_n = n >> 1;

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n / (2u * half);
        for (uint32_t tid = global_tid; tid < total_butterflies; tid += total_threads) {
            uint32_t batch_id = tid / half_n;
            uint32_t local_tid = tid - batch_id * half_n;
            uint32_t boff = batch_id * n;
            uint32_t group   = local_tid / half;
            uint32_t j       = local_tid % half;
            uint32_t idx_top = boff + group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;
            BabyBearElement u = data[idx_top];
            BabyBearElement v = bb_mul(data[idx_bot], twiddles[j * stride]);
            data[idx_top] = bb_add(u, v);
            data[idx_bot] = bb_sub(u, v);
        }
        if (s + 1 < end_stage) grid.sync();
    }
}

__global__ void bb_ntt_outer_r4_batch_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, int batch_size, int start_stage, int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t total_r4 = static_cast<uint32_t>(batch_size) * (n >> 2);
    const uint32_t quart_n = n >> 2;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;

        for (uint32_t tid = global_tid; tid < total_r4; tid += total_threads) {
            uint32_t batch_id = tid / quart_n;
            uint32_t local_tid = tid - batch_id * quart_n;
            uint32_t boff = batch_id * n;
            const uint32_t group = local_tid >> s;
            const uint32_t j     = local_tid & (half - 1);
            const uint32_t base  = boff + (group << (s + 2)) + j;

            BabyBearElement a0 = data[base];
            BabyBearElement a1 = data[base + half];
            BabyBearElement w1 = twiddles[j * stride];
            BabyBearElement t1 = bb_mul(a1, w1);
            BabyBearElement a0p = bb_add(a0, t1); BabyBearElement a1p = bb_sub(a0, t1);

            BabyBearElement a2 = data[base + (half << 1)];
            BabyBearElement a3 = data[base + (half << 1) + half];
            BabyBearElement t2 = bb_mul(a3, w1);
            BabyBearElement a2p = bb_add(a2, t2); BabyBearElement a3p = bb_sub(a2, t2);

            BabyBearElement w2 = twiddles[j * stride2];
            BabyBearElement t3 = bb_mul(a2p, w2);
            data[base]              = bb_add(a0p, t3);
            data[base + (half << 1)] = bb_sub(a0p, t3);

            BabyBearElement w3 = twiddles[(j + half) * stride2];
            BabyBearElement t4 = bb_mul(a3p, w3);
            data[base + half]              = bb_add(a1p, t4);
            data[base + (half << 1) + half] = bb_sub(a1p, t4);
        }
        if (pass + 1 < num_r4_passes) grid.sync();
    }
}

__global__ void bb_ntt_outer_r8_batch_kernel(
    BabyBearElement* __restrict__ data,
    const BabyBearElement* __restrict__ twiddles,
    uint32_t n, int batch_size, int start_stage, int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t total_r8 = static_cast<uint32_t>(batch_size) * (n >> 3);
    const uint32_t eighth_n = n >> 3;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;
        const uint32_t stride4 = stride >> 2;

        for (uint32_t tid = global_tid; tid < total_r8; tid += total_threads) {
            uint32_t batch_id = tid / eighth_n;
            uint32_t local_tid = tid - batch_id * eighth_n;
            uint32_t boff = batch_id * n;
            const uint32_t group = local_tid >> s;
            const uint32_t j     = local_tid & (half - 1);
            const uint32_t base  = boff + (group << (s + 3)) + j;

            BabyBearElement a0 = data[base];
            BabyBearElement a1 = data[base + half];
            BabyBearElement a2 = data[base + 2*half];
            BabyBearElement a3 = data[base + 3*half];
            BabyBearElement a4 = data[base + 4*half];
            BabyBearElement a5 = data[base + 5*half];
            BabyBearElement a6 = data[base + 6*half];
            BabyBearElement a7 = data[base + 7*half];

            BabyBearElement ws = twiddles[j * stride];
            BabyBearElement v0 = bb_mul(a1, ws);
            BabyBearElement b0 = bb_add(a0, v0); BabyBearElement b1 = bb_sub(a0, v0);
            BabyBearElement v1 = bb_mul(a3, ws);
            BabyBearElement b2 = bb_add(a2, v1); BabyBearElement b3 = bb_sub(a2, v1);
            BabyBearElement v2 = bb_mul(a5, ws);
            BabyBearElement b4 = bb_add(a4, v2); BabyBearElement b5 = bb_sub(a4, v2);
            BabyBearElement v3 = bb_mul(a7, ws);
            BabyBearElement b6 = bb_add(a6, v3); BabyBearElement b7 = bb_sub(a6, v3);

            BabyBearElement w1a = twiddles[j * stride2];
            BabyBearElement w1b = twiddles[(j + half) * stride2];
            BabyBearElement v4 = bb_mul(b2, w1a);
            BabyBearElement c0 = bb_add(b0, v4); BabyBearElement c2 = bb_sub(b0, v4);
            BabyBearElement v5 = bb_mul(b3, w1b);
            BabyBearElement c1 = bb_add(b1, v5); BabyBearElement c3 = bb_sub(b1, v5);
            BabyBearElement v6 = bb_mul(b6, w1a);
            BabyBearElement c4 = bb_add(b4, v6); BabyBearElement c6 = bb_sub(b4, v6);
            BabyBearElement v7 = bb_mul(b7, w1b);
            BabyBearElement c5 = bb_add(b5, v7); BabyBearElement c7 = bb_sub(b5, v7);

            BabyBearElement w2_0 = twiddles[j * stride4];
            BabyBearElement w2_1 = twiddles[(j + half) * stride4];
            BabyBearElement w2_2 = twiddles[(j + 2*half) * stride4];
            BabyBearElement w2_3 = twiddles[(j + 3*half) * stride4];

            BabyBearElement v8  = bb_mul(c4, w2_0);
            data[base]          = bb_add(c0, v8);  data[base + 4*half] = bb_sub(c0, v8);
            BabyBearElement v9  = bb_mul(c5, w2_1);
            data[base + half]   = bb_add(c1, v9);  data[base + 5*half] = bb_sub(c1, v9);
            BabyBearElement v10 = bb_mul(c6, w2_2);
            data[base + 2*half] = bb_add(c2, v10); data[base + 6*half] = bb_sub(c2, v10);
            BabyBearElement v11 = bb_mul(c7, w2_3);
            data[base + 3*half] = bb_add(c3, v11); data[base + 7*half] = bb_sub(c3, v11);
        }
        if (pass + 1 < num_r8_passes) grid.sync();
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

static constexpr uint32_t BB_BLOCK = 256;
static constexpr int BB_MAX_R8_PASSES = 4;
static constexpr int BB_MAX_R4_PASSES = 7;
static constexpr int BB_MAX_R2_STAGES = 7;

static int bb_select_k(int log_n) {
    if (log_n >= 11) return 11;
    if (log_n >= 10) return 10;
    if (log_n >= 9)  return 9;
    return 8;
}

static void bb_launch_fused(int k, BabyBearElement* d_data, const BabyBearElement* d_twiddles,
                             uint32_t n, uint32_t num_blocks, cudaStream_t stream) {
    switch (k) {
        case 11: bb_ntt_fused_kernel<11><<<num_blocks, 1024, 0, stream>>>(d_data, d_twiddles, n); break;
        case 10: bb_ntt_fused_kernel<10><<<num_blocks, 512, 0, stream>>>(d_data, d_twiddles, n); break;
        case 9:  bb_ntt_fused_kernel<9><<<num_blocks, 256, 0, stream>>>(d_data, d_twiddles, n); break;
        default: bb_ntt_fused_kernel<8><<<num_blocks, 128, 0, stream>>>(d_data, d_twiddles, n); break;
    }
}

// ─── Cooperative Launch Helpers ──────────────────────────────────────────────

static int s_bb_coop_r8 = 0;
static int s_bb_coop_r4 = 0;
static int s_bb_coop_r2 = 0;

static int bb_get_coop_max(void* kernel_func) {
    int blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel_func, BB_BLOCK, 0);
    if (err != cudaSuccess) { cudaGetLastError(); return -1; }
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (!prop.cooperativeLaunch) return -1;
    int max_blocks = blocks_per_sm * prop.multiProcessorCount;
    return max_blocks > 0 ? max_blocks : -1;
}

static int bb_get_coop_r8() {
    if (s_bb_coop_r8 != 0) return s_bb_coop_r8;
    s_bb_coop_r8 = bb_get_coop_max((void*)bb_ntt_outer_r8_kernel);
    return s_bb_coop_r8;
}
static int bb_get_coop_r4() {
    if (s_bb_coop_r4 != 0) return s_bb_coop_r4;
    s_bb_coop_r4 = bb_get_coop_max((void*)bb_ntt_outer_r4_kernel);
    return s_bb_coop_r4;
}
static int bb_get_coop_r2() {
    if (s_bb_coop_r2 != 0) return s_bb_coop_r2;
    s_bb_coop_r2 = bb_get_coop_max((void*)bb_ntt_outer_r2_kernel);
    return s_bb_coop_r2;
}

// ─── Outer Stage Dispatch ────────────────────────────────────────────────────

static void bb_dispatch_outer(
    BabyBearElement* d_data, const BabyBearElement* d_twiddles,
    uint32_t n, int outer_start, int outer_end,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;
    if (num_outer <= 0) return;

    // Try radix-8
    if (num_outer >= 3) {
        int r8_max = bb_get_coop_r8();
        if (r8_max > 0) {
            int num_r8 = num_outer / 3;
            uint32_t needed = (n / 8 + BB_BLOCK - 1) / BB_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                                 ? static_cast<uint32_t>(r8_max) : needed;

            int r8_stage = outer_start;
            for (int done = 0; done < num_r8; ) {
                int passes = num_r8 - done;
                if (passes > BB_MAX_R8_PASSES) passes = BB_MAX_R8_PASSES;
                int ss = r8_stage; int np = passes;
                void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };
                cudaError_t err = cudaLaunchCooperativeKernel(
                    (void*)bb_ntt_outer_r8_kernel,
                    dim3(grid_size), dim3(BB_BLOCK), args, 0, stream);
                if (err != cudaSuccess) goto fallback_r4;
                done += passes; r8_stage += 3 * passes;
            }

            int leftover = num_outer % 3;
            int ls = outer_start + num_r8 * 3;
            if (leftover == 2) {
                int r4_max = bb_get_coop_r4();
                if (r4_max > 0) {
                    uint32_t r4_needed = (n / 4 + BB_BLOCK - 1) / BB_BLOCK;
                    uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                                       ? static_cast<uint32_t>(r4_max) : r4_needed;
                    int ss = ls; int np = 1;
                    void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };
                    cudaLaunchCooperativeKernel((void*)bb_ntt_outer_r4_kernel,
                        dim3(r4_grid), dim3(BB_BLOCK), args, 0, stream);
                } else {
                    for (int s2 = ls; s2 < ls + 2; ++s2) {
                        uint32_t half = 1u << s2; uint32_t stride = n / (2u * half);
                        bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                            d_data, d_twiddles, n, half, stride);
                    }
                }
            } else if (leftover == 1) {
                uint32_t half = 1u << ls; uint32_t stride = n / (2u * half);
                bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
            return;
        }
    }

fallback_r4:
    if (num_outer >= 2) {
        int r4_max = bb_get_coop_r4();
        if (r4_max > 0) {
            int num_r4p = num_outer / 2;
            uint32_t needed = (n / 4 + BB_BLOCK - 1) / BB_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                                 ? static_cast<uint32_t>(r4_max) : needed;
            int r4_stage = outer_start;
            for (int done = 0; done < num_r4p; ) {
                int passes = num_r4p - done;
                if (passes > BB_MAX_R4_PASSES) passes = BB_MAX_R4_PASSES;
                int ss = r4_stage; int np = passes;
                void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };
                cudaError_t err = cudaLaunchCooperativeKernel(
                    (void*)bb_ntt_outer_r4_kernel,
                    dim3(grid_size), dim3(BB_BLOCK), args, 0, stream);
                if (err != cudaSuccess) goto fallback_r2;
                done += passes; r4_stage += 2 * passes;
            }
            if (num_outer % 2 == 1) {
                int last = outer_end - 1;
                uint32_t half = 1u << last; uint32_t stride = n / (2u * half);
                bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
            return;
        }
    }

fallback_r2:
    {
        int r2_max = bb_get_coop_r2();
        if (r2_max > 0) {
            uint32_t needed = (n / 2 + BB_BLOCK - 1) / BB_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r2_max) < needed)
                                 ? static_cast<uint32_t>(r2_max) : needed;
            for (int s = outer_start; s < outer_end; ) {
                int be = s + BB_MAX_R2_STAGES;
                if (be > outer_end) be = outer_end;
                int ss = s; int se = be;
                void* args[] = { &d_data, &d_twiddles, &n, &ss, &se };
                cudaError_t err = cudaLaunchCooperativeKernel(
                    (void*)bb_ntt_outer_r2_kernel,
                    dim3(grid_size), dim3(BB_BLOCK), args, 0, stream);
                if (err != cudaSuccess) {
                    for (int s2 = s; s2 < outer_end; ++s2) {
                        uint32_t half = 1u << s2; uint32_t stride = n / (2u * half);
                        bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                            d_data, d_twiddles, n, half, stride);
                    }
                    return;
                }
                s = be;
            }
        } else {
            for (int s = outer_start; s < outer_end; ++s) {
                uint32_t half = 1u << s; uint32_t stride = n / (2u * half);
                bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
        }
    }
}

// ─── Batched Outer Dispatch ──────────────────────────────────────────────────

static void bb_dispatch_outer_batch(
    BabyBearElement* d_data, const BabyBearElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int outer_end,
    cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;
    if (num_outer <= 0) return;

    uint32_t total_half = static_cast<uint32_t>(batch_size) * (n >> 1);

    if (num_outer >= 3) {
        int r8_max = bb_get_coop_max((void*)bb_ntt_outer_r8_batch_kernel);
        if (r8_max > 0) {
            int num_r8 = num_outer / 3;
            uint32_t total_r8 = static_cast<uint32_t>(batch_size) * (n >> 3);
            uint32_t needed = (total_r8 + BB_BLOCK - 1) / BB_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                                 ? static_cast<uint32_t>(r8_max) : needed;

            int r8_stage = outer_start;
            for (int done = 0; done < num_r8; ) {
                int passes = num_r8 - done;
                if (passes > BB_MAX_R8_PASSES) passes = BB_MAX_R8_PASSES;
                int ss = r8_stage; int np = passes;
                void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &np };
                cudaLaunchCooperativeKernel((void*)bb_ntt_outer_r8_batch_kernel,
                    dim3(grid_size), dim3(BB_BLOCK), args, 0, stream);
                done += passes; r8_stage += 3 * passes;
            }

            int leftover = num_outer % 3;
            int ls = outer_start + num_r8 * 3;
            if (leftover == 2) {
                int r4_max = bb_get_coop_max((void*)bb_ntt_outer_r4_batch_kernel);
                if (r4_max > 0) {
                    uint32_t r4_total = static_cast<uint32_t>(batch_size) * (n >> 2);
                    uint32_t r4_needed = (r4_total + BB_BLOCK - 1) / BB_BLOCK;
                    uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                                       ? static_cast<uint32_t>(r4_max) : r4_needed;
                    int ss = ls; int np = 1;
                    void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &np };
                    cudaLaunchCooperativeKernel((void*)bb_ntt_outer_r4_batch_kernel,
                        dim3(r4_grid), dim3(BB_BLOCK), args, 0, stream);
                } else {
                    int r2_max = bb_get_coop_max((void*)bb_ntt_outer_r2_batch_kernel);
                    if (r2_max > 0) {
                        uint32_t r2_needed = (total_half + BB_BLOCK - 1) / BB_BLOCK;
                        uint32_t r2_grid = (static_cast<uint32_t>(r2_max) < r2_needed)
                                           ? static_cast<uint32_t>(r2_max) : r2_needed;
                        int ss = ls; int se = ls + 2;
                        void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &se };
                        cudaLaunchCooperativeKernel((void*)bb_ntt_outer_r2_batch_kernel,
                            dim3(r2_grid), dim3(BB_BLOCK), args, 0, stream);
                    }
                }
            } else if (leftover == 1) {
                int r2_max = bb_get_coop_max((void*)bb_ntt_outer_r2_batch_kernel);
                if (r2_max > 0) {
                    uint32_t r2_needed = (total_half + BB_BLOCK - 1) / BB_BLOCK;
                    uint32_t r2_grid = (static_cast<uint32_t>(r2_max) < r2_needed)
                                       ? static_cast<uint32_t>(r2_max) : r2_needed;
                    int ss = ls; int se = ls + 1;
                    void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &se };
                    cudaLaunchCooperativeKernel((void*)bb_ntt_outer_r2_batch_kernel,
                        dim3(r2_grid), dim3(BB_BLOCK), args, 0, stream);
                }
            }
            return;
        }
    }

    // Fallback: radix-2 batch
    {
        int r2_max = bb_get_coop_max((void*)bb_ntt_outer_r2_batch_kernel);
        if (r2_max > 0) {
            uint32_t needed = (total_half + BB_BLOCK - 1) / BB_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r2_max) < needed)
                                 ? static_cast<uint32_t>(r2_max) : needed;
            for (int s = outer_start; s < outer_end; ) {
                int be = s + BB_MAX_R2_STAGES;
                if (be > outer_end) be = outer_end;
                int ss = s; int se = be;
                void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &se };
                cudaLaunchCooperativeKernel((void*)bb_ntt_outer_r2_batch_kernel,
                    dim3(grid_size), dim3(BB_BLOCK), args, 0, stream);
                s = be;
            }
        }
    }
}

// ─── Forward/Inverse NTT Implementation ──────────────────────────────────────

static void bb_ntt_forward_impl(
    BabyBearElement* d_data, size_t n,
    const BabyBearElement* d_twiddles, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = bb_log2(n);
    const uint32_t grid      = (N + BB_BLOCK - 1) / BB_BLOCK;
    const uint32_t grid_half = (N / 2 + BB_BLOCK - 1) / BB_BLOCK;

    bb_ntt_bit_reverse_kernel<<<grid, BB_BLOCK, 0, stream>>>(d_data, N, log_n);

    const int fused_k = bb_select_k(log_n);
    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;
        bb_launch_fused(fused_k, d_data, d_twiddles, N, num_blocks, stream);
        bb_dispatch_outer(d_data, d_twiddles, N, fused_k, log_n, grid_half, stream);
    } else {
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    }
}

static void bb_ntt_inverse_impl(
    BabyBearElement* d_data, size_t n,
    const BabyBearElement* d_inv_twiddles, BabyBearElement n_inv,
    cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = bb_log2(n);
    const uint32_t grid      = (N + BB_BLOCK - 1) / BB_BLOCK;
    const uint32_t grid_half = (N / 2 + BB_BLOCK - 1) / BB_BLOCK;

    bb_ntt_bit_reverse_kernel<<<grid, BB_BLOCK, 0, stream>>>(d_data, N, log_n);

    const int fused_k = bb_select_k(log_n);
    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;
        bb_launch_fused(fused_k, d_data, d_inv_twiddles, N, num_blocks, stream);
        bb_dispatch_outer(d_data, d_inv_twiddles, N, fused_k, log_n, grid_half, stream);
    } else {
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            bb_ntt_butterfly_kernel<<<grid_half, BB_BLOCK, 0, stream>>>(
                d_data, d_inv_twiddles, N, half, stride);
        }
    }

    bb_ntt_scale_kernel<<<grid, BB_BLOCK, 0, stream>>>(d_data, n_inv, N);
}

// ─── Public API ──────────────────────────────────────────────────────────────

void ntt_forward_babybear(BabyBearElement* d_data, size_t n, cudaStream_t stream) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_bb_twiddles(n);
    bb_ntt_forward_impl(d_data, n, s_bb_fwd_twiddles, stream);
}

void ntt_inverse_babybear(BabyBearElement* d_data, size_t n, cudaStream_t stream) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_bb_twiddles(n);
    bb_ntt_inverse_impl(d_data, n, s_bb_inv_twiddles, s_bb_n_inv, stream);
}

void ntt_forward_batch_babybear(BabyBearElement* d_data, int batch_size, size_t n, cudaStream_t stream) {
    if (batch_size <= 0) return;
    if (batch_size == 1) { ntt_forward_babybear(d_data, n, stream); return; }
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_bb_twiddles(n);

    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = bb_log2(n);
    const uint32_t total = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total + BB_BLOCK - 1) / BB_BLOCK;

    bb_ntt_bit_reverse_batch_kernel<<<grid_total, BB_BLOCK, 0, stream>>>(d_data, N, log_n, total);

    const int fused_k = bb_select_k(log_n);
    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);
        bb_launch_fused(fused_k, d_data, s_bb_fwd_twiddles, N, num_blocks, stream);
        bb_dispatch_outer_batch(d_data, s_bb_fwd_twiddles, N, batch_size, fused_k, log_n, stream);
    } else {
        for (int b = 0; b < batch_size; ++b)
            bb_ntt_forward_impl(d_data + b * N, n, s_bb_fwd_twiddles, stream);
    }
}

void ntt_inverse_batch_babybear(BabyBearElement* d_data, int batch_size, size_t n, cudaStream_t stream) {
    if (batch_size <= 0) return;
    if (batch_size == 1) { ntt_inverse_babybear(d_data, n, stream); return; }
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_bb_twiddles(n);

    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = bb_log2(n);
    const uint32_t total = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total + BB_BLOCK - 1) / BB_BLOCK;

    bb_ntt_bit_reverse_batch_kernel<<<grid_total, BB_BLOCK, 0, stream>>>(d_data, N, log_n, total);

    const int fused_k = bb_select_k(log_n);
    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);
        bb_launch_fused(fused_k, d_data, s_bb_inv_twiddles, N, num_blocks, stream);
        bb_dispatch_outer_batch(d_data, s_bb_inv_twiddles, N, batch_size, fused_k, log_n, stream);
    } else {
        for (int b = 0; b < batch_size; ++b)
            bb_ntt_inverse_impl(d_data + b * N, n, s_bb_inv_twiddles, s_bb_n_inv, stream);
    }

    bb_ntt_scale_kernel<<<grid_total, BB_BLOCK, 0, stream>>>(d_data, s_bb_n_inv, total);
}
