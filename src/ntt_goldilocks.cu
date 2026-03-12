// src/ntt_goldilocks.cu
// NTT implementation for the Goldilocks field: p = 2^64 - 2^32 + 1
// Single uint64_t elements (8 bytes). Standard form throughout.
//
// Architecture:
//   - Fused inner kernel: K=8/9/10/11 (warp shuffle stages 0-4 + shmem 5..K-1)
//   - Cooperative outer stages: radix-8 → radix-4 → radix-2 fallback
//   - K=11: 2048 elements × 8B = 16 KB shmem, 1024 threads (max practical K)
//   - For n=2^22 with K=11: 11 outer stages
//
// Goldilocks mul is ~5-8 instructions (vs 528 for BLS12-381 Montgomery),
// so register pressure is minimal and radix-8 outer stages work well.

#include "ntt_goldilocks.cuh"
#include "ff_goldilocks.cuh"
#include "cuda_utils.cuh"
#include "ff_reference.h"

#include <cooperative_groups.h>
#include <cstdio>
#include <cassert>
#include <vector>

// ─── Twiddle Cache ───────────────────────────────────────────────────────────

static GoldilocksElement* s_gl_fwd_twiddles = nullptr;
static GoldilocksElement* s_gl_inv_twiddles = nullptr;
static GoldilocksElement  s_gl_n_inv;
static size_t             s_gl_cached_n = 0;

static void free_gl_twiddles() {
    if (s_gl_fwd_twiddles) { cudaFree(s_gl_fwd_twiddles); s_gl_fwd_twiddles = nullptr; }
    if (s_gl_inv_twiddles) { cudaFree(s_gl_inv_twiddles); s_gl_inv_twiddles = nullptr; }
    s_gl_cached_n = 0;
}

static void ensure_gl_twiddles(size_t n) {
    if (s_gl_cached_n == n) return;
    free_gl_twiddles();

    size_t half = n / 2;

    // Compute twiddles on CPU using ff_ref
    ff_ref::GlRef omega     = ff_ref::gl_get_root_of_unity(n);
    ff_ref::GlRef omega_inv = ff_ref::gl_inv(omega);

    std::vector<GoldilocksElement> h_fwd(half), h_inv(half);

    ff_ref::GlRef wk     = ff_ref::GlRef::one();
    ff_ref::GlRef wk_inv = ff_ref::GlRef::one();
    for (size_t k = 0; k < half; ++k) {
        h_fwd[k] = {wk.val};
        h_inv[k] = {wk_inv.val};
        wk     = ff_ref::gl_mul(wk, omega);
        wk_inv = ff_ref::gl_mul(wk_inv, omega_inv);
    }

    CUDA_CHECK(cudaMalloc(&s_gl_fwd_twiddles, half * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&s_gl_inv_twiddles, half * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(s_gl_fwd_twiddles, h_fwd.data(),
                          half * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_gl_inv_twiddles, h_inv.data(),
                          half * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    // n^{-1} mod p
    ff_ref::GlRef n_inv = ff_ref::gl_inv(ff_ref::GlRef::from_u64(static_cast<uint64_t>(n)));
    s_gl_n_inv = {n_inv.val};

    s_gl_cached_n = n;
}

// ─── Helper ──────────────────────────────────────────────────────────────────

static int gl_log2(size_t n) {
    int r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// ─── Warp-Shuffle Helper ─────────────────────────────────────────────────────
// Shuffle a GoldilocksElement (uint64_t = 2 × uint32_t) across lanes.

__device__ __forceinline__
GoldilocksElement gl_shfl_xor(const GoldilocksElement& val, int xor_mask) {
    uint32_t lo = static_cast<uint32_t>(val.val);
    uint32_t hi = static_cast<uint32_t>(val.val >> 32);
    lo = __shfl_xor_sync(0xFFFFFFFF, lo, xor_mask);
    hi = __shfl_xor_sync(0xFFFFFFFF, hi, xor_mask);
    return {(static_cast<uint64_t>(hi) << 32) | lo};
}

// ─── Warp-Shuffle Butterfly ──────────────────────────────────────────────────

__device__ __forceinline__
GoldilocksElement gl_butterfly_shfl(const GoldilocksElement& my_elem,
                                     const GoldilocksElement& twiddle,
                                     int xor_mask, bool is_top) {
    GoldilocksElement partner = gl_shfl_xor(my_elem, xor_mask);
    GoldilocksElement A = is_top ? my_elem : partner;
    GoldilocksElement B = is_top ? partner : my_elem;
    GoldilocksElement v = gl_mul(B, twiddle);
    return is_top ? gl_add(A, v) : gl_sub(A, v);
}

// ─── Fused Inner Kernel ──────────────────────────────────────────────────────
// Fuses K radix-2 butterfly stages into a single kernel launch.
// Stages 0-4: warp shuffle (stride 1..16, intra-warp).
// Stages 5..K-1: shared memory + __syncthreads() (cross-warp).

template <int K>
__global__ void gl_ntt_fused_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
    uint32_t n
) {
    constexpr int ELEMS   = 1 << K;
    constexpr int THREADS = ELEMS >> 1;
    constexpr int SHFL_STAGES = (K < 5) ? K : 5;

    __shared__ GoldilocksElement sdata[ELEMS];

    const uint32_t boff = blockIdx.x * ELEMS;
    const uint32_t t = threadIdx.x;

    // Load from global memory
    GoldilocksElement reg_lo = data[boff + t];
    GoldilocksElement reg_hi = data[boff + t + THREADS];

    // Stages 0-4: warp shuffle
    #pragma unroll
    for (int s = 0; s < SHFL_STAGES; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n >> (s + 1);
        const uint32_t j      = t & (half - 1);
        const bool is_top     = (t & half) == 0;
        const int xor_mask    = static_cast<int>(half);

        GoldilocksElement w = twiddles[j * stride];
        reg_lo = gl_butterfly_shfl(reg_lo, w, xor_mask, is_top);
        reg_hi = gl_butterfly_shfl(reg_hi, w, xor_mask, is_top);
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

        GoldilocksElement w = twiddles[j * stride];
        GoldilocksElement u = sdata[idx_top];
        GoldilocksElement v = gl_mul(sdata[idx_bot], w);
        sdata[idx_top] = gl_add(u, v);
        sdata[idx_bot] = gl_sub(u, v);
        __syncthreads();
    }

    // Store to global memory
    data[boff + t]           = sdata[t];
    data[boff + t + THREADS] = sdata[t + THREADS];
}

// Explicit template instantiations
template __global__ void gl_ntt_fused_kernel<8>(
    GoldilocksElement* __restrict__, const GoldilocksElement* __restrict__, uint32_t);
template __global__ void gl_ntt_fused_kernel<9>(
    GoldilocksElement* __restrict__, const GoldilocksElement* __restrict__, uint32_t);
template __global__ void gl_ntt_fused_kernel<10>(
    GoldilocksElement* __restrict__, const GoldilocksElement* __restrict__, uint32_t);
template __global__ void gl_ntt_fused_kernel<11>(
    GoldilocksElement* __restrict__, const GoldilocksElement* __restrict__, uint32_t);

// ─── Bit-Reverse Kernel ──────────────────────────────────────────────────────

__global__ void gl_ntt_bit_reverse_kernel(GoldilocksElement* data, uint32_t n, uint32_t log_n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t j = __brev(i) >> (32 - log_n);
    if (i < j) {
        GoldilocksElement temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

__global__ void gl_ntt_bit_reverse_batch_kernel(
    GoldilocksElement* data, uint32_t n, uint32_t log_n, uint32_t total_elements
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    uint32_t batch_id = tid / n;
    uint32_t i = tid - batch_id * n;
    uint32_t j = __brev(i) >> (32 - log_n);

    if (i < j) {
        uint32_t base = batch_id * n;
        GoldilocksElement temp = data[base + i];
        data[base + i] = data[base + j];
        data[base + j] = temp;
    }
}

// ─── Scale Kernel ────────────────────────────────────────────────────────────

__global__ void gl_ntt_scale_kernel(GoldilocksElement* data, GoldilocksElement scalar, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = gl_mul(data[i], scalar);
}

// ─── Per-Stage Butterfly Kernel (small-n fallback) ───────────────────────────

__global__ void gl_ntt_butterfly_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
    uint32_t n, uint32_t half, uint32_t stride
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    uint32_t group = tid / half;
    uint32_t j     = tid % half;

    uint32_t idx_top = group * (2 * half) + j;
    uint32_t idx_bot = idx_top + half;

    GoldilocksElement u = data[idx_top];
    GoldilocksElement v = gl_mul(data[idx_bot], twiddles[j * stride]);
    data[idx_top] = gl_add(u, v);
    data[idx_bot] = gl_sub(u, v);
}

// ─── Cooperative Outer Kernels ───────────────────────────────────────────────

// Radix-2 cooperative outer kernel
__global__ void gl_ntt_outer_r2_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
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

            GoldilocksElement u = data[idx_top];
            GoldilocksElement v = gl_mul(data[idx_bot], twiddles[j * stride]);
            data[idx_top] = gl_add(u, v);
            data[idx_bot] = gl_sub(u, v);
        }

        if (s + 1 < end_stage) grid.sync();
    }
}

// Radix-4 cooperative outer kernel
__global__ void gl_ntt_outer_r4_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
    uint32_t n, int start_stage, int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r4_butterflies = n >> 2;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;

        for (uint32_t tid = global_tid; tid < num_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            GoldilocksElement a0 = data[base];
            GoldilocksElement a1 = data[base + half];
            GoldilocksElement w1 = twiddles[j * stride];

            GoldilocksElement t1 = gl_mul(a1, w1);
            GoldilocksElement a0p = gl_add(a0, t1);
            GoldilocksElement a1p = gl_sub(a0, t1);

            GoldilocksElement a2 = data[base + (half << 1)];
            GoldilocksElement a3 = data[base + (half << 1) + half];

            GoldilocksElement t2 = gl_mul(a3, w1);
            GoldilocksElement a2p = gl_add(a2, t2);
            GoldilocksElement a3p = gl_sub(a2, t2);

            GoldilocksElement w2 = twiddles[j * stride2];
            GoldilocksElement t3 = gl_mul(a2p, w2);
            data[base]              = gl_add(a0p, t3);
            data[base + (half << 1)] = gl_sub(a0p, t3);

            GoldilocksElement w3 = twiddles[(j + half) * stride2];
            GoldilocksElement t4 = gl_mul(a3p, w3);
            data[base + half]              = gl_add(a1p, t4);
            data[base + (half << 1) + half] = gl_sub(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) grid.sync();
    }
}

// Radix-8 cooperative outer kernel
__global__ void gl_ntt_outer_r8_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
    uint32_t n, int start_stage, int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r8_butterflies = n >> 3;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;
        const uint32_t stride4 = stride >> 2;

        for (uint32_t tid = global_tid; tid < num_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            // Load 8 elements
            GoldilocksElement a0 = data[base];
            GoldilocksElement a1 = data[base + half];
            GoldilocksElement a2 = data[base + 2 * half];
            GoldilocksElement a3 = data[base + 3 * half];
            GoldilocksElement a4 = data[base + 4 * half];
            GoldilocksElement a5 = data[base + 5 * half];
            GoldilocksElement a6 = data[base + 6 * half];
            GoldilocksElement a7 = data[base + 7 * half];

            // Stage s: 4 radix-2 butterflies
            GoldilocksElement w_s = twiddles[j * stride];
            GoldilocksElement v0 = gl_mul(a1, w_s);
            GoldilocksElement b0 = gl_add(a0, v0); GoldilocksElement b1 = gl_sub(a0, v0);
            GoldilocksElement v1 = gl_mul(a3, w_s);
            GoldilocksElement b2 = gl_add(a2, v1); GoldilocksElement b3 = gl_sub(a2, v1);
            GoldilocksElement v2 = gl_mul(a5, w_s);
            GoldilocksElement b4 = gl_add(a4, v2); GoldilocksElement b5 = gl_sub(a4, v2);
            GoldilocksElement v3 = gl_mul(a7, w_s);
            GoldilocksElement b6 = gl_add(a6, v3); GoldilocksElement b7 = gl_sub(a6, v3);

            // Stage s+1: 4 radix-2 butterflies
            GoldilocksElement w1a = twiddles[j * stride2];
            GoldilocksElement w1b = twiddles[(j + half) * stride2];
            GoldilocksElement v4 = gl_mul(b2, w1a);
            GoldilocksElement c0 = gl_add(b0, v4); GoldilocksElement c2 = gl_sub(b0, v4);
            GoldilocksElement v5 = gl_mul(b3, w1b);
            GoldilocksElement c1 = gl_add(b1, v5); GoldilocksElement c3 = gl_sub(b1, v5);
            GoldilocksElement v6 = gl_mul(b6, w1a);
            GoldilocksElement c4 = gl_add(b4, v6); GoldilocksElement c6 = gl_sub(b4, v6);
            GoldilocksElement v7 = gl_mul(b7, w1b);
            GoldilocksElement c5 = gl_add(b5, v7); GoldilocksElement c7 = gl_sub(b5, v7);

            // Stage s+2: 4 radix-2 butterflies
            GoldilocksElement w2_0 = twiddles[j * stride4];
            GoldilocksElement w2_1 = twiddles[(j + half) * stride4];
            GoldilocksElement w2_2 = twiddles[(j + 2 * half) * stride4];
            GoldilocksElement w2_3 = twiddles[(j + 3 * half) * stride4];

            GoldilocksElement v8  = gl_mul(c4, w2_0);
            data[base]          = gl_add(c0, v8);  data[base + 4*half] = gl_sub(c0, v8);
            GoldilocksElement v9  = gl_mul(c5, w2_1);
            data[base + half]   = gl_add(c1, v9);  data[base + 5*half] = gl_sub(c1, v9);
            GoldilocksElement v10 = gl_mul(c6, w2_2);
            data[base + 2*half] = gl_add(c2, v10); data[base + 6*half] = gl_sub(c2, v10);
            GoldilocksElement v11 = gl_mul(c7, w2_3);
            data[base + 3*half] = gl_add(c3, v11); data[base + 7*half] = gl_sub(c3, v11);
        }

        if (pass + 1 < num_r8_passes) grid.sync();
    }
}

// ─── Batched Cooperative Outer Kernels ───────────────────────────────────────

__global__ void gl_ntt_outer_r2_batch_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
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

            GoldilocksElement u = data[idx_top];
            GoldilocksElement v = gl_mul(data[idx_bot], twiddles[j * stride]);
            data[idx_top] = gl_add(u, v);
            data[idx_bot] = gl_sub(u, v);
        }

        if (s + 1 < end_stage) grid.sync();
    }
}

__global__ void gl_ntt_outer_r4_batch_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
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

            GoldilocksElement a0 = data[base];
            GoldilocksElement a1 = data[base + half];
            GoldilocksElement w1 = twiddles[j * stride];

            GoldilocksElement t1 = gl_mul(a1, w1);
            GoldilocksElement a0p = gl_add(a0, t1);
            GoldilocksElement a1p = gl_sub(a0, t1);

            GoldilocksElement a2 = data[base + (half << 1)];
            GoldilocksElement a3 = data[base + (half << 1) + half];

            GoldilocksElement t2 = gl_mul(a3, w1);
            GoldilocksElement a2p = gl_add(a2, t2);
            GoldilocksElement a3p = gl_sub(a2, t2);

            GoldilocksElement w2 = twiddles[j * stride2];
            GoldilocksElement t3 = gl_mul(a2p, w2);
            data[base]              = gl_add(a0p, t3);
            data[base + (half << 1)] = gl_sub(a0p, t3);

            GoldilocksElement w3 = twiddles[(j + half) * stride2];
            GoldilocksElement t4 = gl_mul(a3p, w3);
            data[base + half]              = gl_add(a1p, t4);
            data[base + (half << 1) + half] = gl_sub(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) grid.sync();
    }
}

__global__ void gl_ntt_outer_r8_batch_kernel(
    GoldilocksElement* __restrict__ data,
    const GoldilocksElement* __restrict__ twiddles,
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

            GoldilocksElement a0 = data[base];
            GoldilocksElement a1 = data[base + half];
            GoldilocksElement a2 = data[base + 2 * half];
            GoldilocksElement a3 = data[base + 3 * half];
            GoldilocksElement a4 = data[base + 4 * half];
            GoldilocksElement a5 = data[base + 5 * half];
            GoldilocksElement a6 = data[base + 6 * half];
            GoldilocksElement a7 = data[base + 7 * half];

            // Stage s
            GoldilocksElement w_s = twiddles[j * stride];
            GoldilocksElement v0 = gl_mul(a1, w_s);
            GoldilocksElement b0 = gl_add(a0, v0); GoldilocksElement b1 = gl_sub(a0, v0);
            GoldilocksElement v1 = gl_mul(a3, w_s);
            GoldilocksElement b2 = gl_add(a2, v1); GoldilocksElement b3 = gl_sub(a2, v1);
            GoldilocksElement v2 = gl_mul(a5, w_s);
            GoldilocksElement b4 = gl_add(a4, v2); GoldilocksElement b5 = gl_sub(a4, v2);
            GoldilocksElement v3 = gl_mul(a7, w_s);
            GoldilocksElement b6 = gl_add(a6, v3); GoldilocksElement b7 = gl_sub(a6, v3);

            // Stage s+1
            GoldilocksElement w1a = twiddles[j * stride2];
            GoldilocksElement w1b = twiddles[(j + half) * stride2];
            GoldilocksElement v4 = gl_mul(b2, w1a);
            GoldilocksElement c0 = gl_add(b0, v4); GoldilocksElement c2 = gl_sub(b0, v4);
            GoldilocksElement v5 = gl_mul(b3, w1b);
            GoldilocksElement c1 = gl_add(b1, v5); GoldilocksElement c3 = gl_sub(b1, v5);
            GoldilocksElement v6 = gl_mul(b6, w1a);
            GoldilocksElement c4 = gl_add(b4, v6); GoldilocksElement c6 = gl_sub(b4, v6);
            GoldilocksElement v7 = gl_mul(b7, w1b);
            GoldilocksElement c5 = gl_add(b5, v7); GoldilocksElement c7 = gl_sub(b5, v7);

            // Stage s+2
            GoldilocksElement w2_0 = twiddles[j * stride4];
            GoldilocksElement w2_1 = twiddles[(j + half) * stride4];
            GoldilocksElement w2_2 = twiddles[(j + 2 * half) * stride4];
            GoldilocksElement w2_3 = twiddles[(j + 3 * half) * stride4];

            GoldilocksElement v8  = gl_mul(c4, w2_0);
            data[base]          = gl_add(c0, v8);  data[base + 4*half] = gl_sub(c0, v8);
            GoldilocksElement v9  = gl_mul(c5, w2_1);
            data[base + half]   = gl_add(c1, v9);  data[base + 5*half] = gl_sub(c1, v9);
            GoldilocksElement v10 = gl_mul(c6, w2_2);
            data[base + 2*half] = gl_add(c2, v10); data[base + 6*half] = gl_sub(c2, v10);
            GoldilocksElement v11 = gl_mul(c7, w2_3);
            data[base + 3*half] = gl_add(c3, v11); data[base + 7*half] = gl_sub(c3, v11);
        }

        if (pass + 1 < num_r8_passes) grid.sync();
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

static constexpr uint32_t GL_BLOCK = 256;
static constexpr int GL_MAX_R8_PASSES = 4;
static constexpr int GL_MAX_R4_PASSES = 7;
static constexpr int GL_MAX_R2_STAGES = 7;

static int gl_select_k(int log_n) {
    if (log_n >= 11) return 11;
    if (log_n >= 10) return 10;
    if (log_n >= 9)  return 9;
    return 8;
}

static void gl_launch_fused(int k, GoldilocksElement* d_data, const GoldilocksElement* d_twiddles,
                             uint32_t n, uint32_t num_blocks, cudaStream_t stream) {
    switch (k) {
        case 11: gl_ntt_fused_kernel<11><<<num_blocks, 1024, 0, stream>>>(d_data, d_twiddles, n); break;
        case 10: gl_ntt_fused_kernel<10><<<num_blocks, 512, 0, stream>>>(d_data, d_twiddles, n); break;
        case 9:  gl_ntt_fused_kernel<9><<<num_blocks, 256, 0, stream>>>(d_data, d_twiddles, n); break;
        default: gl_ntt_fused_kernel<8><<<num_blocks, 128, 0, stream>>>(d_data, d_twiddles, n); break;
    }
}

// ─── Cooperative Launch Helpers ──────────────────────────────────────────────

static int s_gl_coop_r8 = 0;
static int s_gl_coop_r4 = 0;
static int s_gl_coop_r2 = 0;

static int gl_get_coop_max(void* kernel_func) {
    int blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel_func, GL_BLOCK, 0);
    if (err != cudaSuccess) { cudaGetLastError(); return -1; }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (!prop.cooperativeLaunch) return -1;
    int max_blocks = blocks_per_sm * prop.multiProcessorCount;
    return max_blocks > 0 ? max_blocks : -1;
}

static int gl_get_coop_r8() {
    if (s_gl_coop_r8 != 0) return s_gl_coop_r8;
    s_gl_coop_r8 = gl_get_coop_max((void*)gl_ntt_outer_r8_kernel);
    return s_gl_coop_r8;
}

static int gl_get_coop_r4() {
    if (s_gl_coop_r4 != 0) return s_gl_coop_r4;
    s_gl_coop_r4 = gl_get_coop_max((void*)gl_ntt_outer_r4_kernel);
    return s_gl_coop_r4;
}

static int gl_get_coop_r2() {
    if (s_gl_coop_r2 != 0) return s_gl_coop_r2;
    s_gl_coop_r2 = gl_get_coop_max((void*)gl_ntt_outer_r2_kernel);
    return s_gl_coop_r2;
}

// ─── Outer Stage Dispatch ────────────────────────────────────────────────────

static void gl_dispatch_outer(
    GoldilocksElement* d_data, const GoldilocksElement* d_twiddles,
    uint32_t n, int outer_start, int outer_end,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;
    if (num_outer <= 0) return;

    // Try radix-8
    if (num_outer >= 3) {
        int r8_max = gl_get_coop_r8();
        if (r8_max > 0) {
            int num_r8 = num_outer / 3;
            uint32_t needed = (n / 8 + GL_BLOCK - 1) / GL_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                                 ? static_cast<uint32_t>(r8_max) : needed;

            int r8_stage = outer_start;
            for (int done = 0; done < num_r8; ) {
                int passes = num_r8 - done;
                if (passes > GL_MAX_R8_PASSES) passes = GL_MAX_R8_PASSES;

                int ss = r8_stage;
                int np = passes;
                void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };
                cudaError_t err = cudaLaunchCooperativeKernel(
                    (void*)gl_ntt_outer_r8_kernel,
                    dim3(grid_size), dim3(GL_BLOCK), args, 0, stream);
                if (err != cudaSuccess) goto fallback_r4;

                done += passes;
                r8_stage += 3 * passes;
            }

            // Handle leftover: 0, 1 (radix-2), or 2 (radix-4) stages
            int leftover = num_outer % 3;
            int leftover_start = outer_start + num_r8 * 3;

            if (leftover == 2) {
                // One radix-4 pass
                int r4_max = gl_get_coop_r4();
                if (r4_max > 0) {
                    uint32_t r4_needed = (n / 4 + GL_BLOCK - 1) / GL_BLOCK;
                    uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                                       ? static_cast<uint32_t>(r4_max) : r4_needed;
                    int ss = leftover_start;
                    int np = 1;
                    void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };
                    cudaLaunchCooperativeKernel(
                        (void*)gl_ntt_outer_r4_kernel,
                        dim3(r4_grid), dim3(GL_BLOCK), args, 0, stream);
                } else {
                    // fallback to 2 radix-2 stages
                    for (int s = leftover_start; s < leftover_start + 2; ++s) {
                        uint32_t half = 1u << s;
                        uint32_t stride = n / (2u * half);
                        gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                            d_data, d_twiddles, n, half, stride);
                    }
                }
            } else if (leftover == 1) {
                uint32_t half = 1u << leftover_start;
                uint32_t stride = n / (2u * half);
                gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
            return;
        }
    }

fallback_r4:
    // Try radix-4
    if (num_outer >= 2) {
        int r4_max = gl_get_coop_r4();
        if (r4_max > 0) {
            int num_r4 = num_outer / 2;
            uint32_t needed = (n / 4 + GL_BLOCK - 1) / GL_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                                 ? static_cast<uint32_t>(r4_max) : needed;

            int r4_stage = outer_start;
            for (int done = 0; done < num_r4; ) {
                int passes = num_r4 - done;
                if (passes > GL_MAX_R4_PASSES) passes = GL_MAX_R4_PASSES;

                int ss = r4_stage;
                int np = passes;
                void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };
                cudaError_t err = cudaLaunchCooperativeKernel(
                    (void*)gl_ntt_outer_r4_kernel,
                    dim3(grid_size), dim3(GL_BLOCK), args, 0, stream);
                if (err != cudaSuccess) goto fallback_r2;

                done += passes;
                r4_stage += 2 * passes;
            }

            if (num_outer % 2 == 1) {
                int last = outer_end - 1;
                uint32_t half = 1u << last;
                uint32_t stride = n / (2u * half);
                gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
            return;
        }
    }

fallback_r2:
    {
        int r2_max = gl_get_coop_r2();
        if (r2_max > 0) {
            uint32_t needed = (n / 2 + GL_BLOCK - 1) / GL_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r2_max) < needed)
                                 ? static_cast<uint32_t>(r2_max) : needed;

            for (int s = outer_start; s < outer_end; ) {
                int batch_end = s + GL_MAX_R2_STAGES;
                if (batch_end > outer_end) batch_end = outer_end;

                int ss = s;
                int se = batch_end;
                void* args[] = { &d_data, &d_twiddles, &n, &ss, &se };
                cudaError_t err = cudaLaunchCooperativeKernel(
                    (void*)gl_ntt_outer_r2_kernel,
                    dim3(grid_size), dim3(GL_BLOCK), args, 0, stream);

                if (err != cudaSuccess) {
                    for (int ss2 = s; ss2 < outer_end; ++ss2) {
                        uint32_t half = 1u << ss2;
                        uint32_t stride = n / (2u * half);
                        gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                            d_data, d_twiddles, n, half, stride);
                    }
                    return;
                }
                s = batch_end;
            }
        } else {
            for (int s = outer_start; s < outer_end; ++s) {
                uint32_t half = 1u << s;
                uint32_t stride = n / (2u * half);
                gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
        }
    }
}

// ─── Batched Outer Stage Dispatch ────────────────────────────────────────────

static void gl_dispatch_outer_batch(
    GoldilocksElement* d_data, const GoldilocksElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int outer_end,
    cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;
    if (num_outer <= 0) return;

    uint32_t total_half = static_cast<uint32_t>(batch_size) * (n >> 1);

    // Try radix-8
    if (num_outer >= 3) {
        int r8_max = gl_get_coop_r8();
        // Use batch kernel occupancy — for simplicity reuse single kernel occupancy
        // (batch kernels have same register pressure)
        if (r8_max <= 0) r8_max = gl_get_coop_max((void*)gl_ntt_outer_r8_batch_kernel);
        if (r8_max > 0) {
            int num_r8 = num_outer / 3;
            uint32_t total_r8 = static_cast<uint32_t>(batch_size) * (n >> 3);
            uint32_t needed = (total_r8 + GL_BLOCK - 1) / GL_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                                 ? static_cast<uint32_t>(r8_max) : needed;

            int r8_stage = outer_start;
            for (int done = 0; done < num_r8; ) {
                int passes = num_r8 - done;
                if (passes > GL_MAX_R8_PASSES) passes = GL_MAX_R8_PASSES;

                int ss = r8_stage;
                int np = passes;
                void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &np };
                cudaLaunchCooperativeKernel(
                    (void*)gl_ntt_outer_r8_batch_kernel,
                    dim3(grid_size), dim3(GL_BLOCK), args, 0, stream);

                done += passes;
                r8_stage += 3 * passes;
            }

            int leftover = num_outer % 3;
            int leftover_start = outer_start + num_r8 * 3;

            if (leftover == 2) {
                int r4_max = gl_get_coop_max((void*)gl_ntt_outer_r4_batch_kernel);
                if (r4_max > 0) {
                    uint32_t total_r4 = static_cast<uint32_t>(batch_size) * (n >> 2);
                    uint32_t r4_needed = (total_r4 + GL_BLOCK - 1) / GL_BLOCK;
                    uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                                       ? static_cast<uint32_t>(r4_max) : r4_needed;
                    int ss = leftover_start;
                    int np = 1;
                    void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &np };
                    cudaLaunchCooperativeKernel(
                        (void*)gl_ntt_outer_r4_batch_kernel,
                        dim3(r4_grid), dim3(GL_BLOCK), args, 0, stream);
                } else {
                    uint32_t r2_needed = (total_half + GL_BLOCK - 1) / GL_BLOCK;
                    int ss = leftover_start;
                    int se = leftover_start + 2;
                    int r2_max = gl_get_coop_max((void*)gl_ntt_outer_r2_batch_kernel);
                    if (r2_max > 0) {
                        uint32_t r2_grid = (static_cast<uint32_t>(r2_max) < r2_needed)
                                           ? static_cast<uint32_t>(r2_max) : r2_needed;
                        void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &se };
                        cudaLaunchCooperativeKernel(
                            (void*)gl_ntt_outer_r2_batch_kernel,
                            dim3(r2_grid), dim3(GL_BLOCK), args, 0, stream);
                    }
                }
            } else if (leftover == 1) {
                int ss = leftover_start;
                int se = leftover_start + 1;
                int r2_max = gl_get_coop_max((void*)gl_ntt_outer_r2_batch_kernel);
                if (r2_max > 0) {
                    uint32_t r2_needed = (total_half + GL_BLOCK - 1) / GL_BLOCK;
                    uint32_t r2_grid = (static_cast<uint32_t>(r2_max) < r2_needed)
                                       ? static_cast<uint32_t>(r2_max) : r2_needed;
                    void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &se };
                    cudaLaunchCooperativeKernel(
                        (void*)gl_ntt_outer_r2_batch_kernel,
                        dim3(r2_grid), dim3(GL_BLOCK), args, 0, stream);
                }
            }
            return;
        }
    }

    // Fallback: radix-2 batch cooperative
    {
        int r2_max = gl_get_coop_max((void*)gl_ntt_outer_r2_batch_kernel);
        if (r2_max > 0) {
            uint32_t needed = (total_half + GL_BLOCK - 1) / GL_BLOCK;
            uint32_t grid_size = (static_cast<uint32_t>(r2_max) < needed)
                                 ? static_cast<uint32_t>(r2_max) : needed;

            for (int s = outer_start; s < outer_end; ) {
                int batch_end = s + GL_MAX_R2_STAGES;
                if (batch_end > outer_end) batch_end = outer_end;

                int ss = s;
                int se = batch_end;
                void* args[] = { &d_data, &d_twiddles, &n, &batch_size, &ss, &se };
                cudaLaunchCooperativeKernel(
                    (void*)gl_ntt_outer_r2_batch_kernel,
                    dim3(grid_size), dim3(GL_BLOCK), args, 0, stream);
                s = batch_end;
            }
        }
    }
}

// ─── Forward NTT ─────────────────────────────────────────────────────────────

static void gl_ntt_forward_impl(
    GoldilocksElement* d_data, size_t n,
    const GoldilocksElement* d_twiddles, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = gl_log2(n);
    const uint32_t grid      = (N + GL_BLOCK - 1) / GL_BLOCK;
    const uint32_t grid_half = (N / 2 + GL_BLOCK - 1) / GL_BLOCK;

    // Bit-reverse
    gl_ntt_bit_reverse_kernel<<<grid, GL_BLOCK, 0, stream>>>(d_data, N, log_n);

    // Fused inner stages
    const int fused_k = gl_select_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;
        gl_launch_fused(fused_k, d_data, d_twiddles, N, num_blocks, stream);

        // Outer stages
        gl_dispatch_outer(d_data, d_twiddles, N, fused_k, log_n, grid_half, stream);
    } else {
        // Small NTT: per-stage fallback
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    }
}

// ─── Inverse NTT ─────────────────────────────────────────────────────────────

static void gl_ntt_inverse_impl(
    GoldilocksElement* d_data, size_t n,
    const GoldilocksElement* d_inv_twiddles, GoldilocksElement n_inv,
    cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = gl_log2(n);
    const uint32_t grid      = (N + GL_BLOCK - 1) / GL_BLOCK;
    const uint32_t grid_half = (N / 2 + GL_BLOCK - 1) / GL_BLOCK;

    // Bit-reverse
    gl_ntt_bit_reverse_kernel<<<grid, GL_BLOCK, 0, stream>>>(d_data, N, log_n);

    // Fused inner stages
    const int fused_k = gl_select_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;
        gl_launch_fused(fused_k, d_data, d_inv_twiddles, N, num_blocks, stream);

        gl_dispatch_outer(d_data, d_inv_twiddles, N, fused_k, log_n, grid_half, stream);
    } else {
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            gl_ntt_butterfly_kernel<<<grid_half, GL_BLOCK, 0, stream>>>(
                d_data, d_inv_twiddles, N, half, stride);
        }
    }

    // Scale by n^{-1}
    gl_ntt_scale_kernel<<<grid, GL_BLOCK, 0, stream>>>(d_data, n_inv, N);
}

// ─── Public API ──────────────────────────────────────────────────────────────

void ntt_forward_goldilocks(GoldilocksElement* d_data, size_t n, cudaStream_t stream) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_gl_twiddles(n);
    gl_ntt_forward_impl(d_data, n, s_gl_fwd_twiddles, stream);
}

void ntt_inverse_goldilocks(GoldilocksElement* d_data, size_t n, cudaStream_t stream) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_gl_twiddles(n);
    gl_ntt_inverse_impl(d_data, n, s_gl_inv_twiddles, s_gl_n_inv, stream);
}

void ntt_forward_batch_goldilocks(GoldilocksElement* d_data, int batch_size, size_t n, cudaStream_t stream) {
    if (batch_size <= 0) return;
    if (batch_size == 1) { ntt_forward_goldilocks(d_data, n, stream); return; }
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_gl_twiddles(n);

    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = gl_log2(n);
    const uint32_t total = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total + GL_BLOCK - 1) / GL_BLOCK;

    // Batched bit-reverse
    gl_ntt_bit_reverse_batch_kernel<<<grid_total, GL_BLOCK, 0, stream>>>(d_data, N, log_n, total);

    // Fused inner: launch B * (N/ELEMS) blocks
    const int fused_k = gl_select_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);
        gl_launch_fused(fused_k, d_data, s_gl_fwd_twiddles, N, num_blocks, stream);

        // Batched outer stages
        gl_dispatch_outer_batch(d_data, s_gl_fwd_twiddles, N, batch_size, fused_k, log_n, stream);
    } else {
        // Small NTT fallback: just do them sequentially
        for (int b = 0; b < batch_size; ++b) {
            gl_ntt_forward_impl(d_data + b * N, n, s_gl_fwd_twiddles, stream);
        }
    }
}

void ntt_inverse_batch_goldilocks(GoldilocksElement* d_data, int batch_size, size_t n, cudaStream_t stream) {
    if (batch_size <= 0) return;
    if (batch_size == 1) { ntt_inverse_goldilocks(d_data, n, stream); return; }
    assert(n >= 2 && (n & (n - 1)) == 0);
    ensure_gl_twiddles(n);

    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = gl_log2(n);
    const uint32_t total = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total + GL_BLOCK - 1) / GL_BLOCK;

    // Batched bit-reverse
    gl_ntt_bit_reverse_batch_kernel<<<grid_total, GL_BLOCK, 0, stream>>>(d_data, N, log_n, total);

    // Fused inner
    const int fused_k = gl_select_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);
        gl_launch_fused(fused_k, d_data, s_gl_inv_twiddles, N, num_blocks, stream);

        gl_dispatch_outer_batch(d_data, s_gl_inv_twiddles, N, batch_size, fused_k, log_n, stream);
    } else {
        for (int b = 0; b < batch_size; ++b) {
            gl_ntt_inverse_impl(d_data + b * N, n, s_gl_inv_twiddles, s_gl_n_inv, stream);
        }
    }

    // Scale all elements by n^{-1}
    gl_ntt_scale_kernel<<<grid_total, GL_BLOCK, 0, stream>>>(d_data, s_gl_n_inv, total);
}
