// src/ntt_optimized.cu
// Optimized shared-memory NTT: fuses 8 butterfly stages per kernel launch.
// Phase 5: reduces global memory round-trips vs naive (one launch per stage).
//
// Strategy (Cooley-Tukey, after bit-reverse):
//   1. Fused inner kernel: stages 0..7 in shared memory (1 launch, radix-256)
//   2. Outer stages: stages 8..log_n-1 via global-memory butterfly kernel
//
// For n=2^22: 1 fused + 14 outer = 15 launches (vs 22 naive).
// Each fused block processes 256 elements (8 KB shared memory, 128 threads).
// Measured speedup: ~14% at 2^22, ~16% at 2^18..2^20.

#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

#include <cstdio>
#include <cassert>

// ─── External Kernel Declarations (from ntt_naive.cu) ────────────────────────
// Linked via CUDA separable compilation within zkp_ntt_core.
// __restrict__ qualifiers must match definitions for correct MSVC mangling.

extern __global__ void ntt_bit_reverse_kernel(
    FpElement* data, uint32_t n, uint32_t log_n);

extern __global__ void ntt_butterfly_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n, uint32_t half, uint32_t stride);

extern __global__ void ntt_scale_kernel(
    FpElement* data, FpElement scalar, uint32_t n);

// ─── Fused Shared-Memory Butterfly Kernel ───────────────────────────────────
// Fuses 8 radix-2 butterfly stages into a single kernel launch.
// Each block processes 256 contiguous elements using shared memory.
// 128 threads: one butterfly pair per thread per stage.
//
// Within a single stage, each shared memory location is accessed by exactly
// one thread (butterfly pairs are disjoint), so no intra-stage races.
// __syncthreads() between stages ensures inter-stage visibility.
//
// Twiddle access: twiddles[j * stride] from global memory. The same twiddles
// are used by all blocks at the same stage, so L1/L2 cache is effective
// (only 255 unique twiddles across all 8 stages, ~8 KB total).

static constexpr int FUSED_K = 8;

template <int K>
__global__ void ntt_fused_stages_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n
) {
    constexpr int ELEMS   = 1 << K;       // 256 elements per block
    constexpr int THREADS = ELEMS >> 1;    // 128 threads per block

    __shared__ FpElement sdata[ELEMS];

    const uint32_t boff = blockIdx.x * ELEMS;
    const uint32_t t = threadIdx.x;

    // Coalesced load: each thread loads 2 elements
    sdata[t]           = data[boff + t];
    sdata[t + THREADS] = data[boff + t + THREADS];
    __syncthreads();

    // K butterfly stages in shared memory
    #pragma unroll
    for (int s = 0; s < K; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n >> (s + 1);   // N / 2^(s+1)

        const uint32_t group   = t >> s;         // t / half
        const uint32_t j       = t & (half - 1); // t % half
        const uint32_t idx_top = (group << (s + 1)) + j;
        const uint32_t idx_bot = idx_top + half;

        FpElement w = twiddles[j * stride];
        FpElement u = sdata[idx_top];
        FpElement v = ff_mul(sdata[idx_bot], w);
        sdata[idx_top] = ff_add(u, v);
        sdata[idx_bot] = ff_sub(u, v);
        __syncthreads();
    }

    // Coalesced store
    data[boff + t]           = sdata[t];
    data[boff + t + THREADS] = sdata[t + THREADS];
}

// ─── Configuration ──────────────────────────────────────────────────────────

static constexpr uint32_t OPT_BLOCK = 256;

static __host__ int log2_of(size_t n) {
    int r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// ─── Forward NTT (Montgomery domain, in-place) ─────────────────────────────

void ntt_forward_optimized_montgomery(
    FpElement* d_data, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t grid      = (N + OPT_BLOCK - 1) / OPT_BLOCK;
    const uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Bit-reverse permutation
    ntt_bit_reverse_kernel<<<grid, OPT_BLOCK, 0, stream>>>(d_data, N, log_n);

    // Step 2: Fused inner stages (0..7) in shared memory
    if (log_n >= FUSED_K) {
        constexpr int ELEMS   = 1 << FUSED_K;
        constexpr int THREADS = ELEMS >> 1;
        const uint32_t num_blocks = N / ELEMS;

        ntt_fused_stages_kernel<FUSED_K>
            <<<num_blocks, THREADS, 0, stream>>>(d_data, d_twiddles, N);

        // Step 3: Outer stages (8..log_n-1) via global butterfly kernel
        for (int s = FUSED_K; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    } else {
        // Small NTT (n < 256): naive per-stage fallback
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    }
}

// ─── Inverse NTT (Montgomery domain, in-place) ─────────────────────────────

void ntt_inverse_optimized_montgomery(
    FpElement* d_data, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t grid      = (N + OPT_BLOCK - 1) / OPT_BLOCK;
    const uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Bit-reverse permutation
    ntt_bit_reverse_kernel<<<grid, OPT_BLOCK, 0, stream>>>(d_data, N, log_n);

    // Step 2: Fused inner stages
    if (log_n >= FUSED_K) {
        constexpr int ELEMS   = 1 << FUSED_K;
        constexpr int THREADS = ELEMS >> 1;
        const uint32_t num_blocks = N / ELEMS;

        ntt_fused_stages_kernel<FUSED_K>
            <<<num_blocks, THREADS, 0, stream>>>(d_data, d_inv_twiddles, N);

        // Step 3: Outer stages
        for (int s = FUSED_K; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_inv_twiddles, N, half, stride);
        }
    } else {
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_inv_twiddles, N, half, stride);
        }
    }

    // Step 4: Scale by n^{-1}
    ntt_scale_kernel<<<grid, OPT_BLOCK, 0, stream>>>(d_data, n_inv, N);
}
