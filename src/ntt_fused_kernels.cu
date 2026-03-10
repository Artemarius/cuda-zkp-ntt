// src/ntt_fused_kernels.cu
// Fused shared-memory + warp-shuffle butterfly kernels for NTT optimization.
// Compiled WITHOUT separable compilation (-rdc=false) to work around
// the MSVC + CUDA 12.8 RDC template symbol resolution bug (K>=9).
//
// All device functions (ff_mul, ff_add, ff_sub) are __device__ __forceinline__
// in ff_arithmetic.cuh, so they get inlined — no cross-TU device linkage needed.
//
// Provides explicit template instantiations for K=8, K=9, K=10 and
// host-callable launcher functions that ntt_optimized.cu can call.
//
// v1.1.0: Warp-shuffle optimization for stages 0-4 (stride <= 16 < warpSize).
// Stages 5..K-1 use shared memory + __syncthreads() (cross-warp).

#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// ─── Warp-Shuffle Helper ─────────────────────────────────────────────────────
// Shuffle a full FpElement (8 x uint32_t) across lanes using __shfl_xor_sync.
// All 8 limbs are shuffled independently with the same XOR mask.

__device__ __forceinline__
FpElement fp_shfl_xor(const FpElement& val, int xor_mask) {
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result.limbs[i] = __shfl_xor_sync(0xFFFFFFFF, val.limbs[i], xor_mask);
    }
    return result;
}

// ─── Warp-Shuffle Butterfly Helper ───────────────────────────────────────────
// Performs one butterfly stage on an element held in registers, using warp
// shuffle to exchange data with the partner thread.
//
// Thread mapping: thread t permanently owns element at index t (within its
// half of the block). At stage s, butterfly pairs differ in bit s:
//   partner_thread = t ^ (1 << s)
//   is_top = (t & (1 << s)) == 0
//
// Top thread holds A, partner (bottom) holds B. Both receive partner's value
// via shuffle, compute v = ff_mul(w, B), then:
//   top:    result = ff_add(A, v)
//   bottom: result = ff_sub(A, v)

__device__ __forceinline__
FpElement butterfly_shfl(const FpElement& my_elem, const FpElement& twiddle,
                         int xor_mask, bool is_top) {
    FpElement partner = fp_shfl_xor(my_elem, xor_mask);

    FpElement A = is_top ? my_elem : partner;
    FpElement B = is_top ? partner : my_elem;

    FpElement v = ff_mul(B, twiddle);

    return is_top ? ff_add(A, v) : ff_sub(A, v);
}

// ─── Fused Kernel: Warp-Shuffle + Shared-Memory ─────────────────────────────
// Fuses K radix-2 butterfly stages into a single kernel launch.
// Each block processes 2^K contiguous elements (2^(K-1) threads, 2 elems each).
//
// Stages 0-4 (stride 1..16): warp shuffle, no shared memory, no barriers.
//   __shfl_xor_sync provides intra-warp synchronization.
// Stages 5..K-1 (stride 32+): shared memory + __syncthreads() (cross-warp).
//
// Twiddle access: twiddles[j * stride] from global memory. The same twiddles
// are used by all blocks at the same stage, so L1/L2 cache is effective.

template <int K>
__global__ void ntt_fused_stages_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n
) {
    constexpr int ELEMS   = 1 << K;       // elements per block
    constexpr int THREADS = ELEMS >> 1;    // threads per block
    constexpr int SHFL_STAGES = (K < 5) ? K : 5;

    __shared__ FpElement sdata[ELEMS];

    const uint32_t boff = blockIdx.x * ELEMS;
    const uint32_t t = threadIdx.x;

    // ── Load from global memory into registers ──────────────────────────
    // Thread t owns element[t] (lo) and element[t+THREADS] (hi) permanently.
    FpElement reg_lo = data[boff + t];
    FpElement reg_hi = data[boff + t + THREADS];

    // ── Stages 0-4: warp-shuffle butterfly (stride 1..16, intra-warp) ───
    // No shared memory access, no __syncthreads().
    #pragma unroll
    for (int s = 0; s < SHFL_STAGES; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n >> (s + 1);
        const uint32_t j      = t & (half - 1);
        const bool is_top     = (t & half) == 0;
        const int xor_mask    = static_cast<int>(half);

        FpElement w = twiddles[j * stride];

        reg_lo = butterfly_shfl(reg_lo, w, xor_mask, is_top);
        reg_hi = butterfly_shfl(reg_hi, w, xor_mask, is_top);
    }

    // ── Stages 5..K-1: shared-memory butterfly (cross-warp) ────────────
    // Write registers to shared memory, then butterfly via shmem.
    // Uses original indexing: each thread handles one butterfly pair per stage.
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

        FpElement w = twiddles[j * stride];
        FpElement u = sdata[idx_top];
        FpElement v = ff_mul(sdata[idx_bot], w);
        sdata[idx_top] = ff_add(u, v);
        sdata[idx_bot] = ff_sub(u, v);
        __syncthreads();
    }

    // ── Store to global memory ──────────────────────────────────────────
    data[boff + t]           = sdata[t];
    data[boff + t + THREADS] = sdata[t + THREADS];
}

// ─── Explicit Template Instantiations ────────────────────────────────────────
// These force the compiler to generate the __global__ symbols in this TU,
// avoiding the RDC symbol lookup issue for K>=9.

template __global__ void ntt_fused_stages_kernel<8>(
    FpElement* __restrict__, const FpElement* __restrict__, uint32_t);
template __global__ void ntt_fused_stages_kernel<9>(
    FpElement* __restrict__, const FpElement* __restrict__, uint32_t);
template __global__ void ntt_fused_stages_kernel<10>(
    FpElement* __restrict__, const FpElement* __restrict__, uint32_t);

// ─── Host-Callable Launcher Functions ────────────────────────────────────────
// These are plain host functions (not templates) that ntt_optimized.cu can
// call without needing cross-TU device symbol resolution.

void launch_fused_k8(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream
) {
    constexpr int THREADS = 1 << (8 - 1);  // 128
    ntt_fused_stages_kernel<8>
        <<<num_blocks, THREADS, 0, stream>>>(d_data, d_twiddles, n);
}

void launch_fused_k9(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream
) {
    constexpr int THREADS = 1 << (9 - 1);  // 256
    ntt_fused_stages_kernel<9>
        <<<num_blocks, THREADS, 0, stream>>>(d_data, d_twiddles, n);
}

void launch_fused_k10(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream
) {
    constexpr int THREADS = 1 << (10 - 1);  // 512
    ntt_fused_stages_kernel<10>
        <<<num_blocks, THREADS, 0, stream>>>(d_data, d_twiddles, n);
}
