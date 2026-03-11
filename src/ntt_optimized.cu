// src/ntt_optimized.cu
// Optimized NTT: fused warp-shuffle + shared-memory inner kernel, cooperative
// groups outer kernel fusion. v1.1.0 combines all three optimization directions.
//
// Strategy (Cooley-Tukey, after bit-reverse):
//   1. Fused inner kernel: stages 0..K-1 (K=8/9/10 selected by NTT size)
//      - Stages 0-4: warp-level __shfl_xor_sync (no shared memory / barriers)
//      - Stages 5..K-1: shared memory + __syncthreads() (cross-warp)
//   2. Outer stages: stages K..log_n-1 via cooperative groups fused kernel
//      (multiple stages per launch with grid.sync() global barrier)
//
// K selection: K=10 for n >= 2^10, K=9 for n=2^9, K=8 for n=2^8.
//
// Launch count for n=2^22:
//   v1.0: 1 bit-reverse + 1 fused(K=8) + 14 outer = 16
//   v1.1: 1 bit-reverse + 1 fused(K=10) + 2 cooperative outer = 4

#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "ff_barrett.cuh"
#include "cuda_utils.cuh"

#include <cooperative_groups.h>
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

// ─── Fused Kernel Launchers (from ntt_fused_kernels.cu) ─────────────────────
// Compiled without RDC to avoid template symbol resolution bugs on MSVC+CUDA 12.8.
// Linked as plain host functions.

extern void launch_fused_k8(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream);
extern void launch_fused_k9(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream);
extern void launch_fused_k10(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream);

// Barrett fused launchers (from ntt_fused_kernels.cu)
extern void launch_fused_barrett_k8(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream);
extern void launch_fused_barrett_k9(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream);
extern void launch_fused_barrett_k10(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream);

// ─── Fused Multi-Stage Outer Kernel (Cooperative Groups) ─────────────────────
// Processes multiple consecutive outer butterfly stages in a single kernel launch.
// Uses cooperative groups grid.sync() for global barrier between stages.
// Launched with cudaLaunchCooperativeKernel; grid size limited to device occupancy.
// Each block + its threads process multiple butterfly pairs via a strided loop.

__global__ void ntt_outer_fused_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    int start_stage,
    int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_butterflies = n >> 1;  // n/2 butterfly pairs per stage

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n / (2u * half);

        // Strided loop: each thread handles multiple butterfly pairs
        for (uint32_t tid = global_tid; tid < num_butterflies; tid += total_threads) {
            uint32_t group   = tid / half;
            uint32_t j       = tid % half;
            uint32_t idx_top = group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;
            uint32_t tw_idx  = j * stride;

            FpElement u = data[idx_top];
            FpElement v = ff_mul(data[idx_bot], twiddles[tw_idx]);
            data[idx_top] = ff_add(u, v);
            data[idx_bot] = ff_sub(u, v);
        }

        // Global barrier between stages (all blocks must complete before next stage)
        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

static constexpr uint32_t OPT_BLOCK = 256;

// Max outer stages per cooperative launch. 7 stages per launch means
// 14 outer stages (K=8) -> 2 launches, 12 outer stages (K=10) -> 2 launches.
static constexpr int MAX_STAGES_PER_COOP_LAUNCH = 7;

static __host__ int log2_of(size_t n) {
    int r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// Select the best fused K for the given NTT size.
static __host__ int select_fused_k(int log_n) {
    if (log_n >= 10) return 10;
    if (log_n >= 9)  return 9;
    return 8;
}

// Launch the fused kernel for the selected K value.
static __host__ void launch_fused(
    int k, FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream
) {
    switch (k) {
        case 10: launch_fused_k10(d_data, d_twiddles, n, num_blocks, stream); break;
        case 9:  launch_fused_k9(d_data, d_twiddles, n, num_blocks, stream); break;
        default: launch_fused_k8(d_data, d_twiddles, n, num_blocks, stream); break;
    }
}

// ─── Cooperative Launch Helpers ──────────────────────────────────────────────

static int s_coop_max_blocks = 0;

static __host__ int get_coop_max_blocks() {
    if (s_coop_max_blocks > 0) return s_coop_max_blocks;

    int num_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_fused_kernel,
        OPT_BLOCK,
        0
    ));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks = num_blocks_per_sm * prop.multiProcessorCount;

    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "WARNING: Device does not support cooperative launch. "
                        "Falling back to per-stage kernels.\n");
        s_coop_max_blocks = 0;
    }

    return s_coop_max_blocks;
}

static __host__ bool launch_outer_fused(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int start_stage, int end_stage, cudaStream_t stream
) {
    int max_blocks = get_coop_max_blocks();
    if (max_blocks <= 0) return false;

    uint32_t needed_blocks = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(max_blocks) < needed_blocks)
                         ? static_cast<uint32_t>(max_blocks)
                         : needed_blocks;

    int ss = start_stage;
    int se = end_stage;
    void* args[] = {
        &d_data, &d_twiddles, &n, &ss, &se
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)ntt_outer_fused_kernel,
        dim3(grid_size), dim3(OPT_BLOCK),
        args, 0, stream
    );

    return (err == cudaSuccess);
}

// ─── Outer Stage Dispatch ───────────────────────────────────────────────────

static __host__ void dispatch_outer_stages(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t N, int outer_start, int outer_end,
    uint32_t grid_half, cudaStream_t stream
) {
    bool coop_ok = (get_coop_max_blocks() > 0);

    if (coop_ok) {
        for (int s = outer_start; s < outer_end; ) {
            int batch_end = s + MAX_STAGES_PER_COOP_LAUNCH;
            if (batch_end > outer_end) batch_end = outer_end;

            bool ok = launch_outer_fused(d_data, d_twiddles, N, s, batch_end, stream);

            if (!ok) {
                for (int ss = s; ss < outer_end; ++ss) {
                    uint32_t half   = 1u << ss;
                    uint32_t stride = N / (2u * half);
                    ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                        d_data, d_twiddles, N, half, stride);
                }
                break;
            }
            s = batch_end;
        }
    } else {
        for (int s = outer_start; s < outer_end; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    }
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

    // Step 2: Select and launch fused inner stages
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;

        launch_fused(fused_k, d_data, d_twiddles, N, num_blocks, stream);

        // Step 3: Outer stages via cooperative grouped launches
        dispatch_outer_stages(d_data, d_twiddles, N, fused_k, log_n, grid_half, stream);
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

    // Step 2: Select and launch fused inner stages
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;

        launch_fused(fused_k, d_data, d_inv_twiddles, N, num_blocks, stream);

        // Step 3: Outer stages via cooperative grouped launches
        dispatch_outer_stages(d_data, d_inv_twiddles, N, fused_k, log_n, grid_half, stream);
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

// ═════════════════════════════════════════════════════════════════════════════
// Barrett NTT Path — standard-form data, no Montgomery conversion
// ═════════════════════════════════════════════════════════════════════════════

// Barrett outer cooperative kernel (uses ff_mul_barrett)
__global__ void ntt_outer_fused_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    int start_stage,
    int end_stage
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
            uint32_t tw_idx  = j * stride;

            FpElement u = data[idx_top];
            FpElement v = ff_mul_barrett(data[idx_bot], twiddles[tw_idx]);
            data[idx_top] = ff_add(u, v);
            data[idx_bot] = ff_sub(u, v);
        }

        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// Barrett per-stage butterfly kernel (small-n fallback)
__global__ void ntt_butterfly_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t half,
    uint32_t stride
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    uint32_t group = tid / half;
    uint32_t j     = tid % half;

    uint32_t idx_top = group * (2 * half) + j;
    uint32_t idx_bot = idx_top + half;
    uint32_t tw_idx  = j * stride;

    FpElement u = data[idx_top];
    FpElement v = ff_mul_barrett(data[idx_bot], twiddles[tw_idx]);
    data[idx_top] = ff_add(u, v);
    data[idx_bot] = ff_sub(u, v);
}

// Barrett scale kernel (for n^{-1} in inverse NTT)
__global__ void ntt_scale_barrett_kernel(
    FpElement* data, FpElement scalar, uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = ff_mul_barrett(data[i], scalar);
}

// ─── Barrett Cooperative Launch Helpers ──────────────────────────────────────

static int s_coop_max_blocks_barrett = 0;

static __host__ int get_coop_max_blocks_barrett() {
    if (s_coop_max_blocks_barrett > 0) return s_coop_max_blocks_barrett;

    int num_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_fused_barrett_kernel,
        OPT_BLOCK,
        0
    ));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_barrett = num_blocks_per_sm * prop.multiProcessorCount;

    if (!prop.cooperativeLaunch) {
        s_coop_max_blocks_barrett = 0;
    }

    return s_coop_max_blocks_barrett;
}

static __host__ bool launch_outer_fused_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int start_stage, int end_stage, cudaStream_t stream
) {
    int max_blocks = get_coop_max_blocks_barrett();
    if (max_blocks <= 0) return false;

    uint32_t needed_blocks = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(max_blocks) < needed_blocks)
                         ? static_cast<uint32_t>(max_blocks)
                         : needed_blocks;

    int ss = start_stage;
    int se = end_stage;
    void* args[] = {
        &d_data, &d_twiddles, &n, &ss, &se
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)ntt_outer_fused_barrett_kernel,
        dim3(grid_size), dim3(OPT_BLOCK),
        args, 0, stream
    );

    return (err == cudaSuccess);
}

// Barrett fused launcher dispatch
static __host__ void launch_fused_barrett(
    int k, FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, uint32_t num_blocks, cudaStream_t stream
) {
    switch (k) {
        case 10: launch_fused_barrett_k10(d_data, d_twiddles, n, num_blocks, stream); break;
        case 9:  launch_fused_barrett_k9(d_data, d_twiddles, n, num_blocks, stream); break;
        default: launch_fused_barrett_k8(d_data, d_twiddles, n, num_blocks, stream); break;
    }
}

// Barrett outer stage dispatch
static __host__ void dispatch_outer_stages_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t N, int outer_start, int outer_end,
    uint32_t grid_half, cudaStream_t stream
) {
    bool coop_ok = (get_coop_max_blocks_barrett() > 0);

    if (coop_ok) {
        for (int s = outer_start; s < outer_end; ) {
            int batch_end = s + MAX_STAGES_PER_COOP_LAUNCH;
            if (batch_end > outer_end) batch_end = outer_end;

            bool ok = launch_outer_fused_barrett(d_data, d_twiddles, N, s, batch_end, stream);

            if (!ok) {
                for (int ss = s; ss < outer_end; ++ss) {
                    uint32_t half   = 1u << ss;
                    uint32_t stride = N / (2u * half);
                    ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                        d_data, d_twiddles, N, half, stride);
                }
                break;
            }
            s = batch_end;
        }
    } else {
        for (int s = outer_start; s < outer_end; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    }
}

// ─── Forward NTT (Barrett, standard-form, in-place) ─────────────────────────

void ntt_forward_optimized_barrett(
    FpElement* d_data, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t grid      = (N + OPT_BLOCK - 1) / OPT_BLOCK;
    const uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Bit-reverse permutation (domain-agnostic)
    ntt_bit_reverse_kernel<<<grid, OPT_BLOCK, 0, stream>>>(d_data, N, log_n);

    // Step 2: Fused inner stages (Barrett)
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;

        launch_fused_barrett(fused_k, d_data, d_twiddles, N, num_blocks, stream);

        // Step 3: Outer stages (Barrett)
        dispatch_outer_stages_barrett(d_data, d_twiddles, N, fused_k, log_n, grid_half, stream);
    } else {
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_twiddles, N, half, stride);
        }
    }
}

// ─── Inverse NTT (Barrett, standard-form, in-place) ─────────────────────────

void ntt_inverse_optimized_barrett(
    FpElement* d_data, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t grid      = (N + OPT_BLOCK - 1) / OPT_BLOCK;
    const uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Bit-reverse permutation
    ntt_bit_reverse_kernel<<<grid, OPT_BLOCK, 0, stream>>>(d_data, N, log_n);

    // Step 2: Fused inner stages (Barrett)
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = N / elems;

        launch_fused_barrett(fused_k, d_data, d_inv_twiddles, N, num_blocks, stream);

        // Step 3: Outer stages (Barrett)
        dispatch_outer_stages_barrett(d_data, d_inv_twiddles, N, fused_k, log_n, grid_half, stream);
    } else {
        for (int s = 0; s < log_n; ++s) {
            uint32_t half   = 1u << s;
            uint32_t stride = N / (2u * half);
            ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data, d_inv_twiddles, N, half, stride);
        }
    }

    // Step 4: Scale by n^{-1} using Barrett multiply
    ntt_scale_barrett_kernel<<<grid, OPT_BLOCK, 0, stream>>>(d_data, n_inv, N);
}
