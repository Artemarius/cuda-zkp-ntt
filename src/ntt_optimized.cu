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
#include "twiddle_otf.cuh"
#include "cuda_utils.cuh"

#include <cooperative_groups.h>
#include <cstdio>
#include <cassert>

// ─── OTF Twiddle Constant Memory ─────────────────────────────────────────────
// Per-stage roots of unity: c_otf_roots[s] = omega_n^(n / 2^(s+1))
// Uploaded before each NTT dispatch (forward or inverse, Montgomery or Barrett).
// Total: 32 × 32 bytes = 1 KB (vs 64 MB precomputed table at n=2^22).
__constant__ FpElement c_otf_roots[32];

// Fixed field constants for deriving radix-8/radix-4 twiddles:
// [0] = omega_4  (primitive 4th root of unity)
// [1] = omega_8  (primitive 8th root of unity)
// [2] = omega_8^3
__constant__ FpElement c_otf_consts[3];

// Host function to upload OTF roots and constants to __constant__ memory.
// Called from ntt_naive.cu before launching OTF outer-stage kernels.
void otf_upload_roots(const FpElement* roots, int num_roots, const FpElement* consts) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_otf_roots, roots, num_roots * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_otf_consts, consts, 3 * sizeof(FpElement)));
}

// ─── External Kernel Declarations (from ntt_naive.cu) ────────────────────────
// Linked via CUDA separable compilation within zkp_ntt_core.
// __restrict__ qualifiers must match definitions for correct MSVC mangling.

extern __global__ void ntt_bit_reverse_kernel(
    FpElement* data, uint32_t n, uint32_t log_n);

extern __global__ void ntt_bit_reverse_batch_kernel(
    FpElement* data, uint32_t n, uint32_t log_n, uint32_t total_elements);

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
            FpElement v = ff_mul_ptx(data[idx_bot], twiddles[tw_idx]);
            data[idx_top] = ff_add_v2(u, v);
            data[idx_bot] = ff_sub_v2(u, v);
        }

        // Global barrier between stages (all blocks must complete before next stage)
        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── Radix-4 Outer Kernel (Montgomery, Cooperative) ──────────────────────────
// Fuses pairs of consecutive outer stages into radix-4 butterflies.
// Each radix-4 unit processes 4 elements and 2 stages with 4 loads + 4 stores,
// halving DRAM passes compared to radix-2 (which needs 2× (4 loads + 4 stores)).
//
// Radix-4 butterfly for stages (s, s+1):
//   4 data elements at indices base+{0, half, 2*half, 3*half}
//   3 twiddle factors: w_s(j), w_{s+1}(j), w_{s+1}(j+half)
//   Stage s:   2 radix-2 butterflies on (a0,a1) and (a2,a3) with w_s(j)
//   Stage s+1: 2 radix-2 butterflies on (a0',a2') and (a1',a3')

__global__ void ntt_outer_radix4_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    int start_stage,
    int num_r4_passes
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

            // Load + stage s: two radix-2 butterflies with same twiddle
            FpElement a0 = data[base];
            FpElement a1 = data[base + half];
            FpElement w1 = twiddles[j * stride];

            FpElement t1 = ff_mul_ptx(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_ptx(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            // Stage s+1: two radix-2 butterflies with different twiddles
            FpElement w2 = twiddles[j * stride2];
            FpElement t3 = ff_mul_ptx(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement w3 = twiddles[(j + half) * stride2];
            FpElement t4 = ff_mul_ptx(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

static constexpr uint32_t OPT_BLOCK = 256;

// Max outer stages per cooperative launch. 7 stages per launch means
// 14 outer stages (K=8) -> 2 launches, 12 outer stages (K=10) -> 2 launches.
static constexpr int MAX_STAGES_PER_COOP_LAUNCH = 7;

// Max radix-4 passes per cooperative launch. Each pass fuses 2 stages.
// 7 passes = 14 stages, sufficient for n up to 2^24.
static constexpr int MAX_R4_PASSES_PER_COOP_LAUNCH = 7;

// Max radix-8 passes per cooperative launch. Each pass fuses 3 stages.
// 4 passes = 12 stages, sufficient for n up to 2^22.
static constexpr int MAX_R8_PASSES_PER_COOP_LAUNCH = 4;

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

// ─── Radix-4 Cooperative Launch Helpers (Montgomery) ─────────────────────────

static int s_coop_max_blocks_r4 = 0;

static __host__ int get_coop_max_blocks_r4() {
    if (s_coop_max_blocks_r4 != 0) return s_coop_max_blocks_r4;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix4_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();  // Clear error state
        s_coop_max_blocks_r4 = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r4 = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r4 == 0) {
        s_coop_max_blocks_r4 = -1;
    }
    return s_coop_max_blocks_r4;
}

// Try radix-4 dispatch. Returns true on success, false to fall back to radix-2.
static __host__ bool launch_outer_radix4(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_r4 = num_outer / 2;
    if (num_r4 == 0) return false;

    int r4_max = get_coop_max_blocks_r4();
    if (r4_max <= 0) return false;

    uint32_t needed = (n / 4 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                         ? static_cast<uint32_t>(r4_max) : needed;

    int r4_stage = outer_start;
    for (int done = 0; done < num_r4; ) {
        int passes = num_r4 - done;
        if (passes > MAX_R4_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R4_PASSES_PER_COOP_LAUNCH;

        int ss = r4_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix4_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r4_stage += 2 * passes;
    }

    // Leftover radix-2 stage (if odd number of outer stages)
    if (num_outer % 2 == 1) {
        int last = outer_start + num_outer - 1;
        uint32_t half   = 1u << last;
        uint32_t stride = n / (2u * half);
        ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
            d_data, d_twiddles, n, half, stride);
    }
    return true;
}

// ─── Forward Declarations: Radix-8 Launch Helpers ───────────────────────────
// Defined later in the file (after radix-8 kernels and occupancy queries).

static __host__ bool launch_outer_radix8(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream);

static __host__ bool launch_outer_radix8_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream);

static __host__ bool launch_outer_radix8_batch(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream);

static __host__ bool launch_outer_radix8_batch_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream);

// ─── Forward Declarations: OTF Launch Helpers ────────────────────────────────

static __host__ bool launch_outer_radix8_otf(
    FpElement* d_data, uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream);

static __host__ bool launch_outer_radix4_otf_barrett(
    FpElement* d_data, uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream);

static __host__ bool launch_outer_radix8_otf_batch(
    FpElement* d_data, uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream);

static __host__ bool launch_outer_radix4_otf_batch_barrett(
    FpElement* d_data, uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream);

// ─── Outer Stage Dispatch ───────────────────────────────────────────────────

static __host__ void dispatch_outer_stages(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t N, int outer_start, int outer_end,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;

    // NOTE: OTF (on-the-fly twiddle) disabled for BLS12-381.
    // ff_pow_mont_u32 does ~29 Montgomery muls per butterfly at large stages
    // (256-bit × 8-limb CIOS), far exceeding the DRAM savings from eliminating
    // twiddle table loads. Measured: 56.9ms OTF vs 15.6ms precomputed at n=2^22.
    // OTF infrastructure retained for future multi-field work (Goldilocks/BabyBear
    // where multiply is 1-2 instructions instead of ~128 MADs).

    // Precomputed radix-8
    if (num_outer >= 3 &&
        launch_outer_radix8(d_data, d_twiddles, N, outer_start, num_outer,
                            grid_half, stream))
    {
        return;
    }

    // Try radix-4 (halves DRAM passes for outer stages)
    if (num_outer >= 2 &&
        launch_outer_radix4(d_data, d_twiddles, N, outer_start, num_outer,
                            grid_half, stream))
    {
        return;
    }

    // Radix-2 cooperative fallback
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
            FpElement v = ff_mul_barrett_v2(data[idx_bot], twiddles[tw_idx]);
            data[idx_top] = ff_add_v2(u, v);
            data[idx_bot] = ff_sub_v2(u, v);
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
    FpElement v = ff_mul_barrett_v2(data[idx_bot], twiddles[tw_idx]);
    data[idx_top] = ff_add_v2(u, v);
    data[idx_bot] = ff_sub_v2(u, v);
}

// Barrett scale kernel (for n^{-1} in inverse NTT)
__global__ void ntt_scale_barrett_kernel(
    FpElement* data, FpElement scalar, uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = ff_mul_barrett_v2(data[i], scalar);
}

// ─── Radix-4 Outer Kernel (Barrett, Cooperative) ─────────────────────────────
// Same radix-4 structure as Montgomery variant, using ff_mul_barrett_v2.

__global__ void ntt_outer_radix4_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    int start_stage,
    int num_r4_passes
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

            FpElement a0 = data[base];
            FpElement a1 = data[base + half];
            FpElement w1 = twiddles[j * stride];

            FpElement t1 = ff_mul_barrett_v2(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_barrett_v2(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            FpElement w2 = twiddles[j * stride2];
            FpElement t3 = ff_mul_barrett_v2(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement w3 = twiddles[(j + half) * stride2];
            FpElement t4 = ff_mul_barrett_v2(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
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

// ─── Radix-4 Cooperative Launch Helpers (Barrett) ────────────────────────────

static int s_coop_max_blocks_r4_barrett = 0;

static __host__ int get_coop_max_blocks_r4_barrett() {
    if (s_coop_max_blocks_r4_barrett != 0) return s_coop_max_blocks_r4_barrett;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix4_barrett_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r4_barrett = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r4_barrett = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r4_barrett == 0) {
        s_coop_max_blocks_r4_barrett = -1;
    }
    return s_coop_max_blocks_r4_barrett;
}

static __host__ bool launch_outer_radix4_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_r4 = num_outer / 2;
    if (num_r4 == 0) return false;

    int r4_max = get_coop_max_blocks_r4_barrett();
    if (r4_max <= 0) return false;

    uint32_t needed = (n / 4 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                         ? static_cast<uint32_t>(r4_max) : needed;

    int r4_stage = outer_start;
    for (int done = 0; done < num_r4; ) {
        int passes = num_r4 - done;
        if (passes > MAX_R4_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R4_PASSES_PER_COOP_LAUNCH;

        int ss = r4_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix4_barrett_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r4_stage += 2 * passes;
    }

    if (num_outer % 2 == 1) {
        int last = outer_start + num_outer - 1;
        uint32_t half   = 1u << last;
        uint32_t stride = n / (2u * half);
        ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
            d_data, d_twiddles, n, half, stride);
    }
    return true;
}

// Barrett outer stage dispatch
static __host__ void dispatch_outer_stages_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t N, int outer_start, int outer_end,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;

    // NOTE: Barrett radix-8 disabled — 174 registers causes I-cache thrashing.
    // Barrett stays on radix-4.
    // NOTE: OTF disabled for BLS12-381 — exponentiation too expensive (see dispatch_outer_stages).

    // Precomputed radix-4
    if (num_outer >= 2 &&
        launch_outer_radix4_barrett(d_data, d_twiddles, N, outer_start, num_outer,
                                    grid_half, stream))
    {
        return;
    }

    // Radix-2 cooperative fallback
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

// ═════════════════════════════════════════════════════════════════════════════
// Batched NTT — process B independent NTTs in a single set of kernel launches
// ═════════════════════════════════════════════════════════════════════════════
//
// Data layout: contiguous, d_data[b*n .. (b+1)*n - 1] is NTT #b.
// Twiddle factors are shared across all NTTs (same size n).
//
// Key insight: the butterfly addressing formula (group*2*half + j, +half)
// naturally partitions by NTT boundaries for contiguous data because n/2
// is always a multiple of half = 2^s. Proof: n is power-of-2, s < log_n,
// so half <= n/2, and n/2 / half = n / 2^(s+1) is an integer.
//
// Fused kernel: existing kernel works unchanged — just launch with
// batch_size * (n/ELEMS) blocks. boff = blockIdx.x * ELEMS gives the
// correct offset into the contiguous batched array.
//
// Outer cooperative kernel: new kernel with total_butterflies parameter
// (= batch_size * n/2). Same butterfly formula, more iterations per thread.

// ─── Batched Outer Cooperative Kernel (Montgomery) ───────────────────────────

__global__ void ntt_outer_fused_batch_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t total_butterflies,
    int start_stage,
    int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n / (2u * half);

        for (uint32_t tid = global_tid; tid < total_butterflies; tid += total_threads) {
            uint32_t group   = tid / half;
            uint32_t j       = tid % half;
            uint32_t idx_top = group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;
            uint32_t tw_idx  = j * stride;

            FpElement u = data[idx_top];
            FpElement v = ff_mul_ptx(data[idx_bot], twiddles[tw_idx]);
            data[idx_top] = ff_add_v2(u, v);
            data[idx_bot] = ff_sub_v2(u, v);
        }

        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── Batched Outer Cooperative Kernel (Barrett) ──────────────────────────────

__global__ void ntt_outer_fused_batch_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t total_butterflies,
    int start_stage,
    int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half   = 1u << s;
        const uint32_t stride = n / (2u * half);

        for (uint32_t tid = global_tid; tid < total_butterflies; tid += total_threads) {
            uint32_t group   = tid / half;
            uint32_t j       = tid % half;
            uint32_t idx_top = group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;
            uint32_t tw_idx  = j * stride;

            FpElement u = data[idx_top];
            FpElement v = ff_mul_barrett_v2(data[idx_bot], twiddles[tw_idx]);
            data[idx_top] = ff_add_v2(u, v);
            data[idx_bot] = ff_sub_v2(u, v);
        }

        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── Batched Radix-4 Outer Kernel (Montgomery, Cooperative) ──────────────────

__global__ void ntt_outer_radix4_batch_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t total_r4_butterflies,
    int start_stage,
    int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;

        for (uint32_t tid = global_tid; tid < total_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            FpElement a0 = data[base];
            FpElement a1 = data[base + half];
            FpElement w1 = twiddles[j * stride];

            FpElement t1 = ff_mul_ptx(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_ptx(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            FpElement w2 = twiddles[j * stride2];
            FpElement t3 = ff_mul_ptx(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement w3 = twiddles[(j + half) * stride2];
            FpElement t4 = ff_mul_ptx(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── Batched Radix-4 Outer Kernel (Barrett, Cooperative) ─────────────────────

__global__ void ntt_outer_radix4_batch_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t total_r4_butterflies,
    int start_stage,
    int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half    = 1u << s;
        const uint32_t stride  = n >> (s + 1);
        const uint32_t stride2 = stride >> 1;

        for (uint32_t tid = global_tid; tid < total_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            FpElement a0 = data[base];
            FpElement a1 = data[base + half];
            FpElement w1 = twiddles[j * stride];

            FpElement t1 = ff_mul_barrett_v2(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_barrett_v2(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            FpElement w2 = twiddles[j * stride2];
            FpElement t3 = ff_mul_barrett_v2(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement w3 = twiddles[(j + half) * stride2];
            FpElement t4 = ff_mul_barrett_v2(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── Radix-8 Outer Kernel (Montgomery, Cooperative) ──────────────────────────
// Fuses triples of consecutive outer stages into radix-8 butterflies.
// Each radix-8 unit processes 8 elements and 3 stages with 8 loads + 8 stores,
// reducing DRAM passes to 1/3 compared to radix-2 (which needs 3× (4 loads + 4 stores)).
//
// Radix-8 butterfly for stages (s, s+1, s+2):
//   8 data elements at indices base+{0, h, 2h, 3h, 4h, 5h, 6h, 7h}, h = 2^s
//   Stage s:   4 radix-2 butterflies on (a0,a4),(a1,a5),(a2,a6),(a3,a7) with w_s(j)
//   Stage s+1: 4 radix-2 butterflies on (a0,a2),(a1,a3),(a4,a6),(a5,a7)
//   Stage s+2: 4 radix-2 butterflies on (a0,a4),(a1,a5),(a2,a6),(a3,a7) [re-paired]

__global__ void ntt_outer_radix8_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    int start_stage,
    int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r8_butterflies = n >> 3;  // n/8

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t h = 1u << s;
        const uint32_t stride_s  = n >> (s + 1);
        const uint32_t stride_s1 = stride_s >> 1;
        const uint32_t stride_s2 = stride_s >> 2;

        for (uint32_t tid = global_tid; tid < num_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (h - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            // Load 8 elements
            FpElement a0 = data[base];
            FpElement a1 = data[base + h];
            FpElement a2 = data[base + (h << 1)];
            FpElement a3 = data[base + (h << 1) + h];
            FpElement a4 = data[base + (h << 2)];
            FpElement a5 = data[base + (h << 2) + h];
            FpElement a6 = data[base + (h << 2) + (h << 1)];
            FpElement a7 = data[base + (h << 2) + (h << 1) + h];

            // Stage s: 4 radix-2 butterflies with w1 — pairs h apart
            FpElement w1 = twiddles[j * stride_s];
            {
                FpElement t;
                t = ff_mul_ptx(a1, w1); { FpElement tmp = ff_add_v2(a0, t); a1 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w1); { FpElement tmp = ff_add_v2(a2, t); a3 = ff_sub_v2(a2, t); a2 = tmp; }
                t = ff_mul_ptx(a5, w1); { FpElement tmp = ff_add_v2(a4, t); a5 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w1); { FpElement tmp = ff_add_v2(a6, t); a7 = ff_sub_v2(a6, t); a6 = tmp; }
            }

            // Stage s+1: 4 radix-2 butterflies — pairs 2h apart
            FpElement w2 = twiddles[j * stride_s1];
            FpElement w3 = twiddles[(j + h) * stride_s1];
            {
                FpElement t;
                t = ff_mul_ptx(a2, w2); { FpElement tmp = ff_add_v2(a0, t); a2 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w3); { FpElement tmp = ff_add_v2(a1, t); a3 = ff_sub_v2(a1, t); a1 = tmp; }
                t = ff_mul_ptx(a6, w2); { FpElement tmp = ff_add_v2(a4, t); a6 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w3); { FpElement tmp = ff_add_v2(a5, t); a7 = ff_sub_v2(a5, t); a5 = tmp; }
            }

            // Stage s+2: 4 radix-2 butterflies — pairs 4h apart
            FpElement w4 = twiddles[j * stride_s2];
            FpElement w5 = twiddles[(j + h) * stride_s2];
            FpElement w6 = twiddles[(j + (h << 1)) * stride_s2];
            FpElement w7 = twiddles[(j + (h << 1) + h) * stride_s2];
            {
                FpElement t;
                t = ff_mul_ptx(a4, w4); data[base]                                   = ff_add_v2(a0, t); data[base + (h << 2)]                    = ff_sub_v2(a0, t);
                t = ff_mul_ptx(a5, w5); data[base + h]                               = ff_add_v2(a1, t); data[base + (h << 2) + h]                = ff_sub_v2(a1, t);
                t = ff_mul_ptx(a6, w6); data[base + (h << 1)]                        = ff_add_v2(a2, t); data[base + (h << 2) + (h << 1)]         = ff_sub_v2(a2, t);
                t = ff_mul_ptx(a7, w7); data[base + (h << 1) + h]                    = ff_add_v2(a3, t); data[base + (h << 2) + (h << 1) + h]     = ff_sub_v2(a3, t);
            }
        }

        if (pass + 1 < num_r8_passes) {
            grid.sync();
        }
    }
}

// ─── Radix-8 Outer Kernel (Barrett, Cooperative) ─────────────────────────────

__global__ void ntt_outer_radix8_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    int start_stage,
    int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r8_butterflies = n >> 3;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t h = 1u << s;
        const uint32_t stride_s  = n >> (s + 1);
        const uint32_t stride_s1 = stride_s >> 1;
        const uint32_t stride_s2 = stride_s >> 2;

        for (uint32_t tid = global_tid; tid < num_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (h - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            FpElement a0 = data[base];
            FpElement a1 = data[base + h];
            FpElement a2 = data[base + (h << 1)];
            FpElement a3 = data[base + (h << 1) + h];
            FpElement a4 = data[base + (h << 2)];
            FpElement a5 = data[base + (h << 2) + h];
            FpElement a6 = data[base + (h << 2) + (h << 1)];
            FpElement a7 = data[base + (h << 2) + (h << 1) + h];

            // Stage s
            FpElement w1 = twiddles[j * stride_s];
            {
                FpElement t;
                t = ff_mul_barrett_v2(a1, w1); { FpElement tmp = ff_add_v2(a0, t); a1 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_barrett_v2(a3, w1); { FpElement tmp = ff_add_v2(a2, t); a3 = ff_sub_v2(a2, t); a2 = tmp; }
                t = ff_mul_barrett_v2(a5, w1); { FpElement tmp = ff_add_v2(a4, t); a5 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_barrett_v2(a7, w1); { FpElement tmp = ff_add_v2(a6, t); a7 = ff_sub_v2(a6, t); a6 = tmp; }
            }

            // Stage s+1
            FpElement w2 = twiddles[j * stride_s1];
            FpElement w3 = twiddles[(j + h) * stride_s1];
            {
                FpElement t;
                t = ff_mul_barrett_v2(a2, w2); { FpElement tmp = ff_add_v2(a0, t); a2 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_barrett_v2(a3, w3); { FpElement tmp = ff_add_v2(a1, t); a3 = ff_sub_v2(a1, t); a1 = tmp; }
                t = ff_mul_barrett_v2(a6, w2); { FpElement tmp = ff_add_v2(a4, t); a6 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_barrett_v2(a7, w3); { FpElement tmp = ff_add_v2(a5, t); a7 = ff_sub_v2(a5, t); a5 = tmp; }
            }

            // Stage s+2
            FpElement w4 = twiddles[j * stride_s2];
            FpElement w5 = twiddles[(j + h) * stride_s2];
            FpElement w6 = twiddles[(j + (h << 1)) * stride_s2];
            FpElement w7 = twiddles[(j + (h << 1) + h) * stride_s2];
            {
                FpElement t;
                t = ff_mul_barrett_v2(a4, w4); data[base]                                   = ff_add_v2(a0, t); data[base + (h << 2)]                    = ff_sub_v2(a0, t);
                t = ff_mul_barrett_v2(a5, w5); data[base + h]                               = ff_add_v2(a1, t); data[base + (h << 2) + h]                = ff_sub_v2(a1, t);
                t = ff_mul_barrett_v2(a6, w6); data[base + (h << 1)]                        = ff_add_v2(a2, t); data[base + (h << 2) + (h << 1)]         = ff_sub_v2(a2, t);
                t = ff_mul_barrett_v2(a7, w7); data[base + (h << 1) + h]                    = ff_add_v2(a3, t); data[base + (h << 2) + (h << 1) + h]     = ff_sub_v2(a3, t);
            }
        }

        if (pass + 1 < num_r8_passes) {
            grid.sync();
        }
    }
}

// ─── Batched Radix-8 Outer Kernel (Montgomery, Cooperative) ──────────────────

__global__ void ntt_outer_radix8_batch_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t total_r8_butterflies,
    int start_stage,
    int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t h = 1u << s;
        const uint32_t stride_s  = n >> (s + 1);
        const uint32_t stride_s1 = stride_s >> 1;
        const uint32_t stride_s2 = stride_s >> 2;

        for (uint32_t tid = global_tid; tid < total_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (h - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            FpElement a0 = data[base];
            FpElement a1 = data[base + h];
            FpElement a2 = data[base + (h << 1)];
            FpElement a3 = data[base + (h << 1) + h];
            FpElement a4 = data[base + (h << 2)];
            FpElement a5 = data[base + (h << 2) + h];
            FpElement a6 = data[base + (h << 2) + (h << 1)];
            FpElement a7 = data[base + (h << 2) + (h << 1) + h];

            // Stage s
            FpElement w1 = twiddles[j * stride_s];
            {
                FpElement t;
                t = ff_mul_ptx(a1, w1); { FpElement tmp = ff_add_v2(a0, t); a1 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w1); { FpElement tmp = ff_add_v2(a2, t); a3 = ff_sub_v2(a2, t); a2 = tmp; }
                t = ff_mul_ptx(a5, w1); { FpElement tmp = ff_add_v2(a4, t); a5 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w1); { FpElement tmp = ff_add_v2(a6, t); a7 = ff_sub_v2(a6, t); a6 = tmp; }
            }

            // Stage s+1
            FpElement w2 = twiddles[j * stride_s1];
            FpElement w3 = twiddles[(j + h) * stride_s1];
            {
                FpElement t;
                t = ff_mul_ptx(a2, w2); { FpElement tmp = ff_add_v2(a0, t); a2 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w3); { FpElement tmp = ff_add_v2(a1, t); a3 = ff_sub_v2(a1, t); a1 = tmp; }
                t = ff_mul_ptx(a6, w2); { FpElement tmp = ff_add_v2(a4, t); a6 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w3); { FpElement tmp = ff_add_v2(a5, t); a7 = ff_sub_v2(a5, t); a5 = tmp; }
            }

            // Stage s+2
            FpElement w4 = twiddles[j * stride_s2];
            FpElement w5 = twiddles[(j + h) * stride_s2];
            FpElement w6 = twiddles[(j + (h << 1)) * stride_s2];
            FpElement w7 = twiddles[(j + (h << 1) + h) * stride_s2];
            {
                FpElement t;
                t = ff_mul_ptx(a4, w4); data[base]                                   = ff_add_v2(a0, t); data[base + (h << 2)]                    = ff_sub_v2(a0, t);
                t = ff_mul_ptx(a5, w5); data[base + h]                               = ff_add_v2(a1, t); data[base + (h << 2) + h]                = ff_sub_v2(a1, t);
                t = ff_mul_ptx(a6, w6); data[base + (h << 1)]                        = ff_add_v2(a2, t); data[base + (h << 2) + (h << 1)]         = ff_sub_v2(a2, t);
                t = ff_mul_ptx(a7, w7); data[base + (h << 1) + h]                    = ff_add_v2(a3, t); data[base + (h << 2) + (h << 1) + h]     = ff_sub_v2(a3, t);
            }
        }

        if (pass + 1 < num_r8_passes) {
            grid.sync();
        }
    }
}

// ─── Batched Radix-8 Outer Kernel (Barrett, Cooperative) ─────────────────────

__global__ void ntt_outer_radix8_batch_barrett_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t total_r8_butterflies,
    int start_stage,
    int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t h = 1u << s;
        const uint32_t stride_s  = n >> (s + 1);
        const uint32_t stride_s1 = stride_s >> 1;
        const uint32_t stride_s2 = stride_s >> 2;

        for (uint32_t tid = global_tid; tid < total_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (h - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            FpElement a0 = data[base];
            FpElement a1 = data[base + h];
            FpElement a2 = data[base + (h << 1)];
            FpElement a3 = data[base + (h << 1) + h];
            FpElement a4 = data[base + (h << 2)];
            FpElement a5 = data[base + (h << 2) + h];
            FpElement a6 = data[base + (h << 2) + (h << 1)];
            FpElement a7 = data[base + (h << 2) + (h << 1) + h];

            // Stage s
            FpElement w1 = twiddles[j * stride_s];
            {
                FpElement t;
                t = ff_mul_barrett_v2(a1, w1); { FpElement tmp = ff_add_v2(a0, t); a1 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_barrett_v2(a3, w1); { FpElement tmp = ff_add_v2(a2, t); a3 = ff_sub_v2(a2, t); a2 = tmp; }
                t = ff_mul_barrett_v2(a5, w1); { FpElement tmp = ff_add_v2(a4, t); a5 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_barrett_v2(a7, w1); { FpElement tmp = ff_add_v2(a6, t); a7 = ff_sub_v2(a6, t); a6 = tmp; }
            }

            // Stage s+1
            FpElement w2 = twiddles[j * stride_s1];
            FpElement w3 = twiddles[(j + h) * stride_s1];
            {
                FpElement t;
                t = ff_mul_barrett_v2(a2, w2); { FpElement tmp = ff_add_v2(a0, t); a2 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_barrett_v2(a3, w3); { FpElement tmp = ff_add_v2(a1, t); a3 = ff_sub_v2(a1, t); a1 = tmp; }
                t = ff_mul_barrett_v2(a6, w2); { FpElement tmp = ff_add_v2(a4, t); a6 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_barrett_v2(a7, w3); { FpElement tmp = ff_add_v2(a5, t); a7 = ff_sub_v2(a5, t); a5 = tmp; }
            }

            // Stage s+2
            FpElement w4 = twiddles[j * stride_s2];
            FpElement w5 = twiddles[(j + h) * stride_s2];
            FpElement w6 = twiddles[(j + (h << 1)) * stride_s2];
            FpElement w7 = twiddles[(j + (h << 1) + h) * stride_s2];
            {
                FpElement t;
                t = ff_mul_barrett_v2(a4, w4); data[base]                                   = ff_add_v2(a0, t); data[base + (h << 2)]                    = ff_sub_v2(a0, t);
                t = ff_mul_barrett_v2(a5, w5); data[base + h]                               = ff_add_v2(a1, t); data[base + (h << 2) + h]                = ff_sub_v2(a1, t);
                t = ff_mul_barrett_v2(a6, w6); data[base + (h << 1)]                        = ff_add_v2(a2, t); data[base + (h << 2) + (h << 1)]         = ff_sub_v2(a2, t);
                t = ff_mul_barrett_v2(a7, w7); data[base + (h << 1) + h]                    = ff_add_v2(a3, t); data[base + (h << 2) + (h << 1) + h]     = ff_sub_v2(a3, t);
            }
        }

        if (pass + 1 < num_r8_passes) {
            grid.sync();
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// OTF (On-the-Fly) Twiddle Outer Kernels
// ═════════════════════════════════════════════════════════════════════════════
// Replace precomputed twiddle table loads with on-the-fly computation from
// per-stage roots in __constant__ memory. Eliminates twiddle DRAM traffic.
//
// Twiddle derivation for radix-8 butterfly at stages (s, s+1, s+2):
//   w4 = c_otf_roots[s+2]^j              (exponentiation, ~s iterations)
//   w2 = w4^2                              (1 ff_mul — since root[s+1] = root[s+2]^2)
//   w1 = w2^2                              (1 ff_mul)
//   w3 = w2 * c_otf_consts[0]             (omega_4: g^((r-1)/4))
//   w5 = w4 * c_otf_consts[1]             (omega_8: g^((r-1)/8))
//   w6 = w4 * c_otf_consts[0]             (omega_4)
//   w7 = w4 * c_otf_consts[2]             (omega_8^3)

// ─── OTF Radix-8 Outer Kernel (Montgomery, Single, Cooperative) ─────────────

__global__ void ntt_outer_radix8_otf_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    int start_stage,
    int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r8_butterflies = n >> 3;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t h = 1u << s;

        for (uint32_t tid = global_tid; tid < num_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (h - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            // Load 8 elements
            FpElement a0 = data[base];
            FpElement a1 = data[base + h];
            FpElement a2 = data[base + (h << 1)];
            FpElement a3 = data[base + (h << 1) + h];
            FpElement a4 = data[base + (h << 2)];
            FpElement a5 = data[base + (h << 2) + h];
            FpElement a6 = data[base + (h << 2) + (h << 1)];
            FpElement a7 = data[base + (h << 2) + (h << 1) + h];

            // OTF twiddle computation: derive all 7 from single exponentiation
            FpElement w4 = ff_pow_mont_u32(c_otf_roots[s + 2], j);
            FpElement w2 = ff_mul_ptx(w4, w4);
            FpElement w1 = ff_mul_ptx(w2, w2);
            FpElement w3 = ff_mul_ptx(w2, c_otf_consts[0]);
            FpElement w5 = ff_mul_ptx(w4, c_otf_consts[1]);
            FpElement w6 = ff_mul_ptx(w4, c_otf_consts[0]);
            FpElement w7 = ff_mul_ptx(w4, c_otf_consts[2]);

            // Stage s: 4 radix-2 butterflies with w1
            {
                FpElement t;
                t = ff_mul_ptx(a1, w1); { FpElement tmp = ff_add_v2(a0, t); a1 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w1); { FpElement tmp = ff_add_v2(a2, t); a3 = ff_sub_v2(a2, t); a2 = tmp; }
                t = ff_mul_ptx(a5, w1); { FpElement tmp = ff_add_v2(a4, t); a5 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w1); { FpElement tmp = ff_add_v2(a6, t); a7 = ff_sub_v2(a6, t); a6 = tmp; }
            }

            // Stage s+1: 4 radix-2 butterflies
            {
                FpElement t;
                t = ff_mul_ptx(a2, w2); { FpElement tmp = ff_add_v2(a0, t); a2 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w3); { FpElement tmp = ff_add_v2(a1, t); a3 = ff_sub_v2(a1, t); a1 = tmp; }
                t = ff_mul_ptx(a6, w2); { FpElement tmp = ff_add_v2(a4, t); a6 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w3); { FpElement tmp = ff_add_v2(a5, t); a7 = ff_sub_v2(a5, t); a5 = tmp; }
            }

            // Stage s+2: 4 radix-2 butterflies — write results to global memory
            {
                FpElement t;
                t = ff_mul_ptx(a4, w4); data[base]                                   = ff_add_v2(a0, t); data[base + (h << 2)]                    = ff_sub_v2(a0, t);
                t = ff_mul_ptx(a5, w5); data[base + h]                               = ff_add_v2(a1, t); data[base + (h << 2) + h]                = ff_sub_v2(a1, t);
                t = ff_mul_ptx(a6, w6); data[base + (h << 1)]                        = ff_add_v2(a2, t); data[base + (h << 2) + (h << 1)]         = ff_sub_v2(a2, t);
                t = ff_mul_ptx(a7, w7); data[base + (h << 1) + h]                    = ff_add_v2(a3, t); data[base + (h << 2) + (h << 1) + h]     = ff_sub_v2(a3, t);
            }
        }

        if (pass + 1 < num_r8_passes) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-4 Outer Kernel (Barrett, Single, Cooperative) ────────────────

__global__ void ntt_outer_radix4_otf_barrett_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    int start_stage,
    int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r4_butterflies = n >> 2;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half = 1u << s;

        for (uint32_t tid = global_tid; tid < num_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            // OTF twiddle computation: derive 3 twiddles from single exponentiation
            FpElement w2 = ff_pow_barrett_u32(c_otf_roots[s + 1], j);
            FpElement w1 = ff_mul_barrett_v2(w2, w2);
            FpElement w3 = ff_mul_barrett_v2(w2, c_otf_consts[0]);

            // Stage s: two radix-2 butterflies with w1
            FpElement a0 = data[base];
            FpElement a1 = data[base + half];

            FpElement t1 = ff_mul_barrett_v2(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_barrett_v2(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            // Stage s+1: two radix-2 butterflies with w2, w3
            FpElement t3 = ff_mul_barrett_v2(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement t4 = ff_mul_barrett_v2(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-4 Outer Kernel (Montgomery, Single, Cooperative) ─────────────

__global__ void ntt_outer_radix4_otf_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    int start_stage,
    int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t num_r4_butterflies = n >> 2;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half = 1u << s;

        for (uint32_t tid = global_tid; tid < num_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            // OTF twiddle computation
            FpElement w2 = ff_pow_mont_u32(c_otf_roots[s + 1], j);
            FpElement w1 = ff_mul_ptx(w2, w2);
            FpElement w3 = ff_mul_ptx(w2, c_otf_consts[0]);

            FpElement a0 = data[base];
            FpElement a1 = data[base + half];

            FpElement t1 = ff_mul_ptx(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_ptx(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            FpElement t3 = ff_mul_ptx(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement t4 = ff_mul_ptx(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-8 Outer Kernel (Montgomery, Batched, Cooperative) ────────────

__global__ void ntt_outer_radix8_otf_batch_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    uint32_t total_r8_butterflies,
    int start_stage,
    int num_r8_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r8_passes; ++pass) {
        const int s = start_stage + 3 * pass;
        const uint32_t h = 1u << s;

        for (uint32_t tid = global_tid; tid < total_r8_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (h - 1);
            const uint32_t base  = (group << (s + 3)) + j;

            FpElement a0 = data[base];
            FpElement a1 = data[base + h];
            FpElement a2 = data[base + (h << 1)];
            FpElement a3 = data[base + (h << 1) + h];
            FpElement a4 = data[base + (h << 2)];
            FpElement a5 = data[base + (h << 2) + h];
            FpElement a6 = data[base + (h << 2) + (h << 1)];
            FpElement a7 = data[base + (h << 2) + (h << 1) + h];

            // OTF twiddle computation
            FpElement w4 = ff_pow_mont_u32(c_otf_roots[s + 2], j);
            FpElement w2 = ff_mul_ptx(w4, w4);
            FpElement w1 = ff_mul_ptx(w2, w2);
            FpElement w3 = ff_mul_ptx(w2, c_otf_consts[0]);
            FpElement w5 = ff_mul_ptx(w4, c_otf_consts[1]);
            FpElement w6 = ff_mul_ptx(w4, c_otf_consts[0]);
            FpElement w7 = ff_mul_ptx(w4, c_otf_consts[2]);

            // Stage s
            {
                FpElement t;
                t = ff_mul_ptx(a1, w1); { FpElement tmp = ff_add_v2(a0, t); a1 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w1); { FpElement tmp = ff_add_v2(a2, t); a3 = ff_sub_v2(a2, t); a2 = tmp; }
                t = ff_mul_ptx(a5, w1); { FpElement tmp = ff_add_v2(a4, t); a5 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w1); { FpElement tmp = ff_add_v2(a6, t); a7 = ff_sub_v2(a6, t); a6 = tmp; }
            }

            // Stage s+1
            {
                FpElement t;
                t = ff_mul_ptx(a2, w2); { FpElement tmp = ff_add_v2(a0, t); a2 = ff_sub_v2(a0, t); a0 = tmp; }
                t = ff_mul_ptx(a3, w3); { FpElement tmp = ff_add_v2(a1, t); a3 = ff_sub_v2(a1, t); a1 = tmp; }
                t = ff_mul_ptx(a6, w2); { FpElement tmp = ff_add_v2(a4, t); a6 = ff_sub_v2(a4, t); a4 = tmp; }
                t = ff_mul_ptx(a7, w3); { FpElement tmp = ff_add_v2(a5, t); a7 = ff_sub_v2(a5, t); a5 = tmp; }
            }

            // Stage s+2
            {
                FpElement t;
                t = ff_mul_ptx(a4, w4); data[base]                                   = ff_add_v2(a0, t); data[base + (h << 2)]                    = ff_sub_v2(a0, t);
                t = ff_mul_ptx(a5, w5); data[base + h]                               = ff_add_v2(a1, t); data[base + (h << 2) + h]                = ff_sub_v2(a1, t);
                t = ff_mul_ptx(a6, w6); data[base + (h << 1)]                        = ff_add_v2(a2, t); data[base + (h << 2) + (h << 1)]         = ff_sub_v2(a2, t);
                t = ff_mul_ptx(a7, w7); data[base + (h << 1) + h]                    = ff_add_v2(a3, t); data[base + (h << 2) + (h << 1) + h]     = ff_sub_v2(a3, t);
            }
        }

        if (pass + 1 < num_r8_passes) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-4 Outer Kernel (Barrett, Batched, Cooperative) ───────────────

__global__ void ntt_outer_radix4_otf_batch_barrett_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    uint32_t total_r4_butterflies,
    int start_stage,
    int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half = 1u << s;

        for (uint32_t tid = global_tid; tid < total_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            FpElement w2 = ff_pow_barrett_u32(c_otf_roots[s + 1], j);
            FpElement w1 = ff_mul_barrett_v2(w2, w2);
            FpElement w3 = ff_mul_barrett_v2(w2, c_otf_consts[0]);

            FpElement a0 = data[base];
            FpElement a1 = data[base + half];

            FpElement t1 = ff_mul_barrett_v2(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_barrett_v2(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            FpElement t3 = ff_mul_barrett_v2(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement t4 = ff_mul_barrett_v2(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-4 Outer Kernel (Montgomery, Batched, Cooperative) ────────────

__global__ void ntt_outer_radix4_otf_batch_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    uint32_t total_r4_butterflies,
    int start_stage,
    int num_r4_passes
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int pass = 0; pass < num_r4_passes; ++pass) {
        const int s = start_stage + 2 * pass;
        const uint32_t half = 1u << s;

        for (uint32_t tid = global_tid; tid < total_r4_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t base  = (group << (s + 2)) + j;

            FpElement w2 = ff_pow_mont_u32(c_otf_roots[s + 1], j);
            FpElement w1 = ff_mul_ptx(w2, w2);
            FpElement w3 = ff_mul_ptx(w2, c_otf_consts[0]);

            FpElement a0 = data[base];
            FpElement a1 = data[base + half];

            FpElement t1 = ff_mul_ptx(a1, w1);
            FpElement a0p = ff_add_v2(a0, t1);
            FpElement a1p = ff_sub_v2(a0, t1);

            FpElement a2 = data[base + (half << 1)];
            FpElement a3 = data[base + (half << 1) + half];

            FpElement t2 = ff_mul_ptx(a3, w1);
            FpElement a2p = ff_add_v2(a2, t2);
            FpElement a3p = ff_sub_v2(a2, t2);

            FpElement t3 = ff_mul_ptx(a2p, w2);
            data[base]              = ff_add_v2(a0p, t3);
            data[base + (half << 1)] = ff_sub_v2(a0p, t3);

            FpElement t4 = ff_mul_ptx(a3p, w3);
            data[base + half]              = ff_add_v2(a1p, t4);
            data[base + (half << 1) + half] = ff_sub_v2(a1p, t4);
        }

        if (pass + 1 < num_r4_passes) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-2 Outer Kernel (Montgomery, Single, Cooperative) ─────────────
// For leftover single stages after radix-8/radix-4 OTF passes.

__global__ void ntt_outer_radix2_otf_kernel(
    FpElement* __restrict__ data,
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
        for (uint32_t tid = global_tid; tid < num_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t i0    = (group << (s + 1)) + j;
            const uint32_t i1    = i0 + half;

            FpElement w = ff_pow_mont_u32(c_otf_roots[s], j);

            FpElement a = data[i0];
            FpElement b = data[i1];
            FpElement t = ff_mul_ptx(b, w);
            data[i0] = ff_add_v2(a, t);
            data[i1] = ff_sub_v2(a, t);
        }
        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-2 Outer Kernel (Barrett, Single, Cooperative) ────────────────

__global__ void ntt_outer_radix2_otf_barrett_kernel(
    FpElement* __restrict__ data,
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
        for (uint32_t tid = global_tid; tid < num_butterflies; tid += total_threads) {
            const uint32_t group = tid >> s;
            const uint32_t j     = tid & (half - 1);
            const uint32_t i0    = (group << (s + 1)) + j;
            const uint32_t i1    = i0 + half;

            FpElement w = ff_pow_barrett_u32(c_otf_roots[s], j);

            FpElement a = data[i0];
            FpElement b = data[i1];
            FpElement t = ff_mul_barrett_v2(b, w);
            data[i0] = ff_add_v2(a, t);
            data[i1] = ff_sub_v2(a, t);
        }
        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-2 Outer Kernel (Montgomery, Batched, Cooperative) ─────────────

__global__ void ntt_outer_radix2_otf_batch_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    uint32_t total_butterflies,
    int start_stage,
    int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half = 1u << s;
        for (uint32_t tid = global_tid; tid < total_butterflies; tid += total_threads) {
            uint32_t group   = tid / half;
            uint32_t j       = tid % half;
            uint32_t idx_top = group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;

            FpElement w = ff_pow_mont_u32(c_otf_roots[s], j);

            FpElement a = data[idx_top];
            FpElement b = data[idx_bot];
            FpElement t = ff_mul_ptx(b, w);
            data[idx_top] = ff_add_v2(a, t);
            data[idx_bot] = ff_sub_v2(a, t);
        }
        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── OTF Radix-2 Outer Kernel (Barrett, Batched, Cooperative) ────────────────

__global__ void ntt_outer_radix2_otf_batch_barrett_kernel(
    FpElement* __restrict__ data,
    uint32_t n,
    uint32_t total_butterflies,
    int start_stage,
    int end_stage
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_threads = gridDim.x * blockDim.x;

    for (int s = start_stage; s < end_stage; ++s) {
        const uint32_t half = 1u << s;
        for (uint32_t tid = global_tid; tid < total_butterflies; tid += total_threads) {
            uint32_t group   = tid / half;
            uint32_t j       = tid % half;
            uint32_t idx_top = group * (2u * half) + j;
            uint32_t idx_bot = idx_top + half;

            FpElement w = ff_pow_barrett_u32(c_otf_roots[s], j);

            FpElement a = data[idx_top];
            FpElement b = data[idx_bot];
            FpElement t = ff_mul_barrett_v2(b, w);
            data[idx_top] = ff_add_v2(a, t);
            data[idx_bot] = ff_sub_v2(a, t);
        }
        if (s + 1 < end_stage) {
            grid.sync();
        }
    }
}

// ─── Batched Cooperative Launch Helpers ───────────────────────────────────────

static int s_coop_max_blocks_batch = 0;
static int s_coop_max_blocks_batch_barrett = 0;

static __host__ int get_coop_max_blocks_batch() {
    if (s_coop_max_blocks_batch > 0) return s_coop_max_blocks_batch;

    int num_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_fused_batch_kernel,
        OPT_BLOCK,
        0
    ));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_batch = num_blocks_per_sm * prop.multiProcessorCount;

    if (!prop.cooperativeLaunch) {
        s_coop_max_blocks_batch = 0;
    }

    return s_coop_max_blocks_batch;
}

static __host__ int get_coop_max_blocks_batch_barrett() {
    if (s_coop_max_blocks_batch_barrett > 0) return s_coop_max_blocks_batch_barrett;

    int num_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_fused_batch_barrett_kernel,
        OPT_BLOCK,
        0
    ));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_batch_barrett = num_blocks_per_sm * prop.multiProcessorCount;

    if (!prop.cooperativeLaunch) {
        s_coop_max_blocks_batch_barrett = 0;
    }

    return s_coop_max_blocks_batch_barrett;
}

// ─── Batched Radix-4 Cooperative Launch Helpers ──────────────────────────────

static int s_coop_max_blocks_r4_batch = 0;
static int s_coop_max_blocks_r4_batch_barrett = 0;

static __host__ int get_coop_max_blocks_r4_batch() {
    if (s_coop_max_blocks_r4_batch != 0) return s_coop_max_blocks_r4_batch;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix4_batch_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r4_batch = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r4_batch = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r4_batch == 0) {
        s_coop_max_blocks_r4_batch = -1;
    }
    return s_coop_max_blocks_r4_batch;
}

static __host__ int get_coop_max_blocks_r4_batch_barrett() {
    if (s_coop_max_blocks_r4_batch_barrett != 0) return s_coop_max_blocks_r4_batch_barrett;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix4_batch_barrett_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r4_batch_barrett = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r4_batch_barrett = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r4_batch_barrett == 0) {
        s_coop_max_blocks_r4_batch_barrett = -1;
    }
    return s_coop_max_blocks_r4_batch_barrett;
}

static __host__ bool launch_outer_radix4_batch(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream
) {
    int num_r4 = num_outer / 2;
    if (num_r4 == 0) return false;

    int r4_max = get_coop_max_blocks_r4_batch();
    if (r4_max <= 0) return false;

    uint32_t total_r4 = static_cast<uint32_t>(batch_size) * (n >> 2);
    uint32_t needed = (total_r4 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                         ? static_cast<uint32_t>(r4_max) : needed;

    int r4_stage = outer_start;
    for (int done = 0; done < num_r4; ) {
        int passes = num_r4 - done;
        if (passes > MAX_R4_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R4_PASSES_PER_COOP_LAUNCH;

        int ss = r4_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &total_r4, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix4_batch_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r4_stage += 2 * passes;
    }

    if (num_outer % 2 == 1) {
        int last = outer_start + num_outer - 1;
        uint32_t half   = 1u << last;
        uint32_t stride = n / (2u * half);
        uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        // Process each NTT's leftover stage
        for (int b = 0; b < batch_size; ++b) {
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data + b * n, d_twiddles, n, half, stride);
        }
    }
    return true;
}

static __host__ bool launch_outer_radix4_batch_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream
) {
    int num_r4 = num_outer / 2;
    if (num_r4 == 0) return false;

    int r4_max = get_coop_max_blocks_r4_batch_barrett();
    if (r4_max <= 0) return false;

    uint32_t total_r4 = static_cast<uint32_t>(batch_size) * (n >> 2);
    uint32_t needed = (total_r4 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                         ? static_cast<uint32_t>(r4_max) : needed;

    int r4_stage = outer_start;
    for (int done = 0; done < num_r4; ) {
        int passes = num_r4 - done;
        if (passes > MAX_R4_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R4_PASSES_PER_COOP_LAUNCH;

        int ss = r4_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &total_r4, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix4_batch_barrett_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r4_stage += 2 * passes;
    }

    if (num_outer % 2 == 1) {
        int last = outer_start + num_outer - 1;
        uint32_t half   = 1u << last;
        uint32_t stride = n / (2u * half);
        uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data + b * n, d_twiddles, n, half, stride);
        }
    }
    return true;
}

// ─── Radix-8 Cooperative Launch Helpers (Montgomery, single) ─────────────────

static int s_coop_max_blocks_r8 = 0;

static __host__ int get_coop_max_blocks_r8() {
    if (s_coop_max_blocks_r8 != 0) return s_coop_max_blocks_r8;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix8_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r8 = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r8 = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r8 == 0) {
        s_coop_max_blocks_r8 = -1;
    }
    return s_coop_max_blocks_r8;
}

// Try radix-8 dispatch (Montgomery, single). Returns true on success.
static __host__ bool launch_outer_radix8(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_r8 = num_outer / 3;
    if (num_r8 == 0) return false;

    int r8_max = get_coop_max_blocks_r8();
    if (r8_max <= 0) return false;

    uint32_t needed = (n / 8 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                         ? static_cast<uint32_t>(r8_max) : needed;

    int r8_stage = outer_start;
    for (int done = 0; done < num_r8; ) {
        int passes = num_r8 - done;
        if (passes > MAX_R8_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R8_PASSES_PER_COOP_LAUNCH;

        int ss = r8_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix8_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r8_stage += 3 * passes;
    }

    // Handle leftover stages after radix-8 passes
    int leftover = num_outer % 3;
    if (leftover == 2) {
        // Leftover pair: use radix-4 cooperative kernel
        int r4_max = get_coop_max_blocks_r4();
        if (r4_max > 0) {
            uint32_t r4_needed = (n / 4 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                               ? static_cast<uint32_t>(r4_max) : r4_needed;
            int ss = r8_stage;
            int np = 1;
            void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix4_kernel,
                dim3(r4_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) {
                // Fall back to 2 radix-2 stages
                for (int i = 0; i < 2; ++i) {
                    int s = r8_stage + i;
                    uint32_t half   = 1u << s;
                    uint32_t stride = n / (2u * half);
                    ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                        d_data, d_twiddles, n, half, stride);
                }
            }
        } else {
            for (int i = 0; i < 2; ++i) {
                int s = r8_stage + i;
                uint32_t half   = 1u << s;
                uint32_t stride = n / (2u * half);
                ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
        }
    } else if (leftover == 1) {
        int s = r8_stage;
        uint32_t half   = 1u << s;
        uint32_t stride = n / (2u * half);
        ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
            d_data, d_twiddles, n, half, stride);
    }
    return true;
}

// ─── Radix-8 Cooperative Launch Helpers (Barrett, single) ────────────────────

static int s_coop_max_blocks_r8_barrett = 0;

static __host__ int get_coop_max_blocks_r8_barrett() {
    if (s_coop_max_blocks_r8_barrett != 0) return s_coop_max_blocks_r8_barrett;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix8_barrett_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r8_barrett = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r8_barrett = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r8_barrett == 0) {
        s_coop_max_blocks_r8_barrett = -1;
    }
    return s_coop_max_blocks_r8_barrett;
}

// Try radix-8 dispatch (Barrett, single). Returns true on success.
static __host__ bool launch_outer_radix8_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_r8 = num_outer / 3;
    if (num_r8 == 0) return false;

    int r8_max = get_coop_max_blocks_r8_barrett();
    if (r8_max <= 0) return false;

    uint32_t needed = (n / 8 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                         ? static_cast<uint32_t>(r8_max) : needed;

    int r8_stage = outer_start;
    for (int done = 0; done < num_r8; ) {
        int passes = num_r8 - done;
        if (passes > MAX_R8_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R8_PASSES_PER_COOP_LAUNCH;

        int ss = r8_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix8_barrett_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r8_stage += 3 * passes;
    }

    // Handle leftover stages after radix-8 passes
    int leftover = num_outer % 3;
    if (leftover == 2) {
        int r4_max = get_coop_max_blocks_r4_barrett();
        if (r4_max > 0) {
            uint32_t r4_needed = (n / 4 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                               ? static_cast<uint32_t>(r4_max) : r4_needed;
            int ss = r8_stage;
            int np = 1;
            void* args[] = { &d_data, &d_twiddles, &n, &ss, &np };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix4_barrett_kernel,
                dim3(r4_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) {
                for (int i = 0; i < 2; ++i) {
                    int s = r8_stage + i;
                    uint32_t half   = 1u << s;
                    uint32_t stride = n / (2u * half);
                    ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                        d_data, d_twiddles, n, half, stride);
                }
            }
        } else {
            for (int i = 0; i < 2; ++i) {
                int s = r8_stage + i;
                uint32_t half   = 1u << s;
                uint32_t stride = n / (2u * half);
                ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_data, d_twiddles, n, half, stride);
            }
        }
    } else if (leftover == 1) {
        int s = r8_stage;
        uint32_t half   = 1u << s;
        uint32_t stride = n / (2u * half);
        ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
            d_data, d_twiddles, n, half, stride);
    }
    return true;
}

// ─── Radix-8 Cooperative Launch Helpers (Montgomery, batch) ──────────────────

static int s_coop_max_blocks_r8_batch = 0;

static __host__ int get_coop_max_blocks_r8_batch() {
    if (s_coop_max_blocks_r8_batch != 0) return s_coop_max_blocks_r8_batch;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix8_batch_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r8_batch = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r8_batch = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r8_batch == 0) {
        s_coop_max_blocks_r8_batch = -1;
    }
    return s_coop_max_blocks_r8_batch;
}

static __host__ bool launch_outer_radix8_batch(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream
) {
    int num_r8 = num_outer / 3;
    if (num_r8 == 0) return false;

    int r8_max = get_coop_max_blocks_r8_batch();
    if (r8_max <= 0) return false;

    uint32_t total_r8 = static_cast<uint32_t>(batch_size) * (n >> 3);
    uint32_t needed = (total_r8 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                         ? static_cast<uint32_t>(r8_max) : needed;

    int r8_stage = outer_start;
    for (int done = 0; done < num_r8; ) {
        int passes = num_r8 - done;
        if (passes > MAX_R8_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R8_PASSES_PER_COOP_LAUNCH;

        int ss = r8_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &total_r8, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix8_batch_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r8_stage += 3 * passes;
    }

    // Handle leftover stages after radix-8 passes
    int leftover = num_outer % 3;
    if (leftover == 2) {
        // Use batched radix-4 cooperative kernel for leftover pair
        int r4_max = get_coop_max_blocks_r4_batch();
        if (r4_max > 0) {
            uint32_t total_r4 = static_cast<uint32_t>(batch_size) * (n >> 2);
            uint32_t r4_needed = (total_r4 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                               ? static_cast<uint32_t>(r4_max) : r4_needed;
            int ss = r8_stage;
            int np = 1;
            void* args[] = { &d_data, &d_twiddles, &n, &total_r4, &ss, &np };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix4_batch_kernel,
                dim3(r4_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) {
                uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < 2; ++i) {
                        int s = r8_stage + i;
                        uint32_t half   = 1u << s;
                        uint32_t stride = n / (2u * half);
                        ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                            d_data + b * n, d_twiddles, n, half, stride);
                    }
                }
            }
        } else {
            uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < 2; ++i) {
                    int s = r8_stage + i;
                    uint32_t half   = 1u << s;
                    uint32_t stride = n / (2u * half);
                    ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                        d_data + b * n, d_twiddles, n, half, stride);
                }
            }
        }
    } else if (leftover == 1) {
        uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        int s = r8_stage;
        uint32_t half   = 1u << s;
        uint32_t stride = n / (2u * half);
        for (int b = 0; b < batch_size; ++b) {
            ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data + b * n, d_twiddles, n, half, stride);
        }
    }
    return true;
}

// ─── Radix-8 Cooperative Launch Helpers (Barrett, batch) ─────────────────────

static int s_coop_max_blocks_r8_batch_barrett = 0;

static __host__ int get_coop_max_blocks_r8_batch_barrett() {
    if (s_coop_max_blocks_r8_batch_barrett != 0) return s_coop_max_blocks_r8_batch_barrett;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix8_batch_barrett_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r8_batch_barrett = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r8_batch_barrett = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r8_batch_barrett == 0) {
        s_coop_max_blocks_r8_batch_barrett = -1;
    }
    return s_coop_max_blocks_r8_batch_barrett;
}

static __host__ bool launch_outer_radix8_batch_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream
) {
    int num_r8 = num_outer / 3;
    if (num_r8 == 0) return false;

    int r8_max = get_coop_max_blocks_r8_batch_barrett();
    if (r8_max <= 0) return false;

    uint32_t total_r8 = static_cast<uint32_t>(batch_size) * (n >> 3);
    uint32_t needed = (total_r8 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                         ? static_cast<uint32_t>(r8_max) : needed;

    int r8_stage = outer_start;
    for (int done = 0; done < num_r8; ) {
        int passes = num_r8 - done;
        if (passes > MAX_R8_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R8_PASSES_PER_COOP_LAUNCH;

        int ss = r8_stage;
        int np = passes;
        void* args[] = { &d_data, &d_twiddles, &n, &total_r8, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix8_batch_barrett_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r8_stage += 3 * passes;
    }

    // Handle leftover stages after radix-8 passes
    int leftover = num_outer % 3;
    if (leftover == 2) {
        int r4_max = get_coop_max_blocks_r4_batch_barrett();
        if (r4_max > 0) {
            uint32_t total_r4 = static_cast<uint32_t>(batch_size) * (n >> 2);
            uint32_t r4_needed = (total_r4 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                               ? static_cast<uint32_t>(r4_max) : r4_needed;
            int ss = r8_stage;
            int np = 1;
            void* args[] = { &d_data, &d_twiddles, &n, &total_r4, &ss, &np };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix4_batch_barrett_kernel,
                dim3(r4_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) {
                uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < 2; ++i) {
                        int s = r8_stage + i;
                        uint32_t half   = 1u << s;
                        uint32_t stride = n / (2u * half);
                        ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                            d_data + b * n, d_twiddles, n, half, stride);
                    }
                }
            }
        } else {
            uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < 2; ++i) {
                    int s = r8_stage + i;
                    uint32_t half   = 1u << s;
                    uint32_t stride = n / (2u * half);
                    ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                        d_data + b * n, d_twiddles, n, half, stride);
                }
            }
        }
    } else if (leftover == 1) {
        uint32_t grid_half = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        int s = r8_stage;
        uint32_t half   = 1u << s;
        uint32_t stride = n / (2u * half);
        for (int b = 0; b < batch_size; ++b) {
            ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                d_data + b * n, d_twiddles, n, half, stride);
        }
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// OTF Launch Helpers
// ═════════════════════════════════════════════════════════════════════════════

// ─── OTF Radix-8 (Montgomery, single) ────────────────────────────────────────

static int s_coop_max_blocks_r8_otf = 0;

static __host__ int get_coop_max_blocks_r8_otf() {
    if (s_coop_max_blocks_r8_otf != 0) return s_coop_max_blocks_r8_otf;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix8_otf_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r8_otf = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r8_otf = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r8_otf == 0) {
        s_coop_max_blocks_r8_otf = -1;
    }
    return s_coop_max_blocks_r8_otf;
}

static __host__ bool launch_outer_radix8_otf(
    FpElement* d_data, uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_r8 = num_outer / 3;
    if (num_r8 == 0) return false;

    int r8_max = get_coop_max_blocks_r8_otf();
    if (r8_max <= 0) return false;

    uint32_t needed = (n / 8 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                         ? static_cast<uint32_t>(r8_max) : needed;

    int r8_stage = outer_start;
    for (int done = 0; done < num_r8; ) {
        int passes = num_r8 - done;
        if (passes > MAX_R8_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R8_PASSES_PER_COOP_LAUNCH;

        int ss = r8_stage;
        int np = passes;
        void* args[] = { &d_data, &n, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix8_otf_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r8_stage += 3 * passes;
    }

    // Handle leftover stages after radix-8 OTF passes
    int leftover = num_outer % 3;
    if (leftover == 2) {
        // Use OTF radix-4 cooperative kernel for leftover pair
        int r4_max = get_coop_max_blocks_r4();
        if (r4_max > 0) {
            uint32_t r4_needed = (n / 4 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                               ? static_cast<uint32_t>(r4_max) : r4_needed;
            int ss = r8_stage;
            int np = 1;
            void* args[] = { &d_data, &n, &ss, &np };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix4_otf_kernel,
                dim3(r4_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) {
                // Fall back to precomputed radix-2
                // (twiddles not available here — would need to pass them)
                // In practice this should not happen.
                return false;
            }
        } else {
            return false;
        }
    } else if (leftover == 1) {
        // Use OTF radix-2 for single leftover stage
        int coop_max = get_coop_max_blocks();
        if (coop_max > 0) {
            uint32_t r2_needed = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r2_grid = (static_cast<uint32_t>(coop_max) < r2_needed)
                               ? static_cast<uint32_t>(coop_max) : r2_needed;
            int ss = r8_stage;
            int se = r8_stage + 1;
            void* args[] = { &d_data, &n, &ss, &se };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix2_otf_kernel,
                dim3(r2_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) return false;
        } else {
            return false;
        }
    }
    return true;
}

// ─── OTF Radix-4 (Barrett, single) ──────────────────────────────────────────

static int s_coop_max_blocks_r4_otf_barrett = 0;

static __host__ int get_coop_max_blocks_r4_otf_barrett() {
    if (s_coop_max_blocks_r4_otf_barrett != 0) return s_coop_max_blocks_r4_otf_barrett;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix4_otf_barrett_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r4_otf_barrett = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r4_otf_barrett = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r4_otf_barrett == 0) {
        s_coop_max_blocks_r4_otf_barrett = -1;
    }
    return s_coop_max_blocks_r4_otf_barrett;
}

static __host__ bool launch_outer_radix4_otf_barrett(
    FpElement* d_data, uint32_t n, int outer_start, int num_outer,
    uint32_t grid_half, cudaStream_t stream
) {
    int num_r4 = num_outer / 2;
    if (num_r4 == 0) return false;

    int r4_max = get_coop_max_blocks_r4_otf_barrett();
    if (r4_max <= 0) return false;

    uint32_t needed = (n / 4 + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                         ? static_cast<uint32_t>(r4_max) : needed;

    int r4_stage = outer_start;
    for (int done = 0; done < num_r4; ) {
        int passes = num_r4 - done;

        int ss = r4_stage;
        int np = passes;
        void* args[] = { &d_data, &n, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix4_otf_barrett_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r4_stage += 2 * passes;
    }

    // Handle leftover single stage
    if (num_outer % 2 == 1) {
        int last = outer_start + num_outer - 1;
        int coop_max = get_coop_max_blocks_barrett();
        if (coop_max > 0) {
            uint32_t r2_needed = (n / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r2_grid = (static_cast<uint32_t>(coop_max) < r2_needed)
                               ? static_cast<uint32_t>(coop_max) : r2_needed;
            int ss = last;
            int se = last + 1;
            void* args[] = { &d_data, &n, &ss, &se };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix2_otf_barrett_kernel,
                dim3(r2_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) return false;
        } else {
            return false;
        }
    }
    return true;
}

// ─── OTF Radix-8 (Montgomery, batched) ──────────────────────────────────────

static int s_coop_max_blocks_r8_otf_batch = 0;

static __host__ int get_coop_max_blocks_r8_otf_batch() {
    if (s_coop_max_blocks_r8_otf_batch != 0) return s_coop_max_blocks_r8_otf_batch;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix8_otf_batch_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r8_otf_batch = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r8_otf_batch = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r8_otf_batch == 0) {
        s_coop_max_blocks_r8_otf_batch = -1;
    }
    return s_coop_max_blocks_r8_otf_batch;
}

static __host__ bool launch_outer_radix8_otf_batch(
    FpElement* d_data, uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream
) {
    int num_r8 = num_outer / 3;
    if (num_r8 == 0) return false;

    int r8_max = get_coop_max_blocks_r8_otf_batch();
    if (r8_max <= 0) return false;

    uint32_t total_r8_butterflies = static_cast<uint32_t>(batch_size) * (n >> 3);
    uint32_t needed = (total_r8_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r8_max) < needed)
                         ? static_cast<uint32_t>(r8_max) : needed;

    int r8_stage = outer_start;
    for (int done = 0; done < num_r8; ) {
        int passes = num_r8 - done;
        if (passes > MAX_R8_PASSES_PER_COOP_LAUNCH)
            passes = MAX_R8_PASSES_PER_COOP_LAUNCH;

        int ss = r8_stage;
        int np = passes;
        void* args[] = { &d_data, &n, &total_r8_butterflies, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix8_otf_batch_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r8_stage += 3 * passes;
    }

    // Handle leftover stages
    int leftover = num_outer % 3;
    if (leftover == 2) {
        int r4_max = get_coop_max_blocks_r4_batch();
        if (r4_max > 0) {
            uint32_t total_r4_butterflies = static_cast<uint32_t>(batch_size) * (n >> 2);
            uint32_t r4_needed = (total_r4_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r4_grid = (static_cast<uint32_t>(r4_max) < r4_needed)
                               ? static_cast<uint32_t>(r4_max) : r4_needed;
            int ss = r8_stage;
            int np = 1;
            void* args[] = { &d_data, &n, &total_r4_butterflies, &ss, &np };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix4_otf_batch_kernel,
                dim3(r4_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) return false;
        } else {
            return false;
        }
    } else if (leftover == 1) {
        // Use OTF radix-2 batch cooperative kernel for single leftover stage
        int coop_max = get_coop_max_blocks_batch();
        if (coop_max > 0) {
            uint32_t total_r2_butterflies = static_cast<uint32_t>(batch_size) * (n >> 1);
            uint32_t r2_needed = (total_r2_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r2_grid = (static_cast<uint32_t>(coop_max) < r2_needed)
                               ? static_cast<uint32_t>(coop_max) : r2_needed;
            int ss = r8_stage;
            int se = r8_stage + 1;
            void* args[] = { &d_data, &n, &total_r2_butterflies, &ss, &se };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix2_otf_batch_kernel,
                dim3(r2_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) return false;
        } else {
            return false;
        }
    }
    return true;
}

// ─── OTF Radix-4 (Barrett, batched) ─────────────────────────────────────────

static int s_coop_max_blocks_r4_otf_batch_barrett = 0;

static __host__ int get_coop_max_blocks_r4_otf_batch_barrett() {
    if (s_coop_max_blocks_r4_otf_batch_barrett != 0) return s_coop_max_blocks_r4_otf_batch_barrett;

    int num_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        ntt_outer_radix4_otf_batch_barrett_kernel,
        OPT_BLOCK,
        0
    );
    if (err != cudaSuccess) {
        cudaGetLastError();
        s_coop_max_blocks_r4_otf_batch_barrett = -1;
        return -1;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    s_coop_max_blocks_r4_otf_batch_barrett = num_blocks_per_sm * prop.multiProcessorCount;
    if (!prop.cooperativeLaunch || s_coop_max_blocks_r4_otf_batch_barrett == 0) {
        s_coop_max_blocks_r4_otf_batch_barrett = -1;
    }
    return s_coop_max_blocks_r4_otf_batch_barrett;
}

static __host__ bool launch_outer_radix4_otf_batch_barrett(
    FpElement* d_data, uint32_t n, int batch_size, int outer_start, int num_outer,
    cudaStream_t stream
) {
    int num_r4 = num_outer / 2;
    if (num_r4 == 0) return false;

    int r4_max = get_coop_max_blocks_r4_otf_batch_barrett();
    if (r4_max <= 0) return false;

    uint32_t total_r4_butterflies = static_cast<uint32_t>(batch_size) * (n >> 2);
    uint32_t needed = (total_r4_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
    uint32_t grid_size = (static_cast<uint32_t>(r4_max) < needed)
                         ? static_cast<uint32_t>(r4_max) : needed;

    int r4_stage = outer_start;
    for (int done = 0; done < num_r4; ) {
        int passes = num_r4 - done;

        int ss = r4_stage;
        int np = passes;
        void* args[] = { &d_data, &n, &total_r4_butterflies, &ss, &np };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)ntt_outer_radix4_otf_batch_barrett_kernel,
            dim3(grid_size), dim3(OPT_BLOCK),
            args, 0, stream
        );
        if (err != cudaSuccess) return false;

        done += passes;
        r4_stage += 2 * passes;
    }

    // Handle leftover single stage
    if (num_outer % 2 == 1) {
        int last = outer_start + num_outer - 1;
        int coop_max = get_coop_max_blocks_batch_barrett();
        if (coop_max > 0) {
            uint32_t total_r2_butterflies = static_cast<uint32_t>(batch_size) * (n >> 1);
            uint32_t r2_needed = (total_r2_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
            uint32_t r2_grid = (static_cast<uint32_t>(coop_max) < r2_needed)
                               ? static_cast<uint32_t>(coop_max) : r2_needed;
            int ss = last;
            int se = last + 1;
            void* args[] = { &d_data, &n, &total_r2_butterflies, &ss, &se };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_radix2_otf_batch_barrett_kernel,
                dim3(r2_grid), dim3(OPT_BLOCK),
                args, 0, stream
            );
            if (err != cudaSuccess) return false;
        } else {
            return false;
        }
    }
    return true;
}

// ─── Batched Outer Stage Dispatch (Montgomery) ──────────────────────────────

static __host__ void dispatch_outer_stages_batch(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t N, int batch_size, int outer_start, int outer_end,
    cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;

    // NOTE: OTF disabled for BLS12-381 — exponentiation too expensive (see dispatch_outer_stages).

    // Precomputed radix-8
    if (num_outer >= 3 &&
        launch_outer_radix8_batch(d_data, d_twiddles, N, batch_size,
                                  outer_start, num_outer, stream))
    {
        return;
    }

    // Try radix-4 (halves DRAM passes for outer stages)
    if (num_outer >= 2 &&
        launch_outer_radix4_batch(d_data, d_twiddles, N, batch_size,
                                  outer_start, num_outer, stream))
    {
        return;
    }

    // Radix-2 cooperative fallback
    uint32_t total_butterflies = static_cast<uint32_t>(batch_size) * (N >> 1);
    int max_blocks = get_coop_max_blocks_batch();

    if (max_blocks > 0) {
        uint32_t needed_blocks = (total_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
        uint32_t grid_size = (static_cast<uint32_t>(max_blocks) < needed_blocks)
                             ? static_cast<uint32_t>(max_blocks)
                             : needed_blocks;

        for (int s = outer_start; s < outer_end; ) {
            int batch_end = s + MAX_STAGES_PER_COOP_LAUNCH;
            if (batch_end > outer_end) batch_end = outer_end;

            int ss = s;
            int se = batch_end;
            void* args[] = {
                &d_data, &d_twiddles, &N, &total_butterflies, &ss, &se
            };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_fused_batch_kernel,
                dim3(grid_size), dim3(OPT_BLOCK),
                args, 0, stream
            );

            if (err != cudaSuccess) {
                uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
                for (int b = 0; b < batch_size; ++b) {
                    FpElement* d_ntt = d_data + b * N;
                    for (int ss2 = s; ss2 < outer_end; ++ss2) {
                        uint32_t half   = 1u << ss2;
                        uint32_t stride = N / (2u * half);
                        ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                            d_ntt, d_twiddles, N, half, stride);
                    }
                }
                return;
            }
            s = batch_end;
        }
    } else {
        uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            FpElement* d_ntt = d_data + b * N;
            for (int ss = outer_start; ss < outer_end; ++ss) {
                uint32_t half   = 1u << ss;
                uint32_t stride = N / (2u * half);
                ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_ntt, d_twiddles, N, half, stride);
            }
        }
    }
}

// ─── Batched Outer Stage Dispatch (Barrett) ─────────────────────────────────

static __host__ void dispatch_outer_stages_batch_barrett(
    FpElement* d_data, const FpElement* d_twiddles,
    uint32_t N, int batch_size, int outer_start, int outer_end,
    cudaStream_t stream
) {
    int num_outer = outer_end - outer_start;

    // NOTE: Barrett radix-8 disabled. Barrett stays on radix-4.
    // NOTE: OTF disabled for BLS12-381 — exponentiation too expensive (see dispatch_outer_stages).

    // Precomputed radix-4
    if (num_outer >= 2 &&
        launch_outer_radix4_batch_barrett(d_data, d_twiddles, N, batch_size,
                                          outer_start, num_outer, stream))
    {
        return;
    }

    // Radix-2 cooperative fallback
    uint32_t total_butterflies = static_cast<uint32_t>(batch_size) * (N >> 1);
    int max_blocks = get_coop_max_blocks_batch_barrett();

    if (max_blocks > 0) {
        uint32_t needed_blocks = (total_butterflies + OPT_BLOCK - 1) / OPT_BLOCK;
        uint32_t grid_size = (static_cast<uint32_t>(max_blocks) < needed_blocks)
                             ? static_cast<uint32_t>(max_blocks)
                             : needed_blocks;

        for (int s = outer_start; s < outer_end; ) {
            int batch_end = s + MAX_STAGES_PER_COOP_LAUNCH;
            if (batch_end > outer_end) batch_end = outer_end;

            int ss = s;
            int se = batch_end;
            void* args[] = {
                &d_data, &d_twiddles, &N, &total_butterflies, &ss, &se
            };

            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)ntt_outer_fused_batch_barrett_kernel,
                dim3(grid_size), dim3(OPT_BLOCK),
                args, 0, stream
            );

            if (err != cudaSuccess) {
                uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
                for (int b = 0; b < batch_size; ++b) {
                    FpElement* d_ntt = d_data + b * N;
                    for (int ss2 = s; ss2 < outer_end; ++ss2) {
                        uint32_t half   = 1u << ss2;
                        uint32_t stride = N / (2u * half);
                        ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                            d_ntt, d_twiddles, N, half, stride);
                    }
                }
                return;
            }
            s = batch_end;
        }
    } else {
        uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            FpElement* d_ntt = d_data + b * N;
            for (int ss = outer_start; ss < outer_end; ++ss) {
                uint32_t half   = 1u << ss;
                uint32_t stride = N / (2u * half);
                ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_ntt, d_twiddles, N, half, stride);
            }
        }
    }
}

// ─── Batched Forward NTT (Montgomery, in-place) ─────────────────────────────

void ntt_forward_batch_montgomery(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t total_elements = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total_elements + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Batched bit-reverse permutation
    ntt_bit_reverse_batch_kernel<<<grid_total, OPT_BLOCK, 0, stream>>>(
        d_data, N, log_n, total_elements);

    // Step 2: Fused inner stages (existing kernel, B * blocks)
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);

        launch_fused(fused_k, d_data, d_twiddles, N, num_blocks, stream);

        // Step 3: Batched outer stages
        dispatch_outer_stages_batch(d_data, d_twiddles, N, batch_size,
                                    fused_k, log_n, stream);
    } else {
        // Small NTT fallback: process each NTT separately
        uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            FpElement* d_ntt = d_data + b * N;
            for (int s = 0; s < log_n; ++s) {
                uint32_t half   = 1u << s;
                uint32_t stride = N / (2u * half);
                ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_ntt, d_twiddles, N, half, stride);
            }
        }
    }
}

// ─── Batched Inverse NTT (Montgomery, in-place) ─────────────────────────────

void ntt_inverse_batch_montgomery(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t total_elements = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total_elements + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Batched bit-reverse permutation
    ntt_bit_reverse_batch_kernel<<<grid_total, OPT_BLOCK, 0, stream>>>(
        d_data, N, log_n, total_elements);

    // Step 2: Fused inner stages
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);

        launch_fused(fused_k, d_data, d_inv_twiddles, N, num_blocks, stream);

        dispatch_outer_stages_batch(d_data, d_inv_twiddles, N, batch_size,
                                    fused_k, log_n, stream);
    } else {
        uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            FpElement* d_ntt = d_data + b * N;
            for (int s = 0; s < log_n; ++s) {
                uint32_t half   = 1u << s;
                uint32_t stride = N / (2u * half);
                ntt_butterfly_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_ntt, d_inv_twiddles, N, half, stride);
            }
        }
    }

    // Step 3: Scale by n^{-1} (element-wise, process all B*N elements)
    ntt_scale_kernel<<<grid_total, OPT_BLOCK, 0, stream>>>(d_data, n_inv, total_elements);
}

// ─── Batched Forward NTT (Barrett, standard-form, in-place) ─────────────────

void ntt_forward_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t total_elements = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total_elements + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Batched bit-reverse permutation (domain-agnostic)
    ntt_bit_reverse_batch_kernel<<<grid_total, OPT_BLOCK, 0, stream>>>(
        d_data, N, log_n, total_elements);

    // Step 2: Fused inner stages (Barrett, B * blocks)
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);

        launch_fused_barrett(fused_k, d_data, d_twiddles, N, num_blocks, stream);

        // Step 3: Batched outer stages (Barrett)
        dispatch_outer_stages_batch_barrett(d_data, d_twiddles, N, batch_size,
                                            fused_k, log_n, stream);
    } else {
        uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            FpElement* d_ntt = d_data + b * N;
            for (int s = 0; s < log_n; ++s) {
                uint32_t half   = 1u << s;
                uint32_t stride = N / (2u * half);
                ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_ntt, d_twiddles, N, half, stride);
            }
        }
    }
}

// ─── Batched Inverse NTT (Barrett, standard-form, in-place) ─────────────────

void ntt_inverse_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    const int log_n = log2_of(n);
    const uint32_t total_elements = static_cast<uint32_t>(batch_size) * N;
    const uint32_t grid_total = (total_elements + OPT_BLOCK - 1) / OPT_BLOCK;

    // Step 1: Batched bit-reverse permutation
    ntt_bit_reverse_batch_kernel<<<grid_total, OPT_BLOCK, 0, stream>>>(
        d_data, N, log_n, total_elements);

    // Step 2: Fused inner stages (Barrett)
    const int fused_k = select_fused_k(log_n);

    if (log_n >= fused_k) {
        const int elems = 1 << fused_k;
        const uint32_t num_blocks = static_cast<uint32_t>(batch_size) * (N / elems);

        launch_fused_barrett(fused_k, d_data, d_inv_twiddles, N, num_blocks, stream);

        dispatch_outer_stages_batch_barrett(d_data, d_inv_twiddles, N, batch_size,
                                            fused_k, log_n, stream);
    } else {
        uint32_t grid_half = (N / 2 + OPT_BLOCK - 1) / OPT_BLOCK;
        for (int b = 0; b < batch_size; ++b) {
            FpElement* d_ntt = d_data + b * N;
            for (int s = 0; s < log_n; ++s) {
                uint32_t half   = 1u << s;
                uint32_t stride = N / (2u * half);
                ntt_butterfly_barrett_kernel<<<grid_half, OPT_BLOCK, 0, stream>>>(
                    d_ntt, d_inv_twiddles, N, half, stride);
            }
        }
    }

    // Step 3: Scale by n^{-1} using Barrett multiply (element-wise)
    ntt_scale_barrett_kernel<<<grid_total, OPT_BLOCK, 0, stream>>>(d_data, n_inv, total_elements);
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
