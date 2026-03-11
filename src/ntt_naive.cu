// src/ntt_naive.cu
// Radix-2 Cooley-Tukey NTT (global memory only) + public API dispatch
// Phase 4: naive GPU NTT baseline. One kernel launch per butterfly stage.

#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"
#include "ff_reference.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

// Forward declarations from ntt_optimized.cu (linked via separable compilation)
extern void ntt_forward_optimized_montgomery(
    FpElement* d_data, size_t n, const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_optimized_montgomery(
    FpElement* d_data, size_t n, const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);

// Barrett NTT dispatch (from ntt_optimized.cu)
extern void ntt_forward_optimized_barrett(
    FpElement* d_data, size_t n, const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_optimized_barrett(
    FpElement* d_data, size_t n, const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);

// ─── Internal Twiddle Cache ──────────────────────────────────────────────────
// Twiddles are computed on CPU (using ff_ref) and uploaded to device.
// Forward: twiddles[k] = omega_n^k  for k = 0..n/2-1
// Inverse: twiddles[k] = omega_n^{-k} for k = 0..n/2-1
// Cached across calls for the same n.

static FpElement* s_d_fwd_twiddles = nullptr;
static FpElement* s_d_inv_twiddles = nullptr;
static FpElement  s_n_inv;          // n^{-1} in Montgomery form (host copy)
static size_t     s_cached_n = 0;

// Barrett twiddle cache: twiddles in standard form (no Montgomery)
static FpElement* s_d_fwd_twiddles_barrett = nullptr;
static FpElement* s_d_inv_twiddles_barrett = nullptr;
static FpElement  s_n_inv_barrett;  // n^{-1} in standard form (host copy)
static size_t     s_cached_n_barrett = 0;

static void free_cached_twiddles() {
    if (s_d_fwd_twiddles) { cudaFree(s_d_fwd_twiddles); s_d_fwd_twiddles = nullptr; }
    if (s_d_inv_twiddles) { cudaFree(s_d_inv_twiddles); s_d_inv_twiddles = nullptr; }
    s_cached_n = 0;
}

static void free_cached_twiddles_barrett() {
    if (s_d_fwd_twiddles_barrett) { cudaFree(s_d_fwd_twiddles_barrett); s_d_fwd_twiddles_barrett = nullptr; }
    if (s_d_inv_twiddles_barrett) { cudaFree(s_d_inv_twiddles_barrett); s_d_inv_twiddles_barrett = nullptr; }
    s_cached_n_barrett = 0;
}

static int compute_log2(size_t n) {
    int log_n = 0;
    size_t tmp = n;
    while (tmp > 1) { tmp >>= 1; ++log_n; }
    return log_n;
}

static void ensure_twiddles(size_t n) {
    if (s_cached_n == n) return;
    free_cached_twiddles();

    size_t half = n / 2;

    // Compute twiddles on CPU using ff_ref
    ff_ref::FpRef omega     = ff_ref::get_root_of_unity(n);
    ff_ref::FpRef omega_inv = ff_ref::fp_inv(omega);
    ff_ref::FpRef one;
    one.limbs = ff_ref::R_MOD;  // Montgomery(1)

    std::vector<FpElement> h_fwd(half), h_inv(half);

    ff_ref::FpRef wk     = one;
    ff_ref::FpRef wk_inv = one;
    for (size_t k = 0; k < half; ++k) {
        wk.to_u32(h_fwd[k].limbs);
        wk_inv.to_u32(h_inv[k].limbs);
        wk     = ff_ref::fp_mul(wk, omega);
        wk_inv = ff_ref::fp_mul(wk_inv, omega_inv);
    }

    // Upload to device
    CUDA_CHECK(cudaMalloc(&s_d_fwd_twiddles, half * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&s_d_inv_twiddles, half * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(s_d_fwd_twiddles, h_fwd.data(),
                          half * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_d_inv_twiddles, h_inv.data(),
                          half * sizeof(FpElement), cudaMemcpyHostToDevice));

    // Compute n^{-1} in Montgomery form
    ff_ref::FpRef n_mont   = ff_ref::to_montgomery(ff_ref::FpRef::from_u64(static_cast<uint64_t>(n)));
    ff_ref::FpRef n_inv_ref = ff_ref::fp_inv(n_mont);
    n_inv_ref.to_u32(s_n_inv.limbs);

    s_cached_n = n;
}

// Precompute twiddle factors in STANDARD form for Barrett NTT path.
// Computes in Montgomery domain (well-tested), then converts to standard form.
static void ensure_twiddles_barrett(size_t n) {
    if (s_cached_n_barrett == n) return;
    free_cached_twiddles_barrett();

    size_t half = n / 2;

    // Compute twiddles in Montgomery form (reuse existing logic)
    ff_ref::FpRef omega     = ff_ref::get_root_of_unity(n);
    ff_ref::FpRef omega_inv = ff_ref::fp_inv(omega);
    ff_ref::FpRef one_mont;
    one_mont.limbs = ff_ref::R_MOD;  // Montgomery(1)

    std::vector<FpElement> h_fwd(half), h_inv(half);

    ff_ref::FpRef wk     = one_mont;
    ff_ref::FpRef wk_inv = one_mont;
    for (size_t k = 0; k < half; ++k) {
        // Convert from Montgomery to standard form for Barrett path
        ff_ref::FpRef wk_std     = ff_ref::from_montgomery(wk);
        ff_ref::FpRef wk_inv_std = ff_ref::from_montgomery(wk_inv);

        wk_std.to_u32(h_fwd[k].limbs);
        wk_inv_std.to_u32(h_inv[k].limbs);

        wk     = ff_ref::fp_mul(wk, omega);
        wk_inv = ff_ref::fp_mul(wk_inv, omega_inv);
    }

    // Upload to device
    CUDA_CHECK(cudaMalloc(&s_d_fwd_twiddles_barrett, half * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&s_d_inv_twiddles_barrett, half * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(s_d_fwd_twiddles_barrett, h_fwd.data(),
                          half * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_d_inv_twiddles_barrett, h_inv.data(),
                          half * sizeof(FpElement), cudaMemcpyHostToDevice));

    // Compute n^{-1} in standard form
    ff_ref::FpRef n_mont     = ff_ref::to_montgomery(ff_ref::FpRef::from_u64(static_cast<uint64_t>(n)));
    ff_ref::FpRef n_inv_mont = ff_ref::fp_inv(n_mont);
    ff_ref::FpRef n_inv_std  = ff_ref::from_montgomery(n_inv_mont);
    n_inv_std.to_u32(s_n_inv_barrett.limbs);

    s_cached_n_barrett = n;
}

// ─── Kernels ────────────────────────────────────────────────────────────────

// Bit-reversal permutation (in-place). Uses __brev() intrinsic.
__global__ void ntt_bit_reverse_kernel(FpElement* data, uint32_t n, uint32_t log_n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t j = __brev(i) >> (32 - log_n);

    if (i < j) {
        FpElement temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

// Single butterfly stage kernel.
// Each thread handles one butterfly pair. Total n/2 threads per stage.
// half = len/2 where len = 2^(stage+1) is the butterfly group size.
// stride = n / len = twiddle stride for this stage.
__global__ void ntt_butterfly_kernel(
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
    FpElement v = ff_mul(data[idx_bot], twiddles[tw_idx]);
    data[idx_top] = ff_add(u, v);
    data[idx_bot] = ff_sub(u, v);
}

// Convert standard form → Montgomery form (element-wise)
__global__ void ntt_to_montgomery_kernel(FpElement* data, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = ff_to_montgomery(data[i]);
}

// Convert Montgomery form → standard form (element-wise)
__global__ void ntt_from_montgomery_kernel(FpElement* data, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = ff_from_montgomery(data[i]);
}

// Scale each element by a scalar (for n^{-1} in inverse NTT)
__global__ void ntt_scale_kernel(FpElement* data, FpElement scalar, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = ff_mul(data[i], scalar);
}

// ─── Internal NTT Implementation ────────────────────────────────────────────

static constexpr uint32_t NTT_BLOCK_SIZE = 256;

// Forward NTT on Montgomery-form data (in-place on device)
static void ntt_forward_naive_montgomery(
    FpElement* d_data, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream
) {
    uint32_t N = static_cast<uint32_t>(n);
    int log_n = compute_log2(n);
    uint32_t grid     = (N + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;
    uint32_t grid_half = (N / 2 + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

    // Bit-reverse permutation
    ntt_bit_reverse_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N, log_n);

    // Butterfly stages: stage s → len = 2^(s+1), half = 2^s
    for (int s = 0; s < log_n; ++s) {
        uint32_t half   = 1u << s;
        uint32_t stride = N / (2u * half);  // = N / len
        ntt_butterfly_kernel<<<grid_half, NTT_BLOCK_SIZE, 0, stream>>>(
            d_data, d_twiddles, N, half, stride);
    }
}

// Inverse NTT on Montgomery-form data (in-place on device)
static void ntt_inverse_naive_montgomery(
    FpElement* d_data, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream
) {
    uint32_t N = static_cast<uint32_t>(n);
    int log_n = compute_log2(n);
    uint32_t grid     = (N + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;
    uint32_t grid_half = (N / 2 + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

    // Bit-reverse permutation
    ntt_bit_reverse_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N, log_n);

    // Butterfly stages (using inverse twiddles)
    for (int s = 0; s < log_n; ++s) {
        uint32_t half   = 1u << s;
        uint32_t stride = N / (2u * half);
        ntt_butterfly_kernel<<<grid_half, NTT_BLOCK_SIZE, 0, stream>>>(
            d_data, d_inv_twiddles, N, half, stride);
    }

    // Scale by n^{-1}
    ntt_scale_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, n_inv, N);
}

// ─── Public API ──────────────────────────────────────────────────────────────

void ntt_forward(FpElement* d_data, size_t n, NTTMode mode, cudaStream_t stream) {
    switch (mode) {
        case NTTMode::NAIVE: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles(n);
            uint32_t N = static_cast<uint32_t>(n);
            uint32_t grid = (N + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

            // Convert input to Montgomery form
            ntt_to_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);

            // Run NTT on Montgomery-form data
            ntt_forward_naive_montgomery(d_data, n, s_d_fwd_twiddles, stream);

            // Convert output back to standard form
            ntt_from_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);
            break;
        }
        case NTTMode::OPTIMIZED: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles(n);
            uint32_t N = static_cast<uint32_t>(n);
            uint32_t grid = (N + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

            ntt_to_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);
            ntt_forward_optimized_montgomery(d_data, n, s_d_fwd_twiddles, stream);
            ntt_from_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);
            break;
        }
        case NTTMode::BARRETT: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles_barrett(n);
            // No Montgomery conversion — Barrett operates in standard form
            ntt_forward_optimized_barrett(d_data, n, s_d_fwd_twiddles_barrett, stream);
            break;
        }
        case NTTMode::ASYNC:
            fprintf(stderr, "ntt_forward(ASYNC): use AsyncNTTPipeline class for async pipeline NTT\n");
            break;
    }
}

void ntt_inverse(FpElement* d_data, size_t n, NTTMode mode, cudaStream_t stream) {
    switch (mode) {
        case NTTMode::NAIVE: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles(n);
            uint32_t N = static_cast<uint32_t>(n);
            uint32_t grid = (N + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

            // Convert input to Montgomery form
            ntt_to_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);

            // Run inverse NTT
            ntt_inverse_naive_montgomery(d_data, n, s_d_inv_twiddles, s_n_inv, stream);

            // Convert back to standard form
            ntt_from_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);
            break;
        }
        case NTTMode::OPTIMIZED: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles(n);
            uint32_t N = static_cast<uint32_t>(n);
            uint32_t grid = (N + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

            ntt_to_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);
            ntt_inverse_optimized_montgomery(d_data, n, s_d_inv_twiddles, s_n_inv, stream);
            ntt_from_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, N);
            break;
        }
        case NTTMode::BARRETT: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles_barrett(n);
            // No Montgomery conversion — Barrett operates in standard form
            ntt_inverse_optimized_barrett(d_data, n, s_d_inv_twiddles_barrett, s_n_inv_barrett, stream);
            break;
        }
        case NTTMode::ASYNC:
            fprintf(stderr, "ntt_inverse(ASYNC): use AsyncNTTPipeline class for async pipeline NTT\n");
            break;
    }
}

FpElement* ntt_precompute_twiddles(size_t n) {
    ensure_twiddles(n);
    return s_d_fwd_twiddles;
}

void ntt_free_twiddles(FpElement* d_twiddles) {
    if (d_twiddles == s_d_fwd_twiddles) {
        free_cached_twiddles();
    } else if (d_twiddles) {
        CUDA_CHECK(cudaFree(d_twiddles));
    }
}
