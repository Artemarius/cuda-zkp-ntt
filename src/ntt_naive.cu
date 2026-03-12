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

// Batched NTT dispatch (from ntt_optimized.cu)
extern void ntt_forward_batch_montgomery(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_batch_montgomery(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);
extern void ntt_forward_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);

// 4-Step NTT dispatch (from ntt_4step.cu)
extern void ntt_4step_forward_barrett(
    FpElement* d_data, size_t n,
    const FpElement* d_twiddles_n1, const FpElement* d_twiddles_n2,
    const FpElement* d_twiddles_4step, cudaStream_t stream);
extern void ntt_4step_inverse_barrett(
    FpElement* d_data, size_t n,
    const FpElement* d_inv_twiddles_n1, const FpElement* d_inv_twiddles_n2,
    const FpElement* d_inv_twiddles_4step,
    FpElement n1_inv, FpElement n2_inv, cudaStream_t stream);
extern void ntt_4step_forward_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles_n1, const FpElement* d_twiddles_n2,
    const FpElement* d_twiddles_4step, cudaStream_t stream);
extern void ntt_4step_inverse_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles_n1, const FpElement* d_inv_twiddles_n2,
    const FpElement* d_inv_twiddles_4step,
    FpElement n1_inv, FpElement n2_inv, cudaStream_t stream);
extern void ntt_4step_get_split(uint32_t n, uint32_t& n1, uint32_t& n2);

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

// ─── 4-Step Twiddle Cache ────────────────────────────────────────────────────
// Caches sub-NTT twiddles for n1-point and n2-point transforms, plus the
// 4-step twiddle table omega_n^(i*j). All in standard form (Barrett).
// Keyed by n (from which n1, n2 are derived).

static FpElement* s_d_fwd_twiddles_4s_n1 = nullptr;  // sub-NTT twiddles for n1-point (size n1/2)
static FpElement* s_d_inv_twiddles_4s_n1 = nullptr;
static FpElement* s_d_fwd_twiddles_4s_n2 = nullptr;  // sub-NTT twiddles for n2-point (size n2/2)
static FpElement* s_d_inv_twiddles_4s_n2 = nullptr;
static FpElement* s_d_fwd_twiddles_4step = nullptr;   // omega_n^(i*j), size n
static FpElement* s_d_inv_twiddles_4step = nullptr;   // omega_n^(-i*j), size n
static FpElement  s_n1_inv_4step;                      // n1^{-1} in standard form
static FpElement  s_n2_inv_4step;                      // n2^{-1} in standard form
static size_t     s_cached_n_4step = 0;

static void free_cached_twiddles_4step() {
    // Free n2 twiddles first (only if distinct from n1, i.e., n1 != n2)
    if (s_d_fwd_twiddles_4s_n2 && s_d_fwd_twiddles_4s_n2 != s_d_fwd_twiddles_4s_n1) {
        cudaFree(s_d_fwd_twiddles_4s_n2);
    }
    s_d_fwd_twiddles_4s_n2 = nullptr;
    if (s_d_inv_twiddles_4s_n2 && s_d_inv_twiddles_4s_n2 != s_d_inv_twiddles_4s_n1) {
        cudaFree(s_d_inv_twiddles_4s_n2);
    }
    s_d_inv_twiddles_4s_n2 = nullptr;
    // Now free n1 twiddles
    if (s_d_fwd_twiddles_4s_n1) { cudaFree(s_d_fwd_twiddles_4s_n1); s_d_fwd_twiddles_4s_n1 = nullptr; }
    if (s_d_inv_twiddles_4s_n1) { cudaFree(s_d_inv_twiddles_4s_n1); s_d_inv_twiddles_4s_n1 = nullptr; }
    if (s_d_fwd_twiddles_4step) { cudaFree(s_d_fwd_twiddles_4step); s_d_fwd_twiddles_4step = nullptr; }
    if (s_d_inv_twiddles_4step) { cudaFree(s_d_inv_twiddles_4step); s_d_inv_twiddles_4step = nullptr; }
    s_cached_n_4step = 0;
}

// Helper: compute standard-form twiddles for a given sub-NTT size and upload to device.
static void compute_sub_twiddles_barrett(
    size_t sub_n,
    FpElement** d_fwd, FpElement** d_inv, FpElement* h_sub_inv
) {
    size_t half = sub_n / 2;

    ff_ref::FpRef omega     = ff_ref::get_root_of_unity(sub_n);
    ff_ref::FpRef omega_inv = ff_ref::fp_inv(omega);
    ff_ref::FpRef one_mont;
    one_mont.limbs = ff_ref::R_MOD;

    std::vector<FpElement> h_fwd(half), h_inv(half);
    ff_ref::FpRef wk     = one_mont;
    ff_ref::FpRef wk_inv = one_mont;
    for (size_t k = 0; k < half; ++k) {
        ff_ref::FpRef wk_std     = ff_ref::from_montgomery(wk);
        ff_ref::FpRef wk_inv_std = ff_ref::from_montgomery(wk_inv);
        wk_std.to_u32(h_fwd[k].limbs);
        wk_inv_std.to_u32(h_inv[k].limbs);
        wk     = ff_ref::fp_mul(wk, omega);
        wk_inv = ff_ref::fp_mul(wk_inv, omega_inv);
    }

    CUDA_CHECK(cudaMalloc(d_fwd, half * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(d_inv, half * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(*d_fwd, h_fwd.data(), half * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_inv, h_inv.data(), half * sizeof(FpElement), cudaMemcpyHostToDevice));

    // Compute sub_n^{-1} in standard form
    ff_ref::FpRef n_mont     = ff_ref::to_montgomery(ff_ref::FpRef::from_u64(static_cast<uint64_t>(sub_n)));
    ff_ref::FpRef n_inv_mont = ff_ref::fp_inv(n_mont);
    ff_ref::FpRef n_inv_std  = ff_ref::from_montgomery(n_inv_mont);
    n_inv_std.to_u32(h_sub_inv->limbs);
}

static void ensure_twiddles_4step(size_t n) {
    if (s_cached_n_4step == n) return;
    free_cached_twiddles_4step();

    uint32_t n1, n2;
    ntt_4step_get_split(static_cast<uint32_t>(n), n1, n2);

    // Sub-NTT twiddles for n1-point and n2-point transforms
    compute_sub_twiddles_barrett(n1, &s_d_fwd_twiddles_4s_n1, &s_d_inv_twiddles_4s_n1, &s_n1_inv_4step);
    if (n2 != n1) {
        compute_sub_twiddles_barrett(n2, &s_d_fwd_twiddles_4s_n2, &s_d_inv_twiddles_4s_n2, &s_n2_inv_4step);
    } else {
        // n1 == n2 (even log_n): reuse same twiddle tables
        s_d_fwd_twiddles_4s_n2 = s_d_fwd_twiddles_4s_n1;
        s_d_inv_twiddles_4s_n2 = s_d_inv_twiddles_4s_n1;
        s_n2_inv_4step = s_n1_inv_4step;
    }

    // 4-step twiddle table: omega_n^(i*j) for i in [0,n1), j in [0,n2)
    // Stored in row-major: index = i*n2 + j
    ff_ref::FpRef omega_n = ff_ref::get_root_of_unity(n);
    ff_ref::FpRef omega_n_inv = ff_ref::fp_inv(omega_n);

    std::vector<FpElement> h_fwd_4s(n), h_inv_4s(n);

    // omega_n^i for each row
    ff_ref::FpRef one_mont;
    one_mont.limbs = ff_ref::R_MOD;
    ff_ref::FpRef omega_i = one_mont;  // omega_n^0 = 1
    ff_ref::FpRef omega_inv_i = one_mont;

    for (uint32_t i = 0; i < n1; ++i) {
        // For row i: twiddle[i][j] = omega_n^(i*j)
        ff_ref::FpRef omega_ij = one_mont;  // omega_n^(i*0) = 1
        ff_ref::FpRef omega_inv_ij = one_mont;
        for (uint32_t j = 0; j < n2; ++j) {
            ff_ref::FpRef fwd_std = ff_ref::from_montgomery(omega_ij);
            ff_ref::FpRef inv_std = ff_ref::from_montgomery(omega_inv_ij);
            fwd_std.to_u32(h_fwd_4s[i * n2 + j].limbs);
            inv_std.to_u32(h_inv_4s[i * n2 + j].limbs);

            omega_ij = ff_ref::fp_mul(omega_ij, omega_i);
            omega_inv_ij = ff_ref::fp_mul(omega_inv_ij, omega_inv_i);
        }
        omega_i = ff_ref::fp_mul(omega_i, omega_n);
        omega_inv_i = ff_ref::fp_mul(omega_inv_i, omega_n_inv);
    }

    CUDA_CHECK(cudaMalloc(&s_d_fwd_twiddles_4step, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&s_d_inv_twiddles_4step, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(s_d_fwd_twiddles_4step, h_fwd_4s.data(),
                          n * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s_d_inv_twiddles_4step, h_inv_4s.data(),
                          n * sizeof(FpElement), cudaMemcpyHostToDevice));

    s_cached_n_4step = n;
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

// Batched bit-reversal permutation: each NTT in the batch is bit-reversed independently.
// total_elements = batch_size * n.
__global__ void ntt_bit_reverse_batch_kernel(
    FpElement* data, uint32_t n, uint32_t log_n, uint32_t total_elements
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    uint32_t batch_id = tid / n;
    uint32_t i = tid - batch_id * n;  // local index within this NTT

    uint32_t j = __brev(i) >> (32 - log_n);

    if (i < j) {
        uint32_t base = batch_id * n;
        FpElement temp = data[base + i];
        data[base + i] = data[base + j];
        data[base + j] = temp;
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
        case NTTMode::FOUR_STEP: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            // 4-step requires sub-NTTs of at least 256 elements (K=8 fused kernel).
            // Minimum full NTT size: 2^16 (sub-NTTs = 2^8 = 256).
            // For smaller sizes, fall back to Barrett (same arithmetic, no conversion overhead).
            if (n < (1u << 16)) {
                ensure_twiddles_barrett(n);
                ntt_forward_optimized_barrett(d_data, n, s_d_fwd_twiddles_barrett, stream);
            } else {
                ensure_twiddles_4step(n);
                ntt_4step_forward_barrett(d_data, n,
                    s_d_fwd_twiddles_4s_n1, s_d_fwd_twiddles_4s_n2,
                    s_d_fwd_twiddles_4step, stream);
            }
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
        case NTTMode::FOUR_STEP: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            if (n < (1u << 16)) {
                ensure_twiddles_barrett(n);
                ntt_inverse_optimized_barrett(d_data, n, s_d_inv_twiddles_barrett, s_n_inv_barrett, stream);
            } else {
                ensure_twiddles_4step(n);
                ntt_4step_inverse_barrett(d_data, n,
                    s_d_inv_twiddles_4s_n1, s_d_inv_twiddles_4s_n2,
                    s_d_inv_twiddles_4step,
                    s_n1_inv_4step, s_n2_inv_4step, stream);
            }
            break;
        }
        case NTTMode::ASYNC:
            fprintf(stderr, "ntt_inverse(ASYNC): use AsyncNTTPipeline class for async pipeline NTT\n");
            break;
    }
}

// ─── Batched Public API ──────────────────────────────────────────────────────

void ntt_forward_batch(FpElement* d_data, int batch_size, size_t n, NTTMode mode, cudaStream_t stream) {
    if (batch_size <= 0) return;
    if (batch_size == 1) { ntt_forward(d_data, n, mode, stream); return; }

    switch (mode) {
        case NTTMode::OPTIMIZED: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles(n);
            uint32_t N = static_cast<uint32_t>(n);
            uint32_t total = static_cast<uint32_t>(batch_size) * N;
            uint32_t grid = (total + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

            // Convert all elements to Montgomery form
            ntt_to_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, total);

            // Batched NTT on Montgomery-form data
            ntt_forward_batch_montgomery(d_data, batch_size, n, s_d_fwd_twiddles, stream);

            // Convert back to standard form
            ntt_from_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, total);
            break;
        }
        case NTTMode::BARRETT: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles_barrett(n);
            ntt_forward_batch_barrett(d_data, batch_size, n, s_d_fwd_twiddles_barrett, stream);
            break;
        }
        case NTTMode::FOUR_STEP: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            if (n < (1u << 16)) {
                ensure_twiddles_barrett(n);
                ntt_forward_batch_barrett(d_data, batch_size, n, s_d_fwd_twiddles_barrett, stream);
            } else {
                ensure_twiddles_4step(n);
                ntt_4step_forward_batch_barrett(d_data, batch_size, n,
                    s_d_fwd_twiddles_4s_n1, s_d_fwd_twiddles_4s_n2,
                    s_d_fwd_twiddles_4step, stream);
            }
            break;
        }
        default:
            fprintf(stderr, "ntt_forward_batch: mode %d not supported for batching\n",
                    static_cast<int>(mode));
            break;
    }
}

void ntt_inverse_batch(FpElement* d_data, int batch_size, size_t n, NTTMode mode, cudaStream_t stream) {
    if (batch_size <= 0) return;
    if (batch_size == 1) { ntt_inverse(d_data, n, mode, stream); return; }

    switch (mode) {
        case NTTMode::OPTIMIZED: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles(n);
            uint32_t N = static_cast<uint32_t>(n);
            uint32_t total = static_cast<uint32_t>(batch_size) * N;
            uint32_t grid = (total + NTT_BLOCK_SIZE - 1) / NTT_BLOCK_SIZE;

            ntt_to_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, total);
            ntt_inverse_batch_montgomery(d_data, batch_size, n, s_d_inv_twiddles, s_n_inv, stream);
            ntt_from_montgomery_kernel<<<grid, NTT_BLOCK_SIZE, 0, stream>>>(d_data, total);
            break;
        }
        case NTTMode::BARRETT: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            ensure_twiddles_barrett(n);
            ntt_inverse_batch_barrett(d_data, batch_size, n,
                s_d_inv_twiddles_barrett, s_n_inv_barrett, stream);
            break;
        }
        case NTTMode::FOUR_STEP: {
            assert(n >= 2 && (n & (n - 1)) == 0);
            if (n < (1u << 16)) {
                ensure_twiddles_barrett(n);
                ntt_inverse_batch_barrett(d_data, batch_size, n,
                    s_d_inv_twiddles_barrett, s_n_inv_barrett, stream);
            } else {
                ensure_twiddles_4step(n);
                ntt_4step_inverse_batch_barrett(d_data, batch_size, n,
                    s_d_inv_twiddles_4s_n1, s_d_inv_twiddles_4s_n2,
                    s_d_inv_twiddles_4step,
                    s_n1_inv_4step, s_n2_inv_4step, stream);
            }
            break;
        }
        default:
            fprintf(stderr, "ntt_inverse_batch: mode %d not supported for batching\n",
                    static_cast<int>(mode));
            break;
    }
}

// ─── CUDA Graph Cache ────────────────────────────────────────────────────────
// Captures NTT kernel launch sequences as CUDA Graphs for replay with minimal
// CPU overhead (~5us vs ~20-40us for individual launches).

struct NTTGraphEntry {
    FpElement*      d_data;
    uint32_t        n;
    int             batch_size;   // 1 for single NTT
    NTTMode         mode;
    bool            forward;
    cudaGraphExec_t graphExec;
};

static std::vector<NTTGraphEntry> s_graph_cache;
static cudaStream_t s_capture_stream = nullptr;

static NTTGraphEntry* find_graph_entry(
    FpElement* d_data, uint32_t n, int batch_size, NTTMode mode, bool forward
) {
    for (auto& e : s_graph_cache) {
        if (e.d_data == d_data && e.n == n && e.batch_size == batch_size &&
            e.mode == mode && e.forward == forward) {
            return &e;
        }
    }
    return nullptr;
}

static void ensure_capture_stream() {
    if (!s_capture_stream) {
        CUDA_CHECK(cudaStreamCreate(&s_capture_stream));
    }
}

// Pre-warm twiddle caches so ensure_twiddles* inside ntt_forward/ntt_inverse
// are no-ops during capture (no cudaMalloc/cudaMemcpy).
static void prewarm_twiddles(size_t n, NTTMode mode) {
    switch (mode) {
        case NTTMode::NAIVE:
        case NTTMode::OPTIMIZED:
            ensure_twiddles(n);
            break;
        case NTTMode::BARRETT:
            ensure_twiddles_barrett(n);
            break;
        default:
            break;
    }
}

static cudaGraphExec_t capture_ntt_graph(
    FpElement* d_data, size_t n, int batch_size, NTTMode mode, bool forward
) {
    ensure_capture_stream();
    prewarm_twiddles(n, mode);

    // Also pre-warm Montgomery twiddles if mode needs them
    if (mode == NTTMode::OPTIMIZED || mode == NTTMode::NAIVE) {
        ensure_twiddles(n);
    }

    CUDA_CHECK(cudaStreamBeginCapture(s_capture_stream, cudaStreamCaptureModeRelaxed));

    if (batch_size <= 1) {
        if (forward)
            ntt_forward(d_data, n, mode, s_capture_stream);
        else
            ntt_inverse(d_data, n, mode, s_capture_stream);
    } else {
        if (forward)
            ntt_forward_batch(d_data, batch_size, n, mode, s_capture_stream);
        else
            ntt_inverse_batch(d_data, batch_size, n, mode, s_capture_stream);
    }

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(s_capture_stream, &graph));

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    return graphExec;
}

void ntt_forward_graph(FpElement* d_data, size_t n, NTTMode mode, cudaStream_t stream) {
    assert(mode != NTTMode::FOUR_STEP && "FOUR_STEP not supported with CUDA Graphs (internal cudaMalloc)");
    assert(mode != NTTMode::ASYNC && "ASYNC not supported with CUDA Graphs");
    uint32_t N = static_cast<uint32_t>(n);

    NTTGraphEntry* entry = find_graph_entry(d_data, N, 1, mode, true);
    if (!entry) {
        cudaGraphExec_t ge = capture_ntt_graph(d_data, n, 1, mode, true);
        s_graph_cache.push_back({d_data, N, 1, mode, true, ge});
        entry = &s_graph_cache.back();
    }
    CUDA_CHECK(cudaGraphLaunch(entry->graphExec, stream));
}

void ntt_inverse_graph(FpElement* d_data, size_t n, NTTMode mode, cudaStream_t stream) {
    assert(mode != NTTMode::FOUR_STEP && "FOUR_STEP not supported with CUDA Graphs (internal cudaMalloc)");
    assert(mode != NTTMode::ASYNC && "ASYNC not supported with CUDA Graphs");
    uint32_t N = static_cast<uint32_t>(n);

    NTTGraphEntry* entry = find_graph_entry(d_data, N, 1, mode, false);
    if (!entry) {
        cudaGraphExec_t ge = capture_ntt_graph(d_data, n, 1, mode, false);
        s_graph_cache.push_back({d_data, N, 1, mode, false, ge});
        entry = &s_graph_cache.back();
    }
    CUDA_CHECK(cudaGraphLaunch(entry->graphExec, stream));
}

void ntt_forward_batch_graph(FpElement* d_data, int batch_size, size_t n,
                             NTTMode mode, cudaStream_t stream) {
    assert(mode != NTTMode::FOUR_STEP && "FOUR_STEP not supported with CUDA Graphs (internal cudaMalloc)");
    assert(mode != NTTMode::ASYNC && "ASYNC not supported with CUDA Graphs");
    uint32_t N = static_cast<uint32_t>(n);

    NTTGraphEntry* entry = find_graph_entry(d_data, N, batch_size, mode, true);
    if (!entry) {
        cudaGraphExec_t ge = capture_ntt_graph(d_data, n, batch_size, mode, true);
        s_graph_cache.push_back({d_data, N, batch_size, mode, true, ge});
        entry = &s_graph_cache.back();
    }
    CUDA_CHECK(cudaGraphLaunch(entry->graphExec, stream));
}

void ntt_inverse_batch_graph(FpElement* d_data, int batch_size, size_t n,
                             NTTMode mode, cudaStream_t stream) {
    assert(mode != NTTMode::FOUR_STEP && "FOUR_STEP not supported with CUDA Graphs (internal cudaMalloc)");
    assert(mode != NTTMode::ASYNC && "ASYNC not supported with CUDA Graphs");
    uint32_t N = static_cast<uint32_t>(n);

    NTTGraphEntry* entry = find_graph_entry(d_data, N, batch_size, mode, false);
    if (!entry) {
        cudaGraphExec_t ge = capture_ntt_graph(d_data, n, batch_size, mode, false);
        s_graph_cache.push_back({d_data, N, batch_size, mode, false, ge});
        entry = &s_graph_cache.back();
    }
    CUDA_CHECK(cudaGraphLaunch(entry->graphExec, stream));
}

void ntt_graph_clear_cache() {
    for (auto& e : s_graph_cache) {
        cudaGraphExecDestroy(e.graphExec);
    }
    s_graph_cache.clear();
    if (s_capture_stream) {
        cudaStreamDestroy(s_capture_stream);
        s_capture_stream = nullptr;
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
