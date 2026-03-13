// src/msm.cu
// Pippenger's bucket method for GPU-accelerated Multi-Scalar Multiplication.
//
// Algorithm:
//   1. Window decomposition: each 255-bit scalar -> W windows of c bits
//   2. For each window w:
//      a. Extract bucket IDs (c-bit window of each scalar)
//      b. Sort (bucket_id, point_index) by bucket_id using CUB radix sort
//      c. Bucket accumulation: sequential EC adds per bucket
//      d. Bucket reduction: running sum from top bucket down
//   3. Window combination: Horner's method across windows
//
// Processing windows sequentially to limit sort memory usage.

#include "msm.cuh"
#include "cuda_utils.cuh"

#include <cub/cub.cuh>

// ─── Constants ─────────────────────────────────────────────────────────────

static constexpr int SCALAR_BITS = 255;  // BLS12-381 Fr is ~255 bits

// ─── Kernel: Extract window bits ───────────────────────────────────────────
// For each point i, extract c-bit window starting at bit_offset from scalar[i].
// Output: d_bucket_ids[i] = window bits (0..2^c-1)
//         d_point_indices[i] = i

__global__ void extract_window_kernel(
    const uint32_t* __restrict__ d_scalars,  // n * 8 uint32_t
    uint32_t* __restrict__ d_bucket_ids,
    uint32_t* __restrict__ d_point_indices,
    uint32_t n,
    int bit_offset,
    int window_bits)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Extract window_bits bits starting at bit_offset from scalar[idx]
    const uint32_t* scalar = &d_scalars[idx * 8];

    uint32_t bucket_id = 0;
    for (int b = 0; b < window_bits; ++b) {
        int global_bit = bit_offset + b;
        if (global_bit >= 256) break;
        int limb = global_bit / 32;
        int bit_in_limb = global_bit % 32;
        if (scalar[limb] & (1u << bit_in_limb)) {
            bucket_id |= (1u << b);
        }
    }

    d_bucket_ids[idx] = bucket_id;
    d_point_indices[idx] = idx;
}

// ─── Kernel: Bucket accumulation ───────────────────────────────────────────
// Sorted (bucket_id, point_index) arrays. Find contiguous runs per bucket,
// accumulate points in each bucket using EC addition.
// One thread per bucket.

__global__ void bucket_accumulate_kernel(
    const G1Affine* __restrict__ d_bases,
    const uint32_t* __restrict__ d_sorted_bucket_ids,
    const uint32_t* __restrict__ d_sorted_point_indices,
    G1Jacobian* __restrict__ d_buckets,  // num_buckets entries
    uint32_t n,
    uint32_t num_buckets)
{
    uint32_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= num_buckets) return;

    // Bucket 0 = zero scalar window, skip (contributes nothing)
    if (bucket == 0) {
        d_buckets[0] = G1Jacobian::identity();
        return;
    }

    G1Jacobian acc = G1Jacobian::identity();

    // Binary search for first occurrence of this bucket_id
    // (sorted_bucket_ids is sorted, so contiguous runs)
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (d_sorted_bucket_ids[mid] < bucket) lo = mid + 1;
        else hi = mid;
    }

    // Accumulate all points in this bucket
    while (lo < n && d_sorted_bucket_ids[lo] == bucket) {
        uint32_t pt_idx = d_sorted_point_indices[lo];
        acc = g1_add_mixed(acc, d_bases[pt_idx]);
        ++lo;
    }

    d_buckets[bucket] = acc;
}

// ─── Kernel: Bucket reduction ──────────────────────────────────────────────
// Running sum from top bucket down:
//   S = bucket[num_buckets-1]
//   R = S
//   for b = num_buckets-2 down to 1:
//     S = S + bucket[b]
//     R = R + S
// Result = R (the window contribution)
// Single thread per window (num_buckets is small enough).

__global__ void bucket_reduce_kernel(
    const G1Jacobian* __restrict__ d_buckets,
    G1Jacobian* __restrict__ d_window_result,
    uint32_t num_buckets)
{
    // Single thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    G1Jacobian S = d_buckets[num_buckets - 1];
    G1Jacobian R = S;

    for (int b = (int)num_buckets - 2; b >= 1; --b) {
        S = g1_add(S, d_buckets[b]);
        R = g1_add(R, S);
    }

    d_window_result[0] = R;
}

// ─── Kernel: Window combination (Horner's method) ──────────────────────────
// result = window_results[W-1]
// for w = W-2 down to 0:
//   result = result * 2^c + window_results[w]
// Where "result * 2^c" = c doublings.

__global__ void window_combine_kernel(
    const G1Jacobian* __restrict__ d_window_results,  // W entries
    G1Jacobian* __restrict__ d_final_result,
    int num_windows,
    int window_bits)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    G1Jacobian result = d_window_results[num_windows - 1];

    for (int w = num_windows - 2; w >= 0; --w) {
        // result *= 2^window_bits (c doublings)
        for (int d = 0; d < window_bits; ++d) {
            result = g1_double(result);
        }
        // result += window_results[w]
        result = g1_add(result, d_window_results[w]);
    }

    d_final_result[0] = result;
}

// ─── Local helper kernels (avoid cross-TU extern issues with CUB + RDC) ──

__global__ void msm_scalar_mul_kernel(const G1Jacobian* __restrict__ bases,
                                       const uint32_t* __restrict__ scalars,
                                       G1Jacobian* __restrict__ out,
                                       uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_scalar_mul(bases[idx], &scalars[idx * 8]);
}

__global__ void msm_to_affine_kernel(const G1Jacobian* __restrict__ in,
                                      G1Affine* __restrict__ out,
                                      uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = g1_to_affine(in[idx]);
}

// ─── Host function: MSM ────────────────────────────────────────────────────

void msm_g1(G1Affine* result,
            const G1Affine* d_bases,
            const uint32_t* d_scalars,
            size_t n,
            cudaStream_t stream)
{
    if (n == 0) {
        *result = G1Affine::point_at_infinity();
        return;
    }

    // Single point: just scalar_mul
    if (n == 1) {
        G1Affine h_base;
        uint32_t h_scalar[8];
        CUDA_CHECK(cudaMemcpyAsync(&h_base, d_bases, sizeof(G1Affine), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_scalar, d_scalars, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        G1Jacobian base_jac;
        base_jac.x = h_base.x;
        base_jac.y = h_base.y;
        base_jac.z = FqElement::one_mont();

        G1Jacobian *d_base_jac, *d_out_jac;
        uint32_t *d_sc;
        G1Affine *d_out_aff;
        CUDA_CHECK(cudaMalloc(&d_base_jac, sizeof(G1Jacobian)));
        CUDA_CHECK(cudaMalloc(&d_out_jac, sizeof(G1Jacobian)));
        CUDA_CHECK(cudaMalloc(&d_sc, 8 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_out_aff, sizeof(G1Affine)));
        CUDA_CHECK(cudaMemcpy(d_base_jac, &base_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, h_scalar, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        msm_scalar_mul_kernel<<<1, 1, 0, stream>>>(d_base_jac, d_sc, d_out_jac, 1);
        msm_to_affine_kernel<<<1, 1, 0, stream>>>(d_out_jac, d_out_aff, 1);
        CUDA_CHECK(cudaMemcpyAsync(result, d_out_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_base_jac));
        CUDA_CHECK(cudaFree(d_out_jac));
        CUDA_CHECK(cudaFree(d_sc));
        CUDA_CHECK(cudaFree(d_out_aff));
        return;
    }

    int c = msm_optimal_window(n);
    int num_windows = (SCALAR_BITS + c - 1) / c;
    uint32_t num_buckets = 1u << c;  // 2^c buckets (including bucket 0)

    // Allocate per-window working memory
    uint32_t *d_bucket_ids = nullptr, *d_point_indices = nullptr;
    uint32_t *d_sorted_bucket_ids = nullptr, *d_sorted_point_indices = nullptr;
    G1Jacobian *d_buckets = nullptr;
    G1Jacobian *d_window_results = nullptr;
    G1Jacobian *d_final_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bucket_ids, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_point_indices, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_bucket_ids, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_point_indices, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_buckets, num_buckets * sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_window_results, num_windows * sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(G1Jacobian)));

    // CUB sort temp storage
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_sort_temp, sort_temp_bytes,
        d_bucket_ids, d_sorted_bucket_ids,
        d_point_indices, d_sorted_point_indices,
        (int)n, 0, c, stream);
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));

    const int BLOCK = 256;
    int grid_n = ((int)n + BLOCK - 1) / BLOCK;
    int grid_buckets = ((int)num_buckets + BLOCK - 1) / BLOCK;

    // Process each window sequentially
    for (int w = 0; w < num_windows; ++w) {
        int bit_offset = w * c;
        int effective_bits = c;
        if (bit_offset + effective_bits > SCALAR_BITS) {
            effective_bits = SCALAR_BITS - bit_offset;
            if (effective_bits <= 0) {
                // This window is all zeros - store identity
                G1Jacobian id = G1Jacobian::identity();
                CUDA_CHECK(cudaMemcpyAsync(
                    &d_window_results[w], &id, sizeof(G1Jacobian),
                    cudaMemcpyHostToDevice, stream));
                continue;
            }
        }

        // Step 1: Extract window bits
        extract_window_kernel<<<grid_n, BLOCK, 0, stream>>>(
            d_scalars, d_bucket_ids, d_point_indices,
            (uint32_t)n, bit_offset, effective_bits);

        // Step 2: Sort by bucket_id
        cub::DeviceRadixSort::SortPairs(
            d_sort_temp, sort_temp_bytes,
            d_bucket_ids, d_sorted_bucket_ids,
            d_point_indices, d_sorted_point_indices,
            (int)n, 0, c, stream);

        // Step 3: Bucket accumulation
        // Initialize all buckets to identity
        // Use actual num_buckets for this window (may be smaller for last window)
        uint32_t effective_num_buckets = 1u << effective_bits;
        bucket_accumulate_kernel<<<grid_buckets, BLOCK, 0, stream>>>(
            d_bases, d_sorted_bucket_ids, d_sorted_point_indices,
            d_buckets, (uint32_t)n, effective_num_buckets);

        // Step 4: Bucket reduction (single thread)
        bucket_reduce_kernel<<<1, 1, 0, stream>>>(
            d_buckets, &d_window_results[w], effective_num_buckets);
    }

    // Step 5: Window combination (single thread)
    window_combine_kernel<<<1, 1, 0, stream>>>(
        d_window_results, d_final_result, num_windows, c);

    // Convert to affine
    G1Affine *d_result_aff;
    CUDA_CHECK(cudaMalloc(&d_result_aff, sizeof(G1Affine)));

    msm_to_affine_kernel<<<1, 1, 0, stream>>>(d_final_result, d_result_aff, 1);

    CUDA_CHECK(cudaMemcpyAsync(result, d_result_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup
    CUDA_CHECK(cudaFree(d_bucket_ids));
    CUDA_CHECK(cudaFree(d_point_indices));
    CUDA_CHECK(cudaFree(d_sorted_bucket_ids));
    CUDA_CHECK(cudaFree(d_sorted_point_indices));
    CUDA_CHECK(cudaFree(d_buckets));
    CUDA_CHECK(cudaFree(d_window_results));
    CUDA_CHECK(cudaFree(d_final_result));
    CUDA_CHECK(cudaFree(d_sort_temp));
    CUDA_CHECK(cudaFree(d_result_aff));
}
