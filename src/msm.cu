// src/msm.cu
// Pippenger's bucket method for GPU-accelerated Multi-Scalar Multiplication.
//
// v2.1.0 Session 26 improvements:
//   - Signed-digit window recoding: halves bucket count (2^c -> 2^(c-1))
//   - Segment-offset accumulation: replaces O(log n) binary search with O(1) lookup
//   - Improved window sizing: c = floor(log2(n)/2) + 1
//
// Algorithm:
//   1. For each window w (sequentially):
//      a. Extract signed window digits (with carry propagation)
//      b. Sort (|digit|, packed_value) by bucket_id using CUB radix sort
//      c. Compute segment offsets from sorted data
//      d. Bucket accumulation: one thread per bucket using segment offsets
//         Points with negative digits are negated before adding
//      e. Bucket reduction: running sum from top bucket down
//   2. Window combination: Horner's method across windows

#include "msm.cuh"
#include "cuda_utils.cuh"

#include <cub/cub.cuh>

// ─── Constants ─────────────────────────────────────────────────────────────

static constexpr int SCALAR_BITS = 255;  // BLS12-381 Fr is ~255 bits

// ─── Kernel: Extract signed window digits ─────────────────────────────────
// Signed-digit recoding: if raw window value >= 2^(c-1), subtract 2^c and
// carry +1 to next window. This halves the bucket count.
// Output packed_value: point_index in bits [0..30], sign in bit 31.

__global__ void extract_signed_window_kernel(
    const uint32_t* __restrict__ d_scalars,  // n * 8 uint32_t
    uint32_t* __restrict__ d_bucket_ids,     // |digit| (0..2^(c-1))
    uint32_t* __restrict__ d_packed_values,  // point_idx | (sign << 31)
    uint8_t* __restrict__ d_carries,         // carry buffer (in/out)
    uint32_t n,
    int bit_offset,
    int window_bits,
    int is_first_window)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t* scalar = &d_scalars[idx * 8];

    // Extract window_bits bits starting at bit_offset
    uint32_t raw = 0;
    for (int b = 0; b < window_bits; ++b) {
        int global_bit = bit_offset + b;
        if (global_bit >= 256) break;
        int limb = global_bit / 32;
        int bit_in_limb = global_bit % 32;
        if (scalar[limb] & (1u << bit_in_limb)) {
            raw |= (1u << b);
        }
    }

    // Add carry from previous window
    uint32_t carry_in = is_first_window ? 0u : (uint32_t)d_carries[idx];
    raw += carry_in;

    uint32_t half = 1u << (window_bits - 1);
    uint32_t full = 1u << window_bits;

    int32_t digit;
    uint8_t carry_out;

    if (raw >= half) {
        digit = (int32_t)raw - (int32_t)full;
        carry_out = 1;
    } else {
        digit = (int32_t)raw;
        carry_out = 0;
    }

    d_carries[idx] = carry_out;

    uint32_t sign = (digit < 0) ? 1u : 0u;
    uint32_t abs_digit = (uint32_t)((digit < 0) ? -digit : digit);

    d_bucket_ids[idx] = abs_digit;
    d_packed_values[idx] = idx | (sign << 31);
}

// ─── Kernel: Handle final carry window ───────────────────────────────────
// After all regular windows, any remaining carry contributes to an extra
// "virtual" window with digit = carry (0 or 1).
// For bucket_id = 1 entries with carry, negate is never set (carry is +1).

__global__ void extract_carry_window_kernel(
    const uint8_t* __restrict__ d_carries,
    uint32_t* __restrict__ d_bucket_ids,
    uint32_t* __restrict__ d_packed_values,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t carry = (uint32_t)d_carries[idx];
    d_bucket_ids[idx] = carry;  // 0 or 1
    d_packed_values[idx] = idx; // sign = 0 (positive)
}

// ─── Kernel: Find segment start offsets from sorted data ─────────────────
// After CUB sort by bucket_id, marks where each bucket's run begins.
// Uses atomicMin to handle parallel writes correctly.

__global__ void find_segment_starts_kernel(
    const uint32_t* __restrict__ d_sorted_bucket_ids,
    uint32_t* __restrict__ d_segment_offsets,  // num_buckets + 1 entries
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t curr = d_sorted_bucket_ids[idx];

    if (idx == 0 || d_sorted_bucket_ids[idx - 1] != curr) {
        atomicMin(&d_segment_offsets[curr], idx);
    }
}

// ─── Kernel: Backward-fill empty segments ────────────────────────────────
// For empty buckets (offset == UINT32_MAX after atomicMin), set their offset
// to the next non-empty bucket's offset. Ensures monotonicity.
// num_buckets is small (≤ 2^15), single thread is fine.

__global__ void fill_segments_kernel(
    uint32_t* __restrict__ d_segment_offsets,
    uint32_t num_buckets,
    uint32_t n)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    d_segment_offsets[num_buckets] = n;
    for (int b = (int)num_buckets - 1; b >= 0; --b) {
        if (d_segment_offsets[b] == 0xFFFFFFFFu) {
            d_segment_offsets[b] = d_segment_offsets[b + 1];
        }
    }
}

// ─── Kernel: Bucket accumulation with segment offsets ────────────────────
// One thread per bucket. Uses precomputed segment offsets for O(1) lookup.
// Handles signed-digit: negates point y-coordinate when sign bit is set.

__global__ void bucket_accumulate_kernel(
    const G1Affine* __restrict__ d_bases,
    const uint32_t* __restrict__ d_sorted_packed_values,
    const uint32_t* __restrict__ d_segment_offsets,
    G1Jacobian* __restrict__ d_buckets,
    uint32_t num_buckets)
{
    uint32_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= num_buckets) return;

    if (bucket == 0) {
        d_buckets[0] = G1Jacobian::identity();
        return;
    }

    uint32_t start = d_segment_offsets[bucket];
    uint32_t end = d_segment_offsets[bucket + 1];

    G1Jacobian acc = G1Jacobian::identity();

    for (uint32_t i = start; i < end; ++i) {
        uint32_t val = d_sorted_packed_values[i];
        uint32_t pt_idx = val & 0x7FFFFFFFu;
        bool negate = (val >> 31) != 0;

        G1Affine base = d_bases[pt_idx];
        if (negate) {
            base.y = fq_neg(base.y);
        }
        acc = g1_add_mixed(acc, base);
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
    int total_windows = num_windows + 1;  // +1 for carry overflow window
    uint32_t num_buckets = (1u << (c - 1)) + 1;  // signed: buckets 0..2^(c-1)

    // ─── Allocate working memory ───────────────────────────────────────────
    uint32_t *d_bucket_ids = nullptr, *d_packed_values = nullptr;
    uint32_t *d_sorted_bucket_ids = nullptr, *d_sorted_packed_values = nullptr;
    uint32_t *d_segment_offsets = nullptr;
    uint8_t *d_carries = nullptr;
    G1Jacobian *d_buckets = nullptr;
    G1Jacobian *d_window_results = nullptr;
    G1Jacobian *d_final_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bucket_ids, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_packed_values, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_bucket_ids, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_packed_values, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, (num_buckets + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_carries, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_buckets, num_buckets * sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_window_results, total_windows * sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(G1Jacobian)));

    // CUB sort temp storage — query once, reuse for all windows
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_sort_temp, sort_temp_bytes,
        d_bucket_ids, d_sorted_bucket_ids,
        d_packed_values, d_sorted_packed_values,
        (int)n, 0, c, stream);
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));

    const int BLOCK = 256;
    int grid_n = ((int)n + BLOCK - 1) / BLOCK;
    int grid_buckets = ((int)num_buckets + BLOCK - 1) / BLOCK;

    // ─── Process each window sequentially ──────────────────────────────────
    for (int w = 0; w < num_windows; ++w) {
        int bit_offset = w * c;
        int effective_bits = c;
        if (bit_offset + effective_bits > SCALAR_BITS) {
            effective_bits = SCALAR_BITS - bit_offset;
            if (effective_bits <= 0) {
                // This window is all zeros — store identity
                G1Jacobian id = G1Jacobian::identity();
                CUDA_CHECK(cudaMemcpyAsync(
                    &d_window_results[w], &id, sizeof(G1Jacobian),
                    cudaMemcpyHostToDevice, stream));
                continue;
            }
        }

        // Clamp to at least 2 bits for signed recoding (need half >= 1)
        int recode_bits = (effective_bits < 2) ? 2 : effective_bits;

        // Step 1: Extract signed window digits
        extract_signed_window_kernel<<<grid_n, BLOCK, 0, stream>>>(
            d_scalars, d_bucket_ids, d_packed_values, d_carries,
            (uint32_t)n, bit_offset, recode_bits, (w == 0) ? 1 : 0);

        // Step 2: CUB radix sort by bucket_id
        int sort_bits = c;  // sort key width
        cub::DeviceRadixSort::SortPairs(
            d_sort_temp, sort_temp_bytes,
            d_bucket_ids, d_sorted_bucket_ids,
            d_packed_values, d_sorted_packed_values,
            (int)n, 0, sort_bits, stream);

        // Step 3: Compute segment offsets
        // Initialize offsets to UINT32_MAX (marks "empty")
        CUDA_CHECK(cudaMemsetAsync(d_segment_offsets, 0xFF,
            (num_buckets + 1) * sizeof(uint32_t), stream));

        find_segment_starts_kernel<<<grid_n, BLOCK, 0, stream>>>(
            d_sorted_bucket_ids, d_segment_offsets, (uint32_t)n);

        fill_segments_kernel<<<1, 1, 0, stream>>>(
            d_segment_offsets, num_buckets, (uint32_t)n);

        // Step 4: Bucket accumulation (one thread per bucket)
        bucket_accumulate_kernel<<<grid_buckets, BLOCK, 0, stream>>>(
            d_bases, d_sorted_packed_values, d_segment_offsets,
            d_buckets, num_buckets);

        // Step 5: Bucket reduction (single thread)
        bucket_reduce_kernel<<<1, 1, 0, stream>>>(
            d_buckets, &d_window_results[w], num_buckets);
    }

    // ─── Carry overflow window ─────────────────────────────────────────────
    // After all regular windows, handle remaining carries.
    // Carry window: digit = carry (0 or 1), always positive.
    // Uses bucket 0 (skip) and bucket 1 only, with just 2 buckets.
    {
        int cw = total_windows - 1;  // carry window index
        uint32_t carry_buckets = 2;  // just bucket 0 and 1

        extract_carry_window_kernel<<<grid_n, BLOCK, 0, stream>>>(
            d_carries, d_bucket_ids, d_packed_values, (uint32_t)n);

        // Sort with 1 bit (bucket 0 or 1)
        cub::DeviceRadixSort::SortPairs(
            d_sort_temp, sort_temp_bytes,
            d_bucket_ids, d_sorted_bucket_ids,
            d_packed_values, d_sorted_packed_values,
            (int)n, 0, 1, stream);

        CUDA_CHECK(cudaMemsetAsync(d_segment_offsets, 0xFF,
            (carry_buckets + 1) * sizeof(uint32_t), stream));

        find_segment_starts_kernel<<<grid_n, BLOCK, 0, stream>>>(
            d_sorted_bucket_ids, d_segment_offsets, (uint32_t)n);

        fill_segments_kernel<<<1, 1, 0, stream>>>(
            d_segment_offsets, carry_buckets, (uint32_t)n);

        bucket_accumulate_kernel<<<1, BLOCK, 0, stream>>>(
            d_bases, d_sorted_packed_values, d_segment_offsets,
            d_buckets, carry_buckets);

        // For carry window: only bucket 1 matters.
        // Reduction with 2 buckets: result = bucket[1]
        bucket_reduce_kernel<<<1, 1, 0, stream>>>(
            d_buckets, &d_window_results[cw], carry_buckets);
    }

    // ─── Window combination (Horner's method) ──────────────────────────────
    window_combine_kernel<<<1, 1, 0, stream>>>(
        d_window_results, d_final_result, total_windows, c);

    // Convert to affine
    G1Affine *d_result_aff;
    CUDA_CHECK(cudaMalloc(&d_result_aff, sizeof(G1Affine)));

    msm_to_affine_kernel<<<1, 1, 0, stream>>>(d_final_result, d_result_aff, 1);

    CUDA_CHECK(cudaMemcpyAsync(result, d_result_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup
    CUDA_CHECK(cudaFree(d_bucket_ids));
    CUDA_CHECK(cudaFree(d_packed_values));
    CUDA_CHECK(cudaFree(d_sorted_bucket_ids));
    CUDA_CHECK(cudaFree(d_sorted_packed_values));
    CUDA_CHECK(cudaFree(d_segment_offsets));
    CUDA_CHECK(cudaFree(d_carries));
    CUDA_CHECK(cudaFree(d_buckets));
    CUDA_CHECK(cudaFree(d_window_results));
    CUDA_CHECK(cudaFree(d_final_result));
    CUDA_CHECK(cudaFree(d_sort_temp));
    CUDA_CHECK(cudaFree(d_result_aff));
}
