// src/ntt_4step.cu
// 4-Step NTT (Bailey's algorithm) for BLS12-381 scalar field.
// Decomposes n-point NTT into sub-NTTs that fit entirely in shared memory,
// eliminating the memory-bound outer stages that dominate execution time.
//
// Algorithm for n = n1 * n2:
//   1. View input as n1 x n2 matrix (row-major)
//   2. Column NTTs: n2 independent n1-point NTTs (on columns)
//      → Transpose to row-major, then batch n2 sub-NTTs on contiguous rows
//   3. Twiddle multiply: pointwise data[i*n2+j] *= omega_n^(i*j)
//   4. Row NTTs: n1 independent n2-point NTTs (on rows)
//      → Transpose to column-major, batch n1 sub-NTTs, transpose back
//
// Decomposition strategy:
//   - Even log_n: n1 = n2 = sqrt(n) = 2^(log_n/2)
//   - Odd log_n:  n1 = 2^((log_n-1)/2), n2 = 2^((log_n+1)/2)  (n2 > n1)
//   - Sub-NTT size capped at 2^11 = 2048 (fits in fused K=10 kernel with shmem)
//   - Minimum n for 4-step: n >= 2^16 (below that, fused+cooperative is fine)
//
// v1.3.0 Session 5: Transpose kernel + architecture skeleton

#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "ff_barrett.cuh"
#include "cuda_utils.cuh"

#include <cstdio>
#include <cassert>

// ─── Transpose Kernel ────────────────────────────────────────────────────────
// Transpose an n1 x n2 matrix of FpElement in-place (via out-of-place copy).
//
// Classic GPU transpose pattern:
//   - Each thread block handles a TILE x TILE tile
//   - Coalesced read from src into shared memory
//   - Coalesced write from shared memory to dst (transposed position)
//   - Shared memory padded by 1 column to avoid bank conflicts
//
// FpElement is 32 bytes (8 x uint32_t). At TILE=16, shmem per block =
// 16 * 17 * 32 = 8704 bytes (well within 48 KB limit).
// We use TILE=16 because FpElement is large (32B); TILE=32 would use 35 KB.

static constexpr int TRANSPOSE_TILE = 16;
static constexpr int TRANSPOSE_BLOCK = 256;  // 16x16 threads

__global__ void ntt_transpose_kernel(
    const FpElement* __restrict__ src,
    FpElement* __restrict__       dst,
    uint32_t rows,    // n1 (src rows)
    uint32_t cols     // n2 (src cols)
) {
    // Shared memory: TILE x (TILE+1) to avoid bank conflicts
    __shared__ FpElement tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    // Block position in tile grid
    const uint32_t bx = blockIdx.x;  // tile column index
    const uint32_t by = blockIdx.y;  // tile row index
    const uint32_t tx = threadIdx.x % TRANSPOSE_TILE;
    const uint32_t ty = threadIdx.x / TRANSPOSE_TILE;

    // Source coordinates (row-major: src[row * cols + col])
    const uint32_t src_col = bx * TRANSPOSE_TILE + tx;
    const uint32_t src_row = by * TRANSPOSE_TILE + ty;

    // Load tile from source (coalesced reads along columns)
    if (src_row < rows && src_col < cols) {
        tile[ty][tx] = src[src_row * cols + src_col];
    }

    __syncthreads();

    // Destination coordinates: transposed (cols x rows matrix)
    // dst[col * rows + row] = src[row * cols + col]
    // After transpose, we write tile rows as columns
    const uint32_t dst_col = by * TRANSPOSE_TILE + tx;  // was src row
    const uint32_t dst_row = bx * TRANSPOSE_TILE + ty;  // was src col

    if (dst_row < cols && dst_col < rows) {
        dst[dst_row * rows + dst_col] = tile[tx][ty];
    }
}

// ─── Batched Transpose ───────────────────────────────────────────────────────
// Transpose B independent matrices, each n1 x n2, stored contiguously.
// src[b * n1 * n2 + ...] → dst[b * n1 * n2 + ...]

__global__ void ntt_transpose_batch_kernel(
    const FpElement* __restrict__ src,
    FpElement* __restrict__       dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t batch_size
) {
    __shared__ FpElement tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    const uint32_t bx = blockIdx.x;
    const uint32_t by = blockIdx.y;
    const uint32_t bz = blockIdx.z;  // batch index
    const uint32_t tx = threadIdx.x % TRANSPOSE_TILE;
    const uint32_t ty = threadIdx.x / TRANSPOSE_TILE;

    if (bz >= batch_size) return;

    const uint32_t mat_size = rows * cols;
    const uint32_t base = bz * mat_size;

    const uint32_t src_col = bx * TRANSPOSE_TILE + tx;
    const uint32_t src_row = by * TRANSPOSE_TILE + ty;

    if (src_row < rows && src_col < cols) {
        tile[ty][tx] = src[base + src_row * cols + src_col];
    }

    __syncthreads();

    const uint32_t dst_col = by * TRANSPOSE_TILE + tx;
    const uint32_t dst_row = bx * TRANSPOSE_TILE + ty;

    if (dst_row < cols && dst_col < rows) {
        dst[base + dst_row * rows + dst_col] = tile[tx][ty];
    }
}

// ─── Twiddle Multiply Kernel ─────────────────────────────────────────────────
// Pointwise multiplication: data[i*n2 + j] *= twiddle_4step[i*n2 + j]
// where twiddle_4step[i*n2 + j] = omega_n^(i*j)
//
// For inverse NTT: uses omega_n^(-i*j) (conjugate twiddles)

__global__ void ntt_twiddle_multiply_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles_4step,
    uint32_t total_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    data[idx] = ff_mul_barrett(data[idx], twiddles_4step[idx]);
}

// Batched twiddle multiply: B matrices, each n1*n2 elements.
// All B matrices use the same twiddle table (twiddles are size n1*n2).
__global__ void ntt_twiddle_multiply_batch_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles_4step,
    uint32_t n,            // n1 * n2 (single NTT size)
    uint32_t total_elements // batch_size * n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    uint32_t tw_idx = idx % n;  // twiddle index wraps per NTT
    data[idx] = ff_mul_barrett(data[idx], twiddles_4step[tw_idx]);
}

// ─── Montgomery Twiddle Multiply Kernels ─────────────────────────────────────

__global__ void ntt_twiddle_multiply_mont_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles_4step,
    uint32_t total_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    data[idx] = ff_mul(data[idx], twiddles_4step[idx]);
}

__global__ void ntt_twiddle_multiply_batch_mont_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles_4step,
    uint32_t n,
    uint32_t total_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    uint32_t tw_idx = idx % n;
    data[idx] = ff_mul(data[idx], twiddles_4step[tw_idx]);
}

// ─── Host Helper Functions ───────────────────────────────────────────────────

static __host__ int log2_4step(uint32_t n) {
    int r = 0;
    while (n > 1) { n >>= 1; ++r; }
    return r;
}

// Compute the n1, n2 decomposition for 4-step NTT.
// Strategy: prefer balanced split (n1 ~ n2 ~ sqrt(n)).
// Both n1 and n2 must be powers of 2.
// n1 = number of rows, n2 = number of columns.
// Sub-NTTs are of size n1 (column NTTs) and n2 (row NTTs).
static __host__ void compute_4step_split(uint32_t n, uint32_t& n1, uint32_t& n2) {
    int log_n = log2_4step(n);
    assert(log_n >= 2);

    // Balanced split: n1 = 2^(log_n/2), n2 = 2^(log_n - log_n/2)
    int log_n1 = log_n / 2;
    int log_n2 = log_n - log_n1;

    n1 = 1u << log_n1;
    n2 = 1u << log_n2;

    assert(n1 * n2 == n);
}

// ─── Transpose Host Launch ───────────────────────────────────────────────────

static __host__ void launch_transpose(
    const FpElement* src, FpElement* dst,
    uint32_t rows, uint32_t cols,
    cudaStream_t stream
) {
    dim3 grid(
        (cols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (rows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE
    );
    dim3 block(TRANSPOSE_TILE * TRANSPOSE_TILE);  // 256 threads

    ntt_transpose_kernel<<<grid, block, 0, stream>>>(src, dst, rows, cols);
}

static __host__ void launch_transpose_batch(
    const FpElement* src, FpElement* dst,
    uint32_t rows, uint32_t cols, uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 grid(
        (cols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (rows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        batch_size
    );
    dim3 block(TRANSPOSE_TILE * TRANSPOSE_TILE);

    ntt_transpose_batch_kernel<<<grid, block, 0, stream>>>(
        src, dst, rows, cols, batch_size);
}

// ─── 4-Step Twiddle Cache ────────────────────────────────────────────────────
// Precomputed omega_n^(i*j) for the middle twiddle multiply step.
// Separate caches for Barrett (standard form) and Montgomery.

// Forward declarations from ntt_naive.cu (linked via separable compilation)
extern void ntt_forward_optimized_barrett(
    FpElement* d_data, size_t n, const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_optimized_barrett(
    FpElement* d_data, size_t n, const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);
extern void ntt_forward_optimized_montgomery(
    FpElement* d_data, size_t n, const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_optimized_montgomery(
    FpElement* d_data, size_t n, const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);

// Batched sub-NTT dispatch (from ntt_optimized.cu)
extern void ntt_forward_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_batch_barrett(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);
extern void ntt_forward_batch_montgomery(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_twiddles, cudaStream_t stream);
extern void ntt_inverse_batch_montgomery(
    FpElement* d_data, int batch_size, size_t n,
    const FpElement* d_inv_twiddles, FpElement n_inv, cudaStream_t stream);

// Extern conversion kernels from ntt_naive.cu
extern __global__ void ntt_to_montgomery_kernel(FpElement* data, uint32_t n);
extern __global__ void ntt_from_montgomery_kernel(FpElement* data, uint32_t n);

// ─── 4-Step Forward NTT (Barrett, standard-form) ─────────────────────────────
//
// Input: d_data[0..n-1] in standard form
// Output: NTT(d_data) in standard form
//
// Steps:
//   1. Interpret as n1 x n2 row-major matrix
//   2. Transpose to n2 x n1 → column NTTs become contiguous row sub-NTTs
//   3. Batch n2 independent n1-point sub-NTTs (column NTTs on transposed data)
//   4. Transpose back to n1 x n2
//   5. Twiddle multiply: data[i*n2+j] *= omega_n^(i*j)
//   6. Transpose to n2 x n1
//   7. Batch n1 independent n2-point sub-NTTs (row NTTs on transposed data)
//   8. Transpose back to n1 x n2 (final output in original layout)
//
// Optimization: steps 4-6 can be fused to reduce transposes. For now,
// keep them separate for correctness — fuse in Session 6/7 after validation.
//
// Temporary buffer: one n-element buffer for out-of-place transpose.

void ntt_4step_forward_barrett(
    FpElement* d_data, size_t n,
    const FpElement* d_twiddles_sub,       // sub-NTT twiddles (size max(n1,n2)/2)
    const FpElement* d_twiddles_4step,     // 4-step twiddles (size n1*n2 = n)
    FpElement n1_inv,                      // n1^{-1} for sub-NTT inverse (unused for forward)
    cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    uint32_t n1, n2;
    compute_4step_split(N, n1, n2);

    static constexpr uint32_t BLOCK = 256;

    // Allocate temporary buffer for transpose
    FpElement* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(FpElement)));

    // Step 1-2: Transpose n1 x n2 → n2 x n1 (columns become rows)
    launch_transpose(d_data, d_tmp, n1, n2, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    // Step 3: Batch n2 independent n1-point sub-NTTs (Barrett)
    // Data layout: n2 contiguous sub-arrays of size n1
    ntt_forward_batch_barrett(d_data, static_cast<int>(n2), n1, d_twiddles_sub, stream);

    // Step 4: Transpose back n2 x n1 → n1 x n2
    launch_transpose(d_data, d_tmp, n2, n1, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    // Step 5: Twiddle multiply: data[i*n2+j] *= omega_n^(i*j)
    {
        uint32_t grid = (N + BLOCK - 1) / BLOCK;
        ntt_twiddle_multiply_kernel<<<grid, BLOCK, 0, stream>>>(
            d_data, d_twiddles_4step, N);
    }

    // Step 6: Transpose n1 x n2 → n2 x n1
    launch_transpose(d_data, d_tmp, n1, n2, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    // Step 7: Batch n1 independent n2-point sub-NTTs (Barrett)
    ntt_forward_batch_barrett(d_data, static_cast<int>(n1), n2, d_twiddles_sub, stream);

    // Step 8: Transpose back n2 x n1 → n1 x n2 (restore original layout)
    // Actually for forward NTT, we want the output in the standard order.
    // The 4-step NTT output is naturally in transposed order after row NTTs.
    // Final transpose: n1 x n2 layout restored.
    launch_transpose(d_data, d_tmp, static_cast<uint32_t>(n1), n2, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaFree(d_tmp));
}

// ─── 4-Step Inverse NTT (Barrett, standard-form) ─────────────────────────────
// Mirror of forward: same structure with inverse twiddles and n^{-1} scaling.

void ntt_4step_inverse_barrett(
    FpElement* d_data, size_t n,
    const FpElement* d_inv_twiddles_sub,   // inverse sub-NTT twiddles
    const FpElement* d_inv_twiddles_4step, // inverse 4-step twiddles: omega_n^(-i*j)
    FpElement n1_inv,                      // n1^{-1} for sub-NTT scaling
    FpElement n2_inv,                      // n2^{-1} for sub-NTT scaling
    FpElement n_inv,                       // n^{-1} = (n1*n2)^{-1} for final scaling
    cudaStream_t stream
) {
    const uint32_t N = static_cast<uint32_t>(n);
    uint32_t n1, n2;
    compute_4step_split(N, n1, n2);

    static constexpr uint32_t BLOCK = 256;

    FpElement* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(FpElement)));

    // Step 1-2: Transpose n1 x n2 → n2 x n1
    launch_transpose(d_data, d_tmp, n1, n2, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    // Step 3: Batch n2 independent n1-point inverse sub-NTTs
    ntt_inverse_batch_barrett(d_data, static_cast<int>(n2), n1,
                              d_inv_twiddles_sub, n1_inv, stream);

    // Step 4: Transpose back n2 x n1 → n1 x n2
    launch_transpose(d_data, d_tmp, n2, n1, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    // Step 5: Inverse twiddle multiply: data[i*n2+j] *= omega_n^(-i*j)
    {
        uint32_t grid = (N + BLOCK - 1) / BLOCK;
        ntt_twiddle_multiply_kernel<<<grid, BLOCK, 0, stream>>>(
            d_data, d_inv_twiddles_4step, N);
    }

    // Step 6: Transpose n1 x n2 → n2 x n1
    launch_transpose(d_data, d_tmp, n1, n2, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    // Step 7: Batch n1 independent n2-point inverse sub-NTTs
    ntt_inverse_batch_barrett(d_data, static_cast<int>(n1), n2,
                              d_inv_twiddles_sub, n2_inv, stream);

    // Step 8: Transpose back
    launch_transpose(d_data, d_tmp, static_cast<uint32_t>(n1), n2, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_tmp, N * sizeof(FpElement),
                                cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaFree(d_tmp));
}

// ─── Expose transpose for testing ────────────────────────────────────────────

void ntt_transpose(
    const FpElement* d_src, FpElement* d_dst,
    uint32_t rows, uint32_t cols,
    cudaStream_t stream
) {
    launch_transpose(d_src, d_dst, rows, cols, stream);
}

void ntt_transpose_batch(
    const FpElement* d_src, FpElement* d_dst,
    uint32_t rows, uint32_t cols, uint32_t batch_size,
    cudaStream_t stream
) {
    launch_transpose_batch(d_src, d_dst, rows, cols, batch_size, stream);
}

// ─── Expose split computation for testing ────────────────────────────────────

void ntt_4step_get_split(uint32_t n, uint32_t& n1, uint32_t& n2) {
    compute_4step_split(n, n1, n2);
}
