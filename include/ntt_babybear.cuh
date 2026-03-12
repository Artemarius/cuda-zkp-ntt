// include/ntt_babybear.cuh — BabyBear NTT public interface
// NTT over the BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001
// TWO_ADICITY = 27, supports NTT sizes up to 2^27.
// All data in standard form (no Montgomery conversion).
#pragma once
#include "ff_babybear.cuh"
#include <cuda_runtime.h>
#include <cstddef>

// Forward NTT (in-place on device buffer)
void ntt_forward_babybear(BabyBearElement* d_data, size_t n, cudaStream_t stream = 0);

// Inverse NTT (in-place, scales by n^{-1})
void ntt_inverse_babybear(BabyBearElement* d_data, size_t n, cudaStream_t stream = 0);

// Batched NTT: process batch_size independent NTTs in parallel
// d_data layout: contiguous, d_data[b*n .. (b+1)*n - 1] is NTT #b
void ntt_forward_batch_babybear(BabyBearElement* d_data, int batch_size, size_t n, cudaStream_t stream = 0);
void ntt_inverse_batch_babybear(BabyBearElement* d_data, int batch_size, size_t n, cudaStream_t stream = 0);
