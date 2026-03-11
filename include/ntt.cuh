// include/ntt.cuh — NTT public interface
#pragma once
#include "ff_arithmetic.cuh"
#include <cuda_runtime.h>
#include <cstddef>

enum class NTTMode {
    NAIVE,      // radix-2, global memory only
    OPTIMIZED,  // radix-256, shared memory twiddles
    ASYNC,      // async double-buffered pipeline (Direction A)
    BARRETT     // Barrett arithmetic, no Montgomery conversion overhead
};

// Forward NTT: computes NTT in-place on device buffer
void ntt_forward(FpElement* d_data, size_t n, NTTMode mode = NTTMode::OPTIMIZED, cudaStream_t stream = 0);

// Inverse NTT: computes INTT in-place
void ntt_inverse(FpElement* d_data, size_t n, NTTMode mode = NTTMode::OPTIMIZED, cudaStream_t stream = 0);

// Batched NTT: process batch_size independent NTTs in parallel
// d_data layout: contiguous, d_data[b*n .. (b+1)*n - 1] is NTT #b
// All NTTs share the same twiddle factors (same size n)
// Supported modes: OPTIMIZED, BARRETT
void ntt_forward_batch(FpElement* d_data, int batch_size, size_t n,
                       NTTMode mode = NTTMode::OPTIMIZED, cudaStream_t stream = 0);
void ntt_inverse_batch(FpElement* d_data, int batch_size, size_t n,
                       NTTMode mode = NTTMode::OPTIMIZED, cudaStream_t stream = 0);

// Precompute twiddle factors on device (call once per NTT size)
FpElement* ntt_precompute_twiddles(size_t n);
void       ntt_free_twiddles(FpElement* d_twiddles);
