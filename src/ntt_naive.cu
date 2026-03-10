// src/ntt_naive.cu
// Radix-2 NTT implementation (global memory only) + public API dispatch
// Phase 1: stubs. Radix-2 kernel in Phase 4. Dispatch filled per phase.

#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"
#include <cstdio>

// ─── Public API ──────────────────────────────────────────────────────────────
// Single dispatch point for all NTT modes. Each mode's implementation
// is in its respective .cu file; here we dispatch and provide the naive path.

void ntt_forward(FpElement* d_data, size_t n, NTTMode mode, cudaStream_t stream) {
    switch (mode) {
        case NTTMode::NAIVE:
            // TODO: Phase 4 — radix-2 Cooley-Tukey kernel
            fprintf(stderr, "ntt_forward(NAIVE): stub — not yet implemented\n");
            break;
        case NTTMode::OPTIMIZED:
            // TODO: Phase 5 — radix-256 shared-memory kernel
            fprintf(stderr, "ntt_forward(OPTIMIZED): stub — not yet implemented\n");
            break;
        case NTTMode::ASYNC:
            // TODO: Phase 6 — async pipeline (uses pipeline.cuh)
            fprintf(stderr, "ntt_forward(ASYNC): stub — not yet implemented\n");
            break;
    }
}

void ntt_inverse(FpElement* d_data, size_t n, NTTMode mode, cudaStream_t stream) {
    switch (mode) {
        case NTTMode::NAIVE:
            fprintf(stderr, "ntt_inverse(NAIVE): stub — not yet implemented\n");
            break;
        case NTTMode::OPTIMIZED:
            fprintf(stderr, "ntt_inverse(OPTIMIZED): stub — not yet implemented\n");
            break;
        case NTTMode::ASYNC:
            fprintf(stderr, "ntt_inverse(ASYNC): stub — not yet implemented\n");
            break;
    }
}

FpElement* ntt_precompute_twiddles(size_t n) {
    // TODO: Phase 4 — compute twiddle table on device
    fprintf(stderr, "ntt_precompute_twiddles: stub — not yet implemented\n");
    return nullptr;
}

void ntt_free_twiddles(FpElement* d_twiddles) {
    if (d_twiddles) {
        CUDA_CHECK(cudaFree(d_twiddles));
    }
}
