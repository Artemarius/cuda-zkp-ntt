// src/ntt_async.cu
// Async double-buffered NTT pipeline (Direction A)
// Phase 1: stub. Implementation in Phase 6.

#include "pipeline.cuh"
#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// ─── AsyncNTTPipeline stub implementation ────────────────────────────────────

AsyncNTTPipeline::AsyncNTTPipeline(size_t batch_n)
    : batch_n_(batch_n), d_twiddles_(nullptr)
{
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        d_buf_[i] = nullptr;
        h_pinned_[i] = nullptr;
        streams_[i] = nullptr;
        events_[i] = nullptr;
    }
    // TODO: Phase 6 — allocate()
}

AsyncNTTPipeline::~AsyncNTTPipeline() {
    free_resources();
}

void AsyncNTTPipeline::process(
    const FpElement* h_input,
    FpElement*       h_output,
    size_t           total_n,
    size_t           ntt_size
) {
    // TODO: Phase 6 — double-buffered pipeline loop
    fprintf(stderr, "AsyncNTTPipeline::process: stub — not yet implemented\n");
}

void AsyncNTTPipeline::allocate() {
    // TODO: Phase 6
}

void AsyncNTTPipeline::free_resources() {
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        if (d_buf_[i])    cudaFree(d_buf_[i]);
        if (h_pinned_[i]) cudaFreeHost(h_pinned_[i]);
        if (streams_[i])  cudaStreamDestroy(streams_[i]);
        if (events_[i])   cudaEventDestroy(events_[i]);
        d_buf_[i] = nullptr;
        h_pinned_[i] = nullptr;
        streams_[i] = nullptr;
        events_[i] = nullptr;
    }
    if (d_twiddles_) {
        cudaFree(d_twiddles_);
        d_twiddles_ = nullptr;
    }
}
