// include/pipeline.cuh — Async double-buffered NTT pipeline
// Phase 6 stub

#pragma once
#include "ff_arithmetic.cuh"
#include <cuda_runtime.h>
#include <cstddef>

// AsyncNTTPipeline: overlaps CPU→GPU transfer of batch k+1
// with GPU NTT computation of batch k using CUDA streams.
//
// Usage:
//   AsyncNTTPipeline pipe(batch_size);
//   pipe.process(h_input, h_output, total_n, ntt_size);
//
class AsyncNTTPipeline {
public:
    static constexpr int NUM_BUFFERS = 2;

    explicit AsyncNTTPipeline(size_t batch_n);
    ~AsyncNTTPipeline();

    // Process multiple NTT batches with compute/transfer overlap.
    // h_input and h_output must be pinned (cudaMallocHost) or regular host memory.
    void process(
        const FpElement* h_input,   // all batches, host memory
        FpElement*       h_output,  // all batches, host memory
        size_t           total_n,   // total elements across all batches
        size_t           ntt_size   // size of each individual NTT
    );

private:
    size_t       batch_n_;
    FpElement*   d_buf_[NUM_BUFFERS];
    FpElement*   h_pinned_[NUM_BUFFERS];
    cudaStream_t streams_[NUM_BUFFERS];
    cudaEvent_t  events_[NUM_BUFFERS];
    FpElement*   d_twiddles_;

    void allocate();
    void free_resources();
};
