// include/pipeline.cuh — Async double-buffered NTT pipeline
// Phase 6: Direction A — eliminate CPU-GPU transfer bottleneck

#pragma once
#include "ff_arithmetic.cuh"
#include <cuda_runtime.h>
#include <cstddef>

// AsyncNTTPipeline: overlaps CPU→GPU transfer of batch k+1
// with GPU NTT computation of batch k using CUDA streams.
//
// Architecture (3-stream pipeline):
//   - Dedicated H2D stream, D2H stream, and per-slot compute streams
//   - Cross-stream events enforce H2D→compute→D2H dependencies
//   - With multiple copy engines (asyncEngineCount ≥ 2), H2D and D2H
//     can run simultaneously on separate DMA engines
//   - NTT compute overlaps with both H2D and D2H on the compute engine
//
// Usage:
//   AsyncNTTPipeline pipe(ntt_size);
//   pipe.process(h_input, h_output, total_n, ntt_size);
//
class AsyncNTTPipeline {
public:
    static constexpr int NUM_BUFFERS = 2;

    explicit AsyncNTTPipeline(size_t batch_n);
    ~AsyncNTTPipeline();

    // Process multiple NTT batches with compute/transfer overlap.
    // h_input and h_output can be regular (pageable) or pinned host memory.
    // total_n must be a multiple of ntt_size; any remainder is silently skipped.
    void process(
        const FpElement* h_input,   // all batches, host memory
        FpElement*       h_output,  // all batches, host memory
        size_t           total_n,   // total elements across all batches
        size_t           ntt_size   // size of each individual NTT (power of 2)
    );

    // Pipelined NTT for pinned host memory (cudaMallocHost).
    // Eliminates CPU staging copies — H2D/D2H transfer directly to/from
    // h_input/h_output. Requires both arrays to be page-locked.
    void process_pinned(
        const FpElement* h_input,   // pinned host memory (cudaMallocHost)
        FpElement*       h_output,  // pinned host memory (cudaMallocHost)
        size_t           total_n,
        size_t           ntt_size
    );

    // Non-pipelined reference: same work but serialized on a single stream.
    // Used as baseline for benchmark comparison (no compute/transfer overlap).
    void process_sequential(
        const FpElement* h_input,
        FpElement*       h_output,
        size_t           total_n,
        size_t           ntt_size
    );

private:
    size_t       batch_n_;
    FpElement*   d_buf_[NUM_BUFFERS];
    FpElement*   h_pinned_[NUM_BUFFERS];

    // 3-stream pipeline: separate H2D, compute, D2H streams
    cudaStream_t stream_h2d_;                   // all H2D transfers (DMA engine 0)
    cudaStream_t stream_compute_[NUM_BUFFERS];  // NTT compute (double-buffered)
    cudaStream_t stream_d2h_;                   // all D2H transfers (DMA engine 1)

    // Cross-stream synchronization events (per-slot, reused each pipeline cycle)
    cudaEvent_t  h2d_done_[NUM_BUFFERS];        // H2D complete → compute can start
    cudaEvent_t  compute_done_[NUM_BUFFERS];    // NTT complete → D2H can start
    cudaEvent_t  d2h_done_[NUM_BUFFERS];        // D2H complete → CPU can drain

    FpElement*   d_twiddles_;

    void allocate();
    void free_resources();
};
