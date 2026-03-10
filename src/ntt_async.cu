// src/ntt_async.cu
// Async double-buffered NTT pipeline (Direction A)
// Phase 6: overlaps CPU→GPU transfer with GPU NTT computation.
//
// 3-stream pipeline architecture:
//   stream_h2d_:           H2D transfers (uses DMA engine for host→device)
//   stream_compute_[0,1]:  NTT kernels (uses compute engine, double-buffered)
//   stream_d2h_:           D2H transfers (uses DMA engine for device→host)
//
// With asyncEngineCount >= 2 (RTX 3060 has 5), H2D and D2H can overlap
// on separate DMA engines. Both can overlap with NTT compute.
//
// Cross-stream dependencies enforced via cudaStreamWaitEvent:
//   H2D(k) done → NTT(k) starts on compute stream
//   NTT(k) done → D2H(k) starts on D2H stream
//
// Timeline (steady state):
//   H2D stream:  |--H2D(k)--|--H2D(k+1)--|--H2D(k+2)--|...
//   Compute[0]:       |--NTT(k)--|              |--NTT(k+2)--|...
//   Compute[1]:            |--NTT(k+1)--|              |--NTT(k+3)--|...
//   D2H stream:                 |--D2H(k)--|--D2H(k+1)--|...

#include "pipeline.cuh"
#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

#include <cstring>
#include <cassert>
#include <cstdio>

// ─── AsyncNTTPipeline implementation ─────────────────────────────────────────

AsyncNTTPipeline::AsyncNTTPipeline(size_t batch_n)
    : batch_n_(batch_n), d_twiddles_(nullptr),
      stream_h2d_(nullptr), stream_d2h_(nullptr)
{
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        d_buf_[i] = nullptr;
        h_pinned_[i] = nullptr;
        stream_compute_[i] = nullptr;
        h2d_done_[i] = nullptr;
        compute_done_[i] = nullptr;
        d2h_done_[i] = nullptr;
    }
    allocate();
}

AsyncNTTPipeline::~AsyncNTTPipeline() {
    free_resources();
}

void AsyncNTTPipeline::allocate() {
    const size_t bytes = batch_n_ * sizeof(FpElement);

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        CUDA_CHECK(cudaMalloc(&d_buf_[i], bytes));
        CUDA_CHECK(cudaMallocHost(&h_pinned_[i], bytes));
        CUDA_CHECK(cudaStreamCreate(&stream_compute_[i]));
        // Disable timing for lower sync overhead
        CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done_[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done_[i], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaStreamCreate(&stream_h2d_));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h_));
}

void AsyncNTTPipeline::process(
    const FpElement* h_input,
    FpElement*       h_output,
    size_t           total_n,
    size_t           ntt_size
) {
    assert(ntt_size <= batch_n_ && "ntt_size exceeds pipeline buffer capacity");
    assert(ntt_size >= 2 && (ntt_size & (ntt_size - 1)) == 0 && "ntt_size must be power of 2");

    const size_t num_batches = total_n / ntt_size;
    const size_t batch_bytes = ntt_size * sizeof(FpElement);

    if (num_batches == 0) return;

    // Pre-warm twiddle cache (synchronous, one-time cost)
    ntt_precompute_twiddles(ntt_size);

    // ─── 3-stream double-buffered pipeline loop ──────────────────────────────
    for (size_t k = 0; k < num_batches; ++k) {
        const int slot = static_cast<int>(k % NUM_BUFFERS);

        // Wait for previous D2H on this slot to complete, then drain result
        if (k >= NUM_BUFFERS) {
            CUDA_CHECK(cudaEventSynchronize(d2h_done_[slot]));
            memcpy(h_output + (k - NUM_BUFFERS) * ntt_size,
                   h_pinned_[slot], batch_bytes);
        }

        // Stage input into pinned buffer (CPU memcpy, fast)
        memcpy(h_pinned_[slot], h_input + k * ntt_size, batch_bytes);

        // ── H2D on dedicated H2D stream ──
        CUDA_CHECK(cudaMemcpyAsync(d_buf_[slot], h_pinned_[slot], batch_bytes,
                                   cudaMemcpyHostToDevice, stream_h2d_));
        CUDA_CHECK(cudaEventRecord(h2d_done_[slot], stream_h2d_));

        // ── NTT compute on slot's compute stream (waits for H2D) ──
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute_[slot], h2d_done_[slot]));
        ntt_forward(d_buf_[slot], ntt_size, NTTMode::OPTIMIZED, stream_compute_[slot]);
        CUDA_CHECK(cudaEventRecord(compute_done_[slot], stream_compute_[slot]));

        // ── D2H on dedicated D2H stream (waits for compute) ──
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h_, compute_done_[slot]));
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_[slot], d_buf_[slot], batch_bytes,
                                   cudaMemcpyDeviceToHost, stream_d2h_));
        CUDA_CHECK(cudaEventRecord(d2h_done_[slot], stream_d2h_));
    }

    // ─── Drain remaining batches (last NUM_BUFFERS worth) ────────────────────
    const size_t drain_start = (num_batches >= NUM_BUFFERS)
                             ? (num_batches - NUM_BUFFERS) : 0;
    for (size_t k = drain_start; k < num_batches; ++k) {
        const int slot = static_cast<int>(k % NUM_BUFFERS);
        CUDA_CHECK(cudaEventSynchronize(d2h_done_[slot]));
        memcpy(h_output + k * ntt_size, h_pinned_[slot], batch_bytes);
    }
}

void AsyncNTTPipeline::process_pinned(
    const FpElement* h_input,
    FpElement*       h_output,
    size_t           total_n,
    size_t           ntt_size
) {
    assert(ntt_size <= batch_n_ && "ntt_size exceeds pipeline buffer capacity");
    assert(ntt_size >= 2 && (ntt_size & (ntt_size - 1)) == 0);

    const size_t num_batches = total_n / ntt_size;
    const size_t batch_bytes = ntt_size * sizeof(FpElement);

    if (num_batches == 0) return;

    ntt_precompute_twiddles(ntt_size);

    // ─── 3-stream pipeline, no CPU staging (pinned memory direct) ────────────
    // H2D reads directly from h_input, D2H writes directly to h_output.
    // No intermediate pinned buffer copies needed.
    for (size_t k = 0; k < num_batches; ++k) {
        const int slot = static_cast<int>(k % NUM_BUFFERS);

        // Wait for previous D2H on this slot's device buffer to complete
        if (k >= NUM_BUFFERS) {
            CUDA_CHECK(cudaEventSynchronize(d2h_done_[slot]));
        }

        // H2D directly from pinned input
        CUDA_CHECK(cudaMemcpyAsync(d_buf_[slot],
                                   h_input + k * ntt_size, batch_bytes,
                                   cudaMemcpyHostToDevice, stream_h2d_));
        CUDA_CHECK(cudaEventRecord(h2d_done_[slot], stream_h2d_));

        // NTT compute (waits for H2D)
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute_[slot], h2d_done_[slot]));
        ntt_forward(d_buf_[slot], ntt_size, NTTMode::OPTIMIZED, stream_compute_[slot]);
        CUDA_CHECK(cudaEventRecord(compute_done_[slot], stream_compute_[slot]));

        // D2H directly to pinned output (waits for compute)
        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h_, compute_done_[slot]));
        CUDA_CHECK(cudaMemcpyAsync(h_output + k * ntt_size,
                                   d_buf_[slot], batch_bytes,
                                   cudaMemcpyDeviceToHost, stream_d2h_));
        CUDA_CHECK(cudaEventRecord(d2h_done_[slot], stream_d2h_));
    }

    // Wait for all remaining D2H to complete
    for (int slot = 0; slot < NUM_BUFFERS; ++slot) {
        CUDA_CHECK(cudaEventSynchronize(d2h_done_[slot]));
    }
}

void AsyncNTTPipeline::process_sequential(
    const FpElement* h_input,
    FpElement*       h_output,
    size_t           total_n,
    size_t           ntt_size
) {
    assert(ntt_size <= batch_n_ && "ntt_size exceeds pipeline buffer capacity");
    assert(ntt_size >= 2 && (ntt_size & (ntt_size - 1)) == 0);

    const size_t num_batches = total_n / ntt_size;
    const size_t batch_bytes = ntt_size * sizeof(FpElement);

    if (num_batches == 0) return;

    // Pre-warm twiddle cache
    ntt_precompute_twiddles(ntt_size);

    // ─── Single-stream sequential processing (no overlap) ────────────────────
    for (size_t k = 0; k < num_batches; ++k) {
        // Stage input into pinned buffer
        memcpy(h_pinned_[0], h_input + k * ntt_size, batch_bytes);

        // H2D, NTT, D2H all on one stream (serialized)
        CUDA_CHECK(cudaMemcpyAsync(d_buf_[0], h_pinned_[0], batch_bytes,
                                   cudaMemcpyHostToDevice, stream_h2d_));
        ntt_forward(d_buf_[0], ntt_size, NTTMode::OPTIMIZED, stream_h2d_);
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_[0], d_buf_[0], batch_bytes,
                                   cudaMemcpyDeviceToHost, stream_h2d_));

        // Wait for everything to finish before reusing pinned buffer
        CUDA_CHECK(cudaStreamSynchronize(stream_h2d_));

        // Drain result
        memcpy(h_output + k * ntt_size, h_pinned_[0], batch_bytes);
    }
}

void AsyncNTTPipeline::free_resources() {
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        if (d_buf_[i])         cudaFree(d_buf_[i]);
        if (h_pinned_[i])     cudaFreeHost(h_pinned_[i]);
        if (stream_compute_[i]) cudaStreamDestroy(stream_compute_[i]);
        if (h2d_done_[i])     cudaEventDestroy(h2d_done_[i]);
        if (compute_done_[i]) cudaEventDestroy(compute_done_[i]);
        if (d2h_done_[i])     cudaEventDestroy(d2h_done_[i]);
        d_buf_[i] = nullptr;
        h_pinned_[i] = nullptr;
        stream_compute_[i] = nullptr;
        h2d_done_[i] = nullptr;
        compute_done_[i] = nullptr;
        d2h_done_[i] = nullptr;
    }
    if (stream_h2d_)   { cudaStreamDestroy(stream_h2d_);  stream_h2d_ = nullptr; }
    if (stream_d2h_)   { cudaStreamDestroy(stream_d2h_);  stream_d2h_ = nullptr; }
    if (d_twiddles_)   { cudaFree(d_twiddles_);           d_twiddles_ = nullptr; }
}
