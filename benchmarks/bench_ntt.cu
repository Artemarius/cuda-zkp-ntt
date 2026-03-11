// benchmarks/bench_ntt.cu
// Google Benchmark harness for NTT variants
// Phase 4: naive/optimized NTT benchmarks.
// Phase 6: pipelined vs sequential (end-to-end including H2D/D2H transfers).

#include <benchmark/benchmark.h>
#include <chrono>
#include "ntt.cuh"
#include "pipeline.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// ─── Naive Radix-2 NTT Benchmark ────────────────────────────────────────────

static void BM_NttForwardNaive(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));

    // Allocate and zero-initialize device data
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(FpElement)));

    // Precompute twiddles outside the timing loop
    ntt_precompute_twiddles(n);

    // Warm up
    ntt_forward(d_data, n, NTTMode::NAIVE);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward(d_data, n, NTTMode::NAIVE);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.counters["elements"] = static_cast<double>(n);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
}
BENCHMARK(BM_NttForwardNaive)
    ->RangeMultiplier(4)
    ->Range(1 << 15, 1 << 22)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

static void BM_NttForwardOptimized(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(FpElement)));

    ntt_precompute_twiddles(n);

    // Warm up
    ntt_forward(d_data, n, NTTMode::OPTIMIZED);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward(d_data, n, NTTMode::OPTIMIZED);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.counters["elements"] = static_cast<double>(n);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
}
BENCHMARK(BM_NttForwardOptimized)
    ->RangeMultiplier(4)
    ->Range(1 << 15, 1 << 22)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// ─── Barrett NTT Benchmark ──────────────────────────────────────────────────
// No Montgomery conversion overhead — standard-form throughout.

static void BM_NttForwardBarrett(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(FpElement)));

    ntt_precompute_twiddles(n);

    // Warm up
    ntt_forward(d_data, n, NTTMode::BARRETT);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward(d_data, n, NTTMode::BARRETT);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.counters["elements"] = static_cast<double>(n);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
}
BENCHMARK(BM_NttForwardBarrett)
    ->RangeMultiplier(4)
    ->Range(1 << 15, 1 << 22)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// ─── Batched NTT Benchmarks ──────────────────────────────────────────────────
// Args: (ntt_size, batch_size)

static void BM_NttBatchOptimized(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    int batch_size = static_cast<int>(state.range(1));
    size_t total = static_cast<size_t>(batch_size) * n;

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, total * sizeof(FpElement)));

    ntt_precompute_twiddles(n);

    // Warm up
    ntt_forward_batch(d_data, batch_size, n, NTTMode::OPTIMIZED);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_batch(d_data, batch_size, n, NTTMode::OPTIMIZED);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total));
    state.counters["ntt_size"] = static_cast<double>(n);
    state.counters["batch"] = static_cast<double>(batch_size);
    state.counters["per_ntt_ms"] = benchmark::Counter(0, benchmark::Counter::kAvgIterations);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
}
BENCHMARK(BM_NttBatchOptimized)
    ->Args({1 << 15, 1})->Args({1 << 15, 4})->Args({1 << 15, 8})->Args({1 << 15, 16})
    ->Args({1 << 18, 1})->Args({1 << 18, 4})->Args({1 << 18, 8})
    ->Args({1 << 20, 1})->Args({1 << 20, 4})->Args({1 << 20, 8})
    ->Args({1 << 22, 1})->Args({1 << 22, 2})->Args({1 << 22, 4})->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

static void BM_NttBatchBarrett(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    int batch_size = static_cast<int>(state.range(1));
    size_t total = static_cast<size_t>(batch_size) * n;

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, total * sizeof(FpElement)));

    ntt_precompute_twiddles(n);

    // Warm up
    ntt_forward_batch(d_data, batch_size, n, NTTMode::BARRETT);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_batch(d_data, batch_size, n, NTTMode::BARRETT);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total));
    state.counters["ntt_size"] = static_cast<double>(n);
    state.counters["batch"] = static_cast<double>(batch_size);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
}
BENCHMARK(BM_NttBatchBarrett)
    ->Args({1 << 15, 1})->Args({1 << 15, 4})->Args({1 << 15, 8})->Args({1 << 15, 16})
    ->Args({1 << 18, 1})->Args({1 << 18, 4})->Args({1 << 18, 8})
    ->Args({1 << 20, 1})->Args({1 << 20, 4})->Args({1 << 20, 8})
    ->Args({1 << 22, 1})->Args({1 << 22, 2})->Args({1 << 22, 4})->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Sequential baseline: B separate ntt_forward calls
static void BM_NttSequentialLoop(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    int batch_size = static_cast<int>(state.range(1));

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, batch_size * n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, batch_size * n * sizeof(FpElement)));

    ntt_precompute_twiddles(n);

    // Warm up
    for (int b = 0; b < batch_size; ++b)
        ntt_forward(d_data + b * n, n, NTTMode::BARRETT);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int b = 0; b < batch_size; ++b)
            ntt_forward(d_data + b * n, n, NTTMode::BARRETT);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(ms / 1000.0);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size) * static_cast<int64_t>(n));
    state.counters["ntt_size"] = static_cast<double>(n);
    state.counters["batch"] = static_cast<double>(batch_size);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
}
BENCHMARK(BM_NttSequentialLoop)
    ->Args({1 << 15, 8})->Args({1 << 18, 8})
    ->Args({1 << 20, 8})->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// ─── Phase 6: Pipelined NTT Benchmark ────────────────────────────────────────
// Measures end-to-end latency including H2D + NTT compute + D2H transfers.
// Args: (ntt_size, num_batches)

static void BM_NttPipelined(benchmark::State& state) {
    size_t ntt_size    = static_cast<size_t>(state.range(0));
    int    num_batches = static_cast<int>(state.range(1));
    size_t total_n     = ntt_size * num_batches;

    std::vector<FpElement> h_input(total_n, FpElement::zero());
    std::vector<FpElement> h_output(total_n);

    AsyncNTTPipeline pipe(ntt_size);

    // Warm up (pre-warms twiddle cache + GPU caches)
    pipe.process(h_input.data(), h_output.data(), total_n, ntt_size);

    for (auto _ : state) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        pipe.process(h_input.data(), h_output.data(), total_n, ntt_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(
            std::chrono::duration<double>(t1 - t0).count());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_n));
    state.counters["ntt_size"]  = static_cast<double>(ntt_size);
    state.counters["batches"]   = static_cast<double>(num_batches);
    state.counters["total_MB"]  = static_cast<double>(total_n * sizeof(FpElement))
                                / (1024.0 * 1024.0);
}
BENCHMARK(BM_NttPipelined)
    ->Args({1 << 18, 8})
    ->Args({1 << 20, 8})
    ->Args({1 << 22, 4})
    ->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// ─── Sequential (Non-Pipelined) NTT Benchmark ───────────────────────────────
// Same work as pipelined but serialized on a single stream (no overlap).

static void BM_NttSequential(benchmark::State& state) {
    size_t ntt_size    = static_cast<size_t>(state.range(0));
    int    num_batches = static_cast<int>(state.range(1));
    size_t total_n     = ntt_size * num_batches;

    std::vector<FpElement> h_input(total_n, FpElement::zero());
    std::vector<FpElement> h_output(total_n);

    AsyncNTTPipeline pipe(ntt_size);

    // Warm up
    pipe.process_sequential(h_input.data(), h_output.data(), total_n, ntt_size);

    for (auto _ : state) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        pipe.process_sequential(h_input.data(), h_output.data(), total_n, ntt_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(
            std::chrono::duration<double>(t1 - t0).count());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_n));
    state.counters["ntt_size"]  = static_cast<double>(ntt_size);
    state.counters["batches"]   = static_cast<double>(num_batches);
    state.counters["total_MB"]  = static_cast<double>(total_n * sizeof(FpElement))
                                / (1024.0 * 1024.0);
}
BENCHMARK(BM_NttSequential)
    ->Args({1 << 18, 8})
    ->Args({1 << 20, 8})
    ->Args({1 << 22, 4})
    ->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// ─── Pinned-Memory Pipeline Benchmark ────────────────────────────────────────
// Best-case pipeline: input/output in pinned memory, no CPU staging copies.

static void BM_NttPipelinedPinned(benchmark::State& state) {
    size_t ntt_size    = static_cast<size_t>(state.range(0));
    int    num_batches = static_cast<int>(state.range(1));
    size_t total_n     = ntt_size * num_batches;
    size_t total_bytes = total_n * sizeof(FpElement);

    FpElement *h_input, *h_output;
    CUDA_CHECK(cudaMallocHost(&h_input, total_bytes));
    CUDA_CHECK(cudaMallocHost(&h_output, total_bytes));
    memset(h_input, 0, total_bytes);

    AsyncNTTPipeline pipe(ntt_size);
    pipe.process_pinned(h_input, h_output, total_n, ntt_size);

    for (auto _ : state) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        pipe.process_pinned(h_input, h_output, total_n, ntt_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(
            std::chrono::duration<double>(t1 - t0).count());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_n));
    state.counters["ntt_size"]  = static_cast<double>(ntt_size);
    state.counters["batches"]   = static_cast<double>(num_batches);
    state.counters["total_MB"]  = static_cast<double>(total_bytes) / (1024.0 * 1024.0);

    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
}
BENCHMARK(BM_NttPipelinedPinned)
    ->Args({1 << 18, 8})
    ->Args({1 << 20, 8})
    ->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// ─── Pinned-Memory Sequential Baseline ───────────────────────────────────────

static void BM_NttSequentialPinned(benchmark::State& state) {
    size_t ntt_size    = static_cast<size_t>(state.range(0));
    int    num_batches = static_cast<int>(state.range(1));
    size_t total_n     = ntt_size * num_batches;
    size_t total_bytes = total_n * sizeof(FpElement);

    FpElement *h_input, *h_output;
    CUDA_CHECK(cudaMallocHost(&h_input, total_bytes));
    CUDA_CHECK(cudaMallocHost(&h_output, total_bytes));
    memset(h_input, 0, total_bytes);

    AsyncNTTPipeline pipe(ntt_size);
    pipe.process_sequential(h_input, h_output, total_n, ntt_size);

    for (auto _ : state) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        pipe.process_sequential(h_input, h_output, total_n, ntt_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(
            std::chrono::duration<double>(t1 - t0).count());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(total_n));
    state.counters["ntt_size"]  = static_cast<double>(ntt_size);
    state.counters["batches"]   = static_cast<double>(num_batches);
    state.counters["total_MB"]  = static_cast<double>(total_bytes) / (1024.0 * 1024.0);

    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
}
BENCHMARK(BM_NttSequentialPinned)
    ->Args({1 << 18, 8})
    ->Args({1 << 20, 8})
    ->Args({1 << 22, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_MAIN();
