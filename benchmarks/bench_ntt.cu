// benchmarks/bench_ntt.cu
// Google Benchmark harness for NTT variants
// Phase 4: naive radix-2 NTT benchmark.

#include <benchmark/benchmark.h>
#include "ntt.cuh"
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
    // TODO: Phase 5 — radix-256 NTT benchmark
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NttForwardOptimized)
    ->RangeMultiplier(4)
    ->Range(1 << 15, 1 << 22)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
