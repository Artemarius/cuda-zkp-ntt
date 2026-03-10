// benchmarks/bench_ntt.cu
// Google Benchmark harness for NTT variants
// Phase 1: stub benchmarks. Real measurements in Phase 4+.

#include <benchmark/benchmark.h>
#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// ─── Stub Benchmarks ─────────────────────────────────────────────────────────

static void BM_NttForwardNaive(benchmark::State& state) {
    // TODO: Phase 4 — allocate device buffer, run ntt_forward(NAIVE)
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NttForwardNaive)->RangeMultiplier(4)->Range(1 << 15, 1 << 22)->Unit(benchmark::kMillisecond);

static void BM_NttForwardOptimized(benchmark::State& state) {
    // TODO: Phase 5 — radix-256 NTT benchmark
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NttForwardOptimized)->RangeMultiplier(4)->Range(1 << 15, 1 << 22)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
