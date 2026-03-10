// benchmarks/ff_microbench.cu
// Isolated finite-field operation throughput measurement
// Phase 1: stub. Real FF benchmarks in Phase 2.

#include <benchmark/benchmark.h>
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// ─── Stub Benchmarks ─────────────────────────────────────────────────────────

static void BM_FfAdd(benchmark::State& state) {
    // TODO: Phase 2 — launch ff_add_throughput_kernel, measure
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
}
BENCHMARK(BM_FfAdd)->Unit(benchmark::kMicrosecond);

static void BM_FfMul(benchmark::State& state) {
    // TODO: Phase 2 — launch ff_mul_throughput_kernel, measure
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
}
BENCHMARK(BM_FfMul)->Unit(benchmark::kMicrosecond);

static void BM_FfSqr(benchmark::State& state) {
    // TODO: Phase 2
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
}
BENCHMARK(BM_FfSqr)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
