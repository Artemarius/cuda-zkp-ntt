// benchmarks/ff_microbench.cu
// Isolated finite-field operation throughput measurement
// Phase 2: GPU throughput benchmarks for FF add, sub, mul, sqr

#include <benchmark/benchmark.h>
#include <vector>
#include <cstdint>
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"

// ─── Extern Kernel Declarations ─────────────────────────────────────────────
// Kernels defined in src/ff_mul.cu (linked via zkp_ntt_core)

extern __global__ void ff_add_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

extern __global__ void ff_sub_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

extern __global__ void ff_mul_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

extern __global__ void ff_sqr_throughput_kernel(
    const FpElement* __restrict__ a,
    FpElement* __restrict__ out,
    uint32_t n);

// ─── Helper: fill host arrays with deterministic nonzero data ───────────────

static void fill_test_data(std::vector<FpElement>& h_a,
                           std::vector<FpElement>& h_b,
                           uint32_t n) {
    h_a.resize(n);
    h_b.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        for (int j = 0; j < 8; ++j) {
            h_a[i].limbs[j] = i * 8 + j + 1;
            h_b[i].limbs[j] = (i + 1) * 8 + j + 3;
        }
    }
}

// ─── BM_FfAdd ───────────────────────────────────────────────────────────────

static void BM_FfAdd(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    // Setup: allocate and initialize
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement* d_a = nullptr;
    FpElement* d_b = nullptr;
    FpElement* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Benchmark loop
    for (auto _ : state) {
        ff_add_throughput_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));

    // Teardown
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfAdd)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

// ─── BM_FfSub ───────────────────────────────────────────────────────────────

static void BM_FfSub(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    // Setup: allocate and initialize
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement* d_a = nullptr;
    FpElement* d_b = nullptr;
    FpElement* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Benchmark loop
    for (auto _ : state) {
        ff_sub_throughput_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));

    // Teardown
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSub)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

// ─── BM_FfMul ───────────────────────────────────────────────────────────────

static void BM_FfMul(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    // Setup: allocate and initialize
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement* d_a = nullptr;
    FpElement* d_b = nullptr;
    FpElement* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Benchmark loop
    for (auto _ : state) {
        ff_mul_throughput_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));

    // Teardown
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfMul)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

// ─── BM_FfSqr ──────────────────────────────────────────────────────────────

static void BM_FfSqr(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    // Setup: allocate and initialize (only need one input array for squaring)
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement* d_a = nullptr;
    FpElement* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Benchmark loop
    for (auto _ : state) {
        ff_sqr_throughput_kernel<<<gridSize, blockSize>>>(d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));

    // Teardown
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSqr)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

// ─── Main ───────────────────────────────────────────────────────────────────

BENCHMARK_MAIN();
