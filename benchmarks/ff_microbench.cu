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

// SoA kernel declarations
extern __global__ void ff_add_soa_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ out, uint32_t n);
extern __global__ void ff_sub_soa_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ out, uint32_t n);
extern __global__ void ff_mul_soa_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_soa_kernel(const uint32_t* __restrict__ a, uint32_t* __restrict__ out, uint32_t n);

// v2 (branchless reduction) kernel declarations
extern __global__ void ff_mul_v2_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_add_v2_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sub_v2_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_v2_kernel(const FpElement* __restrict__ a, FpElement* __restrict__ out, uint32_t n);

// Barrett kernel declarations
extern __global__ void ff_mul_barrett_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_barrett_kernel(const FpElement* __restrict__ a, FpElement* __restrict__ out, uint32_t n);

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

// ─── AoS → SoA transpose on host ───────────────────────────────────────────
// Output layout: soa[limb * n + element_idx]

static void aos_to_soa(const std::vector<FpElement>& aos,
                        std::vector<uint32_t>& soa, uint32_t n) {
    soa.resize(8u * n);
    for (uint32_t i = 0; i < n; ++i) {
        for (int j = 0; j < 8; ++j) {
            soa[j * n + i] = aos[i].limbs[j];
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

// ═══════════════════════════════════════════════════════════════════════════
// v2 benchmarks — branchless conditional reduction (PTX sub.cc + lop3)
// ═══════════════════════════════════════════════════════════════════════════

static void BM_FfMul_v2(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    for (auto _ : state) {
        ff_mul_v2_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfMul_v2)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22);

static void BM_FfAdd_v2(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    for (auto _ : state) {
        ff_add_v2_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfAdd_v2)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22);

static void BM_FfSub_v2(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    for (auto _ : state) {
        ff_sub_v2_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSub_v2)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22);

static void BM_FfSqr_v2(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    for (auto _ : state) {
        ff_sqr_v2_kernel<<<gridSize, blockSize>>>(d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSqr_v2)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22);

// ═══════════════════════════════════════════════════════════════════════════
// SoA (Structure-of-Arrays) benchmarks — coalesced memory access pattern
// ═══════════════════════════════════════════════════════════════════════════

static void BM_FfMul_SoA(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    std::vector<uint32_t> soa_a, soa_b;
    aos_to_soa(h_a, soa_a, N);
    aos_to_soa(h_b, soa_b, N);

    uint32_t *d_a, *d_b, *d_out;
    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, soa_b.data(), soa_bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    for (auto _ : state) {
        ff_mul_soa_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfMul_SoA)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

static void BM_FfAdd_SoA(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    std::vector<uint32_t> soa_a, soa_b;
    aos_to_soa(h_a, soa_a, N);
    aos_to_soa(h_b, soa_b, N);

    uint32_t *d_a, *d_b, *d_out;
    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, soa_b.data(), soa_bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    for (auto _ : state) {
        ff_add_soa_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfAdd_SoA)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

static void BM_FfSub_SoA(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    std::vector<uint32_t> soa_a, soa_b;
    aos_to_soa(h_a, soa_a, N);
    aos_to_soa(h_b, soa_b, N);

    uint32_t *d_a, *d_b, *d_out;
    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, soa_b.data(), soa_bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    for (auto _ : state) {
        ff_sub_soa_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSub_SoA)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

static void BM_FfSqr_SoA(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));

    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    std::vector<uint32_t> soa_a;
    aos_to_soa(h_a, soa_a, N);

    uint32_t *d_a, *d_out;
    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    for (auto _ : state) {
        ff_sqr_soa_kernel<<<gridSize, blockSize>>>(d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSqr_SoA)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)
    ->Arg(1 << 18)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

// ═══════════════════════════════════════════════════════════════════════════
// Barrett benchmarks — standard-form modular multiplication (no Montgomery)
// ═══════════════════════════════════════════════════════════════════════════

static void BM_FfMul_Barrett(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    for (auto _ : state) {
        ff_mul_barrett_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfMul_Barrett)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22);

static void BM_FfSqr_Barrett(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<FpElement> h_a, h_b;
    fill_test_data(h_a, h_b, N);

    FpElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    for (auto _ : state) {
        ff_sqr_barrett_kernel<<<gridSize, blockSize>>>(d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_FfSqr_Barrett)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1 << 16)->Arg(1 << 18)->Arg(1 << 20)->Arg(1 << 22);

// ═══════════════════════════════════════════════════════════════════════════
// Goldilocks (64-bit, p = 2^64 - 2^32 + 1) benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#include "ff_goldilocks.cuh"
#include "ff_babybear.cuh"

extern __global__ void gl_add_kernel(const GoldilocksElement* __restrict__ a, const GoldilocksElement* __restrict__ b, GoldilocksElement* __restrict__ out, uint32_t n);
extern __global__ void gl_sub_kernel(const GoldilocksElement* __restrict__ a, const GoldilocksElement* __restrict__ b, GoldilocksElement* __restrict__ out, uint32_t n);
extern __global__ void gl_mul_kernel(const GoldilocksElement* __restrict__ a, const GoldilocksElement* __restrict__ b, GoldilocksElement* __restrict__ out, uint32_t n);
extern __global__ void gl_sqr_kernel(const GoldilocksElement* __restrict__ a, GoldilocksElement* __restrict__ out, uint32_t n);

extern __global__ void bb_add_kernel(const BabyBearElement* __restrict__ a, const BabyBearElement* __restrict__ b, BabyBearElement* __restrict__ out, uint32_t n);
extern __global__ void bb_sub_kernel(const BabyBearElement* __restrict__ a, const BabyBearElement* __restrict__ b, BabyBearElement* __restrict__ out, uint32_t n);
extern __global__ void bb_mul_kernel(const BabyBearElement* __restrict__ a, const BabyBearElement* __restrict__ b, BabyBearElement* __restrict__ out, uint32_t n);
extern __global__ void bb_sqr_kernel(const BabyBearElement* __restrict__ a, BabyBearElement* __restrict__ out, uint32_t n);

static void fill_gl_data(std::vector<GoldilocksElement>& h_a,
                         std::vector<GoldilocksElement>& h_b, uint32_t n) {
    h_a.resize(n); h_b.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        h_a[i].val = static_cast<uint64_t>(i) * 0x123456789ABCULL + 1;
        h_b[i].val = static_cast<uint64_t>(i) * 0xDEADBEEF1234ULL + 3;
    }
}

static void fill_bb_data(std::vector<BabyBearElement>& h_a,
                         std::vector<BabyBearElement>& h_b, uint32_t n) {
    h_a.resize(n); h_b.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        h_a[i].val = (i * 123457u + 1u) % 0x78000001u;
        h_b[i].val = (i * 654321u + 3u) % 0x78000001u;
    }
}

// ─── Goldilocks Add ─────────────────────────────────────────────────────────
static void BM_GlAdd(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<GoldilocksElement> h_a, h_b;
    fill_gl_data(h_a, h_b, N);

    GoldilocksElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    const int bs = 256, gs = (N + bs - 1) / bs;
    for (auto _ : state) {
        gl_add_kernel<<<gs, bs>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_GlAdd)->Unit(benchmark::kMicrosecond)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)->Arg(1<<22);

// ─── Goldilocks Mul ─────────────────────────────────────────────────────────
static void BM_GlMul(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<GoldilocksElement> h_a, h_b;
    fill_gl_data(h_a, h_b, N);

    GoldilocksElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    const int bs = 256, gs = (N + bs - 1) / bs;
    for (auto _ : state) {
        gl_mul_kernel<<<gs, bs>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_GlMul)->Unit(benchmark::kMicrosecond)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)->Arg(1<<22);

// ─── Goldilocks Sqr ─────────────────────────────────────────────────────────
static void BM_GlSqr(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<GoldilocksElement> h_a, h_b;
    fill_gl_data(h_a, h_b, N);

    GoldilocksElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    const int bs = 256, gs = (N + bs - 1) / bs;
    for (auto _ : state) {
        gl_sqr_kernel<<<gs, bs>>>(d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_GlSqr)->Unit(benchmark::kMicrosecond)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)->Arg(1<<22);

// ═══════════════════════════════════════════════════════════════════════════
// BabyBear (31-bit, p = 2^31 - 2^27 + 1) benchmarks
// ═══════════════════════════════════════════════════════════════════════════

// ─── BabyBear Add ───────────────────────────────────────────────────────────
static void BM_BbAdd(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<BabyBearElement> h_a, h_b;
    fill_bb_data(h_a, h_b, N);

    BabyBearElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    const int bs = 256, gs = (N + bs - 1) / bs;
    for (auto _ : state) {
        bb_add_kernel<<<gs, bs>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_BbAdd)->Unit(benchmark::kMicrosecond)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)->Arg(1<<22);

// ─── BabyBear Mul ───────────────────────────────────────────────────────────
static void BM_BbMul(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<BabyBearElement> h_a, h_b;
    fill_bb_data(h_a, h_b, N);

    BabyBearElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    const int bs = 256, gs = (N + bs - 1) / bs;
    for (auto _ : state) {
        bb_mul_kernel<<<gs, bs>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_BbMul)->Unit(benchmark::kMicrosecond)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)->Arg(1<<22);

// ─── BabyBear Sqr ───────────────────────────────────────────────────────────
static void BM_BbSqr(benchmark::State& state) {
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    std::vector<BabyBearElement> h_a, h_b;
    fill_bb_data(h_a, h_b, N);

    BabyBearElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    const int bs = 256, gs = (N + bs - 1) / bs;
    for (auto _ : state) {
        bb_sqr_kernel<<<gs, bs>>>(d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_out));
}
BENCHMARK(BM_BbSqr)->Unit(benchmark::kMicrosecond)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)->Arg(1<<22);

// ─── Main ───────────────────────────────────────────────────────────────────

BENCHMARK_MAIN();
