cmake_minimum_required(VERSION 3.20)

project(cuda_zkp_ntt
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

# ─── Standards ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── CUDA Architecture ────────────────────────────────────────────────────────
# sm_86 = RTX 3060 / RTX 3090 (Ampere)
# Override with: cmake -DCMAKE_CUDA_ARCHITECTURES=75 for Turing
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# ─── Compiler Flags ───────────────────────────────────────────────────────────
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Suppress warnings from third-party headers (Google Benchmark) in nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=940")

# ─── Dependencies ─────────────────────────────────────────────────────────────
include(FetchContent)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ─── Include Directories ──────────────────────────────────────────────────────
include_directories(include)

# ─── Fused NTT Kernels (NO RDC) ──────────────────────────────────────────────
# Compiled WITHOUT separable compilation to work around the MSVC+CUDA 12.8
# RDC template symbol resolution bug for __global__ template functions with K>=9.
# All device functions (ff_mul, ff_add, ff_sub) are __device__ __forceinline__
# in the header, so they get inlined — no cross-TU device linkage needed.
add_library(ntt_fused_lib STATIC
    src/ntt_fused_kernels.cu
)
set_target_properties(ntt_fused_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# ─── Core Library ─────────────────────────────────────────────────────────────
# Finite-field arithmetic and NTT kernels compiled as a static library
add_library(zkp_ntt_core STATIC
    src/ff_mul.cu
    src/ff_multi_field.cu
    src/ntt_naive.cu
    src/ntt_optimized.cu
    src/ntt_async.cu
    src/ntt_4step.cu
    src/ntt_goldilocks.cu
    src/ntt_babybear.cu
)
set_target_properties(zkp_ntt_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
# Link fused kernel library (compiled without RDC)
target_link_libraries(zkp_ntt_core PUBLIC ntt_fused_lib)
# ntt_naive.cu uses ff_reference.h for CPU-side twiddle computation
target_include_directories(zkp_ntt_core PRIVATE ${CMAKE_SOURCE_DIR}/tests)

# ─── Correctness Tests ────────────────────────────────────────────────────────
add_executable(test_correctness
    tests/test_correctness.cu
)
target_link_libraries(test_correctness PRIVATE zkp_ntt_core)
set_target_properties(test_correctness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ─── Google Benchmark Harness ─────────────────────────────────────────────────
add_executable(bench_ntt
    benchmarks/bench_ntt.cu
)
target_link_libraries(bench_ntt PRIVATE zkp_ntt_core benchmark::benchmark)
set_target_properties(bench_ntt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ─── Finite-Field Microbenchmark ──────────────────────────────────────────────
add_executable(ff_microbench
    benchmarks/ff_microbench.cu
)
target_link_libraries(ff_microbench PRIVATE zkp_ntt_core benchmark::benchmark)
set_target_properties(ff_microbench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ─── Multi-Field NTT Benchmark ───────────────────────────────────────────────
add_executable(bench_multifield
    benchmarks/bench_multifield.cu
)
target_link_libraries(bench_multifield PRIVATE zkp_ntt_core)
set_target_properties(bench_multifield PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ─── Clean Profiling Binary ───────────────────────────────────────────────────
# Minimal binary with command-line mode/size args for Nsight Compute
add_executable(ntt_profile
    src/benchmark.cu
)
target_link_libraries(ntt_profile PRIVATE zkp_ntt_core)
set_target_properties(ntt_profile PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ─── Print Configuration Summary ──────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== cuda-zkp-ntt build configuration ===")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard:    ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA flags:       ${CMAKE_CUDA_FLAGS}")
message(STATUS "")
message(STATUS "  Targets:")
message(STATUS "    test_correctness  — NTT correctness validation")
message(STATUS "    bench_ntt         — Google Benchmark harness")
message(STATUS "    bench_multifield  — 3-field NTT benchmark (BLS/GL/BB)")
message(STATUS "    ff_microbench     — Finite-field op throughput")
message(STATUS "    ntt_profile       — Clean binary for Nsight Compute")
message(STATUS "==========================================")
message(STATUS "")
