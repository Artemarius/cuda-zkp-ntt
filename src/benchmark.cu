// src/benchmark.cu
// Minimal profiling binary for Nsight Compute / Nsight Systems
// Supports FF arithmetic profiling (Phase 2) and NTT profiling (Phase 7).

#include "ntt.cuh"
#include "pipeline.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Kernel declarations from ff_mul.cu — baseline
extern __global__ void ff_mul_throughput_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

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

extern __global__ void ff_sqr_throughput_kernel(
    const FpElement* __restrict__ a,
    FpElement* __restrict__ out,
    uint32_t n);

// Kernel declarations from ff_mul.cu — v2 (branchless PTX reduction)
extern __global__ void ff_mul_v2_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

extern __global__ void ff_add_v2_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

extern __global__ void ff_sub_v2_kernel(
    const FpElement* __restrict__ a,
    const FpElement* __restrict__ b,
    FpElement* __restrict__ out,
    uint32_t n);

extern __global__ void ff_sqr_v2_kernel(
    const FpElement* __restrict__ a,
    FpElement* __restrict__ out,
    uint32_t n);

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s --mode <mode> [--size N]\n", prog);
    fprintf(stderr, "  --mode   ff_mul | ff_add | ff_sub | ff_sqr\n");
    fprintf(stderr, "           ff_mul_v2 | ff_add_v2 | ff_sub_v2 | ff_sqr_v2\n");
    fprintf(stderr, "           naive | optimized | async\n");
    fprintf(stderr, "           device_info  (PCIe bandwidth + copy engine count)\n");
    fprintf(stderr, "  --size   log2 of element count (default: 20)\n");
}

// ─── FF Profiling (Phase 2) ─────────────────────────────────────────────────

static void profile_ff(const char* mode, uint32_t n) {
    size_t bytes = (size_t)n * sizeof(FpElement);

    // Host data: deterministic nonzero pattern
    FpElement* h_a = (FpElement*)malloc(bytes);
    FpElement* h_b = (FpElement*)malloc(bytes);
    for (uint32_t i = 0; i < n; ++i) {
        for (int j = 0; j < 8; ++j) {
            h_a[i].limbs[j] = i * 8 + j + 1;
            h_b[i].limbs[j] = (i + 1) * 8 + j + 3;
        }
    }

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Helper lambda to dispatch kernel by mode name
    auto launch = [&]() {
        if (strcmp(mode, "ff_mul") == 0)
            ff_mul_throughput_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        else if (strcmp(mode, "ff_add") == 0)
            ff_add_throughput_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        else if (strcmp(mode, "ff_sub") == 0)
            ff_sub_throughput_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        else if (strcmp(mode, "ff_sqr") == 0)
            ff_sqr_throughput_kernel<<<blocks, threads>>>(d_a, d_out, n);
        else if (strcmp(mode, "ff_mul_v2") == 0)
            ff_mul_v2_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        else if (strcmp(mode, "ff_add_v2") == 0)
            ff_add_v2_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        else if (strcmp(mode, "ff_sub_v2") == 0)
            ff_sub_v2_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        else if (strcmp(mode, "ff_sqr_v2") == 0)
            ff_sqr_v2_kernel<<<blocks, threads>>>(d_a, d_out, n);
    };

    // Warmup (not profiled by ncu --launch-skip)
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profiled kernel launch
    printf("Launching %s kernel: %d blocks x %d threads, %u elements\n", mode, blocks, threads, n);
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Done.\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    free(h_a);
    free(h_b);
}

int main(int argc, char** argv) {
    const char* mode_str = "ff_mul";
    int log_size = 20;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            log_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    uint32_t n = 1u << log_size;
    printf("=== ntt_profile ===\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, VRAM: %zu MB\n", prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));
    printf("Mode: %s, Size: 2^%d = %u elements\n", mode_str, log_size, n);
    printf("\n");

    if (strcmp(mode_str, "device_info") == 0) {
        cudaDeviceProp dp;
        CUDA_CHECK(cudaGetDeviceProperties(&dp, device));
        printf("asyncEngineCount: %d\n", dp.asyncEngineCount);
        printf("concurrentKernels: %d\n", dp.concurrentKernels);
        printf("deviceOverlap: %d\n", dp.deviceOverlap);

        size_t bw_bytes = (size_t)n * sizeof(FpElement);
        void *h_bw, *d_bw;
        CUDA_CHECK(cudaMallocHost(&h_bw, bw_bytes));
        CUDA_CHECK(cudaMalloc(&d_bw, bw_bytes));
        CUDA_CHECK(cudaMemcpy(d_bw, h_bw, bw_bytes, cudaMemcpyHostToDevice)); // warmup

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_bw, h_bw, bw_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms_h2d;
        CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, t0, t1));
        printf("H2D %zu MB: %.1f ms (%.1f GB/s)\n",
            bw_bytes / (1024*1024), ms_h2d, bw_bytes / ms_h2d / 1e6);

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(h_bw, d_bw, bw_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms_d2h;
        CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, t0, t1));
        printf("D2H %zu MB: %.1f ms (%.1f GB/s)\n",
            bw_bytes / (1024*1024), ms_d2h, bw_bytes / ms_d2h / 1e6);

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFreeHost(h_bw));
        CUDA_CHECK(cudaFree(d_bw));
    } else if (strncmp(mode_str, "ff_", 3) == 0) {
        profile_ff(mode_str, n);
    } else if (strcmp(mode_str, "naive") == 0 ||
               strcmp(mode_str, "optimized") == 0) {
        // ─── Single NTT Profiling (Phase 7) ─────────────────────────────
        NTTMode ntt_mode = (strcmp(mode_str, "naive") == 0)
                               ? NTTMode::NAIVE
                               : NTTMode::OPTIMIZED;

        size_t bytes = (size_t)n * sizeof(FpElement);

        // Deterministic nonzero host data
        FpElement* h_data = (FpElement*)malloc(bytes);
        for (uint32_t i = 0; i < n; ++i)
            for (int j = 0; j < 8; ++j)
                h_data[i].limbs[j] = i * 8 + j + 1;

        FpElement* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

        // Warmup (initializes twiddle cache)
        ntt_forward(d_data, n, ntt_mode);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Re-upload fresh data for profiled run
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

        printf("Launching NTT forward (%s): n=2^%d = %u elements\n",
               mode_str, log_size, n);
        ntt_forward(d_data, n, ntt_mode);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Done.\n");

        CUDA_CHECK(cudaFree(d_data));
        free(h_data);

    } else if (strcmp(mode_str, "async") == 0) {
        // ─── Async Pipeline Profiling (Phase 7) ─────────────────────────
        int num_batches = 8;
        size_t total_n = (size_t)n * num_batches;
        size_t total_bytes = total_n * sizeof(FpElement);

        FpElement *h_in, *h_out;
        CUDA_CHECK(cudaMallocHost(&h_in,  total_bytes));
        CUDA_CHECK(cudaMallocHost(&h_out, total_bytes));

        // Fill with deterministic data
        for (size_t i = 0; i < total_n; ++i)
            for (int j = 0; j < 8; ++j)
                h_in[i].limbs[j] = (uint32_t)((i * 8 + j + 1) & 0xFFFFFFFF);

        printf("Launching async pipeline: %d batches x 2^%d = %zu elements\n",
               num_batches, log_size, total_n);

        AsyncNTTPipeline pipeline(n);
        pipeline.process_pinned(h_in, h_out, total_n, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Done.\n");

        CUDA_CHECK(cudaFreeHost(h_in));
        CUDA_CHECK(cudaFreeHost(h_out));
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode_str);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
