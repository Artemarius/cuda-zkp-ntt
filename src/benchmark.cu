// src/benchmark.cu
// Minimal profiling binary for Nsight Compute / Nsight Systems
// Phase 1: stub with device query. Profiling modes added in Phase 7.

#include "ntt.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [--mode naive|optimized|async] [--size N]\n", prog);
    fprintf(stderr, "  --mode   NTT variant to profile (default: naive)\n");
    fprintf(stderr, "  --size   log2 of NTT size, e.g. 20 for 2^20 (default: 20)\n");
}

int main(int argc, char** argv) {
    // Parse command line
    const char* mode_str = "naive";
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

    // Print device info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== ntt_profile ===\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, VRAM: %zu MB\n", prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));
    printf("Mode: %s, Size: 2^%d = %u elements\n", mode_str, log_size, 1u << log_size);
    printf("\n");

    // TODO: Phase 7 — allocate, run NTT, measure
    printf("Profiling stub — implement in Phase 7.\n");

    return 0;
}
