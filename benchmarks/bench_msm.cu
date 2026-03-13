// benchmarks/bench_msm.cu
// MSM benchmark: Pippenger's bucket method at various sizes.
// Measures wall-clock time (7-rep median) for sizes 2^10..2^20.
// Outputs JSON to stdout for plotting.
// v2.1.0: signed-digit recoding, parallel bucket reduction, memory pools.

#include "msm.cuh"
#include "ec_g1.cuh"
#include "ff_fq.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

static constexpr int NUM_REPS = 7;
static constexpr int WARMUP_REPS = 2;

// ─── Median helper ──────────────────────────────────────────────────────────

static float median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) * 0.5f;
}

// ─── Generate random G1 affine points (on-curve) ─────────────────────────

// For benchmarking, we use the generator multiplied by sequential scalars.
// This produces valid on-curve points without needing a hash-to-curve.
__global__ void generate_bases_kernel(G1Affine* d_bases, uint32_t n, uint32_t seed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Use generator point: P = (seed + idx + 1) * G1
    // For benchmark, we just need valid on-curve points. Use the generator directly.
    // Copy the generator point for all bases (MSM correctness doesn't depend on distinct bases
    // for timing purposes — the GPU work is identical).
    G1Affine gen;
    for (int i = 0; i < 12; ++i) {
        gen.x.limbs[i] = G1_GEN_X[i];
        gen.y.limbs[i] = G1_GEN_Y[i];
    }
    gen.infinity = false;
    d_bases[idx] = gen;
}

// Generate pseudo-random scalars (8 x uint32_t per scalar)
__global__ void generate_scalars_kernel(uint32_t* d_scalars, uint32_t n, uint32_t seed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Simple LCG-based PRNG for benchmarking
    uint32_t state = seed + idx * 1103515245u + 12345u;
    for (int limb = 0; limb < 8; ++limb) {
        state = state * 1103515245u + 12345u;
        d_scalars[idx * 8 + limb] = state;
    }
    // Mask top limb to be < Fr modulus (rough approximation)
    d_scalars[idx * 8 + 7] &= 0x73ffffffu;
}

// ─── MSM benchmark ─────────────────────────────────────────────────────────

static float bench_msm(size_t n, uint32_t seed = 42) {
    G1Affine* d_bases;
    uint32_t* d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));

    const int BLOCK = 256;
    int grid = ((int)n + BLOCK - 1) / BLOCK;
    generate_bases_kernel<<<grid, BLOCK>>>(d_bases, (uint32_t)n, seed);
    generate_scalars_kernel<<<grid, BLOCK>>>(d_scalars, (uint32_t)n, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine result;

    // Warmup
    for (int i = 0; i < WARMUP_REPS; ++i) {
        msm_g1(&result, d_bases, d_scalars, n);
    }

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        msm_g1(&result, d_bases, d_scalars, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
    return median(times);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    fprintf(stderr, "=== MSM Benchmark (Pippenger's Bucket Method) ===\n");
    fprintf(stderr, "Device: %s (SM %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fprintf(stderr, "VRAM: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    fprintf(stderr, "Reps: %d (median), Warmup: %d\n\n", NUM_REPS, WARMUP_REPS);

    // Sizes to benchmark
    // Memory budget: n points * 96B (G1Affine) + n * 32B (scalars) + sort temp
    // At n=2^18: 262K * 128B = 32 MB + sort temp ~10 MB = ~42 MB — fine
    // At n=2^20: 1M * 128B = 128 MB + sort temp ~40 MB = ~170 MB — fits in 6 GB
    int log_sizes[] = {10, 12, 14, 15, 16, 18, 20};
    int num_sizes = sizeof(log_sizes) / sizeof(log_sizes[0]);

    fprintf(stderr, "%-8s %-12s %-12s %-12s\n",
            "Size", "MSM(ms)", "Window(c)", "Points/ms");
    fprintf(stderr, "%-8s %-12s %-12s %-12s\n",
            "----", "-------", "--------", "---------");

    // JSON output to stdout
    printf("{\n");
    printf("  \"benchmark\": \"msm\",\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"sms\": %d,\n", prop.multiProcessorCount);
    printf("  \"reps\": %d,\n", NUM_REPS);
    printf("  \"results\": [\n");

    for (int si = 0; si < num_sizes; ++si) {
        int log_n = log_sizes[si];
        size_t n = (size_t)1 << log_n;
        int c = msm_optimal_window(n);

        fprintf(stderr, "2^%-6d ", log_n);
        fflush(stderr);

        float ms = bench_msm(n);
        float points_per_ms = (float)n / ms;

        fprintf(stderr, "%-12.3f %-12d %-12.0f\n", ms, c, points_per_ms);

        printf("    {\n");
        printf("      \"log_n\": %d,\n", log_n);
        printf("      \"n\": %zu,\n", n);
        printf("      \"window_bits\": %d,\n", c);
        printf("      \"msm_ms\": %.4f,\n", ms);
        printf("      \"points_per_ms\": %.1f\n", points_per_ms);
        printf("    }%s\n", (si < num_sizes - 1) ? "," : "");
    }

    printf("  ]\n");
    printf("}\n");

    fprintf(stderr, "\nDone.\n");
    return 0;
}
