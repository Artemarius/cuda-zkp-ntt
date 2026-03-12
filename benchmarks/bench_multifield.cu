// benchmarks/bench_multifield.cu
// 3-way NTT benchmark: BLS12-381 vs Goldilocks vs BabyBear
// Measures wall-clock time (7-rep median) across sizes 2^10..2^22.
// Outputs JSON to stdout for plotting.

#include "ntt.cuh"
#include "ntt_goldilocks.cuh"
#include "ntt_babybear.cuh"
#include "ff_arithmetic.cuh"
#include "ff_goldilocks.cuh"
#include "ff_babybear.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ─── BLS12-381 NTT benchmark ───────────────────────────────────────────────

static float bench_bls_single(size_t n) {
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(FpElement)));
    ntt_precompute_twiddles(n);

    for (int i = 0; i < WARMUP_REPS; ++i) {
        ntt_forward(d_data, n, NTTMode::OPTIMIZED);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward(d_data, n, NTTMode::OPTIMIZED);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return median(times);
}

static float bench_bls_batch(size_t n, int batch_size) {
    size_t total = (size_t)batch_size * n;
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, total * sizeof(FpElement)));
    ntt_precompute_twiddles(n);

    for (int i = 0; i < WARMUP_REPS; ++i) {
        ntt_forward_batch(d_data, batch_size, n, NTTMode::OPTIMIZED);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_batch(d_data, batch_size, n, NTTMode::OPTIMIZED);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return median(times);
}

// ─── Goldilocks NTT benchmark ───────────────────────────────────────────────

static float bench_gl_single(size_t n) {
    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(GoldilocksElement)));

    for (int i = 0; i < WARMUP_REPS; ++i) {
        ntt_forward_goldilocks(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_goldilocks(d_data, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return median(times);
}

static float bench_gl_batch(size_t n, int batch_size) {
    size_t total = (size_t)batch_size * n;
    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, total * sizeof(GoldilocksElement)));

    for (int i = 0; i < WARMUP_REPS; ++i) {
        ntt_forward_batch_goldilocks(d_data, batch_size, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_batch_goldilocks(d_data, batch_size, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return median(times);
}

// ─── BabyBear NTT benchmark ────────────────────────────────────────────────

static float bench_bb_single(size_t n) {
    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(BabyBearElement)));

    for (int i = 0; i < WARMUP_REPS; ++i) {
        ntt_forward_babybear(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_babybear(d_data, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return median(times);
}

static float bench_bb_batch(size_t n, int batch_size) {
    size_t total = (size_t)batch_size * n;
    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemset(d_data, 0, total * sizeof(BabyBearElement)));

    for (int i = 0; i < WARMUP_REPS; ++i) {
        ntt_forward_batch_babybear(d_data, batch_size, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(NUM_REPS);
    for (int r = 0; r < NUM_REPS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        ntt_forward_batch_babybear(d_data, batch_size, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return median(times);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    fprintf(stderr, "=== Multi-Field NTT Benchmark ===\n");
    fprintf(stderr, "Device: %s (SM %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fprintf(stderr, "VRAM: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    fprintf(stderr, "Reps: %d (median), Warmup: %d\n\n", NUM_REPS, WARMUP_REPS);

    // Sizes to benchmark
    int log_sizes[] = {10, 12, 15, 16, 18, 20, 22};
    int num_sizes = sizeof(log_sizes) / sizeof(log_sizes[0]);
    int batch_size = 8;

    // Print header to stderr for progress
    fprintf(stderr, "%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n",
            "Size", "BLS(ms)", "GL(ms)", "BB(ms)",
            "BLS B8(ms)", "GL B8(ms)", "BB B8(ms)");
    fprintf(stderr, "%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n",
            "----", "-------", "------", "------",
            "----------", "---------", "---------");

    // JSON output to stdout
    printf("{\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"sms\": %d,\n", prop.multiProcessorCount);
    printf("  \"reps\": %d,\n", NUM_REPS);
    printf("  \"batch_size\": %d,\n", batch_size);
    printf("  \"results\": [\n");

    for (int si = 0; si < num_sizes; ++si) {
        int log_n = log_sizes[si];
        size_t n = (size_t)1 << log_n;

        fprintf(stderr, "2^%-6d ", log_n);
        fflush(stderr);

        // Single NTT
        float bls_s = bench_bls_single(n);
        fprintf(stderr, "%-12.3f ", bls_s); fflush(stderr);

        float gl_s = bench_gl_single(n);
        fprintf(stderr, "%-12.3f ", gl_s); fflush(stderr);

        float bb_s = bench_bb_single(n);
        fprintf(stderr, "%-12.3f ", bb_s); fflush(stderr);

        // Batched NTT (skip batch for very small sizes that would OOM at batch*n for BLS)
        float bls_b = -1, gl_b = -1, bb_b = -1;
        // BLS: 32 bytes/element, at 2^22 * 8 = 1 GB — may be tight for 6 GB VRAM
        // For sizes <= 2^20, safe. For 2^22, BLS batch of 8 = 1 GB — should fit.
        size_t bls_batch_bytes = (size_t)batch_size * n * sizeof(FpElement);
        if (bls_batch_bytes <= (size_t)4 * 1024 * 1024 * 1024) {
            bls_b = bench_bls_batch(n, batch_size);
        }
        fprintf(stderr, "%-12.3f ", bls_b); fflush(stderr);

        gl_b = bench_gl_batch(n, batch_size);
        fprintf(stderr, "%-12.3f ", gl_b); fflush(stderr);

        bb_b = bench_bb_batch(n, batch_size);
        fprintf(stderr, "%-12.3f", bb_b);

        fprintf(stderr, "\n");

        // JSON entry
        printf("    {\n");
        printf("      \"log_n\": %d,\n", log_n);
        printf("      \"n\": %zu,\n", n);
        printf("      \"bls_single_ms\": %.4f,\n", bls_s);
        printf("      \"gl_single_ms\": %.4f,\n", gl_s);
        printf("      \"bb_single_ms\": %.4f,\n", bb_s);
        printf("      \"bls_batch_ms\": %.4f,\n", bls_b);
        printf("      \"gl_batch_ms\": %.4f,\n", gl_b);
        printf("      \"bb_batch_ms\": %.4f,\n", bb_b);
        printf("      \"gl_speedup\": %.2f,\n", bls_s / gl_s);
        printf("      \"bb_speedup\": %.2f\n", bls_s / bb_s);
        printf("    }%s\n", (si < num_sizes - 1) ? "," : "");
    }

    printf("  ]\n");
    printf("}\n");

    fprintf(stderr, "\nDone.\n");
    return 0;
}
