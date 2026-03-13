// benchmarks/bench_groth16.cu
// End-to-end Groth16 pipeline benchmark with phase breakdown.
// Measures: trusted setup, QAP construction, H(x) quotient, proof assembly.
// Outputs JSON to stdout for plotting.

#include "groth16.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <chrono>

static constexpr int NUM_REPS = 5;
static constexpr int WARMUP_REPS = 1;

// ─── Timing helpers ──────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) * 0.5;
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    fprintf(stderr, "=== Groth16 Pipeline Benchmark ===\n");
    fprintf(stderr, "Device: %s (SM %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fprintf(stderr, "Circuit: x^3 + x + 5 = y (4 R1CS constraints, 6 variables)\n");
    fprintf(stderr, "Reps: %d (median), Warmup: %d\n\n", NUM_REPS, WARMUP_REPS);

    // Domain sizes to test
    int domain_sizes[] = {256, 512, 1024};
    int num_domains = sizeof(domain_sizes) / sizeof(domain_sizes[0]);

    fprintf(stderr, "%-10s %-12s %-12s %-12s %-12s %-12s\n",
            "Domain", "Setup(ms)", "GPU(ms)", "CPU(ms)", "GPU/CPU", "Total(ms)");
    fprintf(stderr, "%-10s %-12s %-12s %-12s %-12s %-12s\n",
            "------", "--------", "------", "------", "-------", "--------");

    // JSON output to stdout
    printf("{\n");
    printf("  \"benchmark\": \"groth16\",\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"circuit\": \"x^3 + x + 5 = y\",\n");
    printf("  \"constraints\": 4,\n");
    printf("  \"variables\": 6,\n");
    printf("  \"reps\": %d,\n", NUM_REPS);
    printf("  \"results\": [\n");

    for (int di = 0; di < num_domains; ++di) {
        int domain_size = domain_sizes[di];

        // Build R1CS and witness
        R1CS r1cs = make_toy_r1cs((size_t)domain_size);
        std::vector<FpElement> witness = compute_witness(3);

        // ── Phase 1: Trusted setup (CPU) ──
        std::vector<double> setup_times(NUM_REPS);
        ProvingKey pk;
        for (int r = 0; r < NUM_REPS; ++r) {
            auto t0 = Clock::now();
            pk = generate_proving_key(r1cs, 42);
            setup_times[r] = ms_since(t0);
        }
        double setup_ms = median(setup_times);

        // ── Phase 2: GPU proof generation ──
        // Warmup
        for (int i = 0; i < WARMUP_REPS; ++i) {
            Groth16Proof p = groth16_prove(r1cs, pk, witness, 17, 23);
            (void)p;
        }

        std::vector<double> gpu_times(NUM_REPS);
        for (int r = 0; r < NUM_REPS; ++r) {
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = Clock::now();
            Groth16Proof p = groth16_prove(r1cs, pk, witness, 17, 23);
            CUDA_CHECK(cudaDeviceSynchronize());
            gpu_times[r] = ms_since(t0);
            (void)p;
        }
        double gpu_ms = median(gpu_times);

        // ── Phase 3: CPU proof generation (for comparison) ──
        std::vector<double> cpu_times(NUM_REPS);
        for (int r = 0; r < NUM_REPS; ++r) {
            auto t0 = Clock::now();
            Groth16Proof p = groth16_prove_cpu(r1cs, pk, witness, 17, 23);
            cpu_times[r] = ms_since(t0);
            (void)p;
        }
        double cpu_ms = median(cpu_times);

        double total_ms = setup_ms + gpu_ms;
        double ratio = gpu_ms / cpu_ms;

        fprintf(stderr, "%-10d %-12.2f %-12.2f %-12.2f %-12.3f %-12.2f\n",
                domain_size, setup_ms, gpu_ms, cpu_ms, ratio, total_ms);

        printf("    {\n");
        printf("      \"domain_size\": %d,\n", domain_size);
        printf("      \"setup_ms\": %.4f,\n", setup_ms);
        printf("      \"gpu_prove_ms\": %.4f,\n", gpu_ms);
        printf("      \"cpu_prove_ms\": %.4f,\n", cpu_ms);
        printf("      \"gpu_cpu_ratio\": %.4f,\n", ratio);
        printf("      \"total_ms\": %.4f\n", total_ms);
        printf("    }%s\n", (di < num_domains - 1) ? "," : "");
    }

    printf("  ]\n");
    printf("}\n");

    fprintf(stderr, "\nDone.\n");
    return 0;
}
