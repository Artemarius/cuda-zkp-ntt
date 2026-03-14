// benchmarks/bench_groth16.cu
// End-to-end Groth16 pipeline benchmark with phase breakdown.
// Supports both toy circuit (x^3+x+5=y) and Fibonacci circuit (a_{i+2}=a_i+a_{i+1}).
// Outputs JSON to stdout for plotting.

#include "groth16.cuh"
#include "ff_arithmetic.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <chrono>

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

    fprintf(stderr, "=== Groth16 Pipeline Benchmark (v2.2.0) ===\n");
    fprintf(stderr, "Device: %s (SM %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // JSON header
    printf("{\n");
    printf("  \"benchmark\": \"groth16_v220\",\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);

    // ─── Fibonacci Circuit Benchmark ─────────────────────────────────────────
    fprintf(stderr, "\n=== Fibonacci Circuit: a_{i+2} = a_i + a_{i+1} ===\n");
    fprintf(stderr, "%-10s %-10s %-10s %-12s %-12s %-12s %-12s\n",
            "NumConst", "Domain", "NumVars", "Setup(s)", "GPU(ms)", "CPU(ms)", "CPU/GPU");
    fprintf(stderr, "%-10s %-10s %-10s %-12s %-12s %-12s %-12s\n",
            "--------", "------", "-------", "--------", "------", "------", "-------");

    printf("  \"fibonacci\": [\n");

    size_t sizes[] = {256, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    constexpr int GPU_REPS = 3;
    constexpr int CPU_REPS = 1;  // CPU proof is slow (sequential scalar muls)

    for (int si = 0; si < num_sizes; ++si) {
        size_t nc = sizes[si];

        SparseR1CS r1cs = make_fibonacci_r1cs(nc);
        auto witness = compute_fibonacci_witness(1, 1, nc);

        // Setup
        fprintf(stderr, "  Setting up nc=%d...", (int)nc);
        fflush(stderr);
        auto t_setup = Clock::now();
        ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
        double setup_ms = ms_since(t_setup);
        fprintf(stderr, " %.1fs\n", setup_ms / 1000.0);

        // GPU proof (warmup + measured)
        {
            Groth16Proof p = groth16_prove_sparse(r1cs, pk, witness, 17, 23);
            (void)p;
        }

        std::vector<double> gpu_times(GPU_REPS);
        for (int r = 0; r < GPU_REPS; ++r) {
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = Clock::now();
            Groth16Proof p = groth16_prove_sparse(r1cs, pk, witness, 17, 23);
            CUDA_CHECK(cudaDeviceSynchronize());
            gpu_times[r] = ms_since(t0);
            (void)p;
        }
        double gpu_ms = median(gpu_times);

        // CPU proof (1 rep — sequential scalar muls are very slow)
        fprintf(stderr, "  CPU prove nc=%d...", (int)nc);
        fflush(stderr);
        auto t_cpu = Clock::now();
        {
            Groth16Proof p = groth16_prove_cpu_sparse(r1cs, pk, witness, 17, 23);
            (void)p;
        }
        double cpu_ms = ms_since(t_cpu);
        fprintf(stderr, " %.1fs\n", cpu_ms / 1000.0);

        double ratio = cpu_ms / gpu_ms;

        fprintf(stderr, "%-10d %-10d %-10d %-12.1f %-12.1f %-12.1f %-12.1fx\n",
                (int)nc, (int)r1cs.domain_size, (int)r1cs.num_variables,
                setup_ms / 1000.0, gpu_ms, cpu_ms, ratio);

        printf("    {\n");
        printf("      \"num_constraints\": %d,\n", (int)nc);
        printf("      \"domain_size\": %d,\n", (int)r1cs.domain_size);
        printf("      \"num_variables\": %d,\n", (int)r1cs.num_variables);
        printf("      \"setup_ms\": %.4f,\n", setup_ms);
        printf("      \"gpu_prove_ms\": %.4f,\n", gpu_ms);
        printf("      \"cpu_prove_ms\": %.4f,\n", cpu_ms);
        printf("      \"cpu_over_gpu\": %.4f\n", ratio);
        printf("    }%s\n", (si < num_sizes - 1) ? "," : "");
    }

    printf("  ]\n");
    printf("}\n");

    fprintf(stderr, "\nDone.\n");
    return 0;
}
