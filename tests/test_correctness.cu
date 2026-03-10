// tests/test_correctness.cu
// Validates NTT output against CPU reference implementation
// Phase 1: stub with basic structure. Real tests added in Phase 2/4.

#include "ff_arithmetic.cuh"
#include "ntt.cuh"
#include "cuda_utils.cuh"
#include "ff_reference.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// ─── Extern kernel declarations (defined in src/ff_mul.cu, linked via zkp_ntt_core) ─
// Note: __restrict__ qualifiers must match the definitions in ff_mul.cu exactly
// for correct MSVC name mangling.
extern __global__ void ff_mul_throughput_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_add_throughput_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sub_throughput_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_throughput_kernel(const FpElement* __restrict__ a, FpElement* __restrict__ out, uint32_t n);

// ─── Test Harness ────────────────────────────────────────────────────────────

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg)                                      \
    do {                                                            \
        tests_run++;                                                \
        if (!(cond)) {                                              \
            fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        } else {                                                    \
            tests_passed++;                                         \
        }                                                           \
    } while (0)

// ─── Phase 1: Smoke Tests ────────────────────────────────────────────────────

void test_fp_element_zero() {
    printf("test_fp_element_zero...\n");
    FpElement z = FpElement::zero();
    bool all_zero = true;
    for (int i = 0; i < 8; ++i) {
        if (z.limbs[i] != 0) all_zero = false;
    }
    TEST_ASSERT(all_zero, "FpElement::zero() should have all limbs = 0");
}

void test_fp_element_equality() {
    printf("test_fp_element_equality...\n");
    FpElement a = FpElement::zero();
    FpElement b = FpElement::zero();
    TEST_ASSERT(a == b, "Two zero elements should be equal");

    a.limbs[0] = 1;
    TEST_ASSERT(a != b, "Different elements should not be equal");
}

void test_cuda_device_available() {
    printf("test_cuda_device_available...\n");
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    TEST_ASSERT(err == cudaSuccess, "cudaGetDeviceCount should succeed");
    TEST_ASSERT(device_count > 0, "At least one CUDA device should be available");

    if (device_count > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("  Device 0: %s (SM %d.%d, %d SMs)\n",
               prop.name, prop.major, prop.minor,
               prop.multiProcessorCount);
    }
}

// ─── Phase 2: CPU Reference Self-Test ────────────────────────────────────────
// Validates that ff_reference.h arithmetic is correct (especially MSVC path).

void test_cpu_reference_self_test() {
    using namespace ff_ref;
    printf("test_cpu_reference_self_test...\n");

    // Test 1: Montgomery round-trip: to_mont(from_mont(x)) == x
    FpRef val = to_montgomery(FpRef::from_u64(42));
    FpRef back = from_montgomery(val);
    TEST_ASSERT(back == FpRef::from_u64(42),
        "Montgomery round-trip: to_mont then from_mont should recover original");

    // Test 2: 1 * 1 = 1 in Montgomery form
    FpRef one_mont;
    one_mont.limbs = R_MOD;  // Montgomery(1)
    FpRef prod = fp_mul(one_mont, one_mont);
    TEST_ASSERT(prod == one_mont,
        "Montgomery(1) * Montgomery(1) should equal Montgomery(1)");

    // Test 3: a + 0 = a
    FpRef a = to_montgomery(FpRef::from_u64(12345));
    FpRef zero_mont = to_montgomery(FpRef::zero());
    TEST_ASSERT(fp_add(a, zero_mont) == a,
        "a + 0 should equal a");

    // Test 4: a - a = 0
    FpRef diff = fp_sub(a, a);
    TEST_ASSERT(diff == FpRef::zero(),
        "a - a should equal 0");

    // Test 5: a * a^{-1} = 1
    FpRef a_inv = fp_inv(a);
    FpRef should_be_one = fp_mul(a, a_inv);
    TEST_ASSERT(should_be_one == one_mont,
        "a * a^{-1} should equal 1");

    // Test 6: (a + b) in Montgomery = to_mont(from_mont(a) + from_mont(b))
    FpRef b = to_montgomery(FpRef::from_u64(67890));
    FpRef sum_mont = fp_add(a, b);
    FpRef sum_plain = from_montgomery(sum_mont);
    // 12345 + 67890 = 80235
    TEST_ASSERT(sum_plain == FpRef::from_u64(80235),
        "12345 + 67890 should equal 80235");

    // Test 7: (a * b) = to_mont(12345 * 67890)
    FpRef prod_mont = fp_mul(a, b);
    FpRef prod_plain = from_montgomery(prod_mont);
    // 12345 * 67890 = 838102050
    TEST_ASSERT(prod_plain == FpRef::from_u64(838102050ULL),
        "12345 * 67890 should equal 838102050");
}

// ─── Phase 2: GPU FF Arithmetic Tests ───────────────────────────────────────

// Generate a valid field element in Montgomery form from a deterministic seed.
// Picks a small value (seed) and converts to Montgomery form via CPU reference.
static FpElement make_random_fp(uint32_t seed) {
    ff_ref::FpRef val = ff_ref::to_montgomery(ff_ref::FpRef::from_u64(static_cast<uint64_t>(seed)));
    FpElement fp;
    val.to_u32(fp.limbs);
    return fp;
}

void test_ff_add_gpu() {
    printf("test_ff_add_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N), h_out(N);

    // Generate deterministic test vectors
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
        h_b[i] = make_random_fp(i * 2 + 1000);
    }

    // Allocate device memory
    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    // Launch kernel
    ff_add_throughput_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Verify against CPU reference
    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_add(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_add GPU vs CPU reference mismatch");
    printf("  ff_add: %d/%u matched\n", pass_count, N);

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_sub_gpu() {
    printf("test_ff_sub_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
        h_b[i] = make_random_fp(i * 2 + 1000);
    }

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_sub_throughput_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_sub(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_sub GPU vs CPU reference mismatch");
    printf("  ff_sub: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_mul_gpu() {
    printf("test_ff_mul_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
        h_b[i] = make_random_fp(i * 2 + 1000);
    }

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_mul_throughput_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_mul(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_mul GPU vs CPU reference mismatch");
    printf("  ff_mul: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_sqr_gpu() {
    printf("test_ff_sqr_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
    }

    FpElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_sqr_throughput_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_sqr(ref_a);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_sqr GPU vs CPU reference mismatch");
    printf("  ff_sqr: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    printf("=== cuda-zkp-ntt correctness tests ===\n\n");

    // Phase 1: basic smoke tests
    test_fp_element_zero();
    test_fp_element_equality();
    test_cuda_device_available();

    // Phase 2: CPU reference self-test (validates MSVC intrinsics path)
    test_cpu_reference_self_test();

    // Phase 2: GPU FF arithmetic tests
    test_ff_add_gpu();
    test_ff_sub_gpu();
    test_ff_mul_gpu();
    test_ff_sqr_gpu();

    // Phase 4: NTT correctness tests (GPU NTT vs CPU DFT reference)

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
