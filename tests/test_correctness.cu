// tests/test_correctness.cu
// Validates NTT output against CPU reference implementation
// Phase 1: stub with basic structure. Real tests added in Phase 2/4.

#include "ff_arithmetic.cuh"
#include "ntt.cuh"
#include "pipeline.cuh"
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
extern __global__ void ff_mul_barrett_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_barrett_kernel(const FpElement* __restrict__ a, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_throughput_kernel(const FpElement* __restrict__ a, FpElement* __restrict__ out, uint32_t n);

// SoA kernel declarations
extern __global__ void ff_add_soa_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ out, uint32_t n);
extern __global__ void ff_sub_soa_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ out, uint32_t n);
extern __global__ void ff_mul_soa_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_soa_kernel(const uint32_t* __restrict__ a, uint32_t* __restrict__ out, uint32_t n);

// v2 (branchless) kernel declarations
extern __global__ void ff_mul_v2_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_add_v2_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sub_v2_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);
extern __global__ void ff_sqr_v2_kernel(const FpElement* __restrict__ a, FpElement* __restrict__ out, uint32_t n);

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

// ─── Phase 3: v2 (branchless) Kernel Correctness Tests ──────────────────────

// Generic helper: test a binary v2 kernel against CPU reference
template<typename KernelFn, typename RefFn>
void test_v2_binary(const char* name, KernelFn kernel, RefFn ref_op) {
    printf("%s...\n", name);

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

    kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ref_op(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }
    TEST_ASSERT(pass_count == (int)N, name);
    printf("  %s: %d/%u matched\n", name, pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_mul_v2_gpu() {
    test_v2_binary("test_ff_mul_v2_gpu", ff_mul_v2_kernel, ff_ref::fp_mul);
}

void test_ff_add_v2_gpu() {
    test_v2_binary("test_ff_add_v2_gpu", ff_add_v2_kernel, ff_ref::fp_add);
}

void test_ff_sub_v2_gpu() {
    test_v2_binary("test_ff_sub_v2_gpu", ff_sub_v2_kernel, ff_ref::fp_sub);
}

void test_ff_sqr_v2_gpu() {
    printf("test_ff_sqr_v2_gpu...\n");
    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_out(N);
    for (uint32_t i = 0; i < N; ++i)
        h_a[i] = make_random_fp(i * 2 + 1);

    FpElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_sqr_v2_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_sqr(ref_a);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }
    TEST_ASSERT(pass_count == (int)N, "ff_sqr_v2 GPU vs CPU reference mismatch");
    printf("  ff_sqr_v2: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── SoA Helpers ─────────────────────────────────────────────────────────────

static void aos_to_soa(const std::vector<FpElement>& aos,
                        std::vector<uint32_t>& soa, uint32_t n) {
    soa.resize(8u * n);
    for (uint32_t i = 0; i < n; ++i)
        for (int j = 0; j < 8; ++j)
            soa[j * n + i] = aos[i].limbs[j];
}

static void soa_to_aos(const std::vector<uint32_t>& soa,
                        std::vector<FpElement>& aos, uint32_t n) {
    aos.resize(n);
    for (uint32_t i = 0; i < n; ++i)
        for (int j = 0; j < 8; ++j)
            aos[i].limbs[j] = soa[j * n + i];
}

// ─── Phase 3: SoA Kernel Correctness Tests ──────────────────────────────────

void test_ff_mul_soa_gpu() {
    printf("test_ff_mul_soa_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
        h_b[i] = make_random_fp(i * 2 + 1000);
    }

    std::vector<uint32_t> soa_a, soa_b;
    aos_to_soa(h_a, soa_a, N);
    aos_to_soa(h_b, soa_b, N);

    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    uint32_t *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, soa_b.data(), soa_bytes, cudaMemcpyHostToDevice));

    ff_mul_soa_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> soa_out(8u * N);
    CUDA_CHECK(cudaMemcpy(soa_out.data(), d_out, soa_bytes, cudaMemcpyDeviceToHost));

    std::vector<FpElement> h_out;
    soa_to_aos(soa_out, h_out, N);

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_mul(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }
    TEST_ASSERT(pass_count == (int)N, "ff_mul SoA GPU vs CPU reference mismatch");
    printf("  ff_mul_soa: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_add_soa_gpu() {
    printf("test_ff_add_soa_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
        h_b[i] = make_random_fp(i * 2 + 1000);
    }

    std::vector<uint32_t> soa_a, soa_b;
    aos_to_soa(h_a, soa_a, N);
    aos_to_soa(h_b, soa_b, N);

    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    uint32_t *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, soa_b.data(), soa_bytes, cudaMemcpyHostToDevice));

    ff_add_soa_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> soa_out(8u * N);
    CUDA_CHECK(cudaMemcpy(soa_out.data(), d_out, soa_bytes, cudaMemcpyDeviceToHost));

    std::vector<FpElement> h_out;
    soa_to_aos(soa_out, h_out, N);

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_add(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }
    TEST_ASSERT(pass_count == (int)N, "ff_add SoA GPU vs CPU reference mismatch");
    printf("  ff_add_soa: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_sub_soa_gpu() {
    printf("test_ff_sub_soa_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
        h_b[i] = make_random_fp(i * 2 + 1000);
    }

    std::vector<uint32_t> soa_a, soa_b;
    aos_to_soa(h_a, soa_a, N);
    aos_to_soa(h_b, soa_b, N);

    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    uint32_t *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, soa_b.data(), soa_bytes, cudaMemcpyHostToDevice));

    ff_sub_soa_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> soa_out(8u * N);
    CUDA_CHECK(cudaMemcpy(soa_out.data(), d_out, soa_bytes, cudaMemcpyDeviceToHost));

    std::vector<FpElement> h_out;
    soa_to_aos(soa_out, h_out, N);

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef ref_b = ff_ref::FpRef::from_u32(h_b[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_sub(ref_a, ref_b);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }
    TEST_ASSERT(pass_count == (int)N, "ff_sub SoA GPU vs CPU reference mismatch");
    printf("  ff_sub_soa: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_sqr_soa_gpu() {
    printf("test_ff_sqr_soa_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_random_fp(i * 2 + 1);
    }

    std::vector<uint32_t> soa_a;
    aos_to_soa(h_a, soa_a, N);

    const size_t soa_bytes = 8u * N * sizeof(uint32_t);
    uint32_t *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, soa_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, soa_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, soa_a.data(), soa_bytes, cudaMemcpyHostToDevice));

    ff_sqr_soa_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> soa_out(8u * N);
    CUDA_CHECK(cudaMemcpy(soa_out.data(), d_out, soa_bytes, cudaMemcpyDeviceToHost));

    std::vector<FpElement> h_out;
    soa_to_aos(soa_out, h_out, N);

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef ref_a = ff_ref::FpRef::from_u32(h_a[i].limbs);
        ff_ref::FpRef expected = ff_ref::fp_sqr(ref_a);
        ff_ref::FpRef gpu_result = ff_ref::FpRef::from_u32(h_out[i].limbs);
        if (gpu_result == expected) ++pass_count;
    }
    TEST_ASSERT(pass_count == (int)N, "ff_sqr SoA GPU vs CPU reference mismatch");
    printf("  ff_sqr_soa: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Phase 4: NTT Correctness Tests ──────────────────────────────────────────

static const char* mode_name(NTTMode m) {
    switch (m) {
        case NTTMode::NAIVE:     return "NAIVE";
        case NTTMode::OPTIMIZED: return "OPTIMIZED";
        case NTTMode::ASYNC:     return "ASYNC";
        case NTTMode::BARRETT:   return "BARRETT";
        default:                 return "UNKNOWN";
    }
}

// Forward NTT: GPU result vs CPU reference
void test_ntt_forward_gpu(int log_n, NTTMode mode = NTTMode::NAIVE) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_forward_gpu [%s] (n=2^%d=%zu)...\n", mode_name(mode), log_n, n);

    // Generate test data in standard form (small deterministic values)
    std::vector<FpElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
    }

    // CPU reference: convert to Montgomery, run NTT, convert back
    std::vector<ff_ref::FpRef> cpu_data(n);
    for (size_t i = 0; i < n; ++i) {
        cpu_data[i] = ff_ref::to_montgomery(ff_ref::FpRef::from_u32(h_data[i].limbs));
    }
    ff_ref::ntt_forward_reference(cpu_data, n);
    for (size_t i = 0; i < n; ++i) {
        cpu_data[i] = ff_ref::from_montgomery(cpu_data[i]);
    }

    // GPU path: upload standard form, ntt_forward converts to/from Montgomery internally
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_forward(d_data, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_result(n);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Compare
    int pass_count = 0;
    for (size_t i = 0; i < n; ++i) {
        ff_ref::FpRef gpu_val = ff_ref::FpRef::from_u32(h_result[i].limbs);
        if (gpu_val == cpu_data[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(n),
        "NTT forward GPU vs CPU mismatch");
    printf("  NTT forward [%s] (n=2^%d): %d/%zu matched\n", mode_name(mode), log_n, pass_count, n);

    if (pass_count != static_cast<int>(n) && n <= 1024) {
        // Print first few mismatches for debugging
        int printed = 0;
        for (size_t i = 0; i < n && printed < 5; ++i) {
            ff_ref::FpRef gpu_val = ff_ref::FpRef::from_u32(h_result[i].limbs);
            if (gpu_val != cpu_data[i]) {
                printf("    [%zu] GPU: %08x%08x... CPU: %08x%08x...\n",
                    i,
                    static_cast<uint32_t>(gpu_val.limbs[3] >> 32),
                    static_cast<uint32_t>(gpu_val.limbs[3]),
                    static_cast<uint32_t>(cpu_data[i].limbs[3] >> 32),
                    static_cast<uint32_t>(cpu_data[i].limbs[3]));
                ++printed;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_data));
}

// Roundtrip test: INTT(NTT(x)) == x
void test_ntt_roundtrip_gpu(int log_n, NTTMode mode = NTTMode::NAIVE) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_roundtrip_gpu [%s] (n=2^%d=%zu)...\n", mode_name(mode), log_n, n);

    // Generate test data in standard form
    std::vector<FpElement> h_original(n);
    for (size_t i = 0; i < n; ++i) {
        h_original[i] = FpElement::zero();
        h_original[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
    }

    // GPU: forward NTT then inverse NTT
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_original.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_forward(d_data, n, mode);
    ntt_inverse(d_data, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_result(n);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Compare with original
    int pass_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_result[i] == h_original[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(n),
        "NTT roundtrip mismatch");
    printf("  NTT roundtrip [%s] (n=2^%d): %d/%zu matched\n", mode_name(mode), log_n, pass_count, n);

    if (pass_count != static_cast<int>(n) && n <= 1024) {
        int printed = 0;
        for (size_t i = 0; i < n && printed < 5; ++i) {
            if (h_result[i] != h_original[i]) {
                printf("    [%zu] got: %08x %08x... expected: %08x %08x...\n",
                    i, h_result[i].limbs[7], h_result[i].limbs[6],
                    h_original[i].limbs[7], h_original[i].limbs[6]);
                ++printed;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_data));
}

// ─── Phase 6: Async Pipeline NTT Tests ───────────────────────────────────────

// Forward NTT through pipeline, compare each batch against CPU reference
void test_async_pipeline_forward(int log_n, int num_batches) {
    size_t ntt_size = static_cast<size_t>(1) << log_n;
    size_t total_n = ntt_size * num_batches;
    printf("test_async_pipeline_forward (n=2^%d, %d batches)...\n", log_n, num_batches);

    // Generate test data in standard form
    std::vector<FpElement> h_input(total_n);
    for (size_t i = 0; i < total_n; ++i) {
        h_input[i] = FpElement::zero();
        h_input[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
    }

    std::vector<FpElement> h_output(total_n);

    // Process through async pipeline
    {
        AsyncNTTPipeline pipe(ntt_size);
        pipe.process(h_input.data(), h_output.data(), total_n, ntt_size);
    }

    // Verify each batch against CPU reference NTT
    int total_pass = 0;
    for (int b = 0; b < num_batches; ++b) {
        std::vector<ff_ref::FpRef> cpu_data(ntt_size);
        for (size_t i = 0; i < ntt_size; ++i) {
            cpu_data[i] = ff_ref::to_montgomery(
                ff_ref::FpRef::from_u32(h_input[b * ntt_size + i].limbs));
        }
        ff_ref::ntt_forward_reference(cpu_data, ntt_size);
        for (size_t i = 0; i < ntt_size; ++i) {
            cpu_data[i] = ff_ref::from_montgomery(cpu_data[i]);
        }

        for (size_t i = 0; i < ntt_size; ++i) {
            ff_ref::FpRef gpu_val = ff_ref::FpRef::from_u32(
                h_output[b * ntt_size + i].limbs);
            if (gpu_val == cpu_data[i]) ++total_pass;
        }
    }

    TEST_ASSERT(total_pass == static_cast<int>(total_n),
        "Async pipeline forward NTT mismatch");
    printf("  Async pipeline forward (n=2^%d, %d batches): %d/%zu matched\n",
        log_n, num_batches, total_pass, total_n);
}

// Verify pipeline and sequential produce identical output
void test_async_pipeline_vs_sequential(int log_n) {
    size_t ntt_size = static_cast<size_t>(1) << log_n;
    int num_batches = 4;
    size_t total_n = ntt_size * num_batches;
    printf("test_async_pipeline_vs_sequential (n=2^%d, %d batches)...\n",
        log_n, num_batches);

    std::vector<FpElement> h_input(total_n);
    for (size_t i = 0; i < total_n; ++i) {
        h_input[i] = FpElement::zero();
        h_input[i].limbs[0] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
    }

    std::vector<FpElement> h_pipe(total_n), h_seq(total_n);

    {
        AsyncNTTPipeline pipe(ntt_size);
        pipe.process(h_input.data(), h_pipe.data(), total_n, ntt_size);
        pipe.process_sequential(h_input.data(), h_seq.data(), total_n, ntt_size);
    }

    int pass_count = 0;
    for (size_t i = 0; i < total_n; ++i) {
        if (h_pipe[i] == h_seq[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(total_n),
        "Pipeline vs sequential output mismatch");
    printf("  Pipeline vs sequential (n=2^%d): %d/%zu matched\n",
        log_n, pass_count, total_n);
}

// ─── v1.2.0: Barrett Arithmetic Correctness Tests ────────────────────────────

// Generate a standard-form field element from a seed (NOT Montgomery form)
static FpElement make_standard_fp(uint32_t seed) {
    FpElement fp = FpElement::zero();
    fp.limbs[0] = seed;
    return fp;
}

// Generate a large standard-form element (all limbs populated, but < p)
static FpElement make_large_standard_fp(uint32_t seed) {
    FpElement fp;
    // Fill limbs with pseudo-random data derived from seed
    uint32_t state = seed;
    for (int i = 0; i < 8; ++i) {
        state = state * 1664525u + 1013904223u;  // LCG
        fp.limbs[i] = state;
    }
    // Ensure < p: clear top 2 bits of MSB (p starts with 0x73 = 0111_0011)
    fp.limbs[7] &= 0x3FFFFFFFu;
    return fp;
}

// CPU reference: Barrett mul on standard-form elements
static FpElement cpu_barrett_mul(const FpElement& a, const FpElement& b) {
    ff_ref::FpRef ra = ff_ref::FpRef::from_u32(a.limbs);
    ff_ref::FpRef rb = ff_ref::FpRef::from_u32(b.limbs);
    ff_ref::FpRef rc = ff_ref::fp_mul_barrett(ra, rb);
    FpElement result;
    rc.to_u32(result.limbs);
    return result;
}

void test_cpu_barrett_self_test() {
    using namespace ff_ref;
    printf("test_cpu_barrett_self_test...\n");

    // Test 1: Barrett(a, 1) = a for small a
    {
        FpRef a = FpRef::from_u64(42);
        FpRef one = FpRef::from_u64(1);
        FpRef result = fp_mul_barrett(a, one);
        TEST_ASSERT(result == a, "Barrett: a * 1 = a");
    }

    // Test 2: Barrett(a, 0) = 0
    {
        FpRef a = FpRef::from_u64(12345);
        FpRef zero = FpRef::zero();
        FpRef result = fp_mul_barrett(a, zero);
        TEST_ASSERT(result == zero, "Barrett: a * 0 = 0");
    }

    // Test 3: Barrett matches Montgomery for small values
    {
        FpRef a_std = FpRef::from_u64(12345);
        FpRef b_std = FpRef::from_u64(67890);
        FpRef barrett_result = fp_mul_barrett(a_std, b_std);

        // Montgomery: convert to mont, multiply, convert back
        FpRef a_mont = to_montgomery(a_std);
        FpRef b_mont = to_montgomery(b_std);
        FpRef mont_result = from_montgomery(fp_mul(a_mont, b_mont));

        TEST_ASSERT(barrett_result == mont_result,
            "Barrett matches Montgomery: 12345 * 67890");
    }

    // Test 4: Barrett matches Montgomery for 1000 random pairs
    {
        int pass = 0;
        for (int i = 0; i < 1000; ++i) {
            FpRef a_std = FpRef::from_u64(static_cast<uint64_t>(i * 7919 + 104729));
            FpRef b_std = FpRef::from_u64(static_cast<uint64_t>(i * 6271 + 299993));
            FpRef barrett = fp_mul_barrett(a_std, b_std);
            FpRef mont = from_montgomery(fp_mul(to_montgomery(a_std), to_montgomery(b_std)));
            if (barrett == mont) ++pass;
        }
        TEST_ASSERT(pass == 1000, "Barrett vs Montgomery: 1000 random pairs");
    }

    // Test 5: Commutativity a*b = b*a
    {
        FpRef a = FpRef::from_u64(999999937);
        FpRef b = FpRef::from_u64(1000000007);
        TEST_ASSERT(fp_mul_barrett(a, b) == fp_mul_barrett(b, a),
            "Barrett commutativity");
    }

    // Test 6: (p-1) * (p-1) mod p
    {
        FpRef pm1;
        pm1.limbs = MOD;
        // p - 1
        uint128_t borrow = 0;
        borrow = uint128_t(pm1.limbs[0]) - 1 - borrow;
        pm1.limbs[0] = static_cast<uint64_t>(borrow);
        borrow = (borrow >> 64) & 1;
        for (int i = 1; i < 4; ++i) {
            borrow = uint128_t(pm1.limbs[i]) - borrow;
            pm1.limbs[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
        // (p-1)^2 mod p = 1  (since (p-1) = -1 mod p, (-1)^2 = 1)
        FpRef result = fp_mul_barrett(pm1, pm1);
        TEST_ASSERT(result == FpRef::from_u64(1), "Barrett: (p-1)*(p-1) = 1");
    }

    // Test 7: Associativity (a*b)*c = a*(b*c)
    {
        FpRef a = FpRef::from_u64(111111);
        FpRef b = FpRef::from_u64(222222);
        FpRef c = FpRef::from_u64(333333);
        FpRef lhs = fp_mul_barrett(fp_mul_barrett(a, b), c);
        FpRef rhs = fp_mul_barrett(a, fp_mul_barrett(b, c));
        TEST_ASSERT(lhs == rhs, "Barrett associativity");
    }

    // Test 8: Distributivity a*(b+c) = a*b + a*c
    {
        FpRef a = FpRef::from_u64(12345);
        FpRef b = FpRef::from_u64(67890);
        FpRef c = FpRef::from_u64(11111);
        FpRef lhs = fp_mul_barrett(a, fp_add(b, c));
        FpRef rhs = fp_add(fp_mul_barrett(a, b), fp_mul_barrett(a, c));
        TEST_ASSERT(lhs == rhs, "Barrett distributivity");
    }
}

void test_ff_mul_barrett_gpu() {
    printf("test_ff_mul_barrett_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_b(N), h_out(N);

    // Standard-form inputs (not Montgomery)
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_large_standard_fp(i * 2 + 1);
        h_b[i] = make_large_standard_fp(i * 2 + 1000);
    }

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_mul_barrett_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FpElement expected = cpu_barrett_mul(h_a[i], h_b[i]);
        if (h_out[i] == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_mul_barrett GPU vs CPU reference mismatch");
    printf("  ff_mul_barrett: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_sqr_barrett_gpu() {
    printf("test_ff_sqr_barrett_gpu...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_large_standard_fp(i * 2 + 1);
    }

    FpElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_sqr_barrett_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FpElement expected = cpu_barrett_mul(h_a[i], h_a[i]);
        if (h_out[i] == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_sqr_barrett GPU vs CPU reference mismatch");
    printf("  ff_sqr_barrett: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_mul_barrett_vs_montgomery() {
    printf("test_ff_mul_barrett_vs_montgomery...\n");

    // Barrett(a, b) in standard form should equal
    // from_montgomery(Montgomery(to_mont(a), to_mont(b)))
    const uint32_t N = 1024;
    std::vector<FpElement> h_a_std(N), h_b_std(N), h_barrett_out(N);
    std::vector<FpElement> h_a_mont(N), h_b_mont(N), h_mont_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a_std[i] = make_large_standard_fp(i * 3 + 7);
        h_b_std[i] = make_large_standard_fp(i * 3 + 2000);
        // Convert to Montgomery form on CPU for Montgomery path
        ff_ref::FpRef ra = ff_ref::to_montgomery(ff_ref::FpRef::from_u32(h_a_std[i].limbs));
        ff_ref::FpRef rb = ff_ref::to_montgomery(ff_ref::FpRef::from_u32(h_b_std[i].limbs));
        ra.to_u32(h_a_mont[i].limbs);
        rb.to_u32(h_b_mont[i].limbs);
    }

    // Barrett GPU path
    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a_std.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_std.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    ff_mul_barrett_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_barrett_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Montgomery GPU path
    CUDA_CHECK(cudaMemcpy(d_a, h_a_mont.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_mont.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    ff_mul_throughput_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_mont_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Convert Montgomery result back to standard form and compare
    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ff_ref::FpRef mont_result = ff_ref::from_montgomery(
            ff_ref::FpRef::from_u32(h_mont_out[i].limbs));
        FpElement mont_std;
        mont_result.to_u32(mont_std.limbs);
        if (h_barrett_out[i] == mont_std) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N,
        "Barrett vs Montgomery GPU cross-validation mismatch");
    printf("  Barrett vs Montgomery: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_ff_mul_barrett_edge_cases() {
    printf("test_ff_mul_barrett_edge_cases...\n");

    // Prepare edge-case pairs on CPU, run on GPU, compare
    struct TestCase {
        FpElement a, b;
        const char* name;
    };

    std::vector<TestCase> cases;

    // Zero * anything = 0
    cases.push_back({FpElement::zero(), make_large_standard_fp(42), "0 * a"});
    cases.push_back({make_large_standard_fp(42), FpElement::zero(), "a * 0"});
    cases.push_back({FpElement::zero(), FpElement::zero(), "0 * 0"});

    // Identity: a * 1 = a
    {
        FpElement one = FpElement::zero();
        one.limbs[0] = 1;
        cases.push_back({make_large_standard_fp(42), one, "a * 1"});
        cases.push_back({one, make_large_standard_fp(42), "1 * a"});
        cases.push_back({one, one, "1 * 1"});
    }

    // Small values
    {
        FpElement two = FpElement::zero(); two.limbs[0] = 2;
        FpElement three = FpElement::zero(); three.limbs[0] = 3;
        cases.push_back({two, three, "2 * 3"});
    }

    // Near-modulus: p-1
    {
        FpElement pm1;
        for (int i = 0; i < 8; ++i) pm1.limbs[i] = 0; // will be set below
        ff_ref::FpRef pm1_ref;
        pm1_ref.limbs = ff_ref::MOD;
        pm1_ref = ff_ref::fp_sub(pm1_ref, ff_ref::FpRef::from_u64(1));
        pm1_ref.to_u32(pm1.limbs);

        FpElement pm2;
        ff_ref::FpRef pm2_ref;
        pm2_ref.limbs = ff_ref::MOD;
        pm2_ref = ff_ref::fp_sub(pm2_ref, ff_ref::FpRef::from_u64(2));
        pm2_ref.to_u32(pm2.limbs);

        cases.push_back({pm1, pm1, "(p-1)*(p-1)"});
        cases.push_back({pm1, pm2, "(p-1)*(p-2)"});

        FpElement two = FpElement::zero(); two.limbs[0] = 2;
        cases.push_back({pm1, two, "(p-1)*2"});
    }

    const uint32_t N = static_cast<uint32_t>(cases.size());
    std::vector<FpElement> h_a(N), h_b(N), h_out(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = cases[i].a;
        h_b[i] = cases[i].b;
    }

    FpElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_mul_barrett_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FpElement expected = cpu_barrett_mul(h_a[i], h_b[i]);
        if (h_out[i] == expected) {
            ++pass_count;
        } else {
            printf("  FAIL: %s — GPU[0]: %08x, CPU[0]: %08x\n",
                cases[i].name, h_out[i].limbs[0], expected.limbs[0]);
        }
    }

    TEST_ASSERT(pass_count == (int)N, "Barrett edge cases mismatch");
    printf("  Barrett edge cases: %d/%u passed\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── v1.2.0 Session 2: Barrett NTT cross-validation ─────────────────────────
// Barrett NTT and Montgomery NTT must produce bitwise-identical output
// for the same standard-form input.

void test_ntt_barrett_vs_montgomery(int log_n) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_barrett_vs_montgomery (n=2^%d=%zu)...\n", log_n, n);

    // Generate test data in standard form
    std::vector<FpElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
    }

    // Run Montgomery NTT
    FpElement* d_mont;
    CUDA_CHECK(cudaMalloc(&d_mont, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_mont, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_mont, n, NTTMode::OPTIMIZED);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_mont(n);
    CUDA_CHECK(cudaMemcpy(h_mont.data(), d_mont, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mont));

    // Run Barrett NTT
    FpElement* d_barrett;
    CUDA_CHECK(cudaMalloc(&d_barrett, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_barrett, n, NTTMode::BARRETT);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_barrett(n);
    CUDA_CHECK(cudaMemcpy(h_barrett.data(), d_barrett, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_barrett));

    // Compare bitwise
    int pass_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_barrett[i] == h_mont[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(n),
        "Barrett NTT vs Montgomery NTT mismatch");
    printf("  Barrett vs Montgomery NTT (n=2^%d): %d/%zu matched\n",
           log_n, pass_count, n);

    if (pass_count != static_cast<int>(n) && n <= 4096) {
        int printed = 0;
        for (size_t i = 0; i < n && printed < 5; ++i) {
            if (h_barrett[i] != h_mont[i]) {
                printf("    [%zu] Barrett: %08x%08x... Montgomery: %08x%08x...\n",
                    i, h_barrett[i].limbs[7], h_barrett[i].limbs[6],
                    h_mont[i].limbs[7], h_mont[i].limbs[6]);
                ++printed;
            }
        }
    }
}

// Barrett NTT inverse roundtrip cross-validation with Montgomery
void test_ntt_barrett_roundtrip_vs_montgomery(int log_n) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_barrett_roundtrip_vs_montgomery (n=2^%d=%zu)...\n", log_n, n);

    // Generate multi-limb test data
    std::vector<FpElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
    }

    // Montgomery roundtrip
    FpElement* d_mont;
    CUDA_CHECK(cudaMalloc(&d_mont, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_mont, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_mont, n, NTTMode::OPTIMIZED);
    ntt_inverse(d_mont, n, NTTMode::OPTIMIZED);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_mont_rt(n);
    CUDA_CHECK(cudaMemcpy(h_mont_rt.data(), d_mont, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mont));

    // Barrett roundtrip
    FpElement* d_barrett;
    CUDA_CHECK(cudaMalloc(&d_barrett, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_barrett, n, NTTMode::BARRETT);
    ntt_inverse(d_barrett, n, NTTMode::BARRETT);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_barrett_rt(n);
    CUDA_CHECK(cudaMemcpy(h_barrett_rt.data(), d_barrett, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_barrett));

    // Both roundtrips should match original data
    int mont_pass = 0, barrett_pass = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_mont_rt[i] == h_data[i]) ++mont_pass;
        if (h_barrett_rt[i] == h_data[i]) ++barrett_pass;
    }

    TEST_ASSERT(barrett_pass == static_cast<int>(n),
        "Barrett roundtrip mismatch");
    printf("  Barrett roundtrip (n=2^%d): %d/%zu, Montgomery: %d/%zu\n",
           log_n, barrett_pass, n, mont_pass, n);
}

void test_ff_mul_barrett_algebraic() {
    printf("test_ff_mul_barrett_algebraic...\n");

    // Test commutativity, associativity, distributivity on GPU
    // using large elements that exercise all limbs
    const uint32_t N = 256;
    std::vector<FpElement> h_a(N), h_b(N), h_c(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_large_standard_fp(i * 5 + 1);
        h_b[i] = make_large_standard_fp(i * 5 + 2);
        h_c[i] = make_large_standard_fp(i * 5 + 3);
    }

    // Commutativity: a*b = b*a
    {
        std::vector<FpElement> h_ab(N), h_ba(N);
        FpElement *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
        CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(FpElement)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

        ff_mul_barrett_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ab.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

        ff_mul_barrett_kernel<<<(N + 255) / 256, 256>>>(d_b, d_a, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ba.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

        int pass = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (h_ab[i] == h_ba[i]) ++pass;
        TEST_ASSERT(pass == (int)N, "Barrett GPU commutativity");
        printf("  Commutativity: %d/%u\n", pass, N);

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_out));
    }

    // Associativity: (a*b)*c = a*(b*c) — on CPU (GPU would need two kernel launches)
    {
        int pass = 0;
        for (uint32_t i = 0; i < N; ++i) {
            FpElement ab = cpu_barrett_mul(h_a[i], h_b[i]);
            FpElement ab_c = cpu_barrett_mul(ab, h_c[i]);
            FpElement bc = cpu_barrett_mul(h_b[i], h_c[i]);
            FpElement a_bc = cpu_barrett_mul(h_a[i], bc);
            if (ab_c == a_bc) ++pass;
        }
        TEST_ASSERT(pass == (int)N, "Barrett associativity (CPU)");
        printf("  Associativity: %d/%u\n", pass, N);
    }

    // Distributivity: a*(b+c) = a*b + a*c — on CPU
    {
        int pass = 0;
        for (uint32_t i = 0; i < N; ++i) {
            // b + c in standard form via CPU reference
            ff_ref::FpRef rb = ff_ref::FpRef::from_u32(h_b[i].limbs);
            ff_ref::FpRef rc = ff_ref::FpRef::from_u32(h_c[i].limbs);
            ff_ref::FpRef bc_sum = ff_ref::fp_add(rb, rc);
            FpElement bpc;
            bc_sum.to_u32(bpc.limbs);

            FpElement lhs = cpu_barrett_mul(h_a[i], bpc);
            FpElement ab = cpu_barrett_mul(h_a[i], h_b[i]);
            FpElement ac = cpu_barrett_mul(h_a[i], h_c[i]);

            ff_ref::FpRef rab = ff_ref::FpRef::from_u32(ab.limbs);
            ff_ref::FpRef rac = ff_ref::FpRef::from_u32(ac.limbs);
            ff_ref::FpRef rhs_ref = ff_ref::fp_add(rab, rac);
            FpElement rhs;
            rhs_ref.to_u32(rhs.limbs);

            if (lhs == rhs) ++pass;
        }
        TEST_ASSERT(pass == (int)N, "Barrett distributivity (CPU)");
        printf("  Distributivity: %d/%u\n", pass, N);
    }
}

// ─── v1.2.0 Session 3: Batched NTT Tests ─────────────────────────────────────

// Batched NTT forward: each NTT in the batch must match sequential single NTT
void test_ntt_batch_vs_sequential(int log_n, int batch_size, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_ntt_batch_vs_sequential (n=2^%d, B=%d, %s)...\n",
           log_n, batch_size, mode_name(mode));

    // Generate distinct random input for each NTT in the batch
    std::vector<FpElement> h_data(total);
    for (size_t i = 0; i < total; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
    }

    // Run batched NTT
    FpElement* d_batch;
    CUDA_CHECK(cudaMalloc(&d_batch, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_batch, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward_batch(d_batch, batch_size, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_batch(total);
    CUDA_CHECK(cudaMemcpy(h_batch.data(), d_batch, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_batch));

    // Run sequential single NTTs for reference
    std::vector<FpElement> h_seq(total);
    FpElement* d_single;
    CUDA_CHECK(cudaMalloc(&d_single, n * sizeof(FpElement)));
    for (int b = 0; b < batch_size; ++b) {
        CUDA_CHECK(cudaMemcpy(d_single, h_data.data() + b * n,
                              n * sizeof(FpElement), cudaMemcpyHostToDevice));
        ntt_forward(d_single, n, mode);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_seq.data() + b * n, d_single,
                              n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_single));

    // Compare: batch output must be bitwise identical to sequential
    int pass_count = 0;
    for (size_t i = 0; i < total; ++i) {
        if (h_batch[i] == h_seq[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(total),
        "Batched NTT vs sequential mismatch");
    printf("  Batch vs sequential (%s, n=2^%d, B=%d): %d/%zu matched\n",
           mode_name(mode), log_n, batch_size, pass_count, total);

    if (pass_count != static_cast<int>(total)) {
        int printed = 0;
        for (size_t i = 0; i < total && printed < 5; ++i) {
            if (h_batch[i] != h_seq[i]) {
                int b = static_cast<int>(i / n);
                size_t local = i % n;
                printf("    NTT[%d][%zu] batch: %08x%08x seq: %08x%08x\n",
                    b, local,
                    h_batch[i].limbs[7], h_batch[i].limbs[6],
                    h_seq[i].limbs[7], h_seq[i].limbs[6]);
                ++printed;
            }
        }
    }
}

// Batched NTT roundtrip: INTT(NTT(x)) = x for all B arrays
void test_ntt_batch_roundtrip(int log_n, int batch_size, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_ntt_batch_roundtrip (n=2^%d, B=%d, %s)...\n",
           log_n, batch_size, mode_name(mode));

    std::vector<FpElement> h_original(total);
    for (size_t i = 0; i < total; ++i) {
        h_original[i] = FpElement::zero();
        h_original[i].limbs[0] = static_cast<uint32_t>((i * 7919u + 104729u) % 1000000007u);
        h_original[i].limbs[1] = static_cast<uint32_t>((i * 6271u + 299993u) % 999999937u);
    }

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_original.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_forward_batch(d_data, batch_size, n, mode);
    ntt_inverse_batch(d_data, batch_size, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_result(total);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    int pass_count = 0;
    for (size_t i = 0; i < total; ++i) {
        if (h_result[i] == h_original[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(total),
        "Batched NTT roundtrip mismatch");
    printf("  Batch roundtrip (%s, n=2^%d, B=%d): %d/%zu matched\n",
           mode_name(mode), log_n, batch_size, pass_count, total);
}

// Batched Barrett NTT == Batched Montgomery NTT (cross-validation)
void test_ntt_batch_barrett_vs_montgomery(int log_n, int batch_size) {
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_ntt_batch_barrett_vs_montgomery (n=2^%d, B=%d)...\n", log_n, batch_size);

    std::vector<FpElement> h_data(total);
    for (size_t i = 0; i < total; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 31337u + 42u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 65537u + 7u) % 999999937u);
    }

    // Batched Montgomery
    FpElement* d_mont;
    CUDA_CHECK(cudaMalloc(&d_mont, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_mont, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward_batch(d_mont, batch_size, n, NTTMode::OPTIMIZED);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_mont(total);
    CUDA_CHECK(cudaMemcpy(h_mont.data(), d_mont, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mont));

    // Batched Barrett
    FpElement* d_barrett;
    CUDA_CHECK(cudaMalloc(&d_barrett, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward_batch(d_barrett, batch_size, n, NTTMode::BARRETT);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_barrett(total);
    CUDA_CHECK(cudaMemcpy(h_barrett.data(), d_barrett, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_barrett));

    int pass_count = 0;
    for (size_t i = 0; i < total; ++i) {
        if (h_barrett[i] == h_mont[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(total),
        "Batched Barrett vs Montgomery mismatch");
    printf("  Batch Barrett vs Montgomery (n=2^%d, B=%d): %d/%zu matched\n",
           log_n, batch_size, pass_count, total);
}

// Batch isolation test: corrupting one NTT should not affect others
void test_ntt_batch_isolation(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    int batch_size = 4;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_ntt_batch_isolation (n=2^%d, B=%d, %s)...\n",
           log_n, batch_size, mode_name(mode));

    // Create two identical input arrays
    std::vector<FpElement> h_clean(total), h_dirty(total);
    for (size_t i = 0; i < total; ++i) {
        h_clean[i] = FpElement::zero();
        h_clean[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
        h_dirty[i] = h_clean[i];
    }

    // Corrupt NTT #2 in the dirty copy (set all elements to p-1)
    for (size_t i = 2 * n; i < 3 * n; ++i) {
        h_dirty[i] = FpElement::zero();
        h_dirty[i].limbs[0] = 0x00000000u;  // p-1 (limbs[0])
        h_dirty[i].limbs[7] = 0x73eda753u;  // different high limb
    }

    // Run batched NTT on both
    FpElement *d_clean, *d_dirty;
    CUDA_CHECK(cudaMalloc(&d_clean, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_dirty, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_clean, h_clean.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dirty, h_dirty.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_forward_batch(d_clean, batch_size, n, mode);
    ntt_forward_batch(d_dirty, batch_size, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_clean_out(total), h_dirty_out(total);
    CUDA_CHECK(cudaMemcpy(h_clean_out.data(), d_clean, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dirty_out.data(), d_dirty, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_clean));
    CUDA_CHECK(cudaFree(d_dirty));

    // NTTs #0, #1, #3 should be identical in both runs (NTT #2 was corrupted)
    int pass_count = 0;
    int total_checked = 0;
    for (int b = 0; b < batch_size; ++b) {
        if (b == 2) continue;  // skip the corrupted one
        for (size_t i = 0; i < n; ++i) {
            if (h_clean_out[b * n + i] == h_dirty_out[b * n + i]) ++pass_count;
            ++total_checked;
        }
    }

    TEST_ASSERT(pass_count == total_checked,
        "Batched NTT isolation failure: corrupting one NTT affected others");
    printf("  Batch isolation (%s, n=2^%d): %d/%d unaffected elements matched\n",
           mode_name(mode), log_n, pass_count, total_checked);
}

// Batch size=1 must match single NTT exactly
void test_ntt_batch_size_one(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_batch_size_one (n=2^%d, %s)...\n", log_n, mode_name(mode));

    std::vector<FpElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
    }

    // Single NTT
    FpElement* d_single;
    CUDA_CHECK(cudaMalloc(&d_single, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_single, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_single, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_single(n);
    CUDA_CHECK(cudaMemcpy(h_single.data(), d_single, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_single));

    // Batch size=1
    FpElement* d_batch;
    CUDA_CHECK(cudaMalloc(&d_batch, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_batch, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward_batch(d_batch, 1, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_batch(n);
    CUDA_CHECK(cudaMemcpy(h_batch.data(), d_batch, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_batch));

    int pass_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_single[i] == h_batch[i]) ++pass_count;
    }

    TEST_ASSERT(pass_count == static_cast<int>(n),
        "Batch size=1 vs single NTT mismatch");
    printf("  Batch B=1 vs single (%s, n=2^%d): %d/%zu matched\n",
           mode_name(mode), log_n, pass_count, n);
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

    // Phase 3: v2 (branchless reduction) kernel correctness tests
    test_ff_mul_v2_gpu();
    test_ff_add_v2_gpu();
    test_ff_sub_v2_gpu();
    test_ff_sqr_v2_gpu();

    // Phase 3: SoA kernel correctness tests
    test_ff_add_soa_gpu();
    test_ff_sub_soa_gpu();
    test_ff_mul_soa_gpu();
    test_ff_sqr_soa_gpu();

    // Phase 4: NTT correctness tests (GPU NTT vs CPU DFT reference)
    test_ntt_forward_gpu(10, NTTMode::NAIVE);  // 2^10 = 1024
    test_ntt_forward_gpu(12, NTTMode::NAIVE);  // 2^12 = 4096
    test_ntt_forward_gpu(15, NTTMode::NAIVE);  // 2^15 = 32768
    test_ntt_forward_gpu(17, NTTMode::NAIVE);  // 2^17 = 131072
    test_ntt_forward_gpu(20, NTTMode::NAIVE);  // 2^20 = 1048576
    test_ntt_roundtrip_gpu(10, NTTMode::NAIVE);
    test_ntt_roundtrip_gpu(12, NTTMode::NAIVE);
    test_ntt_roundtrip_gpu(15, NTTMode::NAIVE);
    test_ntt_roundtrip_gpu(17, NTTMode::NAIVE);
    test_ntt_roundtrip_gpu(20, NTTMode::NAIVE);

    // Phase 5: Optimized shared-memory NTT correctness tests
    test_ntt_forward_gpu(10, NTTMode::OPTIMIZED);
    test_ntt_forward_gpu(12, NTTMode::OPTIMIZED);
    test_ntt_forward_gpu(15, NTTMode::OPTIMIZED);
    test_ntt_forward_gpu(17, NTTMode::OPTIMIZED);
    test_ntt_forward_gpu(20, NTTMode::OPTIMIZED);
    test_ntt_roundtrip_gpu(10, NTTMode::OPTIMIZED);
    test_ntt_roundtrip_gpu(12, NTTMode::OPTIMIZED);
    test_ntt_roundtrip_gpu(15, NTTMode::OPTIMIZED);
    test_ntt_roundtrip_gpu(17, NTTMode::OPTIMIZED);
    test_ntt_roundtrip_gpu(20, NTTMode::OPTIMIZED);

    // v1.1.0: Larger fused kernel tests (K=8, K=9, K=10 paths)
    test_ntt_forward_gpu(8, NTTMode::OPTIMIZED);    // K=8 path (256 elems)
    test_ntt_roundtrip_gpu(8, NTTMode::OPTIMIZED);
    test_ntt_forward_gpu(9, NTTMode::OPTIMIZED);    // K=9 path (512 elems)
    test_ntt_roundtrip_gpu(9, NTTMode::OPTIMIZED);

    // v1.1.0: Large-scale test (K=10 + cooperative outer fusion)
    test_ntt_forward_gpu(22, NTTMode::OPTIMIZED);   // K=10 + 12 outer -> 2 coop launches
    test_ntt_roundtrip_gpu(22, NTTMode::OPTIMIZED);

    // Phase 6: Async pipeline NTT correctness tests
    test_async_pipeline_forward(10, 4);  // 4 batches of 2^10
    test_async_pipeline_forward(15, 4);  // 4 batches of 2^15
    test_async_pipeline_forward(18, 2);  // 2 batches of 2^18
    test_async_pipeline_vs_sequential(12); // pipeline vs sequential match

    // Phase 6: Pinned-memory pipeline test
    {
        size_t ntt_size = 1u << 12;
        int num_batches = 4;
        size_t total_n = ntt_size * num_batches;
        printf("test_async_pipeline_pinned (n=2^12, %d batches)...\n", num_batches);

        FpElement *h_in, *h_out;
        CUDA_CHECK(cudaMallocHost(&h_in, total_n * sizeof(FpElement)));
        CUDA_CHECK(cudaMallocHost(&h_out, total_n * sizeof(FpElement)));
        for (size_t i = 0; i < total_n; ++i) {
            h_in[i] = FpElement::zero();
            h_in[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
        }

        { AsyncNTTPipeline pipe(ntt_size); pipe.process_pinned(h_in, h_out, total_n, ntt_size); }

        // Compare with pipeline (pageable) output
        std::vector<FpElement> h_ref(total_n);
        { AsyncNTTPipeline pipe(ntt_size); pipe.process(h_in, h_ref.data(), total_n, ntt_size); }

        int pass_count = 0;
        for (size_t i = 0; i < total_n; ++i)
            if (h_out[i] == h_ref[i]) ++pass_count;

        TEST_ASSERT(pass_count == static_cast<int>(total_n),
            "Pinned pipeline vs pageable pipeline mismatch");
        printf("  Pinned pipeline (n=2^12): %d/%zu matched\n", pass_count, total_n);

        CUDA_CHECK(cudaFreeHost(h_in));
        CUDA_CHECK(cudaFreeHost(h_out));
    }

    // v1.2.0 Session 1: Barrett arithmetic correctness tests
    test_cpu_barrett_self_test();
    test_ff_mul_barrett_gpu();
    test_ff_sqr_barrett_gpu();
    test_ff_mul_barrett_vs_montgomery();
    test_ff_mul_barrett_edge_cases();
    test_ff_mul_barrett_algebraic();

    // v1.2.0 Session 2: Barrett NTT integration tests
    // Forward correctness: Barrett NTT vs CPU reference
    test_ntt_forward_gpu(10, NTTMode::BARRETT);   // K=10 fused only (1024 elements)
    test_ntt_forward_gpu(12, NTTMode::BARRETT);   // K=10 + outer stages
    test_ntt_forward_gpu(15, NTTMode::BARRETT);
    test_ntt_forward_gpu(17, NTTMode::BARRETT);
    test_ntt_forward_gpu(20, NTTMode::BARRETT);

    // Roundtrip: INTT(NTT(x)) = x for Barrett path
    test_ntt_roundtrip_gpu(10, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(12, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(15, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(17, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(20, NTTMode::BARRETT);

    // Fused kernel path tests (K=8, K=9)
    test_ntt_forward_gpu(8, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(8, NTTMode::BARRETT);
    test_ntt_forward_gpu(9, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(9, NTTMode::BARRETT);

    // Large-scale: K=10 + cooperative outer (2^22)
    test_ntt_forward_gpu(22, NTTMode::BARRETT);
    test_ntt_roundtrip_gpu(22, NTTMode::BARRETT);

    // Cross-validation: Barrett NTT == Montgomery NTT (bitwise)
    test_ntt_barrett_vs_montgomery(10);
    test_ntt_barrett_vs_montgomery(15);
    test_ntt_barrett_vs_montgomery(20);
    test_ntt_barrett_vs_montgomery(22);

    // Roundtrip cross-validation: both paths recover original data
    test_ntt_barrett_roundtrip_vs_montgomery(10);
    test_ntt_barrett_roundtrip_vs_montgomery(15);
    test_ntt_barrett_roundtrip_vs_montgomery(20);

    // v1.2.0 Session 3: Batched NTT tests
    // Batch vs sequential equivalence (Montgomery)
    test_ntt_batch_vs_sequential(10, 4, NTTMode::OPTIMIZED);   // K=10, no outer
    test_ntt_batch_vs_sequential(15, 4, NTTMode::OPTIMIZED);   // K=10 + outer
    test_ntt_batch_vs_sequential(18, 2, NTTMode::OPTIMIZED);   // larger size
    test_ntt_batch_vs_sequential(20, 2, NTTMode::OPTIMIZED);

    // Batch vs sequential equivalence (Barrett)
    test_ntt_batch_vs_sequential(10, 4, NTTMode::BARRETT);
    test_ntt_batch_vs_sequential(15, 4, NTTMode::BARRETT);
    test_ntt_batch_vs_sequential(18, 2, NTTMode::BARRETT);
    test_ntt_batch_vs_sequential(20, 2, NTTMode::BARRETT);

    // Batch roundtrip (Montgomery)
    test_ntt_batch_roundtrip(10, 4, NTTMode::OPTIMIZED);
    test_ntt_batch_roundtrip(15, 4, NTTMode::OPTIMIZED);
    test_ntt_batch_roundtrip(20, 2, NTTMode::OPTIMIZED);

    // Batch roundtrip (Barrett)
    test_ntt_batch_roundtrip(10, 4, NTTMode::BARRETT);
    test_ntt_batch_roundtrip(15, 4, NTTMode::BARRETT);
    test_ntt_batch_roundtrip(20, 2, NTTMode::BARRETT);

    // Batch Barrett vs Montgomery cross-validation
    test_ntt_batch_barrett_vs_montgomery(10, 4);
    test_ntt_batch_barrett_vs_montgomery(15, 4);
    test_ntt_batch_barrett_vs_montgomery(20, 2);

    // Batch size=1 degenerate case
    test_ntt_batch_size_one(15, NTTMode::OPTIMIZED);
    test_ntt_batch_size_one(15, NTTMode::BARRETT);

    // Batch isolation: corrupting one NTT doesn't affect others
    test_ntt_batch_isolation(15, NTTMode::OPTIMIZED);
    test_ntt_batch_isolation(15, NTTMode::BARRETT);

    // Larger batch sizes
    test_ntt_batch_vs_sequential(15, 8, NTTMode::BARRETT);
    test_ntt_batch_roundtrip(15, 8, NTTMode::BARRETT);

    // Large-scale batch test (2^22, B=2)
    test_ntt_batch_vs_sequential(22, 2, NTTMode::OPTIMIZED);
    test_ntt_batch_vs_sequential(22, 2, NTTMode::BARRETT);
    test_ntt_batch_roundtrip(22, 2, NTTMode::BARRETT);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
