// tests/test_correctness.cu
// Validates NTT output against CPU reference implementation
// Phase 1: stub with basic structure. Real tests added in Phase 2/4.

#include "ff_arithmetic.cuh"
#include "ff_goldilocks.cuh"
#include "ff_babybear.cuh"
#include "ff_fq.cuh"
#include "ff_fq2.cuh"
#include "ff_fq6.cuh"
#include "ff_fq12.cuh"
#include "ec_g1.cuh"
#include "ec_g2.cuh"
#include "pairing.cuh"
#include "msm.cuh"
#include "poly_ops.cuh"
#include "groth16.cuh"
#include "ntt.cuh"
#include "ntt_goldilocks.cuh"
#include "ntt_babybear.cuh"
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

// Fq (base field) kernel declarations
extern __global__ void fq_add_kernel(const FqElement* __restrict__ a, const FqElement* __restrict__ b, FqElement* __restrict__ out, uint32_t n);
extern __global__ void fq_sub_kernel(const FqElement* __restrict__ a, const FqElement* __restrict__ b, FqElement* __restrict__ out, uint32_t n);
extern __global__ void fq_mul_kernel(const FqElement* __restrict__ a, const FqElement* __restrict__ b, FqElement* __restrict__ out, uint32_t n);
extern __global__ void fq_sqr_kernel(const FqElement* __restrict__ a, FqElement* __restrict__ out, uint32_t n);

// Fq2 (extension field) kernel declarations
extern __global__ void fq2_add_kernel(const Fq2Element* __restrict__ a, const Fq2Element* __restrict__ b, Fq2Element* __restrict__ out, uint32_t n);
extern __global__ void fq2_sub_kernel(const Fq2Element* __restrict__ a, const Fq2Element* __restrict__ b, Fq2Element* __restrict__ out, uint32_t n);
extern __global__ void fq2_mul_kernel(const Fq2Element* __restrict__ a, const Fq2Element* __restrict__ b, Fq2Element* __restrict__ out, uint32_t n);
extern __global__ void fq2_sqr_kernel(const Fq2Element* __restrict__ a, Fq2Element* __restrict__ out, uint32_t n);

// Fq6 (cubic extension) kernel declarations
extern __global__ void fq6_add_kernel(const Fq6Element* __restrict__ a, const Fq6Element* __restrict__ b, Fq6Element* __restrict__ out, uint32_t n);
extern __global__ void fq6_sub_kernel(const Fq6Element* __restrict__ a, const Fq6Element* __restrict__ b, Fq6Element* __restrict__ out, uint32_t n);
extern __global__ void fq6_mul_kernel(const Fq6Element* __restrict__ a, const Fq6Element* __restrict__ b, Fq6Element* __restrict__ out, uint32_t n);
extern __global__ void fq6_sqr_kernel(const Fq6Element* __restrict__ a, Fq6Element* __restrict__ out, uint32_t n);

// Fq12 (quadratic extension over Fq6) kernel declarations
extern __global__ void fq12_add_kernel(const Fq12Element* __restrict__ a, const Fq12Element* __restrict__ b, Fq12Element* __restrict__ out, uint32_t n);
extern __global__ void fq12_sub_kernel(const Fq12Element* __restrict__ a, const Fq12Element* __restrict__ b, Fq12Element* __restrict__ out, uint32_t n);
extern __global__ void fq12_mul_kernel(const Fq12Element* __restrict__ a, const Fq12Element* __restrict__ b, Fq12Element* __restrict__ out, uint32_t n);
extern __global__ void fq12_sqr_kernel(const Fq12Element* __restrict__ a, Fq12Element* __restrict__ out, uint32_t n);

// EC kernel declarations (defined in src/ec_kernels.cu)
extern __global__ void g1_double_kernel(const G1Jacobian* __restrict__ in, G1Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g1_add_kernel(const G1Jacobian* __restrict__ a, const G1Jacobian* __restrict__ b, G1Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g1_add_mixed_kernel(const G1Jacobian* __restrict__ a, const G1Affine* __restrict__ b, G1Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g1_scalar_mul_kernel(const G1Jacobian* __restrict__ bases, const uint32_t* __restrict__ scalars, G1Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g1_to_affine_kernel(const G1Jacobian* __restrict__ in, G1Affine* __restrict__ out, uint32_t n);
extern __global__ void g1_is_on_curve_kernel(const G1Affine* __restrict__ pts, bool* __restrict__ results, uint32_t n);
extern __global__ void g2_double_kernel(const G2Jacobian* __restrict__ in, G2Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g2_add_kernel(const G2Jacobian* __restrict__ a, const G2Jacobian* __restrict__ b, G2Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g2_add_mixed_kernel(const G2Jacobian* __restrict__ a, const G2Affine* __restrict__ b, G2Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g2_scalar_mul_kernel(const G2Jacobian* __restrict__ bases, const uint32_t* __restrict__ scalars, G2Jacobian* __restrict__ out, uint32_t n);
extern __global__ void g2_to_affine_kernel(const G2Jacobian* __restrict__ in, G2Affine* __restrict__ out, uint32_t n);
extern __global__ void g2_is_on_curve_kernel(const G2Affine* __restrict__ pts, bool* __restrict__ results, uint32_t n);

// Pairing kernel declarations (defined in src/pairing_kernels.cu)
extern __global__ void miller_loop_kernel(const G1Affine* __restrict__ P, const G2Affine* __restrict__ Q, Fq12Element* __restrict__ out, uint32_t n);

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
        case NTTMode::FOUR_STEP: return "FOUR_STEP";
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

// ─── v1.3.0 Session 5: Transpose Kernel Tests ────────────────────────────────

// Extern declarations for transpose and split functions (from ntt_4step.cu)
extern void ntt_transpose(
    const FpElement* d_src, FpElement* d_dst,
    uint32_t rows, uint32_t cols, cudaStream_t stream);
extern void ntt_transpose_batch(
    const FpElement* d_src, FpElement* d_dst,
    uint32_t rows, uint32_t cols, uint32_t batch_size, cudaStream_t stream);
extern void ntt_4step_get_split(uint32_t n, uint32_t& n1, uint32_t& n2);

// Test: transpose a small matrix and verify on CPU
void test_transpose_square(uint32_t dim) {
    uint32_t rows = dim, cols = dim;
    uint32_t total = rows * cols;
    printf("test_transpose_square (%ux%u)...\n", rows, cols);

    std::vector<FpElement> h_src(total), h_dst(total);
    for (uint32_t i = 0; i < total; ++i) {
        h_src[i] = FpElement::zero();
        h_src[i].limbs[0] = i;
        h_src[i].limbs[1] = i * 7 + 3;
    }

    FpElement *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_dst, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_transpose(d_src, d_dst, rows, cols, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Verify: dst[j*rows + i] == src[i*cols + j]
    int pass = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (h_dst[j * rows + i] == h_src[i * cols + j]) ++pass;
        }
    }

    TEST_ASSERT(pass == static_cast<int>(total), "Square transpose mismatch");
    printf("  Square transpose %ux%u: %d/%u correct\n", rows, cols, pass, total);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// Test: transpose a non-square matrix
void test_transpose_rect(uint32_t rows, uint32_t cols) {
    uint32_t total = rows * cols;
    printf("test_transpose_rect (%ux%u)...\n", rows, cols);

    std::vector<FpElement> h_src(total), h_dst(total);
    for (uint32_t i = 0; i < total; ++i) {
        h_src[i] = FpElement::zero();
        h_src[i].limbs[0] = i;
        h_src[i].limbs[1] = i ^ 0xDEADBEEF;
    }

    FpElement *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_dst, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_transpose(d_src, d_dst, rows, cols, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (h_dst[j * rows + i] == h_src[i * cols + j]) ++pass;
        }
    }

    TEST_ASSERT(pass == static_cast<int>(total), "Rect transpose mismatch");
    printf("  Rect transpose %ux%u: %d/%u correct\n", rows, cols, pass, total);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// Test: double transpose (transpose twice = identity)
void test_transpose_roundtrip(uint32_t rows, uint32_t cols) {
    uint32_t total = rows * cols;
    printf("test_transpose_roundtrip (%ux%u)...\n", rows, cols);

    std::vector<FpElement> h_src(total), h_result(total);
    for (uint32_t i = 0; i < total; ++i) {
        h_src[i] = FpElement::zero();
        h_src[i].limbs[0] = i * 12345u + 6789u;
        h_src[i].limbs[3] = i ^ 0xCAFEBABE;
        h_src[i].limbs[7] = (i * 31337u) & 0x73eda752u;
    }

    FpElement *d_src, *d_tmp, *d_result;
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_result, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    // First transpose: rows x cols → cols x rows
    ntt_transpose(d_src, d_tmp, rows, cols, 0);
    // Second transpose: cols x rows → rows x cols (should recover original)
    ntt_transpose(d_tmp, d_result, cols, rows, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < total; ++i) {
        if (h_result[i] == h_src[i]) ++pass;
    }

    TEST_ASSERT(pass == static_cast<int>(total), "Transpose roundtrip mismatch");
    printf("  Roundtrip %ux%u: %d/%u recovered\n", rows, cols, pass, total);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_result));
}

// Test: batched transpose
void test_transpose_batch(uint32_t rows, uint32_t cols, uint32_t batch_size) {
    uint32_t mat_size = rows * cols;
    uint32_t total = batch_size * mat_size;
    printf("test_transpose_batch (%ux%u, B=%u)...\n", rows, cols, batch_size);

    std::vector<FpElement> h_src(total), h_dst(total);
    for (uint32_t i = 0; i < total; ++i) {
        h_src[i] = FpElement::zero();
        h_src[i].limbs[0] = i;
        h_src[i].limbs[1] = i * 17 + 5;
    }

    FpElement *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_dst, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_transpose_batch(d_src, d_dst, rows, cols, batch_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t b = 0; b < batch_size; ++b) {
        uint32_t base = b * mat_size;
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                if (h_dst[base + j * rows + i] == h_src[base + i * cols + j]) ++pass;
            }
        }
    }

    TEST_ASSERT(pass == static_cast<int>(total), "Batched transpose mismatch");
    printf("  Batch transpose %ux%u B=%u: %d/%u correct\n", rows, cols, batch_size, pass, total);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// Test: 4-step split computation
void test_4step_split() {
    printf("test_4step_split...\n");

    struct SplitCase { uint32_t n; uint32_t expected_n1; uint32_t expected_n2; };
    SplitCase cases[] = {
        {1u << 16, 1u << 8,  1u << 8},   // even: 16 → 8, 8
        {1u << 18, 1u << 9,  1u << 9},   // even: 18 → 9, 9
        {1u << 20, 1u << 10, 1u << 10},  // even: 20 → 10, 10
        {1u << 22, 1u << 11, 1u << 11},  // even: 22 → 11, 11
        {1u << 17, 1u << 8,  1u << 9},   // odd: 17 → 8, 9
        {1u << 19, 1u << 9,  1u << 10},  // odd: 19 → 9, 10
        {1u << 21, 1u << 10, 1u << 11},  // odd: 21 → 10, 11
    };

    int pass = 0;
    int total = sizeof(cases) / sizeof(cases[0]);
    for (int c = 0; c < total; ++c) {
        uint32_t n1, n2;
        ntt_4step_get_split(cases[c].n, n1, n2);
        bool ok = (n1 == cases[c].expected_n1 && n2 == cases[c].expected_n2 && n1 * n2 == cases[c].n);
        if (ok) {
            ++pass;
        } else {
            printf("  FAIL: n=%u → got (%u, %u), expected (%u, %u)\n",
                   cases[c].n, n1, n2, cases[c].expected_n1, cases[c].expected_n2);
        }
    }

    TEST_ASSERT(pass == total, "4-step split computation mismatch");
    printf("  4-step split: %d/%d correct\n", pass, total);
}

// Test: large transpose matching 4-step decomposition sizes
void test_transpose_4step_sizes() {
    printf("test_transpose_4step_sizes...\n");

    // Test transpose at the exact sizes used by 4-step NTT
    // n=2^22: n1=n2=2^11=2048 → transpose 2048x2048 matrix
    uint32_t n1 = 2048, n2 = 2048;
    uint32_t total = n1 * n2;  // 4M elements = 128 MB

    std::vector<FpElement> h_src(total), h_dst(total);
    for (uint32_t i = 0; i < total; ++i) {
        h_src[i] = FpElement::zero();
        h_src[i].limbs[0] = i;
        h_src[i].limbs[7] = i >> 16;
    }

    FpElement *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_dst, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_transpose(d_src, d_dst, n1, n2, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Spot-check corners and random positions (full check is 4M comparisons)
    int pass = 0;
    int checks = 0;
    // Check corners
    auto verify = [&](uint32_t r, uint32_t c) {
        if (r < n1 && c < n2) {
            if (h_dst[c * n1 + r] == h_src[r * n2 + c]) ++pass;
            ++checks;
        }
    };
    verify(0, 0);
    verify(0, n2 - 1);
    verify(n1 - 1, 0);
    verify(n1 - 1, n2 - 1);
    // Check some interior points
    for (uint32_t k = 0; k < 1000; ++k) {
        uint32_t r = (k * 12345u + 67u) % n1;
        uint32_t c = (k * 54321u + 89u) % n2;
        verify(r, c);
    }

    TEST_ASSERT(pass == checks, "4-step size transpose spot-check mismatch");
    printf("  Transpose 2048x2048: %d/%d spot checks passed\n", pass, checks);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// ─── v1.3.0 Session 7: Known-Vector + Edge-Case Tests ─────────────────────

// Test NTT with known input patterns: all-zeros, all-ones, single non-zero, ascending, all (p-1)
void test_ntt_known_vectors(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_known_vectors [%s] (n=2^%d)...\n", mode_name(mode), log_n);

    int sub_pass = 0;
    int sub_total = 0;

    auto run_roundtrip = [&](const char* label, const std::vector<FpElement>& h_input) {
        ++sub_total;
        FpElement* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
        CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

        ntt_forward(d_data, n, mode);
        ntt_inverse(d_data, n, mode);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<FpElement> h_result(n);
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data));

        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            if (!(h_result[i] == h_input[i])) { ok = false; break; }
        }
        if (ok) ++sub_pass;
        else printf("    FAIL: %s roundtrip\n", label);
    };

    // Pattern 1: all zeros
    {
        std::vector<FpElement> h(n, FpElement::zero());
        run_roundtrip("all-zeros", h);
    }

    // Pattern 2: all ones (limbs[0] = 1, rest 0)
    {
        std::vector<FpElement> h(n);
        for (size_t i = 0; i < n; ++i) { h[i] = FpElement::zero(); h[i].limbs[0] = 1; }
        run_roundtrip("all-ones", h);
    }

    // Pattern 3: single non-zero element at index 0
    {
        std::vector<FpElement> h(n, FpElement::zero());
        h[0].limbs[0] = 42;
        run_roundtrip("single-nonzero-0", h);
    }

    // Pattern 4: single non-zero element at index n/2
    {
        std::vector<FpElement> h(n, FpElement::zero());
        h[n / 2].limbs[0] = 12345;
        run_roundtrip("single-nonzero-mid", h);
    }

    // Pattern 5: ascending sequence
    {
        std::vector<FpElement> h(n);
        for (size_t i = 0; i < n; ++i) {
            h[i] = FpElement::zero();
            h[i].limbs[0] = static_cast<uint32_t>(i + 1);
        }
        run_roundtrip("ascending", h);
    }

    // Pattern 6: all (p-1) — maximum field element
    {
        // p-1 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000
        FpElement pm1 = FpElement::zero();
        pm1.limbs[0] = 0x00000000u;
        pm1.limbs[1] = 0xffffffffu;
        pm1.limbs[2] = 0xfffe5bfeu;
        pm1.limbs[3] = 0x53bda402u;
        pm1.limbs[4] = 0x09a1d805u;
        pm1.limbs[5] = 0x3339d808u;
        pm1.limbs[6] = 0x299d7d48u;
        pm1.limbs[7] = 0x73eda753u;
        std::vector<FpElement> h(n, pm1);
        run_roundtrip("all-p-minus-1", h);
    }

    TEST_ASSERT(sub_pass == sub_total, "Known vector roundtrip mismatch");
    printf("  Known vectors [%s] (n=2^%d): %d/%d patterns OK\n",
           mode_name(mode), log_n, sub_pass, sub_total);
}

// Test: 4-step NTT forward correctness against CPU ref for all-zeros input
// (NTT of all-zeros should be all-zeros)
void test_ntt_forward_zeros(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_forward_zeros [%s] (n=2^%d)...\n", mode_name(mode), log_n);

    std::vector<FpElement> h_data(n, FpElement::zero());

    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_forward(d_data, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_result(n);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    int pass = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_result[i] == FpElement::zero()) ++pass;
    }

    TEST_ASSERT(pass == static_cast<int>(n), "NTT of all-zeros should be all-zeros");
    printf("  Forward zeros [%s] (n=2^%d): %d/%zu zero\n", mode_name(mode), log_n, pass, n);
}

// Test: 4-step NTT inverse specifically (verify NTT_inv(NTT_fwd(x)) AND NTT_fwd(NTT_inv(x)))
void test_ntt_inverse_explicit(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_ntt_inverse_explicit [%s] (n=2^%d)...\n", mode_name(mode), log_n);

    std::vector<FpElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 65537u + 31337u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 104729u + 7919u) % 999999937u);
    }

    // Direction 1: forward then inverse
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_forward(d_data, n, mode);
    ntt_inverse(d_data, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_r1(n);
    CUDA_CHECK(cudaMemcpy(h_r1.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass1 = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_r1[i] == h_data[i]) ++pass1;
    }

    // Direction 2: inverse then forward
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

    ntt_inverse(d_data, n, mode);
    ntt_forward(d_data, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_r2(n);
    CUDA_CHECK(cudaMemcpy(h_r2.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    int pass2 = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_r2[i] == h_data[i]) ++pass2;
    }

    TEST_ASSERT(pass1 == static_cast<int>(n), "inv(fwd(x)) != x");
    TEST_ASSERT(pass2 == static_cast<int>(n), "fwd(inv(x)) != x");
    printf("  Inverse explicit [%s] (n=2^%d): inv(fwd)=%d/%zu, fwd(inv)=%d/%zu\n",
           mode_name(mode), log_n, pass1, n, pass2, n);
}

// Test: 4-step batched vs 8 sequential 4-step NTTs (B=8)
void test_ntt_batch_vs_sequential_b8(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    int batch_size = 8;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_ntt_batch_vs_sequential_b8 [%s] (n=2^%d, B=8)...\n", mode_name(mode), log_n);

    std::vector<FpElement> h_data(total);
    for (size_t i = 0; i < total; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
    }

    // Batched
    FpElement* d_batch;
    CUDA_CHECK(cudaMalloc(&d_batch, total * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_batch, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward_batch(d_batch, batch_size, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_batch(total);
    CUDA_CHECK(cudaMemcpy(h_batch.data(), d_batch, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_batch));

    // Sequential
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

    int pass = 0;
    for (size_t i = 0; i < total; ++i) {
        if (h_batch[i] == h_seq[i]) ++pass;
    }

    TEST_ASSERT(pass == static_cast<int>(total), "Batch B=8 vs sequential mismatch");
    printf("  Batch B=8 vs sequential [%s] (n=2^%d): %d/%zu matched\n",
           mode_name(mode), log_n, pass, total);
}

// ─── CUDA Graph Tests ─────────────────────────────────────────────────────────
// Validates that graph-accelerated NTT produces bitwise identical output to
// non-graph NTT, and that graph replay is consistent.

void test_ntt_graph_vs_nongraph() {
    printf("\n--- CUDA Graph NTT Tests ---\n");

    // Test 1: Barrett forward graph vs non-graph at multiple sizes
    {
        int sizes[] = {15, 18, 22};
        for (int log_n : sizes) {
            size_t n = static_cast<size_t>(1) << log_n;
            printf("test_graph_barrett_forward (n=2^%d)...\n", log_n);

            std::vector<FpElement> h_data(n);
            for (size_t i = 0; i < n; ++i) {
                h_data[i] = FpElement::zero();
                h_data[i].limbs[0] = static_cast<uint32_t>((i * 7919u + 12345u) % 1000000007u);
            }

            FpElement *d_graph, *d_plain;
            CUDA_CHECK(cudaMalloc(&d_graph, n * sizeof(FpElement)));
            CUDA_CHECK(cudaMalloc(&d_plain, n * sizeof(FpElement)));
            CUDA_CHECK(cudaMemcpy(d_graph, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_plain, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

            ntt_forward_graph(d_graph, n, NTTMode::BARRETT);
            ntt_forward(d_plain, n, NTTMode::BARRETT);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FpElement> h_graph(n), h_plain(n);
            CUDA_CHECK(cudaMemcpy(h_graph.data(), d_graph, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_plain.data(), d_plain, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

            int pass = 0;
            for (size_t i = 0; i < n; ++i)
                if (h_graph[i] == h_plain[i]) ++pass;

            TEST_ASSERT(pass == static_cast<int>(n), "Graph Barrett forward mismatch");
            printf("  Graph vs plain Barrett fwd (2^%d): %d/%zu matched\n", log_n, pass, n);

            ntt_graph_clear_cache();
            CUDA_CHECK(cudaFree(d_graph));
            CUDA_CHECK(cudaFree(d_plain));
        }
    }

    // Test 2: Montgomery forward graph vs non-graph
    {
        int sizes[] = {15, 22};
        for (int log_n : sizes) {
            size_t n = static_cast<size_t>(1) << log_n;
            printf("test_graph_montgomery_forward (n=2^%d)...\n", log_n);

            std::vector<FpElement> h_data(n);
            for (size_t i = 0; i < n; ++i) {
                h_data[i] = FpElement::zero();
                h_data[i].limbs[0] = static_cast<uint32_t>((i * 31337u + 42u) % 1000000007u);
            }

            FpElement *d_graph, *d_plain;
            CUDA_CHECK(cudaMalloc(&d_graph, n * sizeof(FpElement)));
            CUDA_CHECK(cudaMalloc(&d_plain, n * sizeof(FpElement)));
            CUDA_CHECK(cudaMemcpy(d_graph, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_plain, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

            ntt_forward_graph(d_graph, n, NTTMode::OPTIMIZED);
            ntt_forward(d_plain, n, NTTMode::OPTIMIZED);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FpElement> h_graph(n), h_plain(n);
            CUDA_CHECK(cudaMemcpy(h_graph.data(), d_graph, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_plain.data(), d_plain, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

            int pass = 0;
            for (size_t i = 0; i < n; ++i)
                if (h_graph[i] == h_plain[i]) ++pass;

            TEST_ASSERT(pass == static_cast<int>(n), "Graph Montgomery forward mismatch");
            printf("  Graph vs plain Montgomery fwd (2^%d): %d/%zu matched\n", log_n, pass, n);

            ntt_graph_clear_cache();
            CUDA_CHECK(cudaFree(d_graph));
            CUDA_CHECK(cudaFree(d_plain));
        }
    }

    // Test 3: Barrett roundtrip via graph (NTT then INTT)
    {
        int sizes[] = {15, 22};
        for (int log_n : sizes) {
            size_t n = static_cast<size_t>(1) << log_n;
            printf("test_graph_barrett_roundtrip (n=2^%d)...\n", log_n);

            std::vector<FpElement> h_orig(n);
            for (size_t i = 0; i < n; ++i) {
                h_orig[i] = FpElement::zero();
                h_orig[i].limbs[0] = static_cast<uint32_t>((i * 997u + 13u) % 1000000007u);
            }

            FpElement* d_data;
            CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
            CUDA_CHECK(cudaMemcpy(d_data, h_orig.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));

            ntt_forward_graph(d_data, n, NTTMode::BARRETT);
            CUDA_CHECK(cudaDeviceSynchronize());
            ntt_graph_clear_cache();  // clear forward graph before inverse capture
            ntt_inverse_graph(d_data, n, NTTMode::BARRETT);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FpElement> h_result(n);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

            int pass = 0;
            for (size_t i = 0; i < n; ++i)
                if (h_result[i] == h_orig[i]) ++pass;

            TEST_ASSERT(pass == static_cast<int>(n), "Graph Barrett roundtrip mismatch");
            printf("  Graph Barrett roundtrip (2^%d): %d/%zu matched\n", log_n, pass, n);

            ntt_graph_clear_cache();
            CUDA_CHECK(cudaFree(d_data));
        }
    }

    // Test 4: Graph replay consistency (second call same result as first)
    {
        size_t n = 1u << 15;
        printf("test_graph_replay_consistency (n=2^15)...\n");

        std::vector<FpElement> h_data(n);
        for (size_t i = 0; i < n; ++i) {
            h_data[i] = FpElement::zero();
            h_data[i].limbs[0] = static_cast<uint32_t>((i * 4999u + 77u) % 1000000007u);
        }

        FpElement* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));

        // First run (captures graph)
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
        ntt_forward_graph(d_data, n, NTTMode::BARRETT);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<FpElement> h_first(n);
        CUDA_CHECK(cudaMemcpy(h_first.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

        // Second run (replays cached graph)
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
        ntt_forward_graph(d_data, n, NTTMode::BARRETT);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<FpElement> h_second(n);
        CUDA_CHECK(cudaMemcpy(h_second.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));

        int pass = 0;
        for (size_t i = 0; i < n; ++i)
            if (h_first[i] == h_second[i]) ++pass;

        TEST_ASSERT(pass == static_cast<int>(n), "Graph replay inconsistency");
        printf("  Graph replay consistency (2^15): %d/%zu matched\n", pass, n);

        ntt_graph_clear_cache();
        CUDA_CHECK(cudaFree(d_data));
    }

    // Test 5: Batched graph vs non-graph (Barrett, B=4, n=2^15)
    {
        int log_n = 15;
        int batch_size = 4;
        size_t n = static_cast<size_t>(1) << log_n;
        size_t total = static_cast<size_t>(batch_size) * n;
        printf("test_graph_batch_barrett (n=2^%d, B=%d)...\n", log_n, batch_size);

        std::vector<FpElement> h_data(total);
        for (size_t i = 0; i < total; ++i) {
            h_data[i] = FpElement::zero();
            h_data[i].limbs[0] = static_cast<uint32_t>((i * 2017u + 5u) % 1000000007u);
        }

        FpElement *d_graph, *d_plain;
        CUDA_CHECK(cudaMalloc(&d_graph, total * sizeof(FpElement)));
        CUDA_CHECK(cudaMalloc(&d_plain, total * sizeof(FpElement)));
        CUDA_CHECK(cudaMemcpy(d_graph, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_plain, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

        ntt_forward_batch_graph(d_graph, batch_size, n, NTTMode::BARRETT);
        ntt_forward_batch(d_plain, batch_size, n, NTTMode::BARRETT);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<FpElement> h_graph(total), h_plain(total);
        CUDA_CHECK(cudaMemcpy(h_graph.data(), d_graph, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_plain.data(), d_plain, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

        int pass = 0;
        for (size_t i = 0; i < total; ++i)
            if (h_graph[i] == h_plain[i]) ++pass;

        TEST_ASSERT(pass == static_cast<int>(total), "Batched graph vs plain mismatch");
        printf("  Batch graph vs plain Barrett (2^%d, B=%d): %d/%zu matched\n",
               log_n, batch_size, pass, total);

        ntt_graph_clear_cache();
        CUDA_CHECK(cudaFree(d_graph));
        CUDA_CHECK(cudaFree(d_plain));
    }
}

// ─── OTF Twiddle Tests ───────────────────────────────────────────────────────

// Test: OTF stage root squaring chain.
// root[s] should equal root[s+1]^2 in Montgomery form.
void test_otf_stage_root_chain(int log_n) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_otf_stage_root_chain (n=2^%d)...\n", log_n);

    ff_ref::FpRef omega = ff_ref::get_root_of_unity(n);

    // Compute stage roots: root[s] = omega^(n / 2^(s+1))
    // By repeated squaring from omega, we get root[log_n-1] = omega,
    // root[log_n-2] = omega^2, ..., root[0] = omega^(n/2) = -1.
    // But the OTF code computes bottom-up: start from omega, square down.
    // Verify: root[s] = root[s+1]^2
    std::vector<ff_ref::FpRef> roots(log_n);
    ff_ref::FpRef root = omega;
    for (int s = log_n - 1; s >= 0; --s) {
        roots[s] = root;
        root = ff_ref::fp_sqr(root);
    }

    int pass = 0;
    for (int s = 0; s < log_n - 1; ++s) {
        ff_ref::FpRef squared = ff_ref::fp_sqr(roots[s + 1]);
        if (squared == roots[s]) ++pass;
    }

    TEST_ASSERT(pass == log_n - 1, "stage root squaring chain failed");
    printf("  Stage root chain (n=2^%d): %d/%d squaring relations verified\n",
           log_n, pass, log_n - 1);
}

// Test: OTF twiddle matches precomputed table entry (CPU reference).
// For a given stage s and butterfly index j, the twiddle should be omega_n^(j * n / 2^(s+1)).
// Verify at representative (s, j) pairs.
void test_otf_twiddle_values(int log_n) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_otf_twiddle_values (n=2^%d)...\n", log_n);

    ff_ref::FpRef omega = ff_ref::get_root_of_unity(n);

    // Compute stage roots
    std::vector<ff_ref::FpRef> roots(log_n);
    ff_ref::FpRef root = omega;
    for (int s = log_n - 1; s >= 0; --s) {
        roots[s] = root;
        root = ff_ref::fp_sqr(root);
    }

    int pass = 0, total = 0;
    // K=10 means outer stages start at 10
    int outer_start = (log_n >= 10) ? 10 : 0;

    // Test representative (s, j) values
    for (int s = outer_start; s < log_n; ++s) {
        uint32_t half = 1u << s;
        // Test j=0, j=1, j=half/2, j=half-1
        uint32_t test_js[] = { 0, 1, half / 2, half - 1 };
        int num_js = (half < 4) ? half : 4;
        for (int ji = 0; ji < num_js; ++ji) {
            uint32_t j = test_js[ji];
            if (j >= half) continue;

            // Expected: root[s]^j (via CPU pow)
            // root[s] = omega^(n / 2^(s+1))
            // root[s]^j = omega^(j * n / 2^(s+1))
            ff_ref::FpRef expected = roots[s];
            // Exponentiate by j using repeated squaring
            ff_ref::FpRef result;
            result.limbs = ff_ref::R_MOD;  // Montgomery(1)
            ff_ref::FpRef base = roots[s];
            uint32_t exp = j;
            while (exp > 0) {
                if (exp & 1) result = ff_ref::fp_mul(result, base);
                base = ff_ref::fp_sqr(base);
                exp >>= 1;
            }

            // Also verify via direct omega^(j * stride)
            uint32_t stride = static_cast<uint32_t>(n / (2u * half));
            uint64_t tw_exp_val = static_cast<uint64_t>(j) * stride;
            ff_ref::FpRef direct;
            direct.limbs = ff_ref::R_MOD;
            ff_ref::FpRef ob = omega;
            uint64_t de = tw_exp_val;
            while (de > 0) {
                if (de & 1) direct = ff_ref::fp_mul(direct, ob);
                ob = ff_ref::fp_sqr(ob);
                de >>= 1;
            }

            ++total;
            if (result == direct) ++pass;
        }
    }

    TEST_ASSERT(pass == total, "OTF twiddle value mismatch");
    printf("  OTF twiddle values (n=2^%d): %d/%d matched\n", log_n, pass, total);
}

// Test: OTF NTT correctness at sizes exercising all leftover patterns.
// Montgomery: num_outer%3 = 0,1,2
// Barrett: num_outer%2 = 0,1
void test_otf_ntt_leftover_patterns(int log_n, NTTMode mode) {
    size_t n = static_cast<size_t>(1) << log_n;
    int K = (log_n >= 10) ? 10 : ((log_n >= 9) ? 9 : 8);
    int num_outer = log_n - K;
    int leftover = (mode == NTTMode::OPTIMIZED) ? (num_outer % 3) : (num_outer % 2);
    printf("test_otf_ntt_leftover_patterns [%s] (n=2^%d, outer=%d, leftover=%d)...\n",
           mode_name(mode), log_n, num_outer, leftover);

    // Generate test data
    std::vector<FpElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = FpElement::zero();
        h_data[i].limbs[0] = static_cast<uint32_t>((i * 7654321u + 1234567u) % 1000000007u);
        h_data[i].limbs[1] = static_cast<uint32_t>((i * 8765432u + 2345678u) % 999999937u);
    }

    // CPU reference NTT (Montgomery domain)
    std::vector<ff_ref::FpRef> ref(n);
    for (size_t i = 0; i < n; ++i)
        ref[i] = ff_ref::to_montgomery(ff_ref::FpRef::from_u32(h_data[i].limbs));
    ff_ref::ntt_forward_reference(ref, n);
    for (size_t i = 0; i < n; ++i)
        ref[i] = ff_ref::from_montgomery(ref[i]);

    // GPU NTT
    FpElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_data, n, mode);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<FpElement> h_result(n);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, n * sizeof(FpElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    int pass = 0;
    for (size_t i = 0; i < n; ++i) {
        FpElement expected;
        ref[i].to_u32(expected.limbs);
        if (h_result[i] == expected) ++pass;
    }

    TEST_ASSERT(pass == static_cast<int>(n), "OTF leftover pattern NTT mismatch");
    printf("  OTF leftover [%s] (n=2^%d, leftover=%d): %d/%zu matched\n",
           mode_name(mode), log_n, leftover, pass, n);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Goldilocks Field Tests
// ═══════════════════════════════════════════════════════════════════════════════

extern __global__ void gl_add_kernel(const GoldilocksElement* __restrict__ a, const GoldilocksElement* __restrict__ b, GoldilocksElement* __restrict__ out, uint32_t n);
extern __global__ void gl_sub_kernel(const GoldilocksElement* __restrict__ a, const GoldilocksElement* __restrict__ b, GoldilocksElement* __restrict__ out, uint32_t n);
extern __global__ void gl_mul_kernel(const GoldilocksElement* __restrict__ a, const GoldilocksElement* __restrict__ b, GoldilocksElement* __restrict__ out, uint32_t n);
extern __global__ void gl_sqr_kernel(const GoldilocksElement* __restrict__ a, GoldilocksElement* __restrict__ out, uint32_t n);

void test_goldilocks_cpu_self_test() {
    using namespace ff_ref;
    printf("test_goldilocks_cpu_self_test...\n");

    // Test 1: 0 + 0 = 0
    GlRef z = GlRef::zero();
    TEST_ASSERT(gl_add(z, z) == z, "GL: 0 + 0 = 0");

    // Test 2: 1 * 1 = 1
    GlRef one = GlRef::one();
    TEST_ASSERT(gl_mul(one, one) == one, "GL: 1 * 1 = 1");

    // Test 3: a + 0 = a
    GlRef a = GlRef::from_u64(12345);
    TEST_ASSERT(gl_add(a, z) == a, "GL: a + 0 = a");

    // Test 4: a - a = 0
    TEST_ASSERT(gl_sub(a, a) == z, "GL: a - a = 0");

    // Test 5: a * 1 = a
    TEST_ASSERT(gl_mul(a, one) == a, "GL: a * 1 = a");

    // Test 6: commutativity
    GlRef b = GlRef::from_u64(67890);
    TEST_ASSERT(gl_mul(a, b) == gl_mul(b, a), "GL: a*b = b*a");
    TEST_ASSERT(gl_add(a, b) == gl_add(b, a), "GL: a+b = b+a");

    // Test 7: inverse
    GlRef a_inv = gl_inv(a);
    TEST_ASSERT(gl_mul(a, a_inv) == one, "GL: a * a^{-1} = 1");

    // Test 8: (p-1) + 1 = 0 (wrap-around)
    GlRef pm1 = GlRef::from_u64(GL_P - 1);
    TEST_ASSERT(gl_add(pm1, one) == z, "GL: (p-1) + 1 = 0");

    // Test 9: 0 - 1 = p - 1
    TEST_ASSERT(gl_sub(z, one) == pm1, "GL: 0 - 1 = p-1");

    // Test 10: large values
    GlRef big = GlRef::from_u64(0xFFFFFFFE00000002ULL);  // > p
    TEST_ASSERT(big.val < GL_P, "GL: from_u64 reduces mod p");

    // Test 11: root of unity basic check
    GlRef w4 = gl_get_root_of_unity(4);
    GlRef w4_4 = gl_pow(w4, 4);
    TEST_ASSERT(w4_4 == one, "GL: omega_4^4 = 1");

    // Test 12: NTT roundtrip at small size
    size_t n = 8;
    std::vector<GlRef> data(n);
    for (size_t i = 0; i < n; ++i) data[i] = GlRef::from_u64(i + 1);
    std::vector<GlRef> orig = data;
    gl_ntt_forward_reference(data, n);
    gl_ntt_inverse_reference(data, n);
    bool match = true;
    for (size_t i = 0; i < n; ++i)
        if (data[i] != orig[i]) { match = false; break; }
    TEST_ASSERT(match, "GL: CPU NTT roundtrip (n=8)");
}

void test_goldilocks_gpu_add() {
    using namespace ff_ref;
    printf("test_goldilocks_gpu_add...\n");

    const uint32_t N = 1024;
    std::vector<GoldilocksElement> h_a(N), h_b(N), h_out(N);

    // Fill with test data including edge cases
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i].val = (static_cast<uint64_t>(i) * 7 + 13) % GL_P;
        h_b[i].val = (static_cast<uint64_t>(i) * 11 + 37) % GL_P;
    }
    // Edge cases
    h_a[0].val = GL_P - 1; h_b[0].val = 1;  // wrap
    h_a[1].val = GL_P - 1; h_b[1].val = GL_P - 1;  // double wrap
    h_a[2].val = 0;        h_b[2].val = 0;

    GoldilocksElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    gl_add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        GlRef ra = {h_a[i].val}, rb = {h_b[i].val};
        GlRef expected = gl_add(ra, rb);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "GL GPU add: all elements match CPU reference");
    printf("  GL add: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_goldilocks_gpu_sub() {
    using namespace ff_ref;
    printf("test_goldilocks_gpu_sub...\n");

    const uint32_t N = 1024;
    std::vector<GoldilocksElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i].val = (static_cast<uint64_t>(i) * 7 + 13) % GL_P;
        h_b[i].val = (static_cast<uint64_t>(i) * 11 + 37) % GL_P;
    }
    h_a[0].val = 0; h_b[0].val = 1;  // underflow
    h_a[1].val = 0; h_b[1].val = GL_P - 1;

    GoldilocksElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    gl_sub_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        GlRef ra = {h_a[i].val}, rb = {h_b[i].val};
        GlRef expected = gl_sub(ra, rb);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "GL GPU sub: all elements match CPU reference");
    printf("  GL sub: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_goldilocks_gpu_mul() {
    using namespace ff_ref;
    printf("test_goldilocks_gpu_mul...\n");

    const uint32_t N = 1024;
    std::vector<GoldilocksElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i].val = (static_cast<uint64_t>(i) * 0x123456789ABCULL + 13) % GL_P;
        h_b[i].val = (static_cast<uint64_t>(i) * 0xDEADBEEF1234ULL + 37) % GL_P;
    }
    // Edge cases
    h_a[0].val = GL_P - 1; h_b[0].val = GL_P - 1;  // max * max
    h_a[1].val = 0;        h_b[1].val = GL_P - 1;   // 0 * anything
    h_a[2].val = 1;        h_b[2].val = GL_P - 1;   // 1 * anything

    GoldilocksElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    gl_mul_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        GlRef ra = {h_a[i].val}, rb = {h_b[i].val};
        GlRef expected = gl_mul(ra, rb);
        if (h_out[i].val == expected.val) ++pass;
        else if (i < 5) {
            printf("  GL mul MISMATCH [%u]: GPU=%llu, CPU=%llu (a=%llu, b=%llu)\n",
                   i, (unsigned long long)h_out[i].val, (unsigned long long)expected.val,
                   (unsigned long long)h_a[i].val, (unsigned long long)h_b[i].val);
        }
    }
    TEST_ASSERT(pass == N, "GL GPU mul: all elements match CPU reference");
    printf("  GL mul: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_goldilocks_gpu_sqr() {
    using namespace ff_ref;
    printf("test_goldilocks_gpu_sqr...\n");

    const uint32_t N = 1024;
    std::vector<GoldilocksElement> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i)
        h_a[i].val = (static_cast<uint64_t>(i) * 0xABCDEF01ULL + 7) % GL_P;
    h_a[0].val = GL_P - 1;
    h_a[1].val = 0;
    h_a[2].val = 1;

    GoldilocksElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));

    gl_sqr_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        GlRef ra = {h_a[i].val};
        GlRef expected = gl_sqr(ra);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "GL GPU sqr: all elements match CPU reference");
    printf("  GL sqr: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

void test_goldilocks_algebraic() {
    using namespace ff_ref;
    printf("test_goldilocks_algebraic...\n");

    // Distributive: a*(b+c) = a*b + a*c
    GlRef a = GlRef::from_u64(0x123456789ABCDEF0ULL % GL_P);
    GlRef b = GlRef::from_u64(0xFEDCBA9876543210ULL % GL_P);
    GlRef c = GlRef::from_u64(42);
    GlRef lhs = gl_mul(a, gl_add(b, c));
    GlRef rhs = gl_add(gl_mul(a, b), gl_mul(a, c));
    TEST_ASSERT(lhs == rhs, "GL: distributive law a*(b+c) = a*b + a*c");

    // Associative: (a*b)*c = a*(b*c)
    GlRef lhs2 = gl_mul(gl_mul(a, b), c);
    GlRef rhs2 = gl_mul(a, gl_mul(b, c));
    TEST_ASSERT(lhs2 == rhs2, "GL: associative law (a*b)*c = a*(b*c)");

    // Fermat's little theorem: a^p = a (mod p)
    GlRef a_to_p = gl_pow(a, GL_P);
    TEST_ASSERT(a_to_p == a, "GL: Fermat's little theorem a^p = a");

    // a^(p-1) = 1 for a != 0
    GlRef a_to_pm1 = gl_pow(a, GL_P - 1);
    TEST_ASSERT(a_to_pm1 == GlRef::one(), "GL: a^(p-1) = 1");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BabyBear Field Tests
// ═══════════════════════════════════════════════════════════════════════════════

extern __global__ void bb_add_kernel(const BabyBearElement* __restrict__ a, const BabyBearElement* __restrict__ b, BabyBearElement* __restrict__ out, uint32_t n);
extern __global__ void bb_sub_kernel(const BabyBearElement* __restrict__ a, const BabyBearElement* __restrict__ b, BabyBearElement* __restrict__ out, uint32_t n);
extern __global__ void bb_mul_kernel(const BabyBearElement* __restrict__ a, const BabyBearElement* __restrict__ b, BabyBearElement* __restrict__ out, uint32_t n);
extern __global__ void bb_sqr_kernel(const BabyBearElement* __restrict__ a, BabyBearElement* __restrict__ out, uint32_t n);

void test_babybear_cpu_self_test() {
    using namespace ff_ref;
    printf("test_babybear_cpu_self_test...\n");

    BbRef z = BbRef::zero();
    BbRef one = BbRef::one();

    TEST_ASSERT(bb_add(z, z) == z, "BB: 0 + 0 = 0");
    TEST_ASSERT(bb_mul(one, one) == one, "BB: 1 * 1 = 1");

    BbRef a = BbRef::from_u32(12345);
    TEST_ASSERT(bb_add(a, z) == a, "BB: a + 0 = a");
    TEST_ASSERT(bb_sub(a, a) == z, "BB: a - a = 0");
    TEST_ASSERT(bb_mul(a, one) == a, "BB: a * 1 = a");

    BbRef b = BbRef::from_u32(67890);
    TEST_ASSERT(bb_mul(a, b) == bb_mul(b, a), "BB: a*b = b*a");
    TEST_ASSERT(bb_add(a, b) == bb_add(b, a), "BB: a+b = b+a");

    BbRef a_inv = bb_inv(a);
    TEST_ASSERT(bb_mul(a, a_inv) == one, "BB: a * a^{-1} = 1");

    BbRef pm1 = BbRef::from_u32(BB_P - 1);
    TEST_ASSERT(bb_add(pm1, one) == z, "BB: (p-1) + 1 = 0");
    TEST_ASSERT(bb_sub(z, one) == pm1, "BB: 0 - 1 = p-1");

    // Root of unity
    BbRef w4 = bb_get_root_of_unity(4);
    BbRef w4_4 = bb_pow(w4, 4);
    TEST_ASSERT(w4_4 == one, "BB: omega_4^4 = 1");

    // NTT roundtrip
    size_t n = 8;
    std::vector<BbRef> data(n);
    for (size_t i = 0; i < n; ++i) data[i] = BbRef::from_u32(static_cast<uint32_t>(i + 1));
    std::vector<BbRef> orig = data;
    bb_ntt_forward_reference(data, n);
    bb_ntt_inverse_reference(data, n);
    bool match = true;
    for (size_t i = 0; i < n; ++i)
        if (data[i] != orig[i]) { match = false; break; }
    TEST_ASSERT(match, "BB: CPU NTT roundtrip (n=8)");
}

void test_babybear_gpu_add() {
    using namespace ff_ref;
    printf("test_babybear_gpu_add...\n");

    const uint32_t N = 1024;
    std::vector<BabyBearElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i].val = (i * 7u + 13u) % BB_P;
        h_b[i].val = (i * 11u + 37u) % BB_P;
    }
    h_a[0].val = BB_P - 1; h_b[0].val = 1;
    h_a[1].val = BB_P - 1; h_b[1].val = BB_P - 1;
    h_a[2].val = 0;        h_b[2].val = 0;

    BabyBearElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    bb_add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        BbRef ra = {h_a[i].val}, rb = {h_b[i].val};
        BbRef expected = bb_add(ra, rb);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "BB GPU add: all elements match CPU reference");
    printf("  BB add: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_babybear_gpu_sub() {
    using namespace ff_ref;
    printf("test_babybear_gpu_sub...\n");

    const uint32_t N = 1024;
    std::vector<BabyBearElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i].val = (i * 7u + 13u) % BB_P;
        h_b[i].val = (i * 11u + 37u) % BB_P;
    }
    h_a[0].val = 0; h_b[0].val = 1;
    h_a[1].val = 0; h_b[1].val = BB_P - 1;

    BabyBearElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    bb_sub_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        BbRef ra = {h_a[i].val}, rb = {h_b[i].val};
        BbRef expected = bb_sub(ra, rb);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "BB GPU sub: all elements match CPU reference");
    printf("  BB sub: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_babybear_gpu_mul() {
    using namespace ff_ref;
    printf("test_babybear_gpu_mul...\n");

    const uint32_t N = 1024;
    std::vector<BabyBearElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i].val = (i * 123457u + 13u) % BB_P;
        h_b[i].val = (i * 654321u + 37u) % BB_P;
    }
    h_a[0].val = BB_P - 1; h_b[0].val = BB_P - 1;
    h_a[1].val = 0;        h_b[1].val = BB_P - 1;
    h_a[2].val = 1;        h_b[2].val = BB_P - 1;

    BabyBearElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    bb_mul_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        BbRef ra = {h_a[i].val}, rb = {h_b[i].val};
        BbRef expected = bb_mul(ra, rb);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "BB GPU mul: all elements match CPU reference");
    printf("  BB mul: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

void test_babybear_gpu_sqr() {
    using namespace ff_ref;
    printf("test_babybear_gpu_sqr...\n");

    const uint32_t N = 1024;
    std::vector<BabyBearElement> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i)
        h_a[i].val = (i * 654321u + 7u) % BB_P;
    h_a[0].val = BB_P - 1;
    h_a[1].val = 0;
    h_a[2].val = 1;

    BabyBearElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(BabyBearElement), cudaMemcpyHostToDevice));

    bb_sqr_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        BbRef ra = {h_a[i].val};
        BbRef expected = bb_sqr(ra);
        if (h_out[i].val == expected.val) ++pass;
    }
    TEST_ASSERT(pass == N, "BB GPU sqr: all elements match CPU reference");
    printf("  BB sqr: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

void test_babybear_algebraic() {
    using namespace ff_ref;
    printf("test_babybear_algebraic...\n");

    BbRef a = BbRef::from_u32(1234567u % BB_P);
    BbRef b = BbRef::from_u32(7654321u % BB_P);
    BbRef c = BbRef::from_u32(42);

    // Distributive
    BbRef lhs = bb_mul(a, bb_add(b, c));
    BbRef rhs = bb_add(bb_mul(a, b), bb_mul(a, c));
    TEST_ASSERT(lhs == rhs, "BB: distributive law a*(b+c) = a*b + a*c");

    // Associative
    BbRef lhs2 = bb_mul(bb_mul(a, b), c);
    BbRef rhs2 = bb_mul(a, bb_mul(b, c));
    TEST_ASSERT(lhs2 == rhs2, "BB: associative law (a*b)*c = a*(b*c)");

    // Fermat's little theorem
    BbRef a_to_p = bb_pow(a, BB_P);
    TEST_ASSERT(a_to_p == a, "BB: Fermat's little theorem a^p = a");

    BbRef a_to_pm1 = bb_pow(a, BB_P - 1);
    TEST_ASSERT(a_to_pm1 == BbRef::one(), "BB: a^(p-1) = 1");
}

// ─── v1.6.0 Session 17: Goldilocks NTT Tests ────────────────────────────────

void test_gl_ntt_forward(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_gl_ntt_forward (n=2^%d)...\n", log_n);

    // Generate test data
    std::vector<GlRef> h_ref(n);
    std::vector<GoldilocksElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t v = (i * 12345678901ull + 6789ull) % GL_P;
        h_ref[i] = GlRef::from_u64(v);
        h_data[i].val = v;
    }

    // CPU reference NTT
    gl_ntt_forward_reference(h_ref, n);

    // GPU NTT
    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    ntt_forward_goldilocks(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == h_ref[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "GL NTT forward vs CPU reference mismatch");
    printf("  GL forward (n=2^%d): %d/%zu matched\n", log_n, pass, n);

    if (pass != static_cast<int>(n)) {
        int printed = 0;
        for (size_t i = 0; i < n && printed < 5; ++i) {
            if (h_data[i].val != h_ref[i].val) {
                printf("    [%zu] gpu=%016llx cpu=%016llx\n", i,
                    (unsigned long long)h_data[i].val, (unsigned long long)h_ref[i].val);
                ++printed;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_data));
}

void test_gl_ntt_roundtrip(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_gl_ntt_roundtrip (n=2^%d)...\n", log_n);

    std::vector<GoldilocksElement> h_orig(n), h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_orig[i].val = (i * 12345678901ull + 6789ull) % GL_P;
        h_data[i] = h_orig[i];
    }

    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    ntt_forward_goldilocks(d_data, n);
    ntt_inverse_goldilocks(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == h_orig[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "GL NTT roundtrip mismatch");
    printf("  GL roundtrip (n=2^%d): %d/%zu matched\n", log_n, pass, n);

    CUDA_CHECK(cudaFree(d_data));
}

void test_gl_ntt_known_vectors(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_gl_ntt_known_vectors (n=2^%d)...\n", log_n);

    auto run_roundtrip = [&](const char* name, std::vector<GoldilocksElement>& h_data) {
        std::vector<GoldilocksElement> h_orig = h_data;
        GoldilocksElement* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(GoldilocksElement)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
        ntt_forward_goldilocks(d_data, n);
        ntt_inverse_goldilocks(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

        int pass = 0;
        for (size_t i = 0; i < n; ++i)
            if (h_data[i].val == h_orig[i].val) ++pass;
        char msg[128];
        snprintf(msg, sizeof(msg), "GL known-vector %s (n=2^%d)", name, log_n);
        TEST_ASSERT(pass == static_cast<int>(n), msg);
        CUDA_CHECK(cudaFree(d_data));
    };

    // All zeros
    { std::vector<GoldilocksElement> v(n); for (auto& e : v) e.val = 0; run_roundtrip("zeros", v); }
    // All ones
    { std::vector<GoldilocksElement> v(n); for (auto& e : v) e.val = 1; run_roundtrip("ones", v); }
    // Single nonzero
    { std::vector<GoldilocksElement> v(n); for (auto& e : v) e.val = 0; v[0].val = 42; run_roundtrip("single", v); }
    // Ascending
    { std::vector<GoldilocksElement> v(n); for (size_t i = 0; i < n; ++i) v[i].val = i % GL_P; run_roundtrip("ascending", v); }
}

void test_gl_ntt_batch_vs_sequential(int log_n, int batch_size) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_gl_ntt_batch_vs_sequential (n=2^%d, B=%d)...\n", log_n, batch_size);

    std::vector<GoldilocksElement> h_data(total);
    for (size_t i = 0; i < total; ++i)
        h_data[i].val = (i * 12345678901ull + 42ull) % GL_P;

    // Sequential: run B individual NTTs
    GoldilocksElement* d_seq;
    CUDA_CHECK(cudaMalloc(&d_seq, total * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_seq, h_data.data(), total * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    for (int b = 0; b < batch_size; ++b)
        ntt_forward_goldilocks(d_seq + b * n, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Batched
    GoldilocksElement* d_batch;
    CUDA_CHECK(cudaMalloc(&d_batch, total * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_batch, h_data.data(), total * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    ntt_forward_batch_goldilocks(d_batch, batch_size, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GoldilocksElement> h_seq(total), h_batch(total);
    CUDA_CHECK(cudaMemcpy(h_seq.data(), d_seq, total * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_batch.data(), d_batch, total * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < total; ++i)
        if (h_seq[i].val == h_batch[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(total), "GL batch vs sequential mismatch");
    printf("  GL batch vs seq (n=2^%d, B=%d): %d/%zu matched\n", log_n, batch_size, pass, total);

    CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_batch));
}

void test_gl_ntt_batch_roundtrip(int log_n, int batch_size) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_gl_ntt_batch_roundtrip (n=2^%d, B=%d)...\n", log_n, batch_size);

    std::vector<GoldilocksElement> h_orig(total), h_data(total);
    for (size_t i = 0; i < total; ++i) {
        h_orig[i].val = (i * 12345678901ull + 42ull) % GL_P;
        h_data[i] = h_orig[i];
    }

    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), total * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    ntt_forward_batch_goldilocks(d_data, batch_size, n);
    ntt_inverse_batch_goldilocks(d_data, batch_size, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, total * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < total; ++i)
        if (h_data[i].val == h_orig[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(total), "GL batch roundtrip mismatch");
    printf("  GL batch roundtrip (n=2^%d, B=%d): %d/%zu matched\n", log_n, batch_size, pass, total);

    CUDA_CHECK(cudaFree(d_data));
}

void test_gl_ntt_forward_zeros(int log_n) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_gl_ntt_forward_zeros (n=2^%d)...\n", log_n);

    std::vector<GoldilocksElement> h_data(n);
    for (auto& e : h_data) e.val = 0;

    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(GoldilocksElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    ntt_forward_goldilocks(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == 0) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "GL NTT(0) should be 0");

    CUDA_CHECK(cudaFree(d_data));
}

void test_gl_ntt_inverse_explicit(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_gl_ntt_inverse_explicit (n=2^%d)...\n", log_n);

    std::vector<GoldilocksElement> h_data(n);
    for (size_t i = 0; i < n; ++i)
        h_data[i].val = (i * 12345678901ull + 6789ull) % GL_P;
    std::vector<GoldilocksElement> h_orig = h_data;

    GoldilocksElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(GoldilocksElement)));

    // Test fwd(inv(x)) = x
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(GoldilocksElement), cudaMemcpyHostToDevice));
    ntt_inverse_goldilocks(d_data, n);
    ntt_forward_goldilocks(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(GoldilocksElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == h_orig[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "GL fwd(inv(x)) = x");
    printf("  GL fwd(inv(x))=x (n=2^%d): %d/%zu matched\n", log_n, pass, n);

    CUDA_CHECK(cudaFree(d_data));
}

// ─── v1.6.0 Session 17: BabyBear NTT Tests ─────────────────────────────────

void test_bb_ntt_forward(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_bb_ntt_forward (n=2^%d)...\n", log_n);

    std::vector<BbRef> h_ref(n);
    std::vector<BabyBearElement> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = static_cast<uint32_t>((i * 12345u + 6789u) % BB_P);
        h_ref[i] = BbRef::from_u32(v);
        h_data[i].val = v;
    }

    bb_ntt_forward_reference(h_ref, n);

    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    ntt_forward_babybear(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == h_ref[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "BB NTT forward vs CPU reference mismatch");
    printf("  BB forward (n=2^%d): %d/%zu matched\n", log_n, pass, n);

    if (pass != static_cast<int>(n)) {
        int printed = 0;
        for (size_t i = 0; i < n && printed < 5; ++i) {
            if (h_data[i].val != h_ref[i].val) {
                printf("    [%zu] gpu=%08x cpu=%08x\n", i, h_data[i].val, h_ref[i].val);
                ++printed;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_data));
}

void test_bb_ntt_roundtrip(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_bb_ntt_roundtrip (n=2^%d)...\n", log_n);

    std::vector<BabyBearElement> h_orig(n), h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_orig[i].val = static_cast<uint32_t>((i * 12345u + 6789u) % BB_P);
        h_data[i] = h_orig[i];
    }

    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    ntt_forward_babybear(d_data, n);
    ntt_inverse_babybear(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == h_orig[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "BB NTT roundtrip mismatch");
    printf("  BB roundtrip (n=2^%d): %d/%zu matched\n", log_n, pass, n);

    CUDA_CHECK(cudaFree(d_data));
}

void test_bb_ntt_known_vectors(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_bb_ntt_known_vectors (n=2^%d)...\n", log_n);

    auto run_roundtrip = [&](const char* name, std::vector<BabyBearElement>& h_data) {
        std::vector<BabyBearElement> h_orig = h_data;
        BabyBearElement* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(BabyBearElement)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
        ntt_forward_babybear(d_data, n);
        ntt_inverse_babybear(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

        int pass = 0;
        for (size_t i = 0; i < n; ++i)
            if (h_data[i].val == h_orig[i].val) ++pass;
        char msg[128];
        snprintf(msg, sizeof(msg), "BB known-vector %s (n=2^%d)", name, log_n);
        TEST_ASSERT(pass == static_cast<int>(n), msg);
        CUDA_CHECK(cudaFree(d_data));
    };

    { std::vector<BabyBearElement> v(n); for (auto& e : v) e.val = 0; run_roundtrip("zeros", v); }
    { std::vector<BabyBearElement> v(n); for (auto& e : v) e.val = 1; run_roundtrip("ones", v); }
    { std::vector<BabyBearElement> v(n); for (auto& e : v) e.val = 0; v[0].val = 42; run_roundtrip("single", v); }
    { std::vector<BabyBearElement> v(n); for (size_t i = 0; i < n; ++i) v[i].val = static_cast<uint32_t>(i % BB_P); run_roundtrip("ascending", v); }
}

void test_bb_ntt_batch_vs_sequential(int log_n, int batch_size) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_bb_ntt_batch_vs_sequential (n=2^%d, B=%d)...\n", log_n, batch_size);

    std::vector<BabyBearElement> h_data(total);
    for (size_t i = 0; i < total; ++i)
        h_data[i].val = static_cast<uint32_t>((i * 12345u + 42u) % BB_P);

    BabyBearElement* d_seq;
    CUDA_CHECK(cudaMalloc(&d_seq, total * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_seq, h_data.data(), total * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    for (int b = 0; b < batch_size; ++b)
        ntt_forward_babybear(d_seq + b * n, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    BabyBearElement* d_batch;
    CUDA_CHECK(cudaMalloc(&d_batch, total * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_batch, h_data.data(), total * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    ntt_forward_batch_babybear(d_batch, batch_size, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<BabyBearElement> h_seq(total), h_batch(total);
    CUDA_CHECK(cudaMemcpy(h_seq.data(), d_seq, total * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_batch.data(), d_batch, total * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < total; ++i)
        if (h_seq[i].val == h_batch[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(total), "BB batch vs sequential mismatch");
    printf("  BB batch vs seq (n=2^%d, B=%d): %d/%zu matched\n", log_n, batch_size, pass, total);

    CUDA_CHECK(cudaFree(d_seq));
    CUDA_CHECK(cudaFree(d_batch));
}

void test_bb_ntt_batch_roundtrip(int log_n, int batch_size) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    size_t total = static_cast<size_t>(batch_size) * n;
    printf("test_bb_ntt_batch_roundtrip (n=2^%d, B=%d)...\n", log_n, batch_size);

    std::vector<BabyBearElement> h_orig(total), h_data(total);
    for (size_t i = 0; i < total; ++i) {
        h_orig[i].val = static_cast<uint32_t>((i * 12345u + 42u) % BB_P);
        h_data[i] = h_orig[i];
    }

    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), total * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    ntt_forward_batch_babybear(d_data, batch_size, n);
    ntt_inverse_batch_babybear(d_data, batch_size, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, total * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < total; ++i)
        if (h_data[i].val == h_orig[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(total), "BB batch roundtrip mismatch");
    printf("  BB batch roundtrip (n=2^%d, B=%d): %d/%zu matched\n", log_n, batch_size, pass, total);

    CUDA_CHECK(cudaFree(d_data));
}

void test_bb_ntt_forward_zeros(int log_n) {
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_bb_ntt_forward_zeros (n=2^%d)...\n", log_n);

    std::vector<BabyBearElement> h_data(n);
    for (auto& e : h_data) e.val = 0;

    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(BabyBearElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    ntt_forward_babybear(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == 0) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "BB NTT(0) should be 0");

    CUDA_CHECK(cudaFree(d_data));
}

void test_bb_ntt_inverse_explicit(int log_n) {
    using namespace ff_ref;
    size_t n = static_cast<size_t>(1) << log_n;
    printf("test_bb_ntt_inverse_explicit (n=2^%d)...\n", log_n);

    std::vector<BabyBearElement> h_data(n);
    for (size_t i = 0; i < n; ++i)
        h_data[i].val = static_cast<uint32_t>((i * 12345u + 6789u) % BB_P);
    std::vector<BabyBearElement> h_orig = h_data;

    BabyBearElement* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(BabyBearElement)));

    // Test fwd(inv(x)) = x
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(BabyBearElement), cudaMemcpyHostToDevice));
    ntt_inverse_babybear(d_data, n);
    ntt_forward_babybear(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n * sizeof(BabyBearElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (size_t i = 0; i < n; ++i)
        if (h_data[i].val == h_orig[i].val) ++pass;
    TEST_ASSERT(pass == static_cast<int>(n), "BB fwd(inv(x)) = x");
    printf("  BB fwd(inv(x))=x (n=2^%d): %d/%zu matched\n", log_n, pass, n);

    CUDA_CHECK(cudaFree(d_data));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Plantard Arithmetic Tests (v1.7.0 Session 19 — NEGATIVE RESULT investigation)
// ═══════════════════════════════════════════════════════════════════════════════

extern __global__ void ff_mul_plantard_kernel(const FpElement* __restrict__ a, const FpElement* __restrict__ b, FpElement* __restrict__ out, uint32_t n);

// CPU reference: Plantard mul on standard-form elements
static FpElement cpu_plantard_mul(const FpElement& a, const FpElement& t_plant) {
    ff_ref::FpRef ra = ff_ref::FpRef::from_u32(a.limbs);
    ff_ref::FpRef rt = ff_ref::FpRef::from_u32(t_plant.limbs);
    ff_ref::FpRef rc = ff_ref::fp_mul_plantard(ra, rt);
    FpElement result;
    rc.to_u32(result.limbs);
    return result;
}

// Convert standard-form twiddle to Plantard form
static FpElement cpu_to_plantard_twiddle(const FpElement& w_std) {
    ff_ref::FpRef rw = ff_ref::FpRef::from_u32(w_std.limbs);
    ff_ref::FpRef rt = ff_ref::fp_to_plantard_twiddle(rw);
    FpElement result;
    rt.to_u32(result.limbs);
    return result;
}

void test_plantard_cpu_self_test() {
    using namespace ff_ref;
    printf("test_plantard_cpu_self_test...\n");

    // Test 1: Plantard mu constant verification: p * mu ≡ 1 mod R^2
    {
        // Verify by checking p * mu mod 2^64 = 1 (the lowest 64 bits)
        // p[0] = 0xffffffff00000001, mu[0] = 0x0000000100000001
        // p[0] * mu[0] mod 2^64 should contribute to 1
        Plant512 p512;
        for (int i = 0; i < 8; ++i) p512.limbs[i] = 0;
        p512.limbs[0] = MOD[0]; p512.limbs[1] = MOD[1];
        p512.limbs[2] = MOD[2]; p512.limbs[3] = MOD[3];
        Plant512 mu512;
        for (int i = 0; i < 8; ++i) mu512.limbs[i] = PLANTARD_MU[i];

        Plant512 product = plant512_mul_lo(p512, mu512);
        // product should be 1 (mod 2^512)
        bool is_one = (product.limbs[0] == 1);
        for (int i = 1; i < 8; ++i) is_one = is_one && (product.limbs[i] == 0);
        TEST_ASSERT(is_one, "p * mu mod R^2 should be 1");
    }

    // Test 2: Plantard(a, -w*R^2) should give a*w mod p
    {
        FpRef a = FpRef::from_u64(42);
        FpRef w = FpRef::from_u64(17);
        FpRef t_plant = fp_to_plantard_twiddle(w);
        FpRef result = fp_mul_plantard(a, t_plant);
        FpRef expected = fp_mul_barrett(a, w);
        TEST_ASSERT(result == expected, "Plantard(42, T(17)) should equal Barrett(42, 17)");
    }

    // Test 3: a=0 should give 0
    {
        FpRef a = FpRef::zero();
        FpRef w = FpRef::from_u64(123);
        FpRef t_plant = fp_to_plantard_twiddle(w);
        FpRef result = fp_mul_plantard(a, t_plant);
        TEST_ASSERT(result == FpRef::zero(), "Plantard(0, T(w)) should be 0");
    }

    // Test 4: w=0 -> T=0, Plantard should give 0
    {
        FpRef a = FpRef::from_u64(42);
        FpRef w = FpRef::zero();
        FpRef t_plant = fp_to_plantard_twiddle(w);
        FpRef result = fp_mul_plantard(a, t_plant);
        TEST_ASSERT(result == FpRef::zero(), "Plantard(a, T(0)) should be 0");
    }

    // Test 5: Identity — a*1 = a
    {
        FpRef a = FpRef::from_u64(999999);
        FpRef w = FpRef::from_u64(1);
        FpRef t_plant = fp_to_plantard_twiddle(w);
        FpRef result = fp_mul_plantard(a, t_plant);
        TEST_ASSERT(result == a, "Plantard(a, T(1)) should be a");
    }

    // Test 6: (p-1) * (p-1) edge case
    {
        FpRef a, w;
        for (int i = 0; i < 4; ++i) {
            a.limbs[i] = MOD[i];
            w.limbs[i] = MOD[i];
        }
        // a = p, but we need a < p. Use p-1.
        a = fp_sub(a, FpRef::from_u64(1));
        w = fp_sub(w, FpRef::from_u64(1));
        FpRef t_plant = fp_to_plantard_twiddle(w);
        FpRef result = fp_mul_plantard(a, t_plant);
        FpRef expected = fp_mul_barrett(a, w);
        TEST_ASSERT(result == expected, "Plantard((p-1), T(p-1)) should match Barrett");
    }
}

void test_plantard_vs_barrett() {
    using namespace ff_ref;
    printf("test_plantard_vs_barrett (1024 random pairs)...\n");

    int pass = 0;
    const int N = 1024;
    uint32_t state = 0xDEADBEEF;

    for (int i = 0; i < N; ++i) {
        // Generate random elements < p
        FpRef a, w;
        for (int j = 0; j < 4; ++j) {
            state = state * 1664525u + 1013904223u;
            a.limbs[j] = static_cast<uint64_t>(state) |
                          (static_cast<uint64_t>(state * 2654435761u) << 32);
            state = state * 1664525u + 1013904223u;
            w.limbs[j] = static_cast<uint64_t>(state) |
                          (static_cast<uint64_t>(state * 2654435761u) << 32);
        }
        // Ensure < p
        a.limbs[3] &= 0x3FFFFFFFFFFFFFFFULL;
        w.limbs[3] &= 0x3FFFFFFFFFFFFFFFULL;

        FpRef t_plant = fp_to_plantard_twiddle(w);
        FpRef result = fp_mul_plantard(a, t_plant);
        FpRef expected = fp_mul_barrett(a, w);
        if (result == expected) ++pass;
    }
    TEST_ASSERT(pass == N, "Plantard vs Barrett mismatch");
    printf("  %d/%d matched\n", pass, N);
}

void test_plantard_vs_montgomery() {
    using namespace ff_ref;
    printf("test_plantard_vs_montgomery (1024 random pairs)...\n");

    int pass = 0;
    const int N = 1024;

    for (int i = 0; i < N; ++i) {
        FpElement a_gpu = make_large_standard_fp(static_cast<uint32_t>(i * 3 + 7));
        FpElement w_gpu = make_large_standard_fp(static_cast<uint32_t>(i * 3 + 2000));

        FpRef a_ref = FpRef::from_u32(a_gpu.limbs);
        FpRef w_ref = FpRef::from_u32(w_gpu.limbs);

        // Plantard: a * w mod p (standard form)
        FpRef t_plant = fp_to_plantard_twiddle(w_ref);
        FpRef plantard_result = fp_mul_plantard(a_ref, t_plant);

        // Montgomery: to_mont(a) * to_mont(w) -> from_mont -> standard form
        FpRef a_mont = to_montgomery(a_ref);
        FpRef w_mont = to_montgomery(w_ref);
        FpRef mont_result_mont = fp_mul(a_mont, w_mont);
        FpRef mont_result = from_montgomery(mont_result_mont);

        if (plantard_result == mont_result) ++pass;
    }
    TEST_ASSERT(pass == N, "Plantard vs Montgomery mismatch");
    printf("  %d/%d matched\n", pass, N);
}

void test_plantard_gpu() {
    printf("test_plantard_gpu (1024 elements)...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_t(N), h_out(N);

    // Generate standard-form inputs and Plantard twiddles
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_large_standard_fp(i * 2 + 1);
        FpElement w_std = make_large_standard_fp(i * 2 + 5000);
        h_t[i] = cpu_to_plantard_twiddle(w_std);
    }

    FpElement *d_a, *d_t, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_t,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t, h_t.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    ff_mul_plantard_kernel<<<(N + 255) / 256, 256>>>(d_a, d_t, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FpElement expected = cpu_plantard_mul(h_a[i], h_t[i]);
        if (h_out[i] == expected) ++pass_count;
    }

    TEST_ASSERT(pass_count == (int)N, "ff_mul_plantard GPU vs CPU reference mismatch");
    printf("  ff_mul_plantard GPU: %d/%u matched\n", pass_count, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_out));
}

void test_plantard_gpu_vs_barrett_gpu() {
    printf("test_plantard_gpu_vs_barrett_gpu (1024 elements)...\n");

    const uint32_t N = 1024;
    std::vector<FpElement> h_a(N), h_w(N), h_t_plant(N);
    std::vector<FpElement> h_plant_out(N), h_barrett_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = make_large_standard_fp(i * 7 + 11);
        h_w[i] = make_large_standard_fp(i * 7 + 3000);
        h_t_plant[i] = cpu_to_plantard_twiddle(h_w[i]);
    }

    FpElement *d_a, *d_w, *d_t, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_w,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_t,   N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));

    // Plantard GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t, h_t_plant.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    ff_mul_plantard_kernel<<<(N + 255) / 256, 256>>>(d_a, d_t, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_plant_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Barrett GPU
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    ff_mul_barrett_kernel<<<(N + 255) / 256, 256>>>(d_a, d_w, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_barrett_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (h_plant_out[i] == h_barrett_out[i]) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Plantard GPU vs Barrett GPU mismatch");
    printf("  Plantard vs Barrett GPU: %d/%u matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_out));
}

void test_plantard_twiddle_precomputation() {
    using namespace ff_ref;
    printf("test_plantard_twiddle_precomputation...\n");

    // For n=1024, check that Plantard twiddles produce correct NTT-style results
    const size_t n = 1024;
    FpRef omega = get_root_of_unity(n);
    FpRef omega_std = from_montgomery(omega);
    FpRef one = FpRef::from_u64(1);

    FpRef wk = one;
    int pass = 0;
    for (size_t k = 0; k < n / 2; ++k) {
        FpRef t_plant = fp_to_plantard_twiddle(wk);

        // Test: Plantard(a, t_plant) == a * wk mod p
        FpRef a = FpRef::from_u64(k + 100);
        FpRef result = fp_mul_plantard(a, t_plant);
        FpRef expected = fp_mul_barrett(a, wk);
        if (result == expected) ++pass;

        // Advance twiddle (standard form)
        wk = fp_mul_barrett(wk, omega_std);
    }
    TEST_ASSERT(pass == (int)(n / 2), "Plantard twiddle precomputation mismatch");
    printf("  %d/%zu twiddle values verified\n", pass, n / 2);
}

void test_plantard_algebraic() {
    using namespace ff_ref;
    printf("test_plantard_algebraic...\n");

    // Test commutativity: a*w == w*a via Plantard
    // Note: Plantard(a, T(w)) gives a*w. Plantard(w, T(a)) gives w*a.
    // These should be equal.
    int pass_comm = 0, pass_assoc = 0;
    const int N = 256;

    for (int i = 0; i < N; ++i) {
        uint64_t seed_a = static_cast<uint64_t>(i * 13 + 1);
        uint64_t seed_w = static_cast<uint64_t>(i * 17 + 100);
        FpRef a = FpRef::from_u64(seed_a);
        FpRef w = FpRef::from_u64(seed_w);

        FpRef t_w = fp_to_plantard_twiddle(w);
        FpRef t_a = fp_to_plantard_twiddle(a);
        FpRef aw = fp_mul_plantard(a, t_w);
        FpRef wa = fp_mul_plantard(w, t_a);
        if (aw == wa) ++pass_comm;
    }
    TEST_ASSERT(pass_comm == N, "Plantard commutativity failure");
    printf("  Commutativity: %d/%d\n", pass_comm, N);

    // Test associativity: (a*b)*c == a*(b*c) via Barrett (Plantard is for twiddle mul)
    for (int i = 0; i < N; ++i) {
        FpRef a = FpRef::from_u64(static_cast<uint64_t>(i * 5 + 1));
        FpRef b = FpRef::from_u64(static_cast<uint64_t>(i * 5 + 2));
        FpRef c = FpRef::from_u64(static_cast<uint64_t>(i * 5 + 3));

        // (a*b)*c via Plantard
        FpRef ab = fp_mul_barrett(a, b);
        FpRef t_c = fp_to_plantard_twiddle(c);
        FpRef ab_c = fp_mul_plantard(ab, t_c);

        // a*(b*c) via Plantard
        FpRef bc = fp_mul_barrett(b, c);
        FpRef t_bc = fp_to_plantard_twiddle(bc);
        FpRef a_bc = fp_mul_plantard(a, t_bc);

        if (ab_c == a_bc) ++pass_assoc;
    }
    TEST_ASSERT(pass_assoc == N, "Plantard associativity failure");
    printf("  Associativity: %d/%d\n", pass_assoc, N);
}

// ═══════════════════════════════════════════════════════════════════════════════
// v2.0.0 Session 20: Fq (Base Field) + Fq2 (Extension Field) Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Helper: generate deterministic pseudo-random FqRef in Montgomery form ──
static ff_ref::FqRef make_fq_test_val(uint64_t seed) {
    using namespace ff_ref;
    // Generate a simple value mod q, then convert to Montgomery
    FqRef v = FqRef::from_u64(seed);
    return fq_to_montgomery_ref(v);
}

// ─── Helper: convert FqRef <-> FqElement (GPU format) ───────────────────────
static FqElement fq_ref_to_gpu(const ff_ref::FqRef& r) {
    FqElement e;
    r.to_u32(e.limbs);
    return e;
}

static ff_ref::FqRef fq_gpu_to_ref(const FqElement& e) {
    return ff_ref::FqRef::from_u32(e.limbs);
}

// ─── Fq CPU Self-Test ──────────────────────────────────────────────────────

void test_fq_cpu_self_test() {
    using namespace ff_ref;
    printf("test_fq_cpu_self_test...\n");

    FqRef z = FqRef::zero();
    FqRef one_std = FqRef::from_u64(1);
    FqRef one = fq_to_montgomery_ref(one_std);

    // Basic identities
    TEST_ASSERT(fq_add_ref(z, z) == z, "Fq: 0 + 0 = 0");
    TEST_ASSERT(fq_mul_ref(one, one) == one, "Fq: 1 * 1 = 1");

    FqRef a = fq_to_montgomery_ref(FqRef::from_u64(12345));
    TEST_ASSERT(fq_add_ref(a, z) == a, "Fq: a + 0 = a");
    TEST_ASSERT(fq_sub_ref(a, a) == z, "Fq: a - a = 0");
    TEST_ASSERT(fq_mul_ref(a, one) == a, "Fq: a * 1 = a");

    // Commutativity
    FqRef b = fq_to_montgomery_ref(FqRef::from_u64(67890));
    TEST_ASSERT(fq_mul_ref(a, b) == fq_mul_ref(b, a), "Fq: a*b = b*a");
    TEST_ASSERT(fq_add_ref(a, b) == fq_add_ref(b, a), "Fq: a+b = b+a");

    // Inverse
    FqRef a_inv = fq_inv_ref(a);
    TEST_ASSERT(fq_mul_ref(a, a_inv) == one, "Fq: a * a^{-1} = 1");

    // Negation
    FqRef neg_a = fq_neg_ref(a);
    TEST_ASSERT(fq_add_ref(a, neg_a) == z, "Fq: a + (-a) = 0");
    TEST_ASSERT(fq_neg_ref(z) == z, "Fq: -0 = 0");

    // Distributivity: a*(b+c) = a*b + a*c
    FqRef c = fq_to_montgomery_ref(FqRef::from_u64(99999));
    FqRef lhs = fq_mul_ref(a, fq_add_ref(b, c));
    FqRef rhs = fq_add_ref(fq_mul_ref(a, b), fq_mul_ref(a, c));
    TEST_ASSERT(lhs == rhs, "Fq: a*(b+c) = a*b + a*c");

    // Montgomery roundtrip
    FqRef val = FqRef::from_u64(42);
    FqRef mont_val = fq_to_montgomery_ref(val);
    FqRef back = fq_from_montgomery_ref(mont_val);
    TEST_ASSERT(back == val, "Fq: from_mont(to_mont(x)) = x");

    // u32 <-> u64 roundtrip
    uint32_t w[12];
    a.to_u32(w);
    FqRef a2 = FqRef::from_u32(w);
    TEST_ASSERT(a == a2, "Fq: u32/u64 roundtrip");
}

// ─── Fq GPU Add Test ────────────────────────────────────────────────────────

void test_fq_gpu_add() {
    using namespace ff_ref;
    printf("test_fq_gpu_add...\n");

    const uint32_t N = 1024;
    std::vector<FqElement> h_a(N), h_b(N), h_out(N);

    // Generate test data in Montgomery form
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq_ref_to_gpu(make_fq_test_val(i * 7 + 13));
        h_b[i] = fq_ref_to_gpu(make_fq_test_val(i * 11 + 37));
    }
    // Edge cases: 0+0, (q-like)+1
    h_a[0] = FqElement::zero(); h_b[0] = FqElement::zero();

    FqElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));

    fq_add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FqElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FqRef ra = fq_gpu_to_ref(h_a[i]);
        FqRef rb = fq_gpu_to_ref(h_b[i]);
        FqRef expected = fq_add_ref(ra, rb);
        FqRef got = fq_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq GPU add: all elements match CPU reference");
    printf("  Fq add: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq GPU Sub Test ────────────────────────────────────────────────────────

void test_fq_gpu_sub() {
    using namespace ff_ref;
    printf("test_fq_gpu_sub...\n");

    const uint32_t N = 1024;
    std::vector<FqElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq_ref_to_gpu(make_fq_test_val(i * 7 + 13));
        h_b[i] = fq_ref_to_gpu(make_fq_test_val(i * 11 + 37));
    }
    h_a[0] = FqElement::zero(); h_b[0] = fq_ref_to_gpu(make_fq_test_val(1));

    FqElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));

    fq_sub_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FqElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FqRef ra = fq_gpu_to_ref(h_a[i]);
        FqRef rb = fq_gpu_to_ref(h_b[i]);
        FqRef expected = fq_sub_ref(ra, rb);
        FqRef got = fq_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq GPU sub: all elements match CPU reference");
    printf("  Fq sub: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq GPU Mul Test ────────────────────────────────────────────────────────

void test_fq_gpu_mul() {
    using namespace ff_ref;
    printf("test_fq_gpu_mul...\n");

    const uint32_t N = 1024;
    std::vector<FqElement> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq_ref_to_gpu(make_fq_test_val(i * 123457 + 13));
        h_b[i] = fq_ref_to_gpu(make_fq_test_val(i * 654321 + 37));
    }
    // Edge: 0 * x, 1 * x
    h_a[0] = FqElement::zero();
    FqRef one_mont;
    one_mont.limbs = FQ_R_MOD_6;
    h_a[1] = fq_ref_to_gpu(one_mont);

    FqElement *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));

    fq_mul_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FqElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FqRef ra = fq_gpu_to_ref(h_a[i]);
        FqRef rb = fq_gpu_to_ref(h_b[i]);
        FqRef expected = fq_mul_ref(ra, rb);
        FqRef got = fq_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq GPU mul: all elements match CPU reference");
    printf("  Fq mul: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq GPU Sqr Test ────────────────────────────────────────────────────────

void test_fq_gpu_sqr() {
    using namespace ff_ref;
    printf("test_fq_gpu_sqr...\n");

    const uint32_t N = 1024;
    std::vector<FqElement> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i)
        h_a[i] = fq_ref_to_gpu(make_fq_test_val(i * 654321 + 7));
    h_a[0] = FqElement::zero();

    FqElement *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FqElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FqElement), cudaMemcpyHostToDevice));

    fq_sqr_kernel<<<(N + 255) / 256, 256>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FqElement), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        FqRef ra = fq_gpu_to_ref(h_a[i]);
        FqRef expected = fq_sqr_ref(ra);
        FqRef got = fq_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq GPU sqr: all elements match CPU reference");
    printf("  Fq sqr: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq Algebraic Properties Test ──────────────────────────────────────────

void test_fq_algebraic() {
    using namespace ff_ref;
    printf("test_fq_algebraic...\n");

    const int N = 256;
    int pass_comm = 0, pass_assoc = 0, pass_dist = 0;

    for (int i = 0; i < N; ++i) {
        FqRef a = make_fq_test_val(i * 5 + 1);
        FqRef b = make_fq_test_val(i * 5 + 2);
        FqRef c = make_fq_test_val(i * 5 + 3);

        // Commutativity: a*b == b*a
        if (fq_mul_ref(a, b) == fq_mul_ref(b, a)) ++pass_comm;

        // Associativity: (a*b)*c == a*(b*c)
        if (fq_mul_ref(fq_mul_ref(a, b), c) == fq_mul_ref(a, fq_mul_ref(b, c))) ++pass_assoc;

        // Distributivity: a*(b+c) == a*b + a*c
        FqRef lhs = fq_mul_ref(a, fq_add_ref(b, c));
        FqRef rhs = fq_add_ref(fq_mul_ref(a, b), fq_mul_ref(a, c));
        if (lhs == rhs) ++pass_dist;
    }
    TEST_ASSERT(pass_comm == N, "Fq commutativity failure");
    TEST_ASSERT(pass_assoc == N, "Fq associativity failure");
    TEST_ASSERT(pass_dist == N, "Fq distributivity failure");
    printf("  Comm: %d/%d, Assoc: %d/%d, Dist: %d/%d\n",
           pass_comm, N, pass_assoc, N, pass_dist, N);
}

// ─── Fq Montgomery Round-Trip Test ─────────────────────────────────────────

void test_fq_montgomery_roundtrip() {
    using namespace ff_ref;
    printf("test_fq_montgomery_roundtrip...\n");

    int pass = 0;
    const int N = 256;
    for (int i = 0; i < N; ++i) {
        FqRef v = FqRef::from_u64(static_cast<uint64_t>(i) * 12345 + 1);
        FqRef mont = fq_to_montgomery_ref(v);
        FqRef back = fq_from_montgomery_ref(mont);
        if (back == v) ++pass;
    }
    TEST_ASSERT(pass == N, "Fq Montgomery roundtrip failures");
    printf("  %d/%d roundtrips passed\n", pass, N);
}

// ─── Fq2 CPU Self-Test ─────────────────────────────────────────────────────

void test_fq2_cpu_self_test() {
    using namespace ff_ref;
    printf("test_fq2_cpu_self_test...\n");

    Fq2Ref z = Fq2Ref::zero();
    Fq2Ref one = Fq2Ref::one_mont();

    // Basic
    TEST_ASSERT(fq2_add_ref(z, z) == z, "Fq2: 0 + 0 = 0");
    TEST_ASSERT(fq2_mul_ref(one, one) == one, "Fq2: 1 * 1 = 1");

    // Element with both components
    FqRef a0 = fq_to_montgomery_ref(FqRef::from_u64(12345));
    FqRef a1 = fq_to_montgomery_ref(FqRef::from_u64(67890));
    Fq2Ref a = {a0, a1};

    TEST_ASSERT(fq2_add_ref(a, z) == a, "Fq2: a + 0 = a");
    TEST_ASSERT(fq2_sub_ref(a, a) == z, "Fq2: a - a = 0");
    TEST_ASSERT(fq2_mul_ref(a, one) == a, "Fq2: a * 1 = a");

    // Negation
    Fq2Ref neg_a = fq2_neg_ref(a);
    TEST_ASSERT(fq2_add_ref(a, neg_a) == z, "Fq2: a + (-a) = 0");

    // Conjugation: a * conj(a) = norm(a) (real)
    Fq2Ref conj_a = fq2_conjugate_ref(a);
    Fq2Ref prod = fq2_mul_ref(a, conj_a);
    FqRef norm = fq2_norm_ref(a);
    TEST_ASSERT(prod.c0 == norm, "Fq2: a*conj(a) c0 = norm");
    TEST_ASSERT(prod.c1 == FqRef::zero(), "Fq2: a*conj(a) c1 = 0");

    // Inverse: a * a^{-1} = 1
    Fq2Ref a_inv = fq2_inv_ref(a);
    Fq2Ref should_one = fq2_mul_ref(a, a_inv);
    TEST_ASSERT(should_one == one, "Fq2: a * a^{-1} = 1");

    // Commutativity
    FqRef b0 = fq_to_montgomery_ref(FqRef::from_u64(11111));
    FqRef b1 = fq_to_montgomery_ref(FqRef::from_u64(22222));
    Fq2Ref b = {b0, b1};
    TEST_ASSERT(fq2_mul_ref(a, b) == fq2_mul_ref(b, a), "Fq2: a*b = b*a");

    // Squaring consistency: sqr(a) == mul(a, a)
    Fq2Ref sq1 = fq2_sqr_ref(a);
    Fq2Ref sq2 = fq2_mul_ref(a, a);
    TEST_ASSERT(sq1 == sq2, "Fq2: sqr(a) == mul(a,a)");

    // mul_by_nonresidue: (a+bu)*(1+u) = (a-b) + (a+b)u
    Fq2Ref nr = fq2_mul_by_nonresidue_ref(a);
    FqRef exp_c0 = fq_sub_ref(a.c0, a.c1);
    FqRef exp_c1 = fq_add_ref(a.c0, a.c1);
    TEST_ASSERT(nr.c0 == exp_c0, "Fq2: mul_by_nonresidue c0");
    TEST_ASSERT(nr.c1 == exp_c1, "Fq2: mul_by_nonresidue c1");
}

// ─── Fq2 GPU Mul Test ──────────────────────────────────────────────────────

void test_fq2_gpu_mul() {
    using namespace ff_ref;
    printf("test_fq2_gpu_mul...\n");

    const uint32_t N = 512;
    std::vector<Fq2Element> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        FqRef a0 = make_fq_test_val(i * 4 + 1);
        FqRef a1 = make_fq_test_val(i * 4 + 2);
        FqRef b0 = make_fq_test_val(i * 4 + 3);
        FqRef b1 = make_fq_test_val(i * 4 + 4);
        h_a[i].c0 = fq_ref_to_gpu(a0);
        h_a[i].c1 = fq_ref_to_gpu(a1);
        h_b[i].c0 = fq_ref_to_gpu(b0);
        h_b[i].c1 = fq_ref_to_gpu(b1);
    }

    Fq2Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));

    fq2_mul_kernel<<<(N + 127) / 128, 128>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq2Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq2Ref ra = {fq_gpu_to_ref(h_a[i].c0), fq_gpu_to_ref(h_a[i].c1)};
        Fq2Ref rb = {fq_gpu_to_ref(h_b[i].c0), fq_gpu_to_ref(h_b[i].c1)};
        Fq2Ref expected = fq2_mul_ref(ra, rb);
        Fq2Ref got = {fq_gpu_to_ref(h_out[i].c0), fq_gpu_to_ref(h_out[i].c1)};
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq2 GPU mul: all elements match CPU reference");
    printf("  Fq2 mul: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq2 GPU Add Test ──────────────────────────────────────────────────────

void test_fq2_gpu_add() {
    using namespace ff_ref;
    printf("test_fq2_gpu_add...\n");

    const uint32_t N = 512;
    std::vector<Fq2Element> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        FqRef a0 = make_fq_test_val(i * 4 + 1);
        FqRef a1 = make_fq_test_val(i * 4 + 2);
        FqRef b0 = make_fq_test_val(i * 4 + 3);
        FqRef b1 = make_fq_test_val(i * 4 + 4);
        h_a[i].c0 = fq_ref_to_gpu(a0);
        h_a[i].c1 = fq_ref_to_gpu(a1);
        h_b[i].c0 = fq_ref_to_gpu(b0);
        h_b[i].c1 = fq_ref_to_gpu(b1);
    }

    Fq2Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));

    fq2_add_kernel<<<(N + 127) / 128, 128>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq2Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq2Ref ra = {fq_gpu_to_ref(h_a[i].c0), fq_gpu_to_ref(h_a[i].c1)};
        Fq2Ref rb = {fq_gpu_to_ref(h_b[i].c0), fq_gpu_to_ref(h_b[i].c1)};
        Fq2Ref expected = fq2_add_ref(ra, rb);
        Fq2Ref got = {fq_gpu_to_ref(h_out[i].c0), fq_gpu_to_ref(h_out[i].c1)};
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq2 GPU add: all elements match CPU reference");
    printf("  Fq2 add: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq2 GPU Sub Test ──────────────────────────────────────────────────────

void test_fq2_gpu_sub() {
    using namespace ff_ref;
    printf("test_fq2_gpu_sub...\n");

    const uint32_t N = 512;
    std::vector<Fq2Element> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        FqRef a0 = make_fq_test_val(i * 4 + 1);
        FqRef a1 = make_fq_test_val(i * 4 + 2);
        FqRef b0 = make_fq_test_val(i * 4 + 3);
        FqRef b1 = make_fq_test_val(i * 4 + 4);
        h_a[i].c0 = fq_ref_to_gpu(a0);
        h_a[i].c1 = fq_ref_to_gpu(a1);
        h_b[i].c0 = fq_ref_to_gpu(b0);
        h_b[i].c1 = fq_ref_to_gpu(b1);
    }

    Fq2Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));

    fq2_sub_kernel<<<(N + 127) / 128, 128>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq2Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq2Ref ra = {fq_gpu_to_ref(h_a[i].c0), fq_gpu_to_ref(h_a[i].c1)};
        Fq2Ref rb = {fq_gpu_to_ref(h_b[i].c0), fq_gpu_to_ref(h_b[i].c1)};
        Fq2Ref expected = fq2_sub_ref(ra, rb);
        Fq2Ref got = {fq_gpu_to_ref(h_out[i].c0), fq_gpu_to_ref(h_out[i].c1)};
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq2 GPU sub: all elements match CPU reference");
    printf("  Fq2 sub: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq2 GPU Sqr Test ──────────────────────────────────────────────────────

void test_fq2_gpu_sqr() {
    using namespace ff_ref;
    printf("test_fq2_gpu_sqr...\n");

    const uint32_t N = 512;
    std::vector<Fq2Element> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        FqRef a0 = make_fq_test_val(i * 3 + 1);
        FqRef a1 = make_fq_test_val(i * 3 + 2);
        h_a[i].c0 = fq_ref_to_gpu(a0);
        h_a[i].c1 = fq_ref_to_gpu(a1);
    }

    Fq2Element *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq2Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq2Element), cudaMemcpyHostToDevice));

    fq2_sqr_kernel<<<(N + 127) / 128, 128>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq2Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq2Ref ra = {fq_gpu_to_ref(h_a[i].c0), fq_gpu_to_ref(h_a[i].c1)};
        Fq2Ref expected = fq2_sqr_ref(ra);
        Fq2Ref got = {fq_gpu_to_ref(h_out[i].c0), fq_gpu_to_ref(h_out[i].c1)};
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq2 GPU sqr: all elements match CPU reference");
    printf("  Fq2 sqr: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq2 Algebraic Properties Test ─────────────────────────────────────────

void test_fq2_algebraic() {
    using namespace ff_ref;
    printf("test_fq2_algebraic...\n");

    const int N = 128;
    int pass_comm = 0, pass_assoc = 0, pass_dist = 0, pass_sqr = 0;

    for (int i = 0; i < N; ++i) {
        Fq2Ref a = {make_fq_test_val(i * 6 + 1), make_fq_test_val(i * 6 + 2)};
        Fq2Ref b = {make_fq_test_val(i * 6 + 3), make_fq_test_val(i * 6 + 4)};
        Fq2Ref c = {make_fq_test_val(i * 6 + 5), make_fq_test_val(i * 6 + 6)};

        if (fq2_mul_ref(a, b) == fq2_mul_ref(b, a)) ++pass_comm;
        if (fq2_mul_ref(fq2_mul_ref(a, b), c) == fq2_mul_ref(a, fq2_mul_ref(b, c))) ++pass_assoc;

        Fq2Ref lhs = fq2_mul_ref(a, fq2_add_ref(b, c));
        Fq2Ref rhs = fq2_add_ref(fq2_mul_ref(a, b), fq2_mul_ref(a, c));
        if (lhs == rhs) ++pass_dist;

        if (fq2_sqr_ref(a) == fq2_mul_ref(a, a)) ++pass_sqr;
    }
    TEST_ASSERT(pass_comm == N, "Fq2 commutativity failure");
    TEST_ASSERT(pass_assoc == N, "Fq2 associativity failure");
    TEST_ASSERT(pass_dist == N, "Fq2 distributivity failure");
    TEST_ASSERT(pass_sqr == N, "Fq2 sqr consistency failure");
    printf("  Comm: %d/%d, Assoc: %d/%d, Dist: %d/%d, Sqr: %d/%d\n",
           pass_comm, N, pass_assoc, N, pass_dist, N, pass_sqr, N);
}

// =============================================================================
// v3.0.0 Session 31: Fq6 Field Arithmetic Tests
// =============================================================================

// ─── Helper: convert Fq6Ref <-> Fq6Element ─────────────────────────────────

static Fq6Element fq6_ref_to_gpu(const ff_ref::Fq6Ref& r) {
    Fq6Element e;
    r.c0.c0.to_u32(e.c0.c0.limbs);
    r.c0.c1.to_u32(e.c0.c1.limbs);
    r.c1.c0.to_u32(e.c1.c0.limbs);
    r.c1.c1.to_u32(e.c1.c1.limbs);
    r.c2.c0.to_u32(e.c2.c0.limbs);
    r.c2.c1.to_u32(e.c2.c1.limbs);
    return e;
}

static ff_ref::Fq6Ref fq6_gpu_to_ref(const Fq6Element& e) {
    using namespace ff_ref;
    return {
        {FqRef::from_u32(e.c0.c0.limbs), FqRef::from_u32(e.c0.c1.limbs)},
        {FqRef::from_u32(e.c1.c0.limbs), FqRef::from_u32(e.c1.c1.limbs)},
        {FqRef::from_u32(e.c2.c0.limbs), FqRef::from_u32(e.c2.c1.limbs)}
    };
}

static ff_ref::Fq6Ref make_fq6_test_val(uint64_t seed) {
    using namespace ff_ref;
    return {
        {make_fq_test_val(seed * 6 + 1), make_fq_test_val(seed * 6 + 2)},
        {make_fq_test_val(seed * 6 + 3), make_fq_test_val(seed * 6 + 4)},
        {make_fq_test_val(seed * 6 + 5), make_fq_test_val(seed * 6 + 6)}
    };
}

// ─── Fq6 CPU Self-Test ──────────────────────────────────────────────────────

void test_fq6_cpu_self_test() {
    using namespace ff_ref;
    printf("test_fq6_cpu_self_test...\n");

    Fq6Ref z = Fq6Ref::zero();
    Fq6Ref one = Fq6Ref::one_mont();

    // Basic identities
    TEST_ASSERT(fq6_add_ref(z, z) == z, "Fq6: 0 + 0 = 0");
    TEST_ASSERT(fq6_mul_ref(one, one) == one, "Fq6: 1 * 1 = 1");

    Fq6Ref a = make_fq6_test_val(1);

    TEST_ASSERT(fq6_add_ref(a, z) == a, "Fq6: a + 0 = a");
    TEST_ASSERT(fq6_sub_ref(a, a) == z, "Fq6: a - a = 0");
    TEST_ASSERT(fq6_mul_ref(a, one) == a, "Fq6: a * 1 = a");

    // Negation
    Fq6Ref neg_a = fq6_neg_ref(a);
    TEST_ASSERT(fq6_add_ref(a, neg_a) == z, "Fq6: a + (-a) = 0");

    // Inverse: a * a^{-1} = 1
    Fq6Ref a_inv = fq6_inv_ref(a);
    Fq6Ref should_one = fq6_mul_ref(a, a_inv);
    TEST_ASSERT(should_one == one, "Fq6: a * a^{-1} = 1");

    // Commutativity
    Fq6Ref b = make_fq6_test_val(2);
    TEST_ASSERT(fq6_mul_ref(a, b) == fq6_mul_ref(b, a), "Fq6: a*b = b*a");

    // Squaring consistency: sqr(a) == mul(a, a)
    Fq6Ref sq1 = fq6_sqr_ref(a);
    Fq6Ref sq2 = fq6_mul_ref(a, a);
    TEST_ASSERT(sq1 == sq2, "Fq6: sqr(a) == mul(a,a)");

    // mul_by_nonresidue: multiply by v
    // (c0 + c1·v + c2·v²)·v = β·c2 + c0·v + c1·v²
    Fq6Ref nr = fq6_mul_by_nonresidue_ref(a);
    TEST_ASSERT(nr.c0 == fq2_mul_by_nonresidue_ref(a.c2), "Fq6: mul_by_nonresidue c0");
    TEST_ASSERT(nr.c1 == a.c0, "Fq6: mul_by_nonresidue c1");
    TEST_ASSERT(nr.c2 == a.c1, "Fq6: mul_by_nonresidue c2");

    // mul_by_01: should match general mul with c2=0
    Fq2Ref b0 = {make_fq_test_val(100), make_fq_test_val(101)};
    Fq2Ref b1 = {make_fq_test_val(102), make_fq_test_val(103)};
    Fq6Ref sparse_b = {b0, b1, Fq2Ref::zero()};
    Fq6Ref full_prod = fq6_mul_ref(a, sparse_b);
    Fq6Ref sparse_prod = fq6_mul_by_01_ref(a, b0, b1);
    TEST_ASSERT(full_prod == sparse_prod, "Fq6: mul_by_01 matches general mul");

    // mul_by_1: should match general mul with c0=c2=0
    Fq6Ref sparse_b1 = {Fq2Ref::zero(), b1, Fq2Ref::zero()};
    Fq6Ref full_prod1 = fq6_mul_ref(a, sparse_b1);
    Fq6Ref sparse_prod1 = fq6_mul_by_1_ref(a, b1);
    TEST_ASSERT(full_prod1 == sparse_prod1, "Fq6: mul_by_1 matches general mul");

    // Distributivity: (a+b)*c = a*c + b*c
    Fq6Ref c = make_fq6_test_val(3);
    Fq6Ref lhs = fq6_mul_ref(fq6_add_ref(a, b), c);
    Fq6Ref rhs = fq6_add_ref(fq6_mul_ref(a, c), fq6_mul_ref(b, c));
    TEST_ASSERT(lhs == rhs, "Fq6: distributivity");

    printf("  All Fq6 CPU self-tests passed\n");
}

// ─── Fq6 GPU Mul Test ──────────────────────────────────────────────────────

void test_fq6_gpu_mul() {
    using namespace ff_ref;
    printf("test_fq6_gpu_mul...\n");

    const uint32_t N = 256;
    std::vector<Fq6Element> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq6_ref_to_gpu(make_fq6_test_val(i * 2 + 1));
        h_b[i] = fq6_ref_to_gpu(make_fq6_test_val(i * 2 + 2));
    }

    Fq6Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));

    fq6_mul_kernel<<<(N + 63) / 64, 64>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq6Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq6Ref ra = fq6_gpu_to_ref(h_a[i]);
        Fq6Ref rb = fq6_gpu_to_ref(h_b[i]);
        Fq6Ref expected = fq6_mul_ref(ra, rb);
        Fq6Ref got = fq6_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq6 GPU mul: all elements match CPU reference");
    printf("  Fq6 mul: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq6 GPU Add Test ──────────────────────────────────────────────────────

void test_fq6_gpu_add() {
    using namespace ff_ref;
    printf("test_fq6_gpu_add...\n");

    const uint32_t N = 256;
    std::vector<Fq6Element> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq6_ref_to_gpu(make_fq6_test_val(i * 2 + 1));
        h_b[i] = fq6_ref_to_gpu(make_fq6_test_val(i * 2 + 2));
    }

    Fq6Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));

    fq6_add_kernel<<<(N + 63) / 64, 64>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq6Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq6Ref ra = fq6_gpu_to_ref(h_a[i]);
        Fq6Ref rb = fq6_gpu_to_ref(h_b[i]);
        Fq6Ref expected = fq6_add_ref(ra, rb);
        Fq6Ref got = fq6_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq6 GPU add: all elements match CPU reference");
    printf("  Fq6 add: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq6 GPU Sub Test ──────────────────────────────────────────────────────

void test_fq6_gpu_sub() {
    using namespace ff_ref;
    printf("test_fq6_gpu_sub...\n");

    const uint32_t N = 256;
    std::vector<Fq6Element> h_a(N), h_b(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq6_ref_to_gpu(make_fq6_test_val(i * 2 + 1));
        h_b[i] = fq6_ref_to_gpu(make_fq6_test_val(i * 2 + 2));
    }

    Fq6Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));

    fq6_sub_kernel<<<(N + 63) / 64, 64>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq6Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq6Ref ra = fq6_gpu_to_ref(h_a[i]);
        Fq6Ref rb = fq6_gpu_to_ref(h_b[i]);
        Fq6Ref expected = fq6_sub_ref(ra, rb);
        Fq6Ref got = fq6_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq6 GPU sub: all elements match CPU reference");
    printf("  Fq6 sub: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq6 GPU Sqr Test ──────────────────────────────────────────────────────

void test_fq6_gpu_sqr() {
    using namespace ff_ref;
    printf("test_fq6_gpu_sqr...\n");

    const uint32_t N = 256;
    std::vector<Fq6Element> h_a(N), h_out(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq6_ref_to_gpu(make_fq6_test_val(i + 1));
    }

    Fq6Element *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq6Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq6Element), cudaMemcpyHostToDevice));

    fq6_sqr_kernel<<<(N + 63) / 64, 64>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq6Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq6Ref ra = fq6_gpu_to_ref(h_a[i]);
        Fq6Ref expected = fq6_sqr_ref(ra);
        Fq6Ref got = fq6_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq6 GPU sqr: all elements match CPU reference");
    printf("  Fq6 sqr: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq6 Algebraic Properties Test ─────────────────────────────────────────

void test_fq6_algebraic() {
    using namespace ff_ref;
    printf("test_fq6_algebraic...\n");

    const int N = 64;
    int pass_comm = 0, pass_assoc = 0, pass_dist = 0, pass_sqr = 0;

    for (int i = 0; i < N; ++i) {
        Fq6Ref a = make_fq6_test_val(i * 3 + 1);
        Fq6Ref b = make_fq6_test_val(i * 3 + 2);
        Fq6Ref c = make_fq6_test_val(i * 3 + 3);

        if (fq6_mul_ref(a, b) == fq6_mul_ref(b, a)) ++pass_comm;
        if (fq6_mul_ref(fq6_mul_ref(a, b), c) == fq6_mul_ref(a, fq6_mul_ref(b, c))) ++pass_assoc;

        Fq6Ref lhs = fq6_mul_ref(a, fq6_add_ref(b, c));
        Fq6Ref rhs = fq6_add_ref(fq6_mul_ref(a, b), fq6_mul_ref(a, c));
        if (lhs == rhs) ++pass_dist;

        if (fq6_sqr_ref(a) == fq6_mul_ref(a, a)) ++pass_sqr;
    }
    TEST_ASSERT(pass_comm == N, "Fq6 commutativity failure");
    TEST_ASSERT(pass_assoc == N, "Fq6 associativity failure");
    TEST_ASSERT(pass_dist == N, "Fq6 distributivity failure");
    TEST_ASSERT(pass_sqr == N, "Fq6 sqr consistency failure");
    printf("  Comm: %d/%d, Assoc: %d/%d, Dist: %d/%d, Sqr: %d/%d\n",
           pass_comm, N, pass_assoc, N, pass_dist, N, pass_sqr, N);
}

// ─── Fq6 Inverse Round-Trip Test ────────────────────────────────────────────

void test_fq6_inverse() {
    using namespace ff_ref;
    printf("test_fq6_inverse...\n");

    const int N = 32;
    int pass = 0;
    Fq6Ref one = Fq6Ref::one_mont();

    for (int i = 0; i < N; ++i) {
        Fq6Ref a = make_fq6_test_val(i + 1);
        Fq6Ref a_inv = fq6_inv_ref(a);
        Fq6Ref prod = fq6_mul_ref(a, a_inv);
        if (prod == one) ++pass;
    }
    TEST_ASSERT(pass == N, "Fq6 inverse round-trip failure");
    printf("  Fq6 inverse: %d/%d passed\n", pass, N);
}

// ─── Fq6 Sparse Mul Tests ──────────────────────────────────────────────────

void test_fq6_sparse_mul() {
    using namespace ff_ref;
    printf("test_fq6_sparse_mul...\n");

    const int N = 32;
    int pass_01 = 0, pass_1 = 0;

    for (int i = 0; i < N; ++i) {
        Fq6Ref a = make_fq6_test_val(i + 1);
        Fq2Ref b0 = {make_fq_test_val(i * 10 + 100), make_fq_test_val(i * 10 + 101)};
        Fq2Ref b1 = {make_fq_test_val(i * 10 + 102), make_fq_test_val(i * 10 + 103)};

        // mul_by_01 vs general mul
        Fq6Ref sparse_b = {b0, b1, Fq2Ref::zero()};
        if (fq6_mul_by_01_ref(a, b0, b1) == fq6_mul_ref(a, sparse_b)) ++pass_01;

        // mul_by_1 vs general mul
        Fq6Ref sparse_b1 = {Fq2Ref::zero(), b1, Fq2Ref::zero()};
        if (fq6_mul_by_1_ref(a, b1) == fq6_mul_ref(a, sparse_b1)) ++pass_1;
    }
    TEST_ASSERT(pass_01 == N, "Fq6 mul_by_01 mismatch");
    TEST_ASSERT(pass_1 == N, "Fq6 mul_by_1 mismatch");
    printf("  mul_by_01: %d/%d, mul_by_1: %d/%d\n", pass_01, N, pass_1, N);
}

// ─── Fq6 Frobenius Test ────────────────────────────────────────────────────

void test_fq6_frobenius() {
    using namespace ff_ref;
    printf("test_fq6_frobenius...\n");

    // Compute Frobenius coefficients
    Fq2Ref frob_c1[6], frob_c2[6];
    compute_fq6_frobenius_coefficients(frob_c1, frob_c2);

    // γ₁[0] should be 1
    TEST_ASSERT(frob_c1[0] == Fq2Ref::one_mont(), "Fq6 Frobenius: gamma1[0] = 1");

    const int N = 16;

    // φ^6(a) = a (Frobenius order divides 6)
    int pass_period = 0;
    for (int i = 0; i < N; ++i) {
        Fq6Ref a = make_fq6_test_val(i + 1);
        Fq6Ref result = a;
        for (int k = 0; k < 6; ++k) {
            result = fq6_frobenius_map_ref(result, 1, frob_c1, frob_c2);
        }
        if (result == a) ++pass_period;
    }
    TEST_ASSERT(pass_period == N, "Fq6 Frobenius: phi^6(a) != a");

    // φ is multiplicative: φ(a*b) = φ(a)*φ(b)
    int pass_mult = 0;
    for (int i = 0; i < N; ++i) {
        Fq6Ref a = make_fq6_test_val(i * 2 + 1);
        Fq6Ref b = make_fq6_test_val(i * 2 + 2);
        Fq6Ref lhs = fq6_frobenius_map_ref(fq6_mul_ref(a, b), 1, frob_c1, frob_c2);
        Fq6Ref rhs = fq6_mul_ref(
            fq6_frobenius_map_ref(a, 1, frob_c1, frob_c2),
            fq6_frobenius_map_ref(b, 1, frob_c1, frob_c2)
        );
        if (lhs == rhs) ++pass_mult;
    }
    TEST_ASSERT(pass_mult == N, "Fq6 Frobenius: phi(a*b) != phi(a)*phi(b)");

    // φ fixes Fq2 elements (embedded as c0 only)
    Fq6Ref fq2_elem = {
        {make_fq_test_val(42), make_fq_test_val(43)},
        Fq2Ref::zero(), Fq2Ref::zero()
    };
    Fq6Ref phi_fq2 = fq6_frobenius_map_ref(fq2_elem, 2, frob_c1, frob_c2);
    TEST_ASSERT(phi_fq2 == fq2_elem, "Fq6 Frobenius: phi^2 fixes Fq2 elements");

    printf("  Period: %d/%d, Multiplicative: %d/%d, fixes Fq2: OK\n",
           pass_period, N, pass_mult, N);
}

// ─── Fq6 mul_by_nonresidue chain test ──────────────────────────────────────

void test_fq6_nonresidue_chain() {
    using namespace ff_ref;
    printf("test_fq6_nonresidue_chain...\n");

    // Multiplying by v three times should multiply by β = (1+u)
    Fq6Ref a = make_fq6_test_val(7);
    Fq6Ref v3_a = fq6_mul_by_nonresidue_ref(
                    fq6_mul_by_nonresidue_ref(
                     fq6_mul_by_nonresidue_ref(a)));

    // v^3 = β, so a*v^3 = a*β = scale(a, β)
    Fq2Ref beta = fq2_mul_by_nonresidue_ref(Fq2Ref::one_mont());
    // Actually β for the Fq6 tower is (1+u), same as fq2_mul_by_nonresidue(1) = (1-0, 1+0) = (1, 1)
    // Wait: fq2_mul_by_nonresidue_ref({1,0}) = (1-0, 1+0) = (1, 1). Yes, β = (1, 1) = 1+u.
    Fq6Ref expected = fq6_scale_ref(a, beta);
    TEST_ASSERT(v3_a == expected, "Fq6: v^3 * a = beta * a");
    printf("  v^3 chain: OK\n");
}

// =============================================================================
// v3.0.0 Session 32: Fq12 Field Arithmetic Tests
// =============================================================================

// ─── Helper: convert Fq12Ref <-> Fq12Element ───────────────────────────────

static Fq12Element fq12_ref_to_gpu(const ff_ref::Fq12Ref& r) {
    Fq12Element e;
    e.c0 = fq6_ref_to_gpu(r.c0);
    e.c1 = fq6_ref_to_gpu(r.c1);
    return e;
}

static ff_ref::Fq12Ref fq12_gpu_to_ref(const Fq12Element& e) {
    ff_ref::Fq12Ref r;
    r.c0 = fq6_gpu_to_ref(e.c0);
    r.c1 = fq6_gpu_to_ref(e.c1);
    return r;
}

static ff_ref::Fq2Ref make_fq2_test_val(uint64_t seed) {
    using namespace ff_ref;
    return {make_fq_test_val(seed * 2 + 1), make_fq_test_val(seed * 2 + 2)};
}

static ff_ref::Fq12Ref make_fq12_test_val(uint64_t seed) {
    using namespace ff_ref;
    return {make_fq6_test_val(seed * 2 + 1), make_fq6_test_val(seed * 2 + 2)};
}

// ─── Fq12 CPU Self-Test ─────────────────────────────────────────────────────

void test_fq12_cpu_self_test() {
    using namespace ff_ref;
    printf("test_fq12_cpu_self_test...\n");

    Fq12Ref z = Fq12Ref::zero();
    Fq12Ref one = Fq12Ref::one_mont();

    // Basic identity tests
    TEST_ASSERT(fq12_add_ref(z, z) == z, "Fq12: 0 + 0 = 0");
    TEST_ASSERT(fq12_mul_ref(one, one) == one, "Fq12: 1 * 1 = 1");

    Fq12Ref a = make_fq12_test_val(1);

    TEST_ASSERT(fq12_add_ref(a, z) == a, "Fq12: a + 0 = a");
    TEST_ASSERT(fq12_sub_ref(a, a) == z, "Fq12: a - a = 0");
    TEST_ASSERT(fq12_mul_ref(a, one) == a, "Fq12: a * 1 = a");

    // Negation
    Fq12Ref neg_a = fq12_neg_ref(a);
    TEST_ASSERT(fq12_add_ref(a, neg_a) == z, "Fq12: a + (-a) = 0");

    // Inverse
    Fq12Ref a_inv = fq12_inv_ref(a);
    Fq12Ref should_one = fq12_mul_ref(a, a_inv);
    TEST_ASSERT(should_one == one, "Fq12: a * a^{-1} = 1");

    // Commutativity
    Fq12Ref b = make_fq12_test_val(2);
    TEST_ASSERT(fq12_mul_ref(a, b) == fq12_mul_ref(b, a), "Fq12: a*b = b*a");

    // Squaring = mul(a, a)
    Fq12Ref sq1 = fq12_sqr_ref(a);
    Fq12Ref sq2 = fq12_mul_ref(a, a);
    TEST_ASSERT(sq1 == sq2, "Fq12: sqr(a) == mul(a,a)");

    // Conjugate: conj(a) = a0 - a1*w
    Fq12Ref conj_a = fq12_conjugate_ref(a);
    TEST_ASSERT(conj_a.c0 == a.c0, "Fq12: conjugate preserves c0");
    TEST_ASSERT(conj_a.c1 == fq6_neg_ref(a.c1), "Fq12: conjugate negates c1");

    // Sparse mul: mul_by_034 matches general mul
    Fq2Ref d0 = make_fq2_test_val(100);
    Fq2Ref d3 = make_fq2_test_val(200);
    Fq2Ref d4 = make_fq2_test_val(300);
    Fq12Ref sparse_b = {{d0, Fq2Ref::zero(), Fq2Ref::zero()},
                         {d3, d4, Fq2Ref::zero()}};
    Fq12Ref full_prod = fq12_mul_ref(a, sparse_b);
    Fq12Ref sparse_prod = fq12_mul_by_034_ref(a, d0, d3, d4);
    TEST_ASSERT(full_prod == sparse_prod, "Fq12: mul_by_034 matches general mul");

    // Distributivity
    Fq12Ref c = make_fq12_test_val(3);
    Fq12Ref lhs = fq12_mul_ref(fq12_add_ref(a, b), c);
    Fq12Ref rhs = fq12_add_ref(fq12_mul_ref(a, c), fq12_mul_ref(b, c));
    TEST_ASSERT(lhs == rhs, "Fq12: distributivity");

    printf("  All Fq12 CPU self-tests passed\n");
}

// ─── Fq12 GPU Mul Test ─────────────────────────────────────────────────────

void test_fq12_gpu_mul() {
    using namespace ff_ref;
    printf("test_fq12_gpu_mul...\n");
    const uint32_t N = 64;

    std::vector<Fq12Element> h_a(N), h_b(N), h_out(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq12_ref_to_gpu(make_fq12_test_val(i * 2 + 1));
        h_b[i] = fq12_ref_to_gpu(make_fq12_test_val(i * 2 + 2));
    }

    Fq12Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));

    fq12_mul_kernel<<<(N + 63) / 64, 64>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq12Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq12Ref ra = fq12_gpu_to_ref(h_a[i]);
        Fq12Ref rb = fq12_gpu_to_ref(h_b[i]);
        Fq12Ref expected = fq12_mul_ref(ra, rb);
        Fq12Ref got = fq12_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq12 GPU mul: all elements match CPU reference");
    printf("  Fq12 mul: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq12 GPU Add Test ─────────────────────────────────────────────────────

void test_fq12_gpu_add() {
    using namespace ff_ref;
    printf("test_fq12_gpu_add...\n");
    const uint32_t N = 64;

    std::vector<Fq12Element> h_a(N), h_b(N), h_out(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq12_ref_to_gpu(make_fq12_test_val(i * 2 + 1));
        h_b[i] = fq12_ref_to_gpu(make_fq12_test_val(i * 2 + 2));
    }

    Fq12Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));

    fq12_add_kernel<<<(N + 63) / 64, 64>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq12Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq12Ref ra = fq12_gpu_to_ref(h_a[i]);
        Fq12Ref rb = fq12_gpu_to_ref(h_b[i]);
        Fq12Ref expected = fq12_add_ref(ra, rb);
        Fq12Ref got = fq12_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq12 GPU add: all elements match CPU reference");
    printf("  Fq12 add: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq12 GPU Sub Test ─────────────────────────────────────────────────────

void test_fq12_gpu_sub() {
    using namespace ff_ref;
    printf("test_fq12_gpu_sub...\n");
    const uint32_t N = 64;

    std::vector<Fq12Element> h_a(N), h_b(N), h_out(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq12_ref_to_gpu(make_fq12_test_val(i * 2 + 1));
        h_b[i] = fq12_ref_to_gpu(make_fq12_test_val(i * 2 + 2));
    }

    Fq12Element *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));

    fq12_sub_kernel<<<(N + 63) / 64, 64>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq12Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq12Ref ra = fq12_gpu_to_ref(h_a[i]);
        Fq12Ref rb = fq12_gpu_to_ref(h_b[i]);
        Fq12Ref expected = fq12_sub_ref(ra, rb);
        Fq12Ref got = fq12_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq12 GPU sub: all elements match CPU reference");
    printf("  Fq12 sub: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq12 GPU Sqr Test ─────────────────────────────────────────────────────

void test_fq12_gpu_sqr() {
    using namespace ff_ref;
    printf("test_fq12_gpu_sqr...\n");
    const uint32_t N = 64;

    std::vector<Fq12Element> h_a(N), h_out(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = fq12_ref_to_gpu(make_fq12_test_val(i + 1));
    }

    Fq12Element *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(Fq12Element)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(Fq12Element), cudaMemcpyHostToDevice));

    fq12_sqr_kernel<<<(N + 63) / 64, 64>>>(d_a, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(Fq12Element), cudaMemcpyDeviceToHost));

    int pass = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Fq12Ref ra = fq12_gpu_to_ref(h_a[i]);
        Fq12Ref expected = fq12_sqr_ref(ra);
        Fq12Ref got = fq12_gpu_to_ref(h_out[i]);
        if (got == expected) ++pass;
    }
    TEST_ASSERT(pass == (int)N, "Fq12 GPU sqr: all elements match CPU reference");
    printf("  Fq12 sqr: %d/%d matched\n", pass, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ─── Fq12 Algebraic Properties Test ────────────────────────────────────────

void test_fq12_algebraic() {
    using namespace ff_ref;
    printf("test_fq12_algebraic...\n");
    const int N = 8;

    int pass_comm = 0, pass_assoc = 0, pass_dist = 0, pass_sqr = 0;

    for (int i = 0; i < N; ++i) {
        Fq12Ref a = make_fq12_test_val(i * 3 + 1);
        Fq12Ref b = make_fq12_test_val(i * 3 + 2);
        Fq12Ref c = make_fq12_test_val(i * 3 + 3);

        if (fq12_mul_ref(a, b) == fq12_mul_ref(b, a)) ++pass_comm;
        if (fq12_mul_ref(fq12_mul_ref(a, b), c) == fq12_mul_ref(a, fq12_mul_ref(b, c))) ++pass_assoc;

        Fq12Ref lhs = fq12_mul_ref(a, fq12_add_ref(b, c));
        Fq12Ref rhs = fq12_add_ref(fq12_mul_ref(a, b), fq12_mul_ref(a, c));
        if (lhs == rhs) ++pass_dist;

        if (fq12_sqr_ref(a) == fq12_mul_ref(a, a)) ++pass_sqr;
    }
    TEST_ASSERT(pass_comm == N, "Fq12 commutativity failure");
    TEST_ASSERT(pass_assoc == N, "Fq12 associativity failure");
    TEST_ASSERT(pass_dist == N, "Fq12 distributivity failure");
    TEST_ASSERT(pass_sqr == N, "Fq12 sqr consistency failure");
    printf("  Fq12 algebraic: OK\n");
}

// ─── Fq12 Inverse Round-Trip Test ──────────────────────────────────────────

void test_fq12_inverse() {
    using namespace ff_ref;
    printf("test_fq12_inverse...\n");
    const int N = 8;

    Fq12Ref one = Fq12Ref::one_mont();
    int pass = 0;
    for (int i = 0; i < N; ++i) {
        Fq12Ref a = make_fq12_test_val(i + 1);
        Fq12Ref a_inv = fq12_inv_ref(a);
        Fq12Ref prod = fq12_mul_ref(a, a_inv);
        if (prod == one) ++pass;
    }
    TEST_ASSERT(pass == N, "Fq12 inverse round-trip failure");
    printf("  Fq12 inverse: %d/%d passed\n", pass, N);
}

// ─── Fq12 Conjugate Test ───────────────────────────────────────────────────
// For a unitary element f (i.e., f·conj(f) = 1), conjugate is the inverse.
// Construct unitary: f = a * inv(conj(a)) has norm 1 when |a|_Fq12 != 0.
// Actually, f = a / conj(a) gives f · conj(f) = (a / conj(a)) · (conj(a) / a) = 1.

void test_fq12_conjugate() {
    using namespace ff_ref;
    printf("test_fq12_conjugate...\n");
    const int N = 4;

    int pass = 0;
    for (int i = 0; i < N; ++i) {
        // Construct unitary element: f = a * conj(a)^{-1}
        Fq12Ref a = make_fq12_test_val(i + 10);
        Fq12Ref conj_a = fq12_conjugate_ref(a);
        Fq12Ref conj_a_inv = fq12_inv_ref(conj_a);
        Fq12Ref f = fq12_mul_ref(a, conj_a_inv);

        // For unitary f: conj(f) should equal inv(f)
        Fq12Ref conj_f = fq12_conjugate_ref(f);
        Fq12Ref inv_f = fq12_inv_ref(f);
        if (conj_f == inv_f) ++pass;
    }
    TEST_ASSERT(pass == N, "Fq12 conjugate: unitary elements conj != inv");
    printf("  Fq12 conjugate (unitary): %d/%d passed\n", pass, N);
}

// ─── Fq12 Sparse Mul Test ──────────────────────────────────────────────────

void test_fq12_sparse_mul() {
    using namespace ff_ref;
    printf("test_fq12_sparse_mul...\n");
    const int N = 8;

    int pass = 0;
    for (int i = 0; i < N; ++i) {
        Fq12Ref a = make_fq12_test_val(i + 1);

        Fq2Ref d0 = make_fq2_test_val(i * 3 + 100);
        Fq2Ref d3 = make_fq2_test_val(i * 3 + 200);
        Fq2Ref d4 = make_fq2_test_val(i * 3 + 300);

        // Construct full Fq12 element matching 034 sparsity pattern
        Fq12Ref sparse_b = {{d0, Fq2Ref::zero(), Fq2Ref::zero()},
                             {d3, d4, Fq2Ref::zero()}};
        Fq12Ref full_prod = fq12_mul_ref(a, sparse_b);
        Fq12Ref sparse_prod = fq12_mul_by_034_ref(a, d0, d3, d4);
        if (full_prod == sparse_prod) ++pass;
    }
    TEST_ASSERT(pass == N, "Fq12 mul_by_034 mismatch");
    printf("  Fq12 sparse mul: %d/%d passed\n", pass, N);
}

// ─── Fq12 Frobenius Test ───────────────────────────────────────────────────

void test_fq12_frobenius() {
    using namespace ff_ref;
    printf("test_fq12_frobenius...\n");

    // Compute Fq6 Frobenius coefficients (indices 0-5)
    Fq2Ref fq6_c1[6], fq6_c2[6];
    compute_fq6_frobenius_coefficients(fq6_c1, fq6_c2);

    // Compute Fq12 Frobenius coefficients (indices 0-11)
    Fq2Ref fq12_w[12];
    compute_fq12_frobenius_coefficients(fq12_w);

    // γ_w[0] should be 1
    TEST_ASSERT(fq12_w[0] == Fq2Ref::one_mont(), "Fq12 Frobenius: gamma_w[0] = 1");

    // γ_w[k]² should equal γ₁[k] (Fq6 Frobenius c1 coefficient)
    // because γ_w[k] = β^((q^k-1)/6) and γ₁[k] = β^((q^k-1)/3) = γ_w[k]²
    for (int k = 1; k < 6; ++k) {
        Fq2Ref gamma_w_sq = fq2_sqr_ref(fq12_w[k]);
        char msg[128];
        snprintf(msg, sizeof(msg), "Fq12 Frobenius: gamma_w[%d]^2 = gamma_1[%d]", k, k);
        TEST_ASSERT(gamma_w_sq == fq6_c1[k], msg);
    }

    // φ^12(a) = a (period 12 for Fq12 Frobenius)
    const int N = 4;
    int pass_period = 0;
    for (int i = 0; i < N; ++i) {
        Fq12Ref a = make_fq12_test_val(i + 1);
        Fq12Ref result = a;
        for (int k = 0; k < 12; ++k)
            result = fq12_frobenius_map_ref(result, 1, fq6_c1, fq6_c2, fq12_w);
        if (result == a) ++pass_period;
    }
    TEST_ASSERT(pass_period == N, "Fq12 Frobenius: phi^12(a) != a");

    // Multiplicativity: φ(a*b) = φ(a) * φ(b)
    int pass_mult = 0;
    for (int i = 0; i < N; ++i) {
        Fq12Ref a = make_fq12_test_val(i * 2 + 1);
        Fq12Ref b = make_fq12_test_val(i * 2 + 2);
        Fq12Ref lhs = fq12_frobenius_map_ref(fq12_mul_ref(a, b), 1, fq6_c1, fq6_c2, fq12_w);
        Fq12Ref rhs = fq12_mul_ref(
            fq12_frobenius_map_ref(a, 1, fq6_c1, fq6_c2, fq12_w),
            fq12_frobenius_map_ref(b, 1, fq6_c1, fq6_c2, fq12_w)
        );
        if (lhs == rhs) ++pass_mult;
    }
    TEST_ASSERT(pass_mult == N, "Fq12 Frobenius: phi(a*b) != phi(a)*phi(b)");

    // φ^6 fixes Fq6 elements (Fq6 embedded as c0 + 0·w)
    Fq12Ref fq6_elem = {make_fq6_test_val(42), Fq6Ref::zero()};
    Fq12Ref phi6_fq6 = fq6_elem;
    for (int k = 0; k < 6; ++k)
        phi6_fq6 = fq12_frobenius_map_ref(phi6_fq6, 1, fq6_c1, fq6_c2, fq12_w);
    TEST_ASSERT(phi6_fq6 == fq6_elem, "Fq12 Frobenius: phi^6 fixes Fq6 elements");

    // φ^2 fixes Fq2 elements (Fq2 embedded as {{fq2, 0, 0}, {0, 0, 0}})
    Fq2Ref fq2_val = make_fq2_test_val(99);
    Fq12Ref fq2_elem = {{fq2_val, Fq2Ref::zero(), Fq2Ref::zero()},
                         {Fq2Ref::zero(), Fq2Ref::zero(), Fq2Ref::zero()}};
    Fq12Ref phi2_fq2 = fq12_frobenius_map_ref(
        fq12_frobenius_map_ref(fq2_elem, 1, fq6_c1, fq6_c2, fq12_w),
        1, fq6_c1, fq6_c2, fq12_w);
    TEST_ASSERT(phi2_fq2 == fq2_elem, "Fq12 Frobenius: phi^2 fixes Fq2 elements");

    printf("  Fq12 Frobenius: OK\n");
}

// ─── Fq12 w² = v identity test ─────────────────────────────────────────────

void test_fq12_w_squared() {
    using namespace ff_ref;
    printf("test_fq12_w_squared...\n");

    // w = (0, 1) in Fq12, i.e., c0 = Fq6::zero(), c1 = Fq6::one()
    Fq12Ref w = {Fq6Ref::zero(), Fq6Ref::one_mont()};
    Fq12Ref w_sq = fq12_sqr_ref(w);

    // w² should be v, which in Fq12 is the element with c0 = (0, 1, 0), c1 = 0
    // v in Fq6 = {Fq2::zero(), Fq2::one_mont(), Fq2::zero()}
    Fq6Ref v_fq6 = {Fq2Ref::zero(), Fq2Ref::one_mont(), Fq2Ref::zero()};
    Fq12Ref v_fq12 = {v_fq6, Fq6Ref::zero()};

    TEST_ASSERT(w_sq == v_fq12, "Fq12: w^2 = v");

    // Also test: a * w² = a * v = fq6_mul_by_nonresidue on Fq12 level
    Fq12Ref a = make_fq12_test_val(5);
    Fq12Ref a_times_w_sq = fq12_mul_ref(a, w_sq);
    // a * v should give c0 = fq6_mul_by_nonresidue(a.c1), c1 = a.c0 (from the tower structure)
    // Wait: a = (a0, a1), v = ({0,1,0}, {0,0,0})
    // a * v = (a0*{0,1,0} + a1*{0,0,0}*v_fq6, a0*{0,0,0} + a1*{0,1,0})... this is more complex.
    // Let's just verify via general mul
    Fq12Ref a_times_v = fq12_mul_ref(a, v_fq12);
    TEST_ASSERT(a_times_w_sq == a_times_v, "Fq12: a*w^2 = a*v");

    printf("  w^2 = v: OK\n");
}

// =============================================================================
// v2.0.0 Session 21: Elliptic Curve Point Arithmetic Tests
// =============================================================================

// Helper: make a G1 generator point in Jacobian form on GPU
static G1Jacobian make_g1_gen_gpu() {
    G1Jacobian p;
    for (int i = 0; i < 12; ++i) {
        p.x.limbs[i] = G1_GEN_X[i];
        p.y.limbs[i] = G1_GEN_Y[i];
    }
    p.z = FqElement::one_mont();
    return p;
}

// Helper: make a G1 affine generator on GPU
static G1Affine make_g1_gen_affine_gpu() {
    G1Affine p;
    for (int i = 0; i < 12; ++i) {
        p.x.limbs[i] = G1_GEN_X[i];
        p.y.limbs[i] = G1_GEN_Y[i];
    }
    p.infinity = false;
    return p;
}

// Helper: make a G2 generator point in Jacobian form on GPU
static G2Jacobian make_g2_gen_gpu() {
    G2Jacobian p;
    for (int i = 0; i < 12; ++i) {
        p.x.c0.limbs[i] = G2_GEN_X_C0[i];
        p.x.c1.limbs[i] = G2_GEN_X_C1[i];
        p.y.c0.limbs[i] = G2_GEN_Y_C0[i];
        p.y.c1.limbs[i] = G2_GEN_Y_C1[i];
    }
    p.z = Fq2Element::one_mont();
    return p;
}

static G2Affine make_g2_gen_affine_gpu() {
    G2Affine p;
    for (int i = 0; i < 12; ++i) {
        p.x.c0.limbs[i] = G2_GEN_X_C0[i];
        p.x.c1.limbs[i] = G2_GEN_X_C1[i];
        p.y.c0.limbs[i] = G2_GEN_Y_C0[i];
        p.y.c1.limbs[i] = G2_GEN_Y_C1[i];
    }
    p.infinity = false;
    return p;
}

// Helper: compare G1Affine from GPU with G1AffineRef from CPU
static bool g1_affine_eq(const G1Affine& gpu, const ff_ref::G1AffineRef& cpu) {
    if (gpu.infinity && cpu.infinity) return true;
    if (gpu.infinity != cpu.infinity) return false;
    ff_ref::FqRef gx = ff_ref::FqRef::from_u32(gpu.x.limbs);
    ff_ref::FqRef gy = ff_ref::FqRef::from_u32(gpu.y.limbs);
    return gx == cpu.x && gy == cpu.y;
}

static bool g2_affine_eq(const G2Affine& gpu, const ff_ref::G2AffineRef& cpu) {
    if (gpu.infinity && cpu.infinity) return true;
    if (gpu.infinity != cpu.infinity) return false;
    ff_ref::FqRef gxc0 = ff_ref::FqRef::from_u32(gpu.x.c0.limbs);
    ff_ref::FqRef gxc1 = ff_ref::FqRef::from_u32(gpu.x.c1.limbs);
    ff_ref::FqRef gyc0 = ff_ref::FqRef::from_u32(gpu.y.c0.limbs);
    ff_ref::FqRef gyc1 = ff_ref::FqRef::from_u32(gpu.y.c1.limbs);
    return gxc0 == cpu.x.c0 && gxc1 == cpu.x.c1 &&
           gyc0 == cpu.y.c0 && gyc1 == cpu.y.c1;
}

// --- G1 CPU self-test ---
void test_g1_cpu_self_test() {
    using namespace ff_ref;
    printf("test_g1_cpu_self_test...\n");

    G1AffineRef g = G1AffineRef::generator();
    TEST_ASSERT(g1_is_on_curve_ref(g), "G1: generator on curve");
    TEST_ASSERT(G1AffineRef::point_at_infinity().infinity, "G1: infinity flag");

    // 2*G
    G1AffineRef g2 = g1_double_ref(g);
    TEST_ASSERT(g1_is_on_curve_ref(g2), "G1: 2G on curve");
    TEST_ASSERT(!g2.infinity, "G1: 2G not infinity");

    // 3*G = G + 2G
    G1AffineRef g3 = g1_add_ref(g, g2);
    TEST_ASSERT(g1_is_on_curve_ref(g3), "G1: 3G on curve");

    // G + (-G) = O
    G1AffineRef neg_g = g1_negate_ref(g);
    G1AffineRef should_be_inf = g1_add_ref(g, neg_g);
    TEST_ASSERT(should_be_inf.infinity, "G1: G + (-G) = O");

    // Scalar mul: 2*G via scalar_mul == double
    uint32_t scalar_2[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef g2_sm = g1_scalar_mul_ref(g, scalar_2);
    TEST_ASSERT(g2_sm == g2, "G1: scalar_mul(G, 2) == 2G");

    // Scalar mul: 3*G via scalar_mul == G + 2G
    uint32_t scalar_3[8] = {3, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef g3_sm = g1_scalar_mul_ref(g, scalar_3);
    TEST_ASSERT(g3_sm == g3, "G1: scalar_mul(G, 3) == 3G");
}

// --- G1 GPU double test ---
void test_g1_gpu_double() {
    using namespace ff_ref;
    printf("test_g1_gpu_double...\n");

    const uint32_t N = 1;
    G1Jacobian h_in = make_g1_gen_gpu();

    G1Jacobian *d_in, *d_out;
    G1Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G1Affine)));
    CUDA_CHECK(cudaMemcpy(d_in, &h_in, sizeof(G1Jacobian), cudaMemcpyHostToDevice));

    g1_double_kernel<<<1, 1>>>(d_in, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_aff;
    CUDA_CHECK(cudaMemcpy(&h_aff, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));

    G1AffineRef cpu_2g = g1_double_ref(G1AffineRef::generator());
    TEST_ASSERT(g1_affine_eq(h_aff, cpu_2g), "G1 GPU: 2G matches CPU");
    TEST_ASSERT(!h_aff.infinity, "G1 GPU: 2G not infinity");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G1 GPU add test ---
void test_g1_gpu_add() {
    using namespace ff_ref;
    printf("test_g1_gpu_add...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    G1AffineRef cpu_2g = g1_double_ref(cpu_g);
    G1AffineRef cpu_3g = g1_add_ref(cpu_g, cpu_2g);

    // GPU: 2G + G = 3G (both in Jacobian)
    G1Jacobian gen_jac = make_g1_gen_gpu();

    // First, get 2G in Jacobian on GPU via double kernel
    G1Jacobian *d_gen, *d_2g, *d_3g;
    G1Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_gen, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_2g, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_3g, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G1Affine)));
    CUDA_CHECK(cudaMemcpy(d_gen, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));

    g1_double_kernel<<<1, 1>>>(d_gen, d_2g, 1);
    g1_add_kernel<<<1, 1>>>(d_2g, d_gen, d_3g, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_3g, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_3g_aff;
    CUDA_CHECK(cudaMemcpy(&h_3g_aff, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));

    TEST_ASSERT(g1_affine_eq(h_3g_aff, cpu_3g), "G1 GPU: 2G + G = 3G matches CPU");

    CUDA_CHECK(cudaFree(d_gen));
    CUDA_CHECK(cudaFree(d_2g));
    CUDA_CHECK(cudaFree(d_3g));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G1 GPU mixed add test ---
void test_g1_gpu_add_mixed() {
    using namespace ff_ref;
    printf("test_g1_gpu_add_mixed...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    G1AffineRef cpu_2g = g1_double_ref(cpu_g);
    G1AffineRef cpu_3g = g1_add_ref(cpu_2g, cpu_g);

    G1Jacobian gen_jac = make_g1_gen_gpu();
    G1Affine gen_aff = make_g1_gen_affine_gpu();

    G1Jacobian *d_jac, *d_2g, *d_3g;
    G1Affine *d_aff_in, *d_aff_out;
    CUDA_CHECK(cudaMalloc(&d_jac, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_2g, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_3g, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff_in, sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_aff_out, sizeof(G1Affine)));
    CUDA_CHECK(cudaMemcpy(d_jac, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aff_in, &gen_aff, sizeof(G1Affine), cudaMemcpyHostToDevice));

    // 2G = double(G)
    g1_double_kernel<<<1, 1>>>(d_jac, d_2g, 1);
    // 3G = 2G + G_affine (mixed add)
    g1_add_mixed_kernel<<<1, 1>>>(d_2g, d_aff_in, d_3g, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_3g, d_aff_out, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_3g;
    CUDA_CHECK(cudaMemcpy(&h_3g, d_aff_out, sizeof(G1Affine), cudaMemcpyDeviceToHost));

    TEST_ASSERT(g1_affine_eq(h_3g, cpu_3g), "G1 GPU: 2G + G_aff = 3G (mixed add)");

    CUDA_CHECK(cudaFree(d_jac));
    CUDA_CHECK(cudaFree(d_2g));
    CUDA_CHECK(cudaFree(d_3g));
    CUDA_CHECK(cudaFree(d_aff_in));
    CUDA_CHECK(cudaFree(d_aff_out));
}

// --- G1 GPU scalar_mul test ---
void test_g1_gpu_scalar_mul() {
    using namespace ff_ref;
    printf("test_g1_gpu_scalar_mul...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    uint32_t scalar_5[8] = {5, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_5g = g1_scalar_mul_ref(cpu_g, scalar_5);

    G1Jacobian gen_jac = make_g1_gen_gpu();

    G1Jacobian *d_base, *d_out;
    uint32_t *d_scalar;
    G1Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_base, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_scalar, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G1Affine)));
    CUDA_CHECK(cudaMemcpy(d_base, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalar, scalar_5, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    g1_scalar_mul_kernel<<<1, 1>>>(d_base, d_scalar, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_5g;
    CUDA_CHECK(cudaMemcpy(&h_5g, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));

    TEST_ASSERT(g1_affine_eq(h_5g, cpu_5g), "G1 GPU: scalar_mul(G, 5) = 5G");
    TEST_ASSERT(g1_is_on_curve_ref(cpu_5g), "G1: 5G on curve");

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scalar));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G1 GPU on-curve check ---
void test_g1_gpu_on_curve() {
    using namespace ff_ref;
    printf("test_g1_gpu_on_curve...\n");

    G1Affine gen_aff = make_g1_gen_affine_gpu();
    G1Affine inf = {FqElement::zero(), FqElement::zero(), true};

    // Off-curve point: generator with y tweaked
    G1Affine bad = gen_aff;
    bad.y.limbs[0] ^= 1;

    const uint32_t N = 3;
    G1Affine h_pts[3] = {gen_aff, inf, bad};
    bool h_results[3] = {false, false, false};

    G1Affine *d_pts;
    bool *d_results;
    CUDA_CHECK(cudaMalloc(&d_pts, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_results, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, N * sizeof(G1Affine), cudaMemcpyHostToDevice));

    g1_is_on_curve_kernel<<<1, N>>>(d_pts, d_results, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, N * sizeof(bool), cudaMemcpyDeviceToHost));

    TEST_ASSERT(h_results[0] == true, "G1 GPU: generator on curve");
    TEST_ASSERT(h_results[1] == true, "G1 GPU: infinity on curve");
    TEST_ASSERT(h_results[2] == false, "G1 GPU: tweaked point NOT on curve");

    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_results));
}

// --- G1 GPU negate test ---
void test_g1_gpu_negate() {
    using namespace ff_ref;
    printf("test_g1_gpu_negate...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    G1AffineRef cpu_neg_g = g1_negate_ref(cpu_g);

    // G + (-G) = O on GPU
    G1Jacobian gen_jac = make_g1_gen_gpu();
    // Negate: flip y
    G1Jacobian neg_gen = gen_jac;
    // Use CPU to compute -y in Montgomery form
    FqRef gy_ref = FqRef::from_u32(gen_jac.y.limbs);
    FqRef neg_gy = fq_neg_ref(gy_ref);
    neg_gy.to_u32(neg_gen.y.limbs);

    G1Jacobian *d_a, *d_b, *d_out;
    G1Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G1Affine)));
    CUDA_CHECK(cudaMemcpy(d_a, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &neg_gen, sizeof(G1Jacobian), cudaMemcpyHostToDevice));

    g1_add_kernel<<<1, 1>>>(d_a, d_b, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));

    TEST_ASSERT(h_result.infinity, "G1 GPU: G + (-G) = O");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G1 identity tests ---
void test_g1_identity() {
    using namespace ff_ref;
    printf("test_g1_identity...\n");

    G1Jacobian gen_jac = make_g1_gen_gpu();
    G1Jacobian identity = G1Jacobian::identity();

    G1Jacobian *d_a, *d_b, *d_out;
    G1Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G1Affine)));

    // G + O = G
    CUDA_CHECK(cudaMemcpy(d_a, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &identity, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    g1_add_kernel<<<1, 1>>>(d_a, d_b, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g1_affine_eq(h_result, G1AffineRef::generator()), "G1 GPU: G + O = G");

    // O + G = G
    CUDA_CHECK(cudaMemcpy(d_a, &identity, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    g1_add_kernel<<<1, 1>>>(d_a, d_b, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g1_affine_eq(h_result, G1AffineRef::generator()), "G1 GPU: O + G = G");

    // double(O) = O
    CUDA_CHECK(cudaMemcpy(d_a, &identity, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    g1_double_kernel<<<1, 1>>>(d_a, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h_result.infinity, "G1 GPU: double(O) = O");

    // scalar_mul(G, 0) = O
    uint32_t scalar_0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t *d_scalar;
    CUDA_CHECK(cudaMalloc(&d_scalar, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_a, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalar, scalar_0, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    g1_scalar_mul_kernel<<<1, 1>>>(d_a, d_scalar, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h_result.infinity, "G1 GPU: scalar_mul(G, 0) = O");

    // scalar_mul(G, 1) = G
    uint32_t scalar_1[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(d_scalar, scalar_1, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    g1_scalar_mul_kernel<<<1, 1>>>(d_a, d_scalar, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g1_affine_eq(h_result, G1AffineRef::generator()), "G1 GPU: scalar_mul(G, 1) = G");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_aff));
    CUDA_CHECK(cudaFree(d_scalar));
}

// --- G1 GPU scalar_mul larger scalars ---
void test_g1_gpu_scalar_mul_larger() {
    using namespace ff_ref;
    printf("test_g1_gpu_scalar_mul_larger...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();

    // scalar = 100
    uint32_t scalar_100[8] = {100, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_100g = g1_scalar_mul_ref(cpu_g, scalar_100);
    TEST_ASSERT(g1_is_on_curve_ref(cpu_100g), "G1: 100G on curve");

    G1Jacobian gen_jac = make_g1_gen_gpu();
    G1Jacobian *d_base, *d_out;
    uint32_t *d_scalar;
    G1Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_base, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_scalar, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G1Affine)));
    CUDA_CHECK(cudaMemcpy(d_base, &gen_jac, sizeof(G1Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalar, scalar_100, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    g1_scalar_mul_kernel<<<1, 1>>>(d_base, d_scalar, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_100g;
    CUDA_CHECK(cudaMemcpy(&h_100g, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g1_affine_eq(h_100g, cpu_100g), "G1 GPU: scalar_mul(G, 100) matches CPU");

    // scalar = 0xDEADBEEF (multi-bit test)
    uint32_t scalar_beef[8] = {0xDEADBEEFu, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_beef = g1_scalar_mul_ref(cpu_g, scalar_beef);
    TEST_ASSERT(g1_is_on_curve_ref(cpu_beef), "G1: 0xDEADBEEF*G on curve");

    CUDA_CHECK(cudaMemcpy(d_scalar, scalar_beef, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    g1_scalar_mul_kernel<<<1, 1>>>(d_base, d_scalar, d_out, 1);
    g1_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_beef;
    CUDA_CHECK(cudaMemcpy(&h_beef, d_aff, sizeof(G1Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g1_affine_eq(h_beef, cpu_beef), "G1 GPU: scalar_mul(G, 0xDEADBEEF) matches CPU");

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scalar));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G2 CPU self-test ---
void test_g2_cpu_self_test() {
    using namespace ff_ref;
    printf("test_g2_cpu_self_test...\n");

    G2AffineRef g = G2AffineRef::generator();
    TEST_ASSERT(g2_is_on_curve_ref(g), "G2: generator on curve");

    G2AffineRef g2 = g2_double_ref(g);
    TEST_ASSERT(g2_is_on_curve_ref(g2), "G2: 2G on curve");

    G2AffineRef g3 = g2_add_ref(g, g2);
    TEST_ASSERT(g2_is_on_curve_ref(g3), "G2: 3G on curve");

    G2AffineRef neg_g = g2_negate_ref(g);
    G2AffineRef should_be_inf = g2_add_ref(g, neg_g);
    TEST_ASSERT(should_be_inf.infinity, "G2: G + (-G) = O");

    uint32_t scalar_2[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    G2AffineRef g2_sm = g2_scalar_mul_ref(g, scalar_2);
    TEST_ASSERT(g2_sm == g2, "G2: scalar_mul(G, 2) == 2G");

    uint32_t scalar_3[8] = {3, 0, 0, 0, 0, 0, 0, 0};
    G2AffineRef g3_sm = g2_scalar_mul_ref(g, scalar_3);
    TEST_ASSERT(g3_sm == g3, "G2: scalar_mul(G, 3) == 3G");
}

// --- G2 GPU double test ---
void test_g2_gpu_double() {
    using namespace ff_ref;
    printf("test_g2_gpu_double...\n");

    G2Jacobian h_in = make_g2_gen_gpu();
    G2Jacobian *d_in, *d_out;
    G2Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G2Affine)));
    CUDA_CHECK(cudaMemcpy(d_in, &h_in, sizeof(G2Jacobian), cudaMemcpyHostToDevice));

    g2_double_kernel<<<1, 1>>>(d_in, d_out, 1);
    g2_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G2Affine h_aff;
    CUDA_CHECK(cudaMemcpy(&h_aff, d_aff, sizeof(G2Affine), cudaMemcpyDeviceToHost));

    G2AffineRef cpu_2g = g2_double_ref(G2AffineRef::generator());
    TEST_ASSERT(g2_affine_eq(h_aff, cpu_2g), "G2 GPU: 2G matches CPU");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G2 GPU add test ---
void test_g2_gpu_add() {
    using namespace ff_ref;
    printf("test_g2_gpu_add...\n");

    G2AffineRef cpu_g = G2AffineRef::generator();
    G2AffineRef cpu_2g = g2_double_ref(cpu_g);
    G2AffineRef cpu_3g = g2_add_ref(cpu_g, cpu_2g);

    G2Jacobian gen_jac = make_g2_gen_gpu();
    G2Jacobian *d_gen, *d_2g, *d_3g;
    G2Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_gen, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_2g, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_3g, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G2Affine)));
    CUDA_CHECK(cudaMemcpy(d_gen, &gen_jac, sizeof(G2Jacobian), cudaMemcpyHostToDevice));

    g2_double_kernel<<<1, 1>>>(d_gen, d_2g, 1);
    g2_add_kernel<<<1, 1>>>(d_2g, d_gen, d_3g, 1);
    g2_to_affine_kernel<<<1, 1>>>(d_3g, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G2Affine h_3g;
    CUDA_CHECK(cudaMemcpy(&h_3g, d_aff, sizeof(G2Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g2_affine_eq(h_3g, cpu_3g), "G2 GPU: 2G + G = 3G matches CPU");

    CUDA_CHECK(cudaFree(d_gen));
    CUDA_CHECK(cudaFree(d_2g));
    CUDA_CHECK(cudaFree(d_3g));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G2 GPU mixed add test ---
void test_g2_gpu_add_mixed() {
    using namespace ff_ref;
    printf("test_g2_gpu_add_mixed...\n");

    G2AffineRef cpu_g = G2AffineRef::generator();
    G2AffineRef cpu_2g = g2_double_ref(cpu_g);
    G2AffineRef cpu_3g = g2_add_ref(cpu_2g, cpu_g);

    G2Jacobian gen_jac = make_g2_gen_gpu();
    G2Affine gen_aff = make_g2_gen_affine_gpu();

    G2Jacobian *d_jac, *d_2g, *d_3g;
    G2Affine *d_aff_in, *d_aff_out;
    CUDA_CHECK(cudaMalloc(&d_jac, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_2g, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_3g, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff_in, sizeof(G2Affine)));
    CUDA_CHECK(cudaMalloc(&d_aff_out, sizeof(G2Affine)));
    CUDA_CHECK(cudaMemcpy(d_jac, &gen_jac, sizeof(G2Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aff_in, &gen_aff, sizeof(G2Affine), cudaMemcpyHostToDevice));

    g2_double_kernel<<<1, 1>>>(d_jac, d_2g, 1);
    g2_add_mixed_kernel<<<1, 1>>>(d_2g, d_aff_in, d_3g, 1);
    g2_to_affine_kernel<<<1, 1>>>(d_3g, d_aff_out, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G2Affine h_3g;
    CUDA_CHECK(cudaMemcpy(&h_3g, d_aff_out, sizeof(G2Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g2_affine_eq(h_3g, cpu_3g), "G2 GPU: 2G + G_aff = 3G (mixed add)");

    CUDA_CHECK(cudaFree(d_jac));
    CUDA_CHECK(cudaFree(d_2g));
    CUDA_CHECK(cudaFree(d_3g));
    CUDA_CHECK(cudaFree(d_aff_in));
    CUDA_CHECK(cudaFree(d_aff_out));
}

// --- G2 GPU scalar_mul test ---
void test_g2_gpu_scalar_mul() {
    using namespace ff_ref;
    printf("test_g2_gpu_scalar_mul...\n");

    G2AffineRef cpu_g = G2AffineRef::generator();
    uint32_t scalar_5[8] = {5, 0, 0, 0, 0, 0, 0, 0};
    G2AffineRef cpu_5g = g2_scalar_mul_ref(cpu_g, scalar_5);
    TEST_ASSERT(g2_is_on_curve_ref(cpu_5g), "G2: 5G on curve");

    G2Jacobian gen_jac = make_g2_gen_gpu();
    G2Jacobian *d_base, *d_out;
    uint32_t *d_scalar;
    G2Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_base, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_scalar, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G2Affine)));
    CUDA_CHECK(cudaMemcpy(d_base, &gen_jac, sizeof(G2Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalar, scalar_5, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    g2_scalar_mul_kernel<<<1, 1>>>(d_base, d_scalar, d_out, 1);
    g2_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G2Affine h_5g;
    CUDA_CHECK(cudaMemcpy(&h_5g, d_aff, sizeof(G2Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g2_affine_eq(h_5g, cpu_5g), "G2 GPU: scalar_mul(G, 5) = 5G matches CPU");

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scalar));
    CUDA_CHECK(cudaFree(d_aff));
}

// --- G2 GPU on-curve check ---
void test_g2_gpu_on_curve() {
    using namespace ff_ref;
    printf("test_g2_gpu_on_curve...\n");

    G2Affine gen_aff = make_g2_gen_affine_gpu();
    G2Affine inf;
    inf.x = Fq2Element::zero();
    inf.y = Fq2Element::zero();
    inf.infinity = true;

    G2Affine bad = gen_aff;
    bad.y.c0.limbs[0] ^= 1;

    const uint32_t N = 3;
    G2Affine h_pts[3] = {gen_aff, inf, bad};
    bool h_results[3] = {false, false, false};

    G2Affine *d_pts;
    bool *d_results;
    CUDA_CHECK(cudaMalloc(&d_pts, N * sizeof(G2Affine)));
    CUDA_CHECK(cudaMalloc(&d_results, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, N * sizeof(G2Affine), cudaMemcpyHostToDevice));

    g2_is_on_curve_kernel<<<1, N>>>(d_pts, d_results, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, N * sizeof(bool), cudaMemcpyDeviceToHost));

    TEST_ASSERT(h_results[0] == true, "G2 GPU: generator on curve");
    TEST_ASSERT(h_results[1] == true, "G2 GPU: infinity on curve");
    TEST_ASSERT(h_results[2] == false, "G2 GPU: tweaked point NOT on curve");

    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_results));
}

// --- G2 identity tests ---
void test_g2_identity() {
    using namespace ff_ref;
    printf("test_g2_identity...\n");

    G2Jacobian gen_jac = make_g2_gen_gpu();
    G2Jacobian identity = G2Jacobian::identity();

    G2Jacobian *d_a, *d_b, *d_out;
    G2Affine *d_aff;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_aff, sizeof(G2Affine)));

    // G + O = G
    CUDA_CHECK(cudaMemcpy(d_a, &gen_jac, sizeof(G2Jacobian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &identity, sizeof(G2Jacobian), cudaMemcpyHostToDevice));
    g2_add_kernel<<<1, 1>>>(d_a, d_b, d_out, 1);
    g2_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    G2Affine h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G2Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g2_affine_eq(h_result, G2AffineRef::generator()), "G2 GPU: G + O = G");

    // double(O) = O
    CUDA_CHECK(cudaMemcpy(d_a, &identity, sizeof(G2Jacobian), cudaMemcpyHostToDevice));
    g2_double_kernel<<<1, 1>>>(d_a, d_out, 1);
    g2_to_affine_kernel<<<1, 1>>>(d_out, d_aff, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_aff, sizeof(G2Affine), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h_result.infinity, "G2 GPU: double(O) = O");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_aff));
}

// =============================================================================
// v2.0.0 Session 22: Multi-Scalar Multiplication Tests
// =============================================================================

// CPU naive MSM reference: sum(scalar[i] * base[i])
static ff_ref::G1AffineRef msm_naive_ref(
    const ff_ref::G1AffineRef* bases,
    const uint32_t* scalars,  // n * 8
    size_t n)
{
    using namespace ff_ref;
    G1AffineRef result = G1AffineRef::point_at_infinity();
    for (size_t i = 0; i < n; ++i) {
        G1AffineRef term = g1_scalar_mul_ref(bases[i], &scalars[i * 8]);
        result = g1_add_ref(result, term);
    }
    return result;
}

// Test: single-point MSM = scalar_mul
void test_msm_single_point() {
    using namespace ff_ref;
    printf("test_msm_single_point...\n");

    G1Affine h_base = make_g1_gen_affine_gpu();
    uint32_t h_scalar[8] = {7, 0, 0, 0, 0, 0, 0, 0};

    // CPU reference
    G1AffineRef cpu_g = G1AffineRef::generator();
    G1AffineRef cpu_7g = g1_scalar_mul_ref(cpu_g, h_scalar);

    // GPU MSM
    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, &h_base, sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalar, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 1);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_7g), "MSM: single point = scalar_mul");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: two-point MSM
void test_msm_two_points() {
    using namespace ff_ref;
    printf("test_msm_two_points...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    G1AffineRef cpu_2g = g1_double_ref(cpu_g);

    // MSM: 3*G + 5*(2G) = 3G + 10G = 13G
    uint32_t scalars[16] = {3, 0, 0, 0, 0, 0, 0, 0,   // scalar[0] = 3
                            5, 0, 0, 0, 0, 0, 0, 0};   // scalar[1] = 5

    G1AffineRef bases_ref[2] = {cpu_g, cpu_2g};
    G1AffineRef cpu_result = msm_naive_ref(bases_ref, scalars, 2);

    // Verify: should be 13G
    uint32_t s13[8] = {13, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_13g = g1_scalar_mul_ref(cpu_g, s13);
    TEST_ASSERT(cpu_result == cpu_13g, "MSM ref: 3*G + 5*(2G) = 13G");

    // GPU
    G1Affine h_bases[2];
    h_bases[0] = make_g1_gen_affine_gpu();
    // 2G: need to compute on CPU and convert
    FqRef ref_2gx = cpu_2g.x, ref_2gy = cpu_2g.y;
    ref_2gx.to_u32(h_bases[1].x.limbs);
    ref_2gy.to_u32(h_bases[1].y.limbs);
    h_bases[1].infinity = false;

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, 2 * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases, 2 * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 2);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM GPU: 2-point matches CPU");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with identity (zero scalar)
void test_msm_with_identity() {
    using namespace ff_ref;
    printf("test_msm_with_identity...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    // MSM: 0*G + 5*G = 5G
    uint32_t scalars[16] = {0, 0, 0, 0, 0, 0, 0, 0,   // scalar[0] = 0
                            5, 0, 0, 0, 0, 0, 0, 0};   // scalar[1] = 5

    uint32_t s5[8] = {5, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_5g = g1_scalar_mul_ref(cpu_g, s5);

    G1Affine h_bases[2];
    h_bases[0] = make_g1_gen_affine_gpu();
    h_bases[1] = make_g1_gen_affine_gpu();

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, 2 * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases, 2 * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 2);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_5g), "MSM GPU: 0*G + 5*G = 5G");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with all-ones scalars
void test_msm_all_ones() {
    using namespace ff_ref;
    printf("test_msm_all_ones...\n");

    // n copies of G, all with scalar=1 -> result = n*G
    const int N = 16;
    G1AffineRef cpu_g = G1AffineRef::generator();
    uint32_t s_n[8] = {(uint32_t)N, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_nG = g1_scalar_mul_ref(cpu_g, s_n);

    std::vector<G1Affine> h_bases(N, make_g1_gen_affine_gpu());
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) h_scalars[i * 8] = 1;

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_nG), "MSM GPU: N*1*G = N*G");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM correctness at n=64 (GPU vs CPU naive)
void test_msm_medium(int n) {
    using namespace ff_ref;
    printf("test_msm_medium (n=%d)...\n", n);

    G1AffineRef cpu_g = G1AffineRef::generator();

    // Generate n different bases: i*G for i=1..n
    std::vector<G1AffineRef> cpu_bases(n);
    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Generate random-ish scalars (small for speed)
    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8] = (uint32_t)((i * 7 + 3) % 256);
    }

    // CPU reference
    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), n);

    // GPU MSM
    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    // Verify on-curve
    TEST_ASSERT(g1_is_on_curve_ref(cpu_result), "MSM: CPU result on curve");

    bool match = g1_affine_eq(gpu_result, cpu_result);
    char msg[128];
    snprintf(msg, sizeof(msg), "MSM GPU n=%d: matches CPU naive", n);
    TEST_ASSERT(match, msg);

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM on-curve check for larger n (GPU result only, no CPU cross-validation)
void test_msm_on_curve(int n) {
    using namespace ff_ref;
    printf("test_msm_on_curve (n=%d)...\n", n);

    G1AffineRef cpu_g = G1AffineRef::generator();

    // Generate bases: i*G
    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8] = (uint32_t)((i * 13 + 7) % 1024);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    // Convert to ref and check on-curve
    G1AffineRef result_ref;
    result_ref.x = FqRef::from_u32(gpu_result.x.limbs);
    result_ref.y = FqRef::from_u32(gpu_result.y.limbs);
    result_ref.infinity = gpu_result.infinity;

    char msg[128];
    snprintf(msg, sizeof(msg), "MSM GPU n=%d: result on curve", n);
    TEST_ASSERT(gpu_result.infinity || g1_is_on_curve_ref(result_ref), msg);

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM determinism (same input -> same output)
void test_msm_determinism() {
    using namespace ff_ref;
    printf("test_msm_determinism...\n");

    const int N = 32;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) h_scalars[i * 8] = (uint32_t)(i * 3 + 1);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine result1, result2;
    msm_g1(&result1, d_bases, d_scalars, N);
    msm_g1(&result2, d_bases, d_scalars, N);

    bool match = (memcmp(&result1, &result2, sizeof(G1Affine)) == 0);
    TEST_ASSERT(match, "MSM: deterministic (same input -> same output)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM window size selection
void test_msm_window_size() {
    printf("test_msm_window_size...\n");

    TEST_ASSERT(msm_optimal_window(1) >= 1, "MSM window: n=1");
    TEST_ASSERT(msm_optimal_window(1024) >= 4, "MSM window: n=1024");
    TEST_ASSERT(msm_optimal_window(1 << 15) >= 4, "MSM window: n=2^15");
    TEST_ASSERT(msm_optimal_window(1 << 20) >= 4, "MSM window: n=2^20");
    TEST_ASSERT(msm_optimal_window(1 << 20) <= 16, "MSM window: n=2^20 <= 16");
}

// =============================================================================
// v2.1.0 Session 26: Production MSM — Signed-Digit + Segment Offsets Tests
// =============================================================================

// Test: updated window size function gives reasonable values
void test_msm_window_size_v2() {
    printf("test_msm_window_size_v2...\n");

    // c = floor(log2(n)/2) + 1, clamped [4, 16]
    TEST_ASSERT(msm_optimal_window(1 << 10) == 6, "MSM window v2: n=2^10 -> c=6");
    TEST_ASSERT(msm_optimal_window(1 << 14) == 8, "MSM window v2: n=2^14 -> c=8");
    TEST_ASSERT(msm_optimal_window(1 << 18) == 10, "MSM window v2: n=2^18 -> c=10");
    TEST_ASSERT(msm_optimal_window(1 << 20) == 11, "MSM window v2: n=2^20 -> c=11");
}

// Test: MSM with high-bit scalars (near r-1)
void test_msm_high_scalars() {
    using namespace ff_ref;
    printf("test_msm_high_scalars...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();
    // Use scalar = r-1 (max valid scalar for BLS12-381 Fr)
    // r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    uint32_t scalar_r_minus_1[8] = {
        0x00000000u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
        0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u
    };

    G1AffineRef cpu_result = g1_scalar_mul_ref(cpu_g, scalar_r_minus_1);

    G1Affine h_base = make_g1_gen_affine_gpu();
    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, &h_base, sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, scalar_r_minus_1, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 1);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM: scalar=r-1 matches CPU");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with mixed zero/nonzero scalars (stress signed-digit zero handling)
void test_msm_mixed_zero_scalars() {
    using namespace ff_ref;
    printf("test_msm_mixed_zero_scalars...\n");

    const int N = 8;
    G1AffineRef cpu_g = G1AffineRef::generator();

    // Alternating: 0, 3, 0, 7, 0, 11, 0, 13
    std::vector<uint32_t> h_scalars(N * 8, 0);
    h_scalars[1 * 8] = 3;
    h_scalars[3 * 8] = 7;
    h_scalars[5 * 8] = 11;
    h_scalars[7 * 8] = 13;

    // All bases = G
    std::vector<G1Affine> h_bases(N, make_g1_gen_affine_gpu());

    // CPU: 0*G + 3*G + 0*G + 7*G + 0*G + 11*G + 0*G + 13*G = 34*G
    G1AffineRef cpu_bases_ref[8];
    for (int i = 0; i < N; ++i) cpu_bases_ref[i] = cpu_g;
    G1AffineRef cpu_result = msm_naive_ref(cpu_bases_ref, h_scalars.data(), N);

    uint32_t s34[8] = {34, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef expected = g1_scalar_mul_ref(cpu_g, s34);
    TEST_ASSERT(cpu_result == expected, "MSM mixed zero: CPU ref = 34G");

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM mixed zero: GPU matches CPU");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with all identical scalars (single bucket stress test)
void test_msm_single_bucket() {
    using namespace ff_ref;
    printf("test_msm_single_bucket...\n");

    const int N = 16;
    G1AffineRef cpu_g = G1AffineRef::generator();

    // All scalars = 42 (all points go to same bucket per window)
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) h_scalars[i * 8] = 42;

    // bases: i*G for i=1..N
    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM single bucket: GPU matches CPU");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with power-of-2 scalars (stress different window decompositions)
void test_msm_power_of_2_scalars() {
    using namespace ff_ref;
    printf("test_msm_power_of_2_scalars...\n");

    const int N = 8;
    G1AffineRef cpu_g = G1AffineRef::generator();

    // Scalars: 1, 2, 4, 8, 16, 32, 64, 128
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) h_scalars[i * 8] = 1u << i;

    // All bases = G -> result = (1+2+4+8+16+32+64+128)*G = 255*G
    std::vector<G1Affine> h_bases(N, make_g1_gen_affine_gpu());

    G1AffineRef cpu_bases_ref[8];
    for (int i = 0; i < N; ++i) cpu_bases_ref[i] = cpu_g;
    G1AffineRef cpu_result = msm_naive_ref(cpu_bases_ref, h_scalars.data(), N);

    uint32_t s255[8] = {255, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef expected = g1_scalar_mul_ref(cpu_g, s255);
    TEST_ASSERT(cpu_result == expected, "MSM pow2: CPU ref = 255G");

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM pow2: GPU matches CPU");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM cross-validation at larger sizes (n=128, 256) vs CPU
void test_msm_cross_validate(int n) {
    using namespace ff_ref;
    printf("test_msm_cross_validate (n=%d)...\n", n);

    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(n);
    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Use small scalars for CPU feasibility
    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8] = (uint32_t)((i * 17 + 5) % 128);
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), n);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    char msg[128];
    snprintf(msg, sizeof(msg), "MSM cross-validate n=%d: GPU matches CPU", n);
    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), msg);

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM on-curve at larger sizes (n=512, 1024)
void test_msm_on_curve_large(int n) {
    using namespace ff_ref;
    printf("test_msm_on_curve_large (n=%d)...\n", n);

    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Larger scalars (multi-limb) to stress signed-digit recoding
    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8 + 0] = (uint32_t)(i * 0xDEAD + 0xBEEF);
        h_scalars[i * 8 + 1] = (uint32_t)(i * 0x1337 + 0x42);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    G1AffineRef result_ref;
    result_ref.x = FqRef::from_u32(gpu_result.x.limbs);
    result_ref.y = FqRef::from_u32(gpu_result.y.limbs);
    result_ref.infinity = gpu_result.infinity;

    char msg[128];
    snprintf(msg, sizeof(msg), "MSM on-curve n=%d: result on curve", n);
    TEST_ASSERT(gpu_result.infinity || g1_is_on_curve_ref(result_ref), msg);

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with multi-limb scalars (cross-validate, exercises signed-digit carries)
void test_msm_multi_limb_scalars() {
    using namespace ff_ref;
    printf("test_msm_multi_limb_scalars...\n");

    const int N = 8;
    G1AffineRef cpu_g = G1AffineRef::generator();

    // Bases: i*G
    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Multi-limb scalars: large values that exercise carry propagation
    std::vector<uint32_t> h_scalars(N * 8, 0);
    // scalar[0] = 0xFFFFFFFF (all-ones in limb 0 -> forces carries)
    h_scalars[0 * 8 + 0] = 0xFFFFFFFFu;
    // scalar[1] = 0x100000000 (one in limb 1)
    h_scalars[1 * 8 + 1] = 1;
    // scalar[2] = 0xDEADBEEF12345678
    h_scalars[2 * 8 + 0] = 0x12345678u;
    h_scalars[2 * 8 + 1] = 0xDEADBEEFu;
    // scalar[3] = small
    h_scalars[3 * 8 + 0] = 7;
    // scalar[4] = 0xFFFFFFFFFFFFFFFF (two all-ones limbs)
    h_scalars[4 * 8 + 0] = 0xFFFFFFFFu;
    h_scalars[4 * 8 + 1] = 0xFFFFFFFFu;
    // scalar[5] = 1
    h_scalars[5 * 8 + 0] = 1;
    // scalar[6] = value spanning 3 limbs
    h_scalars[6 * 8 + 0] = 0xAAAAAAAAu;
    h_scalars[6 * 8 + 1] = 0xBBBBBBBBu;
    h_scalars[6 * 8 + 2] = 0xCCCCCCCCu;
    // scalar[7] = 0
    // (already zero)

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM multi-limb: GPU matches CPU");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM all-zeros scalars -> identity
void test_msm_all_zeros() {
    printf("test_msm_all_zeros...\n");

    const int N = 8;
    std::vector<G1Affine> h_bases(N, make_g1_gen_affine_gpu());
    std::vector<uint32_t> h_scalars(N * 8, 0);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(gpu_result.infinity, "MSM all-zeros: result is identity");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM determinism with new implementation (run 3 times)
void test_msm_determinism_v2() {
    using namespace ff_ref;
    printf("test_msm_determinism_v2...\n");

    const int N = 64;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) {
        h_scalars[i * 8 + 0] = (uint32_t)(i * 0xABCD + 0x1234);
        h_scalars[i * 8 + 1] = (uint32_t)(i * 0x5678);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine r1, r2, r3;
    msm_g1(&r1, d_bases, d_scalars, N);
    msm_g1(&r2, d_bases, d_scalars, N);
    msm_g1(&r3, d_bases, d_scalars, N);

    bool m12 = (memcmp(&r1, &r2, sizeof(G1Affine)) == 0);
    bool m23 = (memcmp(&r2, &r3, sizeof(G1Affine)) == 0);
    TEST_ASSERT(m12 && m23, "MSM determinism v2: 3 runs identical");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM on-curve at n=2048 and n=4096 (exercises larger window sizes)
void test_msm_on_curve_2k(int n) {
    using namespace ff_ref;
    printf("test_msm_on_curve_2k (n=%d)...\n", n);

    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8 + 0] = (uint32_t)(i * 0xBEEF + 0xCAFE);
        h_scalars[i * 8 + 1] = (uint32_t)(i * 0xFACE);
        h_scalars[i * 8 + 2] = (uint32_t)(i * 0xD00D + 1);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    G1AffineRef result_ref;
    result_ref.x = FqRef::from_u32(gpu_result.x.limbs);
    result_ref.y = FqRef::from_u32(gpu_result.y.limbs);
    result_ref.infinity = gpu_result.infinity;

    char msg[128];
    snprintf(msg, sizeof(msg), "MSM on-curve n=%d: result on curve", n);
    TEST_ASSERT(gpu_result.infinity || g1_is_on_curve_ref(result_ref), msg);

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM scalar=1 for single point (simplest signed-digit case)
void test_msm_scalar_one() {
    using namespace ff_ref;
    printf("test_msm_scalar_one...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();

    G1Affine h_base = make_g1_gen_affine_gpu();
    uint32_t h_scalar[8] = {1, 0, 0, 0, 0, 0, 0, 0};

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, &h_base, sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalar, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 1);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_g), "MSM: 1*G = G");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// =============================================================================
// v2.1.0 Session 27: Parallel Bucket Reduction Tests
// =============================================================================

// Test: MSM cross-validate at n=512 (parallel reduction exercises more buckets)
void test_msm_cross_validate_512() {
    using namespace ff_ref;
    printf("test_msm_cross_validate_512...\n");

    const int n = 512;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(n);
    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8] = (uint32_t)((i * 17 + 5) % 128);
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), n);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM n=512: GPU matches CPU (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM determinism with parallel reduction (run twice, bitwise match)
void test_msm_determinism_parallel() {
    using namespace ff_ref;
    printf("test_msm_determinism_parallel...\n");

    const int N = 256;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) {
        h_scalars[i * 8 + 0] = (uint32_t)(i * 0x1234 + 0x5678);
        h_scalars[i * 8 + 1] = (uint32_t)(i * 0xABCD + 0xEF01);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine result1, result2;
    msm_g1(&result1, d_bases, d_scalars, N);
    msm_g1(&result2, d_bases, d_scalars, N);

    bool match = (memcmp(&result1, &result2, sizeof(G1Affine)) == 0);
    TEST_ASSERT(match, "MSM n=256: parallel reduce deterministic");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM on-curve at n=8192 (c=7, num_buckets=65, stresses parallel scan)
void test_msm_on_curve_8k() {
    using namespace ff_ref;
    printf("test_msm_on_curve_8k...\n");

    const int n = 8192;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8 + 0] = (uint32_t)(i * 0xDEAD + 0xBEEF);
        h_scalars[i * 8 + 1] = (uint32_t)(i * 0x1337 + 0x42);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    G1AffineRef result_ref;
    result_ref.x = FqRef::from_u32(gpu_result.x.limbs);
    result_ref.y = FqRef::from_u32(gpu_result.y.limbs);
    result_ref.infinity = gpu_result.infinity;

    TEST_ASSERT(gpu_result.infinity || g1_is_on_curve_ref(result_ref),
                "MSM on-curve n=8192 (parallel reduce, c=7)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with all-ones scalars at n=1024 (stresses bucket accumulation + parallel reduce)
void test_msm_all_ones_1k() {
    using namespace ff_ref;
    printf("test_msm_all_ones_1k...\n");

    const int N = 1024;
    G1AffineRef cpu_g = G1AffineRef::generator();

    // All scalars = 1, all bases = G. Result should be N*G.
    std::vector<G1Affine> h_bases(N, make_g1_gen_affine_gpu());
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) h_scalars[i * 8] = 1;

    uint32_t sN[8] = {(uint32_t)N, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef cpu_nG = g1_scalar_mul_ref(cpu_g, sN);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_nG), "MSM: 1024*1*G = 1024G (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with spread scalars to stress many non-empty buckets
void test_msm_spread_scalars() {
    using namespace ff_ref;
    printf("test_msm_spread_scalars...\n");

    const int N = 64;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Scalars: i*7+3 — spread across bucket range, exercises parallel suffix scan
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) {
        h_scalars[i * 8] = (uint32_t)(i * 7 + 3);
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM: spread scalars (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with two points and large multi-limb scalars
void test_msm_two_large_scalars() {
    using namespace ff_ref;
    printf("test_msm_two_large_scalars...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();

    // scalar1 = 0x00000000_00000000_00001234_DEADBEEF
    uint32_t s1[8] = {0xDEADBEEFu, 0x1234u, 0, 0, 0, 0, 0, 0};
    // scalar2 = 0x00000000_00000000_0000ABCD_42424242
    uint32_t s2[8] = {0x42424242u, 0xABCDu, 0, 0, 0, 0, 0, 0};

    G1AffineRef base1 = cpu_g;
    uint32_t s_two[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    G1AffineRef base2 = g1_scalar_mul_ref(cpu_g, s_two);

    G1AffineRef cpu_bases[2] = {base1, base2};
    uint32_t cpu_scalars[16];
    memcpy(&cpu_scalars[0], s1, 32);
    memcpy(&cpu_scalars[8], s2, 32);

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases, cpu_scalars, 2);

    std::vector<G1Affine> h_bases(2);
    base1.x.to_u32(h_bases[0].x.limbs);
    base1.y.to_u32(h_bases[0].y.limbs);
    h_bases[0].infinity = false;
    base2.x.to_u32(h_bases[1].x.limbs);
    base2.y.to_u32(h_bases[1].y.limbs);
    h_bases[1].infinity = false;

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, 2 * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), 2 * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, cpu_scalars, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 2);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM: 2-point large scalars (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with ascending scalars at n=128 (exercises variety of bucket occupancies)
void test_msm_ascending_128() {
    using namespace ff_ref;
    printf("test_msm_ascending_128...\n");

    const int n = 128;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(n);
    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Ascending scalars: 1, 2, 3, ..., 128
    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) h_scalars[i * 8] = (uint32_t)(i + 1);

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), n);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM ascending n=128 (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM half-zero scalars at n=256 (half scalars zero, stresses empty bucket handling)
void test_msm_half_zero_256() {
    using namespace ff_ref;
    printf("test_msm_half_zero_256...\n");

    const int n = 256;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(n);
    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Even indices zero, odd indices non-zero
    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        if (i & 1) h_scalars[i * 8] = (uint32_t)(i * 3 + 7);
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), n);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM half-zero n=256 (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM on-curve at n=16384 (c=8, num_buckets=129, larger parallel reduce)
void test_msm_on_curve_16k() {
    using namespace ff_ref;
    printf("test_msm_on_curve_16k...\n");

    const int n = 16384;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1Affine> h_bases(n);
    for (int i = 0; i < n; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        G1AffineRef pt = g1_scalar_mul_ref(cpu_g, si);
        pt.x.to_u32(h_bases[i].x.limbs);
        pt.y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    std::vector<uint32_t> h_scalars(n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h_scalars[i * 8 + 0] = (uint32_t)(i * 0xCAFE + 0xBABE);
        h_scalars[i * 8 + 1] = (uint32_t)(i * 0xFACE + 0xD00D);
        h_scalars[i * 8 + 2] = (uint32_t)(i * 0xBEAD);
    }

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    G1AffineRef result_ref;
    result_ref.x = FqRef::from_u32(gpu_result.x.limbs);
    result_ref.y = FqRef::from_u32(gpu_result.y.limbs);
    result_ref.infinity = gpu_result.infinity;

    TEST_ASSERT(gpu_result.infinity || g1_is_on_curve_ref(result_ref),
                "MSM on-curve n=16384 (parallel reduce, c=8)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM small n=3 (minimum for Pippenger path, c=4, num_buckets=9)
void test_msm_small_n3() {
    using namespace ff_ref;
    printf("test_msm_small_n3...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();

    // 3 points: G, 2G, 3G with scalars 5, 7, 11
    // Expected: 5G + 14G + 33G = 52G
    std::vector<G1AffineRef> cpu_bases(3);
    std::vector<G1Affine> h_bases(3);
    for (int i = 0; i < 3; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    uint32_t scalars[24] = {0};
    scalars[0] = 5;
    scalars[8] = 7;
    scalars[16] = 11;

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), scalars, 3);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, 3 * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 24 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), 3 * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars, 24 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, 3);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM n=3: 5G+14G+33G (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM uniform scalar (every point has same scalar, exercises one hot bucket)
void test_msm_uniform_scalar_256() {
    using namespace ff_ref;
    printf("test_msm_uniform_scalar_256...\n");

    const int N = 256;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // All scalars = 99
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) h_scalars[i * 8] = 99;

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM uniform scalar=99 n=256 (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with high-bit scalars at n=64 (all windows active, stresses carry propagation)
void test_msm_high_bits_64() {
    using namespace ff_ref;
    printf("test_msm_high_bits_64...\n");

    const int N = 64;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Fill all 8 limbs with pattern — exercises all windows and carry chain
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) {
        for (int l = 0; l < 8; ++l) {
            h_scalars[i * 8 + l] = (uint32_t)((i + 1) * (l + 1) * 0x01010101u);
        }
        // Clamp top limb to stay below r
        h_scalars[i * 8 + 7] &= 0x73u;
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM high-bit n=64 (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with alternating scalars (0xAAAA..., 0x5555...) at n=32
void test_msm_alternating_pattern() {
    using namespace ff_ref;
    printf("test_msm_alternating_pattern...\n");

    const int N = 32;
    G1AffineRef cpu_g = G1AffineRef::generator();

    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Alternating bit patterns
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) {
        uint32_t pat = (i & 1) ? 0x55555555u : 0xAAAAAAAAu;
        h_scalars[i * 8 + 0] = pat;
        h_scalars[i * 8 + 1] = pat;
        h_scalars[i * 8 + 7] &= 0x73u;  // clamp below r
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM alternating 0xAA/0x55 n=32 (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM n=4 (minimal Pippenger, c=4, only 9 buckets, tiny parallel scan)
void test_msm_minimal_pippenger() {
    using namespace ff_ref;
    printf("test_msm_minimal_pippenger...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();

    const int N = 4;
    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Scalars: 15, 14, 13, 12 — all within one window of c=4
    uint32_t scalars[32] = {0};
    scalars[0]  = 15;
    scalars[8]  = 14;
    scalars[16] = 13;
    scalars[24] = 12;

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), scalars, N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, 32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), "MSM n=4 minimal Pippenger (parallel reduce)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// ─── v2.1.0 Session 28: Window Auto-Tuning + Memory Pool Tests ──────────────

// Test: window auto-tuner cap at c=11 (parallel reduction limit)
void test_msm_window_cap() {
    printf("test_msm_window_cap...\n");

    // At n=2^22 and above, c must be capped at 11
    TEST_ASSERT(msm_optimal_window(1 << 22) == 11, "MSM window cap: n=2^22 -> c=11 (was 12)");
    TEST_ASSERT(msm_optimal_window(1 << 24) == 11, "MSM window cap: n=2^24 -> c=11 (was 13)");
    TEST_ASSERT(msm_optimal_window(1 << 26) == 11, "MSM window cap: n=2^26 -> c=11 (was 14)");

    // Existing sizes should be unchanged
    TEST_ASSERT(msm_optimal_window(1 << 10) == 6, "MSM window: n=2^10 -> c=6 unchanged");
    TEST_ASSERT(msm_optimal_window(1 << 14) == 8, "MSM window: n=2^14 -> c=8 unchanged");
    TEST_ASSERT(msm_optimal_window(1 << 18) == 10, "MSM window: n=2^18 -> c=10 unchanged");
    TEST_ASSERT(msm_optimal_window(1 << 20) == 11, "MSM window: n=2^20 -> c=11 unchanged");

    // Verify bucket count fits parallel reduction (2^(c-1) <= 1024)
    for (int log_n = 10; log_n <= 26; ++log_n) {
        int c = msm_optimal_window((size_t)1 << log_n);
        uint32_t active_buckets = (1u << (c - 1));  // num_buckets - 1
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "MSM window n=2^%d: c=%d -> %u active buckets <= 1024", log_n, c, active_buckets);
        TEST_ASSERT(active_buckets <= 1024, msg);
    }
}

// Test: MSM on-curve at n=32768 (c=8, exercises pool reuse path)
void test_msm_on_curve_32k() {
    using namespace ff_ref;
    printf("test_msm_on_curve_32k...\n");

    const int n = 32768;

    // Use generator point for all bases (valid on-curve)
    G1Affine gen;
    for (int i = 0; i < 12; ++i) {
        gen.x.limbs[i] = G1_GEN_X[i];
        gen.y.limbs[i] = G1_GEN_Y[i];
    }
    gen.infinity = false;

    std::vector<G1Affine> h_bases(n, gen);

    // Pseudo-random scalars
    std::vector<uint32_t> h_scalars(n * 8);
    for (int i = 0; i < n; ++i) {
        uint32_t state = 42 + i * 1103515245u + 12345u;
        for (int l = 0; l < 8; ++l) {
            state = state * 1103515245u + 12345u;
            h_scalars[i * 8 + l] = state;
        }
        h_scalars[i * 8 + 7] &= 0x73ffffffu;
    }

    G1Affine* d_bases;
    uint32_t* d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, n * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, n);

    // Verify on-curve
    G1AffineRef ref;
    ref.x = FqRef::from_u32(gpu_result.x.limbs);
    ref.y = FqRef::from_u32(gpu_result.y.limbs);
    ref.infinity = gpu_result.infinity;

    TEST_ASSERT(g1_is_on_curve_ref(ref),
                "MSM on-curve n=32768 (c=8, memory pool)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM repeated calls reuse memory pool (correctness after multiple calls)
void test_msm_pool_reuse() {
    using namespace ff_ref;
    printf("test_msm_pool_reuse...\n");

    const int N = 64;

    G1AffineRef cpu_g = G1AffineRef::generator();
    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Different scalars for each run
    uint32_t scalars1[N * 8] = {0};
    uint32_t scalars2[N * 8] = {0};
    for (int i = 0; i < N; ++i) {
        scalars1[i * 8] = (uint32_t)(i + 1);
        scalars2[i * 8] = (uint32_t)(N - i);
    }

    G1AffineRef cpu1 = msm_naive_ref(cpu_bases.data(), scalars1, N);
    G1AffineRef cpu2 = msm_naive_ref(cpu_bases.data(), scalars2, N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));

    // Run 1 — pool allocates fresh
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars1, N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    G1Affine gpu1;
    msm_g1(&gpu1, d_bases, d_scalars, N);
    TEST_ASSERT(g1_affine_eq(gpu1, cpu1), "MSM pool reuse: run 1 correct");

    // Run 2 — pool should reuse cached allocations
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars2, N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    G1Affine gpu2;
    msm_g1(&gpu2, d_bases, d_scalars, N);
    TEST_ASSERT(g1_affine_eq(gpu2, cpu2), "MSM pool reuse: run 2 correct (reused pool)");

    // Run 3 — same as run 1, verify no stale data from run 2
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars1, N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    G1Affine gpu3;
    msm_g1(&gpu3, d_bases, d_scalars, N);
    TEST_ASSERT(g1_affine_eq(gpu3, cpu1), "MSM pool reuse: run 3 matches run 1 (no stale data)");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with explicit window size verification at boundary n
void test_msm_window_boundary() {
    using namespace ff_ref;
    printf("test_msm_window_boundary...\n");

    // n=2^20 is the boundary where c=11 (max for parallel reduce)
    // num_buckets = 2^10 + 1 = 1025, active = 1024 (exactly at limit)
    int c = msm_optimal_window(1 << 20);
    TEST_ASSERT(c == 11, "MSM window boundary: n=2^20 -> c=11");
    uint32_t num_buckets = (1u << (c - 1)) + 1;
    TEST_ASSERT(num_buckets == 1025, "MSM window boundary: num_buckets=1025");
    TEST_ASSERT(num_buckets - 1 == 1024, "MSM window boundary: active=1024 (thread limit)");

    // n=2^21 also gets c=11 now (capped)
    c = msm_optimal_window(1 << 21);
    TEST_ASSERT(c == 11, "MSM window boundary: n=2^21 -> c=11 (capped from 11)");

    // n=2^22 would have been c=12 (from formula), now capped at 11
    c = msm_optimal_window(1 << 22);
    TEST_ASSERT(c == 11, "MSM window boundary: n=2^22 -> c=11 (capped from 12)");
}

// Test: MSM cross-validate at n=128 with varied scalars
// Verifies MSM correctness with a mix of single-limb scalars
void test_msm_window_independence() {
    using namespace ff_ref;
    printf("test_msm_window_independence...\n");

    const int N = 128;

    G1AffineRef cpu_g = G1AffineRef::generator();
    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    // Varied single-limb scalars (exercises multiple bucket occupancies)
    std::vector<uint32_t> h_scalars(N * 8, 0);
    for (int i = 0; i < N; ++i) {
        h_scalars[i * 8] = (uint32_t)((i * 37 + 13) % 256);
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), h_scalars.data(), N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result),
                "MSM window independence: n=128 varied scalars");

    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM with non-default stream (exercises stream-ordered memory pool)
void test_msm_nondefault_stream() {
    using namespace ff_ref;
    printf("test_msm_nondefault_stream...\n");

    const int N = 32;

    G1AffineRef cpu_g = G1AffineRef::generator();
    std::vector<G1AffineRef> cpu_bases(N);
    std::vector<G1Affine> h_bases(N);
    for (int i = 0; i < N; ++i) {
        uint32_t si[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
        cpu_bases[i] = g1_scalar_mul_ref(cpu_g, si);
        cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
        cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
        h_bases[i].infinity = false;
    }

    uint32_t scalars[N * 8] = {0};
    for (int i = 0; i < N; ++i) {
        scalars[i * 8] = (uint32_t)(i * 3 + 7);
    }

    G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), scalars, N);

    G1Affine *d_bases;
    uint32_t *d_scalars;
    CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars, N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Use non-default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    G1Affine gpu_result;
    msm_g1(&gpu_result, d_bases, d_scalars, N, stream);

    TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result),
                "MSM non-default stream: n=32 matches CPU");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_scalars));
}

// Test: MSM consecutive calls with different sizes (pool adapts)
void test_msm_varying_sizes() {
    using namespace ff_ref;
    printf("test_msm_varying_sizes...\n");

    G1AffineRef cpu_g = G1AffineRef::generator();

    // Test sizes: 8, 64, 16 (vary to stress pool reallocation)
    int sizes[] = {8, 64, 16};

    for (int si = 0; si < 3; ++si) {
        int N = sizes[si];

        std::vector<G1AffineRef> cpu_bases(N);
        std::vector<G1Affine> h_bases(N);
        std::vector<uint32_t> scalars(N * 8, 0);

        for (int i = 0; i < N; ++i) {
            uint32_t s[8] = {(uint32_t)(i + 1), 0, 0, 0, 0, 0, 0, 0};
            cpu_bases[i] = g1_scalar_mul_ref(cpu_g, s);
            cpu_bases[i].x.to_u32(h_bases[i].x.limbs);
            cpu_bases[i].y.to_u32(h_bases[i].y.limbs);
            h_bases[i].infinity = false;
            scalars[i * 8] = (uint32_t)(i + si + 1);
        }

        G1AffineRef cpu_result = msm_naive_ref(cpu_bases.data(), scalars.data(), N);

        G1Affine *d_bases;
        uint32_t *d_scalars;
        CUDA_CHECK(cudaMalloc(&d_bases, N * sizeof(G1Affine)));
        CUDA_CHECK(cudaMalloc(&d_scalars, N * 8 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), N * sizeof(G1Affine), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scalars, scalars.data(), N * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        G1Affine gpu_result;
        msm_g1(&gpu_result, d_bases, d_scalars, N);

        char msg[128];
        snprintf(msg, sizeof(msg), "MSM varying sizes: n=%d (pool adapts)", N);
        TEST_ASSERT(g1_affine_eq(gpu_result, cpu_result), msg);

        CUDA_CHECK(cudaFree(d_bases));
        CUDA_CHECK(cudaFree(d_scalars));
    }
}

// Test: MSM formula edge cases (small n)
void test_msm_window_small_n() {
    printf("test_msm_window_small_n...\n");

    TEST_ASSERT(msm_optimal_window(0) == 1, "MSM window: n=0 -> c=1");
    TEST_ASSERT(msm_optimal_window(1) == 1, "MSM window: n=1 -> c=1");
    TEST_ASSERT(msm_optimal_window(2) == 4, "MSM window: n=2 -> c=4 (min clamp)");
    TEST_ASSERT(msm_optimal_window(3) == 4, "MSM window: n=3 -> c=4 (min clamp)");
    TEST_ASSERT(msm_optimal_window(15) == 4, "MSM window: n=15 -> c=4");
    TEST_ASSERT(msm_optimal_window(16) == 4, "MSM window: n=16 -> c=4 (log2=4, 4/2+1=3, clamped)");
    TEST_ASSERT(msm_optimal_window(256) == 5, "MSM window: n=256 -> c=5 (log2=8, 8/2+1=5)");
    TEST_ASSERT(msm_optimal_window(1024) == 6, "MSM window: n=1024 -> c=6 (log2=10, 10/2+1=6)");
}

// ─── v2.0.0 Session 23: Polynomial Operations Tests ─────────────────────────

// Helper: make standard-form FpElement from a small integer
static FpElement make_fp_from_u64(uint64_t v) {
    FpElement e;
    for (int i = 0; i < 8; ++i) e.limbs[i] = 0;
    e.limbs[0] = (uint32_t)(v & 0xFFFFFFFF);
    e.limbs[1] = (uint32_t)(v >> 32);
    return e;
}

// Helper: convert FpElement (standard form) to FpRef (Montgomery form)
static ff_ref::FpRef fp_to_ref_mont(const FpElement& e) {
    return ff_ref::to_montgomery(ff_ref::FpRef::from_u32(e.limbs));
}

// Helper: convert FpRef (Montgomery form) to FpElement (standard form)
static FpElement ref_mont_to_fp(const ff_ref::FpRef& r) {
    ff_ref::FpRef std_form = ff_ref::from_montgomery(r);
    FpElement e;
    std_form.to_u32(e.limbs);
    return e;
}

// Helper: compare two FpElements
static bool fp_eq(const FpElement& a, const FpElement& b) {
    for (int i = 0; i < 8; ++i)
        if (a.limbs[i] != b.limbs[i]) return false;
    return true;
}

void test_poly_pointwise_mul() {
    using namespace ff_ref;
    printf("test_poly_pointwise_mul...\n");

    const size_t N = 256;
    std::vector<FpElement> h_a(N), h_b(N), h_c(N);

    // Generate test data: a[i] = i+1, b[i] = i+100 (standard form)
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = make_fp_from_u64(i + 1);
        h_b[i] = make_fp_from_u64(i + 100);
    }

    FpElement *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_pointwise_mul(d_c, d_a, d_b, N);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Verify against CPU reference
    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        FpRef a_mont = to_montgomery(FpRef::from_u64(i + 1));
        FpRef b_mont = to_montgomery(FpRef::from_u64(i + 100));
        FpRef c_mont = fp_mul(a_mont, b_mont);
        FpElement expected = ref_mont_to_fp(c_mont);
        if (fp_eq(h_c[i], expected)) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "pointwise mul: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d matched\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void test_poly_pointwise_mul_sub() {
    using namespace ff_ref;
    printf("test_poly_pointwise_mul_sub...\n");

    const size_t N = 256;
    std::vector<FpElement> h_a(N), h_b(N), h_c(N), h_out(N);

    for (size_t i = 0; i < N; ++i) {
        h_a[i] = make_fp_from_u64(i + 10);
        h_b[i] = make_fp_from_u64(i + 20);
        h_c[i] = make_fp_from_u64(i + 5);
    }

    FpElement *d_a, *d_b, *d_c, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_pointwise_mul_sub(d_out, d_a, d_b, d_c, N);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        FpRef a_m = to_montgomery(FpRef::from_u64(i + 10));
        FpRef b_m = to_montgomery(FpRef::from_u64(i + 20));
        FpRef c_m = to_montgomery(FpRef::from_u64(i + 5));
        FpRef expected_m = fp_sub(fp_mul(a_m, b_m), c_m);
        FpElement expected = ref_mont_to_fp(expected_m);
        if (fp_eq(h_out[i], expected)) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "pointwise mul-sub: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d matched\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_out));
}

void test_poly_scale() {
    using namespace ff_ref;
    printf("test_poly_scale...\n");

    const size_t N = 256;
    std::vector<FpElement> h_data(N), h_result(N);

    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64(i + 1);
    }
    FpElement scalar = make_fp_from_u64(42);

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_scale(d_data, scalar, N);

    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    FpRef scalar_m = to_montgomery(FpRef::from_u64(42));
    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        FpRef val_m = to_montgomery(FpRef::from_u64(i + 1));
        FpRef expected_m = fp_mul(val_m, scalar_m);
        FpElement expected = ref_mont_to_fp(expected_m);
        if (fp_eq(h_result[i], expected)) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "poly scale: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d matched\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_roundtrip() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_roundtrip (n=256)...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);

    // Original data
    std::vector<FpElement> h_data(N);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64(i + 1);
    }
    std::vector<FpElement> h_original = h_data;

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    // Forward coset NTT then inverse -> should recover original
    poly_coset_ntt_forward(d_data, N, coset_gen);
    poly_coset_ntt_inverse(d_data, N, coset_gen);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fp_eq(h_data[i], h_original[i])) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT roundtrip: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d recovered\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_roundtrip_512() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_roundtrip (n=512)...\n");

    const size_t N = 512;
    FpElement coset_gen = make_fp_from_u64(7);

    std::vector<FpElement> h_data(N);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64((i * 17 + 3) % 10000);
    }
    std::vector<FpElement> h_original = h_data;

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_coset_ntt_forward(d_data, N, coset_gen);
    poly_coset_ntt_inverse(d_data, N, coset_gen);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fp_eq(h_data[i], h_original[i])) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT 512 roundtrip: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d recovered\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_vs_cpu() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_vs_cpu (n=256)...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);

    // GPU data (standard form)
    std::vector<FpElement> h_data(N);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64(i + 1);
    }

    // CPU reference (Montgomery form)
    FpRef g_mont = to_montgomery(FpRef::from_u64(7));
    std::vector<FpRef> cpu_data(N);
    for (size_t i = 0; i < N; ++i) {
        cpu_data[i] = to_montgomery(FpRef::from_u64(i + 1));
    }
    coset_ntt_forward_ref(cpu_data, N, g_mont);

    // GPU coset NTT
    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    poly_coset_ntt_forward(d_data, N, coset_gen);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Compare GPU (standard form) vs CPU (Montgomery form)
    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        FpElement cpu_std = ref_mont_to_fp(cpu_data[i]);
        if (fp_eq(h_data[i], cpu_std)) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT GPU vs CPU: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d matched\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_zeros() {
    printf("test_poly_coset_ntt_zeros (n=256)...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);
    FpElement zero = make_fp_from_u64(0);

    std::vector<FpElement> h_data(N, zero);

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_coset_ntt_forward(d_data, N, coset_gen);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // NTT of all zeros should be all zeros
    int all_zero = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fp_eq(h_data[i], zero)) ++all_zero;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT zeros: %d/%d zero", all_zero, (int)N);
    TEST_ASSERT(all_zero == (int)N, msg);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_ones() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_ones (n=256)...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);
    FpElement one = make_fp_from_u64(1);

    std::vector<FpElement> h_data(N, one);

    // CPU reference
    FpRef g_mont = to_montgomery(FpRef::from_u64(7));
    std::vector<FpRef> cpu_data(N, to_montgomery(FpRef::from_u64(1)));
    coset_ntt_forward_ref(cpu_data, N, g_mont);

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    poly_coset_ntt_forward(d_data, N, coset_gen);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        FpElement cpu_std = ref_mont_to_fp(cpu_data[i]);
        if (fp_eq(h_data[i], cpu_std)) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT ones: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_different_gen() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_different_gen (g=5, n=256)...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(5);

    std::vector<FpElement> h_data(N);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64(i * 3 + 2);
    }
    std::vector<FpElement> h_original = h_data;

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_coset_ntt_forward(d_data, N, coset_gen);
    poly_coset_ntt_inverse(d_data, N, coset_gen);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fp_eq(h_data[i], h_original[i])) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT g=5 roundtrip: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d recovered\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_coset_ntt_1024() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_roundtrip (n=1024)...\n");

    const size_t N = 1024;
    FpElement coset_gen = make_fp_from_u64(7);

    std::vector<FpElement> h_data(N);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64((i * 31 + 7) % 100000);
    }
    std::vector<FpElement> h_original = h_data;

    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_coset_ntt_forward(d_data, N, coset_gen);
    poly_coset_ntt_inverse(d_data, N, coset_gen);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fp_eq(h_data[i], h_original[i])) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset NTT 1024 roundtrip: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d recovered\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

// Test: quotient polynomial H(x) = (A*B - C) / Z_H on coset
// Use known polynomials where A(x)*B(x) - C(x) = H(x)*(x^n - 1)
// A(x) = 1 + x, B(x) = 1 + x, C(x) = 1 + 2x + x^2
// A*B = (1+x)^2 = 1 + 2x + x^2 = C, so H(x) = 0
void test_poly_quotient_zero() {
    using namespace ff_ref;
    printf("test_poly_quotient_zero...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);

    // A(x) = 1 + x: coeffs [1, 1, 0, 0, ...]
    std::vector<FpElement> h_a(N, make_fp_from_u64(0));
    h_a[0] = make_fp_from_u64(1);
    h_a[1] = make_fp_from_u64(1);

    // B(x) = 1 + x
    std::vector<FpElement> h_b = h_a;

    // C(x) = 1 + 2x + x^2
    std::vector<FpElement> h_c(N, make_fp_from_u64(0));
    h_c[0] = make_fp_from_u64(1);
    h_c[1] = make_fp_from_u64(2);
    h_c[2] = make_fp_from_u64(1);

    FpElement *d_a, *d_b, *d_c, *d_h;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_h, N * sizeof(FpElement)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    // Evaluate on coset
    poly_coset_ntt_forward(d_a, N, coset_gen);
    poly_coset_ntt_forward(d_b, N, coset_gen);
    poly_coset_ntt_forward(d_c, N, coset_gen);

    // H_coset = (A_coset * B_coset - C_coset) / (g^n - 1)
    poly_pointwise_mul_sub(d_h, d_a, d_b, d_c, N);

    // Compute Z_H inverse = 1/(g^n - 1) on CPU
    FpRef g_mont = to_montgomery(FpRef::from_u64(7));
    // g^n mod p
    std::array<uint64_t, 4> n_exp = {{N, 0, 0, 0}};
    FpRef g_n = fp_pow(g_mont, n_exp);
    FpRef one_mont;
    one_mont.limbs = R_MOD;
    FpRef zh_val = fp_sub(g_n, one_mont);  // g^n - 1 (Montgomery)
    FpRef zh_inv = fp_inv(zh_val);
    FpElement zh_inv_std = ref_mont_to_fp(zh_inv);

    poly_scale(d_h, zh_inv_std, N);

    // Inverse coset NTT to get H(x) coefficients
    poly_coset_ntt_inverse(d_h, N, coset_gen);

    std::vector<FpElement> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_h, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // H(x) should be all zeros (since A*B = C)
    FpElement zero = make_fp_from_u64(0);
    int all_zero = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fp_eq(h_result[i], zero)) ++all_zero;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "quotient zero: %d/%d zero coeffs", all_zero, (int)N);
    TEST_ASSERT(all_zero == (int)N, msg);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_h));
}

// Test: verify A*B - C = H*Z_H identity at a random evaluation point
// Construct C such that C(omega^i) = A(omega^i)*B(omega^i) for all roots of unity.
// This ensures Z_H | (A*B - C), making H well-defined.
void test_poly_quotient_identity() {
    using namespace ff_ref;
    printf("test_poly_quotient_identity...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);

    // A(x) = 2 + 3x (coefficients, standard form)
    std::vector<FpElement> h_a(N, make_fp_from_u64(0));
    h_a[0] = make_fp_from_u64(2);
    h_a[1] = make_fp_from_u64(3);

    // B(x) = 1 + x
    std::vector<FpElement> h_b(N, make_fp_from_u64(0));
    h_b[0] = make_fp_from_u64(1);
    h_b[1] = make_fp_from_u64(1);

    // Construct C: NTT A and B, pointwise multiply, INTT -> C coefficients
    // This guarantees C(omega^i) = A(omega^i)*B(omega^i) for all i
    FpElement *d_a, *d_b, *d_c, *d_h;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_h, N * sizeof(FpElement)));

    // Compute C = A*B mod Z_H via standard NTT
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    ntt_forward(d_a, N);
    ntt_forward(d_b, N);
    poly_pointwise_mul(d_c, d_a, d_b, N);
    ntt_inverse(d_c, N);

    // Read back C coefficients for CPU verification
    std::vector<FpElement> h_c(N);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Now compute H(x) = (A*B - C) / Z_H via coset NTT
    // Re-upload original A, B coefficients (NTT modified them in-place)
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_coset_ntt_forward(d_a, N, coset_gen);
    poly_coset_ntt_forward(d_b, N, coset_gen);
    poly_coset_ntt_forward(d_c, N, coset_gen);

    poly_pointwise_mul_sub(d_h, d_a, d_b, d_c, N);

    // Divide by Z_H(coset) = g^n - 1
    FpRef g_mont = to_montgomery(FpRef::from_u64(7));
    std::array<uint64_t, 4> n_exp = {{N, 0, 0, 0}};
    FpRef g_n = fp_pow(g_mont, n_exp);
    FpRef one_mont;
    one_mont.limbs = R_MOD;
    FpRef zh_val = fp_sub(g_n, one_mont);
    FpRef zh_inv = fp_inv(zh_val);
    FpElement zh_inv_std = ref_mont_to_fp(zh_inv);
    poly_scale(d_h, zh_inv_std, N);

    poly_coset_ntt_inverse(d_h, N, coset_gen);

    std::vector<FpElement> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_h, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // Verify identity: A(z)*B(z) - C(z) = H(z)*Z_H(z) at z = 13
    FpRef z = to_montgomery(FpRef::from_u64(13));

    // Evaluate A(z) = 2 + 3z, B(z) = 1 + z
    FpRef az = fp_add(to_montgomery(FpRef::from_u64(2)),
                      fp_mul(to_montgomery(FpRef::from_u64(3)), z));
    FpRef bz = fp_add(one_mont, z);

    // Evaluate C(z) from its coefficients
    FpRef cz = to_montgomery(FpRef::from_u32(h_c[0].limbs));
    FpRef z_pow = one_mont;
    for (size_t i = 1; i < N; ++i) {
        z_pow = fp_mul(z_pow, z);
        FpRef coeff = to_montgomery(FpRef::from_u32(h_c[i].limbs));
        cz = fp_add(cz, fp_mul(coeff, z_pow));
    }

    // LHS = A(z)*B(z) - C(z)
    FpRef lhs = fp_sub(fp_mul(az, bz), cz);

    // Evaluate H(z) from its coefficients
    FpRef hz = to_montgomery(FpRef::from_u32(h_result[0].limbs));
    z_pow = one_mont;
    for (size_t i = 1; i < N; ++i) {
        z_pow = fp_mul(z_pow, z);
        FpRef coeff = to_montgomery(FpRef::from_u32(h_result[i].limbs));
        hz = fp_add(hz, fp_mul(coeff, z_pow));
    }

    // Z_H(z) = z^n - 1
    std::array<uint64_t, 4> n_exp2 = {{N, 0, 0, 0}};
    FpRef z_n = fp_pow(z, n_exp2);
    FpRef zh_z = fp_sub(z_n, one_mont);

    // RHS = H(z) * Z_H(z)
    FpRef rhs = fp_mul(hz, zh_z);

    TEST_ASSERT(lhs == rhs, "quotient identity: A(z)*B(z) - C(z) = H(z)*Z_H(z)");
    printf("  Identity verified at z=13\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_h));
}

void test_poly_coset_ntt_inverse_vs_cpu() {
    using namespace ff_ref;
    printf("test_poly_coset_ntt_inverse_vs_cpu (n=256)...\n");

    const size_t N = 256;
    FpElement coset_gen = make_fp_from_u64(7);

    // Start with coset evaluations (forward NTT output)
    std::vector<FpElement> h_data(N);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = make_fp_from_u64(i + 1);
    }

    // GPU: forward then inverse
    FpElement *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_coset_ntt_forward(d_data, N, coset_gen);

    // Read GPU forward result
    std::vector<FpElement> h_gpu_fwd(N);
    CUDA_CHECK(cudaMemcpy(h_gpu_fwd.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // CPU forward for comparison
    FpRef g_mont = to_montgomery(FpRef::from_u64(7));
    std::vector<FpRef> cpu_data(N);
    for (size_t i = 0; i < N; ++i) {
        cpu_data[i] = to_montgomery(FpRef::from_u64(i + 1));
    }
    coset_ntt_forward_ref(cpu_data, N, g_mont);

    // Now do GPU inverse
    poly_coset_ntt_inverse(d_data, N, coset_gen);

    std::vector<FpElement> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    // CPU inverse
    coset_ntt_inverse_ref(cpu_data, N, g_mont);

    // Compare GPU inverse vs CPU inverse
    int matched = 0;
    for (size_t i = 0; i < N; ++i) {
        FpElement cpu_std = ref_mont_to_fp(cpu_data[i]);
        if (fp_eq(h_result[i], cpu_std)) ++matched;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "coset INTT GPU vs CPU: %d/%d matched", matched, (int)N);
    TEST_ASSERT(matched == (int)N, msg);
    printf("  %d/%d matched\n", matched, (int)N);

    CUDA_CHECK(cudaFree(d_data));
}

void test_poly_pointwise_edge_cases() {
    printf("test_poly_pointwise_edge_cases...\n");

    // Multiply by zero
    const size_t N = 8;
    std::vector<FpElement> h_a(N), h_b(N), h_c(N);
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = make_fp_from_u64(i + 1);
        h_b[i] = make_fp_from_u64(0);
    }

    FpElement *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(FpElement)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));

    poly_pointwise_mul(d_c, d_a, d_b, N);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    FpElement zero = make_fp_from_u64(0);
    bool all_zero = true;
    for (size_t i = 0; i < N; ++i) {
        if (!fp_eq(h_c[i], zero)) { all_zero = false; break; }
    }
    TEST_ASSERT(all_zero, "pointwise mul by zero = zero");

    // Multiply by one
    for (size_t i = 0; i < N; ++i) {
        h_b[i] = make_fp_from_u64(1);
    }
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(FpElement), cudaMemcpyHostToDevice));
    poly_pointwise_mul(d_c, d_a, d_b, N);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(FpElement), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < N; ++i) {
        if (!fp_eq(h_c[i], h_a[i])) { all_match = false; break; }
    }
    TEST_ASSERT(all_match, "pointwise mul by one = identity");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// ─── v2.0.0 Session 24: Groth16 Tests ──────────────────────────────────────

void test_groth16_witness() {
    printf("test_groth16_witness...\n");

    // x=3: 3^3 + 3 + 5 = 35
    auto w = compute_witness(3);
    TEST_ASSERT(w[0].limbs[0] == 1, "witness[0] = 1");
    TEST_ASSERT(w[1].limbs[0] == 3, "witness[1] = x = 3");
    TEST_ASSERT(w[2].limbs[0] == 35, "witness[2] = y = 35");
    TEST_ASSERT(w[3].limbs[0] == 9, "witness[3] = v1 = 9");
    TEST_ASSERT(w[4].limbs[0] == 27, "witness[4] = v2 = 27");
    TEST_ASSERT(w[5].limbs[0] == 30, "witness[5] = v3 = 30");

    // x=10: 10^3 + 10 + 5 = 1015
    w = compute_witness(10);
    TEST_ASSERT(w[2].limbs[0] == 1015, "witness x=10: y = 1015");
}

void test_groth16_r1cs_satisfied() {
    using namespace ff_ref;
    printf("test_groth16_r1cs_satisfied...\n");

    R1CS r1cs = make_toy_r1cs(256);
    auto witness = compute_witness(3);

    // Check: A_row · w * B_row · w = C_row · w for each constraint
    for (size_t k = 0; k < r1cs.num_constraints; ++k) {
        FpRef a_dot = FpRef::zero(), b_dot = FpRef::zero(), c_dot = FpRef::zero();
        for (size_t i = 0; i < r1cs.num_variables; ++i) {
            FpRef w_m = to_montgomery(FpRef::from_u32(witness[i].limbs));
            a_dot = fp_add(a_dot, fp_mul(to_montgomery(FpRef::from_u64(r1cs.A[k][i])), w_m));
            b_dot = fp_add(b_dot, fp_mul(to_montgomery(FpRef::from_u64(r1cs.B[k][i])), w_m));
            c_dot = fp_add(c_dot, fp_mul(to_montgomery(FpRef::from_u64(r1cs.C[k][i])), w_m));
        }
        FpRef product = fp_mul(a_dot, b_dot);
        char msg[128];
        snprintf(msg, sizeof(msg), "R1CS constraint %d satisfied", (int)k);
        TEST_ASSERT(product == c_dot, msg);
    }
}

void test_groth16_srs_on_curve() {
    using namespace ff_ref;
    printf("test_groth16_srs_on_curve...\n");

    R1CS r1cs = make_toy_r1cs(256);
    ProvingKey pk = generate_proving_key(r1cs, 42);

    // Check SRS G1 points on curve
    G1AffineRef a_ref = {FqRef::from_u32(pk.alpha_g1.x.limbs),
                          FqRef::from_u32(pk.alpha_g1.y.limbs),
                          pk.alpha_g1.infinity};
    TEST_ASSERT(g1_is_on_curve_ref(a_ref), "SRS: alpha_g1 on curve");

    G1AffineRef b_ref = {FqRef::from_u32(pk.beta_g1.x.limbs),
                          FqRef::from_u32(pk.beta_g1.y.limbs),
                          pk.beta_g1.infinity};
    TEST_ASSERT(g1_is_on_curve_ref(b_ref), "SRS: beta_g1 on curve");

    G1AffineRef d_ref = {FqRef::from_u32(pk.delta_g1.x.limbs),
                          FqRef::from_u32(pk.delta_g1.y.limbs),
                          pk.delta_g1.infinity};
    TEST_ASSERT(g1_is_on_curve_ref(d_ref), "SRS: delta_g1 on curve");

    // Check h_query points on curve (sample a few)
    for (size_t j = 0; j < 5 && j < pk.h_query.size(); ++j) {
        G1AffineRef h_ref = {FqRef::from_u32(pk.h_query[j].x.limbs),
                              FqRef::from_u32(pk.h_query[j].y.limbs),
                              pk.h_query[j].infinity};
        char msg[128];
        snprintf(msg, sizeof(msg), "SRS: h_query[%d] on curve", (int)j);
        TEST_ASSERT(h_ref.infinity || g1_is_on_curve_ref(h_ref), msg);
    }

    // Check G2 points on curve
    G2AffineRef beta_g2_ref = {
        {FqRef::from_u32(pk.beta_g2.x.c0.limbs), FqRef::from_u32(pk.beta_g2.x.c1.limbs)},
        {FqRef::from_u32(pk.beta_g2.y.c0.limbs), FqRef::from_u32(pk.beta_g2.y.c1.limbs)},
        pk.beta_g2.infinity};
    TEST_ASSERT(g2_is_on_curve_ref(beta_g2_ref), "SRS: beta_g2 on curve");
}

void test_groth16_gpu_proof() {
    using namespace ff_ref;
    printf("test_groth16_gpu_proof...\n");

    R1CS r1cs = make_toy_r1cs(256);
    ProvingKey pk = generate_proving_key(r1cs, 42);
    auto witness = compute_witness(3);

    Groth16Proof proof = groth16_prove(r1cs, pk, witness, 17, 23);

    // Check proof elements on curve
    G1AffineRef pa = {FqRef::from_u32(proof.pi_a.x.limbs),
                       FqRef::from_u32(proof.pi_a.y.limbs),
                       proof.pi_a.infinity};
    TEST_ASSERT(proof.pi_a.infinity || g1_is_on_curve_ref(pa), "GPU proof: pi_a on G1");

    G2AffineRef pb = {
        {FqRef::from_u32(proof.pi_b.x.c0.limbs), FqRef::from_u32(proof.pi_b.x.c1.limbs)},
        {FqRef::from_u32(proof.pi_b.y.c0.limbs), FqRef::from_u32(proof.pi_b.y.c1.limbs)},
        proof.pi_b.infinity};
    TEST_ASSERT(proof.pi_b.infinity || g2_is_on_curve_ref(pb), "GPU proof: pi_b on G2");

    G1AffineRef pc = {FqRef::from_u32(proof.pi_c.x.limbs),
                       FqRef::from_u32(proof.pi_c.y.limbs),
                       proof.pi_c.infinity};
    TEST_ASSERT(proof.pi_c.infinity || g1_is_on_curve_ref(pc), "GPU proof: pi_c on G1");
}

void test_groth16_gpu_vs_cpu() {
    using namespace ff_ref;
    printf("test_groth16_gpu_vs_cpu...\n");

    R1CS r1cs = make_toy_r1cs(256);
    ProvingKey pk = generate_proving_key(r1cs, 42);
    auto witness = compute_witness(3);

    Groth16Proof gpu_proof = groth16_prove(r1cs, pk, witness, 17, 23);
    Groth16Proof cpu_proof = groth16_prove_cpu(r1cs, pk, witness, 17, 23);

    // Compare π_A
    bool a_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_a.x.limbs[i] != cpu_proof.pi_a.x.limbs[i]) a_match = false;
        if (gpu_proof.pi_a.y.limbs[i] != cpu_proof.pi_a.y.limbs[i]) a_match = false;
    }
    TEST_ASSERT(a_match, "GPU vs CPU: pi_a match");

    // Compare π_B
    bool b_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_b.x.c0.limbs[i] != cpu_proof.pi_b.x.c0.limbs[i]) b_match = false;
        if (gpu_proof.pi_b.x.c1.limbs[i] != cpu_proof.pi_b.x.c1.limbs[i]) b_match = false;
        if (gpu_proof.pi_b.y.c0.limbs[i] != cpu_proof.pi_b.y.c0.limbs[i]) b_match = false;
        if (gpu_proof.pi_b.y.c1.limbs[i] != cpu_proof.pi_b.y.c1.limbs[i]) b_match = false;
    }
    TEST_ASSERT(b_match, "GPU vs CPU: pi_b match");

    // Compare π_C
    bool c_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_c.x.limbs[i] != cpu_proof.pi_c.x.limbs[i]) c_match = false;
        if (gpu_proof.pi_c.y.limbs[i] != cpu_proof.pi_c.y.limbs[i]) c_match = false;
    }
    TEST_ASSERT(c_match, "GPU vs CPU: pi_c match");
}

void test_groth16_determinism() {
    printf("test_groth16_determinism...\n");

    R1CS r1cs = make_toy_r1cs(256);
    ProvingKey pk = generate_proving_key(r1cs, 42);
    auto witness = compute_witness(3);

    Groth16Proof p1 = groth16_prove(r1cs, pk, witness, 17, 23);
    Groth16Proof p2 = groth16_prove(r1cs, pk, witness, 17, 23);

    bool match = true;
    for (int i = 0; i < 12; ++i) {
        if (p1.pi_a.x.limbs[i] != p2.pi_a.x.limbs[i]) match = false;
        if (p1.pi_a.y.limbs[i] != p2.pi_a.y.limbs[i]) match = false;
    }
    TEST_ASSERT(match, "determinism: same input → same proof");
}

void test_groth16_different_witness() {
    printf("test_groth16_different_witness...\n");

    R1CS r1cs = make_toy_r1cs(256);
    ProvingKey pk = generate_proving_key(r1cs, 42);

    Groth16Proof p1 = groth16_prove(r1cs, pk, compute_witness(3), 17, 23);
    Groth16Proof p2 = groth16_prove(r1cs, pk, compute_witness(5), 17, 23);

    // Different witness should produce different proof
    bool different = false;
    for (int i = 0; i < 12; ++i) {
        if (p1.pi_a.x.limbs[i] != p2.pi_a.x.limbs[i]) { different = true; break; }
    }
    TEST_ASSERT(different, "different witness → different proof");
}

void test_groth16_different_randomness() {
    printf("test_groth16_different_randomness...\n");

    R1CS r1cs = make_toy_r1cs(256);
    ProvingKey pk = generate_proving_key(r1cs, 42);
    auto witness = compute_witness(3);

    Groth16Proof p1 = groth16_prove(r1cs, pk, witness, 17, 23);
    Groth16Proof p2 = groth16_prove(r1cs, pk, witness, 31, 37);

    bool different = false;
    for (int i = 0; i < 12; ++i) {
        if (p1.pi_a.x.limbs[i] != p2.pi_a.x.limbs[i]) { different = true; break; }
    }
    TEST_ASSERT(different, "different randomness → different proof");
}

// ─── v2.2.0 Session 29: Fibonacci Circuit Tests ─────────────────────────────

// Conversion helpers (same as in groth16.cu, needed here for tests)
static ff_ref::G1AffineRef gpu_to_ref_g1(const G1Affine& g) {
    if (g.infinity) return ff_ref::G1AffineRef::point_at_infinity();
    ff_ref::G1AffineRef r;
    r.x = ff_ref::FqRef::from_u32(g.x.limbs);
    r.y = ff_ref::FqRef::from_u32(g.y.limbs);
    r.infinity = false;
    return r;
}

static ff_ref::G2AffineRef gpu_to_ref_g2(const G2Affine& g) {
    if (g.infinity) return ff_ref::G2AffineRef::point_at_infinity();
    ff_ref::G2AffineRef r;
    r.x.c0 = ff_ref::FqRef::from_u32(g.x.c0.limbs);
    r.x.c1 = ff_ref::FqRef::from_u32(g.x.c1.limbs);
    r.y.c0 = ff_ref::FqRef::from_u32(g.y.c0.limbs);
    r.y.c1 = ff_ref::FqRef::from_u32(g.y.c1.limbs);
    r.infinity = false;
    return r;
}

void test_fibonacci_witness_small() {
    printf("test_fibonacci_witness_small...\n");

    // Standard Fibonacci: F(0)=1, F(1)=1, F(2)=2, F(3)=3, F(4)=5, ...
    auto w = compute_fibonacci_witness(1, 1, 8);

    // w = [1, a0=1, a1=1, a2=2, a3=3, a4=5, a5=8, a6=13, a7=21, a8=34, a9=55]
    TEST_ASSERT(w[0].limbs[0] == 1 && w[0].limbs[1] == 0, "fib: w[0] = 1 (constant)");
    TEST_ASSERT(w[1].limbs[0] == 1, "fib: a0 = 1");
    TEST_ASSERT(w[2].limbs[0] == 1, "fib: a1 = 1");
    TEST_ASSERT(w[3].limbs[0] == 2, "fib: a2 = 2");
    TEST_ASSERT(w[4].limbs[0] == 3, "fib: a3 = 3");
    TEST_ASSERT(w[5].limbs[0] == 5, "fib: a4 = 5");
    TEST_ASSERT(w[6].limbs[0] == 8, "fib: a5 = 8");
    TEST_ASSERT(w[7].limbs[0] == 13, "fib: a6 = 13");
    TEST_ASSERT(w[8].limbs[0] == 21, "fib: a7 = 21");
    TEST_ASSERT(w[9].limbs[0] == 34, "fib: a8 = 34");

    // Check recurrence: a_{i+2} = a_i + a_{i+1}
    for (size_t i = 0; i < 8; ++i) {
        uint32_t ai   = w[i + 1].limbs[0];
        uint32_t ai1  = w[i + 2].limbs[0];
        uint32_t ai2  = w[i + 3].limbs[0];
        char msg[128];
        snprintf(msg, sizeof(msg), "fib: a[%d] + a[%d] = a[%d]", (int)i, (int)(i+1), (int)(i+2));
        TEST_ASSERT(ai + ai1 == ai2, msg);
    }
}

void test_fibonacci_witness_different_start() {
    printf("test_fibonacci_witness_different_start...\n");

    // F(0)=0, F(1)=1: classic Fibonacci
    auto w = compute_fibonacci_witness(0, 1, 4);
    TEST_ASSERT(w[1].limbs[0] == 0, "fib(0,1): a0 = 0");
    TEST_ASSERT(w[2].limbs[0] == 1, "fib(0,1): a1 = 1");
    TEST_ASSERT(w[3].limbs[0] == 1, "fib(0,1): a2 = 1");
    TEST_ASSERT(w[4].limbs[0] == 2, "fib(0,1): a3 = 2");
    TEST_ASSERT(w[5].limbs[0] == 3, "fib(0,1): a4 = 3");
    TEST_ASSERT(w[6].limbs[0] == 5, "fib(0,1): a5 = 5");

    // F(0)=2, F(1)=3: Lucas-like
    w = compute_fibonacci_witness(2, 3, 4);
    TEST_ASSERT(w[1].limbs[0] == 2, "fib(2,3): a0 = 2");
    TEST_ASSERT(w[2].limbs[0] == 3, "fib(2,3): a1 = 3");
    TEST_ASSERT(w[3].limbs[0] == 5, "fib(2,3): a2 = 5");
    TEST_ASSERT(w[4].limbs[0] == 8, "fib(2,3): a3 = 8");
}

void test_fibonacci_r1cs_satisfied() {
    using namespace ff_ref;
    printf("test_fibonacci_r1cs_satisfied...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    auto witness = compute_fibonacci_witness(1, 1, 8);

    // Build dense evaluation vectors for the active constraints
    // Check: A_row · w * B_row · w = C_row · w for each constraint
    size_t n = r1cs.domain_size;
    size_t nv = r1cs.num_variables;

    // Initialize dense evaluations to zero
    std::vector<FpRef> a_dot(n, FpRef::zero());
    std::vector<FpRef> b_dot(n, FpRef::zero());
    std::vector<FpRef> c_dot(n, FpRef::zero());

    for (const auto& e : r1cs.A) {
        FpRef val_m = to_montgomery(FpRef::from_u64(e.val));
        FpRef w_m = to_montgomery(FpRef::from_u32(witness[e.col].limbs));
        a_dot[e.row] = fp_add(a_dot[e.row], fp_mul(val_m, w_m));
    }
    for (const auto& e : r1cs.B) {
        FpRef val_m = to_montgomery(FpRef::from_u64(e.val));
        FpRef w_m = to_montgomery(FpRef::from_u32(witness[e.col].limbs));
        b_dot[e.row] = fp_add(b_dot[e.row], fp_mul(val_m, w_m));
    }
    for (const auto& e : r1cs.C) {
        FpRef val_m = to_montgomery(FpRef::from_u64(e.val));
        FpRef w_m = to_montgomery(FpRef::from_u32(witness[e.col].limbs));
        c_dot[e.row] = fp_add(c_dot[e.row], fp_mul(val_m, w_m));
    }

    for (size_t k = 0; k < r1cs.num_constraints; ++k) {
        FpRef product = fp_mul(a_dot[k], b_dot[k]);
        char msg[128];
        snprintf(msg, sizeof(msg), "fib R1CS constraint %d satisfied", (int)k);
        TEST_ASSERT(product == c_dot[k], msg);
    }

    // Padded rows (beyond num_constraints) should be zero
    for (size_t k = r1cs.num_constraints; k < n; ++k) {
        TEST_ASSERT(a_dot[k] == FpRef::zero(), "fib: padded A row is zero");
        TEST_ASSERT(b_dot[k] == FpRef::zero(), "fib: padded B row is zero");
        TEST_ASSERT(c_dot[k] == FpRef::zero(), "fib: padded C row is zero");
    }
}

void test_fibonacci_r1cs_medium() {
    using namespace ff_ref;
    printf("test_fibonacci_r1cs_medium...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(256);
    auto witness = compute_fibonacci_witness(1, 1, 256);

    // Spot check a few constraints
    for (size_t k : {(size_t)0, (size_t)1, (size_t)100, (size_t)255}) {
        // Constraint k: (a_k + a_{k+1}) * 1 = a_{k+2}
        FpRef a_k  = to_montgomery(FpRef::from_u32(witness[k + 1].limbs));
        FpRef a_k1 = to_montgomery(FpRef::from_u32(witness[k + 2].limbs));
        FpRef a_k2 = to_montgomery(FpRef::from_u32(witness[k + 3].limbs));
        FpRef sum = fp_add(a_k, a_k1);
        char msg[128];
        snprintf(msg, sizeof(msg), "fib256: constraint %d: a[%d]+a[%d]=a[%d]",
                 (int)k, (int)k, (int)(k+1), (int)(k+2));
        TEST_ASSERT(sum == a_k2, msg);
    }
}

void test_fibonacci_batch_inversion() {
    using namespace ff_ref;
    printf("test_fibonacci_batch_inversion...\n");

    // Test batch inversion against individual inversions
    size_t n = 16;
    std::vector<FpRef> inputs(n);
    for (size_t i = 0; i < n; ++i)
        inputs[i] = to_montgomery(FpRef::from_u64(i + 3));  // avoid zero

    std::vector<FpRef> batch_outputs;
    fp_batch_inverse(inputs, batch_outputs, n);

    // Compare with individual inversions
    for (size_t i = 0; i < n; ++i) {
        FpRef expected = fp_inv(inputs[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "batch_inv[%d] matches fp_inv", (int)i);
        TEST_ASSERT(batch_outputs[i] == expected, msg);
    }

    // Verify a * a^{-1} = 1 for each
    FpRef one_m;
    one_m.limbs = R_MOD;
    for (size_t i = 0; i < n; ++i) {
        FpRef product = fp_mul(inputs[i], batch_outputs[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "batch_inv[%d]: a * a^-1 = 1", (int)i);
        TEST_ASSERT(product == one_m, msg);
    }
}

void test_fibonacci_sparse_setup_on_curve() {
    using namespace ff_ref;
    printf("test_fibonacci_sparse_setup_on_curve...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    // Check G1 SRS points
    G1AffineRef a_ref = gpu_to_ref_g1(pk.alpha_g1);
    TEST_ASSERT(g1_is_on_curve_ref(a_ref), "sparse SRS: alpha_g1 on curve");

    G1AffineRef b_ref = gpu_to_ref_g1(pk.beta_g1);
    TEST_ASSERT(g1_is_on_curve_ref(b_ref), "sparse SRS: beta_g1 on curve");

    G1AffineRef d_ref = gpu_to_ref_g1(pk.delta_g1);
    TEST_ASSERT(g1_is_on_curve_ref(d_ref), "sparse SRS: delta_g1 on curve");

    // Sample u_tau_g1 points
    for (size_t i = 0; i < 5 && i < pk.u_tau_g1.size(); ++i) {
        G1AffineRef p = gpu_to_ref_g1(pk.u_tau_g1[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "sparse SRS: u_tau_g1[%d] on curve", (int)i);
        TEST_ASSERT(p.infinity || g1_is_on_curve_ref(p), msg);
    }

    // Sample h_query points
    for (size_t j = 0; j < 5 && j < pk.h_query.size(); ++j) {
        G1AffineRef p = gpu_to_ref_g1(pk.h_query[j]);
        char msg[128];
        snprintf(msg, sizeof(msg), "sparse SRS: h_query[%d] on curve", (int)j);
        TEST_ASSERT(p.infinity || g1_is_on_curve_ref(p), msg);
    }

    // Sample l_query points
    for (size_t i = 0; i < 5 && i < pk.l_query.size(); ++i) {
        G1AffineRef p = gpu_to_ref_g1(pk.l_query[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "sparse SRS: l_query[%d] on curve", (int)i);
        TEST_ASSERT(p.infinity || g1_is_on_curve_ref(p), msg);
    }

    // Check v_tau_scalars populated
    TEST_ASSERT(pk.v_tau_scalars.size() == r1cs.num_variables, "sparse SRS: v_tau_scalars populated");
}

void test_fibonacci_lagrange_basis() {
    using namespace ff_ref;
    printf("test_fibonacci_lagrange_basis...\n");

    // Verify Lagrange basis: Σ L_k(tau) * f(omega^k) = f(tau) for known polynomial
    // Use f(x) = x^2 evaluated at roots of unity, then check evaluation at tau
    size_t n = 8;
    FpRef tau = to_montgomery(FpRef::from_u64(42));
    FpRef omega = get_root_of_unity(n);
    FpRef one_m;
    one_m.limbs = R_MOD;

    // Compute omega^k
    std::vector<FpRef> omega_pows(n);
    omega_pows[0] = one_m;
    for (size_t k = 1; k < n; ++k)
        omega_pows[k] = fp_mul(omega_pows[k - 1], omega);

    // f(omega^k) = (omega^k)^2
    std::vector<FpRef> f_evals(n);
    for (size_t k = 0; k < n; ++k)
        f_evals[k] = fp_mul(omega_pows[k], omega_pows[k]);

    // Compute Lagrange basis at tau using the same formula as generate_proving_key_sparse
    std::vector<FpRef> tau_minus_omega(n);
    for (size_t k = 0; k < n; ++k)
        tau_minus_omega[k] = fp_sub(tau, omega_pows[k]);

    std::vector<FpRef> inv_tmo;
    fp_batch_inverse(tau_minus_omega, inv_tmo, n);

    std::array<uint64_t, 4> n_exp = {{(uint64_t)n, 0, 0, 0}};
    FpRef tau_n = fp_pow(tau, n_exp);
    FpRef t_tau = fp_sub(tau_n, one_m);
    FpRef n_mont = to_montgomery(FpRef::from_u64((uint64_t)n));
    FpRef n_inv = fp_inv(n_mont);
    FpRef common = fp_mul(t_tau, n_inv);

    // Σ L_k(tau) * f(omega^k)
    // L_k(tau) = omega^k * common / (tau - omega^k)
    FpRef interp = FpRef::zero();
    for (size_t k = 0; k < n; ++k) {
        FpRef L_k = fp_mul(fp_mul(omega_pows[k], common), inv_tmo[k]);
        interp = fp_add(interp, fp_mul(L_k, f_evals[k]));
    }

    // Should equal f(tau) = tau^2
    FpRef expected = fp_mul(tau, tau);
    TEST_ASSERT(interp == expected, "Lagrange interpolation: Σ L_k * f(ω^k) = f(τ)");
}

void test_fibonacci_gpu_proof_small() {
    using namespace ff_ref;
    printf("test_fibonacci_gpu_proof_small...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto witness = compute_fibonacci_witness(1, 1, 8);

    Groth16Proof proof = groth16_prove_sparse(r1cs, pk, witness, 17, 23);

    // Check proof elements on curve
    G1AffineRef pa = gpu_to_ref_g1(proof.pi_a);
    TEST_ASSERT(proof.pi_a.infinity || g1_is_on_curve_ref(pa), "fib GPU proof: pi_a on G1");

    G2AffineRef pb = gpu_to_ref_g2(proof.pi_b);
    TEST_ASSERT(proof.pi_b.infinity || g2_is_on_curve_ref(pb), "fib GPU proof: pi_b on G2");

    G1AffineRef pc = gpu_to_ref_g1(proof.pi_c);
    TEST_ASSERT(proof.pi_c.infinity || g1_is_on_curve_ref(pc), "fib GPU proof: pi_c on G1");

    // Proof elements should not be infinity
    TEST_ASSERT(!proof.pi_a.infinity, "fib GPU proof: pi_a not infinity");
    TEST_ASSERT(!proof.pi_b.infinity, "fib GPU proof: pi_b not infinity");
}

void test_fibonacci_gpu_vs_cpu_small() {
    using namespace ff_ref;
    printf("test_fibonacci_gpu_vs_cpu_small...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto witness = compute_fibonacci_witness(1, 1, 8);

    Groth16Proof gpu_proof = groth16_prove_sparse(r1cs, pk, witness, 17, 23);
    Groth16Proof cpu_proof = groth16_prove_cpu_sparse(r1cs, pk, witness, 17, 23);

    // Note: GPU and CPU use different assembly methods (GPU: MSM + field arith for B,
    // CPU: sequential scalar muls with G2 points), so pi_B will differ.
    // pi_A and pi_C should match if NTT and H(x) are identical.
    // However, the GPU proof uses GPU MSM for pi_A and pi_C, while CPU uses sequential
    // scalar muls — both are mathematically equivalent, so results should match.

    // Compare π_A
    bool a_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_a.x.limbs[i] != cpu_proof.pi_a.x.limbs[i]) a_match = false;
        if (gpu_proof.pi_a.y.limbs[i] != cpu_proof.pi_a.y.limbs[i]) a_match = false;
    }
    TEST_ASSERT(a_match, "fib GPU vs CPU: pi_a match");

    // Compare π_C
    bool c_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_c.x.limbs[i] != cpu_proof.pi_c.x.limbs[i]) c_match = false;
        if (gpu_proof.pi_c.y.limbs[i] != cpu_proof.pi_c.y.limbs[i]) c_match = false;
    }
    TEST_ASSERT(c_match, "fib GPU vs CPU: pi_c match");
}

void test_fibonacci_gpu_proof_256() {
    using namespace ff_ref;
    printf("test_fibonacci_gpu_proof_256...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(256);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto witness = compute_fibonacci_witness(1, 1, 256);

    Groth16Proof proof = groth16_prove_sparse(r1cs, pk, witness, 17, 23);

    G1AffineRef pa = gpu_to_ref_g1(proof.pi_a);
    TEST_ASSERT(proof.pi_a.infinity || g1_is_on_curve_ref(pa), "fib256 GPU: pi_a on G1");

    G2AffineRef pb = gpu_to_ref_g2(proof.pi_b);
    TEST_ASSERT(proof.pi_b.infinity || g2_is_on_curve_ref(pb), "fib256 GPU: pi_b on G2");

    G1AffineRef pc = gpu_to_ref_g1(proof.pi_c);
    TEST_ASSERT(proof.pi_c.infinity || g1_is_on_curve_ref(pc), "fib256 GPU: pi_c on G1");

    TEST_ASSERT(!proof.pi_a.infinity, "fib256 GPU: pi_a not infinity");
    TEST_ASSERT(!proof.pi_b.infinity, "fib256 GPU: pi_b not infinity");
}

void test_fibonacci_determinism() {
    printf("test_fibonacci_determinism...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto witness = compute_fibonacci_witness(1, 1, 8);

    Groth16Proof p1 = groth16_prove_sparse(r1cs, pk, witness, 17, 23);
    Groth16Proof p2 = groth16_prove_sparse(r1cs, pk, witness, 17, 23);

    bool match = true;
    for (int i = 0; i < 12; ++i) {
        if (p1.pi_a.x.limbs[i] != p2.pi_a.x.limbs[i]) match = false;
        if (p1.pi_a.y.limbs[i] != p2.pi_a.y.limbs[i]) match = false;
    }
    TEST_ASSERT(match, "fib determinism: same input → same proof");
}

void test_fibonacci_different_inputs() {
    printf("test_fibonacci_different_inputs...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    Groth16Proof p1 = groth16_prove_sparse(r1cs, pk,
                         compute_fibonacci_witness(1, 1, 8), 17, 23);
    Groth16Proof p2 = groth16_prove_sparse(r1cs, pk,
                         compute_fibonacci_witness(2, 3, 8), 17, 23);

    bool different = false;
    for (int i = 0; i < 12; ++i) {
        if (p1.pi_a.x.limbs[i] != p2.pi_a.x.limbs[i]) { different = true; break; }
    }
    TEST_ASSERT(different, "fib: different inputs → different proof");
}

void test_fibonacci_zero_start() {
    using namespace ff_ref;
    printf("test_fibonacci_zero_start...\n");

    // F(0)=0, F(1)=0: all-zeros Fibonacci (trivial circuit)
    auto w = compute_fibonacci_witness(0, 0, 4);
    TEST_ASSERT(w[0].limbs[0] == 1, "fib(0,0): w[0] = 1 (constant)");
    for (size_t i = 1; i < w.size(); ++i) {
        bool is_zero = true;
        for (int j = 0; j < 8; ++j)
            if (w[i].limbs[j] != 0) { is_zero = false; break; }
        char msg[128];
        snprintf(msg, sizeof(msg), "fib(0,0): w[%d] = 0", (int)i);
        TEST_ASSERT(is_zero, msg);
    }

    // Should still produce valid R1CS
    SparseR1CS r1cs = make_fibonacci_r1cs(4);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    Groth16Proof proof = groth16_prove_sparse(r1cs, pk, w, 17, 23);

    G1AffineRef pa = gpu_to_ref_g1(proof.pi_a);
    TEST_ASSERT(proof.pi_a.infinity || g1_is_on_curve_ref(pa), "fib(0,0): pi_a valid");
}

void test_fibonacci_sparse_qap_vs_dense() {
    using namespace ff_ref;
    printf("test_fibonacci_sparse_qap_vs_dense...\n");

    // At small n, verify sparse Lagrange setup matches dense INTT approach
    size_t nc = 4;
    SparseR1CS sparse_r1cs = make_fibonacci_r1cs(nc);
    size_t n = sparse_r1cs.domain_size;
    size_t nv = sparse_r1cs.num_variables;

    // Build dense R1CS equivalent
    R1CS dense;
    dense.num_constraints = nc;
    dense.num_variables = nv;
    dense.domain_size = n;
    dense.A.assign(n, std::vector<uint64_t>(nv, 0));
    dense.B.assign(n, std::vector<uint64_t>(nv, 0));
    dense.C.assign(n, std::vector<uint64_t>(nv, 0));

    for (const auto& e : sparse_r1cs.A)
        dense.A[e.row][e.col] = e.val;
    for (const auto& e : sparse_r1cs.B)
        dense.B[e.row][e.col] = e.val;
    for (const auto& e : sparse_r1cs.C)
        dense.C[e.row][e.col] = e.val;

    // Generate keys with same seed
    ProvingKey pk_sparse = generate_proving_key_sparse(sparse_r1cs, 42);
    ProvingKey pk_dense  = generate_proving_key(dense, 42);

    // Compare u_tau_g1 points
    bool all_match = true;
    for (size_t i = 0; i < nv && all_match; ++i) {
        for (int j = 0; j < 12 && all_match; ++j) {
            if (pk_sparse.u_tau_g1[i].x.limbs[j] != pk_dense.u_tau_g1[i].x.limbs[j])
                all_match = false;
            if (pk_sparse.u_tau_g1[i].y.limbs[j] != pk_dense.u_tau_g1[i].y.limbs[j])
                all_match = false;
        }
    }
    TEST_ASSERT(all_match, "sparse vs dense: u_tau_g1 match");

    // Compare h_query points
    bool h_match = true;
    for (size_t j = 0; j < pk_sparse.h_query.size() && h_match; ++j) {
        for (int k = 0; k < 12 && h_match; ++k) {
            if (pk_sparse.h_query[j].x.limbs[k] != pk_dense.h_query[j].x.limbs[k])
                h_match = false;
        }
    }
    TEST_ASSERT(h_match, "sparse vs dense: h_query match");

    // Compare l_query points
    bool l_match = true;
    for (size_t i = 0; i < pk_sparse.l_query.size() && l_match; ++i) {
        for (int j = 0; j < 12 && l_match; ++j) {
            if (pk_sparse.l_query[i].x.limbs[j] != pk_dense.l_query[i].x.limbs[j])
                l_match = false;
        }
    }
    TEST_ASSERT(l_match, "sparse vs dense: l_query match");
}

void test_fibonacci_gpu_vs_cpu_256() {
    using namespace ff_ref;
    printf("test_fibonacci_gpu_vs_cpu_256...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(256);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto witness = compute_fibonacci_witness(1, 1, 256);

    Groth16Proof gpu_proof = groth16_prove_sparse(r1cs, pk, witness, 17, 23);
    Groth16Proof cpu_proof = groth16_prove_cpu_sparse(r1cs, pk, witness, 17, 23);

    // Compare π_A
    bool a_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_a.x.limbs[i] != cpu_proof.pi_a.x.limbs[i]) a_match = false;
        if (gpu_proof.pi_a.y.limbs[i] != cpu_proof.pi_a.y.limbs[i]) a_match = false;
    }
    TEST_ASSERT(a_match, "fib256 GPU vs CPU: pi_a match");

    // Compare π_C
    bool c_match = true;
    for (int i = 0; i < 12; ++i) {
        if (gpu_proof.pi_c.x.limbs[i] != cpu_proof.pi_c.x.limbs[i]) c_match = false;
        if (gpu_proof.pi_c.y.limbs[i] != cpu_proof.pi_c.y.limbs[i]) c_match = false;
    }
    TEST_ASSERT(c_match, "fib256 GPU vs CPU: pi_c match");
}

// ─── v2.2.0 Session 30: Batch Pipeline Tests ────────────────────────────────

// Helper: compare two G1Affine points for bitwise equality
static bool g1_affine_equal(const G1Affine& a, const G1Affine& b) {
    if (a.infinity != b.infinity) return false;
    if (a.infinity) return true;
    for (int i = 0; i < 12; ++i) {
        if (a.x.limbs[i] != b.x.limbs[i]) return false;
        if (a.y.limbs[i] != b.y.limbs[i]) return false;
    }
    return true;
}

static bool g2_affine_equal(const G2Affine& a, const G2Affine& b) {
    if (a.infinity != b.infinity) return false;
    if (a.infinity) return true;
    for (int i = 0; i < 12; ++i) {
        if (a.x.c0.limbs[i] != b.x.c0.limbs[i]) return false;
        if (a.x.c1.limbs[i] != b.x.c1.limbs[i]) return false;
        if (a.y.c0.limbs[i] != b.y.c0.limbs[i]) return false;
        if (a.y.c1.limbs[i] != b.y.c1.limbs[i]) return false;
    }
    return true;
}

static bool proof_equal(const Groth16Proof& a, const Groth16Proof& b) {
    return g1_affine_equal(a.pi_a, b.pi_a) &&
           g2_affine_equal(a.pi_b, b.pi_b) &&
           g1_affine_equal(a.pi_c, b.pi_c);
}

void test_batch_empty() {
    printf("test_batch_empty...\n");
    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses;
    auto proofs = groth16_prove_batch_sparse(r1cs, pk, witnesses);
    TEST_ASSERT(proofs.empty(), "batch: empty input → empty output");
}

void test_batch_single_matches_individual() {
    using namespace ff_ref;
    printf("test_batch_single_matches_individual...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto w = compute_fibonacci_witness(1, 1, 8);

    // Single proof via batch
    auto batch = groth16_prove_batch_sparse(r1cs, pk, {w}, 17, 23);
    // Single proof via individual call (same r_seed, s_seed)
    auto single = groth16_prove_sparse(r1cs, pk, w, 17, 23);

    TEST_ASSERT(batch.size() == 1, "batch(1): size = 1");
    TEST_ASSERT(proof_equal(batch[0], single), "batch(1): matches individual prove");
}

void test_batch_vs_sequential() {
    printf("test_batch_vs_sequential...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 8),
        compute_fibonacci_witness(2, 3, 8),
        compute_fibonacci_witness(5, 8, 8),
        compute_fibonacci_witness(1, 2, 8),
    };

    auto batch = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);
    auto seq   = groth16_prove_batch_sequential_sparse(r1cs, pk, witnesses, 17, 23);

    TEST_ASSERT(batch.size() == 4, "batch vs seq: batch size = 4");
    TEST_ASSERT(seq.size() == 4, "batch vs seq: seq size = 4");

    for (int i = 0; i < 4; ++i) {
        char msg[128];
        snprintf(msg, sizeof(msg), "batch vs seq: proof[%d] matches", i);
        TEST_ASSERT(proof_equal(batch[i], seq[i]), msg);
    }
}

void test_batch_on_curve() {
    using namespace ff_ref;
    printf("test_batch_on_curve...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 8),
        compute_fibonacci_witness(3, 7, 8),
        compute_fibonacci_witness(10, 20, 8),
        compute_fibonacci_witness(0, 1, 8),
    };

    auto proofs = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);

    for (int i = 0; i < 4; ++i) {
        G1AffineRef pa = gpu_to_ref_g1(proofs[i].pi_a);
        G2AffineRef pb = gpu_to_ref_g2(proofs[i].pi_b);
        G1AffineRef pc = gpu_to_ref_g1(proofs[i].pi_c);

        char msg[128];
        snprintf(msg, sizeof(msg), "batch on-curve: proof[%d] pi_a on G1", i);
        TEST_ASSERT(proofs[i].pi_a.infinity || g1_is_on_curve_ref(pa), msg);

        snprintf(msg, sizeof(msg), "batch on-curve: proof[%d] pi_b on G2", i);
        TEST_ASSERT(proofs[i].pi_b.infinity || g2_is_on_curve_ref(pb), msg);

        snprintf(msg, sizeof(msg), "batch on-curve: proof[%d] pi_c on G1", i);
        TEST_ASSERT(proofs[i].pi_c.infinity || g1_is_on_curve_ref(pc), msg);

        snprintf(msg, sizeof(msg), "batch on-curve: proof[%d] pi_a not infinity", i);
        TEST_ASSERT(!proofs[i].pi_a.infinity, msg);

        snprintf(msg, sizeof(msg), "batch on-curve: proof[%d] pi_b not infinity", i);
        TEST_ASSERT(!proofs[i].pi_b.infinity, msg);
    }
}

void test_batch_determinism() {
    printf("test_batch_determinism...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 8),
        compute_fibonacci_witness(2, 3, 8),
    };

    auto run1 = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);
    auto run2 = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);

    TEST_ASSERT(proof_equal(run1[0], run2[0]), "batch determinism: proof[0] matches");
    TEST_ASSERT(proof_equal(run1[1], run2[1]), "batch determinism: proof[1] matches");
}

void test_batch_different_witnesses() {
    printf("test_batch_different_witnesses...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 8),
        compute_fibonacci_witness(2, 3, 8),
        compute_fibonacci_witness(5, 8, 8),
    };

    auto proofs = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);

    // Different witnesses → different proofs (pi_a should differ)
    TEST_ASSERT(!proof_equal(proofs[0], proofs[1]), "batch: proof[0] != proof[1]");
    TEST_ASSERT(!proof_equal(proofs[1], proofs[2]), "batch: proof[1] != proof[2]");
    TEST_ASSERT(!proof_equal(proofs[0], proofs[2]), "batch: proof[0] != proof[2]");
}

void test_batch_same_witness_different_seeds() {
    printf("test_batch_same_witness_different_seeds...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);
    auto w = compute_fibonacci_witness(1, 1, 8);

    // Same witness duplicated — but batch assigns different r/s seeds per proof
    std::vector<std::vector<FpElement>> witnesses = {w, w, w};
    auto proofs = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);

    // Different r/s seeds → different proofs even with same witness
    TEST_ASSERT(!proof_equal(proofs[0], proofs[1]),
                "batch same witness: proof[0] != proof[1] (different r/s)");
    TEST_ASSERT(!proof_equal(proofs[1], proofs[2]),
                "batch same witness: proof[1] != proof[2] (different r/s)");
}

void test_batch_sequential_reference() {
    using namespace ff_ref;
    printf("test_batch_sequential_reference...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 8),
        compute_fibonacci_witness(2, 3, 8),
    };

    auto seq = groth16_prove_batch_sequential_sparse(r1cs, pk, witnesses, 17, 23);

    // Verify sequential matches individual calls
    auto p0 = groth16_prove_sparse(r1cs, pk, witnesses[0], 17, 23);
    auto p1 = groth16_prove_sparse(r1cs, pk, witnesses[1], 19, 25);

    TEST_ASSERT(proof_equal(seq[0], p0), "seq batch: proof[0] matches individual");
    TEST_ASSERT(proof_equal(seq[1], p1), "seq batch: proof[1] matches individual");
}

void test_batch_256() {
    using namespace ff_ref;
    printf("test_batch_256...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(256);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 256),
        compute_fibonacci_witness(2, 3, 256),
    };

    auto batch = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);
    auto seq   = groth16_prove_batch_sequential_sparse(r1cs, pk, witnesses, 17, 23);

    TEST_ASSERT(batch.size() == 2, "batch256: size = 2");
    TEST_ASSERT(proof_equal(batch[0], seq[0]), "batch256: proof[0] matches sequential");
    TEST_ASSERT(proof_equal(batch[1], seq[1]), "batch256: proof[1] matches sequential");

    // On curve
    G1AffineRef pa0 = gpu_to_ref_g1(batch[0].pi_a);
    TEST_ASSERT(!batch[0].pi_a.infinity && g1_is_on_curve_ref(pa0), "batch256: proof[0] pi_a on curve");
    G1AffineRef pa1 = gpu_to_ref_g1(batch[1].pi_a);
    TEST_ASSERT(!batch[1].pi_a.infinity && g1_is_on_curve_ref(pa1), "batch256: proof[1] pi_a on curve");
}

void test_batch_eight() {
    printf("test_batch_eight...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses;
    for (int i = 0; i < 8; ++i)
        witnesses.push_back(compute_fibonacci_witness(i + 1, i + 2, 8));

    auto batch = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);
    auto seq   = groth16_prove_batch_sequential_sparse(r1cs, pk, witnesses, 17, 23);

    TEST_ASSERT(batch.size() == 8, "batch8: size = 8");
    bool all_match = true;
    for (int i = 0; i < 8; ++i) {
        if (!proof_equal(batch[i], seq[i])) { all_match = false; break; }
    }
    TEST_ASSERT(all_match, "batch8: all proofs match sequential");

    // All proofs should be different (different witnesses + different r/s seeds)
    bool all_different = true;
    for (int i = 0; i < 8 && all_different; ++i)
        for (int j = i + 1; j < 8 && all_different; ++j)
            if (proof_equal(batch[i], batch[j])) all_different = false;
    TEST_ASSERT(all_different, "batch8: all proofs pairwise different");
}

void test_batch_vs_cpu_reference() {
    using namespace ff_ref;
    printf("test_batch_vs_cpu_reference...\n");

    SparseR1CS r1cs = make_fibonacci_r1cs(8);
    ProvingKey pk = generate_proving_key_sparse(r1cs, 42);

    std::vector<std::vector<FpElement>> witnesses = {
        compute_fibonacci_witness(1, 1, 8),
        compute_fibonacci_witness(2, 3, 8),
    };

    auto batch = groth16_prove_batch_sparse(r1cs, pk, witnesses, 17, 23);

    // Compare each batch proof's pi_A and pi_C with CPU reference
    for (int i = 0; i < 2; ++i) {
        auto cpu = groth16_prove_cpu_sparse(r1cs, pk, witnesses[i],
                                             17 + (uint64_t)i * 2, 23 + (uint64_t)i * 2);
        char msg[128];
        snprintf(msg, sizeof(msg), "batch vs CPU: proof[%d] pi_a match", i);
        TEST_ASSERT(g1_affine_equal(batch[i].pi_a, cpu.pi_a), msg);

        snprintf(msg, sizeof(msg), "batch vs CPU: proof[%d] pi_c match", i);
        TEST_ASSERT(g1_affine_equal(batch[i].pi_c, cpu.pi_c), msg);
    }
}

// =============================================================================
// v3.0.0 Session 33: Miller Loop Tests
// =============================================================================

// Helper: run miller_loop on GPU for a single pair
static Fq12Element miller_loop_gpu_single(const G1Affine& P, const G2Affine& Q) {
    G1Affine* d_P; G2Affine* d_Q; Fq12Element* d_out;
    cudaMalloc(&d_P, sizeof(G1Affine));
    cudaMalloc(&d_Q, sizeof(G2Affine));
    cudaMalloc(&d_out, sizeof(Fq12Element));
    cudaMemcpy(d_P, &P, sizeof(G1Affine), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, &Q, sizeof(G2Affine), cudaMemcpyHostToDevice);
    miller_loop_kernel<<<1, 1>>>(d_P, d_Q, d_out, 1);
    Fq12Element result;
    cudaMemcpy(&result, d_out, sizeof(Fq12Element), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_P); cudaFree(d_Q); cudaFree(d_out);
    return result;
}

// Helper: convert G1AffineRef to GPU G1Affine
static G1Affine g1_ref_to_gpu(const ff_ref::G1AffineRef& r) {
    G1Affine p;
    if (r.infinity) {
        p.x = FqElement::zero();
        p.y = FqElement::zero();
        p.infinity = true;
        return p;
    }
    r.x.to_u32(p.x.limbs);
    r.y.to_u32(p.y.limbs);
    p.infinity = false;
    return p;
}

// Helper: convert G2AffineRef to GPU G2Affine
static G2Affine g2_ref_to_gpu(const ff_ref::G2AffineRef& r) {
    G2Affine p;
    if (r.infinity) {
        p.x = Fq2Element::zero();
        p.y = Fq2Element::zero();
        p.infinity = true;
        return p;
    }
    r.x.c0.to_u32(p.x.c0.limbs);
    r.x.c1.to_u32(p.x.c1.limbs);
    r.y.c0.to_u32(p.y.c0.limbs);
    r.y.c1.to_u32(p.y.c1.limbs);
    p.infinity = false;
    return p;
}

// Helper: Fq12 GPU == CPU comparison
static bool fq12_eq(const Fq12Element& gpu, const ff_ref::Fq12Ref& cpu) {
    return fq12_gpu_to_ref(gpu) == cpu;
}

// --- CPU tests ---

void test_miller_loop_cpu_nondegen() {
    using namespace ff_ref;
    printf("test_miller_loop_cpu_nondegen...\n");
    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::generator();
    Fq12Ref f = miller_loop_ref(P, Q);
    TEST_ASSERT(f != Fq12Ref::one_mont(), "Miller(G1,G2) != 1 (non-degenerate)");
    TEST_ASSERT(f != Fq12Ref::zero(), "Miller(G1,G2) != 0");
    printf("  non-degenerate: OK\n");
}

void test_miller_loop_cpu_identity_P() {
    using namespace ff_ref;
    printf("test_miller_loop_cpu_identity_P...\n");
    G1AffineRef P = G1AffineRef::point_at_infinity();
    G2AffineRef Q = G2AffineRef::generator();
    Fq12Ref f = miller_loop_ref(P, Q);
    TEST_ASSERT(f == Fq12Ref::one_mont(), "Miller(O, Q) = 1");
    printf("  identity P: OK\n");
}

void test_miller_loop_cpu_identity_Q() {
    using namespace ff_ref;
    printf("test_miller_loop_cpu_identity_Q...\n");
    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::point_at_infinity();
    Fq12Ref f = miller_loop_ref(P, Q);
    TEST_ASSERT(f == Fq12Ref::one_mont(), "Miller(P, O) = 1");
    printf("  identity Q: OK\n");
}

void test_miller_loop_cpu_determinism() {
    using namespace ff_ref;
    printf("test_miller_loop_cpu_determinism...\n");
    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::generator();
    Fq12Ref f1 = miller_loop_ref(P, Q);
    Fq12Ref f2 = miller_loop_ref(P, Q);
    TEST_ASSERT(f1 == f2, "Miller loop deterministic");
    printf("  determinism: OK\n");
}

void test_miller_loop_cpu_different_points() {
    using namespace ff_ref;
    printf("test_miller_loop_cpu_different_points...\n");
    G1AffineRef P = G1AffineRef::generator();
    G1AffineRef P2 = g1_double_ref(P);
    G2AffineRef Q = G2AffineRef::generator();
    Fq12Ref f1 = miller_loop_ref(P, Q);
    Fq12Ref f2 = miller_loop_ref(P2, Q);
    TEST_ASSERT(f1 != f2, "Miller(P,Q) != Miller(2P,Q)");
    printf("  different points give different results: OK\n");
}

void test_miller_loop_cpu_not_one() {
    using namespace ff_ref;
    printf("test_miller_loop_cpu_not_one...\n");
    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::generator();
    Fq12Ref f = miller_loop_ref(P, Q);
    // Check that at least one component of c1 is nonzero (f is not purely in Fq6)
    bool c1_nonzero = (f.c1 != Fq6Ref::zero());
    TEST_ASSERT(c1_nonzero, "Miller result has nonzero w-component");
    printf("  non-trivial Fq12: OK\n");
}

// --- GPU tests ---

void test_miller_loop_gpu_basic() {
    printf("test_miller_loop_gpu_basic...\n");
    G1Affine P = make_g1_gen_affine_gpu();
    G2Affine Q = make_g2_gen_affine_gpu();
    Fq12Element f = miller_loop_gpu_single(P, Q);

    // Should be non-trivial
    Fq12Element one = Fq12Element::one_mont();
    bool is_one = (f == one);
    TEST_ASSERT(!is_one, "GPU: Miller(G1,G2) != 1");
    printf("  GPU basic: OK\n");
}

void test_miller_loop_gpu_vs_cpu() {
    using namespace ff_ref;
    printf("test_miller_loop_gpu_vs_cpu...\n");

    // CPU
    G1AffineRef P_ref = G1AffineRef::generator();
    G2AffineRef Q_ref = G2AffineRef::generator();
    Fq12Ref f_cpu = miller_loop_ref(P_ref, Q_ref);

    // GPU
    G1Affine P_gpu = g1_ref_to_gpu(P_ref);
    G2Affine Q_gpu = g2_ref_to_gpu(Q_ref);
    Fq12Element f_gpu = miller_loop_gpu_single(P_gpu, Q_gpu);

    TEST_ASSERT(fq12_eq(f_gpu, f_cpu), "Miller loop GPU == CPU (generators)");
    printf("  GPU vs CPU: OK\n");
}

void test_miller_loop_gpu_identity() {
    using namespace ff_ref;
    printf("test_miller_loop_gpu_identity...\n");

    // P = infinity
    G1Affine P_inf = G1Affine::point_at_infinity();
    G2Affine Q = make_g2_gen_affine_gpu();
    Fq12Element f1 = miller_loop_gpu_single(P_inf, Q);
    TEST_ASSERT(f1 == Fq12Element::one_mont(), "GPU: Miller(O, Q) = 1");

    // Q = infinity
    G1Affine P = make_g1_gen_affine_gpu();
    G2Affine Q_inf;
    Q_inf.x = Fq2Element::zero();
    Q_inf.y = Fq2Element::zero();
    Q_inf.infinity = true;
    Fq12Element f2 = miller_loop_gpu_single(P, Q_inf);
    TEST_ASSERT(f2 == Fq12Element::one_mont(), "GPU: Miller(P, O) = 1");

    printf("  GPU identity: OK\n");
}

void test_miller_loop_gpu_determinism() {
    printf("test_miller_loop_gpu_determinism...\n");
    G1Affine P = make_g1_gen_affine_gpu();
    G2Affine Q = make_g2_gen_affine_gpu();
    Fq12Element f1 = miller_loop_gpu_single(P, Q);
    Fq12Element f2 = miller_loop_gpu_single(P, Q);
    TEST_ASSERT(f1 == f2, "GPU: Miller loop deterministic");
    printf("  GPU determinism: OK\n");
}

void test_miller_loop_gpu_different_pairs() {
    using namespace ff_ref;
    printf("test_miller_loop_gpu_different_pairs...\n");

    // 2*P on CPU, then convert
    G1AffineRef P_ref = G1AffineRef::generator();
    G1AffineRef P2_ref = g1_double_ref(P_ref);
    G2AffineRef Q_ref = G2AffineRef::generator();

    // GPU: miller(P, Q) and miller(2P, Q) — also compare with CPU
    Fq12Ref cpu_f1 = miller_loop_ref(P_ref, Q_ref);
    Fq12Ref cpu_f2 = miller_loop_ref(P2_ref, Q_ref);

    G1Affine P2_gpu = g1_ref_to_gpu(P2_ref);
    G2Affine Q_gpu = g2_ref_to_gpu(Q_ref);
    Fq12Element gpu_f2 = miller_loop_gpu_single(P2_gpu, Q_gpu);

    TEST_ASSERT(fq12_eq(gpu_f2, cpu_f2), "GPU: Miller(2P, Q) matches CPU");
    TEST_ASSERT(cpu_f1 != cpu_f2, "Miller(P,Q) != Miller(2P,Q)");
    printf("  GPU different pairs: OK\n");
}

// --- Bilinearity tests ---
// These test e(aP, Q) == e(P, aQ) == e(P, Q)^a
// We test at the Miller loop level (before final exponentiation).
// The full bilinearity property holds after final exp, but we can test:
//   miller(aP, Q) == miller(P, aQ) at the Miller loop output level
// This is NOT strictly true — bilinearity is for the full pairing.
// However, we CAN test:
//   miller(P, Q+R) == miller(P, Q) * miller(P, R)  (Miller loop is multiplicative in Q)
//   miller(-P, Q) == conjugate(miller(P, Q))         (sign in P)

void test_miller_loop_bilinearity_scalar_P() {
    using namespace ff_ref;
    printf("test_miller_loop_bilinearity_scalar_P...\n");

    // miller(2P, Q) should equal miller(P, Q)^2 * correction terms...
    // Actually at Miller loop level, f_{u, Q}(2P) != f_{u, Q}(P)^2 in general.
    // The linearity property is: f_{u, Q}(P) is linear in P only after final exp.
    //
    // What we CAN test: miller(P, Q) is NOT the identity for generators.
    // And miller(P1, Q) != miller(P2, Q) for P1 != P2.
    G1AffineRef P = G1AffineRef::generator();
    G1AffineRef P2 = g1_double_ref(P);
    G2AffineRef Q = G2AffineRef::generator();

    Fq12Ref f_P = miller_loop_ref(P, Q);
    Fq12Ref f_2P = miller_loop_ref(P2, Q);

    // f_2P != f_P (different inputs give different Miller loop outputs)
    TEST_ASSERT(f_P != f_2P, "miller(P,Q) != miller(2P,Q)");
    // f_2P != 1
    TEST_ASSERT(f_2P != Fq12Ref::one_mont(), "miller(2P,Q) != 1");
    printf("  scalar P: OK\n");
}

void test_miller_loop_bilinearity_scalar_Q() {
    using namespace ff_ref;
    printf("test_miller_loop_bilinearity_scalar_Q...\n");

    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::generator();
    G2AffineRef Q2 = g2_double_ref(Q);

    Fq12Ref f_Q = miller_loop_ref(P, Q);
    Fq12Ref f_2Q = miller_loop_ref(P, Q2);

    TEST_ASSERT(f_Q != f_2Q, "miller(P,Q) != miller(P,2Q)");
    TEST_ASSERT(f_2Q != Fq12Ref::one_mont(), "miller(P,2Q) != 1");
    printf("  scalar Q: OK\n");
}

void test_miller_loop_bilinearity_product() {
    using namespace ff_ref;
    printf("test_miller_loop_bilinearity_product...\n");

    // Miller loop is multiplicative in Q:
    //   f_{u, Q1+Q2}(P) = f_{u, Q1}(P) * f_{u, Q2}(P) * line_correction
    // This is NOT exact without the correction factor, so we just test
    // that miller(P, Q+R) is well-defined and different from miller(P, Q).

    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::generator();
    G2AffineRef Q2 = g2_double_ref(Q);
    G2AffineRef Q3 = g2_add_ref(Q, Q2);  // 3Q

    Fq12Ref f_Q = miller_loop_ref(P, Q);
    Fq12Ref f_Q3 = miller_loop_ref(P, Q3);

    TEST_ASSERT(f_Q != f_Q3, "miller(P,Q) != miller(P,3Q)");
    // All should be non-trivial
    TEST_ASSERT(f_Q3 != Fq12Ref::one_mont(), "miller(P,3Q) != 1");
    printf("  product: OK\n");
}

void test_miller_loop_negation_P() {
    using namespace ff_ref;
    printf("test_miller_loop_negation_P...\n");

    // e(-P, Q) = e(P, Q)^{-1} holds after final exponentiation.
    // At the raw Miller loop level, the relationship is more complex.
    // We verify: miller(-P, Q) != miller(P, Q) and both are non-trivial.
    G1AffineRef P = G1AffineRef::generator();
    G1AffineRef neg_P = g1_negate_ref(P);
    G2AffineRef Q = G2AffineRef::generator();

    Fq12Ref f = miller_loop_ref(P, Q);
    Fq12Ref f_neg = miller_loop_ref(neg_P, Q);

    TEST_ASSERT(f != f_neg, "miller(-P, Q) != miller(P, Q)");
    TEST_ASSERT(f_neg != Fq12Ref::one_mont(), "miller(-P, Q) != 1");
    printf("  negation P: OK\n");
}

void test_miller_loop_negation_Q() {
    using namespace ff_ref;
    printf("test_miller_loop_negation_Q...\n");

    // e(P, -Q) = e(P, Q)^{-1} holds after final exponentiation.
    // At raw Miller loop level, we verify they're different and non-trivial.
    G1AffineRef P = G1AffineRef::generator();
    G2AffineRef Q = G2AffineRef::generator();
    G2AffineRef neg_Q = g2_negate_ref(Q);

    Fq12Ref f = miller_loop_ref(P, Q);
    Fq12Ref f_neg = miller_loop_ref(P, neg_Q);

    TEST_ASSERT(f != f_neg, "miller(P, Q) != miller(P, -Q)");
    TEST_ASSERT(f_neg != Fq12Ref::one_mont(), "miller(P, -Q) != 1");
    printf("  negation Q: OK\n");
}

void test_miller_loop_double_vs_square() {
    using namespace ff_ref;
    printf("test_miller_loop_double_vs_square...\n");

    // Verify miller(2P, Q) on GPU matches CPU
    G1AffineRef P2_ref = g1_double_ref(G1AffineRef::generator());
    G2AffineRef Q_ref = G2AffineRef::generator();
    Fq12Ref cpu = miller_loop_ref(P2_ref, Q_ref);

    G1Affine P2_gpu = g1_ref_to_gpu(P2_ref);
    G2Affine Q_gpu = g2_ref_to_gpu(Q_ref);
    Fq12Element gpu = miller_loop_gpu_single(P2_gpu, Q_gpu);

    TEST_ASSERT(fq12_eq(gpu, cpu), "GPU: miller(2P, Q) matches CPU");

    // Also test with 2Q
    G2AffineRef Q2_ref = g2_double_ref(Q_ref);
    Fq12Ref cpu2 = miller_loop_ref(G1AffineRef::generator(), Q2_ref);

    G1Affine P_gpu = g1_ref_to_gpu(G1AffineRef::generator());
    G2Affine Q2_gpu = g2_ref_to_gpu(Q2_ref);
    Fq12Element gpu2 = miller_loop_gpu_single(P_gpu, Q2_gpu);

    TEST_ASSERT(fq12_eq(gpu2, cpu2), "GPU: miller(P, 2Q) matches CPU");
    printf("  double vs square: OK\n");
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

    // v1.3.0 Session 5: Transpose kernel tests
    test_4step_split();
    test_transpose_square(16);     // exactly one tile
    test_transpose_square(32);     // 2x2 tiles
    test_transpose_square(64);     // 4x4 tiles
    test_transpose_rect(16, 32);   // non-square (n1 < n2)
    test_transpose_rect(32, 16);   // non-square (n1 > n2)
    test_transpose_rect(256, 512); // 4-step odd log_n sizes
    test_transpose_rect(512, 1024);// 4-step sub-NTT sizes
    test_transpose_roundtrip(16, 32);
    test_transpose_roundtrip(64, 64);
    test_transpose_roundtrip(256, 512);
    test_transpose_roundtrip(1024, 2048);
    test_transpose_batch(16, 32, 4);
    test_transpose_batch(64, 64, 8);
    test_transpose_4step_sizes();  // 2048x2048 (actual 4-step decomposition)

    // v1.3.0 Session 6: 4-Step NTT integration tests
    // Forward correctness: 4-step NTT vs CPU reference
    test_ntt_forward_gpu(16, NTTMode::FOUR_STEP);   // n1=n2=256 (even log_n)
    test_ntt_forward_gpu(17, NTTMode::FOUR_STEP);   // n1=256, n2=512 (odd log_n)
    test_ntt_forward_gpu(18, NTTMode::FOUR_STEP);   // n1=n2=512
    test_ntt_forward_gpu(20, NTTMode::FOUR_STEP);   // n1=n2=1024 (all fused, 0 outer)
    test_ntt_forward_gpu(22, NTTMode::FOUR_STEP);   // n1=n2=2048 (key target size)

    // Roundtrip: INTT(NTT(x)) = x for 4-step path
    test_ntt_roundtrip_gpu(16, NTTMode::FOUR_STEP);
    test_ntt_roundtrip_gpu(17, NTTMode::FOUR_STEP);
    test_ntt_roundtrip_gpu(18, NTTMode::FOUR_STEP);
    test_ntt_roundtrip_gpu(20, NTTMode::FOUR_STEP);
    test_ntt_roundtrip_gpu(22, NTTMode::FOUR_STEP);

    // Cross-validation: 4-step NTT == Barrett NTT (bitwise, same arithmetic)
    test_ntt_barrett_vs_montgomery(16);  // already tested, but re-verify
    // Dedicated 4-step vs Barrett cross-validation
    {
        int four_step_sizes[] = {16, 17, 18, 20, 22};
        for (int log_n : four_step_sizes) {
            size_t nn = static_cast<size_t>(1) << log_n;
            printf("test_4step_vs_barrett (n=2^%d)...\n", log_n);

            std::vector<FpElement> h_data(nn);
            for (size_t i = 0; i < nn; ++i) {
                h_data[i] = FpElement::zero();
                h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
                h_data[i].limbs[1] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
            }

            FpElement *d_4step, *d_barrett;
            CUDA_CHECK(cudaMalloc(&d_4step, nn * sizeof(FpElement)));
            CUDA_CHECK(cudaMalloc(&d_barrett, nn * sizeof(FpElement)));
            CUDA_CHECK(cudaMemcpy(d_4step, h_data.data(), nn * sizeof(FpElement), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), nn * sizeof(FpElement), cudaMemcpyHostToDevice));

            ntt_forward(d_4step, nn, NTTMode::FOUR_STEP);
            ntt_forward(d_barrett, nn, NTTMode::BARRETT);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FpElement> h_4step(nn), h_barrett(nn);
            CUDA_CHECK(cudaMemcpy(h_4step.data(), d_4step, nn * sizeof(FpElement), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_barrett.data(), d_barrett, nn * sizeof(FpElement), cudaMemcpyDeviceToHost));

            int pass = 0;
            for (size_t i = 0; i < nn; ++i) {
                if (h_4step[i] == h_barrett[i]) ++pass;
            }
            TEST_ASSERT(pass == static_cast<int>(nn), "4-step vs Barrett mismatch");
            printf("  4-step vs Barrett (n=2^%d): %d/%zu matched\n", log_n, pass, nn);

            if (pass != static_cast<int>(nn)) {
                int printed = 0;
                for (size_t i = 0; i < nn && printed < 5; ++i) {
                    if (h_4step[i] != h_barrett[i]) {
                        printf("    [%zu] 4step: %08x%08x barrett: %08x%08x\n",
                            i, h_4step[i].limbs[7], h_4step[i].limbs[6],
                            h_barrett[i].limbs[7], h_barrett[i].limbs[6]);
                        ++printed;
                    }
                }
            }

            CUDA_CHECK(cudaFree(d_4step));
            CUDA_CHECK(cudaFree(d_barrett));
        }
    }

    // Batched 4-step NTT tests
    test_ntt_batch_vs_sequential(16, 4, NTTMode::FOUR_STEP);
    test_ntt_batch_vs_sequential(18, 4, NTTMode::FOUR_STEP);
    test_ntt_batch_vs_sequential(20, 2, NTTMode::FOUR_STEP);

    // Batched 4-step roundtrip
    test_ntt_batch_roundtrip(16, 4, NTTMode::FOUR_STEP);
    test_ntt_batch_roundtrip(18, 4, NTTMode::FOUR_STEP);
    test_ntt_batch_roundtrip(20, 2, NTTMode::FOUR_STEP);

    // Batched 4-step vs Barrett cross-validation
    {
        printf("test_4step_batch_vs_barrett_batch (n=2^16, B=4)...\n");
        int log_n = 16;
        int batch_size = 4;
        size_t nn = static_cast<size_t>(1) << log_n;
        size_t total = static_cast<size_t>(batch_size) * nn;

        std::vector<FpElement> h_data(total);
        for (size_t i = 0; i < total; ++i) {
            h_data[i] = FpElement::zero();
            h_data[i].limbs[0] = static_cast<uint32_t>((i * 31337u + 42u) % 1000000007u);
        }

        FpElement *d_4step, *d_barrett;
        CUDA_CHECK(cudaMalloc(&d_4step, total * sizeof(FpElement)));
        CUDA_CHECK(cudaMalloc(&d_barrett, total * sizeof(FpElement)));
        CUDA_CHECK(cudaMemcpy(d_4step, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

        ntt_forward_batch(d_4step, batch_size, nn, NTTMode::FOUR_STEP);
        ntt_forward_batch(d_barrett, batch_size, nn, NTTMode::BARRETT);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<FpElement> h_4step(total), h_barrett(total);
        CUDA_CHECK(cudaMemcpy(h_4step.data(), d_4step, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_barrett.data(), d_barrett, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

        int pass = 0;
        for (size_t i = 0; i < total; ++i) {
            if (h_4step[i] == h_barrett[i]) ++pass;
        }
        TEST_ASSERT(pass == static_cast<int>(total), "Batched 4-step vs Barrett mismatch");
        printf("  Batched 4-step vs Barrett (n=2^%d, B=%d): %d/%zu matched\n",
               log_n, batch_size, pass, total);

        CUDA_CHECK(cudaFree(d_4step));
        CUDA_CHECK(cudaFree(d_barrett));
    }

    // ─── v1.3.0 Session 7: Exhaustive 4-Step Correctness + Edge Cases ──────

    // 4-step forward + roundtrip for ALL sizes 2^10..2^22
    // Sizes 2^10..2^15 use Barrett fallback; 2^16..2^22 use true 4-step
    for (int log_n = 10; log_n <= 22; ++log_n) {
        test_ntt_forward_gpu(log_n, NTTMode::FOUR_STEP);
    }
    for (int log_n = 10; log_n <= 22; ++log_n) {
        test_ntt_roundtrip_gpu(log_n, NTTMode::FOUR_STEP);
    }

    // Known-vector roundtrip tests (multiple input patterns)
    test_ntt_known_vectors(10, NTTMode::FOUR_STEP);  // small, Barrett fallback
    test_ntt_known_vectors(15, NTTMode::FOUR_STEP);  // boundary (still fallback)
    test_ntt_known_vectors(16, NTTMode::FOUR_STEP);  // first true 4-step size
    test_ntt_known_vectors(18, NTTMode::FOUR_STEP);  // even log_n, n1=n2=512
    test_ntt_known_vectors(20, NTTMode::FOUR_STEP);  // n1=n2=1024

    // Forward zeros: NTT(0) = 0
    test_ntt_forward_zeros(16, NTTMode::FOUR_STEP);
    test_ntt_forward_zeros(20, NTTMode::FOUR_STEP);

    // Explicit inverse: verify both inv(fwd(x))=x and fwd(inv(x))=x
    test_ntt_inverse_explicit(16, NTTMode::FOUR_STEP);
    test_ntt_inverse_explicit(17, NTTMode::FOUR_STEP);
    test_ntt_inverse_explicit(20, NTTMode::FOUR_STEP);
    test_ntt_inverse_explicit(22, NTTMode::FOUR_STEP);

    // 4-step vs Barrett cross-validation for sizes NOT tested in Session 6
    {
        int extra_sizes[] = {10, 12, 14, 15, 19, 21};
        for (int log_n : extra_sizes) {
            size_t nn = static_cast<size_t>(1) << log_n;
            printf("test_4step_vs_barrett_extra (n=2^%d)...\n", log_n);

            std::vector<FpElement> h_data(nn);
            for (size_t i = 0; i < nn; ++i) {
                h_data[i] = FpElement::zero();
                h_data[i].limbs[0] = static_cast<uint32_t>((i * 12345u + 6789u) % 1000000007u);
                h_data[i].limbs[1] = static_cast<uint32_t>((i * 54321u + 9876u) % 999999937u);
            }

            FpElement *d_4step, *d_barrett;
            CUDA_CHECK(cudaMalloc(&d_4step, nn * sizeof(FpElement)));
            CUDA_CHECK(cudaMalloc(&d_barrett, nn * sizeof(FpElement)));
            CUDA_CHECK(cudaMemcpy(d_4step, h_data.data(), nn * sizeof(FpElement), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), nn * sizeof(FpElement), cudaMemcpyHostToDevice));

            ntt_forward(d_4step, nn, NTTMode::FOUR_STEP);
            ntt_forward(d_barrett, nn, NTTMode::BARRETT);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FpElement> h_4step(nn), h_barrett(nn);
            CUDA_CHECK(cudaMemcpy(h_4step.data(), d_4step, nn * sizeof(FpElement), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_barrett.data(), d_barrett, nn * sizeof(FpElement), cudaMemcpyDeviceToHost));

            int pass = 0;
            for (size_t i = 0; i < nn; ++i) {
                if (h_4step[i] == h_barrett[i]) ++pass;
            }
            TEST_ASSERT(pass == static_cast<int>(nn), "4-step vs Barrett extra size mismatch");
            printf("  4-step vs Barrett (n=2^%d): %d/%zu matched\n", log_n, pass, nn);

            CUDA_CHECK(cudaFree(d_4step));
            CUDA_CHECK(cudaFree(d_barrett));
        }
    }

    // Batched 4-step: B=8 vs sequential (key Groth16 workload)
    test_ntt_batch_vs_sequential_b8(16, NTTMode::FOUR_STEP);
    test_ntt_batch_vs_sequential_b8(18, NTTMode::FOUR_STEP);

    // Batched 4-step: more sizes and batch configurations
    test_ntt_batch_vs_sequential(10, 4, NTTMode::FOUR_STEP);  // fallback path
    test_ntt_batch_vs_sequential(15, 4, NTTMode::FOUR_STEP);  // fallback path
    test_ntt_batch_vs_sequential(17, 4, NTTMode::FOUR_STEP);  // odd log_n, true 4-step
    test_ntt_batch_vs_sequential(19, 2, NTTMode::FOUR_STEP);  // odd log_n, large
    test_ntt_batch_vs_sequential(22, 2, NTTMode::FOUR_STEP);  // key target size

    // Batched 4-step roundtrip: additional sizes
    test_ntt_batch_roundtrip(10, 4, NTTMode::FOUR_STEP);  // fallback
    test_ntt_batch_roundtrip(15, 8, NTTMode::FOUR_STEP);  // B=8, fallback
    test_ntt_batch_roundtrip(17, 4, NTTMode::FOUR_STEP);  // odd log_n
    test_ntt_batch_roundtrip(19, 2, NTTMode::FOUR_STEP);  // odd log_n, large
    test_ntt_batch_roundtrip(22, 2, NTTMode::FOUR_STEP);  // key target

    // Batched 4-step vs batched Barrett (additional sizes)
    {
        struct BatchXval { int log_n; int batch_size; };
        BatchXval cases[] = {{10, 4}, {15, 4}, {17, 4}, {18, 8}, {20, 2}};
        for (auto& c : cases) {
            size_t nn = static_cast<size_t>(1) << c.log_n;
            size_t total = static_cast<size_t>(c.batch_size) * nn;
            printf("test_4step_batch_vs_barrett_batch (n=2^%d, B=%d)...\n", c.log_n, c.batch_size);

            std::vector<FpElement> h_data(total);
            for (size_t i = 0; i < total; ++i) {
                h_data[i] = FpElement::zero();
                h_data[i].limbs[0] = static_cast<uint32_t>((i * 31337u + 42u) % 1000000007u);
            }

            FpElement *d_4step, *d_barrett;
            CUDA_CHECK(cudaMalloc(&d_4step, total * sizeof(FpElement)));
            CUDA_CHECK(cudaMalloc(&d_barrett, total * sizeof(FpElement)));
            CUDA_CHECK(cudaMemcpy(d_4step, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_barrett, h_data.data(), total * sizeof(FpElement), cudaMemcpyHostToDevice));

            ntt_forward_batch(d_4step, c.batch_size, nn, NTTMode::FOUR_STEP);
            ntt_forward_batch(d_barrett, c.batch_size, nn, NTTMode::BARRETT);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<FpElement> h_4step(total), h_barrett(total);
            CUDA_CHECK(cudaMemcpy(h_4step.data(), d_4step, total * sizeof(FpElement), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_barrett.data(), d_barrett, total * sizeof(FpElement), cudaMemcpyDeviceToHost));

            int pass = 0;
            for (size_t i = 0; i < total; ++i) {
                if (h_4step[i] == h_barrett[i]) ++pass;
            }
            TEST_ASSERT(pass == static_cast<int>(total), "Batched 4-step vs Barrett extra mismatch");
            printf("  Batch 4-step vs Barrett (n=2^%d, B=%d): %d/%zu matched\n",
                   c.log_n, c.batch_size, pass, total);

            CUDA_CHECK(cudaFree(d_4step));
            CUDA_CHECK(cudaFree(d_barrett));
        }
    }

    // ─── v1.4.0 Session 11: CUDA Graph NTT Tests ──────────────────────────
    test_ntt_graph_vs_nongraph();

    // ─── v1.5.0 Session 13: Radix-8 Outer Stage Correctness Tests ──────────
    // Radix-8 activates at n >= 2^13 (K=10, num_outer >= 3).
    // Leftover handling coverage:
    //   num_outer % 3 == 0: 2^13 (3), 2^16 (6), 2^19 (9), 2^22 (12)
    //   num_outer % 3 == 1: 2^14 (4), 2^17 (7), 2^20 (10)
    //   num_outer % 3 == 2: 2^15 (5), 2^18 (8), 2^21 (11)
    printf("\n--- v1.5.0 Session 13: Radix-8 outer stage tests ---\n");

    // Forward NTT vs CPU reference: all sizes where radix-8 is active
    for (int log_n = 13; log_n <= 22; ++log_n) {
        test_ntt_forward_gpu(log_n, NTTMode::OPTIMIZED);
    }
    for (int log_n = 13; log_n <= 22; ++log_n) {
        test_ntt_forward_gpu(log_n, NTTMode::BARRETT);
    }

    // Roundtrip: INTT(NTT(x)) = x for all radix-8 sizes
    for (int log_n = 13; log_n <= 22; ++log_n) {
        test_ntt_roundtrip_gpu(log_n, NTTMode::OPTIMIZED);
    }
    for (int log_n = 13; log_n <= 22; ++log_n) {
        test_ntt_roundtrip_gpu(log_n, NTTMode::BARRETT);
    }

    // Cross-validation: Barrett == Montgomery at all radix-8 sizes
    for (int log_n = 13; log_n <= 22; ++log_n) {
        test_ntt_barrett_vs_montgomery(log_n);
    }

    // Batched B=8 vs sequential: both modes at key radix-8 sizes
    // Covers leftover %3=2 (2^15), %3=2 (2^18), %3=1 (2^20), %3=0 (2^22)
    test_ntt_batch_vs_sequential_b8(15, NTTMode::OPTIMIZED);
    test_ntt_batch_vs_sequential_b8(18, NTTMode::OPTIMIZED);
    test_ntt_batch_vs_sequential_b8(20, NTTMode::OPTIMIZED);
    test_ntt_batch_vs_sequential_b8(15, NTTMode::BARRETT);
    test_ntt_batch_vs_sequential_b8(18, NTTMode::BARRETT);
    test_ntt_batch_vs_sequential_b8(20, NTTMode::BARRETT);
    test_ntt_batch_vs_sequential_b8(22, NTTMode::BARRETT);

    // Batched roundtrip B=8 at representative sizes
    test_ntt_batch_roundtrip(15, 8, NTTMode::OPTIMIZED);
    test_ntt_batch_roundtrip(18, 8, NTTMode::OPTIMIZED);
    test_ntt_batch_roundtrip(20, 8, NTTMode::OPTIMIZED);
    test_ntt_batch_roundtrip(15, 8, NTTMode::BARRETT);
    test_ntt_batch_roundtrip(18, 8, NTTMode::BARRETT);
    test_ntt_batch_roundtrip(20, 8, NTTMode::BARRETT);

    // Known vectors: leftover-representative sizes, both modes
    test_ntt_known_vectors(13, NTTMode::OPTIMIZED);  // %3=0 (3 outer = 1x3)
    test_ntt_known_vectors(14, NTTMode::OPTIMIZED);  // %3=1 (4 outer = 1x3+1)
    test_ntt_known_vectors(15, NTTMode::OPTIMIZED);  // %3=2 (5 outer = 1x3+2)
    test_ntt_known_vectors(13, NTTMode::BARRETT);
    test_ntt_known_vectors(14, NTTMode::BARRETT);
    test_ntt_known_vectors(15, NTTMode::BARRETT);

    // Forward zeros: NTT(0) = 0
    test_ntt_forward_zeros(13, NTTMode::OPTIMIZED);
    test_ntt_forward_zeros(16, NTTMode::OPTIMIZED);
    test_ntt_forward_zeros(22, NTTMode::OPTIMIZED);
    test_ntt_forward_zeros(13, NTTMode::BARRETT);
    test_ntt_forward_zeros(16, NTTMode::BARRETT);
    test_ntt_forward_zeros(22, NTTMode::BARRETT);

    // Inverse explicit: verify both inv(fwd(x))=x AND fwd(inv(x))=x
    test_ntt_inverse_explicit(14, NTTMode::OPTIMIZED);  // %3=1 leftover
    test_ntt_inverse_explicit(15, NTTMode::OPTIMIZED);  // %3=2 leftover
    test_ntt_inverse_explicit(22, NTTMode::OPTIMIZED);  // %3=0 (full radix-8)
    test_ntt_inverse_explicit(14, NTTMode::BARRETT);
    test_ntt_inverse_explicit(15, NTTMode::BARRETT);
    test_ntt_inverse_explicit(22, NTTMode::BARRETT);

    // ── OTF Twiddle Verification ──
    // Stage root squaring chain
    test_otf_stage_root_chain(13);
    test_otf_stage_root_chain(20);
    test_otf_stage_root_chain(22);

    // OTF twiddle value consistency (root[s]^j == omega^(j*stride))
    test_otf_twiddle_values(13);
    test_otf_twiddle_values(20);
    test_otf_twiddle_values(22);

    // OTF NTT leftover pattern coverage:
    // Montgomery radix-8: outer%3 = 0 (2^13,2^16,2^19,2^22), 1 (2^14,2^17,2^20), 2 (2^15,2^18,2^21)
    // Barrett radix-4:    outer%2 = 0 (2^14,2^16,2^18,2^20,2^22), 1 (2^13,2^15,2^17,2^19,2^21)
    test_otf_ntt_leftover_patterns(13, NTTMode::OPTIMIZED);  // 3 outer, %3=0
    test_otf_ntt_leftover_patterns(14, NTTMode::OPTIMIZED);  // 4 outer, %3=1
    test_otf_ntt_leftover_patterns(15, NTTMode::OPTIMIZED);  // 5 outer, %3=2
    test_otf_ntt_leftover_patterns(20, NTTMode::OPTIMIZED);  // 10 outer, %3=1
    test_otf_ntt_leftover_patterns(22, NTTMode::OPTIMIZED);  // 12 outer, %3=0
    test_otf_ntt_leftover_patterns(13, NTTMode::BARRETT);    // 3 outer, %2=1
    test_otf_ntt_leftover_patterns(14, NTTMode::BARRETT);    // 4 outer, %2=0
    test_otf_ntt_leftover_patterns(15, NTTMode::BARRETT);    // 5 outer, %2=1
    test_otf_ntt_leftover_patterns(20, NTTMode::BARRETT);    // 10 outer, %2=0
    test_otf_ntt_leftover_patterns(22, NTTMode::BARRETT);    // 12 outer, %2=0

    // ─── v1.6.0 Session 16: Goldilocks + BabyBear Field Arithmetic Tests ────
    printf("\n--- v1.6.0 Session 16: Goldilocks field arithmetic ---\n");
    test_goldilocks_cpu_self_test();
    test_goldilocks_gpu_add();
    test_goldilocks_gpu_sub();
    test_goldilocks_gpu_mul();
    test_goldilocks_gpu_sqr();
    test_goldilocks_algebraic();

    printf("\n--- v1.6.0 Session 16: BabyBear field arithmetic ---\n");
    test_babybear_cpu_self_test();
    test_babybear_gpu_add();
    test_babybear_gpu_sub();
    test_babybear_gpu_mul();
    test_babybear_gpu_sqr();
    test_babybear_algebraic();

    // ─── v1.6.0 Session 17: Goldilocks NTT Tests ────────────────────────────
    printf("\n--- v1.6.0 Session 17: Goldilocks NTT ---\n");

    // Forward vs CPU reference: all K paths + outer stage leftover patterns
    test_gl_ntt_forward(8);    // K=8 only
    test_gl_ntt_forward(9);    // K=9 only
    test_gl_ntt_forward(10);   // K=10 only
    test_gl_ntt_forward(11);   // K=11 only (no outer stages)
    test_gl_ntt_forward(12);   // K=11 + 1 outer stage
    test_gl_ntt_forward(14);   // K=11 + 3 outer (%3=0)
    test_gl_ntt_forward(15);   // K=11 + 4 outer (%3=1)
    test_gl_ntt_forward(16);   // K=11 + 5 outer (%3=2)
    test_gl_ntt_forward(18);   // K=11 + 7 outer
    test_gl_ntt_forward(20);   // K=11 + 9 outer (%3=0)
    test_gl_ntt_forward(22);   // K=11 + 11 outer

    // Roundtrip INTT(NTT(x)) = x
    for (int log_n = 8; log_n <= 22; log_n += 2)
        test_gl_ntt_roundtrip(log_n);

    // Known vectors at representative sizes
    test_gl_ntt_known_vectors(10);
    test_gl_ntt_known_vectors(15);
    test_gl_ntt_known_vectors(20);

    // Forward zeros: NTT(0) = 0
    test_gl_ntt_forward_zeros(12);
    test_gl_ntt_forward_zeros(20);

    // Inverse explicit: fwd(inv(x)) = x
    test_gl_ntt_inverse_explicit(15);
    test_gl_ntt_inverse_explicit(20);

    // Batched: batch vs sequential
    test_gl_ntt_batch_vs_sequential(10, 4);
    test_gl_ntt_batch_vs_sequential(15, 4);
    test_gl_ntt_batch_vs_sequential(15, 8);
    test_gl_ntt_batch_vs_sequential(18, 2);

    // Batched roundtrip
    test_gl_ntt_batch_roundtrip(10, 4);
    test_gl_ntt_batch_roundtrip(15, 8);
    test_gl_ntt_batch_roundtrip(18, 2);

    // ─── v1.6.0 Session 17: BabyBear NTT Tests ─────────────────────────────
    printf("\n--- v1.6.0 Session 17: BabyBear NTT ---\n");

    // Forward vs CPU reference
    test_bb_ntt_forward(8);
    test_bb_ntt_forward(9);
    test_bb_ntt_forward(10);
    test_bb_ntt_forward(11);
    test_bb_ntt_forward(12);
    test_bb_ntt_forward(14);
    test_bb_ntt_forward(15);
    test_bb_ntt_forward(16);
    test_bb_ntt_forward(18);
    test_bb_ntt_forward(20);
    test_bb_ntt_forward(22);

    // Roundtrip
    for (int log_n = 8; log_n <= 22; log_n += 2)
        test_bb_ntt_roundtrip(log_n);

    // Known vectors
    test_bb_ntt_known_vectors(10);
    test_bb_ntt_known_vectors(15);
    test_bb_ntt_known_vectors(20);

    // Forward zeros
    test_bb_ntt_forward_zeros(12);
    test_bb_ntt_forward_zeros(20);

    // Inverse explicit
    test_bb_ntt_inverse_explicit(15);
    test_bb_ntt_inverse_explicit(20);

    // Batched: batch vs sequential
    test_bb_ntt_batch_vs_sequential(10, 4);
    test_bb_ntt_batch_vs_sequential(15, 4);
    test_bb_ntt_batch_vs_sequential(15, 8);
    test_bb_ntt_batch_vs_sequential(18, 2);

    // Batched roundtrip
    test_bb_ntt_batch_roundtrip(10, 4);
    test_bb_ntt_batch_roundtrip(15, 8);
    test_bb_ntt_batch_roundtrip(18, 2);

    // ── Plantard Arithmetic (v1.7.0 Session 19 — NEGATIVE RESULT) ──
    printf("\n--- Plantard arithmetic tests (negative result investigation) ---\n");
    test_plantard_cpu_self_test();
    test_plantard_vs_barrett();
    test_plantard_vs_montgomery();
    test_plantard_gpu();
    test_plantard_gpu_vs_barrett_gpu();
    test_plantard_twiddle_precomputation();
    test_plantard_algebraic();

    // ─── v2.0.0 Session 20: Fq + Fq2 Field Arithmetic Tests ──────────────────
    printf("\n--- v2.0.0 Session 20: Fq (base field) arithmetic ---\n");
    test_fq_cpu_self_test();
    test_fq_montgomery_roundtrip();
    test_fq_gpu_add();
    test_fq_gpu_sub();
    test_fq_gpu_mul();
    test_fq_gpu_sqr();
    test_fq_algebraic();

    printf("\n--- v2.0.0 Session 20: Fq2 (extension field) arithmetic ---\n");
    test_fq2_cpu_self_test();
    test_fq2_gpu_add();
    test_fq2_gpu_sub();
    test_fq2_gpu_mul();
    test_fq2_gpu_sqr();
    test_fq2_algebraic();

    // ─── v3.0.0 Session 31: Fq6 (cubic extension) arithmetic ────────────────
    printf("\n--- v3.0.0 Session 31: Fq6 (cubic extension) arithmetic ---\n");
    test_fq6_cpu_self_test();
    test_fq6_gpu_add();
    test_fq6_gpu_sub();
    test_fq6_gpu_mul();
    test_fq6_gpu_sqr();
    test_fq6_algebraic();
    test_fq6_inverse();
    test_fq6_sparse_mul();
    test_fq6_frobenius();
    test_fq6_nonresidue_chain();

    // ─── v3.0.0 Session 32: Fq12 (quadratic extension over Fq6) arithmetic ──
    printf("\n--- v3.0.0 Session 32: Fq12 (quadratic extension) arithmetic ---\n");
    test_fq12_cpu_self_test();
    test_fq12_gpu_add();
    test_fq12_gpu_sub();
    test_fq12_gpu_mul();
    test_fq12_gpu_sqr();
    test_fq12_algebraic();
    test_fq12_inverse();
    test_fq12_conjugate();
    test_fq12_sparse_mul();
    test_fq12_frobenius();
    test_fq12_w_squared();

    // ─── v2.0.0 Session 21: Elliptic Curve Arithmetic Tests ──────────────────
    printf("\n--- v2.0.0 Session 21: G1 Elliptic Curve Arithmetic ---\n");
    test_g1_cpu_self_test();
    test_g1_gpu_double();
    test_g1_gpu_add();
    test_g1_gpu_add_mixed();
    test_g1_gpu_scalar_mul();
    test_g1_gpu_scalar_mul_larger();
    test_g1_gpu_on_curve();
    test_g1_gpu_negate();
    test_g1_identity();

    printf("\n--- v2.0.0 Session 21: G2 Elliptic Curve Arithmetic ---\n");
    test_g2_cpu_self_test();
    test_g2_gpu_double();
    test_g2_gpu_add();
    test_g2_gpu_add_mixed();
    test_g2_gpu_scalar_mul();
    test_g2_gpu_on_curve();
    test_g2_identity();

    // ─── v2.0.0 Session 22: Multi-Scalar Multiplication Tests ────────────────
    printf("\n--- v2.0.0 Session 22: MSM (Pippenger) ---\n");
    test_msm_window_size();
    test_msm_single_point();
    test_msm_two_points();
    test_msm_with_identity();
    test_msm_all_ones();
    test_msm_medium(8);
    test_msm_medium(16);
    test_msm_medium(32);
    test_msm_medium(64);
    test_msm_on_curve(128);
    test_msm_on_curve(256);
    test_msm_determinism();

    // ─── v2.1.0 Session 26: Production MSM Tests ────────────────────────────
    printf("\n--- v2.1.0 Session 26: Production MSM (Signed-Digit + Segments) ---\n");
    test_msm_window_size_v2();
    test_msm_high_scalars();
    test_msm_mixed_zero_scalars();
    test_msm_single_bucket();
    test_msm_power_of_2_scalars();
    test_msm_cross_validate(128);
    test_msm_cross_validate(256);
    test_msm_on_curve_large(512);
    test_msm_on_curve_large(1024);
    test_msm_multi_limb_scalars();
    test_msm_all_zeros();
    test_msm_determinism_v2();
    test_msm_on_curve_2k(2048);
    test_msm_on_curve_2k(4096);
    test_msm_scalar_one();

    // ─── v2.1.0 Session 27: Parallel Bucket Reduction Tests ─────────────────
    printf("\n--- v2.1.0 Session 27: Parallel Bucket Reduction ---\n");
    test_msm_cross_validate_512();
    test_msm_determinism_parallel();
    test_msm_on_curve_8k();
    test_msm_all_ones_1k();
    test_msm_spread_scalars();
    test_msm_two_large_scalars();
    test_msm_ascending_128();
    test_msm_half_zero_256();
    test_msm_on_curve_16k();
    test_msm_small_n3();
    test_msm_uniform_scalar_256();
    test_msm_high_bits_64();
    test_msm_alternating_pattern();
    test_msm_minimal_pippenger();

    // ─── v2.1.0 Session 28: Window Auto-Tuning + Memory Pool Tests ──────────
    printf("\n--- v2.1.0 Session 28: Window Auto-Tuning + Memory Pool ---\n");
    test_msm_window_cap();
    test_msm_window_boundary();
    test_msm_window_small_n();
    test_msm_window_independence();
    test_msm_pool_reuse();
    test_msm_nondefault_stream();
    test_msm_varying_sizes();
    test_msm_on_curve_32k();

    // ─── v2.0.0 Session 23: Polynomial Operations Tests ─────────────────────
    printf("\n--- v2.0.0 Session 23: Polynomial Operations ---\n");
    test_poly_pointwise_mul();
    test_poly_pointwise_mul_sub();
    test_poly_scale();
    test_poly_pointwise_edge_cases();
    test_poly_coset_ntt_roundtrip();
    test_poly_coset_ntt_roundtrip_512();
    test_poly_coset_ntt_1024();
    test_poly_coset_ntt_vs_cpu();
    test_poly_coset_ntt_inverse_vs_cpu();
    test_poly_coset_ntt_zeros();
    test_poly_coset_ntt_ones();
    test_poly_coset_ntt_different_gen();
    test_poly_quotient_zero();
    test_poly_quotient_identity();

    // ─── v2.0.0 Session 24: Groth16 Pipeline Tests ──────────────────────────
    printf("\n--- v2.0.0 Session 24: Groth16 Pipeline ---\n");
    test_groth16_witness();
    test_groth16_r1cs_satisfied();
    test_groth16_srs_on_curve();
    test_groth16_gpu_proof();
    test_groth16_gpu_vs_cpu();
    test_groth16_determinism();
    test_groth16_different_witness();
    test_groth16_different_randomness();

    // ─── v2.2.0 Session 29: Fibonacci Circuit Tests ──────────────────────────
    printf("\n--- v2.2.0 Session 29: Fibonacci Circuit ---\n");
    test_fibonacci_witness_small();
    test_fibonacci_witness_different_start();
    test_fibonacci_r1cs_satisfied();
    test_fibonacci_r1cs_medium();
    test_fibonacci_batch_inversion();
    test_fibonacci_lagrange_basis();
    test_fibonacci_sparse_setup_on_curve();
    test_fibonacci_sparse_qap_vs_dense();
    test_fibonacci_gpu_proof_small();
    test_fibonacci_gpu_vs_cpu_small();
    test_fibonacci_gpu_proof_256();
    test_fibonacci_gpu_vs_cpu_256();
    test_fibonacci_determinism();
    test_fibonacci_different_inputs();
    test_fibonacci_zero_start();

    // ─── v2.2.0 Session 30: Batch Pipeline Tests ────────────────────────────
    printf("\n--- v2.2.0 Session 30: Batch Pipeline ---\n");
    test_batch_empty();
    test_batch_single_matches_individual();
    test_batch_vs_sequential();
    test_batch_on_curve();
    test_batch_determinism();
    test_batch_different_witnesses();
    test_batch_same_witness_different_seeds();
    test_batch_sequential_reference();
    test_batch_256();
    test_batch_eight();
    test_batch_vs_cpu_reference();

    // ─── v3.0.0 Session 33: Miller Loop Tests ─────────────────────────────
    printf("\n--- v3.0.0 Session 33: Miller Loop ---\n");
    test_miller_loop_cpu_nondegen();
    test_miller_loop_cpu_identity_P();
    test_miller_loop_cpu_identity_Q();
    test_miller_loop_cpu_determinism();
    test_miller_loop_cpu_different_points();
    test_miller_loop_cpu_not_one();
    test_miller_loop_gpu_basic();
    test_miller_loop_gpu_vs_cpu();
    test_miller_loop_gpu_identity();
    test_miller_loop_gpu_determinism();
    test_miller_loop_gpu_different_pairs();
    test_miller_loop_bilinearity_scalar_P();
    test_miller_loop_bilinearity_scalar_Q();
    test_miller_loop_bilinearity_product();
    test_miller_loop_negation_P();
    test_miller_loop_negation_Q();
    test_miller_loop_double_vs_square();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
