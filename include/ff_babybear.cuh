// include/ff_babybear.cuh
// Finite-field arithmetic for the BabyBear field: p = 2^31 - 2^27 + 1
// Single uint32_t representation. Used by RISC Zero for STARKs.
//
// Key property: p = 0x78000001, so 2^31 ≡ 2^27 - 1 (mod p).
// p - 1 = 2^27 * 15, so TWO_ADICITY = 27 (NTT-friendly up to 2^27).
//
// All operations are __device__ __forceinline__ for kernel inlining.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ─── BabyBear Field Constants ───────────────────────────────────────────────

// p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
__device__ static constexpr uint32_t BABYBEAR_P = 0x78000001u;

// Generator g = 31 (primitive root modulo p)
// p - 1 = 2^27 * 15, TWO_ADICITY = 27
__device__ static constexpr uint32_t BABYBEAR_GENERATOR = 31u;

// ─── Field Element ──────────────────────────────────────────────────────────

struct BabyBearElement {
    uint32_t val;  // value in [0, p)

    __host__ __device__ __forceinline__
    static BabyBearElement zero() { return {0}; }

    __host__ __device__ __forceinline__
    static BabyBearElement one() { return {1}; }

    __host__ __device__ __forceinline__
    static BabyBearElement from_u32(uint32_t v) { return {v % BABYBEAR_P}; }

    __host__ __device__ __forceinline__
    bool operator==(const BabyBearElement& o) const { return val == o.val; }

    __host__ __device__ __forceinline__
    bool operator!=(const BabyBearElement& o) const { return val != o.val; }
};

// ─── Modular Addition: (a + b) mod p ────────────────────────────────────────
// Branchless: add, then conditionally subtract p.

__device__ __forceinline__
BabyBearElement bb_add(BabyBearElement a, BabyBearElement b) {
    uint32_t sum = a.val + b.val;
    // If sum >= p (or overflow), subtract p.
    // Since a, b < p < 2^31, sum < 2^32, no uint32_t overflow possible.
    // (max a + max b = 2*(p-1) = 2*0x78000000 = 0xF0000000 < 2^32)
    sum -= BABYBEAR_P;
    // If the subtraction underflowed (sum was < p), add p back
    // Use arithmetic right shift to get mask: if underflow, high bit set
    sum += BABYBEAR_P & static_cast<uint32_t>(static_cast<int32_t>(sum) >> 31);
    return {sum};
}

// ─── Modular Subtraction: (a - b) mod p ─────────────────────────────────────
// Branchless: subtract, then conditionally add p.

__device__ __forceinline__
BabyBearElement bb_sub(BabyBearElement a, BabyBearElement b) {
    uint32_t diff = a.val - b.val;
    // If a < b, diff underflowed — add p back
    diff += BABYBEAR_P & static_cast<uint32_t>(static_cast<int32_t>(diff) >> 31);
    return {diff};
}

// ─── Modular Multiplication: (a * b) mod p ──────────────────────────────────
// 32×32→64 bit multiply, then reduce mod p.
//
// Reduction strategy: product = hi * 2^32 + lo where hi < p and lo < 2^32.
// We need (hi * 2^32 + lo) mod p.
//
// Since p = 2^31 - 2^27 + 1:
//   2^32 = 2 * 2^31 ≡ 2*(2^27 - 1) = 2^28 - 2 (mod p)
//   So hi * 2^32 ≡ hi * (2^28 - 2) (mod p)
//
// But simpler: just use 64-bit arithmetic for the full product and reduce.
// product mod p = product - (product / p) * p
// For small p (31 bits), we can use a Barrett-like approach with 64-bit math.

__device__ __forceinline__
BabyBearElement bb_mul(BabyBearElement a, BabyBearElement b) {
    uint64_t prod = static_cast<uint64_t>(a.val) * static_cast<uint64_t>(b.val);
    // Direct modulo: product fits in 62 bits (31+31), p fits in 31 bits.
    // Use hardware 64-bit division (fast on modern GPUs)
    // or manual Barrett reduction for extra speed.
    //
    // Barrett: q = (prod * mu) >> 62 where mu = floor(2^62 / p)
    // But 64-bit division by a constant is well-optimized by the compiler.
    uint32_t r = static_cast<uint32_t>(prod % static_cast<uint64_t>(BABYBEAR_P));
    return {r};
}

// ─── Modular Squaring ───────────────────────────────────────────────────────

__device__ __forceinline__
BabyBearElement bb_sqr(BabyBearElement a) {
    return bb_mul(a, a);
}

// ─── Modular Exponentiation (binary, LSB-first) ────────────────────────────

__device__ __forceinline__
BabyBearElement bb_pow(BabyBearElement base, uint32_t exp) {
    BabyBearElement result = BabyBearElement::one();
    BabyBearElement b = base;
    while (exp > 0) {
        if (exp & 1) result = bb_mul(result, b);
        b = bb_sqr(b);
        exp >>= 1;
    }
    return result;
}

// ─── Modular Inverse (Fermat's little theorem) ─────────────────────────────
// a^{-1} = a^{p-2} mod p

__device__ __forceinline__
BabyBearElement bb_inv(BabyBearElement a) {
    return bb_pow(a, BABYBEAR_P - 2);
}
