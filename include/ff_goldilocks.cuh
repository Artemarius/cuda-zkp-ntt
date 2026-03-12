// include/ff_goldilocks.cuh
// Finite-field arithmetic for the Goldilocks field: p = 2^64 - 2^32 + 1
// Single uint64_t representation. Used by Plonky2/3 for recursive SNARKs.
//
// Key property: p = 2^64 - 2^32 + 1 means 2^64 ≡ 2^32 - 1 (mod p),
// enabling fast reduction of 128-bit products.
//
// All operations are __device__ __forceinline__ for kernel inlining.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

// ─── Goldilocks Field Constants ─────────────────────────────────────────────

// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
__device__ static constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

// Generator g = 7 (primitive root modulo p)
// p - 1 = 2^32 * (2^32 - 1), so TWO_ADICITY = 32
__device__ static constexpr uint64_t GOLDILOCKS_GENERATOR = 7ULL;

// ─── Field Element ──────────────────────────────────────────────────────────

struct GoldilocksElement {
    uint64_t val;  // value in [0, p)

    __host__ __device__ __forceinline__
    static GoldilocksElement zero() { return {0}; }

    __host__ __device__ __forceinline__
    static GoldilocksElement one() { return {1}; }

    __host__ __device__ __forceinline__
    static GoldilocksElement from_u64(uint64_t v) { return {v % GOLDILOCKS_P}; }

    __host__ __device__ __forceinline__
    bool operator==(const GoldilocksElement& o) const { return val == o.val; }

    __host__ __device__ __forceinline__
    bool operator!=(const GoldilocksElement& o) const { return val != o.val; }
};

// ─── Modular Addition: (a + b) mod p ────────────────────────────────────────
// Branchless: compute sum, conditional subtract p using underflow detection.

__device__ __forceinline__
GoldilocksElement gl_add(GoldilocksElement a, GoldilocksElement b) {
    uint64_t sum = a.val + b.val;
    // Detect carry (overflow of uint64_t addition)
    uint64_t carry = (sum < a.val) ? 1ULL : 0ULL;
    // If carry or sum >= p, subtract p. Since p = 2^64 - 2^32 + 1,
    // subtracting p from a 65-bit value (carry:sum) is: sum + 2^32 - 1 (mod 2^64)
    // and clearing the carry. If no overflow and sum < p, add 0.
    uint64_t adj = (carry != 0) ? (0xFFFFFFFFULL) : 0ULL;  // 2^32 - 1
    uint64_t r = sum + adj;
    // Now handle the no-carry case where sum >= p
    if (carry == 0 && sum >= GOLDILOCKS_P) {
        r = sum - GOLDILOCKS_P;
    }
    return {r};
}

// ─── Modular Subtraction: (a - b) mod p ─────────────────────────────────────
// Branchless: compute diff, conditional add p using underflow detection.

__device__ __forceinline__
GoldilocksElement gl_sub(GoldilocksElement a, GoldilocksElement b) {
    uint64_t diff = a.val - b.val;
    // If a < b, we underflowed — add p back
    uint64_t borrow = (a.val < b.val) ? 1ULL : 0ULL;
    uint64_t adj = borrow * GOLDILOCKS_P;
    return {diff + adj};
}

// ─── Goldilocks Reduction: reduce 128-bit product to [0, p) ─────────────────
// Given product = hi:lo (128 bits), compute product mod p.
//
// Since 2^64 ≡ 2^32 - 1 (mod p):
//   hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p)
//                   = lo - hi + hi * 2^32 (mod p)
//
// Let t = hi * 2^32. Then result = lo + t - hi (mod p).
// We compute this carefully to avoid intermediate overflow.

__device__ __forceinline__
GoldilocksElement gl_reduce128(uint64_t lo, uint64_t hi) {
    // hi_lo = low 32 bits of hi, hi_hi = high 32 bits of hi
    uint32_t hi_lo = static_cast<uint32_t>(hi);
    uint32_t hi_hi = static_cast<uint32_t>(hi >> 32);

    // hi * 2^32 = (hi_hi << 64) + (hi_lo << 32)
    // hi * (2^32 - 1) = hi * 2^32 - hi = (hi_hi << 64) + (hi_lo << 32) - hi
    // But hi_hi << 64 itself needs reduction: hi_hi * 2^64 ≡ hi_hi * (2^32 - 1) mod p
    //
    // Two-step approach:
    // Step 1: result = lo + hi_lo * 2^32 - hi (mod p)  [partial, ignoring hi_hi * 2^64]
    // Step 2: result += hi_hi * (2^32 - 1) (mod p)

    // Step 1: t1 = lo - hi (might underflow)
    uint64_t t1 = lo - hi;
    uint64_t borrow1 = (lo < hi) ? 1ULL : 0ULL;

    // Add hi_lo << 32
    uint64_t t2 = static_cast<uint64_t>(hi_lo) << 32;
    uint64_t r1 = t1 + t2;
    uint64_t carry1 = (r1 < t1) ? 1ULL : 0ULL;

    // Step 2: add hi_hi * (2^32 - 1)
    uint64_t adj = static_cast<uint64_t>(hi_hi) * 0xFFFFFFFFULL;
    uint64_t r2 = r1 + adj;
    uint64_t carry2 = (r2 < r1) ? 1ULL : 0ULL;

    // Net carry/borrow: carry1 + carry2 - borrow1
    // Each carry adds 2^64 ≡ 2^32 - 1 (mod p)
    // Each borrow subtracts 2^64 ≡ 2^32 - 1 (mod p) → adds p
    int64_t net = static_cast<int64_t>(carry1 + carry2) - static_cast<int64_t>(borrow1);

    uint64_t result = r2;
    if (net > 0) {
        // Add net * (2^32 - 1)
        for (int64_t i = 0; i < net; ++i) {
            uint64_t prev = result;
            result += 0xFFFFFFFFULL;
            if (result < prev) result += 0xFFFFFFFFULL;  // cascading carry (very rare)
        }
    } else if (net < 0) {
        // Subtract |net| * (2^32 - 1) ≡ add |net| * p
        for (int64_t i = 0; i < -net; ++i) {
            result += GOLDILOCKS_P;
        }
    }

    // Final reduction: at most 1 subtraction needed
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return {result};
}

// ─── Modular Multiplication: (a * b) mod p ──────────────────────────────────
// 64×64→128 bit multiply, then Goldilocks reduction.

__device__ __forceinline__
GoldilocksElement gl_mul(GoldilocksElement a, GoldilocksElement b) {
    // Use PTX mul for exact 64x64->128 on GPU
    uint64_t lo, hi;
#ifdef __CUDA_ARCH__
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a.val), "l"(b.val));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a.val), "l"(b.val));
#else
    // Host fallback — MSVC uses _umul128, GCC/Clang uses __int128
#ifdef _MSC_VER
    lo = _umul128(a.val, b.val, &hi);
#else
    unsigned __int128 prod = static_cast<unsigned __int128>(a.val) * b.val;
    lo = static_cast<uint64_t>(prod);
    hi = static_cast<uint64_t>(prod >> 64);
#endif
#endif
    return gl_reduce128(lo, hi);
}

// ─── Modular Squaring ───────────────────────────────────────────────────────

__device__ __forceinline__
GoldilocksElement gl_sqr(GoldilocksElement a) {
    return gl_mul(a, a);
}

// ─── Modular Exponentiation (binary, MSB-first) ────────────────────────────

__device__ __forceinline__
GoldilocksElement gl_pow(GoldilocksElement base, uint64_t exp) {
    GoldilocksElement result = GoldilocksElement::one();
    GoldilocksElement b = base;
    while (exp > 0) {
        if (exp & 1) result = gl_mul(result, b);
        b = gl_sqr(b);
        exp >>= 1;
    }
    return result;
}

// ─── Modular Inverse (Fermat's little theorem) ─────────────────────────────
// a^{-1} = a^{p-2} mod p

__device__ __forceinline__
GoldilocksElement gl_inv(GoldilocksElement a) {
    return gl_pow(a, GOLDILOCKS_P - 2);
}
