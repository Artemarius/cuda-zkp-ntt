// tests/ff_reference.h
// CPU-only reference implementation of BLS12-381 scalar field arithmetic
// Plain C++17, no CUDA. Uses 4 × uint64_t limbs for simpler carry handling.
// Used as the correctness oracle for GPU implementations.
//
// NOTE: Requires __uint128_t (GCC / Clang). Not available on MSVC.
// This is acceptable since tests build on WSL2 (GCC 11).

#pragma once
#include <cstdint>
#include <cstring>
#include <array>

namespace ff_ref {

// ─── BLS12-381 Scalar Field Constants (4 × uint64_t, little-endian) ─────────
//
// Derived from 8 × uint32_t little-endian representation:
//   {0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
//    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753}
//
// 64-bit packing: limbs64[i] = (limbs32[2i+1] << 32) | limbs32[2i]

// Modulus r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
static constexpr std::array<uint64_t, 4> MOD = {{
    0xffffffff00000001ULL,   // limbs32[0..1]
    0x53bda402fffe5bfeULL,   // limbs32[2..3]
    0x3339d80809a1d805ULL,   // limbs32[4..5]
    0x73eda753299d7d48ULL    // limbs32[6..7]
}};

// R = 2^256 mod r  (Montgomery form of 1)
static constexpr std::array<uint64_t, 4> R_MOD = {{
    0x00000001fffffffeULL,
    0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL,
    0x1824b159acc5056fULL
}};

// R^2 mod r  (for converting to Montgomery form)
static constexpr std::array<uint64_t, 4> R2_MOD = {{
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
}};

// p^{-1} mod 2^64  (Montgomery reduction constant)
// Derived via Hensel lifting from p^{-1} mod 2^32 = 0xffffffff
static constexpr uint64_t P_INV64 = 0xfffffffeffffffffULL;

// ─── FpRef: CPU-only field element ───────────────────────────────────────────

struct FpRef {
    std::array<uint64_t, 4> limbs;  // little-endian

    static FpRef zero() { return FpRef{{0, 0, 0, 0}}; }

    static FpRef from_u64(uint64_t v) { return FpRef{{v, 0, 0, 0}}; }

    bool operator==(const FpRef& o) const { return limbs == o.limbs; }
    bool operator!=(const FpRef& o) const { return limbs != o.limbs; }

    // Convert from 8 × uint32_t (GPU format) to 4 × uint64_t
    static FpRef from_u32(const uint32_t w[8]) {
        FpRef r;
        for (int i = 0; i < 4; ++i)
            r.limbs[i] = (static_cast<uint64_t>(w[2*i+1]) << 32) | w[2*i];
        return r;
    }

    // Convert to 8 × uint32_t (GPU format)
    void to_u32(uint32_t w[8]) const {
        for (int i = 0; i < 4; ++i) {
            w[2*i]   = static_cast<uint32_t>(limbs[i]);
            w[2*i+1] = static_cast<uint32_t>(limbs[i] >> 32);
        }
    }
};

// ─── Comparison ──────────────────────────────────────────────────────────────

inline bool geq(const std::array<uint64_t, 4>& a, const std::array<uint64_t, 4>& b) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;  // equal
}

// ─── Addition: (a + b) mod p ─────────────────────────────────────────────────

inline FpRef fp_add(const FpRef& a, const FpRef& b) {
    FpRef r;
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; ++i) {
        carry += static_cast<unsigned __int128>(a.limbs[i]) + b.limbs[i];
        r.limbs[i] = static_cast<uint64_t>(carry);
        carry >>= 64;
    }
    if (carry || geq(r.limbs, MOD)) {
        unsigned __int128 borrow = 0;
        for (int i = 0; i < 4; ++i) {
            borrow = static_cast<unsigned __int128>(r.limbs[i]) - MOD[i] - borrow;
            r.limbs[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
    }
    return r;
}

// ─── Subtraction: (a - b) mod p ──────────────────────────────────────────────

inline FpRef fp_sub(const FpRef& a, const FpRef& b) {
    FpRef r;
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 4; ++i) {
        borrow = static_cast<unsigned __int128>(a.limbs[i]) - b.limbs[i] - borrow;
        r.limbs[i] = static_cast<uint64_t>(borrow);
        borrow = (borrow >> 64) & 1;
    }
    if (borrow) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            carry += static_cast<unsigned __int128>(r.limbs[i]) + MOD[i];
            r.limbs[i] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
    }
    return r;
}

// ─── Montgomery Multiplication (CIOS, 4-limb 64-bit) ────────────────────────
// Computes MonMul(a, b) = a * b * R^{-1} mod p
// If a = a_real * R and b = b_real * R, result = (a_real * b_real) * R mod p

inline FpRef fp_mul(const FpRef& a, const FpRef& b) {
    uint64_t T[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 4; ++i) {
        // T += a * b[i]
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            carry += static_cast<unsigned __int128>(a.limbs[j]) * b.limbs[i] + T[j];
            T[j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        T[4] += static_cast<uint64_t>(carry);

        // m = T[0] * p_inv mod 2^64
        uint64_t m = T[0] * P_INV64;

        // T = (T + m * p) >> 64
        carry = static_cast<unsigned __int128>(T[0]) +
                static_cast<unsigned __int128>(m) * MOD[0];
        carry >>= 64;

        for (int j = 1; j < 4; ++j) {
            carry += static_cast<unsigned __int128>(T[j]) +
                     static_cast<unsigned __int128>(m) * MOD[j];
            T[j-1] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        carry += T[4];
        T[3] = static_cast<uint64_t>(carry);
        T[4] = static_cast<uint64_t>(carry >> 64);
    }

    FpRef r;
    for (int i = 0; i < 4; ++i) r.limbs[i] = T[i];

    if (T[4] || geq(r.limbs, MOD)) {
        unsigned __int128 borrow = 0;
        for (int i = 0; i < 4; ++i) {
            borrow = static_cast<unsigned __int128>(r.limbs[i]) - MOD[i] - borrow;
            r.limbs[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
    }
    return r;
}

// ─── Montgomery Squaring ─────────────────────────────────────────────────────

inline FpRef fp_sqr(const FpRef& a) {
    return fp_mul(a, a);
}

// ─── Conversions ─────────────────────────────────────────────────────────────

inline FpRef to_montgomery(const FpRef& a) {
    FpRef r2;
    r2.limbs = R2_MOD;
    return fp_mul(a, r2);
}

inline FpRef from_montgomery(const FpRef& a) {
    return fp_mul(a, FpRef::from_u64(1));
}

// ─── Modular Exponentiation (in Montgomery form) ────────────────────────────

inline FpRef fp_pow(const FpRef& base, const std::array<uint64_t, 4>& exp) {
    FpRef result;
    result.limbs = R_MOD;  // Montgomery(1)
    FpRef b = base;
    for (int i = 0; i < 4; ++i) {
        uint64_t bits = exp[i];
        for (int j = 0; j < 64; ++j) {
            if (bits & 1) result = fp_mul(result, b);
            b = fp_sqr(b);
            bits >>= 1;
        }
    }
    return result;
}

} // namespace ff_ref
