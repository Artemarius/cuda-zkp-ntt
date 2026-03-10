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
#include <vector>
#include <cassert>

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

// ─── Modular Inverse (Fermat's little theorem) ──────────────────────────────
// a^{-1} = a^{r-2} mod r.  Input/output in Montgomery form.

inline FpRef fp_inv(const FpRef& a) {
    // r - 2 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfefffffffeffffffff
    std::array<uint64_t, 4> r_minus_2 = {{
        0xfffffffeffffffffULL,
        0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL,
        0x73eda753299d7d48ULL
    }};
    return fp_pow(a, r_minus_2);
}

// ─── CPU Reference NTT ──────────────────────────────────────────────────────
// Radix-2 Cooley-Tukey, in-place.  All elements in Montgomery form.
//
// BLS12-381 scalar field: TWO_ADICITY = 32 (i.e. 2^32 | r-1).
// Multiplicative generator g = 7.
// Primitive 2^32-th root of unity: omega = g^((r-1) / 2^32) mod r.

// 256-bit right shift by k bits (0 < k <= 64)
inline std::array<uint64_t, 4> shr256(const std::array<uint64_t, 4>& a, int k) {
    assert(k > 0 && k <= 64);
    std::array<uint64_t, 4> r = {{0, 0, 0, 0}};
    if (k == 64) {
        for (int i = 0; i < 3; ++i) r[i] = a[i + 1];
    } else {
        for (int i = 0; i < 4; ++i) {
            r[i] = a[i] >> k;
            if (i + 1 < 4) r[i] |= a[i + 1] << (64 - k);
        }
    }
    return r;
}

// Returns the primitive n-th root of unity in Montgomery form.
// n must be a power of 2 with 1 <= log2(n) <= 32.
inline FpRef get_root_of_unity(size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0);  // power of 2

    // Compute log2(n)
    int log_n = 0;
    { size_t tmp = n; while (tmp > 1) { tmp >>= 1; ++log_n; } }
    assert(log_n <= 32);

    // exponent = (r - 1) / n  =  (r - 1) >> log_n
    // r - 1 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000
    std::array<uint64_t, 4> r_minus_1 = {{
        0xffffffff00000000ULL,
        0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL,
        0x73eda753299d7d48ULL
    }};

    // Multi-word right shift by log_n bits (log_n <= 32, so within one word)
    std::array<uint64_t, 4> exp = shr256(r_minus_1, log_n);

    // omega_n = g^exp mod r,  where g = 7
    FpRef g = to_montgomery(FpRef::from_u64(7));
    return fp_pow(g, exp);
}

// In-place bit-reversal permutation
inline void bit_reverse_permute(std::vector<FpRef>& data, size_t n) {
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
}

// Forward NTT (Cooley-Tukey, radix-2, in-place).
// data: n elements in Montgomery form.  n must be a power of 2.
inline void ntt_forward_reference(std::vector<FpRef>& data, size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    assert(data.size() >= n);

    bit_reverse_permute(data, n);

    FpRef one;
    one.limbs = R_MOD;  // Montgomery(1)

    for (size_t len = 2; len <= n; len <<= 1) {
        FpRef w = get_root_of_unity(len);  // primitive len-th root
        for (size_t i = 0; i < n; i += len) {
            FpRef wj = one;
            for (size_t j = 0; j < len / 2; ++j) {
                FpRef u = data[i + j];
                FpRef v = fp_mul(data[i + j + len / 2], wj);
                data[i + j]           = fp_add(u, v);
                data[i + j + len / 2] = fp_sub(u, v);
                wj = fp_mul(wj, w);
            }
        }
    }
}

// Inverse NTT: uses omega^{-1} twiddles, then scales by n^{-1}.
inline void ntt_inverse_reference(std::vector<FpRef>& data, size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    assert(data.size() >= n);

    bit_reverse_permute(data, n);

    FpRef one;
    one.limbs = R_MOD;

    for (size_t len = 2; len <= n; len <<= 1) {
        FpRef w = fp_inv(get_root_of_unity(len));  // inverse root
        for (size_t i = 0; i < n; i += len) {
            FpRef wj = one;
            for (size_t j = 0; j < len / 2; ++j) {
                FpRef u = data[i + j];
                FpRef v = fp_mul(data[i + j + len / 2], wj);
                data[i + j]           = fp_add(u, v);
                data[i + j + len / 2] = fp_sub(u, v);
                wj = fp_mul(wj, w);
            }
        }
    }

    // Scale by n^{-1} mod r
    FpRef n_mont = to_montgomery(FpRef::from_u64(static_cast<uint64_t>(n)));
    FpRef n_inv  = fp_inv(n_mont);
    for (size_t i = 0; i < n; ++i)
        data[i] = fp_mul(data[i], n_inv);
}

// Naive DFT O(n^2) — for small-n cross-validation against Cooley-Tukey.
// data: n elements in Montgomery form.  Returns new vector (not in-place).
inline std::vector<FpRef> dft_naive(const std::vector<FpRef>& data, size_t n) {
    assert(data.size() >= n);

    FpRef one;
    one.limbs = R_MOD;

    FpRef w = get_root_of_unity(n);  // primitive n-th root
    std::vector<FpRef> out(n, FpRef::zero());

    for (size_t i = 0; i < n; ++i) {
        // w^i
        FpRef wi = one;
        for (size_t k = 0; k < i; ++k) wi = fp_mul(wi, w);

        // w^{ij} for each j, accumulate
        FpRef wij = one;
        for (size_t j = 0; j < n; ++j) {
            // out[i] += data[j] * w^{ij}
            out[i] = fp_add(out[i], fp_mul(data[j], wij));
            wij = fp_mul(wij, wi);
        }
    }
    return out;
}

} // namespace ff_ref
