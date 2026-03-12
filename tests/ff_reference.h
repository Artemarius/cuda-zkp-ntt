// tests/ff_reference.h
// CPU-only reference implementation of BLS12-381 scalar field arithmetic
// Plain C++17, no CUDA. Uses 4 x uint64_t limbs for simpler carry handling.
// Used as the correctness oracle for GPU implementations.
//
// Portable: uses unsigned __int128 on GCC/Clang, MSVC intrinsics on MSVC.

#pragma once
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <cassert>

// ─── 128-bit Arithmetic Portability Layer ────────────────────────────────────

#ifdef _MSC_VER
#include <intrin.h>

namespace ff_ref {
namespace detail {

struct u128 {
    uint64_t lo, hi;

    u128() : lo(0), hi(0) {}
    u128(uint64_t v) : lo(v), hi(0) {}
    u128(uint64_t lo_, uint64_t hi_) : lo(lo_), hi(hi_) {}

    explicit operator uint64_t() const { return lo; }
    explicit operator bool() const { return lo || hi; }
};

inline u128 operator+(u128 a, u128 b) {
    u128 r;
    unsigned char c = _addcarry_u64(0, a.lo, b.lo, &r.lo);
    _addcarry_u64(c, a.hi, b.hi, &r.hi);
    return r;
}

inline u128& operator+=(u128& a, u128 b) { a = a + b; return a; }

inline u128 operator-(u128 a, u128 b) {
    u128 r;
    unsigned char borrow = _subborrow_u64(0, a.lo, b.lo, &r.lo);
    _subborrow_u64(borrow, a.hi, b.hi, &r.hi);
    return r;
}

inline u128 operator*(u128 a, u128 b) {
    u128 r;
    r.lo = _umul128(a.lo, b.lo, &r.hi);
    r.hi += a.hi * b.lo + a.lo * b.hi;  // cross terms (mod 2^128)
    return r;
}

inline u128 operator>>(u128 a, int n) {
    if (n == 64) return u128(a.hi, 0);
    if (n == 0) return a;
    return u128((a.lo >> n) | (a.hi << (64 - n)), a.hi >> n);
}

inline u128& operator>>=(u128& a, int n) { a = a >> n; return a; }

inline u128 operator&(u128 a, uint64_t mask) {
    return u128(a.lo & mask, 0);
}

} // namespace detail

using uint128_t = detail::u128;

} // namespace ff_ref

#else // GCC / Clang

namespace ff_ref {
using uint128_t = unsigned __int128;
} // namespace ff_ref

#endif

// ─────────────────────────────────────────────────────────────────────────────

namespace ff_ref {

// ─── BLS12-381 Scalar Field Constants (4 x uint64_t, little-endian) ─────────
//
// Derived from 8 x uint32_t little-endian representation:
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

    // Convert from 8 x uint32_t (GPU format) to 4 x uint64_t
    static FpRef from_u32(const uint32_t w[8]) {
        FpRef r;
        for (int i = 0; i < 4; ++i)
            r.limbs[i] = (static_cast<uint64_t>(w[2*i+1]) << 32) | w[2*i];
        return r;
    }

    // Convert to 8 x uint32_t (GPU format)
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
    uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        carry += uint128_t(a.limbs[i]) + b.limbs[i];
        r.limbs[i] = static_cast<uint64_t>(carry);
        carry >>= 64;
    }
    if (carry || geq(r.limbs, MOD)) {
        uint128_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            borrow = uint128_t(r.limbs[i]) - MOD[i] - borrow;
            r.limbs[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
    }
    return r;
}

// ─── Subtraction: (a - b) mod p ──────────────────────────────────────────────

inline FpRef fp_sub(const FpRef& a, const FpRef& b) {
    FpRef r;
    uint128_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        borrow = uint128_t(a.limbs[i]) - b.limbs[i] - borrow;
        r.limbs[i] = static_cast<uint64_t>(borrow);
        borrow = (borrow >> 64) & 1;
    }
    if (borrow) {
        uint128_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            carry += uint128_t(r.limbs[i]) + MOD[i];
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
        uint128_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            carry += uint128_t(a.limbs[j]) * b.limbs[i] + T[j];
            T[j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        T[4] += static_cast<uint64_t>(carry);

        // m = T[0] * p_inv mod 2^64
        uint64_t m = T[0] * P_INV64;

        // T = (T + m * p) >> 64
        carry = uint128_t(T[0]) + uint128_t(m) * MOD[0];
        carry >>= 64;

        for (int j = 1; j < 4; ++j) {
            carry += uint128_t(T[j]) + uint128_t(m) * MOD[j];
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
        uint128_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            borrow = uint128_t(r.limbs[i]) - MOD[i] - borrow;
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

// Naive DFT O(n^2) -- for small-n cross-validation against Cooley-Tukey.
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

// ─── Barrett Modular Arithmetic ──────────────────────────────────────────────
// Standard-form in/out — no Montgomery domain conversion.
// Uses Barrett reduction: c = a*b mod p via precomputed mu = floor(2^512 / p).

// Barrett constant: mu = floor(2^512 / p), 5 x uint64_t little-endian
// mu = 0x2355094edfede377c38b5dcb707e08ed365043eb4be4bad7142737a020c0d6393
static constexpr std::array<uint64_t, 5> BARRETT_MU = {{
    0x42737a020c0d6393ULL,
    0x65043eb4be4bad71ULL,
    0x38b5dcb707e08ed3ULL,
    0x355094edfede377cULL,
    0x0000000000000002ULL
}};

// 512-bit intermediate type (8 x uint64_t, little-endian)
struct Wide512 {
    uint64_t limbs[8];
};

// Full 256x256 -> 512-bit product (schoolbook)
inline Wide512 fp_wide_mul(const FpRef& a, const FpRef& b) {
    Wide512 z;
    for (int i = 0; i < 8; ++i) z.limbs[i] = 0;

    for (int i = 0; i < 4; ++i) {
        uint128_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            carry += uint128_t(a.limbs[j]) * b.limbs[i] + z.limbs[i + j];
            z.limbs[i + j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        z.limbs[i + 4] = static_cast<uint64_t>(carry);
    }
    return z;
}

// Barrett reduction of a 512-bit value mod p
// Algorithm (HAC 14.42, 64-bit base, k=4):
//   q1 = z >> 192 (shift right by 3 words)
//   q2 = q1 * mu  (5 x 5 -> 10 words)
//   q3 = q2 >> 320 (shift right by 5 words)
//   r1 = z mod 2^320 (low 5 words)
//   r2 = (q3 * p) mod 2^320
//   r = r1 - r2
//   while r >= p: r -= p (at most 2 times)
inline FpRef fp_barrett_reduce(const Wide512& z) {
    // q1 = z >> 192 = z.limbs[3..7] (5 words)
    uint64_t q1[5];
    for (int i = 0; i < 5; ++i) q1[i] = z.limbs[3 + i];

    // q2 = q1 * mu (5 x 5 -> 10 words)
    uint64_t q2[10];
    for (int i = 0; i < 10; ++i) q2[i] = 0;

    for (int i = 0; i < 5; ++i) {
        uint128_t carry = 0;
        for (int j = 0; j < 5; ++j) {
            carry += uint128_t(q1[j]) * BARRETT_MU[i] + q2[i + j];
            q2[i + j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        q2[i + 5] = static_cast<uint64_t>(carry);
    }

    // q3 = q2[5..8] (4 words)

    // r2 = (q3 * p) mod 2^320 (low 5 words)
    uint64_t r2[5];
    for (int i = 0; i < 5; ++i) r2[i] = 0;

    for (int i = 0; i < 4; ++i) {
        uint128_t carry = 0;
        int j_limit = (5 - i) < 4 ? (5 - i) : 4;
        for (int j = 0; j < j_limit; ++j) {
            carry += uint128_t(q2[5 + i]) * MOD[j] + r2[i + j];
            r2[i + j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        if (i + j_limit < 5) {
            r2[i + j_limit] += static_cast<uint64_t>(carry);
        }
    }

    // r = z[0..4] - r2[0..4] (mod 2^320)
    uint64_t r[5];
    {
        uint128_t borrow = 0;
        for (int i = 0; i < 5; ++i) {
            borrow = uint128_t(z.limbs[i]) - r2[i] - borrow;
            r[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
    }

    // Conditional subtraction (at most 2 times)
    for (int iter = 0; iter < 2; ++iter) {
        bool need_sub = (r[4] != 0);
        if (!need_sub) {
            need_sub = true;
            for (int i = 3; i >= 0; --i) {
                if (r[i] > MOD[i]) break;
                if (r[i] < MOD[i]) { need_sub = false; break; }
            }
        }

        if (need_sub) {
            uint128_t borrow = 0;
            for (int i = 0; i < 5; ++i) {
                uint64_t mod_i = (i < 4) ? MOD[i] : 0;
                borrow = uint128_t(r[i]) - mod_i - borrow;
                r[i] = static_cast<uint64_t>(borrow);
                borrow = (borrow >> 64) & 1;
            }
        }
    }

    FpRef result;
    for (int i = 0; i < 4; ++i) result.limbs[i] = r[i];
    return result;
}

// Barrett modular multiplication: (a * b) mod p, standard form in/out
inline FpRef fp_mul_barrett(const FpRef& a, const FpRef& b) {
    Wide512 z = fp_wide_mul(a, b);
    return fp_barrett_reduce(z);
}

// Barrett squaring
inline FpRef fp_sqr_barrett(const FpRef& a) {
    return fp_mul_barrett(a, a);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Goldilocks Field: p = 2^64 - 2^32 + 1
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;
static constexpr uint64_t GL_GENERATOR = 7ULL;  // primitive root

struct GlRef {
    uint64_t val;  // in [0, p)

    static GlRef zero() { return {0}; }
    static GlRef one() { return {1}; }
    static GlRef from_u64(uint64_t v) { return {v % GL_P}; }

    bool operator==(const GlRef& o) const { return val == o.val; }
    bool operator!=(const GlRef& o) const { return val != o.val; }
};

inline GlRef gl_add(GlRef a, GlRef b) {
    uint128_t sum = uint128_t(a.val) + uint128_t(b.val);
    uint64_t r = static_cast<uint64_t>(sum);
    uint64_t carry = static_cast<uint64_t>(sum >> 64);
    if (carry || r >= GL_P) r -= GL_P;
    return {r};
}

inline GlRef gl_sub(GlRef a, GlRef b) {
    if (a.val >= b.val) return {a.val - b.val};
    return {a.val + GL_P - b.val};
}

inline GlRef gl_mul(GlRef a, GlRef b) {
    // 64x64 -> 128-bit product, then Goldilocks reduction
#ifdef _MSC_VER
    uint64_t hi;
    uint64_t lo = _umul128(a.val, b.val, &hi);
#else
    unsigned __int128 prod = static_cast<unsigned __int128>(a.val) * b.val;
    uint64_t lo = static_cast<uint64_t>(prod);
    uint64_t hi = static_cast<uint64_t>(prod >> 64);
#endif

    // Reduce: hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p)
    uint32_t hi_lo = static_cast<uint32_t>(hi);
    uint32_t hi_hi = static_cast<uint32_t>(hi >> 32);

#ifdef _MSC_VER
    // Iterative reduction for MSVC (no __int128 signed arithmetic)
    uint64_t t = lo;
    // Subtract hi
    bool borrow = (t < hi);
    t -= hi;
    if (borrow) t += GL_P;
    // Add hi_lo << 32
    uint64_t add1 = static_cast<uint64_t>(hi_lo) << 32;
    uint64_t prev = t;
    t += add1;
    if (t < prev) {
        // overflow: add 2^64 mod p = 2^32 - 1
        prev = t;
        t += 0xFFFFFFFFULL;
        if (t < prev) t += 0xFFFFFFFFULL;
    }
    // Add hi_hi * (2^32 - 1)
    uint64_t add2 = static_cast<uint64_t>(hi_hi) * 0xFFFFFFFFULL;
    prev = t;
    t += add2;
    if (t < prev) {
        prev = t;
        t += 0xFFFFFFFFULL;
        if (t < prev) t += 0xFFFFFFFFULL;
    }
    if (t >= GL_P) t -= GL_P;
    return {t};
#else
    __int128 sr = static_cast<__int128>(lo) - static_cast<__int128>(hi)
                + (static_cast<__int128>(hi_lo) << 32)
                + static_cast<__int128>(static_cast<uint64_t>(hi_hi) * 0xFFFFFFFFULL);
    while (sr < 0) sr += GL_P;
    while (static_cast<unsigned __int128>(sr) >= GL_P) sr -= GL_P;
    return {static_cast<uint64_t>(sr)};
#endif
}

inline GlRef gl_sqr(GlRef a) { return gl_mul(a, a); }

inline GlRef gl_pow(GlRef base, uint64_t exp) {
    GlRef result = GlRef::one();
    GlRef b = base;
    while (exp > 0) {
        if (exp & 1) result = gl_mul(result, b);
        b = gl_sqr(b);
        exp >>= 1;
    }
    return result;
}

inline GlRef gl_inv(GlRef a) { return gl_pow(a, GL_P - 2); }

// Primitive n-th root of unity for Goldilocks.
// TWO_ADICITY = 32 (p - 1 = 2^32 * (2^32 - 1)).
inline GlRef gl_get_root_of_unity(size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    int log_n = 0;
    { size_t tmp = n; while (tmp > 1) { tmp >>= 1; ++log_n; } }
    assert(log_n <= 32);

    uint64_t exp = (GL_P - 1) / static_cast<uint64_t>(n);
    return gl_pow(GlRef::from_u64(GL_GENERATOR), exp);
}

// Forward NTT (Cooley-Tukey, radix-2) for Goldilocks.
inline void gl_ntt_forward_reference(std::vector<GlRef>& data, size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0 && data.size() >= n);

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        GlRef w = gl_get_root_of_unity(len);
        for (size_t i = 0; i < n; i += len) {
            GlRef wj = GlRef::one();
            for (size_t j = 0; j < len / 2; ++j) {
                GlRef u = data[i + j];
                GlRef v = gl_mul(data[i + j + len / 2], wj);
                data[i + j]           = gl_add(u, v);
                data[i + j + len / 2] = gl_sub(u, v);
                wj = gl_mul(wj, w);
            }
        }
    }
}

// Inverse NTT for Goldilocks.
inline void gl_ntt_inverse_reference(std::vector<GlRef>& data, size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0 && data.size() >= n);

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        GlRef w = gl_inv(gl_get_root_of_unity(len));
        for (size_t i = 0; i < n; i += len) {
            GlRef wj = GlRef::one();
            for (size_t j = 0; j < len / 2; ++j) {
                GlRef u = data[i + j];
                GlRef v = gl_mul(data[i + j + len / 2], wj);
                data[i + j]           = gl_add(u, v);
                data[i + j + len / 2] = gl_sub(u, v);
                wj = gl_mul(wj, w);
            }
        }
    }

    GlRef n_inv = gl_inv(GlRef::from_u64(static_cast<uint64_t>(n)));
    for (size_t i = 0; i < n; ++i)
        data[i] = gl_mul(data[i], n_inv);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BabyBear Field: p = 2^31 - 2^27 + 1 = 0x78000001
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr uint32_t BB_P = 0x78000001u;  // 2013265921
static constexpr uint32_t BB_GENERATOR = 31u;  // primitive root

struct BbRef {
    uint32_t val;  // in [0, p)

    static BbRef zero() { return {0}; }
    static BbRef one() { return {1}; }
    static BbRef from_u32(uint32_t v) { return {v % BB_P}; }

    bool operator==(const BbRef& o) const { return val == o.val; }
    bool operator!=(const BbRef& o) const { return val != o.val; }
};

inline BbRef bb_add(BbRef a, BbRef b) {
    uint32_t sum = a.val + b.val;
    if (sum >= BB_P || sum < a.val) sum -= BB_P;
    return {sum};
}

inline BbRef bb_sub(BbRef a, BbRef b) {
    if (a.val >= b.val) return {a.val - b.val};
    return {a.val + BB_P - b.val};
}

inline BbRef bb_mul(BbRef a, BbRef b) {
    uint64_t prod = static_cast<uint64_t>(a.val) * b.val;
    return {static_cast<uint32_t>(prod % BB_P)};
}

inline BbRef bb_sqr(BbRef a) { return bb_mul(a, a); }

inline BbRef bb_pow(BbRef base, uint32_t exp) {
    BbRef result = BbRef::one();
    BbRef b = base;
    while (exp > 0) {
        if (exp & 1) result = bb_mul(result, b);
        b = bb_sqr(b);
        exp >>= 1;
    }
    return result;
}

inline BbRef bb_inv(BbRef a) { return bb_pow(a, BB_P - 2); }

// Primitive n-th root of unity for BabyBear.
// TWO_ADICITY = 27 (p - 1 = 2^27 * 15).
inline BbRef bb_get_root_of_unity(size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0);
    int log_n = 0;
    { size_t tmp = n; while (tmp > 1) { tmp >>= 1; ++log_n; } }
    assert(log_n <= 27);

    uint32_t exp = static_cast<uint32_t>((static_cast<uint64_t>(BB_P) - 1) / n);
    return bb_pow(BbRef::from_u32(BB_GENERATOR), exp);
}

// Forward NTT (Cooley-Tukey, radix-2) for BabyBear.
inline void bb_ntt_forward_reference(std::vector<BbRef>& data, size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0 && data.size() >= n);

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        BbRef w = bb_get_root_of_unity(len);
        for (size_t i = 0; i < n; i += len) {
            BbRef wj = BbRef::one();
            for (size_t j = 0; j < len / 2; ++j) {
                BbRef u = data[i + j];
                BbRef v = bb_mul(data[i + j + len / 2], wj);
                data[i + j]           = bb_add(u, v);
                data[i + j + len / 2] = bb_sub(u, v);
                wj = bb_mul(wj, w);
            }
        }
    }
}

// Inverse NTT for BabyBear.
inline void bb_ntt_inverse_reference(std::vector<BbRef>& data, size_t n) {
    assert(n >= 2 && (n & (n - 1)) == 0 && data.size() >= n);

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        BbRef w = bb_inv(bb_get_root_of_unity(len));
        for (size_t i = 0; i < n; i += len) {
            BbRef wj = BbRef::one();
            for (size_t j = 0; j < len / 2; ++j) {
                BbRef u = data[i + j];
                BbRef v = bb_mul(data[i + j + len / 2], wj);
                data[i + j]           = bb_add(u, v);
                data[i + j + len / 2] = bb_sub(u, v);
                wj = bb_mul(wj, w);
            }
        }
    }

    BbRef n_inv = bb_inv(BbRef::from_u32(static_cast<uint32_t>(n)));
    for (size_t i = 0; i < n; ++i)
        data[i] = bb_mul(data[i], n_inv);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Plantard Modular Arithmetic for BLS12-381
// NEGATIVE RESULT: 264 MADs vs Montgomery CIOS 136 MADs for 256-bit modulus.
// Plantard's advantage (eliminating one big-int multiply) only applies to
// word-size moduli. For multi-limb 256-bit, both product and reduction are O(n^2).
// ═══════════════════════════════════════════════════════════════════════════════

// 512-bit type for Plantard intermediate values (8 x uint64_t, little-endian)
struct Plant512 {
    uint64_t limbs[8];
};

// mu = p^{-1} mod 2^512 (8 x uint64_t, little-endian)
static constexpr std::array<uint64_t, 8> PLANTARD_MU = {{
    0x0000000100000001ULL,  // limbs64[0]
    0xac45a4000001a402ULL,  // limbs64[1]
    0xe7e4d3e8fffb13f9ULL,  // limbs64[2]
    0xc2bbc54f2840d7c6ULL,  // limbs64[3]
    0x126d1ba9fe75c03fULL,  // limbs64[4]
    0xef9dd73b4a284c49ULL,  // limbs64[5]
    0x81a4751084a126a4ULL,  // limbs64[6]
    0x72fe25abadd63363ULL   // limbs64[7]
}};

// -R^2 mod p (for Plantard twiddle conversion: T = -w * R^2 mod p)
static constexpr std::array<uint64_t, 4> NEG_R2_MOD = {{
    0x3666166e0c0d6394ULL,
    0x2850b637786bffdbULL,
    0x2d66c371974d9e76ULL,
    0x6ca4cd798a437e37ULL
}};

// Multiply two 512-bit values, return low 512 bits (mod 2^512)
inline Plant512 plant512_mul_lo(const Plant512& a, const Plant512& b) {
    Plant512 r;
    for (int i = 0; i < 8; ++i) r.limbs[i] = 0;

    for (int i = 0; i < 8; ++i) {
        uint128_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            if (i + j >= 8) break;
            carry += uint128_t(a.limbs[j]) * b.limbs[i] + r.limbs[i + j];
            r.limbs[i + j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
    }
    return r;
}

// Full 256x256 -> 512-bit product as Plant512
inline Plant512 plant_wide_mul(const FpRef& a, const FpRef& b) {
    Wide512 w = fp_wide_mul(a, b);
    Plant512 r;
    for (int i = 0; i < 8; ++i) r.limbs[i] = w.limbs[i];
    return r;
}

// Plantard reduction: PRedc(W, T) = W * T * (-R^{-2}) mod p
// Algorithm:
//   1. z = W * T (512-bit product)
//   2. c = z * mu mod R^2 (low 512 bits of 512x512 product)
//   3. q = upper(c) + 1 (c[4..7] + 1, in 64-bit limbs)
//   4. r = upper(q * p) (upper 256 bits of q*p)
//   5. if r == p, return 0
inline FpRef fp_plantard_reduce(const Plant512& z) {
    // Step 2: c = z * mu mod R^2
    Plant512 mu_val;
    for (int i = 0; i < 8; ++i) mu_val.limbs[i] = PLANTARD_MU[i];
    Plant512 c = plant512_mul_lo(z, mu_val);

    // Step 3: q = upper half of c + 1 = c[4..7] + 1
    uint64_t q[5];
    uint128_t carry_q = 1; // the "+1"
    for (int i = 0; i < 4; ++i) {
        carry_q += uint128_t(c.limbs[4 + i]);
        q[i] = static_cast<uint64_t>(carry_q);
        carry_q >>= 64;
    }
    q[4] = static_cast<uint64_t>(carry_q);

    // Step 4: r = floor(q * p / R) = upper 256 bits of q*p
    // q is at most 257 bits (5 limbs), p is 256 bits (4 limbs)
    uint64_t qp[9];
    for (int i = 0; i < 9; ++i) qp[i] = 0;

    for (int i = 0; i < 5; ++i) {
        uint128_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            if (i + j >= 9) break;
            carry += uint128_t(q[i]) * MOD[j] + qp[i + j];
            qp[i + j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        if (i + 4 < 9) qp[i + 4] += static_cast<uint64_t>(carry);
    }

    // r = qp[4..7]
    FpRef result;
    for (int i = 0; i < 4; ++i) result.limbs[i] = qp[4 + i];

    // Step 5: if r == p, return 0
    if (result.limbs == MOD) {
        return FpRef::zero();
    }
    return result;
}

// Plantard multiply: PRedc(a, t_plant) where t_plant = -w * R^2 mod p
// Result: a * w mod p (standard form in/out)
inline FpRef fp_mul_plantard(const FpRef& a, const FpRef& t_plant) {
    Plant512 z = plant_wide_mul(a, t_plant);
    return fp_plantard_reduce(z);
}

// Convert twiddle w (standard form) to Plantard form: T = -w * R^2 mod p
// Uses Barrett multiplication: T = w * (-R^2 mod p) mod p
inline FpRef fp_to_plantard_twiddle(const FpRef& w_std) {
    FpRef neg_r2;
    neg_r2.limbs = NEG_R2_MOD;
    return fp_mul_barrett(w_std, neg_r2);
}

} // namespace ff_ref
