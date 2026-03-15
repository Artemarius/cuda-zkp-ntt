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

// ═══════════════════════════════════════════════════════════════════════════════
// BLS12-381 Base Field Fq (381-bit prime, 6 x uint64_t Montgomery)
// q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// R = 2^384, Montgomery form.
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr std::array<uint64_t, 6> FQ_MOD = {{
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
}};

static constexpr std::array<uint64_t, 6> FQ_R_MOD_6 = {{
    0x760900000002fffdULL,
    0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL,
    0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL,
    0x15f65ec3fa80e493ULL
}};

static constexpr std::array<uint64_t, 6> FQ_R2_MOD_6 = {{
    0xf4df1f341c341746ULL,
    0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL,
    0x11988fe592cae3aaULL
}};

// -q^{-1} mod 2^64
static constexpr uint64_t FQ_P_INV64_VAL = 0x89f3fffcfffcfffdULL;

// ─── FqRef: CPU-only base field element ─────────────────────────────────────

struct FqRef {
    std::array<uint64_t, 6> limbs;  // little-endian

    static FqRef zero() { return FqRef{{0, 0, 0, 0, 0, 0}}; }

    static FqRef from_u64(uint64_t v) { return FqRef{{v, 0, 0, 0, 0, 0}}; }

    bool operator==(const FqRef& o) const { return limbs == o.limbs; }
    bool operator!=(const FqRef& o) const { return limbs != o.limbs; }

    // Convert from 12 x uint32_t (GPU format) to 6 x uint64_t
    static FqRef from_u32(const uint32_t w[12]) {
        FqRef r;
        for (int i = 0; i < 6; ++i)
            r.limbs[i] = (static_cast<uint64_t>(w[2*i+1]) << 32) | w[2*i];
        return r;
    }

    // Convert to 12 x uint32_t (GPU format)
    void to_u32(uint32_t w[12]) const {
        for (int i = 0; i < 6; ++i) {
            w[2*i]   = static_cast<uint32_t>(limbs[i]);
            w[2*i+1] = static_cast<uint32_t>(limbs[i] >> 32);
        }
    }
};

// ─── Fq comparison ──────────────────────────────────────────────────────────

inline bool fq_geq(const std::array<uint64_t, 6>& a, const std::array<uint64_t, 6>& b) {
    for (int i = 5; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;  // equal
}

// ─── Fq Addition ────────────────────────────────────────────────────────────

inline FqRef fq_add_ref(const FqRef& a, const FqRef& b) {
    FqRef r;
    uint128_t carry = 0;
    for (int i = 0; i < 6; ++i) {
        carry += uint128_t(a.limbs[i]) + b.limbs[i];
        r.limbs[i] = static_cast<uint64_t>(carry);
        carry >>= 64;
    }
    if (carry || fq_geq(r.limbs, FQ_MOD)) {
        uint128_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            borrow = uint128_t(r.limbs[i]) - FQ_MOD[i] - borrow;
            r.limbs[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
    }
    return r;
}

// ─── Fq Subtraction ────────────────────────────────────────────────────────

inline FqRef fq_sub_ref(const FqRef& a, const FqRef& b) {
    FqRef r;
    uint128_t borrow = 0;
    for (int i = 0; i < 6; ++i) {
        borrow = uint128_t(a.limbs[i]) - b.limbs[i] - borrow;
        r.limbs[i] = static_cast<uint64_t>(borrow);
        borrow = (borrow >> 64) & 1;
    }
    if (borrow) {
        uint128_t carry = 0;
        for (int i = 0; i < 6; ++i) {
            carry += uint128_t(r.limbs[i]) + FQ_MOD[i];
            r.limbs[i] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
    }
    return r;
}

// ─── Fq Negation ───────────────────────────────────────────────────────────

inline FqRef fq_neg_ref(const FqRef& a) {
    bool is_zero = true;
    for (int i = 0; i < 6; ++i) if (a.limbs[i] != 0) { is_zero = false; break; }
    if (is_zero) return FqRef::zero();
    FqRef q;
    q.limbs = FQ_MOD;
    return fq_sub_ref(q, a);
}

// ─── Fq Montgomery Multiplication (CIOS, 6-limb 64-bit) ────────────────────

inline FqRef fq_mul_ref(const FqRef& a, const FqRef& b) {
    uint64_t T[7] = {0};

    for (int i = 0; i < 6; ++i) {
        uint128_t carry = 0;
        for (int j = 0; j < 6; ++j) {
            carry += uint128_t(a.limbs[j]) * b.limbs[i] + T[j];
            T[j] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        T[6] += static_cast<uint64_t>(carry);

        uint64_t m = T[0] * FQ_P_INV64_VAL;

        carry = uint128_t(T[0]) + uint128_t(m) * FQ_MOD[0];
        carry >>= 64;

        for (int j = 1; j < 6; ++j) {
            carry += uint128_t(T[j]) + uint128_t(m) * FQ_MOD[j];
            T[j-1] = static_cast<uint64_t>(carry);
            carry >>= 64;
        }
        carry += T[6];
        T[5] = static_cast<uint64_t>(carry);
        T[6] = static_cast<uint64_t>(carry >> 64);
    }

    FqRef r;
    for (int i = 0; i < 6; ++i) r.limbs[i] = T[i];

    if (T[6] || fq_geq(r.limbs, FQ_MOD)) {
        uint128_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            borrow = uint128_t(r.limbs[i]) - FQ_MOD[i] - borrow;
            r.limbs[i] = static_cast<uint64_t>(borrow);
            borrow = (borrow >> 64) & 1;
        }
    }
    return r;
}

inline FqRef fq_sqr_ref(const FqRef& a) { return fq_mul_ref(a, a); }

// ─── Fq Conversions ────────────────────────────────────────────────────────

inline FqRef fq_to_montgomery_ref(const FqRef& a) {
    FqRef r2;
    r2.limbs = FQ_R2_MOD_6;
    return fq_mul_ref(a, r2);
}

inline FqRef fq_from_montgomery_ref(const FqRef& a) {
    return fq_mul_ref(a, FqRef::from_u64(1));
}

// ─── Fq Exponentiation ────────────────────────────────────────────────────

inline FqRef fq_pow_ref(const FqRef& base, const std::array<uint64_t, 6>& exp) {
    FqRef result;
    result.limbs = FQ_R_MOD_6;  // Montgomery(1)
    FqRef b = base;
    for (int i = 0; i < 6; ++i) {
        uint64_t bits = exp[i];
        for (int j = 0; j < 64; ++j) {
            if (bits & 1) result = fq_mul_ref(result, b);
            b = fq_sqr_ref(b);
            bits >>= 1;
        }
    }
    return result;
}

// ─── Fq Inverse (Fermat) ──────────────────────────────────────────────────

inline FqRef fq_inv_ref(const FqRef& a) {
    std::array<uint64_t, 6> q_minus_2 = {{
        0xb9feffffffffaaa9ULL,
        0x1eabfffeb153ffffULL,
        0x6730d2a0f6b0f624ULL,
        0x64774b84f38512bfULL,
        0x4b1ba7b6434bacd7ULL,
        0x1a0111ea397fe69aULL
    }};
    return fq_pow_ref(a, q_minus_2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fq2 = Fq[u] / (u^2 + 1) CPU Reference
// ═══════════════════════════════════════════════════════════════════════════════

struct Fq2Ref {
    FqRef c0, c1;  // c0 + c1*u

    static Fq2Ref zero() { return {FqRef::zero(), FqRef::zero()}; }
    static Fq2Ref one_mont() {
        FqRef one;
        one.limbs = FQ_R_MOD_6;
        return {one, FqRef::zero()};
    }

    bool operator==(const Fq2Ref& o) const { return c0 == o.c0 && c1 == o.c1; }
    bool operator!=(const Fq2Ref& o) const { return !(*this == o); }
};

inline Fq2Ref fq2_add_ref(const Fq2Ref& a, const Fq2Ref& b) {
    return {fq_add_ref(a.c0, b.c0), fq_add_ref(a.c1, b.c1)};
}

inline Fq2Ref fq2_sub_ref(const Fq2Ref& a, const Fq2Ref& b) {
    return {fq_sub_ref(a.c0, b.c0), fq_sub_ref(a.c1, b.c1)};
}

inline Fq2Ref fq2_neg_ref(const Fq2Ref& a) {
    return {fq_neg_ref(a.c0), fq_neg_ref(a.c1)};
}

inline Fq2Ref fq2_conjugate_ref(const Fq2Ref& a) {
    return {a.c0, fq_neg_ref(a.c1)};
}

inline Fq2Ref fq2_mul_ref(const Fq2Ref& a, const Fq2Ref& b) {
    FqRef v0 = fq_mul_ref(a.c0, b.c0);
    FqRef v1 = fq_mul_ref(a.c1, b.c1);
    FqRef c0 = fq_sub_ref(v0, v1);
    FqRef a01 = fq_add_ref(a.c0, a.c1);
    FqRef b01 = fq_add_ref(b.c0, b.c1);
    FqRef c1 = fq_mul_ref(a01, b01);
    c1 = fq_sub_ref(c1, v0);
    c1 = fq_sub_ref(c1, v1);
    return {c0, c1};
}

inline Fq2Ref fq2_sqr_ref(const Fq2Ref& a) {
    FqRef apb = fq_add_ref(a.c0, a.c1);
    FqRef amb = fq_sub_ref(a.c0, a.c1);
    FqRef c0 = fq_mul_ref(apb, amb);
    FqRef ab = fq_mul_ref(a.c0, a.c1);
    FqRef c1 = fq_add_ref(ab, ab);
    return {c0, c1};
}

inline Fq2Ref fq2_mul_by_nonresidue_ref(const Fq2Ref& a) {
    return {fq_sub_ref(a.c0, a.c1), fq_add_ref(a.c0, a.c1)};
}

inline FqRef fq2_norm_ref(const Fq2Ref& a) {
    return fq_add_ref(fq_sqr_ref(a.c0), fq_sqr_ref(a.c1));
}

inline Fq2Ref fq2_inv_ref(const Fq2Ref& a) {
    FqRef n = fq2_norm_ref(a);
    FqRef n_inv = fq_inv_ref(n);
    return {fq_mul_ref(a.c0, n_inv), fq_mul_ref(fq_neg_ref(a.c1), n_inv)};
}

inline Fq2Ref fq2_scale_ref(const Fq2Ref& a, const FqRef& s) {
    return {fq_mul_ref(a.c0, s), fq_mul_ref(a.c1, s)};
}

// ─── Fq2 Exponentiation (384-bit exponent) ────────────────────────────────

inline Fq2Ref fq2_pow_ref(const Fq2Ref& base, const std::array<uint64_t, 6>& exp) {
    Fq2Ref result = Fq2Ref::one_mont();
    Fq2Ref b = base;
    for (int i = 0; i < 6; ++i) {
        uint64_t bits = exp[i];
        for (int j = 0; j < 64; ++j) {
            if (bits & 1) result = fq2_mul_ref(result, b);
            b = fq2_sqr_ref(b);
            bits >>= 1;
        }
    }
    return result;
}

// ─── Multi-precision div by 3 (for Frobenius exponent computation) ──────────

inline std::array<uint64_t, 6> div_6limb_by_3(const std::array<uint64_t, 6>& a) {
    std::array<uint64_t, 6> q = {{}};
    uint64_t rem = 0;
    for (int i = 5; i >= 0; --i) {
        uint64_t a_hi = a[i] >> 32;
        uint64_t a_lo = a[i] & 0xFFFFFFFFULL;
        uint64_t hi_val = (rem << 32) | a_hi;
        uint64_t q_hi = hi_val / 3;
        uint64_t r_hi = hi_val % 3;
        uint64_t lo_val = (r_hi << 32) | a_lo;
        uint64_t q_lo = lo_val / 3;
        rem = lo_val % 3;
        q[i] = (q_hi << 32) | q_lo;
    }
    return q;
}

// ─── Multi-precision div by 2 (right shift) ─────────────────────────────────

inline std::array<uint64_t, 6> div_6limb_by_2(const std::array<uint64_t, 6>& a) {
    std::array<uint64_t, 6> q = {{}};
    for (int i = 0; i < 6; ++i) {
        q[i] = a[i] >> 1;
        if (i < 5) q[i] |= (a[i+1] & 1) << 63;
    }
    return q;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fq6 = Fq2[v] / (v^3 - β) CPU Reference, β = (1+u)
// ═══════════════════════════════════════════════════════════════════════════════

struct Fq6Ref {
    Fq2Ref c0, c1, c2;  // c0 + c1*v + c2*v^2

    static Fq6Ref zero() { return {Fq2Ref::zero(), Fq2Ref::zero(), Fq2Ref::zero()}; }
    static Fq6Ref one_mont() { return {Fq2Ref::one_mont(), Fq2Ref::zero(), Fq2Ref::zero()}; }

    bool operator==(const Fq6Ref& o) const { return c0 == o.c0 && c1 == o.c1 && c2 == o.c2; }
    bool operator!=(const Fq6Ref& o) const { return !(*this == o); }
};

inline Fq6Ref fq6_add_ref(const Fq6Ref& a, const Fq6Ref& b) {
    return {fq2_add_ref(a.c0, b.c0), fq2_add_ref(a.c1, b.c1), fq2_add_ref(a.c2, b.c2)};
}

inline Fq6Ref fq6_sub_ref(const Fq6Ref& a, const Fq6Ref& b) {
    return {fq2_sub_ref(a.c0, b.c0), fq2_sub_ref(a.c1, b.c1), fq2_sub_ref(a.c2, b.c2)};
}

inline Fq6Ref fq6_neg_ref(const Fq6Ref& a) {
    return {fq2_neg_ref(a.c0), fq2_neg_ref(a.c1), fq2_neg_ref(a.c2)};
}

// Multiply by v: (c0 + c1·v + c2·v²)·v = β·c2 + c0·v + c1·v²
inline Fq6Ref fq6_mul_by_nonresidue_ref(const Fq6Ref& a) {
    return {fq2_mul_by_nonresidue_ref(a.c2), a.c0, a.c1};
}

// Karatsuba multiplication: 6 Fq2 muls
inline Fq6Ref fq6_mul_ref(const Fq6Ref& a, const Fq6Ref& b) {
    Fq2Ref v0 = fq2_mul_ref(a.c0, b.c0);
    Fq2Ref v1 = fq2_mul_ref(a.c1, b.c1);
    Fq2Ref v2 = fq2_mul_ref(a.c2, b.c2);

    Fq2Ref t0 = fq2_mul_ref(fq2_add_ref(a.c1, a.c2), fq2_add_ref(b.c1, b.c2));
    t0 = fq2_sub_ref(fq2_sub_ref(t0, v1), v2);
    Fq2Ref c0 = fq2_add_ref(v0, fq2_mul_by_nonresidue_ref(t0));

    Fq2Ref t1 = fq2_mul_ref(fq2_add_ref(a.c0, a.c1), fq2_add_ref(b.c0, b.c1));
    t1 = fq2_sub_ref(fq2_sub_ref(t1, v0), v1);
    Fq2Ref c1 = fq2_add_ref(t1, fq2_mul_by_nonresidue_ref(v2));

    Fq2Ref t2 = fq2_mul_ref(fq2_add_ref(a.c0, a.c2), fq2_add_ref(b.c0, b.c2));
    Fq2Ref c2 = fq2_add_ref(fq2_sub_ref(fq2_sub_ref(t2, v0), v2), v1);

    return {c0, c1, c2};
}

// Squaring (CH-SQR2): 2 Fq2 muls + 3 Fq2 sqrs
inline Fq6Ref fq6_sqr_ref(const Fq6Ref& a) {
    Fq2Ref s0 = fq2_sqr_ref(a.c0);
    Fq2Ref ab = fq2_mul_ref(a.c0, a.c1);
    Fq2Ref s1 = fq2_add_ref(ab, ab);
    Fq2Ref s2 = fq2_sqr_ref(fq2_add_ref(fq2_sub_ref(a.c0, a.c1), a.c2));
    Fq2Ref bc = fq2_mul_ref(a.c1, a.c2);
    Fq2Ref s3 = fq2_add_ref(bc, bc);
    Fq2Ref s4 = fq2_sqr_ref(a.c2);

    Fq2Ref c0 = fq2_add_ref(s0, fq2_mul_by_nonresidue_ref(s3));
    Fq2Ref c1 = fq2_add_ref(s1, fq2_mul_by_nonresidue_ref(s4));
    Fq2Ref c2 = fq2_add_ref(fq2_sub_ref(fq2_add_ref(s1, s2), s0), fq2_sub_ref(s3, s4));

    return {c0, c1, c2};
}

// Inverse via norm to Fq2
inline Fq6Ref fq6_inv_ref(const Fq6Ref& a) {
    Fq2Ref t0 = fq2_sqr_ref(a.c0);
    Fq2Ref t1 = fq2_sqr_ref(a.c1);
    Fq2Ref t2 = fq2_sqr_ref(a.c2);
    Fq2Ref t3 = fq2_mul_ref(a.c0, a.c1);
    Fq2Ref t4 = fq2_mul_ref(a.c0, a.c2);
    Fq2Ref t5 = fq2_mul_ref(a.c1, a.c2);

    Fq2Ref s0 = fq2_sub_ref(t0, fq2_mul_by_nonresidue_ref(t5));
    Fq2Ref s1 = fq2_sub_ref(fq2_mul_by_nonresidue_ref(t2), t3);
    Fq2Ref s2 = fq2_sub_ref(t1, t4);

    Fq2Ref norm = fq2_add_ref(
        fq2_mul_ref(a.c0, s0),
        fq2_mul_by_nonresidue_ref(
            fq2_add_ref(fq2_mul_ref(a.c2, s1), fq2_mul_ref(a.c1, s2))
        )
    );

    Fq2Ref norm_inv = fq2_inv_ref(norm);

    return {fq2_mul_ref(s0, norm_inv), fq2_mul_ref(s1, norm_inv), fq2_mul_ref(s2, norm_inv)};
}

// Sparse multiply by (b0 + b1·v): 5 Fq2 muls
inline Fq6Ref fq6_mul_by_01_ref(const Fq6Ref& a, const Fq2Ref& b0, const Fq2Ref& b1) {
    Fq2Ref v0 = fq2_mul_ref(a.c0, b0);
    Fq2Ref v1 = fq2_mul_ref(a.c1, b1);

    Fq2Ref t0 = fq2_sub_ref(fq2_mul_ref(fq2_add_ref(a.c1, a.c2), b1), v1);
    Fq2Ref c0 = fq2_add_ref(v0, fq2_mul_by_nonresidue_ref(t0));

    Fq2Ref c1 = fq2_sub_ref(fq2_sub_ref(
        fq2_mul_ref(fq2_add_ref(a.c0, a.c1), fq2_add_ref(b0, b1)), v0), v1);

    Fq2Ref c2 = fq2_add_ref(fq2_sub_ref(fq2_mul_ref(fq2_add_ref(a.c0, a.c2), b0), v0), v1);

    return {c0, c1, c2};
}

// Sparse multiply by b1·v: 3 Fq2 muls
inline Fq6Ref fq6_mul_by_1_ref(const Fq6Ref& a, const Fq2Ref& b1) {
    Fq2Ref c0 = fq2_mul_by_nonresidue_ref(fq2_mul_ref(a.c2, b1));
    Fq2Ref c1 = fq2_mul_ref(a.c0, b1);
    Fq2Ref c2 = fq2_mul_ref(a.c1, b1);
    return {c0, c1, c2};
}

inline Fq6Ref fq6_scale_ref(const Fq6Ref& a, const Fq2Ref& s) {
    return {fq2_mul_ref(a.c0, s), fq2_mul_ref(a.c1, s), fq2_mul_ref(a.c2, s)};
}

// ─── Frobenius Coefficient Computation ─────────────────────────────────────
// γ₁[k] = β^((q^k-1)/3) where β = (1+u).
// γ₁[k] = product_{i=0}^{k-1} φ^i(γ₁[1]) where φ = conjugation on Fq2.
// γ₂[k] = γ₁[k]².

inline Fq2Ref compute_fq6_frobenius_c1(int power) {
    if (power == 0) return Fq2Ref::one_mont();

    // β = (1+u) in Montgomery form
    Fq2Ref beta = {fq_to_montgomery_ref(FqRef::from_u64(1)),
                   fq_to_montgomery_ref(FqRef::from_u64(1))};

    // Compute (q-1)/3 where q = FQ_MOD
    std::array<uint64_t, 6> q_minus_1 = FQ_MOD;
    q_minus_1[0] -= 1;  // q is odd, no borrow
    std::array<uint64_t, 6> exp = div_6limb_by_3(q_minus_1);

    // γ₁[1] = β^((q-1)/3)
    Fq2Ref gamma1 = fq2_pow_ref(beta, exp);
    Fq2Ref conj_gamma1 = fq2_conjugate_ref(gamma1);

    // γ₁[k] = product_{i=0}^{k-1} φ^i(γ₁[1])
    // φ^i = identity if i even, conjugation if i odd
    Fq2Ref result = Fq2Ref::one_mont();
    for (int i = 0; i < power; ++i) {
        if (i % 2 == 0)
            result = fq2_mul_ref(result, gamma1);
        else
            result = fq2_mul_ref(result, conj_gamma1);
    }
    return result;
}

inline void compute_fq6_frobenius_coefficients(Fq2Ref c1_out[6], Fq2Ref c2_out[6]) {
    for (int k = 0; k < 6; ++k) {
        c1_out[k] = compute_fq6_frobenius_c1(k);
        c2_out[k] = fq2_sqr_ref(c1_out[k]);
    }
}

// Frobenius map using precomputed coefficients
inline Fq6Ref fq6_frobenius_map_ref(const Fq6Ref& a, int power,
                                     const Fq2Ref* frob_c1,
                                     const Fq2Ref* frob_c2) {
    Fq2Ref c0 = (power & 1) ? fq2_conjugate_ref(a.c0) : a.c0;
    Fq2Ref c1 = (power & 1) ? fq2_conjugate_ref(a.c1) : a.c1;
    Fq2Ref c2 = (power & 1) ? fq2_conjugate_ref(a.c2) : a.c2;

    c1 = fq2_mul_ref(c1, frob_c1[power]);
    c2 = fq2_mul_ref(c2, frob_c2[power]);

    return {c0, c1, c2};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fq12 = Fq6[w] / (w² − v) CPU Reference
// Top of the BLS12-381 tower: Fq → Fq2 → Fq6 → Fq12.
// w² = v, so multiplying Fq6 by v is fq6_mul_by_nonresidue.
// ═══════════════════════════════════════════════════════════════════════════════

struct Fq12Ref {
    Fq6Ref c0, c1;  // c0 + c1*w

    static Fq12Ref zero() { return {Fq6Ref::zero(), Fq6Ref::zero()}; }
    static Fq12Ref one_mont() { return {Fq6Ref::one_mont(), Fq6Ref::zero()}; }

    bool operator==(const Fq12Ref& o) const { return c0 == o.c0 && c1 == o.c1; }
    bool operator!=(const Fq12Ref& o) const { return !(*this == o); }
};

inline Fq12Ref fq12_add_ref(const Fq12Ref& a, const Fq12Ref& b) {
    return {fq6_add_ref(a.c0, b.c0), fq6_add_ref(a.c1, b.c1)};
}

inline Fq12Ref fq12_sub_ref(const Fq12Ref& a, const Fq12Ref& b) {
    return {fq6_sub_ref(a.c0, b.c0), fq6_sub_ref(a.c1, b.c1)};
}

inline Fq12Ref fq12_neg_ref(const Fq12Ref& a) {
    return {fq6_neg_ref(a.c0), fq6_neg_ref(a.c1)};
}

inline Fq12Ref fq12_conjugate_ref(const Fq12Ref& a) {
    return {a.c0, fq6_neg_ref(a.c1)};
}

// Karatsuba: 3 Fq6 muls = 54 Fq muls
inline Fq12Ref fq12_mul_ref(const Fq12Ref& a, const Fq12Ref& b) {
    Fq6Ref v0 = fq6_mul_ref(a.c0, b.c0);
    Fq6Ref v1 = fq6_mul_ref(a.c1, b.c1);

    Fq6Ref c0 = fq6_add_ref(v0, fq6_mul_by_nonresidue_ref(v1));
    Fq6Ref c1 = fq6_sub_ref(fq6_sub_ref(
        fq6_mul_ref(fq6_add_ref(a.c0, a.c1), fq6_add_ref(b.c0, b.c1)), v0), v1);

    return {c0, c1};
}

// Complex squaring: 2 Fq6 muls = 36 Fq muls
inline Fq12Ref fq12_sqr_ref(const Fq12Ref& a) {
    Fq6Ref ab = fq6_mul_ref(a.c0, a.c1);
    Fq6Ref v_a1 = fq6_mul_by_nonresidue_ref(a.c1);

    Fq6Ref c0 = fq6_sub_ref(fq6_sub_ref(
        fq6_mul_ref(fq6_add_ref(a.c0, a.c1), fq6_add_ref(a.c0, v_a1)), ab),
        fq6_mul_by_nonresidue_ref(ab));
    Fq6Ref c1 = fq6_add_ref(ab, ab);

    return {c0, c1};
}

// Inverse: norm = a0² - v·a1², result = (a0/norm, -a1/norm)
inline Fq12Ref fq12_inv_ref(const Fq12Ref& a) {
    Fq6Ref a0_sq = fq6_sqr_ref(a.c0);
    Fq6Ref a1_sq = fq6_sqr_ref(a.c1);
    Fq6Ref norm = fq6_sub_ref(a0_sq, fq6_mul_by_nonresidue_ref(a1_sq));
    Fq6Ref norm_inv = fq6_inv_ref(norm);

    return {fq6_mul_ref(a.c0, norm_inv), fq6_neg_ref(fq6_mul_ref(a.c1, norm_inv))};
}

// Sparse mul_by_034: 13 Fq2 muls = 39 Fq muls
inline Fq12Ref fq12_mul_by_034_ref(const Fq12Ref& a,
                                     const Fq2Ref& d0,
                                     const Fq2Ref& d3,
                                     const Fq2Ref& d4) {
    Fq6Ref v0 = fq6_scale_ref(a.c0, d0);
    Fq6Ref v1 = fq6_mul_by_01_ref(a.c1, d3, d4);

    Fq6Ref c0 = fq6_add_ref(v0, fq6_mul_by_nonresidue_ref(v1));

    Fq6Ref sum_a = fq6_add_ref(a.c0, a.c1);
    Fq2Ref sum_d03 = fq2_add_ref(d0, d3);
    Fq6Ref c1 = fq6_sub_ref(fq6_sub_ref(
        fq6_mul_by_01_ref(sum_a, sum_d03, d4), v0), v1);

    return {c0, c1};
}

// ─── Fq12 Frobenius Coefficient Computation ────────────────────────────────
// γ_w[k] = β^((q^k-1)/6) where β = (1+u).
// γ_w[k] = product_{i=0}^{k-1} φ^i(γ_w[1])
// where φ^i on Fq2 = conjugation if i odd, identity if even.

inline Fq2Ref compute_fq12_frobenius_w(int power) {
    if (power == 0) return Fq2Ref::one_mont();

    // β = (1+u) in Montgomery form
    Fq2Ref beta = {fq_to_montgomery_ref(FqRef::from_u64(1)),
                   fq_to_montgomery_ref(FqRef::from_u64(1))};

    // Compute (q-1)/6 = div_by_2(q-1) then div_by_3
    std::array<uint64_t, 6> q_minus_1 = FQ_MOD;
    q_minus_1[0] -= 1;  // q is odd, no borrow
    std::array<uint64_t, 6> half = div_6limb_by_2(q_minus_1);
    std::array<uint64_t, 6> exp = div_6limb_by_3(half);

    // γ_w[1] = β^((q-1)/6)
    Fq2Ref gamma_w1 = fq2_pow_ref(beta, exp);
    Fq2Ref conj_gamma_w1 = fq2_conjugate_ref(gamma_w1);

    // γ_w[k] = product_{i=0}^{k-1} φ^i(γ_w[1])
    Fq2Ref result = Fq2Ref::one_mont();
    for (int i = 0; i < power; ++i) {
        if (i % 2 == 0)
            result = fq2_mul_ref(result, gamma_w1);
        else
            result = fq2_mul_ref(result, conj_gamma_w1);
    }
    return result;
}

inline void compute_fq12_frobenius_coefficients(Fq2Ref w_out[12]) {
    for (int k = 0; k < 12; ++k) {
        w_out[k] = compute_fq12_frobenius_w(k);
    }
}

// Frobenius map: φ^k(c0 + c1·w) = φ^k(c0) + φ^k(c1)·γ_w[k]·w
inline Fq12Ref fq12_frobenius_map_ref(const Fq12Ref& a, int power,
                                       const Fq2Ref* fq6_frob_c1,
                                       const Fq2Ref* fq6_frob_c2,
                                       const Fq2Ref* fq12_frob_w) {
    int fq6_power = power % 6;
    Fq6Ref c0 = fq6_frobenius_map_ref(a.c0, fq6_power, fq6_frob_c1, fq6_frob_c2);
    Fq6Ref c1 = fq6_frobenius_map_ref(a.c1, fq6_power, fq6_frob_c1, fq6_frob_c2);
    c1 = fq6_scale_ref(c1, fq12_frob_w[power]);

    return {c0, c1};
}

// =============================================================================
// BLS12-381 G1 Elliptic Curve CPU Reference (Affine coordinates)
// Curve: y^2 = x^3 + 4 over Fq
// All field values in Montgomery form.
// =============================================================================

// G1 generator (standard form, for conversion to Montgomery)
static constexpr std::array<uint64_t, 6> G1_GEN_X_STD = {{
    0xfb3af00adb22c6bbULL, 0x6c55e83ff97a1aefULL,
    0xa14e3a3f171bac58ULL, 0xc3688c4f9774b905ULL,
    0x2695638c4fa9ac0fULL, 0x17f1d3a73197d794ULL
}};
static constexpr std::array<uint64_t, 6> G1_GEN_Y_STD = {{
    0x0caa232946c5e7e1ULL, 0xd03cc744a2888ae4ULL,
    0x00db18cb2c04b3edULL, 0xfcf5e095d5d00af6ULL,
    0xa09e30ed741d8ae4ULL, 0x08b3f481e3aaa0f1ULL
}};

// G2 generator (standard form)
static constexpr std::array<uint64_t, 6> G2_GEN_X_C0_STD = {{
    0xd48056c8c121bdb8ULL, 0x0bac0326a805bbefULL,
    0xb4510b647ae3d177ULL, 0xc6e47ad4fa403b02ULL,
    0x260805272dc51051ULL, 0x024aa2b2f08f0a91ULL
}};
static constexpr std::array<uint64_t, 6> G2_GEN_X_C1_STD = {{
    0xe5ac7d055d042b7eULL, 0x334cf11213945d57ULL,
    0xb5da61bbdc7f5049ULL, 0x596bd0d09920b61aULL,
    0x7dacd3a088274f65ULL, 0x13e02b6052719f60ULL
}};
static constexpr std::array<uint64_t, 6> G2_GEN_Y_C0_STD = {{
    0xe193548608b82801ULL, 0x923ac9cc3baca289ULL,
    0x6d429a695160d12cULL, 0xadfd9baa8cbdd3a7ULL,
    0x8cc9cdc6da2e351aULL, 0x0ce5d527727d6e11ULL
}};
static constexpr std::array<uint64_t, 6> G2_GEN_Y_C1_STD = {{
    0xaaa9075ff05f79beULL, 0x3f370d275cec1da1ULL,
    0x267492ab572e99abULL, 0xcb3e287e85a763afULL,
    0x32acd2b02bc28b99ULL, 0x0606c4a02ea734ccULL
}};

struct G1AffineRef {
    FqRef x, y;
    bool infinity;

    static G1AffineRef point_at_infinity() {
        return {FqRef::zero(), FqRef::zero(), true};
    }

    static G1AffineRef generator() {
        FqRef gx, gy;
        gx.limbs = G1_GEN_X_STD;
        gy.limbs = G1_GEN_Y_STD;
        return {fq_to_montgomery_ref(gx), fq_to_montgomery_ref(gy), false};
    }

    bool operator==(const G1AffineRef& o) const {
        if (infinity && o.infinity) return true;
        if (infinity != o.infinity) return false;
        return x == o.x && y == o.y;
    }
};

inline G1AffineRef g1_double_ref(const G1AffineRef& p) {
    if (p.infinity) return p;
    if (p.y == FqRef::zero()) return G1AffineRef::point_at_infinity();

    FqRef x2 = fq_sqr_ref(p.x);
    FqRef three_x2 = fq_add_ref(x2, fq_add_ref(x2, x2));
    FqRef two_y = fq_add_ref(p.y, p.y);
    FqRef lam = fq_mul_ref(three_x2, fq_inv_ref(two_y));

    FqRef rx = fq_sub_ref(fq_sqr_ref(lam), fq_add_ref(p.x, p.x));
    FqRef ry = fq_sub_ref(fq_mul_ref(lam, fq_sub_ref(p.x, rx)), p.y);
    return {rx, ry, false};
}

inline G1AffineRef g1_add_ref(const G1AffineRef& p, const G1AffineRef& q) {
    if (p.infinity) return q;
    if (q.infinity) return p;
    if (p.x == q.x) {
        if (p.y == q.y) return g1_double_ref(p);
        return G1AffineRef::point_at_infinity();
    }

    FqRef lam = fq_mul_ref(fq_sub_ref(q.y, p.y), fq_inv_ref(fq_sub_ref(q.x, p.x)));
    FqRef rx = fq_sub_ref(fq_sub_ref(fq_sqr_ref(lam), p.x), q.x);
    FqRef ry = fq_sub_ref(fq_mul_ref(lam, fq_sub_ref(p.x, rx)), p.y);
    return {rx, ry, false};
}

inline G1AffineRef g1_negate_ref(const G1AffineRef& p) {
    if (p.infinity) return p;
    return {p.x, fq_neg_ref(p.y), false};
}

inline G1AffineRef g1_scalar_mul_ref(const G1AffineRef& p, const uint32_t scalar[8]) {
    G1AffineRef result = G1AffineRef::point_at_infinity();
    G1AffineRef base = p;
    for (int i = 0; i < 8; ++i) {
        uint32_t bits = scalar[i];
        for (int j = 0; j < 32; ++j) {
            if (bits & 1) result = g1_add_ref(result, base);
            base = g1_double_ref(base);
            bits >>= 1;
        }
    }
    return result;
}

inline bool g1_is_on_curve_ref(const G1AffineRef& p) {
    if (p.infinity) return true;
    FqRef y2 = fq_sqr_ref(p.y);
    FqRef x3 = fq_mul_ref(fq_sqr_ref(p.x), p.x);
    FqRef b = fq_to_montgomery_ref(FqRef::from_u64(4));
    return y2 == fq_add_ref(x3, b);
}

// =============================================================================
// BLS12-381 G2 Elliptic Curve CPU Reference (Affine over Fq2)
// Curve: y^2 = x^3 + 4(1+u) (sextic twist)
// =============================================================================

struct G2AffineRef {
    Fq2Ref x, y;
    bool infinity;

    static G2AffineRef point_at_infinity() {
        return {Fq2Ref::zero(), Fq2Ref::zero(), true};
    }

    static G2AffineRef generator() {
        FqRef xc0, xc1, yc0, yc1;
        xc0.limbs = G2_GEN_X_C0_STD;
        xc1.limbs = G2_GEN_X_C1_STD;
        yc0.limbs = G2_GEN_Y_C0_STD;
        yc1.limbs = G2_GEN_Y_C1_STD;
        Fq2Ref gx = {fq_to_montgomery_ref(xc0), fq_to_montgomery_ref(xc1)};
        Fq2Ref gy = {fq_to_montgomery_ref(yc0), fq_to_montgomery_ref(yc1)};
        return {gx, gy, false};
    }

    bool operator==(const G2AffineRef& o) const {
        if (infinity && o.infinity) return true;
        if (infinity != o.infinity) return false;
        return x == o.x && y == o.y;
    }
};

inline G2AffineRef g2_double_ref(const G2AffineRef& p) {
    if (p.infinity) return p;
    if (p.y == Fq2Ref::zero()) return G2AffineRef::point_at_infinity();

    Fq2Ref x2 = fq2_sqr_ref(p.x);
    Fq2Ref three_x2 = fq2_add_ref(x2, fq2_add_ref(x2, x2));
    Fq2Ref two_y = fq2_add_ref(p.y, p.y);
    Fq2Ref lam = fq2_mul_ref(three_x2, fq2_inv_ref(two_y));

    Fq2Ref rx = fq2_sub_ref(fq2_sqr_ref(lam), fq2_add_ref(p.x, p.x));
    Fq2Ref ry = fq2_sub_ref(fq2_mul_ref(lam, fq2_sub_ref(p.x, rx)), p.y);
    return {rx, ry, false};
}

inline G2AffineRef g2_add_ref(const G2AffineRef& p, const G2AffineRef& q) {
    if (p.infinity) return q;
    if (q.infinity) return p;
    if (p.x == q.x) {
        if (p.y == q.y) return g2_double_ref(p);
        return G2AffineRef::point_at_infinity();
    }

    Fq2Ref lam = fq2_mul_ref(fq2_sub_ref(q.y, p.y), fq2_inv_ref(fq2_sub_ref(q.x, p.x)));
    Fq2Ref rx = fq2_sub_ref(fq2_sub_ref(fq2_sqr_ref(lam), p.x), q.x);
    Fq2Ref ry = fq2_sub_ref(fq2_mul_ref(lam, fq2_sub_ref(p.x, rx)), p.y);
    return {rx, ry, false};
}

inline G2AffineRef g2_negate_ref(const G2AffineRef& p) {
    if (p.infinity) return p;
    return {p.x, fq2_neg_ref(p.y), false};
}

inline G2AffineRef g2_scalar_mul_ref(const G2AffineRef& p, const uint32_t scalar[8]) {
    G2AffineRef result = G2AffineRef::point_at_infinity();
    G2AffineRef base = p;
    for (int i = 0; i < 8; ++i) {
        uint32_t bits = scalar[i];
        for (int j = 0; j < 32; ++j) {
            if (bits & 1) result = g2_add_ref(result, base);
            base = g2_double_ref(base);
            bits >>= 1;
        }
    }
    return result;
}

inline bool g2_is_on_curve_ref(const G2AffineRef& p) {
    if (p.infinity) return true;
    Fq2Ref y2 = fq2_sqr_ref(p.y);
    Fq2Ref x3 = fq2_mul_ref(fq2_sqr_ref(p.x), p.x);
    FqRef b4 = fq_to_montgomery_ref(FqRef::from_u64(4));
    Fq2Ref b_prime = {b4, b4};
    return y2 == fq2_add_ref(x3, b_prime);
}

// ─── Batch Inversion (Montgomery's trick) ────────────────────────────────────
// Computes inverses of n elements using a single fp_inv + 3(n-1) multiplications.
// Input/output in Montgomery form. Zero inputs are left as zero in output.

inline void fp_batch_inverse(const std::vector<FpRef>& inputs,
                             std::vector<FpRef>& outputs, size_t n) {
    outputs.resize(n);
    if (n == 0) return;

    FpRef one;
    one.limbs = R_MOD;  // Montgomery(1)

    // Partial products (skip zeros by carrying forward previous product)
    std::vector<FpRef> products(n);
    products[0] = (inputs[0] == FpRef::zero()) ? one : inputs[0];
    for (size_t i = 1; i < n; ++i) {
        if (inputs[i] == FpRef::zero())
            products[i] = products[i - 1];
        else
            products[i] = fp_mul(products[i - 1], inputs[i]);
    }

    // Single inversion of the total product
    FpRef inv = fp_inv(products[n - 1]);

    // Back-propagate
    for (size_t i = n - 1; i > 0; --i) {
        if (inputs[i] == FpRef::zero()) {
            outputs[i] = FpRef::zero();
        } else {
            outputs[i] = fp_mul(inv, products[i - 1]);
            inv = fp_mul(inv, inputs[i]);
        }
    }
    outputs[0] = (inputs[0] == FpRef::zero()) ? FpRef::zero() : inv;
}

// ─── Polynomial Operations (CPU reference) ──────────────────────────────────

// Coset scale: data[i] *= g^i (Montgomery form in/out)
inline void coset_scale_ref(std::vector<FpRef>& data, size_t n, const FpRef& g_mont) {
    FpRef one;
    one.limbs = R_MOD;  // Montgomery(1)
    FpRef g_pow = one;
    for (size_t i = 0; i < n; ++i) {
        data[i] = fp_mul(data[i], g_pow);
        g_pow = fp_mul(g_pow, g_mont);
    }
}

// Coset unscale: data[i] *= g^{-i} (Montgomery form in/out)
inline void coset_unscale_ref(std::vector<FpRef>& data, size_t n, const FpRef& g_mont) {
    FpRef g_inv = fp_inv(g_mont);
    FpRef one;
    one.limbs = R_MOD;
    FpRef g_inv_pow = one;
    for (size_t i = 0; i < n; ++i) {
        data[i] = fp_mul(data[i], g_inv_pow);
        g_inv_pow = fp_mul(g_inv_pow, g_inv);
    }
}

// Coset NTT forward: scale by g^i then NTT
inline void coset_ntt_forward_ref(std::vector<FpRef>& data, size_t n, const FpRef& g_mont) {
    coset_scale_ref(data, n, g_mont);
    ntt_forward_reference(data, n);
}

// Coset NTT inverse: INTT then unscale by g^{-i}
inline void coset_ntt_inverse_ref(std::vector<FpRef>& data, size_t n, const FpRef& g_mont) {
    ntt_inverse_reference(data, n);
    coset_unscale_ref(data, n, g_mont);
}

// Pointwise multiply: c[i] = a[i] * b[i] (Montgomery form)
inline void pointwise_mul_ref(std::vector<FpRef>& c,
                              const std::vector<FpRef>& a,
                              const std::vector<FpRef>& b, size_t n) {
    c.resize(n);
    for (size_t i = 0; i < n; ++i) {
        c[i] = fp_mul(a[i], b[i]);
    }
}

// Pointwise multiply-subtract: out[i] = a[i]*b[i] - c[i] (Montgomery form)
inline void pointwise_mul_sub_ref(std::vector<FpRef>& out,
                                  const std::vector<FpRef>& a,
                                  const std::vector<FpRef>& b,
                                  const std::vector<FpRef>& c, size_t n) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = fp_sub(fp_mul(a[i], b[i]), c[i]);
    }
}

// =============================================================================
// BLS12-381 Optimal Ate Pairing — Miller Loop (CPU Reference)
// =============================================================================
//
// The optimal Ate pairing for BLS12-381:
//   e: G1 x G2 -> GT (subgroup of Fq12*)
//
// Miller loop parameter: u = -0xd201000000010000 (64-bit, Hamming weight 5)
// |u| = 0xd201000000010000
//
// Uses affine coordinates for the running G2 point T throughout.
// Requires one Fq2 inversion per step, but correct and simple.
//
// For BLS12-381 D-type sextic twist with ξ = (1+u):
//   ψ(x', y') = (x'·v, y'·v·w) maps E'(Fq2) → E(Fq12)
//   where v is the Fq6 generator and w is the Fq12 generator.
//
// The line evaluation at P∈G1 produces sparse Fq12 at positions (0, 3, 4),
// matching our fq12_mul_by_034.

struct MillerLineCoeffs {
    Fq2Ref c0, c3, c4;  // sparse Fq12 coefficients at positions 0, 3, 4
};

// BLS12-381 Miller loop parameter (absolute value)
static constexpr uint64_t BLS12_381_U = 0xd201000000010000ULL;

// Scaling Fq2 by an Fq element: (a + bu) * s = (a*s) + (b*s)*u
inline Fq2Ref fq2_mul_by_fq_ref(const Fq2Ref& a, const FqRef& s) {
    return {fq_mul_ref(a.c0, s), fq_mul_ref(a.c1, s)};
}

// Miller loop doubling step (affine T, affine P).
// T = (xt, yt) on E'(Fq2), P = (xP, yP) on E(Fq)
// Returns sparse line coefficients (d0, d3, d4), updates T to 2T.
//
// Derivation: tangent line at T on untwisted curve evaluated at P:
//   d0 = 2·yt · yP,  d3 = -3·xt² · xP,  d4 = 3·xt³ - 2·yt²
inline MillerLineCoeffs miller_double_step_ref(G2AffineRef& T, const G1AffineRef& P) {
    Fq2Ref xt = T.x, yt = T.y;

    Fq2Ref xt2 = fq2_sqr_ref(xt);
    Fq2Ref three_xt2 = fq2_add_ref(xt2, fq2_add_ref(xt2, xt2));
    Fq2Ref two_yt = fq2_add_ref(yt, yt);
    Fq2Ref lam = fq2_mul_ref(three_xt2, fq2_inv_ref(two_yt));

    // 2T (affine)
    Fq2Ref xr = fq2_sub_ref(fq2_sqr_ref(lam), fq2_add_ref(xt, xt));
    Fq2Ref yr = fq2_sub_ref(fq2_mul_ref(lam, fq2_sub_ref(xt, xr)), yt);
    T.x = xr;
    T.y = yr;

    // Line coefficients
    MillerLineCoeffs line;
    line.c0 = fq2_mul_by_fq_ref(two_yt, P.y);
    line.c3 = fq2_neg_ref(fq2_mul_by_fq_ref(three_xt2, P.x));
    Fq2Ref xt3 = fq2_mul_ref(xt2, xt);
    Fq2Ref three_xt3 = fq2_add_ref(xt3, fq2_add_ref(xt3, xt3));
    Fq2Ref yt2 = fq2_sqr_ref(yt);
    Fq2Ref two_yt2 = fq2_add_ref(yt2, yt2);
    line.c4 = fq2_sub_ref(three_xt3, two_yt2);

    return line;
}

// Miller loop addition step (affine T + affine Q, affine P).
// T = (xt, yt), Q = (xq, yq) on E'(Fq2), P on E(Fq).
// Returns sparse line coefficients, updates T to T+Q.
//
// Derivation: chord line through T,Q on untwisted curve evaluated at P:
//   d0 = (xq-xt)·yP,  d3 = -(yq-yt)·xP,  d4 = (yq-yt)·xt - (xq-xt)·yt
inline MillerLineCoeffs miller_add_step_ref(G2AffineRef& T, const G2AffineRef& Q,
                                              const G1AffineRef& P) {
    Fq2Ref xt = T.x, yt = T.y;
    Fq2Ref xq = Q.x, yq = Q.y;

    Fq2Ref dy = fq2_sub_ref(yq, yt);
    Fq2Ref dx = fq2_sub_ref(xq, xt);
    Fq2Ref lam = fq2_mul_ref(dy, fq2_inv_ref(dx));

    // T + Q (affine)
    Fq2Ref xr = fq2_sub_ref(fq2_sub_ref(fq2_sqr_ref(lam), xt), xq);
    Fq2Ref yr = fq2_sub_ref(fq2_mul_ref(lam, fq2_sub_ref(xt, xr)), yt);
    T.x = xr;
    T.y = yr;

    // Line coefficients
    MillerLineCoeffs line;
    line.c0 = fq2_mul_by_fq_ref(dx, P.y);
    line.c3 = fq2_neg_ref(fq2_mul_by_fq_ref(dy, P.x));
    line.c4 = fq2_sub_ref(fq2_mul_ref(dy, xt), fq2_mul_ref(dx, yt));

    return line;
}

// Miller loop: f_{|u|, Q}(P) with sign correction for negative u.
// Does NOT include the final exponentiation.
inline Fq12Ref miller_loop_ref(const G1AffineRef& P, const G2AffineRef& Q) {
    if (P.infinity || Q.infinity) return Fq12Ref::one_mont();

    Fq12Ref f = Fq12Ref::one_mont();
    G2AffineRef T = Q;

    // |u| = 0xd201000000010000, MSB is bit 63
    uint64_t u_abs = BLS12_381_U;
    int top_bit = 63;

    // Iterate from bit 62 down to 0
    for (int i = top_bit - 1; i >= 0; --i) {
        f = fq12_sqr_ref(f);

        MillerLineCoeffs ld = miller_double_step_ref(T, P);
        f = fq12_mul_by_034_ref(f, ld.c0, ld.c3, ld.c4);

        if ((u_abs >> i) & 1) {
            MillerLineCoeffs la = miller_add_step_ref(T, Q, P);
            f = fq12_mul_by_034_ref(f, la.c0, la.c3, la.c4);
        }
    }

    // u < 0 for BLS12-381: conjugate f (cheap inversion for unitary elements)
    f = fq12_conjugate_ref(f);

    return f;
}

} // namespace ff_ref
