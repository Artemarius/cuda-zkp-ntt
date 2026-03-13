// include/ff_fq.cuh
// Finite-field arithmetic for BLS12-381 base field Fq (381-bit prime)
// 12 x uint32_t Montgomery representation.
// Used for elliptic curve operations on G1 (over Fq) and G2 (over Fq2).

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ─── BLS12-381 Base Field Constants ─────────────────────────────────────────
// q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// All constants in little-endian 12 x uint32_t representation.
// Montgomery form with R = 2^384.

static constexpr int FQ_LIMBS = 12;

__device__ static constexpr uint32_t FQ_MODULUS[12] = {
    0xffffaaabu, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
};

// R mod q = 2^384 mod q
__device__ static constexpr uint32_t FQ_R_MOD[12] = {
    0x0002fffdu, 0x76090000u, 0xc40c0002u, 0xebf4000bu,
    0x53c758bau, 0x5f489857u, 0x70525745u, 0x77ce5853u,
    0xa256ec6du, 0x5c071a97u, 0xfa80e493u, 0x15f65ec3u
};

// R^2 mod q (for converting to Montgomery form)
__device__ static constexpr uint32_t FQ_R2_MOD[12] = {
    0x1c341746u, 0xf4df1f34u, 0x09d104f1u, 0x0a76e6a6u,
    0x4c95b6d5u, 0x8de5476cu, 0x939d83c0u, 0x67eb88a9u,
    0xb519952du, 0x9a793e85u, 0x92cae3aau, 0x11988fe5u
};

// -q^{-1} mod 2^32 (Montgomery reduction constant)
__device__ static constexpr uint32_t FQ_P_INV = 0xfffcfffdu;

// ─── Field Element ──────────────────────────────────────────────────────────

struct __align__(16) FqElement {
    uint32_t limbs[12];  // 381-bit value, little-endian, Montgomery form

    __host__ __device__ __forceinline__
    static FqElement zero() {
        FqElement r;
        #pragma unroll
        for (int i = 0; i < 12; ++i) r.limbs[i] = 0;
        return r;
    }

    __host__ __device__ __forceinline__
    static FqElement one_mont() {
        FqElement r;
        #pragma unroll
        for (int i = 0; i < 12; ++i) r.limbs[i] = FQ_R_MOD[i];
        return r;
    }

    __host__ __device__ __forceinline__
    bool operator==(const FqElement& o) const {
        for (int i = 0; i < 12; ++i)
            if (limbs[i] != o.limbs[i]) return false;
        return true;
    }

    __host__ __device__ __forceinline__
    bool operator!=(const FqElement& o) const {
        return !(*this == o);
    }
};

// ─── Helper: compare >= modulus ─────────────────────────────────────────────

__device__ __forceinline__
bool fq_geq_modulus(const uint32_t* a) {
    for (int i = 11; i >= 0; --i) {
        if (a[i] > FQ_MODULUS[i]) return true;
        if (a[i] < FQ_MODULUS[i]) return false;
    }
    return true;  // equal
}

// ─── fq_add: (a + b) mod q ─────────────────────────────────────────────────

__device__ __forceinline__
FqElement fq_add(const FqElement& a, const FqElement& b) {
    FqElement result;
    uint32_t carry = 0;

    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)b.limbs[i] + carry;
        result.limbs[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }

    uint32_t need_sub = carry;
    if (!need_sub) {
        need_sub = fq_geq_modulus(result.limbs) ? 1u : 0u;
    }

    if (need_sub) {
        uint32_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            uint64_t diff = (uint64_t)result.limbs[i] - (uint64_t)FQ_MODULUS[i] - borrow;
            result.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 32) ? 1u : 0u;
        }
    }
    return result;
}

// ─── fq_sub: (a - b) mod q ─────────────────────────────────────────────────

__device__ __forceinline__
FqElement fq_sub(const FqElement& a, const FqElement& b) {
    FqElement result;
    uint32_t borrow = 0;

    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)b.limbs[i] - borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (diff >> 32) ? 1u : 0u;
    }

    if (borrow) {
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            uint64_t sum = (uint64_t)result.limbs[i] + (uint64_t)FQ_MODULUS[i] + carry;
            result.limbs[i] = (uint32_t)sum;
            carry = (uint32_t)(sum >> 32);
        }
    }
    return result;
}

// ─── fq_neg: -a mod q ──────────────────────────────────────────────────────

__device__ __forceinline__
FqElement fq_neg(const FqElement& a) {
    // Check if a == 0
    bool is_zero = true;
    for (int i = 0; i < 12; ++i) {
        if (a.limbs[i] != 0) { is_zero = false; break; }
    }
    if (is_zero) return FqElement::zero();

    // q - a
    FqElement result;
    uint32_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        uint64_t diff = (uint64_t)FQ_MODULUS[i] - (uint64_t)a.limbs[i] - borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (diff >> 32) ? 1u : 0u;
    }
    return result;
}

// ─── fq_mul: Montgomery multiplication (CIOS, 12-limb) ─────────────────────
// Computes MonPro(a, b) = a * b * R^{-1} mod q
// 12 outer iterations x 12 inner = 144 MADs per multiply.

__device__ __forceinline__
FqElement fq_mul(const FqElement& a, const FqElement& b) {
    uint32_t T[14] = {0};  // T[0..12] used, T[13] for overflow

    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        // Phase 1: T += a * b[i]
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 12; ++j) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)b.limbs[i]
                          + (uint64_t)T[j] + (uint64_t)carry;
            T[j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        uint64_t sum = (uint64_t)T[12] + (uint64_t)carry;
        T[12] = (uint32_t)sum;
        T[13] = (uint32_t)(sum >> 32);

        // Phase 2: Montgomery reduction
        uint32_t m = T[0] * FQ_P_INV;

        uint64_t carry2 = (uint64_t)T[0] + (uint64_t)m * (uint64_t)FQ_MODULUS[0];
        uint32_t C = (uint32_t)(carry2 >> 32);

        #pragma unroll
        for (int j = 1; j < 12; ++j) {
            carry2 = (uint64_t)T[j] + (uint64_t)m * (uint64_t)FQ_MODULUS[j] + (uint64_t)C;
            T[j - 1] = (uint32_t)carry2;
            C = (uint32_t)(carry2 >> 32);
        }

        sum = (uint64_t)T[12] + (uint64_t)C;
        T[11] = (uint32_t)sum;
        T[12] = T[13] + (uint32_t)(sum >> 32);
        T[13] = 0;
    }

    // Conditional subtraction
    FqElement result;
    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        result.limbs[i] = T[i];
    }

    uint32_t need_sub = (T[12] != 0) ? 1u : 0u;
    if (!need_sub) {
        need_sub = fq_geq_modulus(result.limbs) ? 1u : 0u;
    }

    if (need_sub) {
        uint32_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            uint64_t diff = (uint64_t)result.limbs[i] - (uint64_t)FQ_MODULUS[i] - borrow;
            result.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 32) ? 1u : 0u;
        }
    }
    return result;
}

// ─── fq_sqr: squaring via Montgomery multiplication ─────────────────────────

__device__ __forceinline__
FqElement fq_sqr(const FqElement& a) {
    return fq_mul(a, a);
}

// ─── Montgomery form conversions ────────────────────────────────────────────

__device__ __forceinline__
FqElement fq_to_montgomery(const FqElement& a) {
    FqElement r2;
    #pragma unroll
    for (int i = 0; i < 12; ++i) r2.limbs[i] = FQ_R2_MOD[i];
    return fq_mul(a, r2);
}

__device__ __forceinline__
FqElement fq_from_montgomery(const FqElement& a) {
    FqElement one;
    one.limbs[0] = 1;
    #pragma unroll
    for (int i = 1; i < 12; ++i) one.limbs[i] = 0;
    return fq_mul(a, one);
}

// ─── Modular Exponentiation (in Montgomery form) ───────────────────────────
// Binary square-and-multiply, LSB-first, 384-bit exponent.

__device__ __forceinline__
FqElement fq_pow(const FqElement& base, const uint32_t exp[12]) {
    FqElement result = FqElement::one_mont();
    FqElement b = base;
    for (int i = 0; i < 12; ++i) {
        uint32_t bits = exp[i];
        for (int j = 0; j < 32; ++j) {
            if (bits & 1) result = fq_mul(result, b);
            b = fq_sqr(b);
            bits >>= 1;
        }
    }
    return result;
}

// ─── Modular Inverse (Fermat's little theorem) ─────────────────────────────
// a^{-1} = a^{q-2} mod q.  Input/output in Montgomery form.

__device__ __forceinline__
FqElement fq_inv(const FqElement& a) {
    // q - 2 in 12 x uint32_t little-endian
    uint32_t q_minus_2[12] = {
        0xffffaaa9u, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu,
        0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
        0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
    };
    return fq_pow(a, q_minus_2);
}
