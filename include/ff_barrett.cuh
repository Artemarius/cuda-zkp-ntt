// include/ff_barrett.cuh
// Barrett modular arithmetic for BLS12-381 scalar field
// Standard-form in/out — no Montgomery domain conversion needed.
//
// Barrett reduction computes (a * b) mod p directly on standard-form integers,
// avoiding the to/from Montgomery conversion overhead (~3ms at n=2^22).
//
// Algorithm (HAC 14.42, adapted for 8x32-bit limbs):
//   z = a * b                             (full 512-bit schoolbook product)
//   q1 = z >> 224                         (k=8 words, shift right 7 words)
//   q2 = q1 * mu                          (9x9 word multiply)
//   q3 = q2 >> 288                        (shift right 9 words)
//   r1 = z mod 2^288                      (low 9 words of z)
//   r2 = (q3 * p) mod 2^288              (low 9 words of q3*p)
//   r = r1 - r2                           (mod 2^288)
//   while r >= p: r -= p                  (at most 2 times)

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "ff_arithmetic.cuh"  // FpElement, BLS12_381_MODULUS

// ─── Barrett Constant ────────────────────────────────────────────────────────
// mu = floor(2^512 / p), 9 x uint32_t little-endian
// p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// mu = 0x2355094edfede377c38b5dcb707e08ed365043eb4be4bad7142737a020c0d6393
// Verified: mu * p <= 2^512 < (mu+1) * p

__device__ static constexpr uint32_t BLS12_381_BARRETT_MU[9] = {
    0x0c0d6393u,  // limb[0]
    0x42737a02u,  // limb[1]
    0xbe4bad71u,  // limb[2]
    0x65043eb4u,  // limb[3]
    0x07e08ed3u,  // limb[4]
    0x38b5dcb7u,  // limb[5]
    0xfede377cu,  // limb[6]
    0x355094edu,  // limb[7]
    0x00000002u   // limb[8]
};

// ─── Barrett modular multiplication ──────────────────────────────────────────
// Computes (a * b) mod p where a, b are in standard form [0, p).
// Cost: ~209 multiply-accumulate operations (vs 128 for Montgomery CIOS).
// Benefit: no domain conversion needed — saves ~3ms per NTT at n=2^22.

__device__ __forceinline__
FpElement ff_mul_barrett(const FpElement& a, const FpElement& b) {
    // ── Step 1: Full 512-bit product z = a * b (schoolbook, 64 mads) ────────
    uint32_t z[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) z[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)b.limbs[i]
                          + (uint64_t)z[i + j] + (uint64_t)carry;
            z[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        z[i + 8] = carry;
    }

    // ── Step 2: Barrett quotient estimate ────────────────────────────────────
    // q1 = z >> 224 = z[7..15] (9 words)
    // q2 = q1 * mu (9 x 9 -> 18 words, schoolbook, 81 mads)
    // q3 = q2[9..16] (8 words — quotient estimate, since q3 < p < 2^255)

    uint32_t q2[18];
    #pragma unroll
    for (int i = 0; i < 18; ++i) q2[i] = 0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            uint64_t prod = (uint64_t)z[7 + j] * (uint64_t)BLS12_381_BARRETT_MU[i]
                          + (uint64_t)q2[i + j] + (uint64_t)carry;
            q2[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        q2[i + 9] = carry;
    }

    // q3[0..7] = q2[9..16]

    // ── Step 3: r2 = (q3 * p) mod 2^288 (low 9 words only, ≤43 mads) ──────
    uint32_t r2[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) r2[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0;
        // Compute contributions to r2[i..min(i+7, 8)]
        // j_limit: i + j < 9 → j < 9 - i; also j < 8 (p has 8 words)
        const int j_limit = (9 - i) < 8 ? (9 - i) : 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= j_limit) break;
            uint64_t prod = (uint64_t)q2[9 + i] * (uint64_t)BLS12_381_MODULUS[j]
                          + (uint64_t)r2[i + j] + (uint64_t)carry;
            r2[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        // Carry goes to r2[i + j_limit] if within range
        if (i + j_limit < 9) {
            r2[i + j_limit] += carry;
        }
    }

    // ── Step 4: r = z[0..8] - r2[0..8] (mod 2^288) ────────────────────────
    uint32_t r[9];
    {
        uint32_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            uint64_t diff = (uint64_t)z[i] - (uint64_t)r2[i] - (uint64_t)borrow;
            r[i] = (uint32_t)diff;
            borrow = (diff >> 63) ? 1u : 0u;
        }
        // If borrow, add 2^288 (r is mod 2^288, so just ignore overflow)
        // Actually: if r1 < r2 (mod 2^288), we need to add 2^288.
        // But since q3 ≤ true_quotient ≤ q3 + 2, r should be non-negative
        // in practice. If borrow occurs, it means r is in [2^288 - 2p, 2^288),
        // which wraps to the correct positive value.
    }

    // ── Step 5: Conditional subtraction (at most 2 times) ───────────────────
    // r might be up to 2p. Subtract p while r >= p.

    // First check: compare r[0..7] against modulus (r[8] should be 0 or small)
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        // Check if r >= p: r[8] > 0, or r[0..7] >= p
        uint32_t need_sub = (r[8] != 0) ? 1u : 0u;
        if (!need_sub) {
            uint32_t ge = 1;
            #pragma unroll
            for (int i = 7; i >= 0; --i) {
                if (r[i] > BLS12_381_MODULUS[i]) { ge = 1; break; }
                if (r[i] < BLS12_381_MODULUS[i]) { ge = 0; break; }
            }
            need_sub = ge;
        }

        if (need_sub) {
            uint32_t borrow = 0;
            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                uint32_t mod_i = (i < 8) ? BLS12_381_MODULUS[i] : 0u;
                uint64_t diff = (uint64_t)r[i] - (uint64_t)mod_i - (uint64_t)borrow;
                r[i] = (uint32_t)diff;
                borrow = (diff >> 63) ? 1u : 0u;
            }
        }
    }

    // ── Return result ───────────────────────────────────────────────────────
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result.limbs[i] = r[i];
    }
    return result;
}

// ─── Barrett squaring ────────────────────────────────────────────────────────
// Dedicated squaring optimization deferred; delegate to ff_mul_barrett.

__device__ __forceinline__
FpElement ff_sqr_barrett(const FpElement& a) {
    return ff_mul_barrett(a, a);
}
