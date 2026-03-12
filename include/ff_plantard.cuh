// include/ff_plantard.cuh
// Plantard modular arithmetic for BLS12-381 scalar field
//
// Plantard reduction (Plantard, IEEE TETC 2021):
//   PRedc(W, T) = floor(((floor(W*T*mu mod R^2 / R) + 1) * p) / R)
//   Output: r ≡ W * T * (-R^{-2}) mod p, where R = 2^256
//
// For NTT: precompute T = -w * R^2 mod p for each twiddle w.
// Then PRedc(a, T) = a * (-w*R^2) * (-R^{-2}) mod p = a * w mod p.
//
// NEGATIVE RESULT for BLS12-381 (256-bit, 8-limb):
//   Full Plantard requires 264 MADs (vs Montgomery CIOS 136 MADs).
//   The z*mu step (512x512 → low 512 bits) costs 136 MADs alone.
//   With 512-bit precomputed twiddles: 164 MADs but 2x memory footprint.
//   Plantard's advantage (eliminating one big-int multiply) only applies
//   to WORD-SIZE moduli (32/64 bits) where each operation is O(1).
//   For multi-limb 256-bit, both product and reduction are O(n^2).

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "ff_arithmetic.cuh"  // FpElement, BLS12_381_MODULUS

// ─── Plantard Constant ──────────────────────────────────────────────────────
// mu = p^{-1} mod 2^512, 16 x uint32_t little-endian
// Computed via Hensel lifting from p^{-1} mod 2^32 = 1.
// Verified: p * mu mod 2^512 == 1.

__device__ static constexpr uint32_t BLS12_381_PLANTARD_MU[16] = {
    0x00000001u,  // limb[0]
    0x00000001u,  // limb[1]
    0x0001a402u,  // limb[2]
    0xac45a400u,  // limb[3]
    0xfffb13f9u,  // limb[4]
    0xe7e4d3e8u,  // limb[5]
    0x2840d7c6u,  // limb[6]
    0xc2bbc54fu,  // limb[7]
    0xfe75c03fu,  // limb[8]
    0x126d1ba9u,  // limb[9]
    0x4a284c49u,  // limb[10]
    0xef9dd73bu,  // limb[11]
    0x84a126a4u,  // limb[12]
    0x81a47510u,  // limb[13]
    0xadd63363u,  // limb[14]
    0x72fe25abu   // limb[15]
};

// -R^2 mod p (for Plantard twiddle conversion: T = -w * R^2 mod p)
// = 0x6ca4cd798a437e372d66c371974d9e762850b637786bffdb3666166e0c0d6394
__device__ static constexpr uint32_t BLS12_381_NEG_R2_MOD[8] = {
    0x0c0d6394u,  // limb[0]
    0x3666166eu,  // limb[1]
    0x786bffdbu,  // limb[2]
    0x2850b637u,  // limb[3]
    0x974d9e76u,  // limb[4]
    0x2d66c371u,  // limb[5]
    0x8a437e37u,  // limb[6]
    0x6ca4cd79u   // limb[7]
};

// ─── Plantard modular multiplication ────────────────────────────────────────
// Computes PRedc(a, T) = a * T * (-R^{-2}) mod p
// where T is a Plantard-form twiddle: T = -w * R^2 mod p
// Result: a * w mod p (standard form in, standard form out)
//
// Algorithm:
//   Step 1: z = a * T (schoolbook 8x8, 64 MADs, 16 limbs)
//   Step 2: c = z * mu mod R^2 (schoolbook 16x16 low, 136 MADs, 16 limbs)
//   Step 3: q = c[8..15] + 1 (upper 256 bits of c, +1)
//   Step 4: r = floor(q * p / R) = upper 256 bits of q*p (64 MADs)
//   Step 5: if r == p, return 0; branchless select
//
// Total: ~264 MADs — significantly worse than Montgomery CIOS (136 MADs).

__device__ __forceinline__
FpElement ff_mul_plantard(const FpElement& a, const FpElement& t_plant) {
    // ── Step 1: Full 512-bit product z = a * t_plant (schoolbook, 64 MADs) ──
    uint32_t z[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) z[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)t_plant.limbs[i]
                          + (uint64_t)z[i + j] + (uint64_t)carry;
            z[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        z[i + 8] = carry;
    }

    // ── Step 2: c = z * mu mod R^2 (schoolbook 16x16, low 16 limbs, 136 MADs)
    uint32_t c[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = 0;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint32_t carry = 0;
        // Only accumulate terms where i+j < 16 (mod R^2 truncation)
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (i + j >= 16) break;
            uint64_t prod = (uint64_t)z[j] * (uint64_t)BLS12_381_PLANTARD_MU[i]
                          + (uint64_t)c[i + j] + (uint64_t)carry;
            c[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        // carry above limb 15 is discarded (mod R^2)
    }

    // ── Step 3: q = c[8..15] + 1 (upper 256 bits of c, plus 1) ─────────────
    uint32_t q[9]; // 256-bit + possible carry
    uint32_t q_carry = 1; // the "+1"
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = (uint64_t)c[8 + i] + (uint64_t)q_carry;
        q[i] = (uint32_t)sum;
        q_carry = (uint32_t)(sum >> 32);
    }
    q[8] = q_carry;

    // ── Step 4: r = floor(q * p / R) = upper 256 bits of q*p ────────────────
    // q is at most 257 bits (9 limbs), p is 256 bits (8 limbs)
    // Product is at most 513 bits (17 limbs). We need limbs 8..16.
    // But for carry accuracy, compute all 17 limbs.
    uint32_t qp[17];
    #pragma unroll
    for (int i = 0; i < 17; ++i) qp[i] = 0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (i + j >= 17) break;
            uint64_t prod = (uint64_t)q[i] * (uint64_t)BLS12_381_MODULUS[j]
                          + (uint64_t)qp[i + j] + (uint64_t)carry;
            qp[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        if (i + 8 < 17) qp[i + 8] += carry;
    }

    // r = qp[8..15] (upper 256 bits of q*p)
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result.limbs[i] = qp[8 + i];
    }

    // ── Step 5: if r == p, return 0 (branchless) ────────────────────────────
    // Plantard guarantees r ∈ [0, p]. If r == p, set to 0.
    // Check: result == MODULUS? Use branchless comparison.
    uint32_t eq = 1;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        eq &= (result.limbs[i] == BLS12_381_MODULUS[i]) ? 1u : 0u;
    }
    if (eq) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) result.limbs[i] = 0;
    }

    return result;
}

// ─── Plantard squaring (via Plantard multiply) ──────────────────────────────
// Not useful in NTT (no self-multiply), but included for completeness.
// Note: squaring with Plantard requires a = a_plant form, which is unusual.
// This function computes PRedc(a, a) = a^2 * (-R^{-2}) mod p (NOT a^2 mod p).

__device__ __forceinline__
FpElement ff_sqr_plantard(const FpElement& a) {
    return ff_mul_plantard(a, a);
}
