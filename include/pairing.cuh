// include/pairing.cuh
// BLS12-381 optimal Ate pairing — Miller loop + final exponentiation.
// Public types and declarations for the pairing module.
//
// The Miller loop computes f_{|u|, Q}(P) where u = -0xd201000000010000.
// Uses affine G2 coordinates for the running point T.
// Final exponentiation raises f to (q^12 - 1)/r.

#pragma once
#include "ff_fq12.cuh"
#include "ec_g1.cuh"
#include "ec_g2.cuh"

// ─── BLS12-381 Miller Loop Parameter ────────────────────────────────────────
// u = -0xd201000000010000 (64-bit, Hamming weight 5, negative)
static constexpr uint64_t BLS12_381_U_ABS = 0xd201000000010000ULL;

// ─── Line Coefficients ──────────────────────────────────────────────────────
// Sparse Fq12 element at positions (0, 1, 4) for mul_by_014.
// M-type sextic twist: E': y² = x³ + 4(1+u).

struct LineCoeffs {
    Fq2Element d0;  // position 0: Fq12.c0.c0
    Fq2Element d1;  // position 1: Fq12.c0.c1
    Fq2Element d4;  // position 4: Fq12.c1.c1
};

// ─── Frobenius Coefficients ─────────────────────────────────────────────────
// Precomputed on host, uploaded to device for final exponentiation.
// Contains all coefficients needed for Fq6 and Fq12 Frobenius maps.

struct FrobeniusCoeffs {
    Fq2Element fq6_c1[6];   // γ₁[k] for Fq6 Frobenius, k=0..5
    Fq2Element fq6_c2[6];   // γ₂[k] = γ₁[k]² for Fq6 Frobenius, k=0..5
    Fq2Element fq12_w[12];  // γ_w[k] = β^((q^k-1)/6) for Fq12 Frobenius, k=0..11
};
