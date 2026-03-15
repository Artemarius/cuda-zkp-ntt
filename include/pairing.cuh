// include/pairing.cuh
// BLS12-381 optimal Ate pairing — Miller loop.
// Public types and declarations for the pairing module.
//
// The Miller loop computes f_{|u|, Q}(P) where u = -0xd201000000010000.
// Uses affine G2 coordinates for the running point T.
// Does NOT include final exponentiation (Session 34).

#pragma once
#include "ff_fq12.cuh"
#include "ec_g1.cuh"
#include "ec_g2.cuh"

// ─── BLS12-381 Miller Loop Parameter ────────────────────────────────────────
// u = -0xd201000000010000 (64-bit, Hamming weight 5, negative)
static constexpr uint64_t BLS12_381_U_ABS = 0xd201000000010000ULL;

// ─── Line Coefficients ──────────────────────────────────────────────────────
// Sparse Fq12 element at positions (0, 3, 4) for mul_by_034.

struct LineCoeffs {
    Fq2Element c0;  // position 0: Fq12.c0.c0
    Fq2Element c3;  // position 3: Fq12.c1.c0
    Fq2Element c4;  // position 4: Fq12.c1.c1
};
