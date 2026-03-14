// include/ff_fq12.cuh
// Quadratic extension field Fq12 = Fq6[w] / (w² − v) for BLS12-381.
// Elements: c0 + c1*w where c0, c1 in Fq6.
// Top of the pairing tower: Fq → Fq2 → Fq6 → Fq12.
//
// Key identity: w² = v (the Fq6 element {0, 1, 0}).
// Multiplying an Fq6 element by v is fq6_mul_by_nonresidue.
// Karatsuba multiplication: 3 Fq6 muls = 18 Fq2 muls = 54 Fq muls.
// Squaring (complex method): 2 Fq6 muls = 12 Fq2 muls = 36 Fq muls.

#pragma once
#include "ff_fq6.cuh"

// ─── Fq12 Element ───────────────────────────────────────────────────────────

struct Fq12Element {
    Fq6Element c0;  // constant term
    Fq6Element c1;  // coefficient of w

    __host__ __device__ __forceinline__
    static Fq12Element zero() {
        return {Fq6Element::zero(), Fq6Element::zero()};
    }

    __host__ __device__ __forceinline__
    static Fq12Element one_mont() {
        return {Fq6Element::one_mont(), Fq6Element::zero()};
    }

    __host__ __device__ __forceinline__
    bool operator==(const Fq12Element& o) const {
        return c0 == o.c0 && c1 == o.c1;
    }

    __host__ __device__ __forceinline__
    bool operator!=(const Fq12Element& o) const {
        return !(*this == o);
    }
};

// ─── fq12_add: (a + b) in Fq12 ─────────────────────────────────────────────

__device__ __forceinline__
Fq12Element fq12_add(const Fq12Element& a, const Fq12Element& b) {
    return {fq6_add(a.c0, b.c0), fq6_add(a.c1, b.c1)};
}

// ─── fq12_sub: (a - b) in Fq12 ─────────────────────────────────────────────

__device__ __forceinline__
Fq12Element fq12_sub(const Fq12Element& a, const Fq12Element& b) {
    return {fq6_sub(a.c0, b.c0), fq6_sub(a.c1, b.c1)};
}

// ─── fq12_neg: -a in Fq12 ──────────────────────────────────────────────────

__device__ __forceinline__
Fq12Element fq12_neg(const Fq12Element& a) {
    return {fq6_neg(a.c0), fq6_neg(a.c1)};
}

// ─── fq12_conjugate: conj(a0 + a1·w) = a0 - a1·w ──────────────────────────
// For unitary elements (on cyclotomic subgroup, norm=1), conjugate = inverse.

__device__ __forceinline__
Fq12Element fq12_conjugate(const Fq12Element& a) {
    return {a.c0, fq6_neg(a.c1)};
}

// ─── fq12_mul: Karatsuba multiplication in Fq12 ────────────────────────────
// (a0 + a1·w)(b0 + b1·w) = (a0·b0 + a1·b1·v) + ((a0+a1)·(b0+b1) - a0·b0 - a1·b1)·w
// where v·x = fq6_mul_by_nonresidue(x) since w² = v.
// Cost: 3 Fq6 muls = 54 Fq muls.

__device__ __forceinline__
Fq12Element fq12_mul(const Fq12Element& a, const Fq12Element& b) {
    Fq6Element v0 = fq6_mul(a.c0, b.c0);
    Fq6Element v1 = fq6_mul(a.c1, b.c1);

    Fq6Element c0 = fq6_add(v0, fq6_mul_by_nonresidue(v1));
    Fq6Element c1 = fq6_sub(fq6_sub(
        fq6_mul(fq6_add(a.c0, a.c1), fq6_add(b.c0, b.c1)), v0), v1);

    return {c0, c1};
}

// ─── fq12_sqr: squaring in Fq12 (complex method) ───────────────────────────
// Complex squaring (Gauss trick):
//   ab = a0 · a1
//   c0 = (a0 + a1)(a0 + v·a1) - ab - v·ab  = a0² + v·a1²
//   c1 = 2 · ab
// Cost: 2 Fq6 muls = 36 Fq muls (vs 54 for general mul).

__device__ __forceinline__
Fq12Element fq12_sqr(const Fq12Element& a) {
    Fq6Element ab = fq6_mul(a.c0, a.c1);
    Fq6Element v_a1 = fq6_mul_by_nonresidue(a.c1);

    Fq6Element c0 = fq6_sub(fq6_sub(
        fq6_mul(fq6_add(a.c0, a.c1), fq6_add(a.c0, v_a1)), ab),
        fq6_mul_by_nonresidue(ab));
    Fq6Element c1 = fq6_add(ab, ab);

    return {c0, c1};
}

// ─── fq12_inv: inverse in Fq12 ─────────────────────────────────────────────
// (a0 + a1·w)⁻¹ = (a0 - a1·w) / (a0² - v·a1²)
//   norm = a0² - v·a1²  (in Fq6)
//   result = (a0 · norm⁻¹, -a1 · norm⁻¹)

__device__ __forceinline__
Fq12Element fq12_inv(const Fq12Element& a) {
    Fq6Element a0_sq = fq6_sqr(a.c0);
    Fq6Element a1_sq = fq6_sqr(a.c1);
    Fq6Element norm = fq6_sub(a0_sq, fq6_mul_by_nonresidue(a1_sq));
    Fq6Element norm_inv = fq6_inv(norm);

    return {fq6_mul(a.c0, norm_inv), fq6_neg(fq6_mul(a.c1, norm_inv))};
}

// ─── fq12_mul_by_034: sparse multiply for Miller loop line functions ────────
// Multiplies by a sparse Fq12 element with structure:
//   b.c0 = (d0, 0, 0)   — only the Fq2 constant term of c0 is nonzero
//   b.c1 = (d3, d4, 0)  — c1 has c0 and c1 nonzero, c2 = 0
//
// Karatsuba decomposition:
//   v0 = a.c0 · b.c0 = fq6_scale(a.c0, d0)                     [3 Fq2 muls]
//   v1 = a.c1 · b.c1 = fq6_mul_by_01(a.c1, d3, d4)             [5 Fq2 muls]
//   c0 = v0 + v·v1
//   c1 = fq6_mul_by_01(a.c0+a.c1, d0+d3, d4) - v0 - v1         [5 Fq2 muls]
// Cost: 13 Fq2 muls = 39 Fq muls (vs 54 for general mul).

__device__ __forceinline__
Fq12Element fq12_mul_by_034(const Fq12Element& a,
                             const Fq2Element& d0,
                             const Fq2Element& d3,
                             const Fq2Element& d4) {
    Fq6Element v0 = fq6_scale(a.c0, d0);
    Fq6Element v1 = fq6_mul_by_01(a.c1, d3, d4);

    Fq6Element c0 = fq6_add(v0, fq6_mul_by_nonresidue(v1));

    Fq6Element sum_a = fq6_add(a.c0, a.c1);
    Fq2Element sum_d03 = fq2_add(d0, d3);
    Fq6Element c1 = fq6_sub(fq6_sub(
        fq6_mul_by_01(sum_a, sum_d03, d4), v0), v1);

    return {c0, c1};
}

// ─── fq12_frobenius_map: apply Frobenius endomorphism φ^power ───────────────
// φ^k(a0 + a1·w) = φ^k(a0) + φ^k(a1) · γ_w[k] · w
// where φ^k on Fq6 uses fq6_frobenius_map with power (k % 6),
// and γ_w[k] = β^((q^k-1)/6) ∈ Fq2 is the Fq12-specific Frobenius coefficient.
// Fq6 Frobenius has period 6, so we use k % 6 for the Fq6 part.
// Fq12 Frobenius has period 12, so fq12_frob_w has 12 entries.

__device__ __forceinline__
Fq12Element fq12_frobenius_map(const Fq12Element& a, int power,
                                const Fq2Element* fq6_frob_c1,
                                const Fq2Element* fq6_frob_c2,
                                const Fq2Element* fq12_frob_w) {
    int fq6_power = power % 6;
    Fq6Element c0 = fq6_frobenius_map(a.c0, fq6_power, fq6_frob_c1, fq6_frob_c2);
    Fq6Element c1 = fq6_frobenius_map(a.c1, fq6_power, fq6_frob_c1, fq6_frob_c2);
    c1 = fq6_scale(c1, fq12_frob_w[power]);

    return {c0, c1};
}
