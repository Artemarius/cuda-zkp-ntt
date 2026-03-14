// include/ff_fq6.cuh
// Cubic extension field Fq6 = Fq2[v] / (v^3 - β) for BLS12-381.
// Elements: c0 + c1*v + c2*v^2 where c0, c1, c2 in Fq2.
// β = (1+u) is the Fq2 non-residue.
// Used for pairing tower: Fq → Fq2 → Fq6 → Fq12.
//
// Key identity: v^3 = β = (1+u).
// Karatsuba-like multiplication: 6 Fq2 muls = 18 Fq muls.
// Squaring (CH-SQR2): 2 Fq2 muls + 3 Fq2 sqrs = 12 Fq muls.

#pragma once
#include "ff_fq2.cuh"

// ─── Fq6 Element ────────────────────────────────────────────────────────────

struct Fq6Element {
    Fq2Element c0;  // constant term
    Fq2Element c1;  // coefficient of v
    Fq2Element c2;  // coefficient of v^2

    __host__ __device__ __forceinline__
    static Fq6Element zero() {
        return {Fq2Element::zero(), Fq2Element::zero(), Fq2Element::zero()};
    }

    __host__ __device__ __forceinline__
    static Fq6Element one_mont() {
        return {Fq2Element::one_mont(), Fq2Element::zero(), Fq2Element::zero()};
    }

    __host__ __device__ __forceinline__
    bool operator==(const Fq6Element& o) const {
        return c0 == o.c0 && c1 == o.c1 && c2 == o.c2;
    }

    __host__ __device__ __forceinline__
    bool operator!=(const Fq6Element& o) const {
        return !(*this == o);
    }
};

// ─── fq6_add: (a + b) in Fq6 ───────────────────────────────────────────────

__device__ __forceinline__
Fq6Element fq6_add(const Fq6Element& a, const Fq6Element& b) {
    return {fq2_add(a.c0, b.c0), fq2_add(a.c1, b.c1), fq2_add(a.c2, b.c2)};
}

// ─── fq6_sub: (a - b) in Fq6 ───────────────────────────────────────────────

__device__ __forceinline__
Fq6Element fq6_sub(const Fq6Element& a, const Fq6Element& b) {
    return {fq2_sub(a.c0, b.c0), fq2_sub(a.c1, b.c1), fq2_sub(a.c2, b.c2)};
}

// ─── fq6_neg: -a in Fq6 ────────────────────────────────────────────────────

__device__ __forceinline__
Fq6Element fq6_neg(const Fq6Element& a) {
    return {fq2_neg(a.c0), fq2_neg(a.c1), fq2_neg(a.c2)};
}

// ─── fq6_mul_by_nonresidue: multiply by v ───────────────────────────────────
// (c0 + c1·v + c2·v²) · v = β·c2 + c0·v + c1·v²
// where β·x = fq2_mul_by_nonresidue(x) since v³ = β = (1+u).

__device__ __forceinline__
Fq6Element fq6_mul_by_nonresidue(const Fq6Element& a) {
    return {fq2_mul_by_nonresidue(a.c2), a.c0, a.c1};
}

// ─── fq6_mul: Karatsuba multiplication in Fq6 ──────────────────────────────
// v0 = a0·b0, v1 = a1·b1, v2 = a2·b2
// c0 = v0 + β·((a1+a2)·(b1+b2) - v1 - v2)
// c1 = (a0+a1)·(b0+b1) - v0 - v1 + β·v2
// c2 = (a0+a2)·(b0+b2) - v0 - v2 + v1
// Cost: 6 Fq2 muls = 18 Fq muls.

__device__ __forceinline__
Fq6Element fq6_mul(const Fq6Element& a, const Fq6Element& b) {
    Fq2Element v0 = fq2_mul(a.c0, b.c0);
    Fq2Element v1 = fq2_mul(a.c1, b.c1);
    Fq2Element v2 = fq2_mul(a.c2, b.c2);

    // c0 = v0 + β·((a1+a2)·(b1+b2) - v1 - v2)
    Fq2Element t0 = fq2_mul(fq2_add(a.c1, a.c2), fq2_add(b.c1, b.c2));
    t0 = fq2_sub(fq2_sub(t0, v1), v2);
    Fq2Element c0 = fq2_add(v0, fq2_mul_by_nonresidue(t0));

    // c1 = (a0+a1)·(b0+b1) - v0 - v1 + β·v2
    Fq2Element t1 = fq2_mul(fq2_add(a.c0, a.c1), fq2_add(b.c0, b.c1));
    t1 = fq2_sub(fq2_sub(t1, v0), v1);
    Fq2Element c1 = fq2_add(t1, fq2_mul_by_nonresidue(v2));

    // c2 = (a0+a2)·(b0+b2) - v0 - v2 + v1
    Fq2Element t2 = fq2_mul(fq2_add(a.c0, a.c2), fq2_add(b.c0, b.c2));
    Fq2Element c2 = fq2_add(fq2_sub(fq2_sub(t2, v0), v2), v1);

    return {c0, c1, c2};
}

// ─── fq6_sqr: squaring in Fq6 (Chung-Hasan SQR2) ──────────────────────────
// s0 = a0², ab = a0·a1, s1 = 2·ab,
// s2 = (a0 - a1 + a2)², bc = a1·a2, s3 = 2·bc, s4 = a2²
// c0 = s0 + β·s3
// c1 = s1 + β·s4
// c2 = s1 + s2 + s3 - s0 - s4
// Cost: 2 Fq2 muls + 3 Fq2 sqrs = 12 Fq muls.

__device__ __forceinline__
Fq6Element fq6_sqr(const Fq6Element& a) {
    Fq2Element s0 = fq2_sqr(a.c0);
    Fq2Element ab = fq2_mul(a.c0, a.c1);
    Fq2Element s1 = fq2_add(ab, ab);
    Fq2Element s2 = fq2_sqr(fq2_add(fq2_sub(a.c0, a.c1), a.c2));
    Fq2Element bc = fq2_mul(a.c1, a.c2);
    Fq2Element s3 = fq2_add(bc, bc);
    Fq2Element s4 = fq2_sqr(a.c2);

    Fq2Element c0 = fq2_add(s0, fq2_mul_by_nonresidue(s3));
    Fq2Element c1 = fq2_add(s1, fq2_mul_by_nonresidue(s4));
    Fq2Element c2 = fq2_add(fq2_sub(fq2_add(s1, s2), s0), fq2_sub(s3, s4));

    return {c0, c1, c2};
}

// ─── fq6_inv: inverse in Fq6 via norm to Fq2 ───────────────────────────────
// t0 = c0², t1 = c1², t2 = c2²
// t3 = c0·c1, t4 = c0·c2, t5 = c1·c2
// s0 = t0 - β·t5
// s1 = β·t2 - t3
// s2 = t1 - t4
// norm = c0·s0 + β·(c2·s1 + c1·s2)
// result = (s0, s1, s2) / norm

__device__ __forceinline__
Fq6Element fq6_inv(const Fq6Element& a) {
    Fq2Element t0 = fq2_sqr(a.c0);
    Fq2Element t1 = fq2_sqr(a.c1);
    Fq2Element t2 = fq2_sqr(a.c2);
    Fq2Element t3 = fq2_mul(a.c0, a.c1);
    Fq2Element t4 = fq2_mul(a.c0, a.c2);
    Fq2Element t5 = fq2_mul(a.c1, a.c2);

    Fq2Element s0 = fq2_sub(t0, fq2_mul_by_nonresidue(t5));
    Fq2Element s1 = fq2_sub(fq2_mul_by_nonresidue(t2), t3);
    Fq2Element s2 = fq2_sub(t1, t4);

    Fq2Element norm = fq2_add(
        fq2_mul(a.c0, s0),
        fq2_mul_by_nonresidue(
            fq2_add(fq2_mul(a.c2, s1), fq2_mul(a.c1, s2))
        )
    );

    Fq2Element norm_inv = fq2_inv(norm);

    return {fq2_mul(s0, norm_inv), fq2_mul(s1, norm_inv), fq2_mul(s2, norm_inv)};
}

// ─── fq6_mul_by_01: multiply by (b0 + b1·v) ────────────────────────────────
// Sparse multiply where b2 = 0. Used in Miller loop line evaluation.
// c0 = v0 + β·((a1+a2)·b1 - v1)
// c1 = (a0+a1)·(b0+b1) - v0 - v1
// c2 = (a0+a2)·b0 - v0 + v1
// Cost: 5 Fq2 muls (vs 6 for general mul).

__device__ __forceinline__
Fq6Element fq6_mul_by_01(const Fq6Element& a, const Fq2Element& b0, const Fq2Element& b1) {
    Fq2Element v0 = fq2_mul(a.c0, b0);
    Fq2Element v1 = fq2_mul(a.c1, b1);

    // c0 = v0 + β·((a1+a2)·b1 - v1)
    Fq2Element t0 = fq2_sub(fq2_mul(fq2_add(a.c1, a.c2), b1), v1);
    Fq2Element c0 = fq2_add(v0, fq2_mul_by_nonresidue(t0));

    // c1 = (a0+a1)·(b0+b1) - v0 - v1
    Fq2Element c1 = fq2_sub(fq2_sub(
        fq2_mul(fq2_add(a.c0, a.c1), fq2_add(b0, b1)), v0), v1);

    // c2 = (a0+a2)·b0 - v0 + v1
    Fq2Element c2 = fq2_add(fq2_sub(fq2_mul(fq2_add(a.c0, a.c2), b0), v0), v1);

    return {c0, c1, c2};
}

// ─── fq6_mul_by_1: multiply by b1·v ────────────────────────────────────────
// Sparse multiply where b0 = b2 = 0. Used in Miller loop.
// c0 = β·(a2·b1)
// c1 = a0·b1
// c2 = a1·b1
// Cost: 3 Fq2 muls.

__device__ __forceinline__
Fq6Element fq6_mul_by_1(const Fq6Element& a, const Fq2Element& b1) {
    Fq2Element c0 = fq2_mul_by_nonresidue(fq2_mul(a.c2, b1));
    Fq2Element c1 = fq2_mul(a.c0, b1);
    Fq2Element c2 = fq2_mul(a.c1, b1);
    return {c0, c1, c2};
}

// ─── fq6_scale: multiply Fq6 element by Fq2 scalar ─────────────────────────

__device__ __forceinline__
Fq6Element fq6_scale(const Fq6Element& a, const Fq2Element& s) {
    return {fq2_mul(a.c0, s), fq2_mul(a.c1, s), fq2_mul(a.c2, s)};
}

// ─── fq6_frobenius_map: apply Frobenius endomorphism φ^power ────────────────
// φ^k(c0 + c1·v + c2·v²) = φ^k(c0) + φ^k(c1)·γ₁[k]·v + φ^k(c2)·γ₂[k]·v²
// where φ^k on Fq2 = conjugation if k is odd, identity if even.
// γ₁[k] = β^((q^k-1)/3), γ₂[k] = γ₁[k]².
// Coefficients passed as device pointers (precomputed on host).

__device__ __forceinline__
Fq6Element fq6_frobenius_map(const Fq6Element& a, int power,
                              const Fq2Element* frob_c1,
                              const Fq2Element* frob_c2) {
    Fq2Element c0 = (power & 1) ? fq2_conjugate(a.c0) : a.c0;
    Fq2Element c1 = (power & 1) ? fq2_conjugate(a.c1) : a.c1;
    Fq2Element c2 = (power & 1) ? fq2_conjugate(a.c2) : a.c2;

    c1 = fq2_mul(c1, frob_c1[power]);
    c2 = fq2_mul(c2, frob_c2[power]);

    return {c0, c1, c2};
}
