// include/ff_fq2.cuh
// Quadratic extension field Fq2 = Fq[u] / (u^2 + 1) for BLS12-381.
// Elements: c0 + c1*u where c0, c1 in Fq.
// Used for G2 point arithmetic (curve over Fq2).
//
// Key identity: u^2 = -1.
// Karatsuba multiplication: 3 Fq muls + 5 Fq add/subs.

#pragma once
#include "ff_fq.cuh"

// ─── Fq2 Element ────────────────────────────────────────────────────────────

struct Fq2Element {
    FqElement c0;  // real part
    FqElement c1;  // imaginary part (coefficient of u)

    __host__ __device__ __forceinline__
    static Fq2Element zero() {
        return {FqElement::zero(), FqElement::zero()};
    }

    __host__ __device__ __forceinline__
    static Fq2Element one_mont() {
        return {FqElement::one_mont(), FqElement::zero()};
    }

    __host__ __device__ __forceinline__
    bool operator==(const Fq2Element& o) const {
        return c0 == o.c0 && c1 == o.c1;
    }

    __host__ __device__ __forceinline__
    bool operator!=(const Fq2Element& o) const {
        return !(*this == o);
    }
};

// ─── fq2_add: (a + b) in Fq2 ───────────────────────────────────────────────

__device__ __forceinline__
Fq2Element fq2_add(const Fq2Element& a, const Fq2Element& b) {
    return {fq_add(a.c0, b.c0), fq_add(a.c1, b.c1)};
}

// ─── fq2_sub: (a - b) in Fq2 ───────────────────────────────────────────────

__device__ __forceinline__
Fq2Element fq2_sub(const Fq2Element& a, const Fq2Element& b) {
    return {fq_sub(a.c0, b.c0), fq_sub(a.c1, b.c1)};
}

// ─── fq2_neg: -a in Fq2 ────────────────────────────────────────────────────

__device__ __forceinline__
Fq2Element fq2_neg(const Fq2Element& a) {
    return {fq_neg(a.c0), fq_neg(a.c1)};
}

// ─── fq2_conjugate: conj(a + bu) = a - bu ──────────────────────────────────

__device__ __forceinline__
Fq2Element fq2_conjugate(const Fq2Element& a) {
    return {a.c0, fq_neg(a.c1)};
}

// ─── fq2_mul: Karatsuba multiplication in Fq2 ──────────────────────────────
// (a0 + a1*u) * (b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
//
// Karatsuba: let v0 = a0*b0, v1 = a1*b1
//   c0 = v0 - v1
//   c1 = (a0+a1)*(b0+b1) - v0 - v1
// Cost: 3 Fq muls + 5 Fq add/subs (vs 4 muls schoolbook)

__device__ __forceinline__
Fq2Element fq2_mul(const Fq2Element& a, const Fq2Element& b) {
    FqElement v0 = fq_mul(a.c0, b.c0);  // a0 * b0
    FqElement v1 = fq_mul(a.c1, b.c1);  // a1 * b1

    // c0 = v0 - v1  (since u^2 = -1)
    FqElement c0 = fq_sub(v0, v1);

    // c1 = (a0+a1)*(b0+b1) - v0 - v1
    FqElement a01 = fq_add(a.c0, a.c1);
    FqElement b01 = fq_add(b.c0, b.c1);
    FqElement c1 = fq_mul(a01, b01);
    c1 = fq_sub(c1, v0);
    c1 = fq_sub(c1, v1);

    return {c0, c1};
}

// ─── fq2_sqr: squaring in Fq2 ──────────────────────────────────────────────
// (a + bu)^2 = (a^2 - b^2) + 2ab*u = (a+b)(a-b) + 2ab*u
// Cost: 2 Fq muls + 3 Fq add/subs (vs 3 muls for Karatsuba)

__device__ __forceinline__
Fq2Element fq2_sqr(const Fq2Element& a) {
    FqElement apb = fq_add(a.c0, a.c1);  // a + b
    FqElement amb = fq_sub(a.c0, a.c1);  // a - b
    FqElement c0 = fq_mul(apb, amb);     // (a+b)(a-b) = a^2 - b^2

    FqElement ab = fq_mul(a.c0, a.c1);   // a * b
    FqElement c1 = fq_add(ab, ab);       // 2 * a * b

    return {c0, c1};
}

// ─── fq2_mul_by_nonresidue: multiply by (1 + u) ────────────────────────────
// Used in sextic twist for G2.
// (a + bu)(1 + u) = (a - b) + (a + b)u

__device__ __forceinline__
Fq2Element fq2_mul_by_nonresidue(const Fq2Element& a) {
    FqElement c0 = fq_sub(a.c0, a.c1);
    FqElement c1 = fq_add(a.c0, a.c1);
    return {c0, c1};
}

// ─── fq2_norm: |a|^2 = a0^2 + a1^2 (element of Fq) ────────────────────────
// Since u^2 = -1: norm(a + bu) = a*conj(a) = (a+bu)(a-bu) = a^2 + b^2

__device__ __forceinline__
FqElement fq2_norm(const Fq2Element& a) {
    FqElement a2 = fq_sqr(a.c0);
    FqElement b2 = fq_sqr(a.c1);
    return fq_add(a2, b2);
}

// ─── fq2_inv: inverse in Fq2 ───────────────────────────────────────────────
// (a + bu)^{-1} = conj(a + bu) / norm(a + bu)
//               = (a - bu) / (a^2 + b^2)
// Cost: 1 Fq inverse + 2 Fq sqr + 1 Fq add + 2 Fq mul + 1 Fq neg

__device__ __forceinline__
Fq2Element fq2_inv(const Fq2Element& a) {
    FqElement n = fq2_norm(a);       // a^2 + b^2
    FqElement n_inv = fq_inv(n);     // 1 / (a^2 + b^2)
    FqElement c0 = fq_mul(a.c0, n_inv);
    FqElement c1 = fq_mul(fq_neg(a.c1), n_inv);
    return {c0, c1};
}

// ─── fq2_scale: multiply Fq2 element by Fq scalar ──────────────────────────

__device__ __forceinline__
Fq2Element fq2_scale(const Fq2Element& a, const FqElement& s) {
    return {fq_mul(a.c0, s), fq_mul(a.c1, s)};
}
