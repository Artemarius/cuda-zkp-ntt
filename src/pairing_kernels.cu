// src/pairing_kernels.cu
// BLS12-381 optimal Ate pairing — Miller loop + final exponentiation.
// Compiled WITHOUT RDC (separate TU, no cross-TU device linkage needed).
//
// NOTE: Heavy Fq12 operations are wrapped with __noinline__ to prevent cicc
// crash from explosive IR inlining in the 63-iteration Miller loop and the
// 4x exp_by_u calls in the final exponentiation hard part.

#include "pairing.cuh"

// ─── Noinline wrappers (prevent cicc IR explosion) ──────────────────────────

__device__ __noinline__
static FqElement fq_inv_noinline(const FqElement& a) {
    return fq_inv(a);
}

__device__ __noinline__
static Fq2Element fq2_inv_noinline(const Fq2Element& a) {
    FqElement n = fq2_norm(a);
    FqElement n_inv = fq_inv_noinline(n);
    FqElement c0 = fq_mul(a.c0, n_inv);
    FqElement c1 = fq_mul(fq_neg(a.c1), n_inv);
    return {c0, c1};
}

__device__ __noinline__
static Fq12Element fq12_sqr_noinline(const Fq12Element& a) {
    return fq12_sqr(a);
}

__device__ __noinline__
static Fq12Element fq12_mul_noinline(const Fq12Element& a, const Fq12Element& b) {
    return fq12_mul(a, b);
}

__device__ __noinline__
static Fq12Element fq12_inv_noinline(const Fq12Element& a) {
    return fq12_inv(a);
}

__device__ __noinline__
static Fq12Element fq12_mul_by_014_noinline(const Fq12Element& a,
                                             const Fq2Element& d0,
                                             const Fq2Element& d1,
                                             const Fq2Element& d4) {
    return fq12_mul_by_014(a, d0, d1, d4);
}

__device__ __noinline__
static Fq12Element fq12_frobenius_noinline(const Fq12Element& a, int power,
                                            const FrobeniusCoeffs& coeffs) {
    return fq12_frobenius_map(a, power, coeffs.fq6_c1, coeffs.fq6_c2, coeffs.fq12_w);
}

// ─── fq2_mul_by_fq: scale Fq2 element by Fq scalar ────────────────────────

__device__ __forceinline__
static Fq2Element fq2_mul_by_fq(const Fq2Element& a, const FqElement& s) {
    return {fq_mul(a.c0, s), fq_mul(a.c1, s)};
}

// ─── Doubling Step (affine T, affine P) ─────────────────────────────────────
// M-type twist: E': y² = x³ + 4(1+u).  Line at positions (0, 1, 4).
// d0 = 3·b'- yt²  where b' = 4(1+u), so 3·b' = 12(1+u)
// d1 = 3xt² · xP
// d4 = -2yt · yP

__device__ __noinline__
static LineCoeffs miller_double_step(G2Affine& T, const G1Affine& P) {
    Fq2Element xt = T.x, yt = T.y;

    Fq2Element xt2 = fq2_sqr(xt);
    Fq2Element three_xt2 = fq2_add(xt2, fq2_add(xt2, xt2));
    Fq2Element two_yt = fq2_add(yt, yt);
    Fq2Element lam = fq2_mul(three_xt2, fq2_inv_noinline(two_yt));

    Fq2Element xr = fq2_sub(fq2_sqr(lam), fq2_add(xt, xt));
    Fq2Element yr = fq2_sub(fq2_mul(lam, fq2_sub(xt, xr)), yt);
    T.x = xr;
    T.y = yr;

    // 12(1+u) in Montgomery form: {12_mont, 12_mont}
    FqElement one = FqElement::one_mont();
    FqElement two = fq_add(one, one);
    FqElement four = fq_add(two, two);
    FqElement twelve = fq_add(fq_add(four, four), four);
    Fq2Element twelve_beta = {twelve, twelve};  // 12·(1+u)

    Fq2Element yt2 = fq2_sqr(yt);

    LineCoeffs line;
    line.d0 = fq2_sub(twelve_beta, yt2);               // 12(1+u) - yt²
    line.d1 = fq2_mul_by_fq(three_xt2, P.x);           // 3xt² · xP
    line.d4 = fq2_neg(fq2_mul_by_fq(two_yt, P.y));     // -2yt · yP

    return line;
}

// ─── Addition Step (affine T + affine Q, affine P) ──────────────────────────
// M-type twist: line at positions (0, 1, 4).
// d0 = xq·yt - yq·xt
// d1 = (yq - yt) · xP
// d4 = (xt - xq) · yP

__device__ __noinline__
static LineCoeffs miller_add_step(G2Affine& T, const G2Affine& Q, const G1Affine& P) {
    Fq2Element xt = T.x, yt = T.y;
    Fq2Element xq = Q.x, yq = Q.y;

    Fq2Element dy = fq2_sub(yq, yt);
    Fq2Element dx = fq2_sub(xq, xt);
    Fq2Element lam = fq2_mul(dy, fq2_inv_noinline(dx));

    Fq2Element xr = fq2_sub(fq2_sub(fq2_sqr(lam), xt), xq);
    Fq2Element yr = fq2_sub(fq2_mul(lam, fq2_sub(xt, xr)), yt);
    T.x = xr;
    T.y = yr;

    LineCoeffs line;
    line.d0 = fq2_sub(fq2_mul(xq, yt), fq2_mul(yq, xt));  // xq·yt - yq·xt
    line.d1 = fq2_mul_by_fq(dy, P.x);                       // (yq-yt)·xP
    line.d4 = fq2_mul_by_fq(fq2_sub(xt, xq), P.y);         // (xt-xq)·yP

    return line;
}

// ─── Miller Loop ────────────────────────────────────────────────────────────

__device__ __noinline__
static Fq12Element miller_loop(const G1Affine& P, const G2Affine& Q) {
    if (P.infinity || Q.infinity) return Fq12Element::one_mont();

    Fq12Element f = Fq12Element::one_mont();
    G2Affine T = Q;

    for (int i = 62; i >= 0; --i) {
        f = fq12_sqr_noinline(f);

        LineCoeffs ld = miller_double_step(T, P);
        f = fq12_mul_by_014_noinline(f, ld.d0, ld.d1, ld.d4);

        if ((BLS12_381_U_ABS >> i) & 1) {
            LineCoeffs la = miller_add_step(T, Q, P);
            f = fq12_mul_by_014_noinline(f, la.d0, la.d1, la.d4);
        }
    }

    f = fq12_conjugate(f);
    return f;
}

// ─── exp_by_u: f^u via square-and-multiply ──────────────────────────────────
// u = -0xd201000000010000, |u| has Hamming weight 5.
// 63 squarings + 4 multiplications, then conjugate (u < 0).

__device__ __noinline__
static Fq12Element exp_by_u(const Fq12Element& f) {
    Fq12Element result = f;  // f^1 (bit 63 is MSB)
    for (int i = 62; i >= 0; --i) {
        result = fq12_sqr_noinline(result);
        if ((BLS12_381_U_ABS >> i) & 1) {
            result = fq12_mul_noinline(result, f);
        }
    }
    return fq12_conjugate(result);  // u < 0
}

// ─── Final Exponentiation ────────────────────────────────────────────────────
// Raises Miller loop output f to (q^12 - 1)/r.
// Easy part: f^((q^6 - 1)(q^2 + 1)) — conjugate, inverse, Frobenius
// Hard part: Hayashida-Hayasaka-Teruya (eprint 2020/875), gnark decomposition
//   exponent = 3 + (x²+q²-1)(q+x)(x-1)² where x = u

__device__ __noinline__
static Fq12Element final_exponentiation(const Fq12Element& f,
                                         const FrobeniusCoeffs& coeffs) {
    // ─── Easy part: f^((q^6 - 1)(q^2 + 1)) ─────────────────────────────
    Fq12Element t0 = fq12_conjugate(f);           // f^(q^6)
    Fq12Element f_inv = fq12_inv_noinline(f);     // f^(-1)
    t0 = fq12_mul_noinline(t0, f_inv);            // f^(q^6 - 1)

    Fq12Element t1 = fq12_frobenius_noinline(t0, 2, coeffs);
    Fq12Element result = fq12_mul_noinline(t1, t0);  // f^((q^6-1)(q^2+1))

    // ─── Hard part ──────────────────────────────────────────────────────
    t0 = fq12_sqr_noinline(result);               // result²
    t1 = exp_by_u(result);                         // result^x
    Fq12Element t2 = fq12_conjugate(result);       // result^(-1)
    t1 = fq12_mul_noinline(t1, t2);                // result^(x-1)
    t2 = exp_by_u(t1);                             // result^(x²-x)
    t1 = fq12_conjugate(t1);                       // result^(1-x)
    t1 = fq12_mul_noinline(t1, t2);                // result^((x-1)²)
    t2 = exp_by_u(t1);                             // result^(x(x-1)²)
    t1 = fq12_frobenius_noinline(t1, 1, coeffs);   // result^(q(x-1)²)
    t1 = fq12_mul_noinline(t1, t2);                // result^((q+x)(x-1)²)
    result = fq12_mul_noinline(result, t0);         // result³
    t0 = exp_by_u(t1);                             // ^(x(q+x)(x-1)²)
    t2 = exp_by_u(t0);                             // ^(x²(q+x)(x-1)²)
    t0 = fq12_frobenius_noinline(t1, 2, coeffs);   // ^(q²(q+x)(x-1)²)
    t1 = fq12_conjugate(t1);                       // ^(-(q+x)(x-1)²)
    t1 = fq12_mul_noinline(t1, t2);                // ^((x²-1)(q+x)(x-1)²)
    t1 = fq12_mul_noinline(t1, t0);                // ^((x²+q²-1)(q+x)(x-1)²)
    result = fq12_mul_noinline(result, t1);         // 3 + (x²+q²-1)(q+x)(x-1)²

    return result;
}

// ─── Miller Loop Test Kernel ────────────────────────────────────────────────

__global__ void miller_loop_kernel(const G1Affine* __restrict__ P,
                                    const G2Affine* __restrict__ Q,
                                    Fq12Element* __restrict__ out,
                                    uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = miller_loop(P[idx], Q[idx]);
}

// ─── Final Exponentiation Kernel ────────────────────────────────────────────

__global__ void final_exp_kernel(const Fq12Element* __restrict__ in,
                                  Fq12Element* __restrict__ out,
                                  const FrobeniusCoeffs* __restrict__ coeffs,
                                  uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = final_exponentiation(in[idx], *coeffs);
}

// ─── Full Pairing Kernel (Miller Loop + Final Exponentiation) ───────────────

__global__ void pairing_kernel(const G1Affine* __restrict__ P,
                                const G2Affine* __restrict__ Q,
                                Fq12Element* __restrict__ out,
                                const FrobeniusCoeffs* __restrict__ coeffs,
                                uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Fq12Element f = miller_loop(P[idx], Q[idx]);
        out[idx] = final_exponentiation(f, *coeffs);
    }
}
