// src/pairing_kernels.cu
// BLS12-381 optimal Ate pairing — Miller loop device functions + test kernel.
// Compiled WITHOUT RDC (separate TU, no cross-TU device linkage needed).
//
// NOTE: fq12_sqr, fq12_mul_by_034, fq2_inv, and fq_inv are wrapped with
// __noinline__ to prevent cicc crash. The forceinlined Fq12/Fq operations
// generate too much IR when composed in the 63-iteration Miller loop body.

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
static Fq12Element fq12_mul_by_034_noinline(const Fq12Element& a,
                                             const Fq2Element& d0,
                                             const Fq2Element& d3,
                                             const Fq2Element& d4) {
    return fq12_mul_by_034(a, d0, d3, d4);
}

// ─── fq2_mul_by_fq: scale Fq2 element by Fq scalar ────────────────────────

__device__ __forceinline__
static Fq2Element fq2_mul_by_fq(const Fq2Element& a, const FqElement& s) {
    return {fq_mul(a.c0, s), fq_mul(a.c1, s)};
}

// ─── Doubling Step (affine T, affine P) ─────────────────────────────────────
// Line coefficients from tangent at T evaluated at P via D-type twist:
//   c0 = 2·yt · yP,  c3 = -3·xt² · xP,  c4 = 3·xt³ - 2·yt²

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

    LineCoeffs line;
    line.c0 = fq2_mul_by_fq(two_yt, P.y);
    line.c3 = fq2_neg(fq2_mul_by_fq(three_xt2, P.x));
    Fq2Element xt3 = fq2_mul(xt2, xt);
    Fq2Element three_xt3 = fq2_add(xt3, fq2_add(xt3, xt3));
    Fq2Element yt2 = fq2_sqr(yt);
    Fq2Element two_yt2 = fq2_add(yt2, yt2);
    line.c4 = fq2_sub(three_xt3, two_yt2);

    return line;
}

// ─── Addition Step (affine T + affine Q, affine P) ──────────────────────────
// Line coefficients from chord through T,Q evaluated at P via D-type twist:
//   c0 = (xq-xt)·yP,  c3 = -(yq-yt)·xP,  c4 = (yq-yt)·xt - (xq-xt)·yt

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
    line.c0 = fq2_mul_by_fq(dx, P.y);
    line.c3 = fq2_neg(fq2_mul_by_fq(dy, P.x));
    line.c4 = fq2_sub(fq2_mul(dy, xt), fq2_mul(dx, yt));

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
        f = fq12_mul_by_034_noinline(f, ld.c0, ld.c3, ld.c4);

        if ((BLS12_381_U_ABS >> i) & 1) {
            LineCoeffs la = miller_add_step(T, Q, P);
            f = fq12_mul_by_034_noinline(f, la.c0, la.c3, la.c4);
        }
    }

    f = fq12_conjugate(f);
    return f;
}

// ─── Miller Loop Test Kernel ────────────────────────────────────────────────

__global__ void miller_loop_kernel(const G1Affine* __restrict__ P,
                                    const G2Affine* __restrict__ Q,
                                    Fq12Element* __restrict__ out,
                                    uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = miller_loop(P[idx], Q[idx]);
}
