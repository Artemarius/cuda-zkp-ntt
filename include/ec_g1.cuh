// include/ec_g1.cuh
// BLS12-381 G1 elliptic curve point arithmetic over Fq.
// Curve: y^2 = x^3 + 4 (short Weierstrass, b = 4)
// Coordinates: Jacobian projective (X, Y, Z) where affine (x, y) = (X/Z^2, Y/Z^3)
// All field values in Montgomery form.

#pragma once
#include "ff_fq.cuh"

// ─── Curve Constant ─────────────────────────────────────────────────────────
// b = 4 in Montgomery form: 4*R mod q
__device__ static constexpr uint32_t G1_B_MONT[12] = {
    0x000cfff3u, 0xaa270000u, 0xfc34000au, 0x53cc0032u,
    0x6b0a807fu, 0x478fe97au, 0xe6ba24d7u, 0xb1d37ebeu,
    0xbf78ab2fu, 0x8ec9733bu, 0x3d83de7eu, 0x09d64551u
};

// ─── G1 Generator (Montgomery form) ────────────────────────────────────────
// Affine generator: the standard BLS12-381 G1 generator point.
__device__ static constexpr uint32_t G1_GEN_X[12] = {
    0xfd530c16u, 0x5cb38790u, 0x9976fff5u, 0x7817fc67u,
    0x143ba1c1u, 0x154f95c7u, 0xf3d0e747u, 0xf0ae6acdu,
    0x21dbf440u, 0xedce6eccu, 0x9e0bfb75u, 0x12017741u
};
__device__ static constexpr uint32_t G1_GEN_Y[12] = {
    0x0ce72271u, 0xbaac93d5u, 0x7918fd8eu, 0x8c22631au,
    0x570725ceu, 0xdd595f13u, 0x50405194u, 0x51ac5829u,
    0xad0059c0u, 0x0e1c8c3fu, 0x5008a26au, 0x0bbc3efcu
};

// ─── Point Types ────────────────────────────────────────────────────────────

struct G1Jacobian {
    FqElement x, y, z;

    __host__ __device__ __forceinline__
    static G1Jacobian identity() {
        // Point at infinity: (0, 1, 0) in Jacobian
        G1Jacobian p;
        p.x = FqElement::zero();
        p.y = FqElement::one_mont();  // R mod q
        p.z = FqElement::zero();
        return p;
    }

    __host__ __device__ __forceinline__
    bool is_identity() const {
        // Check if Z == 0
        for (int i = 0; i < 12; ++i)
            if (z.limbs[i] != 0) return false;
        return true;
    }
};

struct G1Affine {
    FqElement x, y;
    bool infinity;

    __host__ __device__ __forceinline__
    static G1Affine point_at_infinity() {
        G1Affine p;
        p.x = FqElement::zero();
        p.y = FqElement::zero();
        p.infinity = true;
        return p;
    }
};

// ─── Affine <-> Jacobian conversions ────────────────────────────────────────

__device__ __forceinline__
G1Jacobian g1_affine_to_jacobian(const G1Affine& p) {
    if (p.infinity) return G1Jacobian::identity();
    G1Jacobian j;
    j.x = p.x;
    j.y = p.y;
    j.z = FqElement::one_mont();
    return j;
}

__device__ __forceinline__
G1Affine g1_to_affine(const G1Jacobian& p) {
    if (p.is_identity()) return G1Affine::point_at_infinity();

    FqElement z_inv = fq_inv(p.z);
    FqElement z_inv2 = fq_sqr(z_inv);
    FqElement z_inv3 = fq_mul(z_inv2, z_inv);

    G1Affine a;
    a.x = fq_mul(p.x, z_inv2);
    a.y = fq_mul(p.y, z_inv3);
    a.infinity = false;
    return a;
}

// ─── g1_negate ──────────────────────────────────────────────────────────────

__device__ __forceinline__
G1Jacobian g1_negate(const G1Jacobian& p) {
    G1Jacobian r;
    r.x = p.x;
    r.y = fq_neg(p.y);
    r.z = p.z;
    return r;
}

// ─── g1_double (Jacobian) ───────────────────────────────────────────────────
// Algorithm: dbl-2001-b (4M + 4S)
// For a = 0 (BLS12-381 G1: y^2 = x^3 + 4, a = 0)
// delta = Z1^2, gamma = Y1^2, beta = X1*gamma
// alpha = 3*(X1-delta)*(X1+delta) = 3*X1^2 (since a=0)
// X3 = alpha^2 - 8*beta
// Y3 = alpha*(4*beta - X3) - 8*gamma^2
// Z3 = (Y1+Z1)^2 - gamma - delta

__device__ __forceinline__
G1Jacobian g1_double(const G1Jacobian& p) {
    if (p.is_identity()) return p;

    FqElement delta = fq_sqr(p.z);         // Z1^2
    FqElement gamma = fq_sqr(p.y);         // Y1^2
    FqElement beta = fq_mul(p.x, gamma);   // X1 * Y1^2

    // alpha = 3 * X1^2
    FqElement x2 = fq_sqr(p.x);
    FqElement alpha = fq_add(x2, fq_add(x2, x2));

    // X3 = alpha^2 - 8*beta
    FqElement alpha2 = fq_sqr(alpha);
    FqElement beta2 = fq_add(beta, beta);   // 2*beta
    FqElement beta4 = fq_add(beta2, beta2); // 4*beta
    FqElement beta8 = fq_add(beta4, beta4); // 8*beta
    FqElement x3 = fq_sub(alpha2, beta8);

    // Y3 = alpha*(4*beta - X3) - 8*gamma^2
    FqElement gamma2 = fq_sqr(gamma);
    FqElement gamma2_2 = fq_add(gamma2, gamma2);  // 2*gamma^2
    FqElement gamma2_4 = fq_add(gamma2_2, gamma2_2); // 4*gamma^2
    FqElement gamma2_8 = fq_add(gamma2_4, gamma2_4); // 8*gamma^2
    FqElement y3 = fq_sub(fq_mul(alpha, fq_sub(beta4, x3)), gamma2_8);

    // Z3 = (Y1+Z1)^2 - gamma - delta
    FqElement yz = fq_add(p.y, p.z);
    FqElement yz2 = fq_sqr(yz);
    FqElement z3 = fq_sub(fq_sub(yz2, gamma), delta);

    G1Jacobian r;
    r.x = x3;
    r.y = y3;
    r.z = z3;
    return r;
}

// ─── g1_add (Jacobian + Jacobian) ───────────────────────────────────────────
// Algorithm: add-2007-bl (12M + 4S)

__device__ __forceinline__
G1Jacobian g1_add(const G1Jacobian& p, const G1Jacobian& q) {
    if (p.is_identity()) return q;
    if (q.is_identity()) return p;

    FqElement z1_sq = fq_sqr(p.z);
    FqElement z2_sq = fq_sqr(q.z);

    FqElement u1 = fq_mul(p.x, z2_sq);         // X1 * Z2^2
    FqElement u2 = fq_mul(q.x, z1_sq);         // X2 * Z1^2
    FqElement s1 = fq_mul(p.y, fq_mul(q.z, z2_sq)); // Y1 * Z2^3
    FqElement s2 = fq_mul(q.y, fq_mul(p.z, z1_sq)); // Y2 * Z1^3

    if (u1 == u2) {
        if (s1 == s2) return g1_double(p);
        else return G1Jacobian::identity();
    }

    FqElement h = fq_sub(u2, u1);       // H = U2 - U1
    FqElement i = fq_add(h, h);
    i = fq_sqr(i);                       // I = (2H)^2
    FqElement j = fq_mul(h, i);          // J = H * I
    FqElement r = fq_sub(s2, s1);
    r = fq_add(r, r);                    // r = 2*(S2-S1)
    FqElement v = fq_mul(u1, i);         // V = U1 * I

    // X3 = r^2 - J - 2V
    FqElement r2 = fq_sqr(r);
    FqElement v2 = fq_add(v, v);
    FqElement x3 = fq_sub(fq_sub(r2, j), v2);

    // Y3 = r*(V-X3) - 2*S1*J
    FqElement s1j = fq_mul(s1, j);
    FqElement s1j2 = fq_add(s1j, s1j);
    FqElement y3 = fq_sub(fq_mul(r, fq_sub(v, x3)), s1j2);

    // Z3 = ((Z1+Z2)^2 - Z1^2 - Z2^2) * H
    FqElement z12 = fq_add(p.z, q.z);
    FqElement z12_sq = fq_sqr(z12);
    FqElement z3 = fq_mul(fq_sub(fq_sub(z12_sq, z1_sq), z2_sq), h);

    G1Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// ─── g1_add_mixed (Jacobian + Affine) ───────────────────────────────────────
// Optimized: Z2 = 1, so Z2^2 = 1, Z2^3 = 1. Cost: 8M + 3S.

__device__ __forceinline__
G1Jacobian g1_add_mixed(const G1Jacobian& p, const G1Affine& q_aff) {
    if (q_aff.infinity) return p;
    if (p.is_identity()) return g1_affine_to_jacobian(q_aff);

    FqElement z1_sq = fq_sqr(p.z);

    FqElement u1 = p.x;                                // X1 (already * 1)
    FqElement u2 = fq_mul(q_aff.x, z1_sq);            // X2 * Z1^2
    FqElement s1 = p.y;                                // Y1 (already * 1)
    FqElement s2 = fq_mul(q_aff.y, fq_mul(p.z, z1_sq)); // Y2 * Z1^3

    if (u1 == u2) {
        if (s1 == s2) return g1_double(p);
        else return G1Jacobian::identity();
    }

    FqElement h = fq_sub(u2, u1);
    FqElement hh = fq_sqr(h);
    FqElement i = fq_add(hh, hh);
    i = fq_add(i, i);                    // I = 4*H^2
    FqElement j = fq_mul(h, i);
    FqElement r = fq_sub(s2, s1);
    r = fq_add(r, r);                    // r = 2*(S2-S1)
    FqElement v = fq_mul(u1, i);

    FqElement r2 = fq_sqr(r);
    FqElement v2 = fq_add(v, v);
    FqElement x3 = fq_sub(fq_sub(r2, j), v2);

    FqElement s1j = fq_mul(s1, j);
    FqElement s1j2 = fq_add(s1j, s1j);
    FqElement y3 = fq_sub(fq_mul(r, fq_sub(v, x3)), s1j2);

    FqElement z3 = fq_add(p.z, h);
    z3 = fq_sub(fq_sqr(z3), fq_add(z1_sq, hh));

    G1Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// ─── g1_is_on_curve (Affine) ────────────────────────────────────────────────
// Check y^2 == x^3 + 4 (in Montgomery form)

__device__ __forceinline__
bool g1_is_on_curve(const G1Affine& p) {
    if (p.infinity) return true;

    FqElement y2 = fq_sqr(p.y);
    FqElement x3 = fq_mul(fq_sqr(p.x), p.x);
    FqElement b;
    #pragma unroll
    for (int i = 0; i < 12; ++i) b.limbs[i] = G1_B_MONT[i];
    FqElement rhs = fq_add(x3, b);
    return y2 == rhs;
}

// ─── g1_scalar_mul (double-and-add) ─────────────────────────────────────────
// Simple binary method. For testing only; MSM replaces for performance.
// scalar: 8 x uint32_t little-endian (Fr element in standard form)

__device__ __forceinline__
G1Jacobian g1_scalar_mul(const G1Jacobian& p, const uint32_t scalar[8]) {
    G1Jacobian result = G1Jacobian::identity();
    G1Jacobian base = p;

    for (int i = 0; i < 8; ++i) {
        uint32_t bits = scalar[i];
        for (int j = 0; j < 32; ++j) {
            if (bits & 1) result = g1_add(result, base);
            base = g1_double(base);
            bits >>= 1;
        }
    }
    return result;
}
