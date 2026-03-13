// include/ec_g2.cuh
// BLS12-381 G2 elliptic curve point arithmetic over Fq2.
// Curve: y^2 = x^3 + 4(1+u) (sextic twist)
// Coordinates: Jacobian projective.
// All field values in Montgomery form.

#pragma once
#include "ff_fq2.cuh"
#include "ec_g1.cuh"  // for G1_B_MONT (b = 4 in Montgomery form)

// ─── G2 Curve Constant ──────────────────────────────────────────────────────
// b' = 4(1+u) in Fq2 Montgomery form: c0 = 4*R mod q, c1 = 4*R mod q
// (uses G1_B_MONT from ec_g1.cuh via ff_fq.cuh include chain)

// ─── G2 Generator (Montgomery form) ────────────────────────────────────────
// Standard BLS12-381 G2 generator (EIP-2537 coordinates).
__device__ static constexpr uint32_t G2_GEN_X_C0[12] = {
    0x02940a10u, 0xf5f28fa2u, 0x87b4961au, 0xb3f5fb26u,
    0x3e2ae580u, 0xa1a893b5u, 0x1a3caee9u, 0x9894999du,
    0x1863366bu, 0x6f67b763u, 0x4350bcd7u, 0x05819192u
};
__device__ static constexpr uint32_t G2_GEN_X_C1[12] = {
    0x9e23f606u, 0xa5a9c075u, 0xbccd60c3u, 0xaaa0c59du,
    0xe2867806u, 0x3bb17e18u, 0x8541b367u, 0x1b1ab6ccu,
    0xf2158547u, 0xc2b6ed0eu, 0x7360edf3u, 0x11922a09u
};
__device__ static constexpr uint32_t G2_GEN_Y_C0[12] = {
    0x60494c4au, 0x4c730af8u, 0x5e369c5au, 0x597cfa1fu,
    0xaa0a635au, 0xe7e6856cu, 0x6e0d495fu, 0xbbefb5e9u,
    0xf0ef25a2u, 0x07d3a975u, 0x7e80dae5u, 0x0083fd8eu
};
__device__ static constexpr uint32_t G2_GEN_Y_C1[12] = {
    0xdf64b05du, 0xadc0fc92u, 0x2b1461dcu, 0x18aa270au,
    0x3be4eba0u, 0x86adac6au, 0xc93da33au, 0x79495c4eu,
    0xa43ccaedu, 0xe7175850u, 0x63de1bf2u, 0x0b2bc2a1u
};

// ─── Point Types ────────────────────────────────────────────────────────────

struct G2Jacobian {
    Fq2Element x, y, z;

    __host__ __device__ __forceinline__
    static G2Jacobian identity() {
        G2Jacobian p;
        p.x = Fq2Element::zero();
        p.y = Fq2Element::one_mont();
        p.z = Fq2Element::zero();
        return p;
    }

    __host__ __device__ __forceinline__
    bool is_identity() const {
        return z == Fq2Element::zero();
    }
};

struct G2Affine {
    Fq2Element x, y;
    bool infinity;

    __host__ __device__ __forceinline__
    static G2Affine point_at_infinity() {
        G2Affine p;
        p.x = Fq2Element::zero();
        p.y = Fq2Element::zero();
        p.infinity = true;
        return p;
    }
};

// ─── Affine <-> Jacobian ────────────────────────────────────────────────────

__device__ __forceinline__
G2Jacobian g2_affine_to_jacobian(const G2Affine& p) {
    if (p.infinity) return G2Jacobian::identity();
    G2Jacobian j;
    j.x = p.x;
    j.y = p.y;
    j.z = Fq2Element::one_mont();
    return j;
}

__device__ __forceinline__
G2Affine g2_to_affine(const G2Jacobian& p) {
    if (p.is_identity()) return G2Affine::point_at_infinity();

    Fq2Element z_inv = fq2_inv(p.z);
    Fq2Element z_inv2 = fq2_sqr(z_inv);
    Fq2Element z_inv3 = fq2_mul(z_inv2, z_inv);

    G2Affine a;
    a.x = fq2_mul(p.x, z_inv2);
    a.y = fq2_mul(p.y, z_inv3);
    a.infinity = false;
    return a;
}

// ─── g2_negate ──────────────────────────────────────────────────────────────

__device__ __forceinline__
G2Jacobian g2_negate(const G2Jacobian& p) {
    G2Jacobian r;
    r.x = p.x;
    r.y = fq2_neg(p.y);
    r.z = p.z;
    return r;
}

// ─── g2_double ──────────────────────────────────────────────────────────────
// Same algorithm as G1 (a=0 for BLS12-381 twist).

__device__ __forceinline__
G2Jacobian g2_double(const G2Jacobian& p) {
    if (p.is_identity()) return p;

    Fq2Element delta = fq2_sqr(p.z);
    Fq2Element gamma = fq2_sqr(p.y);
    Fq2Element beta = fq2_mul(p.x, gamma);

    Fq2Element x2 = fq2_sqr(p.x);
    Fq2Element alpha = fq2_add(x2, fq2_add(x2, x2));  // 3*X1^2

    Fq2Element alpha2 = fq2_sqr(alpha);
    Fq2Element beta2 = fq2_add(beta, beta);
    Fq2Element beta4 = fq2_add(beta2, beta2);
    Fq2Element beta8 = fq2_add(beta4, beta4);
    Fq2Element x3 = fq2_sub(alpha2, beta8);

    Fq2Element gamma2 = fq2_sqr(gamma);
    Fq2Element gamma2_2 = fq2_add(gamma2, gamma2);
    Fq2Element gamma2_4 = fq2_add(gamma2_2, gamma2_2);
    Fq2Element gamma2_8 = fq2_add(gamma2_4, gamma2_4);
    Fq2Element y3 = fq2_sub(fq2_mul(alpha, fq2_sub(beta4, x3)), gamma2_8);

    Fq2Element yz = fq2_add(p.y, p.z);
    Fq2Element yz2 = fq2_sqr(yz);
    Fq2Element z3 = fq2_sub(fq2_sub(yz2, gamma), delta);

    G2Jacobian r;
    r.x = x3; r.y = y3; r.z = z3;
    return r;
}

// ─── g2_add (Jacobian + Jacobian) ───────────────────────────────────────────

__device__ __forceinline__
G2Jacobian g2_add(const G2Jacobian& p, const G2Jacobian& q) {
    if (p.is_identity()) return q;
    if (q.is_identity()) return p;

    Fq2Element z1_sq = fq2_sqr(p.z);
    Fq2Element z2_sq = fq2_sqr(q.z);

    Fq2Element u1 = fq2_mul(p.x, z2_sq);
    Fq2Element u2 = fq2_mul(q.x, z1_sq);
    Fq2Element s1 = fq2_mul(p.y, fq2_mul(q.z, z2_sq));
    Fq2Element s2 = fq2_mul(q.y, fq2_mul(p.z, z1_sq));

    if (u1 == u2) {
        if (s1 == s2) return g2_double(p);
        else return G2Jacobian::identity();
    }

    Fq2Element h = fq2_sub(u2, u1);
    Fq2Element i = fq2_add(h, h);
    i = fq2_sqr(i);
    Fq2Element j = fq2_mul(h, i);
    Fq2Element r = fq2_sub(s2, s1);
    r = fq2_add(r, r);
    Fq2Element v = fq2_mul(u1, i);

    Fq2Element r2 = fq2_sqr(r);
    Fq2Element v2 = fq2_add(v, v);
    Fq2Element x3 = fq2_sub(fq2_sub(r2, j), v2);

    Fq2Element s1j = fq2_mul(s1, j);
    Fq2Element s1j2 = fq2_add(s1j, s1j);
    Fq2Element y3 = fq2_sub(fq2_mul(r, fq2_sub(v, x3)), s1j2);

    Fq2Element z12 = fq2_add(p.z, q.z);
    Fq2Element z12_sq = fq2_sqr(z12);
    Fq2Element z3 = fq2_mul(fq2_sub(fq2_sub(z12_sq, z1_sq), z2_sq), h);

    G2Jacobian result;
    result.x = x3; result.y = y3; result.z = z3;
    return result;
}

// ─── g2_add_mixed (Jacobian + Affine) ───────────────────────────────────────
// Optimized: Z2 = 1. Cost: 8M + 3S over Fq2.

__device__ __forceinline__
G2Jacobian g2_add_mixed(const G2Jacobian& p, const G2Affine& q_aff) {
    if (q_aff.infinity) return p;
    if (p.is_identity()) return g2_affine_to_jacobian(q_aff);

    Fq2Element z1_sq = fq2_sqr(p.z);

    Fq2Element u1 = p.x;
    Fq2Element u2 = fq2_mul(q_aff.x, z1_sq);
    Fq2Element s1 = p.y;
    Fq2Element s2 = fq2_mul(q_aff.y, fq2_mul(p.z, z1_sq));

    if (u1 == u2) {
        if (s1 == s2) return g2_double(p);
        else return G2Jacobian::identity();
    }

    Fq2Element h = fq2_sub(u2, u1);
    Fq2Element hh = fq2_sqr(h);
    Fq2Element i = fq2_add(hh, hh);
    i = fq2_add(i, i);                    // I = 4*H^2
    Fq2Element j = fq2_mul(h, i);
    Fq2Element r = fq2_sub(s2, s1);
    r = fq2_add(r, r);                    // r = 2*(S2-S1)
    Fq2Element v = fq2_mul(u1, i);

    Fq2Element r2 = fq2_sqr(r);
    Fq2Element v2 = fq2_add(v, v);
    Fq2Element x3 = fq2_sub(fq2_sub(r2, j), v2);

    Fq2Element s1j = fq2_mul(s1, j);
    Fq2Element s1j2 = fq2_add(s1j, s1j);
    Fq2Element y3 = fq2_sub(fq2_mul(r, fq2_sub(v, x3)), s1j2);

    Fq2Element z3 = fq2_add(p.z, h);
    z3 = fq2_sub(fq2_sqr(z3), fq2_add(z1_sq, hh));

    G2Jacobian result;
    result.x = x3; result.y = y3; result.z = z3;
    return result;
}

// ─── g2_is_on_curve (Affine) ────────────────────────────────────────────────
// Check y^2 == x^3 + 4(1+u), where b' = 4(1+u) = {4_mont, 4_mont}

__device__ __forceinline__
bool g2_is_on_curve(const G2Affine& p) {
    if (p.infinity) return true;

    Fq2Element y2 = fq2_sqr(p.y);
    Fq2Element x3 = fq2_mul(fq2_sqr(p.x), p.x);

    // b' = 4(1+u): c0 = 4*R, c1 = 4*R
    FqElement b4;
    #pragma unroll
    for (int i = 0; i < 12; ++i) b4.limbs[i] = G1_B_MONT[i];
    Fq2Element b_prime = {b4, b4};

    Fq2Element rhs = fq2_add(x3, b_prime);
    return y2 == rhs;
}

// ─── g2_scalar_mul (double-and-add) ─────────────────────────────────────────
// scalar: 8 x uint32_t little-endian (Fr element in standard form)

__device__ __forceinline__
G2Jacobian g2_scalar_mul(const G2Jacobian& p, const uint32_t scalar[8]) {
    G2Jacobian result = G2Jacobian::identity();
    G2Jacobian base = p;

    for (int i = 0; i < 8; ++i) {
        uint32_t bits = scalar[i];
        for (int j = 0; j < 32; ++j) {
            if (bits & 1) result = g2_add(result, base);
            base = g2_double(base);
            bits >>= 1;
        }
    }
    return result;
}
