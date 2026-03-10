// include/ff_arithmetic.cuh
// Finite-field arithmetic for BLS12-381 scalar field
// Phase 2 stub

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// BLS12-381 scalar modulus r (little-endian, 8 × uint32_t)
// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
static constexpr uint32_t BLS12_381_MODULUS[8] = {
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u
};

// Montgomery constant: R mod r = 2^256 mod r
static constexpr uint32_t BLS12_381_R_MOD[8] = {
    0xfffffffe, 0x01fffffe, 0x02000000, 0x88f70000,
    0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x1824b159  // placeholder — verify
};

// p^{-1} mod 2^32 for Montgomery reduction
static constexpr uint32_t BLS12_381_P_INV = 0xffffffffu;

// ─── Field Element ────────────────────────────────────────────────────────────

struct __align__(32) FpElement {
    uint32_t limbs[8];  // 255-bit, little-endian, Montgomery form

    __host__ __device__ __forceinline__
    static FpElement zero() {
        FpElement r;
        for (int i = 0; i < 8; ++i) r.limbs[i] = 0;
        return r;
    }

    __host__ __device__ __forceinline__
    bool operator==(const FpElement& o) const {
        for (int i = 0; i < 8; ++i) if (limbs[i] != o.limbs[i]) return false;
        return true;
    }
};

// ─── Device Function Declarations ─────────────────────────────────────────────
// (implemented in src/ff_mul.cu — Phase 2/3)

__device__ __forceinline__ FpElement ff_add(const FpElement& a, const FpElement& b);
__device__ __forceinline__ FpElement ff_sub(const FpElement& a, const FpElement& b);
__device__ __forceinline__ FpElement ff_mul(const FpElement& a, const FpElement& b);
__device__ __forceinline__ FpElement ff_sqr(const FpElement& a);
