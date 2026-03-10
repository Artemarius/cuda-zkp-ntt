// include/ff_arithmetic.cuh
// Finite-field arithmetic for BLS12-381 scalar field
// Type definitions + stub inline implementations (Phase 1)
// Real implementations replace stubs in Phase 2/3

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ─── BLS12-381 Scalar Field Constants ────────────────────────────────────────
// All constants in little-endian 8 × uint32_t representation.

// Modulus r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// Note: __device__ required so these are accessible from device __forceinline__ functions.
// With -rdc=true, plain constexpr host arrays are not visible in device code.
__device__ static constexpr uint32_t BLS12_381_MODULUS[8] = {
    0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
    0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u
};

// Montgomery constant: R mod r  where R = 2^256
// = 0x1824b159acc5056f998c4fefecbc4ff55884b7fa0003480200000001fffffffe
__device__ static constexpr uint32_t BLS12_381_R_MOD[8] = {
    0xfffffffeu, 0x00000001u, 0x00034802u, 0x5884b7fau,
    0xecbc4ff5u, 0x998c4fefu, 0xacc5056fu, 0x1824b159u
};

// R^2 mod r (for converting to Montgomery form via MonMul(a, R2))
// = 0x0748d9d99f59ff1105d314967254398f2b6cedcb87925c23c999e990f3f29c6d
__device__ static constexpr uint32_t BLS12_381_R2_MOD[8] = {
    0xf3f29c6du, 0xc999e990u, 0x87925c23u, 0x2b6cedcbu,
    0x7254398fu, 0x05d31496u, 0x9f59ff11u, 0x0748d9d9u
};

// p^{-1} mod 2^32  (Montgomery reduction constant)
__device__ static constexpr uint32_t BLS12_381_P_INV = 0xffffffffu;

// ─── Field Element ───────────────────────────────────────────────────────────

struct __align__(32) FpElement {
    uint32_t limbs[8];  // 255-bit value, little-endian, Montgomery form

    __host__ __device__ __forceinline__
    static FpElement zero() {
        FpElement r;
        for (int i = 0; i < 8; ++i) r.limbs[i] = 0;
        return r;
    }

    __host__ __device__ __forceinline__
    bool operator==(const FpElement& o) const {
        for (int i = 0; i < 8; ++i)
            if (limbs[i] != o.limbs[i]) return false;
        return true;
    }

    __host__ __device__ __forceinline__
    bool operator!=(const FpElement& o) const {
        return !(*this == o);
    }
};

// ─── Device Inline Functions ─────────────────────────────────────────────────
// Phase 2: real implementations of finite-field arithmetic.
// All operations assume inputs are in Montgomery form and produce
// results in Montgomery form (except to/from_montgomery conversions).

// ─── ff_add: (a + b) mod p ──────────────────────────────────────────────────

__device__ __forceinline__
FpElement ff_add(const FpElement& a, const FpElement& b) {
    FpElement result;
    uint32_t carry = 0;

    // 8-limb addition with carry propagation
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)b.limbs[i] + carry;
        result.limbs[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }

    // Conditional subtraction: if carry != 0 or result >= modulus, subtract p
    // First, determine if result >= modulus via MSB-to-LSB comparison
    // If carry is set, result is definitely >= p (since p < 2^256)
    uint32_t need_sub = carry;

    if (!need_sub) {
        // Compare result vs modulus from MSB to LSB
        // ge = 1 means result >= modulus so far
        uint32_t ge = 1;
        #pragma unroll
        for (int i = 7; i >= 0; --i) {
            if (result.limbs[i] > BLS12_381_MODULUS[i]) {
                ge = 1;
                break;
            }
            if (result.limbs[i] < BLS12_381_MODULUS[i]) {
                ge = 0;
                break;
            }
            // equal: continue to next lower limb
        }
        need_sub = ge;
    }

    if (need_sub) {
        uint32_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)result.limbs[i] - (uint64_t)BLS12_381_MODULUS[i] - borrow;
            result.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 32) ? 1u : 0u;
        }
    }

    return result;
}

// ─── ff_sub: (a - b) mod p ──────────────────────────────────────────────────

__device__ __forceinline__
FpElement ff_sub(const FpElement& a, const FpElement& b) {
    FpElement result;
    uint32_t borrow = 0;

    // 8-limb subtraction with borrow propagation
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)b.limbs[i] - borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (diff >> 32) ? 1u : 0u;
    }

    // If borrow != 0, result underflowed: add modulus to wrap around
    if (borrow) {
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t sum = (uint64_t)result.limbs[i] + (uint64_t)BLS12_381_MODULUS[i] + carry;
            result.limbs[i] = (uint32_t)sum;
            carry = (uint32_t)(sum >> 32);
        }
    }

    return result;
}

// ─── ff_mul: Montgomery multiplication (CIOS algorithm) ─────────────────────
// Coarsely Integrated Operand Scanning for 8 × 32-bit limbs.
// Computes MonPro(a, b) = a * b * R^{-1} mod p
// where a, b are in Montgomery form.

__device__ __forceinline__
FpElement ff_mul(const FpElement& a, const FpElement& b) {
    uint32_t T[10] = {0};  // Accumulator: T[0..8] used, T[9] for overflow

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Phase 1: T += a * b[i]
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)b.limbs[i]
                          + (uint64_t)T[j] + (uint64_t)carry;
            T[j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        uint64_t sum = (uint64_t)T[8] + (uint64_t)carry;
        T[8] = (uint32_t)sum;
        T[9] = (uint32_t)(sum >> 32);

        // Phase 2: Montgomery reduction
        // m = T[0] * P_INV mod 2^32
        uint32_t m = T[0] * BLS12_381_P_INV;

        // Compute T = (T + m * p) >> 32
        // First limb: low 32 bits are guaranteed zero by Montgomery property
        uint64_t carry2 = (uint64_t)T[0] + (uint64_t)m * (uint64_t)BLS12_381_MODULUS[0];
        uint32_t C = (uint32_t)(carry2 >> 32);

        #pragma unroll
        for (int j = 1; j < 8; ++j) {
            carry2 = (uint64_t)T[j] + (uint64_t)m * (uint64_t)BLS12_381_MODULUS[j] + (uint64_t)C;
            T[j - 1] = (uint32_t)carry2;
            C = (uint32_t)(carry2 >> 32);
        }

        sum = (uint64_t)T[8] + (uint64_t)C;
        T[7] = (uint32_t)sum;
        T[8] = T[9] + (uint32_t)(sum >> 32);
        T[9] = 0;
    }

    // Conditional subtraction: if T[8] != 0 or T[0..7] >= modulus, subtract p
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result.limbs[i] = T[i];
    }

    uint32_t need_sub = (T[8] != 0) ? 1u : 0u;

    if (!need_sub) {
        // Compare T[0..7] vs modulus from MSB to LSB
        uint32_t ge = 1;
        #pragma unroll
        for (int i = 7; i >= 0; --i) {
            if (result.limbs[i] > BLS12_381_MODULUS[i]) {
                ge = 1;
                break;
            }
            if (result.limbs[i] < BLS12_381_MODULUS[i]) {
                ge = 0;
                break;
            }
        }
        need_sub = ge;
    }

    if (need_sub) {
        uint32_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)result.limbs[i] - (uint64_t)BLS12_381_MODULUS[i] - borrow;
            result.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 32) ? 1u : 0u;
        }
    }

    return result;
}

// ─── ff_mul_ptx: CIOS Montgomery mul with branchless conditional reduction ──
// Same CIOS loop as ff_mul (compiler generates good carry chains from uint64_t).
// Key optimization: branchless final reduction using PTX sub.cc chain + lop3 select.
// Eliminates divergent branches in the comparison and conditional subtract.

__device__ __forceinline__
FpElement ff_mul_ptx(const FpElement& a, const FpElement& b) {
    uint32_t T[10] = {0};

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Phase 1: T += a * b[i]
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)b.limbs[i]
                          + (uint64_t)T[j] + (uint64_t)carry;
            T[j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        uint64_t sum = (uint64_t)T[8] + (uint64_t)carry;
        T[8] = (uint32_t)sum;
        T[9] = (uint32_t)(sum >> 32);

        // Phase 2: Montgomery reduction
        uint32_t m = T[0] * BLS12_381_P_INV;

        uint64_t carry2 = (uint64_t)T[0] + (uint64_t)m * (uint64_t)BLS12_381_MODULUS[0];
        uint32_t C = (uint32_t)(carry2 >> 32);

        #pragma unroll
        for (int j = 1; j < 8; ++j) {
            carry2 = (uint64_t)T[j] + (uint64_t)m * (uint64_t)BLS12_381_MODULUS[j] + (uint64_t)C;
            T[j - 1] = (uint32_t)carry2;
            C = (uint32_t)(carry2 >> 32);
        }

        sum = (uint64_t)T[8] + (uint64_t)C;
        T[7] = (uint32_t)sum;
        T[8] = T[9] + (uint32_t)(sum >> 32);
        T[9] = 0;
    }

    // ─── Branchless conditional reduction ────────────────────────────────────
    // Compute S = T - p. If T >= p (no borrow), use S; else keep T.
    // All CC-dependent instructions MUST be in a single asm block (CC is not
    // preserved between separate asm() statements).
    uint32_t S[8];
    uint32_t mask;
    asm("sub.cc.u32   %0, %9,  %18;\n\t"
        "subc.cc.u32  %1, %10, %19;\n\t"
        "subc.cc.u32  %2, %11, %20;\n\t"
        "subc.cc.u32  %3, %12, %21;\n\t"
        "subc.cc.u32  %4, %13, %22;\n\t"
        "subc.cc.u32  %5, %14, %23;\n\t"
        "subc.cc.u32  %6, %15, %24;\n\t"
        "subc.cc.u32  %7, %16, %25;\n\t"
        "subc.u32     %8, %17, 0;\n\t"    // mask = T[8] - 0 - borrow
        "shr.s32      %8, %8, 31;"         // normalize to 0xFFFFFFFF or 0
        : "=r"(S[0]), "=r"(S[1]), "=r"(S[2]), "=r"(S[3]),
          "=r"(S[4]), "=r"(S[5]), "=r"(S[6]), "=r"(S[7]),
          "=r"(mask)
        : "r"(T[0]), "r"(T[1]), "r"(T[2]), "r"(T[3]),
          "r"(T[4]), "r"(T[5]), "r"(T[6]), "r"(T[7]), "r"(T[8]),
          "r"(BLS12_381_MODULUS[0]), "r"(BLS12_381_MODULUS[1]),
          "r"(BLS12_381_MODULUS[2]), "r"(BLS12_381_MODULUS[3]),
          "r"(BLS12_381_MODULUS[4]), "r"(BLS12_381_MODULUS[5]),
          "r"(BLS12_381_MODULUS[6]), "r"(BLS12_381_MODULUS[7])
    );

    // Select: lop3 0xD8 = (a & ~c) | (b & c). mask=0→S, mask=0xFFFFFFFF→T
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        asm("lop3.b32 %0, %1, %2, %3, 0xD8;"
            : "=r"(result.limbs[i])
            : "r"(S[i]), "r"(T[i]), "r"(mask));
    }

    return result;
}

// ─── ff_add_v2: branchless modular addition ─────────────────────────────────
// Compute a + b mod p. Always computes both sum and sum-p, selects branchlessly.

__device__ __forceinline__
FpElement ff_add_v2(const FpElement& a, const FpElement& b) {
    // Step 1: sum = a + b (256-bit, carry out)
    uint32_t sum[8], carry;
    asm("add.cc.u32   %0, %9,  %17;\n\t"
        "addc.cc.u32  %1, %10, %18;\n\t"
        "addc.cc.u32  %2, %11, %19;\n\t"
        "addc.cc.u32  %3, %12, %20;\n\t"
        "addc.cc.u32  %4, %13, %21;\n\t"
        "addc.cc.u32  %5, %14, %22;\n\t"
        "addc.cc.u32  %6, %15, %23;\n\t"
        "addc.cc.u32  %7, %16, %24;\n\t"
        "addc.u32     %8, 0, 0;"
        : "=r"(sum[0]), "=r"(sum[1]), "=r"(sum[2]), "=r"(sum[3]),
          "=r"(sum[4]), "=r"(sum[5]), "=r"(sum[6]), "=r"(sum[7]),
          "=r"(carry)
        : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]), "r"(a.limbs[3]),
          "r"(a.limbs[4]), "r"(a.limbs[5]), "r"(a.limbs[6]), "r"(a.limbs[7]),
          "r"(b.limbs[0]), "r"(b.limbs[1]), "r"(b.limbs[2]), "r"(b.limbs[3]),
          "r"(b.limbs[4]), "r"(b.limbs[5]), "r"(b.limbs[6]), "r"(b.limbs[7])
    );

    // Step 2: S = sum - p, mask from carry - borrow
    uint32_t S[8], mask;
    asm("sub.cc.u32   %0, %9,  %18;\n\t"
        "subc.cc.u32  %1, %10, %19;\n\t"
        "subc.cc.u32  %2, %11, %20;\n\t"
        "subc.cc.u32  %3, %12, %21;\n\t"
        "subc.cc.u32  %4, %13, %22;\n\t"
        "subc.cc.u32  %5, %14, %23;\n\t"
        "subc.cc.u32  %6, %15, %24;\n\t"
        "subc.cc.u32  %7, %16, %25;\n\t"
        "subc.u32     %8, %17, 0;\n\t"
        "shr.s32      %8, %8, 31;"
        : "=r"(S[0]), "=r"(S[1]), "=r"(S[2]), "=r"(S[3]),
          "=r"(S[4]), "=r"(S[5]), "=r"(S[6]), "=r"(S[7]),
          "=r"(mask)
        : "r"(sum[0]), "r"(sum[1]), "r"(sum[2]), "r"(sum[3]),
          "r"(sum[4]), "r"(sum[5]), "r"(sum[6]), "r"(sum[7]),
          "r"(carry),
          "r"(BLS12_381_MODULUS[0]), "r"(BLS12_381_MODULUS[1]),
          "r"(BLS12_381_MODULUS[2]), "r"(BLS12_381_MODULUS[3]),
          "r"(BLS12_381_MODULUS[4]), "r"(BLS12_381_MODULUS[5]),
          "r"(BLS12_381_MODULUS[6]), "r"(BLS12_381_MODULUS[7])
    );

    // mask=0→use S (reduced), mask=0xFFFFFFFF→use sum (no reduction)
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        asm("lop3.b32 %0, %1, %2, %3, 0xD8;"
            : "=r"(result.limbs[i])
            : "r"(S[i]), "r"(sum[i]), "r"(mask));
    }
    return result;
}

// ─── ff_sub_v2: branchless modular subtraction ──────────────────────────────
// Compute a - b mod p. Always computes both diff and diff+p, selects branchlessly.

__device__ __forceinline__
FpElement ff_sub_v2(const FpElement& a, const FpElement& b) {
    // Step 1: diff = a - b (256-bit), borrow = 0xFFFFFFFF if a < b
    uint32_t diff[8], borrow;
    asm("sub.cc.u32   %0, %9,  %17;\n\t"
        "subc.cc.u32  %1, %10, %18;\n\t"
        "subc.cc.u32  %2, %11, %19;\n\t"
        "subc.cc.u32  %3, %12, %20;\n\t"
        "subc.cc.u32  %4, %13, %21;\n\t"
        "subc.cc.u32  %5, %14, %22;\n\t"
        "subc.cc.u32  %6, %15, %23;\n\t"
        "subc.cc.u32  %7, %16, %24;\n\t"
        "subc.u32     %8, 0, 0;"
        : "=r"(diff[0]), "=r"(diff[1]), "=r"(diff[2]), "=r"(diff[3]),
          "=r"(diff[4]), "=r"(diff[5]), "=r"(diff[6]), "=r"(diff[7]),
          "=r"(borrow)
        : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]), "r"(a.limbs[3]),
          "r"(a.limbs[4]), "r"(a.limbs[5]), "r"(a.limbs[6]), "r"(a.limbs[7]),
          "r"(b.limbs[0]), "r"(b.limbs[1]), "r"(b.limbs[2]), "r"(b.limbs[3]),
          "r"(b.limbs[4]), "r"(b.limbs[5]), "r"(b.limbs[6]), "r"(b.limbs[7])
    );

    // Step 2: S = diff + p (correction)
    uint32_t S[8];
    asm("add.cc.u32   %0, %8,  %16;\n\t"
        "addc.cc.u32  %1, %9,  %17;\n\t"
        "addc.cc.u32  %2, %10, %18;\n\t"
        "addc.cc.u32  %3, %11, %19;\n\t"
        "addc.cc.u32  %4, %12, %20;\n\t"
        "addc.cc.u32  %5, %13, %21;\n\t"
        "addc.cc.u32  %6, %14, %22;\n\t"
        "addc.u32     %7, %15, %23;"
        : "=r"(S[0]), "=r"(S[1]), "=r"(S[2]), "=r"(S[3]),
          "=r"(S[4]), "=r"(S[5]), "=r"(S[6]), "=r"(S[7])
        : "r"(diff[0]), "r"(diff[1]), "r"(diff[2]), "r"(diff[3]),
          "r"(diff[4]), "r"(diff[5]), "r"(diff[6]), "r"(diff[7]),
          "r"(BLS12_381_MODULUS[0]), "r"(BLS12_381_MODULUS[1]),
          "r"(BLS12_381_MODULUS[2]), "r"(BLS12_381_MODULUS[3]),
          "r"(BLS12_381_MODULUS[4]), "r"(BLS12_381_MODULUS[5]),
          "r"(BLS12_381_MODULUS[6]), "r"(BLS12_381_MODULUS[7])
    );

    // borrow=0xFFFFFFFF → use S (corrected); borrow=0 → use diff
    // lop3 0xD8 with (diff, S, borrow): (diff & ~borrow) | (S & borrow)
    FpElement result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        asm("lop3.b32 %0, %1, %2, %3, 0xD8;"
            : "=r"(result.limbs[i])
            : "r"(diff[i]), "r"(S[i]), "r"(borrow));
    }
    return result;
}

// ─── ff_sqr: squaring via Montgomery multiplication ─────────────────────────
// Simple delegation; dedicated squaring optimization deferred to Phase 3.

__device__ __forceinline__
FpElement ff_sqr(const FpElement& a) {
    return ff_mul(a, a);
}

// ─── Montgomery form conversions ────────────────────────────────────────────

// Convert a standard-form element to Montgomery form: aR mod p
// Computed as MonMul(a, R^2 mod p)
__device__ __forceinline__
FpElement ff_to_montgomery(const FpElement& a) {
    FpElement r2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r2.limbs[i] = BLS12_381_R2_MOD[i];
    }
    return ff_mul(a, r2);
}

// Convert a Montgomery-form element back to standard form: a * R^{-1} mod p
// Computed as MonMul(a, 1)
__device__ __forceinline__
FpElement ff_from_montgomery(const FpElement& a) {
    FpElement one;
    one.limbs[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        one.limbs[i] = 0;
    }
    return ff_mul(a, one);
}
