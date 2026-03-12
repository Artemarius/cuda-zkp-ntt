// include/twiddle_otf.cuh
// On-the-fly twiddle computation for NTT outer-stage kernels
//
// Instead of loading twiddles from a precomputed n/2-entry global memory table
// (64 MB at n=2^22), computes twiddles on-the-fly from per-stage roots stored
// in __constant__ memory (< 1 KB total). Outer stages are DRAM-bound, so the
// extra compute is hidden by memory latency.
//
// Radix-8 derivation: 1 exponentiation + 6 multiplies for 7 twiddles.
//   w4 = stage_root[s+2]^j       (exponentiation)
//   w2 = w4^2, w1 = w2^2         (2 squarings)
//   w3 = w2 * omega_4             (field constant: primitive 4th root)
//   w5 = w4 * omega_8             (field constant: primitive 8th root)
//   w6 = w4 * omega_4
//   w7 = w4 * omega_8^3
//
// Radix-4 derivation: 1 exponentiation + 2 multiplies for 3 twiddles.
//   w2 = stage_root[s+1]^j       (exponentiation)
//   w1 = w2^2                     (1 squaring)
//   w3 = w2 * omega_4             (1 multiply)

#pragma once
#include "ff_arithmetic.cuh"
#include "ff_barrett.cuh"

// ─── Montgomery modular exponentiation (uint32_t exponent) ──────────────────
// Computes base^exp mod p in Montgomery form using MSB-first binary method.

__device__ __forceinline__
FpElement ff_pow_mont_u32(const FpElement& base, uint32_t exp) {
    if (exp == 0) {
        FpElement one;
        #pragma unroll
        for (int i = 0; i < 8; ++i) one.limbs[i] = BLS12_381_R_MOD[i];
        return one;
    }

    // MSB-first binary exponentiation
    // __clz returns count of leading zeros for uint32_t
    int msb = 31 - __clz(exp);

    // Start with result = base (accounts for the leading 1 bit)
    FpElement result = base;
    for (int bit = msb - 1; bit >= 0; --bit) {
        result = ff_mul_ptx(result, result);    // square
        if ((exp >> bit) & 1) {
            result = ff_mul_ptx(result, base);  // multiply
        }
    }
    return result;
}

// ─── Barrett modular exponentiation (uint32_t exponent) ─────────────────────
// Computes base^exp mod p in standard form using MSB-first binary method.

__device__ __forceinline__
FpElement ff_pow_barrett_u32(const FpElement& base, uint32_t exp) {
    if (exp == 0) {
        FpElement one = FpElement::zero();
        one.limbs[0] = 1;  // standard form identity
        return one;
    }

    int msb = 31 - __clz(exp);

    FpElement result = base;
    for (int bit = msb - 1; bit >= 0; --bit) {
        result = ff_mul_barrett_v2(result, result);
        if ((exp >> bit) & 1) {
            result = ff_mul_barrett_v2(result, base);
        }
    }
    return result;
}
