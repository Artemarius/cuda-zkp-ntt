// include/poly_ops.cuh
// Polynomial operations for Groth16: coset NTT, pointwise arithmetic.
//
// Coset NTT evaluates a polynomial on the coset {g * omega^i} rather than
// on the roots of unity {omega^i}. This avoids Z_H(x) = x^n - 1 being zero
// at evaluation points, enabling H(x) = (A*B - C) / Z_H computation.

#pragma once
#include "ff_arithmetic.cuh"
#include "ntt.cuh"

// ─── Coset NTT ──────────────────────────────────────────────────────────────

// Forward coset NTT: scale element i by g^i, then NTT.
// Evaluates polynomial at {g * omega_n^i} for i = 0..n-1.
// d_data: device pointer to n FpElements (standard form, in-place)
// coset_gen: coset generator in standard form (typically 7)
void poly_coset_ntt_forward(FpElement* d_data, size_t n,
                            const FpElement& coset_gen,
                            NTTMode mode = NTTMode::OPTIMIZED,
                            cudaStream_t stream = 0);

// Inverse coset NTT: INTT, then unscale element i by g^{-i}.
// Recovers polynomial coefficients from coset evaluations.
void poly_coset_ntt_inverse(FpElement* d_data, size_t n,
                            const FpElement& coset_gen,
                            NTTMode mode = NTTMode::OPTIMIZED,
                            cudaStream_t stream = 0);

// ─── Pointwise Operations ───────────────────────────────────────────────────

// Pointwise multiplication: d_c[i] = d_a[i] * d_b[i]
// All arrays in standard form, size n.
void poly_pointwise_mul(FpElement* d_c,
                        const FpElement* d_a, const FpElement* d_b,
                        size_t n, cudaStream_t stream = 0);

// Pointwise multiply-subtract: d_out[i] = d_a[i] * d_b[i] - d_c[i]
// All arrays in standard form, size n.
void poly_pointwise_mul_sub(FpElement* d_out,
                            const FpElement* d_a, const FpElement* d_b,
                            const FpElement* d_c,
                            size_t n, cudaStream_t stream = 0);

// Pointwise scale by constant: d_data[i] *= scalar
// Both d_data and scalar in standard form.
void poly_scale(FpElement* d_data, const FpElement& scalar,
                size_t n, cudaStream_t stream = 0);
