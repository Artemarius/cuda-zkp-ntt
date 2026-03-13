// include/groth16.cuh
// Toy Groth16 prover for x^3 + x + 5 = y.
// Demonstrates GPU-accelerated NTT, polynomial ops, MSM, and EC arithmetic
// working together in an end-to-end proof generation pipeline.
//
// No pairing verification (future work). Correctness verified via:
//   - QAP identity: A*B - C = H*Z_H at random evaluation point
//   - Proof elements on-curve
//   - Determinism: same inputs → same proof

#pragma once
#include "ec_g1.cuh"
#include "ec_g2.cuh"
#include "ff_arithmetic.cuh"
#include <vector>

// ─── R1CS ────────────────────────────────────────────────────────────────────
// Rank-1 Constraint System: A * w ∘ B * w = C * w
// where w is the witness vector and ∘ is the Hadamard (pointwise) product.

struct R1CS {
    size_t num_constraints;  // actual constraints (4 for toy circuit)
    size_t num_variables;    // variables in witness (6 for toy circuit)
    size_t domain_size;      // power-of-2 NTT domain (≥ 256)
    // Dense matrices: A[constraint][variable] as small integers
    // Stored row-major. Padded rows (beyond num_constraints) are zero.
    std::vector<std::vector<uint64_t>> A, B, C;
};

// ─── Proving Key (Structured Reference String) ──────────────────────────────

struct ProvingKey {
    size_t n;  // domain size

    // [α]_1, [β]_1, [δ]_1 — G1 SRS elements
    G1Affine alpha_g1;
    G1Affine beta_g1;
    G1Affine delta_g1;

    // [β]_2, [δ]_2 — G2 SRS elements
    G2Affine beta_g2;
    G2Affine delta_g2;

    // Per-variable commitments: [u_i(τ)]_1 for QAP "left" polynomial
    std::vector<G1Affine> u_tau_g1;   // num_variables entries
    // Per-variable commitments: [v_i(τ)]_2 for QAP "right" polynomial
    std::vector<G2Affine> v_tau_g2;   // num_variables entries

    // H query: [τ^j * t(τ) / δ]_1 for j = 0..n-2
    // Used for H(x) commitment: MSM(h_query, h_coefficients)
    std::vector<G1Affine> h_query;

    // Private input commitments: [(β*u_i(τ) + α*v_i(τ) + w_i(τ))/δ]_1
    // For variables i in the private witness range
    std::vector<G1Affine> l_query;    // private_count entries
    size_t public_count;              // number of public inputs (1=constant + public)
};

// ─── Proof ───────────────────────────────────────────────────────────────────

struct Groth16Proof {
    G1Affine pi_a;   // [A]_1
    G2Affine pi_b;   // [B]_2
    G1Affine pi_c;   // [C]_1
};

// ─── API ─────────────────────────────────────────────────────────────────────

// Construct R1CS for x^3 + x + 5 = y.
// Variables: w = [1, x, y, v1=x*x, v2=v1*x, v3=v2+x]
// Constraints:
//   v1 = x * x
//   v2 = v1 * x
//   v3 = v2 + x   (= (v2+x)*1)
//   y  = v3 + 5    (= (v3+5)*1)
R1CS make_toy_r1cs(size_t domain_size = 256);

// Compute witness for x^3 + x + 5 = y given input x.
// Returns: [1, x, y, x*x, x*x*x, x*x*x+x] in standard form.
std::vector<FpElement> compute_witness(uint64_t x);

// Generate proving key (trusted setup, CPU-side).
// Uses small seed values for deterministic, fast generation.
// tau_seed controls the secret evaluation point τ.
ProvingKey generate_proving_key(const R1CS& r1cs, uint64_t tau_seed = 42);

// Generate Groth16 proof using GPU-accelerated primitives.
// Pipeline: QAP construction → H(x) quotient (coset NTT) → commitments (MSM).
// r_seed and s_seed control proof randomization.
Groth16Proof groth16_prove(const R1CS& r1cs,
                           const ProvingKey& pk,
                           const std::vector<FpElement>& witness,
                           uint64_t r_seed = 17,
                           uint64_t s_seed = 23,
                           cudaStream_t stream = 0);

// CPU-only proof generation (for cross-validation).
Groth16Proof groth16_prove_cpu(const R1CS& r1cs,
                               const ProvingKey& pk,
                               const std::vector<FpElement>& witness,
                               uint64_t r_seed = 17,
                               uint64_t s_seed = 23);
