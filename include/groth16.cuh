// include/groth16.cuh
// Groth16 prover + verifier with GPU-accelerated NTT, polynomial ops, MSM, EC, and pairing.
//
// Supports two circuit types:
//   - Toy circuit: x^3 + x + 5 = y (4 constraints, dense R1CS, domain_size=256)
//   - Fibonacci circuit: a_{i+2} = a_i + a_{i+1} (sparse R1CS, up to 2^18 constraints)
//
// Verification via BLS12-381 optimal Ate pairing:
//   e(π_A, π_B) = e(α, β) · e(L_pub, γ) · e(π_C, δ)
// Implemented as multi-Miller loop + single final exponentiation.

#pragma once
#include "ec_g1.cuh"
#include "ec_g2.cuh"
#include "ff_arithmetic.cuh"
#include <vector>

// ─── R1CS (Dense) ────────────────────────────────────────────────────────────
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

// ─── R1CS (Sparse) ───────────────────────────────────────────────────────────
// COO (coordinate) format for large circuits where dense is infeasible.
// Fibonacci at n=2^18: ~4 nonzeros per constraint vs 2^18 dense entries.

struct SparseEntry {
    uint32_t row;    // constraint index
    uint32_t col;    // variable index
    uint64_t val;    // coefficient value (small integer, typically 1)
};

struct SparseR1CS {
    size_t num_constraints;  // actual constraints
    size_t num_variables;    // variables in witness
    size_t domain_size;      // power-of-2 NTT domain (≥ num_constraints)
    std::vector<SparseEntry> A, B, C;
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

    // Raw scalar values for B_scalar computation (G2 MSM workaround).
    // v_tau_scalars[i] = v_i(τ) in standard form. Used to compute B_scalar on CPU
    // instead of requiring GPU G2 MSM.
    // Populated by both generate_proving_key() and generate_proving_key_sparse().
    std::vector<FpElement> v_tau_scalars;
    FpElement beta_scalar;   // β in standard form
    FpElement delta_scalar;  // δ in standard form
};

// ─── Proof ───────────────────────────────────────────────────────────────────

struct Groth16Proof {
    G1Affine pi_a;   // [A]_1
    G2Affine pi_b;   // [B]_2
    G1Affine pi_c;   // [C]_1
};

// ─── Verifying Key ──────────────────────────────────────────────────────────
// Contains the minimal information needed to verify a Groth16 proof.
// Generated alongside the proving key during trusted setup.

struct VerifyingKey {
    G1Affine alpha_g1;    // [α]_1
    G2Affine beta_g2;     // [β]_2
    G2Affine gamma_g2;    // [γ]_2
    G2Affine delta_g2;    // [δ]_2

    // Public input commitments: ic[i] = [(β*u_i(τ) + α*v_i(τ) + w_i(τ))/γ]_1
    // for i = 0..public_count-1 (public variables including constant "1").
    std::vector<G1Affine> ic;
    size_t public_count;
};

// ─── API: Verification ──────────────────────────────────────────────────────

// Verify a Groth16 proof using CPU pairing (multi-Miller loop + final exp).
// public_inputs: first public_count entries of the witness (w[0]=1, ...).
// Returns true iff e(π_A, π_B) = e(α,β) · e(L_pub, γ) · e(π_C, δ).
bool groth16_verify(const VerifyingKey& vk,
                    const Groth16Proof& proof,
                    const std::vector<FpElement>& public_inputs);

// ─── API: Toy Circuit (x^3 + x + 5 = y) ─────────────────────────────────────

// Construct dense R1CS for x^3 + x + 5 = y.
R1CS make_toy_r1cs(size_t domain_size = 256);

// Compute witness for x^3 + x + 5 = y given input x.
std::vector<FpElement> compute_witness(uint64_t x);

// Generate proving key for dense R1CS (trusted setup, CPU-side).
// If vk_out is non-null, also generates the verifying key (same toxic waste).
ProvingKey generate_proving_key(const R1CS& r1cs, uint64_t tau_seed = 42,
                                VerifyingKey* vk_out = nullptr);

// Generate Groth16 proof using GPU NTT + CPU assembly (toy circuit).
Groth16Proof groth16_prove(const R1CS& r1cs,
                           const ProvingKey& pk,
                           const std::vector<FpElement>& witness,
                           uint64_t r_seed = 17,
                           uint64_t s_seed = 23,
                           cudaStream_t stream = 0);

// CPU-only proof generation (toy circuit, for cross-validation).
Groth16Proof groth16_prove_cpu(const R1CS& r1cs,
                               const ProvingKey& pk,
                               const std::vector<FpElement>& witness,
                               uint64_t r_seed = 17,
                               uint64_t s_seed = 23);

// ─── API: Fibonacci Circuit (a_{i+2} = a_i + a_{i+1}) ───────────────────────

// Construct sparse R1CS for Fibonacci sequence.
// num_constraints Fibonacci recurrence constraints, domain_size = next power of 2.
// Variables: w = [1, a_0, a_1, ..., a_{num_constraints+1}]
// Each constraint i: (a_i + a_{i+1}) * 1 = a_{i+2}
SparseR1CS make_fibonacci_r1cs(size_t num_constraints);

// Compute Fibonacci witness starting from (a0, a1) in the BLS12-381 scalar field.
// Returns: [1, a_0, a_1, a_2, ..., a_{num_constraints+1}] in standard form.
// Fibonacci values are computed modulo the field prime (wraps for large sequences).
std::vector<FpElement> compute_fibonacci_witness(uint64_t a0, uint64_t a1,
                                                  size_t num_constraints);

// Generate proving key for sparse R1CS using Lagrange basis evaluation.
// Uses batch inversion (Montgomery's trick) for O(n) setup instead of O(nv * n log n).
// Stores v_tau_scalars for CPU-side B_scalar computation (no GPU G2 MSM needed).
// If vk_out is non-null, also generates the verifying key (same toxic waste).
ProvingKey generate_proving_key_sparse(const SparseR1CS& r1cs, uint64_t tau_seed = 42,
                                       VerifyingKey* vk_out = nullptr);

// Generate Groth16 proof using GPU MSM + GPU NTT (full GPU acceleration).
// Pipeline: sparse mat-vec → GPU INTT → coset NTT → H(x) → GPU MSM assembly.
Groth16Proof groth16_prove_sparse(const SparseR1CS& r1cs,
                                   const ProvingKey& pk,
                                   const std::vector<FpElement>& witness,
                                   uint64_t r_seed = 17,
                                   uint64_t s_seed = 23,
                                   cudaStream_t stream = 0);

// CPU-only proof for sparse R1CS (for cross-validation and GPU/CPU comparison).
Groth16Proof groth16_prove_cpu_sparse(const SparseR1CS& r1cs,
                                       const ProvingKey& pk,
                                       const std::vector<FpElement>& witness,
                                       uint64_t r_seed = 17,
                                       uint64_t s_seed = 23);

// ─── API: Batch Pipeline (2-stream, pre-allocated device memory) ─────────────

// Batch prove: generates proofs for multiple witnesses against the same circuit.
// Pre-allocates device memory (2 slots with CUDA streams) to eliminate per-proof
// malloc/free overhead. Each proof uses r_seed = r_seed_base + 2*i,
// s_seed = s_seed_base + 2*i. Returns proofs in witness order.
std::vector<Groth16Proof> groth16_prove_batch_sparse(
    const SparseR1CS& r1cs,
    const ProvingKey& pk,
    const std::vector<std::vector<FpElement>>& witnesses,
    uint64_t r_seed_base = 17,
    uint64_t s_seed_base = 23);

// Sequential batch: calls groth16_prove_sparse() in a loop (baseline for
// comparison with the pipelined batch version).
std::vector<Groth16Proof> groth16_prove_batch_sequential_sparse(
    const SparseR1CS& r1cs,
    const ProvingKey& pk,
    const std::vector<std::vector<FpElement>>& witnesses,
    uint64_t r_seed_base = 17,
    uint64_t s_seed_base = 23);
