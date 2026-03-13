# v2.0.0 Performance Analysis — Groth16 GPU Primitives

**Device:** NVIDIA GeForce RTX 3060 Laptop GPU (SM 8.6, 30 SMs, 6 GB VRAM)
**CUDA:** 12.8, MSVC 2022, Release build (`-O3 --use_fast_math`)
**Date:** 2026-03-12

---

## MSM Benchmark (Pippenger's Bucket Method)

| Size | Points | Window (c) | Latency (ms) | Throughput (pts/ms) |
|------|--------|------------|-------------|---------------------|
| 2^10 | 1,024 | 4 | 261 | 3.9 |
| 2^12 | 4,096 | 4 | 749 | 5.5 |
| 2^14 | 16,384 | 4 | 2,697 | 6.1 |
| 2^15 | 32,768 | 4 | 5,308 | 6.2 |
| 2^16 | 65,536 | 4 | 10,574 | 6.2 |
| 2^18 | 262,144 | 4 | 42,603 | 6.2 |

**Scaling:** Approximately linear in n (2x points → 2x time), which matches
theoretical O(n) for Pippenger's bucket method. Throughput plateaus at ~6.2 pts/ms.

**Why slow:** This implementation is correctness-focused, not throughput-optimized:
- Bucket reduction runs on a **single GPU thread** (sequential running sum over 2^c buckets)
- Window combination runs on a **single GPU thread** (sequential Horner's method)
- Bucket accumulation: one thread per bucket, sequential EC additions
- No signed-digit recoding, no batch affine-to-Jacobian conversion

**Context:** Production MSM implementations (ICICLE, bellman-cuda) achieve orders-of-magnitude
higher throughput via signed-digit scalars, cooperative accumulation, and batch affine inversions.
This implementation demonstrates the algorithm structure and correctness.

### v2.1.0 MSM (Sessions 26-27: Signed-Digit + Parallel Reduction)

| Size | v2.0.0 (ms) | v2.1.0-s27 (ms) | Speedup | Window (c) | Throughput (pts/ms) |
|------|-------------|-----------------|---------|------------|---------------------|
| 2^10 | 261 | 121 | 2.2x | 6 | 8.5 |
| 2^12 | 755 | 256 | 2.9x | 7 | 16.0 |
| 2^14 | 2,704 | 557 | 4.9x | 8 | 29.4 |
| 2^15 | 5,317 | 1,066 | 5.0x | 8 | 30.7 |
| 2^16 | 10,633 | 2,318 | 4.6x | 9 | 28.3 |
| 2^18 | 42,714 | 1,180 | **36.2x** | 10 | 222.1 |

**Optimizations:**
- **Signed-digit window recoding** (S26): halves bucket count (2^c → 2^(c-1)+1)
- **Improved window sizing** (S26): c = floor(log2(n)/2)+1, clamped [4,16]
- **Segment-offset accumulation** (S26): O(1) lookup replaces binary search
- **Parallel bucket reduction** (S27): Hillis-Steele suffix scan + tree reduce
  (O(log B) depth vs O(B) single-thread). Key optimization — removed dominant serial bottleneck.

**Why 36.2x at 2^18:** At c=10, 512 active buckets per window. The single-thread running
sum did 1024 EC additions × 27 windows sequentially. Parallel scan reduces this to 9 rounds
of depth, utilizing 512 threads per window.

---

## Groth16 Pipeline Benchmark

Circuit: x^3 + x + 5 = y (4 R1CS constraints, 6 variables)

| Domain | Setup (ms) | GPU Prove (ms) | CPU Prove (ms) | GPU/CPU |
|--------|-----------|----------------|----------------|---------|
| 256 | 5,363 | 5,243 | 5,220 | 1.004 |
| 512 | 10,362 | 10,306 | 10,264 | 1.004 |
| 1024 | 20,427 | 20,324 | 20,319 | 1.000 |

**Phase breakdown (n=256):**
- Trusted setup (CPU): ~5.4 seconds — dominated by CPU EC scalar multiplications
  (n-1 = 255 h_query points + 6 u_tau + 6 v_tau + 5 l_query = ~272 EC scalar muls)
- Proof generation: ~5.2 seconds — also dominated by CPU EC assembly
  (H commitment loop: up to 255 EC scalar muls for h_query)

**Why GPU/CPU ratio ≈ 1.0:**
- The GPU pipeline (NTT/INTT/coset NTT/pointwise ops) completes in <1 ms for n=256
- Both GPU and CPU paths share the same `assemble_proof()` function for EC arithmetic
- At this toy scale, proof assembly (CPU EC scalar muls) dominates 99%+ of runtime
- GPU acceleration would only matter for circuits with thousands+ of constraints where
  MSM replaces sequential scalar muls and NTT operates on larger domains

**Scaling:** Linear in domain size (2x domain → 2x time), matching O(n) setup and
O(n log n) NTT + O(n) MSM expectations (at this scale, constant factors dominate).

---

## Component Summary

### What's New in v2.0.0

| Component | Files | Tests | Purpose |
|-----------|-------|-------|---------|
| Fq (381-bit) + Fq2 arithmetic | `ff_fq.cuh`, `ff_fq2.cuh`, `ff_fq_kernels.cu` | 85 | Base field for EC |
| G1 + G2 elliptic curve ops | `ec_g1.cuh`, `ec_g2.cuh`, `ec_kernels.cu` | 38 | Point arithmetic |
| MSM (Pippenger, G1) | `msm.cuh`, `msm.cu` | 21 | G1 commitments |
| Polynomial operations | `poly_ops.cuh`, `poly_ops.cu` | 15 | Coset NTT, quotient |
| Groth16 pipeline | `groth16.cuh`, `groth16.cu` | 29 | End-to-end prover |
| Benchmarks | `bench_msm.cu`, `bench_groth16.cu` | — | Perf measurement |
| **Total** | **13 new files** | **188 new** | **621 total tests** |

### v1.x NTT Performance (unchanged)

| Version | Best Mode | n=2^22 Latency |
|---------|-----------|---------------|
| v1.1.0 | Montgomery | 25.1 ms |
| v1.4.0 | Montgomery (radix-4) | 17.1 ms |
| v1.5.0 | Montgomery (radix-8) | 15.5 ms |
| v1.6.0 | Goldilocks | 3.6 ms |
| v1.6.0 | BabyBear | 2.4 ms |

---

## Future Work

1. **Optimized MSM:** Signed-digit recoding, cooperative bucket accumulation,
   Montgomery batch inversion for affine conversion, larger window sizes
2. **Pairing verification:** e(A, B) = e(alpha, beta) * e(L, gamma_inv) * e(C, delta_inv)
   requires Miller loop + final exponentiation (complex, ~2000 lines)
3. **Larger circuits:** Scale beyond toy — requires constraint system builder,
   R1CS-to-QAP automation, variable-length witness
4. **GPU proof assembly:** Replace CPU EC scalar muls with GPU MSM for commitments
   (would make GPU path significantly faster for large circuits)
