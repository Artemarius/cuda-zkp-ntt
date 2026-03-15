# CLAUDE.md — cuda-zkp-ntt

## Project Identity

GPU-accelerated ZKP primitives library for BLS12-381 on NVIDIA GPUs.
Includes NTT (3 fields), elliptic curve arithmetic (G1/G2), MSM (Pippenger),
polynomial operations, and end-to-end Groth16 toy prover.

- **v3.0.0** (in progress, S34): Pairing verification — Fq6 cubic extension over Fq2
  (Karatsuba 6 Fq2 muls, CH-SQR2 sqr, inverse via norm, sparse mul_by_01/mul_by_1 for
  Miller loop, Frobenius map); Fq12 quadratic extension over Fq6 (Karatsuba 3 Fq6 muls,
  complex-method sqr, inverse via norm, conjugate, sparse mul_by_014 for Miller loop,
  Frobenius map with γ_w[k] coefficients); Miller loop (optimal Ate, |u|=0xd201000000010000,
  63 iterations, affine G2 line functions, M-type twist → sparse Fq12 via mul_by_014,
  GPU kernel with __noinline__ wrappers for cicc crash workaround); Final exponentiation
  (Hayashida-Hayasaka-Teruya eprint 2020/875, easy part (q^6-1)(q^2+1) + hard part
  3+(x²+q²-1)(q+x)(x-1)², 4x exp_by_u + 2 Frobenius, FrobeniusCoeffs struct,
  GPU kernel with __noinline__ wrappers); Full pairing kernel e(P,Q) = final_exp(miller(P,Q)),
  bilinearity verified: e(2P,Q) = e(P,Q)². 986 tests.
- **v2.2.0**: Fibonacci circuit + batch pipeline — sparse R1CS (COO format),
  Lagrange basis trusted setup (batch inversion), GPU MSM proof assembly, 2-stream batch
  pipeline with pre-allocated device memory. GPU wins 55-139x over CPU at n=256-1024.
  Batch pipeline: 1.03x at nc=256, 1.02x at nc=1024 (limited by cooperative NTT blocking
  all SMs and MSM internal sync). 870 tests.
- **v2.1.0**: Production MSM — signed-digit window recoding (halves bucket
  count), segment-offset parallel accumulation, parallel bucket reduction (Hillis-Steele
  suffix scan), window auto-tuner (c capped at 11 for parallel reduction), stream-ordered
  memory pools. 35.8x vs v2.0.0 at n=2^18 (42.7s→1.2s), 247 pts/ms at n=2^20. 701 tests.
- **v2.0.0**: Groth16 GPU primitives — Fq/Fq2 381-bit field arithmetic,
  G1/G2 elliptic curve ops (Jacobian), GPU MSM (Pippenger's bucket method),
  polynomial ops (coset NTT, pointwise), end-to-end toy prover for x^3+x+5=y.
  621 tests. GPU proof matches CPU proof bitwise.
- **NTT (v1.x)**: Fused radix-1024 inner kernel + radix-8 Montgomery / radix-4 Barrett
  cooperative outer stages + branchless arithmetic + batched NTT + async pipeline + CUDA Graphs.
  At n=2^22: BLS12-381 15.1ms, Goldilocks 3.6ms (4.2x), BabyBear 2.4ms (6.2x).
- **Negative results documented**: OTF twiddles (+265%), Plantard (+79%), 4-Step NTT (+18%).

---

## Development Environment

- **OS**: Windows 10 Pro (primary build target); WSL2 Ubuntu 24.04 available
- **GPU**: NVIDIA RTX 3060 Laptop GPU, 6GB VRAM, Ampere (sm_86), 30 SMs
- **CUDA**: 12.8
- **Compiler**: MSVC 2022 (primary) / GCC 13 (WSL2)
- **CMake**: 3.20+
- **IDE**: VS Code + Remote WSL extension; CUDA files open in Visual Studio 2022 for IntelliSense
- **Profiler**: NVIDIA Nsight Compute (ncu), Nsight Systems (nsys)

---

## Build System

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

# Build all targets
cmake --build build -j$(nproc)

# Build specific target
cmake --build build --target bench_ntt -j$(nproc)
```

CMake targets:
- `test_correctness` — validates all primitives (NTT, EC, MSM, poly ops, Groth16)
- `bench_ntt` — Google Benchmark harness, outputs CSV
- `bench_multifield` — 3-field NTT benchmark (BLS/Goldilocks/BabyBear, JSON output)
- `bench_msm` — MSM (Pippenger) benchmark at various sizes
- `bench_groth16` — Groth16 pipeline benchmark with phase breakdown
- `ff_microbench` — standalone finite-field operation microbenchmark
- `ntt_profile` — minimal binary for clean Nsight Compute profiling

---

## Code Conventions

### CUDA/C++ Style
- C++17 throughout; no C++20 until CUDA 12.x fully supports it in WSL2
- Headers: `.cuh` for CUDA device code, `.h` for host-only
- Kernels: `snake_case`, prefixed by module (e.g., `ntt_butterfly_kernel`, `ff_mul_kernel`)
- Device functions: `__device__ __forceinline__` for hot-path helpers
- No exceptions in device code; use return codes or assert for debug builds
- PTX intrinsics only in `ff_arithmetic.cuh` — isolate from algorithm logic

### Memory Management
- Prefer `thrust::device_vector` for managed allocations in benchmark harness
- Raw `cudaMalloc` / `cudaFree` in kernel implementation files (explicit control)
- Use `cudaMallocAsync` + memory pools for pipeline double-buffer allocations
- Always check CUDA error codes; use `CUDA_CHECK` macro from `include/cuda_utils.cuh`

### Finite Field
- Field modulus: BLS12-381 scalar field `r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001`
- Representation: 8 × uint32_t limbs in Montgomery form (or standard form for Barrett)
- Montgomery constant: R = 2^256 mod r
- Montgomery arithmetic: `include/ff_arithmetic.cuh` (CIOS, PTX intrinsics)
- Barrett arithmetic: `include/ff_barrett.cuh` (standard-form, no domain conversion)
- Plantard arithmetic: `include/ff_plantard.cuh` — **NEGATIVE RESULT** for BLS12-381.
  944 SASS instructions vs Montgomery 528 (+79%). Full algorithm requires z×μ (512×512
  schoolbook, 136 MADs) + q×p (64 MADs) on top of the 64-MAD product. Only viable for
  word-size moduli (32/64 bits). Infrastructure retained for reference.
- Goldilocks field (p = 2^64 − 2^32 + 1): `include/ff_goldilocks.cuh`
  - Representation: single `uint64_t`, standard form (no Montgomery)
  - Mul: PTX `mul.lo/hi.u64` → Goldilocks reduction (2^64 ≡ 2^32 − 1 mod p)
  - ~5-8 instructions per mul vs 528 for BLS12-381
- BabyBear field (p = 2^31 − 2^27 + 1 = 0x78000001): `include/ff_babybear.cuh`
  - Representation: single `uint32_t`, standard form
  - Mul: 32×32→64 product, then `% p` (compiler-optimized constant division)
  - ~3-5 instructions per mul
- CPU reference implementation in `tests/ff_reference.h` (plain C++, no CUDA)
  - BLS12-381: `FpRef` (4×uint64_t Montgomery), Barrett reduction
  - Goldilocks: `GlRef` (uint64_t), NTT reference
  - BabyBear: `BbRef` (uint32_t), NTT reference
- BLS12-381 base field Fq (381-bit, 12×uint32 Montgomery): `include/ff_fq.cuh`
  - CIOS multiplication (12 limbs, 144 MADs), add/sub/mul/sqr/inv/neg
  - Fq2 quadratic extension: `include/ff_fq2.cuh` (Karatsuba, 3 Fq muls per Fq2 mul)
  - Fq6 cubic extension: `include/ff_fq6.cuh` (Fq6 = Fq2[v]/(v³−β), β=(1+u), Karatsuba
    6 Fq2 muls per mul, CH-SQR2 squaring 2 muls + 3 sqrs, inverse via norm to Fq2,
    sparse mul_by_01/mul_by_1 for Miller loop, Frobenius map)
  - Fq12 quadratic extension: `include/ff_fq12.cuh` (Fq12 = Fq6[w]/(w²−v), 576 bytes,
    Karatsuba 3 Fq6 muls per mul = 54 Fq muls, complex-method sqr = 36 Fq muls,
    inverse via norm to Fq6, conjugate a0−a1·w, sparse mul_by_014 for Miller loop
    line functions = 13 Fq2 muls, Frobenius map with γ_w[k] = β^((q^k-1)/6) period 12)
- Elliptic curve G1/G2: `include/ec_g1.cuh`, `include/ec_g2.cuh`
  - Jacobian projective coordinates, affine conversion, on-curve check, scalar_mul
  - G1 over Fq (curve: y^2 = x^3 + 4), G2 over Fq2 (curve: y^2 = x^3 + 4(1+u))
- GPU MSM (Pippenger): `include/msm.cuh`, `src/msm.cu`
  - Signed-digit window recoding → CUB radix sort → segment-offset accumulation → parallel reduce → Horner
  - Signed digits halve bucket count (2^c → 2^(c-1)+1), point negation via packed sign bit
  - Parallel bucket reduction: Hillis-Steele suffix scan + tree reduce (O(log B) depth)
  - Window sizing: c = floor(log2(n)/2) + 1, clamped [4, 11] (cap ensures parallel reduction)
  - Stream-ordered memory pools (cudaMallocAsync/cudaFreeAsync) for allocation reuse
  - Separate TU without RDC (CUB compatibility)
- Polynomial ops: `include/poly_ops.cuh`, `src/poly_ops.cu`
  - Coset NTT: scale by g^i → regular NTT (coset generator g=7)
  - Pointwise mul, mul_sub, scale kernels
- Groth16 prover: `include/groth16.cuh`, `src/groth16.cu`
  - Toy circuit x^3+x+5=y (4 R1CS constraints, 6 variables, domain_size=256)
  - Fibonacci circuit a_{i+2}=a_i+a_{i+1} (up to 2^18 constraints, sparse R1CS)
  - Sparse R1CS: `SparseEntry` (COO format), `SparseR1CS` struct, `make_fibonacci_r1cs()`
  - Lagrange basis trusted setup: `generate_proving_key_sparse()` — O(n) batch inversion
    (Montgomery's trick) instead of O(nv * n log n) INTTs. L_k(τ) = ω^k*(τ^n-1)/(n*(τ-ω^k))
  - GPU MSM proof assembly: `groth16_prove_sparse()` — GPU NTT + GPU MSM for pi_A, pi_C
  - CPU reference proof: `groth16_prove_cpu_sparse()` — CPU NTT + sequential scalar muls
  - G2 MSM workaround: B_scalar computed via field arithmetic + 1 G2 scalar mul (no GPU G2 MSM)
  - Dense pipeline: R1CS×witness → INTT → coset NTT → pointwise → coset INTT → EC assembly
  - GPU proof matches CPU proof bitwise (cross-validated)
  - Batch pipeline: `groth16_prove_batch_sparse()` — 2 CUDA streams, pre-allocated device
    memory (2 slots), reusable host scalar buffers. Eliminates per-proof malloc/free overhead.
    Batch reference: `groth16_prove_batch_sequential_sparse()` (loop baseline).
    Measured speedup: 1.03x at nc=256, 1.02x at nc=1024 (cooperative NTT blocks all SMs,
    MSM internal sync prevents CPU-GPU overlap — true overlap needs async MSM)
- BLS12-381 optimal Ate pairing: `include/pairing.cuh`, `src/pairing_kernels.cu`
  - Parameter: u = -0xd201000000010000 (64-bit, Hamming weight 6)
  - Miller loop: 63 iterations (bit 62 down to 0), plus sign correction (conjugate for u<0)
  - Affine G2 coordinates for running point T (requires Fq2 inversion per step)
  - M-type sextic twist (E': y²=x³+4(1+u)): line evaluations at positions (0, 1, 4)
  - Doubling step: d0=12(1+u)-yt², d1=3xt²·xP, d4=-2yt·yP → fq12_mul_by_014
  - Addition step: d0=xq·yt-yq·xt, d1=(yq-yt)·xP, d4=(xt-xq)·yP → fq12_mul_by_014
  - Final exponentiation: f^((q^12-1)/r), Hayashida-Hayasaka-Teruya (eprint 2020/875)
    - Easy part: f^((q^6-1)(q^2+1)) via conjugate + inverse + Frobenius
    - Hard part: 3+(x²+q²-1)(q+x)(x-1)² via 4× exp_by_u + 2 Frobenius maps
    - exp_by_u: 63 cyclotomic squarings + 4 multiplications, then conjugate (u<0)
    - FrobeniusCoeffs struct: γ₁[6], γ₂[6] (Fq6) + γ_w[12] (Fq12), precomputed on host
  - GPU: __noinline__ wrappers for fq12_sqr, fq12_mul, fq12_mul_by_014, fq12_inv,
    fq12_frobenius, fq2_inv, fq_inv, exp_by_u, final_exponentiation
  - Kernels: miller_loop_kernel, final_exp_kernel, pairing_kernel (Miller+final_exp)
  - pairing_lib compiled without RDC (separate TU, like msm_lib)
  - CPU reference: `miller_loop_ref()`, `final_exponentiation_ref()`, `pairing_ref()`
  - Bilinearity verified: e(2P,Q) = e(P,Q)², e(P,2Q) = e(P,Q)², e(2P,Q) = e(P,2Q)

### NTT
- BLS12-381 NTT: `ntt.cuh` (NTTMode: NAIVE, OPTIMIZED, BARRETT, ASYNC, FOUR_STEP)
- Goldilocks NTT: `ntt_goldilocks.cuh` (standard-form, fused K=8-11 + cooperative radix-8/4/2 outer)
- BabyBear NTT: `ntt_babybear.cuh` (standard-form, fused K=8-11 + cooperative radix-8/4/2 outer)
- Input/output in standard (non-Montgomery) form for all fields
- BLS12-381 three arithmetic paths: `NTTMode::OPTIMIZED` (Montgomery internal, conversion overhead),
  `NTTMode::BARRETT` (standard-form throughout, no conversion),
  and `NTTMode::FOUR_STEP` (4-step Bailey's algorithm, Barrett arithmetic)
- Batched NTT: `ntt_forward_batch()` / `ntt_inverse_batch()` process B independent NTTs
  in a single set of kernel launches. Contiguous memory layout: `d_data[b*n..(b+1)*n-1]`
- Twiddle factors precomputed on device: Montgomery twiddles + separate standard-form cache
  Shared across all NTTs in a batch (single precomputation, B× reuse)
- NTT size must be power of 2, range [2^8 .. 2^26]
- Butterfly operation: `A[i] += w * A[j]; A[j] = A[i] - 2*w*A[j]` (in-place Cooley-Tukey)
- Goldilocks/BabyBear NTT: K=11 max (2048 elements, 1024 threads), radix-8 outer default
  (register pressure trivial: ~40-50 regs GL, ~20-30 BB vs 134 BLS12-381 Montgomery)

---

## Project File Map

```
include/
  cuda_utils.cuh      — CUDA_CHECK macro, timing utilities
  ff_arithmetic.cuh   — Fp element type, Montgomery mul, add, sub, inv
  ff_barrett.cuh      — Barrett modular multiplication (standard-form, no Montgomery)
  ff_plantard.cuh     — Plantard modular multiplication (NEGATIVE RESULT for BLS12-381)
  ff_goldilocks.cuh   — Goldilocks field (p=2^64-2^32+1, uint64_t): add, sub, mul, pow, inv
  ff_babybear.cuh     — BabyBear field (p=2^31-2^27+1, uint32_t): add, sub, mul, pow, inv
  ntt.cuh             — NTT public interface (single + batched + graph, NTTMode: NAIVE, OPTIMIZED, BARRETT, ASYNC, FOUR_STEP)
  ntt_goldilocks.cuh  — Goldilocks NTT public interface (forward/inverse, single/batched)
  ntt_babybear.cuh    — BabyBear NTT public interface (forward/inverse, single/batched)
  ff_fq.cuh           — BLS12-381 base field Fq (381-bit, 12×uint32 Montgomery)
  ff_fq2.cuh          — Fq2 quadratic extension (Karatsuba, 3 Fq muls)
  ff_fq6.cuh          — Fq6 cubic extension (Fq2[v]/(v³−β), Karatsuba 6 Fq2 muls, CH-SQR2, inverse via norm, sparse mul, Frobenius)
  ff_fq12.cuh         — Fq12 quadratic extension (Fq6[w]/(w²−v), Karatsuba 3 Fq6 muls, complex-method sqr, inverse via norm, conjugate, sparse mul_by_014/mul_by_034, Frobenius)
  ec_g1.cuh           — G1 elliptic curve ops (Jacobian, affine, scalar_mul)
  ec_g2.cuh           — G2 elliptic curve ops (over Fq2)
  pairing.cuh         — BLS12-381 pairing types (LineCoeffs, FrobeniusCoeffs, BLS12_381_U_ABS)
  msm.cuh             — GPU MSM (Pippenger's bucket method, G1)
  poly_ops.cuh        — Polynomial ops (coset NTT, pointwise mul/sub, scale)
  groth16.cuh         — Groth16 prover API (R1CS, SparseR1CS, ProvingKey, Proof, Fibonacci)
  pipeline.cuh        — AsyncNTTPipeline class interface
  twiddle_otf.cuh     — On-the-fly twiddle pow functions (OTF — disabled for BLS12-381)

src/
  ff_mul.cu           — FF kernels: baseline AoS, v2 branchless (PTX), SoA variants
  ff_multi_field.cu   — Goldilocks + BabyBear GPU throughput kernels (add/sub/mul/sqr)
  ff_fq_kernels.cu    — Fq/Fq2 GPU throughput kernels
  ntt_naive.cu        — Radix-2 NTT baseline + public API dispatch + twiddle caches + CUDA Graph cache
  ntt_optimized.cu    — NTT host dispatch: K selection, cooperative outer (Montgomery + Barrett)
  ntt_fused_kernels.cu — Fused warp-shuffle + shmem kernel (K=8/9/10, Montgomery + Barrett, no-RDC TU)
  ntt_4step.cu        — 4-step NTT: transpose kernel, twiddle multiply, forward/inverse (single + batched)
  ntt_async.cu        — Double-buffered async pipeline NTT
  ntt_goldilocks.cu   — Goldilocks NTT: fused K=8-11 + cooperative radix-8/4/2 outer + batched
  ntt_babybear.cu     — BabyBear NTT: fused K=8-11 + cooperative radix-8/4/2 outer + batched
  ec_kernels.cu       — G1/G2 GPU test kernels
  pairing_kernels.cu  — Miller loop GPU kernel + device functions (NO RDC, __noinline__ wrappers)
  msm.cu              — Pippenger MSM (separate TU, no RDC — CUB compatibility)
  poly_ops.cu         — Coset NTT, pointwise mul/sub/scale kernels
  groth16.cu          — Groth16 prover: trusted setup + GPU/CPU proof (dense + sparse/Fibonacci)
  benchmark.cu        — Main benchmark entry point

tests/
  test_correctness.cu — Validates all primitives (NTT, EC, MSM, poly ops, Groth16, Fibonacci)
  ff_reference.h      — CPU-only finite field + NTT + EC reference + batch inverse (test oracle)

benchmarks/
  bench_ntt.cu        — Google Benchmark: latency vs scale for all variants
  bench_multifield.cu — 3-way NTT benchmark: BLS12-381 vs Goldilocks vs BabyBear (JSON output)
  bench_msm.cu        — MSM (Pippenger) benchmark at various sizes
  bench_groth16.cu    — Groth16 pipeline benchmark with phase breakdown
  ff_microbench.cu    — Instruction throughput: FF_add, FF_mul, FF_sqr isolated

profiling/
  scripts/
    profile_ntt.sh         — Full ncu profile, saves .ncu-rep files
    collect_metrics.sh     — CSV metrics: sm_throughput, warp_stalls, roofline
    nsys_timeline.sh       — nsys timeline for async pipeline visualization
  README.md           — Profiling methodology

results/
  screenshots/        — Nsight Compute roofline, warp stall charts (PNG)
  charts/             — Generated benchmark comparison charts (matplotlib)
  data/               — Benchmark CSV/JSON files
  analysis.md         — Annotated interpretation of profiling results (v1.x)
  analysis_v200.md    — v2.0.0 performance analysis (MSM, Groth16)

scripts/
  plot_benchmarks.py  — Generate v1.x benchmark bar charts from measured data
  plot_groth16.py     — Generate v2.0.0 charts (MSM scaling, Groth16 pipeline)

.github/
  workflows/
    build.yml          — CI: compile on Linux (CUDA 12.8/12.6) + Windows (MSVC)

GUIDE.md               — Deep-dive: ZKP math, NTT, finite fields, GPU optimization
NTT_OPTIMIZATION_ROADMAP.md — Release plans (v1.2.0-v1.8.0) with session breakdown
LICENSE                — MIT License
```

---

## Key Algorithmic Reference

### Montgomery Multiplication (BLS12-381 scalar field)
- 8-limb, 32-bit words
- CIOS (Coarsely Integrated Operand Scanning) algorithm
- `mad.lo.cc.u32` / `mad.hi.cc.u32` PTX for fused multiply-accumulate
- Conditional reduction: final subtract if result ≥ p (branchless via predication)
- Target: replace IMAD-dominant path → IADD3 accumulation path per ZKProphet §IV-C

### Async Pipeline (Direction A)
- Two pinned host buffers (double-buffer), two device buffers
- Three CUDA streams: `stream_h2d_`, `stream_compute_[0/1]`, `stream_d2h_`
- Cross-stream cudaEvent_t dependencies (`cudaStreamWaitEvent`) for H2D→compute→D2H ordering
- Batch size: tunable (one NTT per batch)
- Ordering: H2D(k+1) overlaps with NTT_compute(k) and D2H(k-1)

### NTT (Cooley-Tukey, fused radix-1024 + cooperative outer stages)
- Fused inner kernel: K=10 (512 threads, 1024 elements, 32 KB shmem per block)
  - Stages 0-4: `__shfl_xor_sync` warp shuffles (no shmem, no barriers)
  - Stages 5-9: shared memory + `__syncthreads()` (cross-warp)
- K selection: K=10 for n>=2^10, K=9 for n=2^9, K=8 for n=2^8
- Fused kernels compiled in separate TU without RDC (MSVC+CUDA 12.8 workaround)
- Outer stages: cooperative groups `grid.sync()` persistent-thread kernel
  - Up to 7 stages per cooperative launch (14 outer stages -> 2 launches)
- Twiddle factors: precomputed table in global memory, cached in L1
- For n=2^22: 4 total launches (1 bit-reverse + 1 fused K=10 + 2 cooperative outer)

### NTT Time Breakdown (n=2^22, v1.1.0 → v1.5.0)
- Bit-reverse: ~0.3 ms (2%), Montgomery conversions: ~3.0 ms (19%)
- Fused K=10 (stages 0-9): ~2.5 ms (16%) — **compute-bound** (69%, IPC 2.41)
- Cooperative outer (stages 10-21): ~19.4 ms (v1.1) → ~11 ms (radix-4) → **~9 ms (radix-8 Montgomery)**
- v1.4.0: 17.1 ms Montgomery / 17.4 ms Barrett (radix-4 outer)
- v1.5.0: **15.5 ms Montgomery** (radix-8 outer) / 17.5 ms Barrett (radix-4, unchanged)
- Barrett radix-8 disabled: 174 regs → I-cache thrashing → +73% regression

### Multi-Field NTT Performance (v1.6.0, n=2^22, 7-rep median)
- BLS12-381 Montgomery: **15.1 ms** (32B/element, 256 MB DRAM traffic)
- Goldilocks: **3.6 ms** (8B/element, 64 MB DRAM traffic) — **4.2x faster**
- BabyBear: **2.4 ms** (4B/element, 32 MB DRAM traffic) — **6.2x faster**
- Speedup converges at large sizes: 19.7x (BB, 2^10) → 6.2x (BB, 2^22)
  because outer stages are memory-bound, not arithmetic-bound
- Batched 8× at 2^22: BLS 120ms, GL 31ms, BB 21ms

### Barrett Reduction (implemented, v1.2.0 Sessions 1-2 — MoMA-inspired)
- Alternative to Montgomery: operates on standard-form integers directly (no domain conversion)
- Eliminates ~3 ms (12%) to/from Montgomery overhead per NTT at n=2^22
- Barrett: c = a·b mod p via precomputed μ = ⌊2^512/p⌋ (HAC 14.42, 8×32-bit limbs)
- μ = 0x2355094edfede377c38b5dcb707e08ed365043eb4be4bad7142737a020c0d6393 (9 × uint32_t)
- Instruction cost: ~188 MADs vs Montgomery CIOS ~128 MADs (~47% more multiply ops)
- SASS: Barrett 888 instructions vs Montgomery v2 528 (1.68×), baseline 592 (1.50×)
- Microbench: identical throughput to Montgomery (both hit 91% DRAM ceiling — memory-bound)
- **NTT integration (Session 2):** `NTTMode::BARRETT` — full NTT pipeline using Barrett
  - Standard-form twiddles (separate cache from Montgomery twiddles)
  - Barrett fused kernels (K=8/9/10) + Barrett cooperative outer kernels
  - Measured NTT impact (n=2^22, RTX 3060): 28.0 ms → 27.3 ms = **-0.7 ms (2.5% faster)**
  - Small sizes (2^15-2^16) ~15% slower (fused kernel is compute-bound, Barrett +68% instructions)
  - 2^18: approximately equal; 2^20+: Barrett slightly faster
  - Net improvement smaller than projected (-0.7ms vs -1.8ms) because fused kernel overhead
    is ~2.3ms (not 1.2ms) — Barrett's extra instructions hit harder at 69% compute utilization
- Implementation: `include/ff_barrett.cuh` (GPU), `tests/ff_reference.h` (CPU reference)
- NTT integration: `NTTMode::BARRETT` in all kernel files, standard-form twiddle cache
- Reference: MoMA (Zhang & Franchetti, CGO 2025) uses Barrett exclusively

### Batched NTT (implemented, v1.2.0 Session 3)
- Process B independent NTTs in a single set of kernel launches (vs B separate calls)
- Groth16 needs ~9 NTTs; MoMA recommends batch_size > 8 for 128-384 bit inputs
- API: `ntt_forward_batch(d_data, batch_size, n, mode)` — contiguous layout, B*n elements
- Fused kernel: existing kernel unchanged — launch B × (n/1024) blocks. `boff = blockIdx.x * ELEMS`
  naturally addresses correct sub-array because butterfly addressing partitions by NTT boundaries
  (n/2 is always a multiple of half = 2^s, so the integer division decomposes cleanly)
- Outer cooperative kernel: new batched kernel with `total_butterflies = B * n/2`. Same butterfly
  formula; strided loop handles more iterations. One cooperative launch for all B NTTs.
- Bit-reverse: new batched kernel maps `tid / n` to batch_id, `tid % n` to local index
- Scale/conversion kernels: element-wise, work unchanged with B*n total elements
- Twiddle factor sharing: all B NTTs reuse the same precomputed table
- **Kernel launches**: 3-4 per batch (vs B × 4 sequential = 32 for B=8)
- **Measured batch throughput (RTX 3060, 8× NTTs, Barrett, median):**
  - 2^15: 1.03 ms batched vs 1.64 ms sequential → **1.59x (37% faster)**
  - 2^18: 9.73 ms vs 10.8 ms → **1.11x (10% faster)**
  - 2^20: 44.9 ms vs 46.8 ms → **1.04x (4% faster)**
  - 2^22: 200 ms vs 205 ms → **1.02x (2% faster)**
- Small sizes benefit most: single NTT at 2^15 barely fills 30 SMs; batch of 8 gives 8× blocks
- Large sizes (2^22): GPU already saturated; batching saves launch overhead only

### 4-Step NTT (complete, v1.3.0 — **negative result**)
- Bailey's algorithm: decompose n = n1 × n2 into sub-NTTs + transpose + twiddle multiply
- For n=2^22 with n1=n2=2^11: each sub-NTT is 2048 elements → mostly fused in shmem
- **Measured (n=2^22)**: 29.5 ms (4-step) vs 24.9 ms (Barrett) = **+18% slower**
- **Root cause**: 3 transpose passes add ~11n DRAM ops; sub-NTTs of 2048 still have 1 outer
  stage hitting DRAM (128 MB total >> 4 MB L2 when batched); cooperative approach has zero
  structural overhead
- Natural synergy with batching: B full NTTs = B×n2 sub-NTTs in step 1
- **Session 5 (complete):** Transpose kernel + architecture skeleton
  - Transpose kernel: TILE=16, shmem padded 16×(16+1), coalesced R/W, supports non-square
  - Batched transpose via z-dimension gridDim
  - Split strategy: n1=2^(log_n/2), n2=2^(log_n-log_n/2) (balanced)
  - Twiddle multiply kernels (Barrett + Montgomery, single + batched)
- **Session 6 (complete):** Sub-NTT integration, 3-transpose algorithm, NTTMode::FOUR_STEP
  - Forward: transpose → n2 column NTTs → transpose → twiddle → n1 row NTTs → transpose
  - Key insight: output is in mixed-radix order k=k1+k2*n1 (column-major); final transpose
    to n2×n1 yields natural order. 3 transposes total (not 2).
  - Inverse: reverse the forward steps with conjugate twiddles and n_sub^{-1} scaling
  - 4-step twiddle cache: separate sub-NTT twiddles for n1 and n2 + omega_n^(i*j) table
  - Shared twiddles when n1==n2 (even log_n); distinct tables for odd log_n
  - Batched 4-step: B full NTTs → B*n2 column sub-NTTs + B*n1 row sub-NTTs
  - `NTTMode::FOUR_STEP` in public API (single + batched, forward + inverse)
  - Implementation: `src/ntt_4step.cu`
- **Session 7 (complete):** Exhaustive correctness + edge cases + fallback logic
  - Fallback: `FOUR_STEP` for n < 2^16 transparently delegates to Barrett (sub-NTTs need K=8 minimum = 256 elements)
  - Forward + roundtrip tested for ALL sizes 2^10..2^22 (13 sizes, even + odd log_n)
  - Known-vector tests: all-zeros, all-ones, single-nonzero, ascending, all-(p-1) at 5 sizes
  - Inverse explicit: verified both inv(fwd(x))=x AND fwd(inv(x))=x at 4 sizes
  - 4-step vs Barrett cross-validation at 11 sizes (bitwise identical)
  - Batched B=8 vs sequential at 2^16, 2^18; batched at 9 additional size×B configurations
  - Batched 4-step vs batched Barrett cross-validation at 5 size×B configurations
- **Session 8 (complete):** Benchmark + analysis + release v1.3.0
  - 4-step slower than Barrett at all sizes (n ≥ 2^16): +64% at 2^16, +18% at 2^22
  - Batched 8× 4-step: 241 ms vs 199 ms Barrett (2^22); 4-step benefits more from batching
    (1.16x batch speedup vs Barrett's 1.05x) but still slower in absolute terms
  - Root cause documented: transpose overhead + sub-NTT outer stages + L2 thrashing

### Branchless Arithmetic (v1.4.0 Session 9)
- Switched all NTT hot-path kernels from branchy to branchless arithmetic:
  - `ff_mul` → `ff_mul_ptx` (branchless conditional reduction via PTX sub.cc + lop3)
  - `ff_add` → `ff_add_v2` (branchless PTX carry chain + lop3 select)
  - `ff_sub` → `ff_sub_v2` (branchless PTX subtract + lop3 select)
  - `ff_mul_barrett` → `ff_mul_barrett_v2` (branchless 2× conditional subtraction via PTX)
- **Eliminates warp divergence** in comparison and conditional reduction loops
- **Register usage**: Montgomery K=10: 68→66 (-2), Barrett K=10: 92→80 (-12, -13%)
- **SASS instruction mix (Montgomery K=10)**: IADD3 36%, IMAD.WIDE 21%, IMAD.X 21%, LOP3 4%
  Compiler generates good carry chains; IADD3-dominant (2-cycle) as MoMA targets
- **Measured (n=2^22, 7-rep median):** Barrett 24.9→23.8 ms (**-4.4%**),
  Montgomery 25.1→24.4 ms (**-2.8%**). Outer stages unchanged (memory-bound).

### Radix-4 Outer Stages (v1.4.0 Session 10)
- Fuses pairs of consecutive outer stages into radix-4 butterflies
- Each radix-4 unit: 4 data loads + 4 stores (vs 8+8 for 2 radix-2 stages) = **~45% DRAM traffic reduction**
- Radix-4 butterfly for stages (s, s+1): 4 elements at base+{0, half, 2·half, 3·half}
  - Stage s: 2 radix-2 butterflies on (a0,a1) and (a2,a3) with w_s(j)
  - Stage s+1: 2 radix-2 butterflies on (a0',a2') and (a1',a3') with w_{s+1}(j) and w_{s+1}(j+half)
- 6 new cooperative kernels: Barrett/Montgomery × single/batch × radix-4
- For n=2^22: 12 outer stages → 6 radix-4 passes in **1 cooperative launch** (was 2 launches)
- Odd outer stage count: radix-4 for pairs + 1 radix-2 leftover
- **Measured (n=2^22, 7-rep median):** Montgomery 24.4→**17.0 ms** (**-30.3%**),
  Barrett 23.8→**17.1 ms** (**-28.2%**). Outer stages ~19.4→~11 ms (-43%).

### Radix-8 Outer Stages (v1.5.0 Sessions 12-13)
- Fuses triples of consecutive outer stages into radix-8 butterflies
- Each radix-8 unit: 8 data loads + 8 stores (vs 6×(4+4) for 3 radix-4 stages)
- **Montgomery only**: radix-8 (134 regs) gives −8.2% at 2^22 (15.6 ms vs 17.0 ms)
- **Barrett disabled**: radix-8 (174 regs) causes I-cache thrashing → +73% regression.
  Barrett stays on radix-4 (~98 regs, 2 blocks/SM).
- Dispatch priority: radix-8 → radix-4 → radix-2 (Montgomery), radix-4 → radix-2 (Barrett)
- Leftover handling: num_outer%3==1 → 1 radix-2, num_outer%3==2 → 1 radix-4
- For n=2^22 Montgomery: 12 outer stages → 4 radix-8 passes in 1 cooperative launch
- **Measured (n=2^22, 7-rep median):** Montgomery **15.6 ms** (−8.2% vs v1.4.0)
- **Tests**: 317/317 pass (87 new radix-8 tests)

### CUDA Graphs (v1.4.0 Session 11)
- Captures NTT kernel launch sequence as CUDA Graph on first call, replays on subsequent calls
- API: `ntt_forward_graph()`, `ntt_inverse_graph()`, `ntt_forward_batch_graph()`,
  `ntt_inverse_batch_graph()`, `ntt_graph_clear_cache()`
- Graph cache keyed by `(d_data, n, batch_size, mode, forward)`
- Capture: `cudaStreamCaptureModeRelaxed` (allows occupancy queries during capture)
- Supported modes: NAIVE, OPTIMIZED, BARRETT (not FOUR_STEP — internal cudaMalloc)
- **Measured impact**: negligible (within ±2% noise). Only 3-4 kernel launches per NTT →
  CPU launch overhead is already minimal compared to GPU compute time.
- **Value**: clean API for embedding NTT in larger CUDA Graph workflows

### On-the-Fly (OTF) Twiddle Computation (v1.5.0 Session 14 — NEGATIVE RESULT)
- Replaces precomputed twiddle table loads with on-the-fly computation from per-stage roots
  stored in `__constant__` memory (~1 KB vs 64 MB precomputed table at n=2^22)
- OTF radix-8 derivation: 1 exponentiation (MSB-first square-and-multiply) + 6 multiplies
- Stage roots: `root[s] = omega_n^(n/2^(s+1))`, computed via repeated squaring from omega_n
- Fixed constants: omega_4, omega_8, omega_8^3 (independent of n)
- Implementation: `include/twiddle_otf.cuh` (pow functions), OTF kernels in `ntt_optimized.cu`
- **NEGATIVE RESULT**: At n=2^22, `ff_pow_mont_u32(root, j)` does ~29 Montgomery muls per
  butterfly at large outer stages (j up to 2^19). Each 256-bit Montgomery CIOS mul costs ~128
  MADs. Total OTF overhead: ~35 muls/butterfly vs 7 DRAM reads (precomputed).
- **Measured**: Montgomery 56.9ms OTF vs 15.6ms precomputed (+265%). Barrett 91.4ms vs 17.9ms.
- **Root cause**: BLS12-381 256-bit arithmetic makes exponentiation prohibitively expensive.
  Warp divergence in the pow loop (`if ((exp >> bit) & 1)`) further degrades throughput.
- **OTF disabled** in all dispatch functions. Infrastructure retained for Goldilocks/BabyBear
  multi-field work (64-bit/31-bit fields where multiply is 1-2 instructions, not 128 MADs).
- **Tests**: 333/333 pass (16 new OTF-specific tests: stage root chain, twiddle values, leftover patterns)

---

## Phase Status

See PROJECT.md (gitignored) for full phase roadmap and strategic context.
See `NTT_OPTIMIZATION_ROADMAP.md` for release plans (v1.0.0-v2.0.0 complete, v2.1.0-v3.0.0 planned).

Phases 1-8 complete. Current version: **v3.0.0-dev** (Session 34 complete, 986 tests).

### In Progress
- **v3.0.0** — Pairing verification: Fq6/Fq12 tower arithmetic, Miller loop (optimal Ate),
  final exponentiation, Groth16 verify equation. End-to-end prove→verify loop.
  Sessions 31-35. **Sessions 31-34 complete**: Fq6 cubic extension + Fq12 quadratic extension
  (Karatsuba mul, complex-method sqr, inverse via norm, conjugate, sparse mul_by_014,
  Frobenius map) + Miller loop (optimal Ate, affine G2, M-type twist line functions) +
  Final exponentiation (easy part + hard part via Hayashida et al.) + Full pairing kernel
  with bilinearity verified. 986 tests.

### Completed Releases
- **v2.2.0** — Fibonacci circuit + batch pipeline. Sparse R1CS (COO format), Lagrange basis
  trusted setup (O(n) batch inversion), GPU MSM proof assembly. 2-stream batch pipeline with
  pre-allocated device memory. GPU wins **55-139x** over CPU at n=256-1024. Batch pipeline
  1.03x at nc=256 (cooperative NTT + MSM sync limit overlap). 870 tests (169 new over v2.1.0).
- **v2.1.0** — Production MSM: signed-digit window recoding, CUB radix sort, segment-offset
  parallel accumulation, Hillis-Steele parallel bucket reduction, window auto-tuner (c capped
  at 11), stream-ordered memory pools. **35.8x speedup** at n=2^18 (42.7s→1.2s vs v2.0.0),
  247 pts/ms at n=2^20. 701 tests (80 new).
- **v1.0.0** — [Released on GitHub](https://github.com/Artemarius/cuda-zkp-ntt/releases/tag/v1.0.0). Fused radix-1024 + cooperative outer + async pipeline.
- **v1.2.0** — Barrett arithmetic + batched NTT. 24.9 ms single (Barrett, 2^22), 1.52x batch throughput at 2^15. 119 tests.
- **v1.3.0** — 4-Step NTT (Bailey's algorithm). **Negative result**: 29.5 ms at 2^22 (+18% vs Barrett). 221 tests.
- **v1.4.0** — Branchless arithmetic + radix-4 outer stages + CUDA Graphs. **17.1 ms Montgomery / 17.4 ms Barrett** at 2^22 (-32% vs v1.1.0). 230 tests.
- **v1.5.0** — Radix-8 outer (Montgomery only; Barrett disabled due to I-cache regression).
  OTF twiddles: **negative result** (56.9ms vs 15.6ms, disabled). **15.5 ms Montgomery** at 2^22 (-9% vs v1.4.0). 333 tests.
- **v1.6.0** — Multi-field NTT: Goldilocks (64-bit) + BabyBear (31-bit). 3-way benchmark at n=2^22:
  BLS12-381 15.1ms, Goldilocks 3.6ms (4.2x), BabyBear 2.4ms (6.2x). Speedup converges at large sizes
  (outer stages memory-bound). 458 tests.
- **v1.7.0** — Plantard reduction: **negative result**. 944 SASS (+79% vs Montgomery 528) for BLS12-381.
  Plantard's advantage (eliminating one multiply) only applies to word-size moduli (32/64-bit).
  NTT integration cancelled. 471 tests.
- **v2.0.0** — Groth16 GPU primitives library. End-to-end toy prover for x^3+x+5=y connecting
  NTT, coset NTT, MSM (Pippenger), and EC arithmetic. New primitives: Fq (381-bit) + Fq2 field
  arithmetic, G1/G2 elliptic curve ops (Jacobian), GPU MSM, polynomial operations (coset NTT,
  pointwise mul/sub), Groth16 pipeline. 621 tests (150 new). GPU proof matches CPU proof bitwise.

---

## Profiling Methodology Notes

Replicating ZKProphet analysis on RTX 3060:
- Roofline: integer throughput (IMAD-weighted) vs DRAM / L2 / L1 bandwidth ceilings
- Warp stall breakdown: Stall_Wait, Stall_Math_Pipe_Throttle, Stall_Not_Selected
- Branch efficiency per FF_op type
- Achieved vs theoretical occupancy
- Instruction mix: % IMAD, % IADD3, % SHF per kernel

Export format: `.ncu-rep` (Nsight Compute report), PNG screenshots for results/screenshots/

---

## Do Not

- Do not use `thrust::complex` or any floating-point in the NTT hot path
- Do not use `__syncthreads()` outside of shared-memory NTT stages
- Do not allocate device memory inside kernel calls
- Do not use `cudaDeviceSynchronize()` in benchmark inner loops (use events)
- Do not commit `.ncu-rep` binary files to git (add to .gitignore)
