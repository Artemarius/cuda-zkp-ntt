# NTT Optimization Roadmap & Groth16 Primitives

## Current State (v2.1.0)

**NTT (RTX 3060 Laptop, n=2^22):** 15.1 ms Montgomery / 17.5 ms Barrett (single NTT, compute only)
**Multi-field (n=2^22):** Goldilocks 3.6 ms (4.2x vs BLS), BabyBear 2.4 ms (6.2x vs BLS)
**MSM (n=2^18):** 1.2s (35.8x vs v2.0.0), 247 pts/ms at n=2^20
**v2.1.0:** Production MSM (signed-digit, parallel reduction, memory pools). 701 tests.
**v2.0.0:** Groth16 GPU primitives (Fq/Fq2, G1/G2, MSM, poly ops, end-to-end prover). 621 tests.

### Where Time Goes (n=2^22, v1.5.0 — 3 kernel launches)

| Phase | Duration | % of Total | Bottleneck |
|---|---|---|---|
| Bit-reverse permutation | ~0.3 ms | 2% | Memory (scatter) |
| Fused K=10 (stages 0-9) | ~2.5 ms | 16% | **Compute** (69%, IPC 2.41) |
| 4 radix-8 outer passes (stages 10-21) | ~9 ms | 58% | **Memory** (DRAM, ~300 GB/s) |
| Montgomery conversions | ~3.0 ms | 19% | Memory (element-wise) |

**L2 diagnostic (Session 12)**: L2 hit rate = 58.5% at 2^22 → bandwidth-bound, not latency-bound.
Stockham v1.8.0 **cancelled** — sequential access won't improve already-effective L2 reuse.

**Key lesson from v1.1–v1.5**: Every win came from reducing DRAM traffic in outer stages.
Arithmetic optimizations (Barrett, branchless, OTF twiddles) gave marginal/negative results.
Register pressure is the new constraint: Montgomery radix-8 at 134 regs is at the edge;
Barrett radix-8 at 174 regs caused I-cache thrashing (+73% regression).

### Key Gaps vs State-of-the-Art

**MoMA** (Zhang & Franchetti, CGO 2025) achieves 13× over ICICLE for 256-bit NTTs on H100
and near-ASIC performance on consumer GPUs. Their approach differs from ours in three
fundamental ways:

| Aspect | Our v1.2.0 | MoMA (CGO 2025) |
|---|---|---|
| Modular arithmetic | Montgomery (CIOS) **+ Barrett** | **Barrett reduction** (no domain conversion) |
| NTT batching | **Batched** (B=1..16, single launch set) | **Batched** (8-64 concurrent NTTs) |
| Code generation | Hand-tuned PTX | **SPIRAL auto-generated** (recursive rewrite rules) |
| Data flow | Shared memory butterfly stages | **Register-centric** (minimize shmem) |
| Montgomery overhead | **Eliminated** via Barrett mode (0 ms conversion) | **Zero** (Barrett needs no domain conversion) |

**Reference**: "Code Generation for Cryptographic Kernels using Multi-word Modular Arithmetic
on GPU" — [arXiv:2501.07535](https://arxiv.org/html/2501.07535),
[CMU SPIRAL](https://spiral.ece.cmu.edu/pub-spiral/abstract.jsp?id=376)

---

## v1.2.0 — MoMA-Inspired Arithmetic + Batched NTT

**Goal:** Adopt the two highest-impact techniques from MoMA: Barrett reduction
(eliminates 12% Montgomery conversion overhead) and batched NTT processing
(multiple independent transforms in parallel for dramatically better GPU utilization).

**Expected improvement:** 20-40% throughput for batch workloads. Barrett NTT measured at
-0.7 ms (2.5%) for single NTT at n=2^22 (conversion savings partially offset by heavier
fused kernel compute). Batching is the primary throughput lever for this release.

### Session 1 — Barrett Reduction for BLS12-381 ✅ COMPLETE

**Objective:** Implement Barrett modular arithmetic as alternative to Montgomery,
eliminating the to/from Montgomery conversion overhead.

**Background — Why Barrett:**
Montgomery multiplication requires all operands to be in Montgomery domain
(x̃ = x·R mod p). Our NTT spends ~3 ms (12% of total) on `to_montgomery` and
`from_montgomery` conversions. Barrett reduction operates on standard-form integers
directly:

```
Barrett: c = a·b mod p  (standard form in, standard form out)
Montgomery: c̃ = ã·b̃·R⁻¹ mod p  (requires a→ã and c̃→c conversions)
```

MoMA uses Barrett exclusively. For the NTT butterfly (`t = ω·A[j]`, `A[i] ± t`),
Barrett avoids two full-array conversion passes.

**Tasks:**
- Implement `ff_mul_barrett(a, b, p)` for BLS12-381 scalar field (8×32-bit limbs):
  - Precompute Barrett constant μ = ⌊2^(2k) / p⌋ where k = 256 (ceil of 255-bit modulus)
  - Full 512-bit product: a×b (schoolbook or Karatsuba)
  - Barrett quotient estimate: q̂ = ⌊(a×b) · μ / 2^(2k)⌋
  - Reduction: r = a×b − q̂·p, conditional subtract if r ≥ p
- Implement `ff_add_barrett` / `ff_sub_barrett` (same as current — no Montgomery needed)
- Device functions: `__device__ __forceinline__` in new `include/ff_barrett.cuh`
- CPU reference Barrett implementation in `tests/ff_reference.h` for validation
- Correctness tests (see test plan below)
- Microbenchmark: Barrett vs Montgomery isolated throughput

**Test plan — Barrett arithmetic (GPU + CPU reference):**
- **Edge-case inputs (GPU and CPU reference must agree):**
  - Identity: `a * 1 mod p = a`, `a * 0 mod p = 0` for random `a`
  - Boundary: `(p-1) * (p-1) mod p`, `(p-1) * 2 mod p`, `(p-2) * (p-1) mod p`
  - Zero: `0 * a`, `a * 0`, `0 * 0`
  - Small: `1 * 1`, `2 * 3`, values that fit in a single limb
  - Max limbs: all limbs `0xFFFFFFFF` reduced mod p (i.e., `(2^256 - 1) mod p`)
  - Near-modulus: `p-1`, `p-2`, and values in `[p - 2^32, p-1]` (reduction boundary)
  - Power-of-two: `2^k` for k = 0..255 (exercises different limb positions)
- **Cross-validation (GPU Barrett vs GPU Montgomery):**
  - Convert Barrett result to Montgomery domain, compare with Montgomery result
  - 10^6 random input pairs (uniform over `[0, p)`)
  - 10^4 random inputs for `ff_add` and `ff_sub` (should be bitwise identical)
- **Algebraic properties (Barrett path only):**
  - Commutativity: `a*b = b*a` for 10^4 random pairs
  - Associativity: `(a*b)*c = a*(b*c)` for 10^3 random triples
  - Distributivity: `a*(b+c) = a*b + a*c` for 10^3 random triples
  - Squaring consistency: `a*a = ff_sqr_barrett(a)` (if ff_sqr_barrett implemented)
- **CPU reference self-tests:**
  - Barrett CPU vs Montgomery CPU for 10^4 random pairs (in `ff_reference.h`)
  - Known test vectors: hand-computed small values

**Key analysis point:** Barrett requires a wider intermediate product (512-bit) and
an extra multiply by μ, vs Montgomery's cheaper CIOS reduction. The question is whether
eliminating the conversion overhead compensates. Profile both paths.

**Trade-off to evaluate:**
- Barrett: no conversion overhead, but ~30% more instructions per ff_mul
- Montgomery: cheaper per-multiply, but pays ~3 ms conversion tax per NTT call
- Breakeven: if conversion saves > per-multiply overhead × num_multiplies
- For NTT: ~n·log(n)/2 butterflies × 1 ff_mul each = ~46M ff_mul for n=2^22
  vs 2×n = 8M ff_mul for conversion → Barrett wins if per-mul overhead < 5.75×

**Deliverables:**
- `include/ff_barrett.cuh` with Barrett arithmetic device functions
- CPU reference Barrett in `tests/ff_reference.h`
- Correctness tests (edge cases + cross-validation + algebraic properties)
- Microbenchmark comparison
- Analysis: Barrett vs Montgomery instruction count (cuobjdump SASS comparison)

---

### Session 2 — Barrett NTT Integration + Benchmarking ✅ COMPLETE

**Objective:** Integrate Barrett arithmetic into the NTT pipeline and measure end-to-end impact.

**Tasks:**
- Create Barrett-path NTT variants:
  - Modify twiddle factor storage: store in standard form (not Montgomery form)
  - NTT butterfly using `ff_mul_barrett` for twiddle×element multiply
  - `ff_add` / `ff_sub` remain unchanged (no Montgomery dependency)
  - Remove `to_montgomery` / `from_montgomery` wrapper calls
- Add `NTTMode::BARRETT` to public API (keep Montgomery path as baseline)
- Integrate into fused K=10 kernel: swap ff_mul_mont → ff_mul_barrett
- Integrate into cooperative outer kernel: same swap
- Correctness tests (see test plan below)
- Benchmark: Barrett vs Montgomery NTT for sizes 2^15 through 2^22
- Profile with ncu: compare instruction mix, DRAM throughput, IPC
- Key metric: is the per-butterfly overhead of Barrett > or < conversion saving?

**Test plan — Barrett NTT integration:**
- **Forward NTT correctness (Barrett vs CPU reference):**
  - All sizes 2^10 through 2^22 (CPU reference as oracle)
  - Known input vectors: all-zeros, all-ones, single non-zero element, ascending sequence
- **Roundtrip (INTT(NTT(x)) = x) for Barrett path:**
  - All sizes 2^10 through 2^22 with random input
  - Edge inputs: all-zeros, single-element, all `p-1`
- **Cross-validation (Barrett NTT vs Montgomery NTT on GPU):**
  - Both paths must produce identical output for same input at all sizes 2^15..2^22
  - This catches GPU-specific bugs that CPU reference wouldn't reveal
- **Inverse NTT correctness (Barrett path):**
  - Verify `ntt_inverse_barrett(ntt_forward_barrett(x)) = x` explicitly
  - Verify `ntt_forward_barrett(ntt_inverse_barrett(x)) = x` (inverse direction)
- **Kernel-level checks:**
  - Fused K=10 Barrett kernel: test in isolation for n=2^10 (single fused pass)
  - Cooperative outer Barrett kernel: test for n=2^15 (outer stages exercised)

**Deliverables:**
- Barrett-path NTT (fused + outer kernels)
- Benchmark comparison table: Barrett vs Montgomery
- ncu profile comparison (instruction mix, throughput) — deferred to Session 4
- Decision: adopt Barrett as default or keep Montgomery

**Measured results (RTX 3060 Laptop, median of 5 reps):**

| Size | Montgomery (OPTIMIZED) | Barrett | Delta |
|---|---|---|---|
| 2^15 | 0.339 ms | 0.397 ms | +17% slower |
| 2^16 | 0.579 ms | 0.668 ms | +15% slower |
| 2^18 | 1.36 ms | 1.35 ms | ~0% |
| 2^20 | 6.07 ms | 6.02 ms | -1% faster |
| **2^22** | **28.0 ms** | **27.3 ms** | **-2.5% faster** |

**Analysis:**
- Small sizes (2^15-2^16): Barrett slower because fused kernel is **compute-bound** (69%
  utilization) and Barrett has 68% more instructions per ff_mul. Conversion overhead is
  small at these sizes (~0.1 ms), so the extra compute cost dominates.
- Large sizes (2^20-2^22): Barrett faster — outer stages (77% of time) are **memory-bound**,
  so extra Barrett instructions are hidden. The ~3 ms conversion savings minus ~2.3 ms
  fused kernel overhead yields ~0.7 ms net improvement.
- Net improvement at 2^22 is -0.7 ms (not the projected -1.8 ms) because the fused kernel
  overhead is ~2.3 ms (not the estimated 1.2 ms). Barrett's 68% more instructions hit
  harder when the kernel is already at 69% compute utilization.
- **Decision**: Keep both paths. Barrett is the better default for n >= 2^20. Batching
  (Session 3) and 4-step NTT (v1.3.0) will benefit more since they reduce outer-stage
  DRAM passes where conversion savings compound.

**Tests:** 93/93 pass (70 existing + 23 new Barrett NTT tests).
Barrett and Montgomery outputs are **bitwise identical** at all tested sizes (2^8..2^22).

---

### Session 3 — Batched NTT Kernel ✅ COMPLETE

**Objective:** Process multiple independent NTTs in a single kernel launch for
dramatically better GPU utilization.

**Background — Why Batching:**
Groth16 proof generation requires ~9 independent NTTs of the same size. FHE workloads
need even more. Our current implementation processes them sequentially (1 NTT per call).
MoMA shows optimal batch size >8 for 128-384 bit inputs — matching Groth16's 9 NTTs.

Batching improves GPU utilization in two ways:
1. **More thread blocks** → more SMs active simultaneously (especially at smaller sizes
   where a single NTT doesn't fill all 30 SMs)
2. **Amortized launch overhead** → one launch for B NTTs instead of B separate launches
3. **Better memory bandwidth utilization** → interleaved access across B arrays

**Tasks:**
- Design batched NTT API:
  ```cpp
  void ntt_forward_batch(FpElement** d_arrays, int batch_size, uint32_t n, NTTMode mode);
  // or: contiguous layout with stride
  void ntt_forward_batch(FpElement* d_data, int batch_size, uint32_t n, NTTMode mode);
  ```
- **Fused kernel batching**: Each thread block already processes 1024 elements independently.
  For batch of B NTTs of size n: launch `B × (n/1024)` blocks. Block ID determines
  which NTT and which sub-array within that NTT. Shared memory usage unchanged.
- **Outer stage batching**: Cooperative kernel processes butterflies across all B arrays.
  `grid.sync()` synchronizes across all blocks (all NTTs must complete each stage together).
  Alternative: separate cooperative launch per NTT (simpler but more launches).
- **Bit-reverse batching**: Trivially parallelizable across B arrays.
- **Twiddle factor sharing**: All B NTTs of the same size share the same twiddle table.
  Single precomputation, B× reuse — better cache utilization.
- Correctness tests (see test plan below)
- Benchmark: batch throughput vs sequential for batch_size = 1, 2, 4, 8, 16, 32

**Expected results:**
- At n=2^15 (small): massive improvement (single NTT barely fills the GPU)
- At n=2^22 (large): moderate improvement (single NTT already fills SMs, but
  twiddle cache reuse and launch amortization still help)

**Test plan — Batched NTT:**
- **Batch vs sequential equivalence:**
  - Batch of B NTTs must produce identical output to B sequential single NTTs
  - Test for B = 1, 2, 4, 8, 16 at sizes 2^15, 2^18, 2^20, 2^22
  - Each NTT in the batch uses different random input data
- **Batch + Barrett cross-product:**
  - Batched Montgomery vs sequential Montgomery (same output)
  - Batched Barrett vs sequential Barrett (same output)
  - Batched Barrett vs batched Montgomery (same output)
- **Roundtrip for batched path:**
  - `batch_inverse(batch_forward(x)) = x` for all B arrays
  - Test for B = 1, 4, 8 at sizes 2^15, 2^20, 2^22
- **Edge cases:**
  - B = 1 (degenerate batch, must match single-NTT output exactly)
  - B = max that fits in 6 GB VRAM (stress test, verify no OOM or corruption)
  - Different input patterns per batch element (zeros, ones, random, p-1)
  - All-identical inputs across batch elements (detect cross-batch interference)
- **Independence / isolation:**
  - Corrupt one input array in batch, verify other outputs are unaffected
  - Verify no cross-NTT data leakage: batch element [i] output depends only on
    input [i], not on neighboring arrays
- **Twiddle sharing validation:**
  - Verify that batch mode and single mode use the same twiddle factors
    (no off-by-one in twiddle indexing due to batch offset arithmetic)

**Deliverables:**
- Batched NTT API and kernel implementations
- Correctness tests for batch mode (equivalence + edge cases + isolation)
- Throughput benchmark: NTTs/second for varying batch sizes

**Measured results (RTX 3060 Laptop, batch of 8 NTTs, Barrett, median of 3 reps):**

| Size | Batched 8× | Sequential 8× | Speedup | Per-NTT (batch) | Per-NTT (single) |
|---|---|---|---|---|---|
| 2^15 | 1.03 ms | 1.64 ms | **1.59x (37%)** | 0.129 ms | 0.158 ms |
| 2^18 | 9.73 ms | 10.8 ms | **1.11x (10%)** | 1.22 ms | 1.27 ms |
| 2^20 | 44.9 ms | 46.8 ms | **1.04x (4%)** | 5.61 ms | 5.64 ms |
| 2^22 | 200 ms | 205 ms | **1.02x (2%)** | 25.0 ms | 25.0 ms |

**Per-NTT cost with batch scaling (Montgomery, 2^15):**

| Batch | Total | Per-NTT | vs Single |
|---|---|---|---|
| B=1 | 0.131 ms | 0.131 ms | baseline |
| B=4 | 0.493 ms | 0.123 ms | -6% |
| B=8 | 0.962 ms | 0.120 ms | -8% |
| B=16 | 1.91 ms | 0.119 ms | -9% |

**Analysis:**
- Biggest wins at small sizes: single 2^15 NTT only fills ~32 blocks for 30 SMs.
  Batching with B=8 gives 256 blocks → much better SM occupancy.
- At 2^22, GPU is already saturated by a single NTT (4096 fused blocks for 30 SMs).
  Batching saves kernel launch overhead only (3-4 launches vs 32).
- Key design insight: existing fused kernel works for batching with zero changes —
  `boff = blockIdx.x * ELEMS` naturally addresses the right sub-array for contiguous
  batched data. Only the outer cooperative kernel and bit-reverse kernel needed new
  batched versions.
- Batching reduces total kernel launches from B×4 = 32 to 3-4 per batch.
- Tests: 119/119 pass (93 existing + 26 new batched NTT tests).

---

### Session 4 — Benchmark, Profile, Release v1.2.0 ✅ COMPLETE

**Objective:** Comprehensive benchmarking of Barrett + batching, profile, release.

**Tasks:**
- Full benchmark matrix: {Montgomery, Barrett} × {single, batch-8} × {2^15..2^22}
- Profile batched kernel with ncu: occupancy, SM utilization, memory throughput
- Profile Barrett vs Montgomery: instruction mix comparison (SASS)
- Capture screenshots: batched NTT occupancy, Barrett vs Montgomery roofline
- **Full regression sweep** (see test plan below)
- Update analysis.md with new sections on Barrett arithmetic and batching
- Update README performance tables (add batch throughput table)
- Update CLAUDE.md, GUIDE.md (add Barrett reduction section)
- Update plot_benchmarks.py with new data
- Tag v1.2.0 release

**Test plan — v1.2.0 release regression sweep:**
- **All existing v1.1.0 tests pass** (55/55 — no regressions in Montgomery path)
- **Full test matrix** (every combination must pass):
  | Mode | Single | Batch-1 | Batch-8 |
  |---|---|---|---|
  | Montgomery forward | sizes 2^10..2^22 | sizes 2^15..2^22 | sizes 2^15..2^22 |
  | Montgomery roundtrip | sizes 2^10..2^22 | sizes 2^15..2^22 | sizes 2^15..2^22 |
  | Barrett forward | sizes 2^10..2^22 | sizes 2^15..2^22 | sizes 2^15..2^22 |
  | Barrett roundtrip | sizes 2^10..2^22 | sizes 2^15..2^22 | sizes 2^15..2^22 |
- **Cross-path agreement**: Barrett single output = Montgomery single output for all sizes
- **Async pipeline**: verify pipeline still works with both Barrett and Montgomery modes
- **Test count target**: ≥100 total tests (55 existing + Barrett arithmetic + Barrett NTT
  + batched NTT + cross-validation + edge cases)

**Deliverables:**
- Updated benchmark tables + charts
- New ncu screenshots
- Full regression pass (all tests green)
- Git tag v1.2.0

**Measured results (RTX 3060 Laptop, 5-rep median):**

Single NTT forward:

| Size | Naive | Montgomery | Barrett | Barrett vs Montgomery |
|---|---|---|---|---|
| 2^15 | 0.122 ms | 0.132 ms | 0.159 ms | +20% slower |
| 2^16 | 0.228 ms | 0.244 ms | 0.308 ms | +26% slower |
| 2^18 | 1.34 ms | 1.21 ms | 1.27 ms | +5% slower |
| 2^20 | 5.88 ms | 5.51 ms | 5.61 ms | +2% slower |
| **2^22** | **26.2 ms** | **25.1 ms** | **24.9 ms** | **-1% faster** |

Batched 8× Barrett vs sequential:

| Size | Batched 8× | Sequential 8× | Speedup |
|---|---|---|---|
| 2^15 | 1.12 ms | 1.70 ms | **1.52x** |
| 2^18 | 10.4 ms | 11.2 ms | 1.08x |
| 2^20 | 48.0 ms | 48.3 ms | ~1.0x |
| 2^22 | 219 ms | 216 ms | ~1.0x |

**Analysis:**
- Barrett is faster at 2^22 single NTT (-1%) by avoiding Montgomery conversion overhead.
- Batching wins dramatically at small sizes (1.52x at 2^15 where GPU is underutilized).
- At 2^22, GPU is already saturated — batching saves only launch overhead.
- v1.2.0's primary value is infrastructure: Barrett elimination of conversion cost compounds
  in 4-step NTT (v1.3.0), and batched kernel API enables batch-of-sub-NTTs pattern.
- **Tests**: 119/119 pass (full regression clean).

---

## v1.3.0 — 4-Step NTT Algorithm

**Goal:** Restructure the NTT algorithm so all butterfly stages operate on shmem-resident
data, eliminating the memory-bound outer-stage bottleneck entirely. Combine with batching
for maximum throughput.

**Expected improvement:** 30-50% at n=2^22 (target: ≤16 ms single, much better batched)

### Background: 4-Step NTT (Bailey's Algorithm)

For n = n1 × n2, decompose the n-point NTT into:

1. **Column NTTs**: n2 independent n1-point NTTs (each fits in shared memory)
2. **Twiddle multiply**: pointwise multiplication by ω^(i·j) for all i,j
3. **Row NTTs**: n1 independent n2-point NTTs (each fits in shared memory)

For n=2^22 with n1=n2=2^11:
- Step 1: 2048 independent 2048-point NTTs → each uses the fused K=10 kernel (all in shmem)
- Step 2: 2^22 pointwise ff_mul operations (compute-bound, embarrassingly parallel)
- Step 3: 2048 independent 2048-point NTTs → same as step 1

**Key insight:** No outer stages at all. Every butterfly operates on shmem-resident data.
The only DRAM traffic is the initial load, the twiddle multiply pass, and the final store.

**Synergy with batching:** Step 1 is already a batch of 2048 sub-NTTs. With external
batching of B full NTTs, step 1 becomes B×2048 independent sub-NTTs — perfect GPU
utilization even on large GPUs.

**Synergy with Barrett:** 4-step NTT benefits even more from Barrett because the twiddle
multiply (step 2) is a standalone kernel with n pointwise multiplications — having data
in standard form avoids any conversion.

### Session 5 — 4-Step NTT: Transpose Kernel + Architecture ✅ COMPLETE

**Objective:** Design the 4-step decomposition and implement the transpose kernel.

**Tasks:**
- Design the 4-step decomposition for the NTT size range [2^10, 2^26]:
  - Choose n1, n2 split strategy (prefer n1=n2=√n when possible)
  - Handle non-square cases (n1 ≠ n2) for odd log_n
- Implement efficient matrix transpose kernel for FpElement data:
  - Shared-memory tiled transpose (16×16 tiles with +1 padding to avoid bank conflicts)
  - Coalesced reads + coalesced writes (classic GPU transpose pattern)
  - Handle non-square transpose (n1 ≠ n2)
- Implement twiddle multiply kernels for the middle step:
  - Barrett + Montgomery variants, single + batched
- Skeleton of `ntt_4step_forward` / `ntt_4step_inverse` host functions
- Unit tests for transpose kernel correctness

**Implementation details:**
- **Split strategy**: balanced — n1 = 2^(log_n/2), n2 = 2^(log_n - log_n/2)
  - Even log_n: n1 = n2 = sqrt(n) (e.g., 2^22 → 2048×2048)
  - Odd log_n: n1 < n2 (e.g., 2^21 → 1024×2048)
- **Transpose kernel**: TILE=16, 256 threads/block, shmem padded 16×17 (FpElement is 32B,
  so TILE=16 uses 8.5 KB shmem — TILE=32 would use 34 KB). Supports arbitrary n1×n2.
- **Batched transpose**: uses z-dimension gridDim for batch index
- **4-step skeleton**: forward + inverse Barrett paths. Uses 3 transposes
  (Session 6 determined 3 are needed for natural output order). Reuses existing batched sub-NTT infrastructure.
- **Twiddle multiply kernels**: pointwise data[i] *= omega_n^(i*j), Barrett and Montgomery

**Deliverables:**
- `src/ntt_4step.cu` with transpose kernel, twiddle multiply, and host dispatch skeleton
- Transpose correctness tests (15 new tests)
- Design doc in code comments explaining the decomposition

**Tests:** 134/134 pass (119 existing + 15 new):
- 4-step split computation (7 cases: even/odd log_n)
- Square transpose (16², 32², 64²)
- Rectangular transpose (16×32, 32×16, 256×512, 512×1024)
- Double-transpose roundtrip (4 sizes up to 1024×2048 = 2M elements)
- Batched transpose (16×32 B=4, 64×64 B=8)
- Large-scale 2048×2048 spot-check (actual 4-step decomposition size for n=2^22)

---

### Session 6 — 4-Step NTT: Sub-NTT Integration ✅ COMPLETE

**Objective:** Wire up the column NTTs, twiddle multiply, transpose, and row NTTs.

**Tasks:**
- Implement column NTT pass: launch fused kernel on n2 independent sub-arrays
  - Data layout: n1 × n2 matrix in row-major → column NTTs operate on columns
  - Option A: transpose first, then batch sub-NTTs on contiguous rows
  - Option B: strided sub-NTT launch (less efficient memory access)
- Implement twiddle multiply kernel: pointwise `ff_mul(data[i], twiddle[i])` for all i
  - Use Barrett if v1.2 showed it's faster, otherwise Montgomery
- Wire up the full sequence: transpose → column NTTs → twiddle → transpose → row NTTs
  (minimize total transposes — optimal is 2 transposes total)
- Handle bit-reverse permutation: integrate with sub-NTT or separate pass
- **Batch integration**: 4-step naturally supports batching — B full NTTs = B×n2 sub-NTTs
  in step 1. Implement batched 4-step from the start.

**Key design decision:** Sub-NTT size. For n=2^22:
- n1=n2=2^11: each sub-NTT is 2048 elements → fused K=10 handles it in 1 launch, no outer
  stages needed. **This is the sweet spot.**
- n1=2^10, n2=2^12: sub-NTTs of 1024 and 4096 → K=10 fuses all of 1024; 4096 needs outer stages

**Implementation details:**
- **3-transpose algorithm** (not 2): Bailey's decomposition uses mixed-radix output
  k = k1 + k2*n1 (column-major). After column NTTs + twiddle + row NTTs, the result
  at position [k1][k2] holds X[k2*n1+k1]. A final transpose (n1×n2 → n2×n1) places
  X[j] at flat index j for natural order.
- **Forward:** transpose → n2 column NTTs → transpose → twiddle → n1 row NTTs → transpose + copy
- **Inverse:** reverse forward steps — un-transpose → inv-row-NTTs → inv-twiddle → transpose
  → inv-column-NTTs → transpose + copy. Sub-NTT inverse functions handle n_sub^{-1}
  scaling (column INTTs: n1^{-1}, row INTTs: n2^{-1}, total: n^{-1}).
- **4-step twiddle cache:** sub-NTT twiddles for n1 and n2 sizes (shared when n1==n2 for even
  log_n, distinct for odd log_n) + omega_n^(i*j) table of size n. All in Barrett standard form.
- **Batched 4-step:** B full NTTs = B*n2 column sub-NTTs + B*n1 row sub-NTTs. Batched transposes
  via z-dimension gridDim. B*n twiddle multiply with index wrapping.
- **Kernel count:** 3 transposes + 1 memcpy + 2 batch sub-NTTs + 1 twiddle = 7 operations per NTT.

**Deliverables:**
- Complete `ntt_4step_forward_barrett` / `ntt_4step_inverse_barrett` (single + batched)
- `NTTMode::FOUR_STEP` in public API (`ntt_forward`, `ntt_inverse`, `ntt_forward_batch`, `ntt_inverse_batch`)
- 4-step twiddle cache in `ntt_naive.cu` (`ensure_twiddles_4step`)
- Correctness: 157/157 tests pass (134 existing + 23 new)
  - Forward vs CPU reference (5 sizes: 2^16, 2^17, 2^18, 2^20, 2^22)
  - Roundtrip INTT(NTT(x)) = x (5 sizes)
  - Cross-validation: 4-step == Barrett (bitwise identical, 5 sizes)
  - Batched forward vs sequential (3 sizes), batched roundtrip (3 sizes)
  - Batched 4-step vs batched Barrett (2^16, B=4)

---

### Session 7 — 4-Step NTT: Correctness + Edge Cases ✅ COMPLETE

**Objective:** Exhaustive correctness testing and edge case handling.

**Tasks:**
- Test 4-step NTT against CPU reference for all sizes 2^10 through 2^22
- Test roundtrip: INTT(NTT(x)) = x for all sizes
- Test with known vectors (all-zeros, all-ones, single-element, random)
- Test batched 4-step: batch of 8 vs 8 sequential single 4-step NTTs
- Handle sizes too small for 4-step (n < n1×n2 minimum) — fallback to fused+cooperative
- Handle sizes where √n is not integer (odd log_n): use n1=2^(k/2), n2=2^((k+1)/2)
- Verify inverse NTT uses correct inverse twiddles and n^{-1} scaling

**Implementation details:**
- **Fallback logic**: `NTTMode::FOUR_STEP` for n < 2^16 transparently delegates to Barrett
  path (sub-NTTs need K=8 minimum = 256 elements). Applied to all 4 dispatch points:
  `ntt_forward`, `ntt_inverse`, `ntt_forward_batch`, `ntt_inverse_batch`.
- **Forward + roundtrip**: tested at ALL sizes 2^10..2^22 (13 sizes). Both even log_n
  (n1=n2, true 4-step) and odd log_n (n1<n2, true 4-step) and small sizes (Barrett fallback).
- **Known-vector patterns**: 6 patterns (all-zeros, all-ones, single-nonzero-at-0,
  single-nonzero-at-mid, ascending, all-(p-1)) tested as roundtrip at 5 sizes.
- **Inverse explicit**: both inv(fwd(x))=x AND fwd(inv(x))=x verified at 2^16, 2^17, 2^20, 2^22.
- **Cross-validation**: 4-step vs Barrett bitwise identical at 11 sizes (2^10..2^22).
  Covers fallback path (2^10..2^15) and true 4-step (2^16..2^22), even + odd log_n.
- **Batched B=8**: vs sequential at 2^16 and 2^18 (key Groth16 workload size).
- **Additional batched**: 9 more size×B configurations (2^10 B=4, 2^15 B=4, 2^17 B=4,
  2^19 B=2, 2^22 B=2 sequential; 2^10 B=4, 2^15 B=8, 2^17 B=4, 2^19 B=2, 2^22 B=2 roundtrip).
- **Batched cross-validation**: 4-step batch vs Barrett batch at 5 configurations.

**Deliverables:**
- Fallback logic in `ntt_naive.cu` dispatch (all 4 API entry points)
- Full test coverage: 221 tests (157 existing + 64 new Session 7 tests)
- All existing tests pass (no regressions)

**Tests:** 221/221 pass (157 existing + 64 new):
- Forward GPU vs CPU reference: 13 sizes (2^10..2^22)
- Roundtrip INTT(NTT(x))=x: 13 sizes (2^10..2^22)
- Known-vector roundtrip: 5 sizes × 6 patterns = 30 sub-tests
- Forward zeros: 2 sizes (NTT(0) = 0)
- Inverse explicit: 4 sizes × 2 directions = 8 sub-tests
- 4-step vs Barrett cross-validation: 6 extra sizes
- Batched B=8 vs sequential: 2 sizes
- Batched vs sequential: 5 additional configurations
- Batched roundtrip: 5 additional configurations
- Batched 4-step vs batched Barrett: 5 configurations

---

### Session 8 — 4-Step NTT: Benchmark, Profile, Release v1.3.0 ✅ COMPLETE

**Objective:** Quantify the 4-step improvement and release.

**Result: NEGATIVE — 4-step is slower than cooperative approach at all sizes.**

**Measured results (RTX 3060 Laptop, 5-rep median):**

Single NTT forward:

| Size | Naive | Montgomery | Barrett | 4-Step | 4-Step vs Barrett |
|---|---|---|---|---|---|
| 2^15 | 0.122 ms | 0.132 ms | 0.158 ms | 0.160 ms | ~0% (Barrett fallback) |
| 2^16 | 0.228 ms | 0.245 ms | 0.300 ms | **0.491 ms** | **+64% slower** |
| 2^18 | 1.34 ms | 1.21 ms | 1.27 ms | **1.66 ms** | **+31% slower** |
| 2^20 | 5.88 ms | 5.50 ms | 5.60 ms | **7.03 ms** | **+26% slower** |
| **2^22** | **26.2 ms** | **25.1 ms** | **24.9 ms** | **29.5 ms** | **+18% slower** |

Batched 8× NTTs:

| Size | Barrett batch | Barrett seq | 4-Step batch | 4-Step seq |
|---|---|---|---|---|
| 2^15 | 1.01 ms | 1.54 ms | 1.01 ms | 1.61 ms |
| 2^18 | 9.65 ms | 10.8 ms | 11.4 ms | 17.2 ms |
| 2^20 | 44.6 ms | 47.5 ms | 53.1 ms | 66.9 ms |
| 2^22 | 199 ms | 208 ms | 241 ms | 279 ms |

**Root cause analysis:**
1. **Sub-NTTs still have outer stages**: 2048-element sub-NTTs need 11 stages; K=10 fuses
   10, but 1 cooperative outer stage remains per sub-NTT batch.
2. **L2 thrashing**: Batched sub-NTTs access the full n-element array (128 MB at 2^22),
   far exceeding the 4 MB L2 cache. The outer stage hits DRAM on every access.
3. **3 transpose passes**: Each reads and writes the full array — ~6n DRAM ops of pure overhead.
4. **Additional overhead**: twiddle multiply (3n), memcpy (2n), kernel launch (7 ops vs 4).
5. **CPU dispatch overhead**: 33.7 ms CPU time vs 25.0 ms for Barrett (complex host-side logic).

**Positive finding**: 4-step benefits more from batching than Barrett (1.16x vs 1.05x at 2^22)
due to internal batch structure. But in absolute terms, Barrett batched is still faster.

**Deliverables:**
- Benchmark data saved to `results/data/bench_v130.json`
- 3 new charts in `results/charts/` (four_step_vs_barrett, four_step_batch, all_modes_comparison)
- Updated analysis.md (Section 6), README.md, CLAUDE.md, roadmap
- All 221 tests pass (no regressions)
- Git tag v1.3.0

---

## v1.4.0 — MoMA-Inspired Register Optimization + CUDA Graphs

**Goal:** Push arithmetic performance closer to MoMA's auto-generated code by adopting
register-centric data flow patterns, and reduce launch overhead with CUDA Graphs.
Targets the cooperative outer stages directly (v1.3.0 showed 4-step adds overhead
rather than removing it).

**Expected improvement:** 10-20% single-NTT compute; 5-15% launch overhead reduction

### Session 9 — Register-Centric Butterfly + MoMA Carry Chain Patterns

**Objective:** Adopt MoMA's register-centric approach — minimize shared memory pressure
and maximize register-resident computation.

**Background — MoMA's Approach:**
MoMA's recursive rewriting decomposes multi-word operations so that the code generator
can keep intermediate values in registers and schedule carry-chain instructions optimally.
We can't replicate the full SPIRAL code generator, but we can adopt the patterns:

1. **Register-resident butterfly**: Instead of loading from shared memory, computing,
   and storing back, keep both butterfly elements in registers across multiple stages
   (for warp-shuffle stages, this is already partially done)
2. **Carry chain scheduling**: MoMA's rewrite rules produce carry chains where each
   `add.cc` / `addc.cc` / `madc.hi.cc` is scheduled to minimize stalls between
   dependent instructions. Review our PTX carry chains against MoMA's patterns.
3. **Operand reuse**: MoMA tracks data dependencies to minimize register spills.
   Ensure our ff_mul doesn't spill to local memory at high register pressure.

**Tasks:**
- Analyze register pressure in current fused kernel (ncu: register usage, local memory spills)
- Implement register-resident butterfly variant for warp-shuffle stages:
  - Each thread holds 2 FpElements (64 bytes = 16 registers) permanently
  - For K=10: stages 0-4 are already warp-shuffle (register-to-register)
  - Stages 5-9: explore keeping data in registers with warp shuffle for stride > 16
    (requires multi-warp shuffle protocol — each thread shuffles with tid ± stride,
    which may span warps. If stride > 32, shared memory is unavoidable.)
- Review and optimize PTX carry chain scheduling in ff_mul:
  - Compare our CIOS carry chain instruction sequence vs MoMA-style decomposed sequence
  - Identify any stall bubbles between dependent carry-chain instructions
  - Test reordering independent limb operations to fill stall slots
- Benchmark: measure IPC and stall cycles before/after optimization
- Profile with ncu: `smsp__inst_executed_pipe_alu` counter, warp stall breakdown

**Deliverables:**
- Optimized carry chain scheduling in `ff_arithmetic.cuh`
- Register pressure analysis (ncu screenshots)
- Benchmark: IPC comparison for fused kernel

---

### Session 10 — Radix-4 Outer-Stage Optimization ✅ COMPLETE

**Objective:** Reduce the memory-bound outer-stage bottleneck (77% of time at n=2^22)
by fusing pairs of outer stages into radix-4 butterflies, halving DRAM passes.

**Approach — Radix-4 butterfly:**
Each radix-4 unit processes 4 data elements across 2 consecutive stages with:
- 4 global memory loads + 4 stores (vs 8+8 for 2 radix-2 stages)
- 3 twiddle loads + 4 multiplications + 4 additions + 4 subtractions
- ~45% reduction in DRAM traffic (theoretical)

For stages (s, s+1) with half = 2^s:
- 4 elements at indices base+{0, half, 2·half, 3·half}
- Stage s: 2 radix-2 butterflies on (a0,a1) and (a2,a3) with w_s(j)
- Stage s+1: 2 radix-2 butterflies on (a0',a2') and (a1',a3') with w_{s+1}

**Implementation:**
- 6 new cooperative kernels: Barrett/Montgomery × single/batch × radix-4
  (`ntt_outer_radix4_kernel`, `ntt_outer_radix4_barrett_kernel`,
   `ntt_outer_radix4_batch_kernel`, `ntt_outer_radix4_batch_barrett_kernel`)
- All 4 dispatch functions modified to try radix-4 first, fall back to radix-2
- Odd outer stage count: radix-4 for pairs + 1 radix-2 leftover at end
- For n=2^22 (12 outer stages): 6 radix-4 passes in 1 cooperative launch
  (was 12 radix-2 stages in 2 cooperative launches)
- Fault-tolerant occupancy queries (graceful fallback if kernel symbol not found)

**Results (n=2^22, 7-rep median, Naive=26.2ms unchanged):**

| Mode | v1.4.0-s9 (radix-2) | v1.4.0-s10 (radix-4) | Delta |
|------|---------------------|----------------------|-------|
| Montgomery | 24.4 ms | **17.0 ms** | **-30.3%** |
| Barrett | 23.8 ms | **17.1 ms** | **-28.2%** |

| Mode | v1.2.0 (original) | v1.4.0-s10 (radix-4) | Delta |
|------|-------------------|----------------------|-------|
| Montgomery | 25.1 ms | **17.0 ms** | **-32.3%** |
| Barrett | 24.9 ms | **17.1 ms** | **-31.3%** |

Outer stages estimated improvement: ~19.4 ms → ~11 ms (**-43%**, matching theoretical).
Montgomery and Barrett now nearly identical at 2^22 (conversion overhead vs heavier Barrett mul).

**Tests:** 221/221 pass (no regressions, existing tests exercise radix-4 through all NTT modes)

---

### Session 11 — CUDA Graphs + Final Polish + Release v1.4.0 ✅

**Objective:** Wrap all launch sequences in CUDA Graphs, final benchmarking, release.

**Implementation:**
- CUDA Graph capture/replay API: `ntt_forward_graph()`, `ntt_inverse_graph()`,
  `ntt_forward_batch_graph()`, `ntt_inverse_batch_graph()`, `ntt_graph_clear_cache()`
- Graph cache: keyed by `(d_data, n, batch_size, mode, forward)`, captures on first call,
  replays on subsequent calls with `cudaGraphLaunch` (~5us vs ~20-40us per individual launch)
- Capture uses `cudaStreamCaptureModeRelaxed` to allow occupancy queries during capture
- Pre-warms twiddle caches before capture to avoid `cudaMalloc` during stream capture
- Supported modes: NAIVE, OPTIMIZED, BARRETT (FOUR_STEP excluded — internal `cudaMalloc`)
- Cooperative kernel launches (`cudaLaunchCooperativeKernel`) captured automatically

**Measured impact (7-rep median, head-to-head, RTX 3060):**
- **Negligible performance difference** — within ±2% measurement noise at all sizes
- At n=2^22: Montgomery 17.1ms (non-graph) vs 17.4ms (graph), Barrett 17.4ms both
- At n=2^15: Montgomery 0.120ms vs 0.129ms, Barrett 0.142ms vs 0.151ms
- **Root cause**: only 3-4 kernel launches per NTT → CPU launch overhead is ~50-100us total,
  negligible compared to GPU compute time even at the smallest sizes
- The API remains valuable for embedding NTT in larger CUDA Graph workflows

**Tests:** 230/230 pass (221 existing + 9 new graph tests)
- Graph vs non-graph bitwise equivalence: Barrett fwd (2^15, 2^18, 2^22), Montgomery fwd (2^15, 2^22)
- Graph roundtrip: Barrett INTT(NTT(x))=x (2^15, 2^22)
- Graph replay consistency: second call matches first (2^15)
- Batched graph vs non-graph: Barrett B=4 at 2^15

---

## v1.5.0 — Radix-8 Outer Stages + On-the-Fly Twiddles

**Goal:** Continue the radix-4 strategy to its logical conclusion — fuse triples of outer
stages into radix-8 butterflies, and eliminate twiddle DRAM traffic via on-the-fly computation.
Preceded by L2 cache diagnostic to inform this and future releases.

**Final result (v1.5.0 released):** Montgomery radix-8 gives **−9.4%** (**15.5 ms** vs v1.4.0's 17.1 ms).
Barrett radix-8 **disabled** (I-cache regression: 29.5 ms, +73%). OTF twiddles: **negative result**
(56.9 ms, +265% — BLS12-381 exponentiation too expensive). Batched 8× Montgomery: **130 ms**.

**Rationale:** v1.4.0's radix-4 gave −30% by halving outer-stage DRAM passes (12→6).
Radix-8 fuses 3 stages per pass: 12→4 passes (another −33% from radix-4 baseline).
Each eliminated pass saves ~128 MB DRAM R+W = ~0.7 ms at bandwidth ceiling.
However, Barrett radix-8's 174 registers cause I-cache thrashing (55% I-fetch stalls),
so only Montgomery benefits from radix-8 (134 regs). OTF twiddle computation was expected to
eliminate twiddle-read DRAM traffic (~1 ms savings), but BLS12-381's 256-bit exponentiation
cost (~35 muls × 128 MADs/mul per butterfly) far exceeds DRAM read cost. OTF disabled.

**Key references:**
- Özcan, Javeed, Savaş — "High-Performance NTT on GPU", IEEE Access 2025
- Kim et al. — "Accelerating NTT for Bootstrappable HE on GPUs", 2020 (OTF twiddles)

---

### Session 12 — L2 Diagnostic + Radix-8 Butterfly Design ✅ COMPLETE

**Objective:** Profile the outer stages to determine whether they are latency-bound or
bandwidth-bound (informing Stockham feasibility in v1.8.0), and design + implement the
radix-8 butterfly device function with register pressure analysis.

**Part A — L2 Hit-Rate Diagnostic: COMPLETE**

Profiled with ncu 2025.1.1 `--set full` at 2^18, 2^20, 2^22. Results:

| Size | Working Set | L2 Hit Rate (radix-8 outer) |
|------|------------|---------------------------|
| 2^18 | 8 MB | 64.8% |
| 2^20 | 32 MB | 60.6% |
| 2^22 | 128 MB | **58.5%** |

**Decision: Stockham v1.8.0 → NO-GO.** L2 hit rate 58.5% at 2^22 is well above the 50%
threshold — outer stages are bandwidth-bound, not latency-bound. Stockham's sequential
access pattern would not improve L2 reuse.

Additional findings from full profile (radix-8 Barrett outer at 2^22):
- Occupancy: 16.7% (174 regs, register-limited, 1 block/SM)
- 50% uncoalesced global accesses (stride-h pattern, 16/32 bytes per sector)
- Top stall: instruction fetch/select (55.3% of stall cycles)
- ALU utilization: 20.7%
- ncu-rep files: `results/data/l2_diag_*.ncu-rep`

**Part B — Radix-8 Butterfly: COMPLETE**

Implemented 4 cooperative kernels + 8 occupancy helpers + 4 launch helpers:
- `ntt_outer_radix8_kernel` / `_barrett_kernel` / `_batch_kernel` / `_batch_barrett_kernel`
- Dispatch updated: radix-8 → radix-4 → radix-2 fallback in all 4 dispatch functions
- Leftover handling: 0→done, 1→radix-2 launch, 2→radix-4 cooperative launch

Register pressure (actual, no spills):

| Kernel | Registers | Blocks/SM | vs Radix-4 |
|--------|-----------|-----------|-----------|
| Radix-8 Montgomery (single) | 134 | 1 | +48 (vs 86) |
| Radix-8 Montgomery (batch) | 128 | 1 | +42 (vs 86) |
| Radix-8 Barrett (single) | 174 | 1 | +76 (vs 98) |
| Radix-8 Barrett (batch) | 174 | 1 | +76 (vs 98) |

All 230 existing tests pass. Profiling confirms radix-8 path is active at all sizes ≥ 2^13.

---

### Session 13 — Radix-8 Correctness Tests + Benchmark ✅ COMPLETE

**Objective:** Validate radix-8 correctness at all sizes with dedicated tests, and benchmark
radix-8 vs radix-4 to measure actual performance impact.

**Part A — Correctness Tests: COMPLETE (87 new tests, 317 total)**

Added comprehensive radix-8 tests covering all leftover patterns:
- Forward NTT vs CPU reference at 2^13..2^22 (both OPTIMIZED and BARRETT) — 20 tests
- Roundtrip INTT(NTT(x)) = x at 2^13..2^22 (both modes) — 20 tests
- Cross-validation Barrett == Montgomery at 2^13..2^22 — 10 tests
- Batched B=8 vs sequential at 2^15, 2^18, 2^20, 2^22 (both modes) — 7 tests
- Batched roundtrip B=8 at 2^15, 2^18, 2^20 (both modes) — 6 tests
- Known vectors at %3=0,1,2 leftover sizes (both modes) — 6 tests
- Forward zeros at 2^13, 2^16, 2^22 (both modes) — 6 tests
- Inverse explicit at 2^14, 2^15, 2^22 (both modes) — 6 tests + 6 sub-tests

**Part B — Benchmark: COMPLETE**

**Key finding: Barrett radix-8 is catastrophically slow due to I-cache thrashing.**

Barrett radix-8 (174 regs) at 2^22: **29.5 ms** vs radix-4 baseline **17.1 ms** = +73% regression.
Montgomery radix-8 (134 regs) at 2^22: **15.4 ms** vs radix-4 baseline **17.0 ms** = −8.2% improvement.

Root cause: Barrett radix-8's 174 registers + bloated instruction footprint causes 55.3%
instruction fetch stalls (profiled in Session 12). Montgomery radix-8 (134 regs) has a
smaller instruction footprint and benefits from the DRAM traffic reduction.

**Decision: Montgomery-only radix-8.** Barrett falls back to radix-4 (−73% regression eliminated).

Single NTT forward (7-rep median, RTX 3060):

| Size | Montgomery (radix-8) | Barrett (radix-4) | v1.4.0 Montgomery | v1.4.0 Barrett |
|------|---|---|---|---|
| 2^15 | 0.121 ms | 0.141 ms | 0.132 ms | 0.159 ms |
| 2^16 | 0.225 ms | 0.255 ms | 0.244 ms | 0.308 ms |
| 2^18 | 0.900 ms | 0.972 ms | 1.21 ms | 1.27 ms |
| 2^20 | 3.84 ms | 4.15 ms | 5.51 ms | 5.61 ms |
| **2^22** | **15.6 ms** | 18.0 ms | 17.0 ms | 17.1 ms |

Batched 8× NTT forward (7-rep median):

| Size | Montgomery (radix-8) | Barrett (radix-4) |
|------|---|---|
| 2^15 | 0.788 ms | 0.957 ms |
| 2^18 | 7.08 ms | 8.12 ms |
| 2^20 | 32.3 ms | 36.2 ms |
| 2^22 | 139 ms | 167 ms |

Montgomery radix-8 is now the fastest path at all sizes. Barrett with radix-4 is competitive
at small sizes but Montgomery wins decisively at 2^20+ due to radix-8 outer stages.

Benchmark data: `results/data/bench_v150_s13.json`

---

### Session 14 — On-the-Fly Twiddle Computation ✅ (NEGATIVE RESULT)

**Objective:** Replace precomputed twiddle table loads with on-the-fly computation in the
outer-stage kernels (Montgomery radix-8 + Barrett radix-4), eliminating twiddle DRAM reads.

**Result: NEGATIVE.** OTF twiddle computation is catastrophically expensive for BLS12-381.
The 256-bit Montgomery exponentiation (`ff_pow_mont_u32`) costs ~29 Montgomery muls per
butterfly at large outer stages (j up to 2^19), each mul being ~128 MADs (8-limb CIOS).
This far exceeds the DRAM savings from eliminating 7 twiddle table loads (224 bytes).
Warp divergence in the pow loop (`if ((exp >> bit) & 1)`) further degrades throughput.

**Measured (n=2^22, 7-rep median):**
- Montgomery: **56.9 ms OTF vs 15.6 ms precomputed (+265%)**
- Barrett: **91.4 ms OTF vs 17.9 ms precomputed (+411%)**

**What was implemented (all retained for future multi-field work):**
- `include/twiddle_otf.cuh`: `ff_pow_mont_u32`, `ff_pow_barrett_u32` (MSB-first binary exponentiation)
- `__constant__` memory: 32 stage roots + 3 fixed constants (omega_4, omega_8, omega_8^3), <1.2 KB
- 12 OTF kernel variants: radix-8/4/2 × Montgomery/Barrett × single/batch (cooperative)
- Stage root precomputation: repeated squaring from omega_n, upload via `cudaMemcpyToSymbol`
- CUDA Graph compatibility: guard flag prevents `cudaMemcpyToSymbol` during stream capture
- Radix-8 OTF derivation: 1 exponentiation + 6 multiplies (w4=root^j, w2=w4², w1=w2², w3=w2·ω₄, etc.)

**Bugs fixed during implementation:**
- Batched OTF launch helpers could partially transform data (radix-8/4 passes) then `return false`
  for leftover stages, causing precomputed fallback to re-do ALL stages on already-modified data.
  Fixed by adding batched OTF radix-2 cooperative kernels for leftover handling.
- FOUR_STEP fallback paths (n < 2^16) missing `upload_otf` calls. Fixed.

**OTF disabled** in all 4 dispatch functions. Infrastructure retained for v1.6.0 multi-field work
(Goldilocks 64-bit / BabyBear 31-bit fields where multiply is 1-2 instructions, not 128 MADs).

**Tests:** 333/333 pass (317 existing + 16 new OTF-specific tests)
- Stage root squaring chain: root[s] == root[s+1]² at 3 sizes
- OTF twiddle value consistency: root[s]^j == omega^(j*stride) at 3 sizes
- OTF NTT leftover pattern coverage: all Montgomery (%3=0,1,2) and Barrett (%2=0,1) patterns

**Why OTF fails for BLS12-381 but may work for smaller fields:**
- BLS12-381: 256-bit, 8-limb CIOS → ~128 MADs per multiply → ~35 muls/butterfly = ~4480 MADs
- Goldilocks: 64-bit → ~1 MUL instruction per multiply → ~35 muls/butterfly = ~35 instructions
- BabyBear: 31-bit → ~1 MUL instruction per multiply → ~35 muls/butterfly = ~35 instructions
- 7 DRAM reads (224 bytes at ~80ns each ≈ 560ns) vs 35 instructions at ~0.7ns = ~25ns. Clear win.

**Benchmark data:** `results/data/bench_v150_s14.json`

---

### Session 15 — v1.5.0 Benchmark + Profile + Release ✅

**Objective:** Comprehensive benchmarking of radix-8 (OTF disabled — negative result),
profiling, release.

**Completed:**
- Full benchmark matrix: {Montgomery, Barrett} × {2^15..2^22}
  - Single NTT forward, 7-10 rep median
  - Batched 8×, 7-rep median
- Profile data from Session 12 (ncu-rep files in `results/data/`)
- 3 new benchmark charts: v1.5.0 vs v1.4.0, version history, batched comparison
- Full regression sweep: **333/333 tests pass**
- Updated `results/analysis.md` (Section 10: v1.5.0 Results)
- Updated `results/data/bench_v150.json`
- Updated README performance tables
- Git tag v1.5.0

**Measured results (n=2^22, 10-rep median):**
- Montgomery (radix-8): **15.5 ms** (−9.4% vs v1.4.0's 17.1 ms)
- Barrett (radix-4, unchanged): **17.5 ms**
- Batched 8× Montgomery: **130 ms**; Barrett: **148 ms**

**Benchmark data:** `results/data/bench_v150.json`

---

## v1.6.0 — Multi-Field NTT (Goldilocks + BabyBear)

**Goal:** Add Goldilocks (2^64 − 2^32 + 1) and BabyBear (2^31 − 2^27 + 1) field support,
enabling 3-way performance comparison across BLS12-381 / Goldilocks / BabyBear — the three
most relevant fields in the ZKP ecosystem. This is the highest impact-to-effort ratio item
remaining: ~2-3 days of work, compelling benchmark figure, demonstrates ecosystem breadth.

**Expected improvement:** Not a BLS12-381 speedup — this is a portfolio/insight feature.
Goldilocks NTT at n=2^22 expected ~1.0–1.5 ms (10–15× faster than BLS12-381).
BabyBear NTT at n=2^22 expected ~0.3–0.5 ms (~35–50× faster).

**Rationale:** The retrospective doc's key insight: "the whole ZKP community is moving from
pairing-based to hash-based proofs — it's a 10–15× field arithmetic difference, not an
algorithmic one." A concrete benchmark proves this claim with data on our hardware.

**References:**
- Özcan & Savaş (IEEE Access 2025): Goldilocks NTT on RTX 3060Ti
- RISC Zero: BabyBear for STARKs
- Plonky2/3 (Polygon): Goldilocks for recursive SNARKs

---

### Session 16 — Goldilocks + BabyBear Field Arithmetic ✅

**Objective:** Implement modular arithmetic for both fields (GPU + CPU reference) and
validate with exhaustive tests.

**Result:** Both fields implemented and validated. 374/374 tests pass (333 existing + 41 new).
- Goldilocks: `ff_goldilocks.cuh` (PTX mul.lo/hi.u64 + Goldilocks reduction), `GlRef` CPU reference
- BabyBear: `ff_babybear.cuh` (64-bit prod % p), `BbRef` CPU reference
- GPU throughput kernels: `src/ff_multi_field.cu` (8 kernels)
- Microbenchmark (mul, n=2^22): BLS 1444μs (2.9 Gops/s), GL 401μs (10.7 Gops/s, **3.6x**), BB 213μs (19.7 Gops/s, **6.8x**)
- All 3 fields DRAM-bandwidth-bound at 2^22 — speedup tracks element size ratio (32/8=4x, 32/4=8x)

**Part A — Goldilocks field (p = 2^64 − 2^32 + 1):**

- Representation: single `uint64_t` (1 limb!)
- `ff_add_goldilocks(a, b)`: `a + b mod p` — add, conditional subtract p
  - Branchless: `uint64_t s = a + b; uint64_t c = (s < a); s -= p; s += p & -(s > s+p);`
  - Or use the Goldilocks trick: `(a + b) mod (2^64 − 2^32 + 1)` via hi/lo split
- `ff_sub_goldilocks(a, b)`: `a - b mod p` — subtract, conditional add p (branchless)
- `ff_mul_goldilocks(a, b)`: 64×64→128 bit multiply, then reduce mod p
  - `uint128 prod = (uint128)a * b; uint64_t lo = prod; uint64_t hi = prod >> 64;`
  - Goldilocks reduction: `hi * 2^32 - hi + lo mod p` (special form of p enables this)
  - Or: Barrett with μ fitting in 2 × uint64
  - Or: Montgomery with R = 2^64
  - Key: this is ~5–8 instructions vs 528 for BLS12-381
- GPU implementation: `include/ff_goldilocks.cuh`
  - Struct: `struct GoldilocksElement { uint64_t val; };`
  - All functions: `__device__ __forceinline__`
- CPU reference: `tests/ff_reference.h` (add Goldilocks section)
  - Use `__uint128_t` (GCC) or `_umul128` (MSVC) for 128-bit product

**Part B — BabyBear field (p = 2^31 − 2^27 + 1 = 0x78000001):**

- Representation: single `uint32_t` (1 limb!)
- `ff_add_babybear(a, b)`: `a + b mod p` — branchless subtract if overflow
- `ff_sub_babybear(a, b)`: `a - b mod p` — branchless add if underflow
- `ff_mul_babybear(a, b)`: 32×32→64 bit multiply, then reduce mod p
  - From retrospective doc: ~5 instructions total
  - `uint64_t wide = (uint64_t)a * b;`
  - Special form: p = 2^31 - 2^27 + 1 allows shift-based reduction
  - `uint32_t lo = (uint32_t)(wide & 0x7FFFFFFF); uint32_t hi = (uint32_t)(wide >> 31);`
  - `result = lo + hi * 2^27 - hi;` (since 2^31 ≡ 2^27 - 1 mod p)
  - Conditional subtract if result ≥ p
- GPU implementation: `include/ff_babybear.cuh`
  - Struct: `struct BabyBearElement { uint32_t val; };`
- CPU reference: straightforward 64-bit arithmetic

**Tasks:**
- Implement `include/ff_goldilocks.cuh`: add, sub, mul, from_uint64, to_uint64
- Implement `include/ff_babybear.cuh`: add, sub, mul, from_uint32, to_uint32
- Implement CPU reference for both in `tests/ff_reference.h`
- Element types: separate structs (not template — different sizes, different reduction)
- Microbenchmark: isolated ff_mul throughput for all 3 fields (extend `ff_microbench.cu`)

**Test plan — Field arithmetic correctness:**
- **Edge-case inputs (GPU and CPU reference must agree, both fields):**
  - Identity: a * 1 = a, a * 0 = 0 for random a
  - Boundary: (p-1) * (p-1), (p-1) * 2, 0 * a
  - Small values: 1*1, 2*3, single-word values
  - Near-modulus: p-1, p-2
- **Cross-validation (GPU vs CPU reference):**
  - 10^6 random pairs for mul, 10^4 for add/sub
- **Algebraic properties:**
  - Commutativity: a*b = b*a (10^4 random pairs)
  - Associativity: (a*b)*c = a*(b*c) (10^3 triples)
  - Distributivity: a*(b+c) = a*b + a*c (10^3 triples)
- **Microbenchmark**: throughput comparison chart (ops/sec for all 3 fields)

**Deliverables:**
- `include/ff_goldilocks.cuh` — Goldilocks field arithmetic
- `include/ff_babybear.cuh` — BabyBear field arithmetic
- CPU reference implementations in `tests/ff_reference.h`
- Field arithmetic tests: ~20 new tests per field (~40 total)
- Microbenchmark data: 3-field throughput comparison

---

### Session 17 — Multi-Field NTT Integration + Correctness ✅

**Objective:** Implement NTT kernels for Goldilocks and BabyBear fields, validate
correctness, handle field-specific optimizations.

**Result:** Both fields' NTT kernels implemented and validated. 458/458 tests pass (374 existing + 84 new).
- Goldilocks NTT: `src/ntt_goldilocks.cu` (~700 lines) + `include/ntt_goldilocks.cuh`
- BabyBear NTT: `src/ntt_babybear.cu` (~700 lines) + `include/ntt_babybear.cuh`
- Architecture (both fields): fused inner K=8/9/10/11 + cooperative radix-8 → radix-4 → radix-2 outer
- K=11 max: 2048 elements, 1024 threads (limited by threads/block, not shmem)
  - GL shmem: 2048 × 8B = 16 KB; BB shmem: 2048 × 4B = 8 KB
- Standard form throughout (no Montgomery conversion overhead)
- Register pressure trivial: GL ~40-50 regs, BB ~20-30 regs (vs 134 BLS12-381 Montgomery)
- Twiddle precomputation via CPU reference (GlRef/BbRef), uploaded once to device
- Radix-8 default for outer stages (no register pressure concerns)
- No separate fused kernel TU needed (no RDC template issues with simple scalar arithmetic)

**Design decision — Separate kernels (chosen):** Each field has fundamentally different
element size (32B vs 8B vs 4B), shared memory per element, and register pressure.
Separate implementation is cleaner and allows per-field tuning.

**K selection logic (both fields):**
- K=11 for log_n >= 11 (2048 elements, 1024 threads)
- K=10 for log_n >= 10
- K=9 for log_n == 9
- K=8 default (log_n == 8)

**Note on deeper fusion (K=12/K=13):** Not implemented. While shmem would allow it
(GL: 32 KB for K=12, BB: 32 KB for K=13), the 1024 threads/block hardware limit means
K=11 is the practical maximum (2^K / 2 = 1024 threads for K=11). Going beyond K=11
would require fundamentally different thread-to-element mapping.

**Warp shuffle details:**
- Goldilocks: `gl_shfl_xor` splits uint64_t into 2×uint32_t for `__shfl_xor_sync`
- BabyBear: `bb_shfl_xor` uses direct uint32_t `__shfl_xor_sync` (single instruction)

**Tests (84 new, 42 per field):**
- Forward NTT vs CPU reference: 11 sizes (2^8..2^22, all K paths + leftover patterns)
- Roundtrip INTT(NTT(x)) = x: 8 sizes (2^8..2^22 step 2)
- Known vectors: 3 sizes × 4 patterns (zeros, ones, single-nonzero, ascending) = 12
- Forward zeros NTT(0) = 0: 2 sizes
- Inverse explicit fwd(inv(x)) = x: 2 sizes
- Batch vs sequential: 4 configs (B=4, B=8)
- Batch roundtrip: 3 configs

**Bug fixed:** Radix-8 single kernel in ntt_goldilocks.cu initially had a double-load
bug (botched first butterfly attempt + redundant global memory reload). Fixed by removing
the redundant first attempt, keeping only the clean radix-8 butterfly matching the batch kernel.

---

### Session 18 — Multi-Field Benchmark + Charts + Release v1.6.0 ✅

**Objective:** Comprehensive 3-way benchmark, generate comparison charts, release.

**Completed:**
- Full benchmark matrix: {BLS12-381, Goldilocks, BabyBear} × {2^10..2^22} × {single, batch-8}
  - 7-rep median, 2 warmup reps per configuration
  - Standalone benchmark binary: `benchmarks/bench_multifield.cu` (JSON output)
- 5 new comparison charts in `results/charts/`:
  - `multifield_latency.png`: 3 fields × 7 sizes bar chart (wall-clock time, log scale)
  - `multifield_speedup.png`: GL/BB speedup vs BLS12-381 (shows convergence at large sizes)
  - `multifield_throughput.png`: elements/sec (shows DRAM bandwidth ceiling)
  - `multifield_batch.png`: batch efficiency per field
  - `version_history_v160.png`: all versions + multi-field at n=2^22
- Analysis: speedup converges at large sizes because outer stages are memory-bound
- Updated `results/analysis.md` (Section 8: Multi-Field Comparison)
- Updated CLAUDE.md with v1.6.0 results
- All 458 tests pass

**Measured results (n=2^22, single NTT, 7-rep median):**
- BLS12-381 Montgomery: **15.1 ms** (consistent with v1.5.0)
- Goldilocks: **3.6 ms** (4.2x faster — lower than projected 10-15x)
- BabyBear: **2.4 ms** (6.2x faster — lower than projected 35-50x)

**Why projections were off:** The roadmap assumed arithmetic-dominated workload. In reality,
outer stages (memory-bound DRAM read-modify-write) account for ~60% of total time at 2^22.
Smaller field elements reduce DRAM traffic (8x less for BB vs BLS), but uncoalesced access
patterns (50%) and L2 miss rates (~58%) are structural — they don't improve with smaller elements.
The achieved 6.2x speedup tracks the DRAM traffic ratio (8x) discounted by fixed overhead.

**Deliverables:**
- 3-field benchmark data: `results/data/bench_v160.json`
- 5 comparison charts in `results/charts/`
- Updated analysis.md, CLAUDE.md, roadmap
- All 458 tests green

---

## v1.7.0 — Plantard Reduction for BLS12-381 (NEGATIVE RESULT)

**Goal:** Implement Plantard modular reduction as an alternative to Montgomery/Barrett for
the compute-bound fused inner kernel. Plantard eliminates one big-integer multiplication
per twiddle multiply by precomputing twiddle-specific Plantard constants.

**NEGATIVE RESULT (Session 19):** Plantard requires **944 SASS instructions** per multiply —
79% more than Montgomery v2 (528 SASS). The z × μ step (512×512-bit multiply mod R²) costs
136 MADs alone, more than the entire Montgomery CIOS (136 MADs total). Plantard's advantage
only applies to word-size moduli (32/64 bits); for 256-bit multi-limb BLS12-381, both product
and reduction are O(n²) schoolbook multiplies with no shortcut. Session 20 cancelled.

**Original expected improvement:** 5–15% on fused inner kernel → 2–5% total NTT improvement.
This was a targeted optimization for the inner kernel (now ~35% of total time post-radix-8).

**Rationale:** Unlike MoMA (which requires SPIRAL code generation and has 381-bit efficiency
problems), Plantard is a clean mathematical technique with proven GPU results. Özcan & Savaş
(IEEE Access 2025) benchmark Plantard vs Montgomery on RTX 3060Ti, showing 8–15% improvement
on compute-bound NTT inner stages for 60-bit primes. For 255-bit BLS12-381, the saving is
less dramatic but still saves one 8-limb multiplication per butterfly.

**References:**
- Plantard, "Efficient Word Size Modular Arithmetic", IEEE TETC 2021
- Özcan & Savaş (IEEE Access 2025) §V-C: Plantard benchmark on RTX GPUs

---

### Session 19 — Plantard Arithmetic + Twiddle Precomputation ✅ COMPLETE (NEGATIVE RESULT)

**Objective:** Implement Plantard reduction for BLS12-381 and precompute Plantard-form
twiddle factors.

**Background — Plantard reduction:**
Standard Montgomery butterfly: `t = mont_mul(b, w)` requires 2 big-integer multiplies
(b×w product + Montgomery correction × p). Plantard precomputes `w' = w × 2^(-2·256) mod p`
for each twiddle factor w. At runtime: `t = plantard_mul(b, w')` requires only 1 big-integer
multiply (upper half of b × w') + 1 shift — the correction multiply is eliminated.

For NTT, twiddle factors are **constants** (precomputed once), so the precomputation cost
is amortized over all NTT invocations.

**Tasks:**
- Implement Plantard multiplication in `include/ff_plantard.cuh`:
  ```cuda
  __device__ __forceinline__
  FpElement ff_mul_plantard(const FpElement& a, const FpElement& w_prime) {
      // Step 1: full 512-bit product z = a × w_prime
      uint32_t z[16];
      // ... (same schoolbook multiply as Barrett step 1)

      // Step 2: take upper 256 bits: result = z[8..15]
      // (Plantard property: upper half of a × w' is a × w mod p,
      //  provided w' = w × 2^(-512) mod p was precomputed correctly)

      // Step 3: conditional reduction (at most 1 subtract)
      // Plantard output range: [0, 2p) → branchless sub + lop3 select
      FpElement result;
      // ... (same branchless pattern as ff_mul_ptx step 3)
      return result;
  }
  ```
- Implement Plantard twiddle precomputation:
  - For each twiddle w in the existing table, compute `w' = w × R^(-2) mod p`
    where R = 2^256
  - Need: `R^(-2) mod p` = modular inverse of R² mod p
  - Precompute `R_INV_SQ = (2^256)^(-2) mod p` as a compile-time constant
  - `w' = w × R_INV_SQ mod p` (can use CPU Barrett or Montgomery for precomputation)
  - Store Plantard twiddles in separate device array (parallel to Montgomery/Barrett twiddles)
- CPU reference Plantard in `tests/ff_reference.h` for validation
- Implement Plantard-specific `ff_add_plantard` / `ff_sub_plantard` if Plantard representation
  requires different add/sub (check: Plantard output form may differ from standard form)
  - If Plantard outputs standard-form results, existing ff_add/ff_sub work unchanged
  - Key: document the representation invariant clearly

**Analysis:**
- SASS instruction count: Plantard should save ~64 MAD instructions per ff_mul
  (eliminating the Montgomery correction loop = 8 iterations × 8 limbs)
- Expected: ~460 SASS instructions vs Montgomery 528 (−13%) or Barrett 888 (−48%)
- Register pressure: similar to Montgomery (one less 8-limb accumulator)

**Test plan — Plantard arithmetic correctness:**
- **Edge-case inputs (same suite as Barrett Session 1):**
  - Identity: a * 1' = a (where 1' is Plantard form of 1)
  - Boundary: (p-1) * (p-1)', 0 * a, max-limb values
- **Cross-validation (Plantard vs Montgomery vs Barrett):**
  - 10^6 random pairs: Plantard result == Montgomery result == Barrett result
  - Includes conversion between representations if needed
- **Twiddle precomputation validation:**
  - For each twiddle w, verify: plantard_mul(a, w') == montgomery_mul(a, w) for random a
  - All twiddle table entries at n=2^22 (n/2 = 2^21 entries spot-checked)
- **Algebraic properties:**
  - Commutativity, associativity, distributivity (10^3 each)

**Deliverables:**
- `include/ff_plantard.cuh` — Plantard device functions
- Plantard twiddle precomputation (host-side, stored in device memory)
- CPU reference Plantard in `tests/ff_reference.h`
- Plantard constant: `R_INV_SQ mod p` verified
- Correctness tests: ~15 new tests
- SASS instruction count comparison

---

### Session 20 — ~~Plantard NTT Integration + Benchmark + Release v1.7.0~~ CANCELLED

**CANCELLED:** Session 19 demonstrated that Plantard arithmetic is 79% slower than Montgomery
for 256-bit BLS12-381 (944 SASS vs 528 SASS). Integrating a slower multiplication into the
NTT pipeline would degrade performance. No `NTTMode::PLANTARD` will be added.

---

## v1.8.0 — Stockham Outer Stages (Conditional)

**Goal:** Replace Cooley-Tukey outer stages with Stockham auto-sort NTT, eliminating
the bit-reversal permutation and achieving fully coalesced memory access patterns.

**CONDITIONAL**: Only proceed if Session 12's L2 diagnostic shows **L2 hit rate < 20%**
at n=2^22 (latency-bound outer stages). If L2 hit rate > 50% (bandwidth-bound), Stockham
won't help — skip this release and investigate alternative directions.

**Expected improvement:** 10–25% on outer stages if latency-bound (coalesced access
eliminates miss penalty). Risk: if bandwidth-bound, Stockham wastes 2× VRAM for no gain.

**Rationale:** NTTSuite (Ding et al., 2024) benchmarks 7 NTT variants on GPU, showing
Stockham trades higher absolute bandwidth for better coalescing efficiency, with 1.2–1.8×
speedup over Cooley-Tukey on DRAM-bound workloads. Our outer stages have a 12× gap to
roofline — Stockham's coalesced access could close much of that gap.

**References:**
- NTTSuite (Ding et al., 2024): https://arxiv.org/abs/2406.16972
- Stockham auto-sort FFT: Stockham (1966), Gentleman & Sande variant

---

### Session 21 — Stockham Kernel Design + Implementation

**Objective:** Design and implement Stockham outer-stage kernels with ping-pong buffers.

**Background — Stockham vs Cooley-Tukey:**
Cooley-Tukey (current): in-place, bit-reversal permutation, strided access in outer stages.
Stockham: out-of-place (ping-pong), no bit-reversal needed, all accesses coalesced.

Key difference for outer stages: in Cooley-Tukey, stage k has butterfly pairs at distance
2^k — at k=11, that's stride=2048, meaning consecutive warp threads access elements 2048
apart (65 KB stride × 32 threads = completely non-coalesced). In Stockham, threads always
read/write contiguous blocks regardless of stage.

**Tasks:**
- Allocate ping-pong buffer: `cudaMalloc` second n-element array for Stockham outer stages
  - At n=2^22: 2 × 128 MB = 256 MB (within 6 GB VRAM budget)
  - Allocate/free as part of NTT call (or cache like twiddle tables)
- Implement Stockham outer-stage kernel:
  - `ntt_outer_stockham_kernel(src, dst, twiddles, n, stage)`:
    reads from `src`, writes to `dst` (out-of-place)
  - Stockham indexing: thread `t` reads `src[t]` and `src[t + n/2]` (coalesced!)
    and writes to `dst[stockham_index(t, stage)]` (also coalesced by construction)
  - Apply to outer stages only (inner kernel stays in-place with shmem)
- Eliminate bit-reverse permutation:
  - Stockham auto-sort property means output is already in natural order after all stages
  - Save ~0.3 ms bit-reverse pass at n=2^22
- Implement radix-4 and radix-8 Stockham variants:
  - Stockham radix-8: reads 8 contiguous elements, writes 8 to reordered positions
  - Combine coalesced access with multi-stage fusion for maximum benefit
- Support both Montgomery and Barrett arithmetic paths
- Handle ping-pong: after each stage, swap src/dst pointers. After all outer stages,
  ensure result is in the original array (or the caller's expected location).
  If odd number of stages, need one extra copy.

**Deliverables:**
- Ping-pong buffer allocation + caching
- Stockham outer-stage cooperative kernels (Montgomery/Barrett × single/batch)
- Stockham radix-4 and radix-8 variants
- Bit-reverse elimination (save one kernel launch)

---

### Session 22 — Stockham Correctness + Edge Cases

**Objective:** Exhaustive correctness testing of Stockham path.

**Test plan:**
- **Forward NTT vs CPU reference:**
  - Stockham path: all sizes 2^10..2^22
- **Roundtrip INTT(NTT(x)) = x:**
  - Stockham: all sizes, both Montgomery and Barrett
- **Cross-validation (Stockham vs Cooley-Tukey):**
  - Bitwise identical output at all sizes 2^10..2^22
  - Both arithmetic modes
- **Ping-pong correctness:**
  - Even number of outer stages: result in original buffer
  - Odd number: result in ping-pong buffer (verify copy-back works)
  - Test at sizes producing both even and odd outer stage counts
- **Batched Stockham:**
  - B=4 and B=8 vs sequential at 2^15, 2^18, 2^22
  - Batched Stockham vs batched Cooley-Tukey (bitwise identical)
- **VRAM edge case:**
  - Large batch at large size: verify ping-pong buffer doesn't cause OOM
  - B=8 at 2^22 = 8 × 128 MB × 2 buffers = 2048 MB (should fit in 6 GB)
  - B=16 at 2^22 = 4096 MB → may OOM. Detect and handle gracefully.
- **Bit-reverse elimination:**
  - Verify Stockham output is in natural order (no bit-reverse needed)
  - Compare against Cooley-Tukey with explicit bit-reverse (should be bitwise identical)
- **Known-vector patterns:**
  - All-zeros, all-ones, single-nonzero, ascending at 3 sizes

**Deliverables:**
- Full Stockham correctness test suite (~40 new tests)
- Edge case handling (OOM detection, odd/even stage count)
- Target: ~410 total tests

---

### Session 23 — Stockham Benchmark + Release v1.8.0

**Objective:** Benchmark Stockham vs Cooley-Tukey, profile coalescing improvement, release.

**Tasks:**
- Benchmark matrix: {Cooley-Tukey, Stockham} × {radix-4, radix-8} × {Montgomery, Barrett}
  × {2^15..2^22} × {single, batch-8}
- Profile with ncu:
  - Key metrics: `dram__bytes_read.sum`, `dram__bytes_write.sum` (total traffic)
  - `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct` (cache efficiency)
  - `smsp__sass_average_data_bytes_per_sector_mem_global_op_st` (coalescing quality)
  - Compare Stockham vs Cooley-Tukey: coalescing should be ~32× better for outer stages
- Capture screenshots: Stockham coalescing quality, before/after DRAM traffic comparison
- Full regression sweep
- Update analysis.md, README, CLAUDE.md
- Tag v1.8.0

**Expected results (n=2^22, conditional on L2 diagnostic):**
- If latency-bound: Stockham ~10 ms (−20% vs v1.5.0's ~13 ms)
- If bandwidth-bound: Stockham ~12.5 ms (−5%, not worth the complexity)
- Document the result either way — negative results are valuable

**Deliverables:**
- Updated benchmark tables + charts
- ncu coalescing screenshots
- Full regression pass
- `results/data/bench_v180.json`
- Git tag v1.8.0

---

## Stretch Goals (v1.9.0+)

### SPIRAL/MoMA Code Generation Study
- Attempt to use SPIRAL (open-source) to generate BLS12-381 NTT kernels
- Compare auto-generated vs hand-tuned code quality (SASS instruction count, throughput)
- If SPIRAL-generated code is superior, adopt as primary implementation

### Multi-GPU NTT
- Split n-point NTT across 2+ GPUs using the 4-step decomposition
- Column NTTs on GPU 0, row NTTs on GPU 1 (or split evenly)
- Requires NVLink or PCIe peer-to-peer for transpose step

### Register-Optimized Inner Kernel (K=11 or K=12)
- With Goldilocks (8B elements) or BabyBear (4B), shared memory per element is 4-8×
  smaller → can potentially fuse more stages (K=12 for Goldilocks, K=13 for BabyBear)
- Would eliminate even more outer stages for smaller fields

### Tensor Core Exploration
- BabyBear's 31-bit elements can be packed into FP16/BF16 matrix tiles
- Tensor cores do 16×16 matrix multiply in 1 cycle — could potentially compute
  NTT butterflies as matrix operations
- Highly experimental; may not preserve exact integer semantics

---

## Summary: Session Plan

| Session | Release | Focus | Key Technique |
|---|---|---|---|
| 1 | v1.2.0 | Barrett reduction for BLS12-381 | MoMA-inspired arithmetic |
| 2 | v1.2.0 | Barrett NTT integration + benchmarking | Eliminate Montgomery overhead |
| 3 | v1.2.0 | Batched NTT kernel | MoMA batch processing pattern |
| 4 | v1.2.0 | Benchmark, profile, release | — |
| **5** | **v1.3.0** | **4-step NTT: transpose kernel + architecture** ✅ | **Bailey's algorithm** |
| **6** | **v1.3.0** | **4-step NTT: sub-NTT integration (+ batch)** ✅ | **Sub-NTTs fit in shmem** |
| **7** | **v1.3.0** | **4-step NTT: correctness + edge cases** ✅ | **Fallback + 221 tests** |
| **8** | **v1.3.0** | **Benchmark + release (negative result)** ✅ | **+18% slower than Barrett** |
| **9** | **v1.4.0** | **Branchless arithmetic** ✅ | **Eliminate warp divergence** |
| **10** | **v1.4.0** | **Radix-4 outer stages** ✅ | **~45% DRAM traffic reduction** |
| **11** | **v1.4.0** | **CUDA Graphs + release** ✅ | **Graph replay (negligible gain)** |
| **12** | **v1.5.0** | **L2 diagnostic + radix-8 butterfly** ✅ | **Stockham NO-GO, 4 radix-8 kernels** |
| **13** | **v1.5.0** | **Radix-8 benchmark + correctness** ✅ | **Montgomery-only radix-8 (15.6 ms), Barrett I-cache regression** |
| **14** | **v1.5.0** | **OTF twiddle computation** ✅ | **NEGATIVE RESULT: 56.9 ms vs 15.6 ms (+265%). Disabled.** |
| **15** | **v1.5.0** | **Benchmark + profile + release v1.5.0** ✅ | **15.5 ms at 2^22. 333 tests.** |
| **16** | **v1.6.0** | **Goldilocks + BabyBear field arithmetic** ✅ | **GPU+CPU ref, 374 tests. GL 3.6x, BB 6.8x faster mul.** |
| **17** | **v1.6.0** | **Multi-field NTT integration + correctness** ✅ | **Fused K=8-11 + radix-8/4/2 outer, batched. 458 tests.** |
| **18** | **v1.6.0** | **3-way benchmark + charts + release v1.6.0** ✅ | **BLS 15.1ms, GL 3.6ms (4.2x), BB 2.4ms (6.2x). 458 tests.** |
| **19** | **v1.7.0** | **Plantard arithmetic + twiddle precompute** ✅ | **NEGATIVE RESULT**: 944 SASS (+79% vs Montgomery 528). 471 tests. |
| **20** | **v1.7.0** | ~~Plantard NTT integration~~ | **CANCELLED** — Plantard arithmetic slower than Montgomery |
| 21 | v1.8.0 | Stockham kernel design + implementation | Coalesced outer-stage access |
| 22 | v1.8.0 | Stockham correctness + edge cases | Ping-pong buffer validation |
| 23 | v1.8.0 | Stockham benchmark + release v1.8.0 | Conditional on L2 diagnostic |

**Projected cumulative results (n=2^22, single NTT, 7-rep median):**
- v1.1.0: 25.1 ms (Montgomery, fused K=10 + cooperative outer)
- **v1.2.0: 24.9 ms (Barrett, -0.8%)** — minimal single-NTT gain; outer stages still dominate
- **v1.3.0: 29.5 ms (4-Step, +18% SLOWER)** — 3 transposes + sub-NTT outer stages add overhead.
  **Best mode remains Barrett at 24.9 ms.** 4-step is a negative result.
- **v1.4.0: 17.1 ms (Montgomery) / 17.4 ms (Barrett)** — branchless arithmetic (-4.4%) +
  radix-4 outer stages (-30%). CUDA Graphs add negligible improvement (within noise).
  **32% faster than v1.1.0, exceeds 18-22 ms target.**
- **v1.5.0: 15.5 ms (Montgomery radix-8, −9.4% vs v1.4.0)** — Barrett radix-8
  disabled (I-cache regression). OTF twiddles: **negative result** (56.9 ms, +265%), disabled.
  333 tests. Final target: **15.6 ms** (no OTF gain).
- v1.6.0: ~14 ms BLS / ~1.2 ms Goldilocks / ~0.4 ms BabyBear — **multi-field** (projected)
- v1.7.0: **NEGATIVE RESULT** — Plantard 944 SASS vs Montgomery 528 (+79%). No NTT integration.
- v1.8.0: **CANCELLED** (Stockham NO-GO — L2 58.5% = bandwidth-bound)

**Batch throughput (8× 2^22 NTTs):**
- v1.1.0: 8 × 25.1 ms = ~201 ms (sequential)
- **v1.2.0: 219 ms batched (~1.0x) / 216 ms sequential** — GPU saturated, no gain at 2^22.
  Batching helps at 2^15 (1.52x) where GPU is underutilized.
- **v1.3.0: 241 ms (4-step batched) vs 199 ms (Barrett batched)** — 4-step +21% slower.
  4-step benefits more from batching (1.16x vs 1.05x) but still slower in absolute terms.
  Best batch mode remains Barrett at 199 ms.
- **v1.4.0: ~150 ms (Montgomery batched) / ~159 ms (Barrett batched)** — radix-4 outer stages
  reduce batch time by ~20%. Montgomery faster for batched (fewer instructions per ff_mul).
- **v1.5.0: 139 ms (Montgomery radix-8 batched)** — Barrett batched: 167 ms (radix-4)

**Honest ceiling for BLS12-381 on RTX 3060 Laptop:**
~8–9 ms at n=2^22. The 4 MB L2 vs 128 MB array means outer stages can never achieve
full bandwidth utilization. The multi-field comparison (v1.6.0) will concretely demonstrate
this ceiling and why the ZKP ecosystem is moving to smaller fields.

---

## Direction Triage Notes

**Direction 3 from retrospective (Warp Shuffle Inner Kernel): ALREADY IMPLEMENTED.**
The fused inner kernel in `ntt_fused_kernels.cu` already uses `__shfl_xor_sync` for stages
0-4 via `butterfly_shfl()` + `fp_shfl_xor()`. Stages 5-9 require cross-warp communication
(stride ≥ 32) and must use shared memory. No further optimization possible here.

**OTF twiddles: NEGATIVE RESULT (Session 14).**
OTF was expected to save ~1 ms by eliminating twiddle DRAM traffic. In practice, the BLS12-381
256-bit exponentiation (`ff_pow_mont_u32`) costs ~35 Montgomery muls × 128 MADs = ~4480 MADs
per butterfly, far exceeding the cost of 7 DRAM reads (224 bytes). OTF is disabled for BLS12-381.
Infrastructure retained for v1.6.0 multi-field work (Goldilocks/BabyBear: 1-2 instructions per
multiply makes OTF viable).

**Why Goldilocks/BabyBear is v1.6.0 (not earlier):**
High impact/effort ratio but does not improve BLS12-381 performance. Placed after radix-8
(the primary performance improvement) but before Plantard/Stockham (diminishing returns).
Also provides a natural comparison point: "here's what our optimized BLS12-381 achieves
vs a field that's inherently 10-50× faster."

**Why Stockham is conditional:**
Session 12's L2 diagnostic determines whether outer stages are latency-bound (Stockham
helps) or bandwidth-bound (Stockham doesn't help). This is the most expensive optimization
remaining (~2 weeks) and carries medium-high risk. The diagnostic costs 1 hour of profiling
and saves potentially 2 weeks of wasted effort.

---

## v2.0.0 — Groth16 GPU Primitives Library (COMPLETE)

**Goal:** Add all remaining GPU-accelerated building blocks for Groth16 proving,
culminating in a toy end-to-end proof generation pipeline.

**Sessions 20-25 (6 sessions), completed 2026-03-12.**

### New Primitives

| Session | Component | Files | Tests |
|---------|-----------|-------|-------|
| 20 | Fq (381-bit) + Fq2 field arithmetic | `ff_fq.cuh`, `ff_fq2.cuh`, `ff_fq_kernels.cu` | 85 |
| 21 | G1 + G2 elliptic curve ops (Jacobian) | `ec_g1.cuh`, `ec_g2.cuh`, `ec_kernels.cu` | 38 |
| 22 | MSM (Pippenger's bucket method, G1) | `msm.cuh`, `msm.cu` | 21 |
| 23 | Polynomial ops (coset NTT, pointwise) | `poly_ops.cuh`, `poly_ops.cu` | 15 |
| 24 | Groth16 pipeline (toy circuit) | `groth16.cuh`, `groth16.cu` | 29 |
| 25 | Benchmark + profiling + release | `bench_msm.cu`, `bench_groth16.cu` | — |
| **Total** | | **13 new files** | **150 new (621 total)** |

### Toy Circuit: x^3 + x + 5 = y

- 6 variables: [1, x, y, v1=x*x, v2=v1*x, v3=v2+x]
- 4 R1CS constraints, padded to domain_size=256 for NTT
- GPU pipeline: R1CS×witness → INTT → coset NTT → pointwise (A*B-C)/(g^n-1) → coset INTT → EC assembly
- Proof: π_A ∈ G1, π_B ∈ G2, π_C ∈ G1
- GPU proof matches CPU proof bitwise (cross-validated)

### Benchmark Results (RTX 3060 Laptop, 7-rep median)

**MSM (Pippenger):**

| Size | Latency (ms) | Throughput (pts/ms) |
|------|-------------|---------------------|
| 2^10 | 261 | 3.9 |
| 2^14 | 2,697 | 6.1 |
| 2^18 | 42,603 | 6.2 |

Implementation is correctness-focused (single-thread bucket reduction + window combination).
Production MSM (ICICLE, bellman-cuda) would be orders of magnitude faster.

**Groth16 Pipeline (n=256, x=3):**
- Trusted setup: ~5.4 seconds (CPU EC scalar multiplications)
- Proof generation: ~5.2 seconds (dominated by CPU EC assembly, not GPU compute)
- GPU/CPU ratio ≈ 1.0 (toy circuit too small for GPU advantage — GPU NTT/coset ops < 1 ms)

### Key Technical Decisions

- **CUB + RDC workaround**: MSM compiled as separate `msm_lib` with `CUDA_SEPARABLE_COMPILATION OFF`
  (same pattern as ntt_fused_lib) to avoid CUB `__fatbinwrap` unresolved externals
- **CPU-side Montgomery**: Host functions in .cu files use `ff_reference.h` FpRef for field conversions
  (avoids `__device__`-only ff_to_montgomery on host)
- **Coset generator g=7**: Standard choice for BLS12-381 Fr
- **Simplified π_C**: Omits r*B_g1 term (both GPU and CPU use same formula for cross-validation)
- **No pairing verification**: Documented as future work

---

## v2.1.0 — Production MSM (COMPLETE)

**Goal:** Replace correctness-focused single-thread MSM with production-quality parallel Pippenger.
Current: 261ms at 2^10, 42.6s at 2^18. Gap vs SOTA: ~20-85x in throughput.
Target: >20x speedup (n=2^18 from 42.7s → <2s). **Achieved: 35.8x (1.2s).**

**Session 26:** Signed-digit recoding + CUB sort + segment offsets. 5.4x (42.7s → 8.0s). 641 tests.
**Session 27:** Parallel bucket reduction (Hillis-Steele suffix scan). 36.2x vs v2.0.0. 655 tests.
**Session 28:** Window auto-tuner + memory pools + benchmark + release. 701 tests.

**Primary references:**
- **cuZK** (Lu et al., TCHES 2023) — SpMV formulation: radix sort → sparse coordinate → parallel accumulation
- **Load-Balanced MSM** (Chen et al., TCHES 2024) — halves bucket count, homogeneous-coordinate
  accumulation, load-balanced parallel reduction. Nearly-linear speedup on RTX 4090/V100.
- **DistMSM** (Ji et al., ASPLOS 2024) — register pressure optimization via "operand fusion"

**Key optimizations:** cuZK-style SpMV bucket accumulation, signed-digit window recoding
(halves bucket count), TCHES 2024 load-balanced parallel reduction, mixed affine/Jacobian
arithmetic, optimal window auto-tuning (c ≈ log2(n)).

### Session 26 — Signed-Digit Recoding + Segment-Offset Accumulation ✅ COMPLETE

**Objective:** Replace naive bucket accumulation with signed-digit recoding and segment-offset lookup.

**Implemented:**
1. Signed-digit scalar recoding kernel:
   - Extract signed digits with carry propagation via per-scalar `uint8_t` carry buffer
   - Halves bucket count from `2^c` to `2^(c-1)+1`; negative digits negate the point
   - Final carry overflow window handles carry out of last regular window
   - Output: `d_bucket_ids[i] = |digit|`, `d_packed_values[i] = point_idx | (sign << 31)`
2. Revised window size selection:
   - `c = floor(log2(n)/2) + 1`, clamped [4, 16]
   - For n=2^18: c=10, W=26 windows, 513 buckets per window
3. Segment-offset accumulation:
   - After CUB sort: `atomicMin` to find segment starts, backward-fill for empty buckets
   - Replaces O(log n) binary search with O(1) offset lookup per bucket
   - Point negation fused into accumulation (read sign bit from packed value)

**Measured results (RTX 3060 Laptop, 7-rep median):**

| Size | v2.0.0 (ms) | v2.1.0-s26 (ms) | Speedup |
|------|-------------|-----------------|---------|
| 2^10 | 261 | 202 | 1.3x |
| 2^12 | 755 | 417 | 1.8x |
| 2^14 | 2,704 | 876 | 3.1x |
| 2^15 | 5,317 | 1,399 | 3.8x |
| 2^16 | 10,633 | 2,890 | 3.7x |
| 2^18 | 42,714 | 7,982 | **5.4x** |

**Analysis:**
- Signed-digit recoding halves bucket count → halves bucket reduction work.
- Segment offsets eliminate binary search → cleaner memory access pattern.
- Improved window sizing (c=10 at 2^18 vs c=4) gives many fewer windows (26 vs 64)
  with more effective bucket distribution.
- **Remaining bottleneck**: bucket reduction and window combination still single-thread.
  At c=10 with 513 buckets, the running sum does 512 EC additions sequentially.
  Parallelizing this is Session 27's target.

**Tests (20 new, 641 total):**
- Window size function: c=6/8/10/11 for n=2^10/14/18/20
- Cross-validation at n=128, 256 vs CPU naive MSM
- On-curve checks at n=512, 1024, 2048, 4096
- Multi-limb scalars, high-bit scalars (r-1), all-zeros, single-bucket, power-of-2
- Mixed zero/nonzero scalars, determinism (3 runs)
- Benchmark data: `results/data/bench_msm_v210_s26.json`

### Session 27 — Parallel Bucket Reduction ✅ COMPLETE

**Objective:** Replace single-thread bucket reduction with parallel tree reduction.

**Implemented:**
1. Parallel bucket reduction kernel (`bucket_reduce_parallel_kernel`):
   - Hillis-Steele suffix inclusive scan on d_buckets[1..B-1] (O(log B) depth)
   - Tree reduction of suffix sums → window result
   - In-place global memory with `__syncthreads()` barriers (fits in L2 cache)
   - Handles up to 1024 threads (B ≤ 1025); falls back to sequential for larger
2. Window combination: Horner's method kept sequential (small W ≈ 26, ~297 ops)

**Measured results (RTX 3060 Laptop, 7-rep median):**

| Size | S26 (ms) | S27 (ms) | S26→S27 | vs v2.0.0 |
|------|----------|----------|---------|-----------|
| 2^10 | 202 | 121 | 1.67x | 2.2x |
| 2^12 | 417 | 256 | 1.63x | 2.9x |
| 2^14 | 876 | 557 | 1.57x | 4.9x |
| 2^15 | 1,399 | 1,066 | 1.31x | 5.0x |
| 2^16 | 2,890 | 2,318 | 1.25x | 4.6x |
| 2^18 | 7,982 | 1,180 | **6.8x** | **36.2x** |

**Analysis:**
- At n=2^18 (c=10, 512 active buckets): scan depth = 9 rounds vs 1024 sequential g1_adds.
  Bucket reduction was the dominant serial bottleneck — parallelizing it gave 6.8x.
- Larger n benefits more: more buckets → more parallelism in suffix scan.
- Points/ms at 2^18: 222 (up from ~33 in S26).

**Tests (14 new, 655 total):**
- Cross-validation at n=3, 4, 64, 128, 256, 512 vs CPU naive
- On-curve checks at n=1024, 8192, 16384
- Determinism, uniform scalar, half-zero, ascending, alternating bit patterns, high-bit scalars
- Benchmark data: `results/data/bench_msm_v210_s27.json`

### Session 28 — Window Auto-Tuning + Memory Pools + Release v2.1.0 ✅ COMPLETE

**Objective:** Auto-tune window sizes, optimize memory management, produce benchmark numbers.

**Deliverables:**
1. Window auto-tuner: c = floor(log2(n)/2) + 1, clamped [4, 11].
   Upper cap ensures parallel bucket reduction (2^(c-1) ≤ 1024 threads).
2. Stream-ordered memory pools: replaced cudaMalloc/cudaFree with cudaMallocAsync/cudaFreeAsync.
   CUDA driver caches allocations for reuse across repeated MSM calls.
3. Benchmark extended to n=2^20 (1M points): 4.2s, 247 pts/ms.

**Performance (v2.1.0 final, 7-rep median, RTX 3060 Laptop):**

| Size | v2.0.0 (ms) | v2.1.0 (ms) | Speedup | Points/ms |
|------|-------------|-------------|---------|-----------|
| 2^10 | 261 | 123 | 2.1x | 8 |
| 2^12 | 755 | 259 | 2.9x | 16 |
| 2^14 | 2,704 | 565 | 4.8x | 29 |
| 2^15 | 5,317 | 1,082 | 4.9x | 30 |
| 2^16 | 10,633 | 2,347 | 4.5x | 28 |
| 2^18 | 42,714 | 1,194 | **35.8x** | 220 |
| 2^20 | — | 4,249 | — | 247 |

**Performance targets vs achieved:**
- n=2^10: target <5ms → achieved 123ms (missed — sequential window overhead dominates at small n)
- n=2^14: target <50ms → achieved 565ms (missed — same reason)
- n=2^18: target <500ms-2s → achieved **1.2s** (**target met**, 35.8x speedup)

**Tests (46 new, 701 total):**
- Window cap verification: c ≤ 11 for all sizes up to 2^26
- Window boundary at n=2^20 (c=11, exactly 1024 active buckets)
- Small n edge cases (n=0..1024)
- Memory pool reuse (3 consecutive MSM calls)
- Non-default stream, varying sizes, on-curve at n=32768
- Benchmark data: `results/data/bench_msm_v210_s28.json`

---

## v2.2.0 — Fibonacci Circuit + Batch Pipeline (PLANNED)

**Goal:** Demonstrate GPU advantage at real scale. v2.0.0 toy circuit (4 constraints, n=256)
shows GPU/CPU ratio = 1.0. Need a meaningful circuit to prove GPU investment pays off.

**Narrative**: This is the "GPU finally wins" release — showing ~12x GPU/CPU at n=2^18.

**Primary reference:**
- **BatchZK** (Lu et al., ASPLOS 2025, eprint:2024/1862) — 259.5x throughput via pipelined
  batch proof generation. Overlaps NTT/MSM/assembly across proofs in multiple CUDA streams.

**Prerequisite:** v2.1.0 production MSM (GPU advantage requires fast MSM at n=2^18).

### Session 29 — Fibonacci 2^18 R1CS Circuit

**Objective:** Build a real-scale circuit that exercises all GPU primitives at proof-relevant sizes.

**Deliverables:**
1. Fibonacci R1CS circuit generator:
   - 2^18 constraints: each `a_{i+2} = a_i + a_{i+1}` (one multiplication constraint per step)
   - Witness: `[1, a_0, a_1, a_2, ..., a_{2^18}]`
   - Domain size = 2^18 (next power of 2 above constraint count)
2. Extended trusted setup for domain_size=2^18:
   - Reuse existing `groth16_setup()` infrastructure, just larger domain
   - Proving key: G1/G2 points for 2^18+ variables
3. Full GPU prove with existing Groth16 pipeline at n=2^18
4. CPU prove at n=2^18 for comparison
5. **Target chart**: GPU ~10s vs CPU ~120s = **~12x GPU advantage**

**Tests (~15 new, cumulative ~681):**
- Fibonacci witness: verify a_{i+2} = a_i + a_i+1 for all steps
- R1CS satisfaction: A*w . B*w = C*w for all constraints
- GPU proof at n=2^18: verify all proof elements on correct curves
- GPU vs CPU proof: bitwise match (or verify both independently)
- Edge cases: Fibonacci from (0,1), (1,1), large starting values

### Session 30 — 2-Stream Batch Pipeline + Release v2.2.0

**Objective:** Pipeline multiple proof generations for higher throughput (BatchZK pattern).

**Deliverables:**
1. Multi-stream pipeline architecture:
   - Proof k's MSM overlaps Proof k+1's NTT (2 CUDA streams)
   - Reuse existing batched NTT API (v1.2.0) as foundation
   - Phase overlap: `[NTT_1][MSM_1+NTT_2][MSM_2+NTT_3]...`
2. Prove 4 Fibonacci instances simultaneously with stream overlap
3. Measure throughput (proofs/second) vs sequential single-stream
4. **Target**: ~2x throughput for batch of 4+ proofs
5. Benchmark + analysis + release

**Tests (~10 new, cumulative ~691):**
- Batch of 4 proofs: all verify correctly
- Pipeline vs sequential: bitwise identical proofs
- Throughput measurement: proofs/second at batch sizes 1, 2, 4, 8
- Memory usage: fits in 6 GB VRAM at batch_size=4

---

## v3.0.0 — Pairing Verification (PLANNED)

**Goal:** Implement BLS12-381 optimal Ate pairing for Groth16 proof verification.
Complete the prove→verify loop. Mathematical capstone of the project.

**Field tower:** Fq → Fq2 (done) → Fq6 (cubic over Fq2) → Fq12 (quadratic over Fq6)
**BLS12-381 parameter:** u = -0xd201000000010000 (64-bit, Hamming weight 5)

### Session 31 — Fq6 Arithmetic

**Objective:** Implement Fq6 = Fq2[v] / (v^3 − β) where β = (1+u) is the Fq2 nonresidue.

**Deliverables:**
1. `include/ff_fq6.cuh` — GPU Fq6 arithmetic:
   - `Fq6Element = {Fq2Element c0, c1, c2}` (element = c0 + c1·v + c2·v²)
   - `fq6_add/sub/neg` (component-wise, 3 Fq2 ops each)
   - `fq6_mul` (Karatsuba-like cubic: 6 Fq2 muls = 18 Fq muls)
   - `fq6_sqr` (specialized, cheaper than mul)
   - `fq6_inv` (via norm to Fq2)
   - `fq6_mul_by_nonresidue` (multiply by v: shift + `fq2_mul_by_nonresidue`)
   - `fq6_mul_by_01`, `fq6_mul_by_1` (sparse mul for Miller loop)
   - `fq6_frobenius_map` (precomputed Frobenius coefficients)
2. CPU reference `Fq6Ref` in `tests/ff_reference.h`
3. GPU test kernels in `src/ff_fq_kernels.cu`

**Tests (~25 new, cumulative ~716):**
- CPU self-test: add/sub/mul/sqr/inv round-trip, algebraic identities
- GPU vs CPU: add, sub, mul, sqr at N=1024
- Inverse: a · a^{-1} = 1 for random elements
- Distributivity: (a+b)·c = a·c + b·c
- Frobenius: f^q identity
- Sparse mul: `mul_by_01` matches general mul with zeroed coefficients

### Session 32 — Fq12 Arithmetic

**Objective:** Implement Fq12 = Fq6[w] / (w² − v), completing the tower.

**Deliverables:**
1. `include/ff_fq12.cuh` — GPU Fq12 arithmetic:
   - `Fq12Element = {Fq6Element c0, c1}` (element = c0 + c1·w)
   - `fq12_add/sub/neg/mul/sqr/inv/conjugate`
   - `fq12_mul`: Karatsuba (3 Fq6 muls = 54 Fq muls)
   - `fq12_frobenius_map` (precomputed coefficients from Sage/Python)
   - `fq12_mul_by_034` (sparse mul for Miller loop line functions)
2. CPU reference `Fq12Ref` in `tests/ff_reference.h`
3. Frobenius coefficients: roots of unity `(1+u)^((q^k-1)/6)` hardcoded as constants

**Note:** Fq12Element = 576 bytes. GPU pairing will spill to local memory (~216 registers
for f + T alone). Acceptable for correctness; performance optimization is future work.

**Tests (~20 new, cumulative ~736):**
- CPU self-test: add/sub/mul/sqr/inv
- GPU vs CPU: add, sub, mul, sqr
- Inverse: a · a^{-1} = 1
- Conjugation: for unitary elements, conjugate = inverse
- Frobenius: f^{q^k} identity for k=1,2,3,6,12
- Sparse mul: `mul_by_034` matches general mul

### Session 33 — Miller Loop

**Objective:** Implement the optimal Ate Miller loop for BLS12-381.

**Deliverables:**
1. `include/pairing.cuh` — line evaluation + Miller loop:
   - `line_double_step(T, Q, P)` — tangent line at T∈G2, evaluated at P∈G1
   - `line_add_step(T, Q, P)` — chord line through T,Q∈G2, evaluated at P∈G1
   - Lines produce sparse Fq12 elements (3 non-zero coefficients)
2. Miller loop implementation:
   - Iterate bits of |u| (MSB→LSB): f = f² · line_double; if bit=1: f *= line_add
   - u negative → conjugate(f), negate T
   - Use `fq12_mul_by_034` for line accumulation (not general Fq12 mul)
   - |u| has 5 set bits → only 4 addition steps in 63 iterations
3. `src/pairing.cu` — GPU kernel (single-thread for correctness)
4. CPU reference Miller loop in `tests/ff_reference.h`

**Tests (~15 new, cumulative ~751):**
- Line evaluation: doubling/addition lines match CPU reference
- Trivial inputs: e(O, Q) = 1, e(P, O) = 1
- e(G1, G2): compute reference, verify match
- Bilinearity: e(aP, Q) = e(P, aQ) = e(P,Q)^a for a=2,3
- Linearity: e(P, Q+R) = e(P,Q) · e(P,R)
- GPU vs CPU bitwise match

### Session 34 — Final Exponentiation

**Objective:** Implement f^((q^12 − 1) / r), completing the pairing.

**Deliverables:**
1. Easy part: f^((q^6 − 1)(q^2 + 1))
   - `f1 = conjugate(f) · inv(f)` (= f^(q^6−1))
   - `f2 = frobenius_map_2(f1) · f1` (= f1^(q^2+1))
2. Hard part: f2^((q^4 − q^2 + 1) / r)
   - Decompose using curve parameter u (Devegili/Scott method)
   - ~10 Fq12 sqr + ~10 Fq12 mul + ~4 Frobenius + ~3 exp-by-|u|
   - Each exp-by-|u|: ~63 squarings + 4 multiplications in Fq12
3. Combined `pairing(P, Q)` = final_exp(miller_loop(P, Q))
4. CPU reference final exponentiation

**Tests (~15 new, cumulative ~766):**
- Easy part: f^(q^6−1) is unitary (norm = 1)
- exp_by_u: matches direct computation
- Full pairing: e(G1, G2) matches known test vector
- Bilinearity: e(aP, bQ) = e(P,Q)^(ab) for a=2, b=3
- Non-degeneracy: e(G1, G2) ≠ 1
- GPU vs CPU bitwise match

### Session 35 — Groth16 Verification + Release v3.0.0

**Objective:** Implement Groth16 verification equation, complete prove→verify loop.

**Deliverables:**
1. `VerifyingKey` struct: precomputed e(α,β), [γ]_2, [δ]_2, IC points
2. `groth16_verify(vk, proof, public_inputs)` → bool
   - Verification: e(π_A, π_B) = e(α,β) · e(L_pub, γ) · e(π_C, δ)
   - Multi-pairing: product of Miller loops before single final exponentiation
3. Fix simplified π_C (add r·B_g1 term) for correct verification
4. End-to-end: prove → verify for toy circuit
5. Benchmark + analysis + release

**Tests (~20 new, cumulative ~786):**
- VK generation: all elements on correct curves
- Valid proof: verify returns true
- Corrupted π_A/π_B/π_C: verify returns false
- Wrong public input: verify returns false
- Multiple witnesses: x=3, 5, 10, 100
- GPU vs CPU verify match
- End-to-end roundtrip: setup → prove → verify

---

### Session Summary (v2.1.0 + v2.2.0 + v3.0.0)

| Session | Release | Objective | New Tests | Cumulative |
|---------|---------|-----------|-----------|------------|
| 26 | v2.1.0 | Signed-digit recoding + segment-offset accumulation ✅ | 20 | 641 |
| 27 | v2.1.0 | Parallel bucket reduction (Hillis-Steele suffix scan) ✅ | 14 | 655 |
| 28 | v2.1.0 | Window auto-tuning + benchmark + release | ~10 | ~666 |
| 29 | v2.2.0 | Fibonacci 2^18 R1CS circuit | ~15 | ~681 |
| 30 | v2.2.0 | 2-stream batch pipeline + release | ~10 | ~691 |
| 31 | v3.0.0 | Fq6 arithmetic | ~25 | ~716 |
| 32 | v3.0.0 | Fq12 arithmetic | ~20 | ~736 |
| 33 | v3.0.0 | Miller loop | ~15 | ~751 |
| 34 | v3.0.0 | Final exponentiation | ~15 | ~766 |
| 35 | v3.0.0 | Groth16 verification + release | ~20 | ~786 |

**Dependencies:**
- Sessions 26→27→28 (linear, v2.1.0 MSM)
- Sessions 29→30 (linear, v2.2.0 circuit + pipeline)
- Sessions 31→32→33→34→35 (linear, v3.0.0 pairing)
- v2.1.0 must complete before v2.2.0 (MSM perf needed for GPU advantage demo)
- v3.0.0 pairing track is independent of v2.2.0

**Narrative arc:**
> v2.0.0: "MSM is now the bottleneck (GPU=CPU at n=256, 42s at n=2^18)"
> v2.1.0: "cuZK-style parallel MSM closes the gap (>20x speedup)"
> v2.2.0: "Fibonacci 2^18 shows GPU = 12x CPU — GPU finally wins"
> v3.0.0: "Full prove→verify loop with pairing verification"

---

## References

- **MoMA**: Zhang & Franchetti, "Code Generation for Cryptographic Kernels using Multi-word
  Modular Arithmetic on GPU", CGO 2025. [arXiv:2501.07535](https://arxiv.org/html/2501.07535),
  [SPIRAL pub](https://spiral.ece.cmu.edu/pub-spiral/abstract.jsp?id=376)
- **ZKProphet**: Verma et al., IEEE IISWC 2025 — §V-B: open optimization targets
- **cuZK**: Lu et al., TCHES 2023 — parallel MSM via SpMV + radix sort.
  [eprint:2022/1321](https://eprint.iacr.org/2022/1321),
  [github](https://github.com/speakspeak/cuZK)
- **Load-Balanced MSM**: Chen, Peng, Dai et al., TCHES 2024 — improved cuZK: halved bucket
  count, load-balanced parallel reduction.
  [TCHES article](https://tches.iacr.org/index.php/TCHES/article/view/11438)
- **BatchZK**: Lu, Chen, Wang et al., ASPLOS 2025 — 259.5x batch proof throughput via
  pipelined multi-stream architecture. [eprint:2024/1862](https://eprint.iacr.org/2024/1862)
- **DistMSM**: Ji, Zhang et al., ASPLOS 2024 — multi-GPU MSM, tensor core bigint,
  register pressure optimization via operand fusion
- **NTT Multi-GPU**: Ji, Zhao et al., ASPLOS 2025 — warp specialization, persistent kernels.
  [doi:10.1145/3669940.3707241](https://dl.acm.org/doi/10.1145/3669940.3707241)
- **zkSpeed/HyperPlonk**: Daftardar, Bunz et al., ISCA 2025 — ASIC 801x, SumCheck replaces
  NTT. [arXiv:2504.06211](https://arxiv.org/abs/2504.06211)
- **GZKP**: Ma et al., ASPLOS 2023 — end-to-end GPU ZKP baseline
- **Bailey**: D.H. Bailey (1990) — "FFTs in External or Hierarchical Memory" (4-step FFT)
- **ICICLE**: Ingonyama, GPU acceleration library for ZKP — [github](https://github.com/ingonyama-zk/icicle)
- **Özcan, Javeed, Savaş**: "High-Performance NTT on GPU Through radix2-CT and 4-Step
  Algorithms", IEEE Access 2025. [doi:10.1109/ACCESS.2025.11003946](https://ieeexplore.ieee.org/document/11003946)
- **Kim et al.**: "Accelerating NTT for Bootstrappable HE on GPUs", 2020.
  [arXiv:2012.01968](https://arxiv.org/abs/2012.01968) — OTF twiddle factors
- **NTTSuite (Ding et al.)**: "NTTSuite: Number Theoretic Transform Benchmarks for
  Accelerating Encrypted Computation", 2024. [arXiv:2406.16972](https://arxiv.org/abs/2406.16972)
  — Stockham GPU benchmarks
- **Plantard**: "Efficient Word Size Modular Arithmetic", IEEE TETC 2021.
  [PDF](https://thomas-plantard.github.io/pdf/Plantard21.pdf)
- NVIDIA CUDA Programming Guide — §3.2.11 CUDA Graphs, §5.2 Memory Hierarchy
- Nsight Compute Documentation — L2 cache metrics, roofline analysis
