# NTT Optimization Roadmap — Future Releases

## Current State (v1.4.0)

**Benchmark (RTX 3060 Laptop, n=2^22):** 17.1 ms Montgomery / 17.4 ms Barrett (single NTT, compute only)
**Previous target:** ≤20 ms — **exceeded** (17.1 ms = −32% vs v1.1.0's 25.2 ms)

### Where Time Goes (n=2^22, v1.4.0 — 3 kernel launches)

| Phase | Duration | % of Total | Bottleneck |
|---|---|---|---|
| Bit-reverse permutation | ~0.3 ms | 2% | Memory (scatter) |
| Fused K=10 (stages 0-9) | ~6 ms | 35% | **Compute** (69%, IPC 2.41) |
| 6 radix-4 outer passes (stages 10-21) | ~11 ms | 65% | **Memory** (DRAM, ~300 GB/s) |

**Roofline gap**: Outer stages at ~11 ms vs theoretical floor of ~0.89 ms (320 MB / 360 GB/s
peak). **Gap to roofline: ~12×.** Suggests memory-latency-bound (L1/L2 miss penalty), not
bandwidth-bound. L2 diagnostic planned for Session 12 to confirm.

**Key lesson from v1.1–v1.4**: Every win came from reducing DRAM traffic in outer stages.
Arithmetic optimizations (Barrett, branchless) gave marginal results. The path forward is
higher-radix fusion (radix-8) and eliminating twiddle DRAM traffic (OTF computation).

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

**Expected improvement:** 15–25% additional over v1.4.0. Target: ~13 ms at n=2²².

**Rationale:** v1.4.0's radix-4 gave −30% by halving outer-stage DRAM passes (12→6).
Radix-8 fuses 3 stages per pass: 12→4 passes (another −33% from radix-4 baseline).
Each eliminated pass saves ~128 MB DRAM R+W = ~0.7 ms at bandwidth ceiling.
OTF twiddle computation eliminates an additional 384 MB twiddle-read traffic across
6 outer passes, worth ~1 ms. The doc recommends deploying OTF as part of a combined
radix-8 + OTF outer-stage rewrite (standalone OTF is too small to measure above noise).

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

### Session 13 — Radix-8 Correctness Tests + Benchmark

**Objective:** Validate radix-8 correctness at all sizes with dedicated tests, and benchmark
radix-8 vs radix-4 to measure actual performance impact. Kernels and dispatch logic are
already implemented (completed in Session 12 ahead of schedule).

**NOTE:** Session 12 ncu profiling revealed 50% uncoalesced access and 16.7% occupancy for
the Barrett radix-8 kernel. If benchmark shows radix-8 is slower than radix-4 due to
register pressure (174 regs Barrett → 1 block/SM vs 2 for radix-4), consider:
- `--maxrregcount=128` to force register spilling and recover 2 blocks/SM occupancy
- Montgomery-only radix-8 (134 regs) with Barrett falling back to radix-4
- Re-profiling with ncu after any changes to verify impact

**Tasks:**

**Part A — Correctness Tests (add to `tests/test_correctness.cu`):**
- Forward NTT vs CPU reference at all sizes 2^10..2^22 (radix-8 active where ≥3 outer)
- Roundtrip INTT(NTT(x)) = x at all sizes, both Montgomery and Barrett
- Batched radix-8: B=8 vs sequential at 2^15, 2^18, 2^20, 2^22
- Leftover handling:
  - num_outer % 3 == 0 (e.g., 2^22: 12 outer = 4×3)
  - num_outer % 3 == 1 (e.g., 2^14: 4 outer = 1×3 + 1)
  - num_outer % 3 == 2 (e.g., 2^15: 5 outer = 1×3 + 2)
- Target: ≥30 new tests (total ~260)

**Part B — Benchmark:**
- Measure radix-8 vs radix-4 at 2^15, 2^18, 2^20, 2^22 (single + batched)
- Both Montgomery and Barrett paths
- Key question: does radix-8's 1 block/SM (vs radix-4's 2 blocks/SM) hurt or help?
  - If net slower: try `--maxrregcount=128` or Montgomery-only radix-8
- Update benchmark data files in `results/data/`

**Part C — Re-profiling (if needed):**
- If `--maxrregcount` or other register pressure mitigations are applied, re-profile
  with ncu to verify occupancy improvement and measure coalescing impact
- Use existing `profiling/scripts/l2_diagnostic.ps1` to compare before/after

**Deliverables:**
- ≥30 new correctness tests for radix-8 paths
- Benchmark comparison: radix-8 vs radix-4 at key sizes
- Decision: keep radix-8 for Barrett, Montgomery, or both
- Updated `results/analysis.md` with benchmark results

---

### Session 14 — On-the-Fly Twiddle Computation

**Objective:** Replace precomputed twiddle table loads with on-the-fly computation in the
outer-stage radix-8 kernel, eliminating ~384 MB of twiddle DRAM reads.

**Background:**
Current outer stages load twiddles from a 64 MB precomputed table (n/2 × 32 bytes at n=2^22).
Access is strided (stage k has stride n/2^(k+1)), defeating L2 cache. OTF replaces this with
per-thread computation: each thread computes its required twiddle ω^j from a single per-stage
root ω_stage using a square-and-multiply chain.

**Tasks:**
- Implement OTF twiddle device function:
  ```cuda
  __device__ __forceinline__
  FpElement compute_twiddle_otf(
      const FpElement& omega_stage,  // ω^(n / 2^(s+1)) = primitive root for stage s
      uint32_t j                     // butterfly index within group → twiddle = omega_stage^j
  ) {
      // Square-and-multiply: compute omega_stage^j
      // For j up to 2^21, this is at most 21 ff_mul operations
      FpElement result = FP_ONE;
      FpElement base = omega_stage;
      while (j > 0) {
          if (j & 1) result = ff_mul(result, base);
          base = ff_mul(base, base);
          j >>= 1;
      }
      return result;
  }
  ```
- **Optimization**: For radix-8, each butterfly needs 7 twiddles. Many share the same
  `omega_stage` base. Compute twiddles incrementally where possible:
  - Stage s twiddles: w, w², w³ → compute w then multiply sequentially
  - Stage s+1 and s+2 twiddles share structure with stage s
  - Target: reduce from 7×21 ff_mul worst case to ~3×21 + 4×1 = ~67 ff_mul per butterfly
- **Precompute stage roots**: Instead of the full n/2 twiddle table, store only
  `log_n` values: `omega_roots[s] = ω^(n / 2^(s+1))` for s = 0..log_n-1.
  Total: 22 × 32 bytes = 704 bytes (fits in constant memory `__constant__`)
- **Apply to outer stages only**: The fused inner kernel (stages 0-9) is compute-bound;
  adding OTF ff_mul would make it slower. Keep precomputed twiddles for inner kernel.
- Implement OTF variant of radix-8 kernel: `ntt_outer_radix8_otf_kernel`
- **Trade-off analysis**: OTF adds ~7 ff_mul per radix-8 butterfly but eliminates 7 global
  memory twiddle loads. For the memory-bound outer stages, this is a net win. Measure:
  - With OTF: extra compute time from ff_mul
  - Without OTF: twiddle DRAM traffic eliminated
  - Net: should be positive for outer stages (memory-bound → compute is "free")
- Test: OTF kernel output must be **bitwise identical** to precomputed-twiddle kernel
  (same twiddle values, just computed differently)

**Test plan — OTF twiddle correctness:**
- **OTF twiddle values vs precomputed table:**
  - For each stage s and 1000 random j values, verify compute_twiddle_otf(ω_stage, j)
    equals twiddles[j * stride] from the precomputed table
  - Edge values: j=0 (should be 1), j=1 (should be ω_stage), j=max
- **NTT with OTF vs precomputed twiddles:**
  - All sizes 2^13..2^22: bitwise identical output
  - Both Montgomery and Barrett
  - Batched B=4 at 2^18, 2^22
- **Roundtrip with OTF twiddles:**
  - INTT(NTT(x)) = x for 2^15, 2^18, 2^20, 2^22
- **Stage root correctness:**
  - Verify omega_roots[s]^(2^s) = ω_n for all s (each root is consistent with ω_n)
  - Verify omega_roots[s]^(n/2^(s+1)) = 1 (correct order)

**Deliverables:**
- OTF twiddle device function in `include/ff_arithmetic.cuh` (or new `include/twiddle_otf.cuh`)
- Stage root precomputation + `__constant__` storage
- OTF variant of radix-8 outer kernel
- OTF correctness tests: ~15 new tests
- Trade-off analysis: OTF compute cost vs DRAM savings

---

### Session 15 — v1.5.0 Benchmark + Profile + Release

**Objective:** Comprehensive benchmarking of radix-8 + OTF, profiling, release.

**Tasks:**
- Full benchmark matrix: {Montgomery, Barrett} × {radix-4, radix-8, radix-8+OTF} × {2^15..2^22}
  - Single NTT forward, 7-rep median
  - Batched 8×, 7-rep median
- Profile with ncu:
  - Radix-8 kernel: register usage, occupancy, DRAM throughput, IPC
  - Compare radix-8 vs radix-4: DRAM bytes read/written (should show ~33% reduction)
  - OTF vs precomputed: DRAM bytes (twiddle traffic should disappear)
  - L2 hit rate comparison: does eliminating twiddle traffic improve coefficient caching?
- Capture screenshots: radix-8 roofline, OTF DRAM comparison
- Generate benchmark charts: v1.5.0 vs v1.4.0 bar chart at all sizes
- Full regression sweep: all 230+ existing tests pass + new tests green
- Update `results/analysis.md` (new section: "Radix-8 + OTF Analysis")
- Update `results/data/bench_v150.json`
- Update README performance tables
- Update CLAUDE.md with radix-8 and OTF documentation
- Tag v1.5.0 release

**Expected results (n=2^22):**
- Radix-8 alone: ~13–15 ms (−15–25% vs v1.4.0's 17.1 ms)
- Radix-8 + OTF: ~12–14 ms (additional −5–10% from twiddle traffic elimination)
- Batched 8× at 2^22: proportional improvement (~110–130 ms vs v1.4.0's ~150 ms)

**Test plan — v1.5.0 release regression sweep:**
- All existing v1.4.0 tests pass (230/230 — no regressions)
- Full test matrix: {NAIVE, OPTIMIZED, BARRETT} × {single, batch-1, batch-8}
  × {forward, roundtrip} × {2^10..2^22}
- Radix-8 cross-validation: radix-8 == radix-4 at all sizes (bitwise)
- OTF cross-validation: OTF == precomputed at all sizes (bitwise)
- CUDA Graph tests still pass (graphs should use new radix-8 kernels automatically)
- **Test count target**: ≥275 total

**Deliverables:**
- Updated benchmark tables + charts (radix-8, OTF comparison)
- ncu screenshots (radix-8 roofline, OTF DRAM savings)
- Full regression pass (all tests green)
- Git tag v1.5.0

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

### Session 16 — Goldilocks + BabyBear Field Arithmetic

**Objective:** Implement modular arithmetic for both fields (GPU + CPU reference) and
validate with exhaustive tests.

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

### Session 17 — Multi-Field NTT Integration + Correctness

**Objective:** Implement NTT kernels for Goldilocks and BabyBear fields, validate
correctness, handle field-specific optimizations.

**Design decision — Separate kernels vs templates:**
- **Separate kernels** (recommended): Each field has fundamentally different element size
  (32B vs 8B vs 4B), different shared memory per element, different register pressure.
  Template instantiation would work but shared memory sizing, thread counts, and K values
  all differ. Separate implementation is cleaner and allows per-field tuning.
- Goldilocks: 8 bytes/element → K=10 fuses 1024 elements × 8B = 8 KB shmem (vs 32 KB for BLS)
  → could potentially go to K=12 (4096 elements × 8B = 32 KB) for deeper fusion
- BabyBear: 4 bytes/element → K=10 fuses 1024 × 4B = 4 KB → could go to K=13 (8192 × 4B = 32 KB)

**Tasks:**
- Implement Goldilocks NTT in new `src/ntt_goldilocks.cu`:
  - Fused inner kernel (K=10 minimum, explore K=12 for deeper fusion)
  - Cooperative outer-stage kernel (radix-4 or radix-8 depending on v1.5.0 results)
  - Bit-reverse permutation kernel
  - Twiddle precomputation: find primitive root for Goldilocks field
    (p-1 = 2^32 × (2^32 - 1), has large power-of-2 factor → NTT-friendly up to 2^32!)
  - Forward + inverse, single + batched
- Implement BabyBear NTT in new `src/ntt_babybear.cu`:
  - Same structure as Goldilocks
  - BabyBear p-1 = 2^27 × (2^4 - 1) → NTT-friendly up to 2^27
  - Explore deeper fusion: K=12 or K=13 (entire NTT may fit in shmem for small sizes)
- Public API additions to `include/ntt.cuh` (or new headers):
  ```cpp
  void ntt_forward_goldilocks(GoldilocksElement* d_data, size_t n, cudaStream_t stream = 0);
  void ntt_inverse_goldilocks(GoldilocksElement* d_data, size_t n, cudaStream_t stream = 0);
  void ntt_forward_babybear(BabyBearElement* d_data, size_t n, cudaStream_t stream = 0);
  void ntt_inverse_babybear(BabyBearElement* d_data, size_t n, cudaStream_t stream = 0);
  ```
- CPU reference NTT for both fields in `tests/ff_reference.h`
  (extend existing CPU NTT with field-generic template or separate functions)

**Test plan — Multi-field NTT correctness:**
- **Forward NTT vs CPU reference:**
  - Goldilocks: sizes 2^10..2^22 (13 sizes)
  - BabyBear: sizes 2^10..2^22 (13 sizes)
- **Roundtrip INTT(NTT(x)) = x:**
  - Both fields, sizes 2^10..2^22
- **Known-vector patterns:**
  - All-zeros, all-ones, single-nonzero, ascending for both fields (3 sizes each)
- **Batched:**
  - B=8 vs sequential at 2^15, 2^18, 2^22 for both fields
- **Cross-field sanity**: at 2^10 where computation is trivially verifiable by hand,
  verify all 3 fields produce correct DFT coefficients for a known input vector

**Deliverables:**
- `src/ntt_goldilocks.cu` — Goldilocks NTT implementation
- `src/ntt_babybear.cu` — BabyBear NTT implementation
- Updated `include/ntt.cuh` (or `include/ntt_goldilocks.cuh`, `include/ntt_babybear.cuh`)
- CPU reference NTT for both fields
- NTT correctness tests: ~30 new tests per field (~60 total)
- Target: ~340 total tests

---

### Session 18 — Multi-Field Benchmark + Charts + Release v1.6.0

**Objective:** Comprehensive 3-way benchmark, generate comparison charts, release.

**Tasks:**
- Full benchmark matrix: {BLS12-381, Goldilocks, BabyBear} × {2^15..2^22} × {single, batch-8}
  - 7-rep median for each configuration
  - Record both wall-clock time and throughput (elements/sec, NTTs/sec)
- Generate comparison charts using `scripts/plot_benchmarks.py`:
  - **Main chart**: 3 fields × 5 sizes bar chart (wall-clock time, log scale)
  - **Speedup chart**: Goldilocks and BabyBear speedup factor vs BLS12-381
  - **Throughput chart**: elements/sec for all 3 fields (shows memory-bound ceiling)
  - **Batch comparison**: batch efficiency (batch/sequential ratio) per field
- Profile with ncu:
  - Compare compute utilization across fields: BLS12-381 (compute-bound inner),
    Goldilocks (where's the bottleneck?), BabyBear (fully memory-bound?)
  - Verify BabyBear is indeed fully DRAM-bound end-to-end (both inner + outer)
- Key analysis question: does smaller element size change the inner/outer bottleneck ratio?
  - BLS12-381: inner=compute-bound, outer=memory-bound
  - Goldilocks: inner should be much faster → outer dominates even more?
  - BabyBear: inner is trivial → outer is essentially entire NTT?
  - This has implications for which optimizations matter per field
- Full regression sweep: all tests pass
- Update `results/analysis.md` (new section: "Multi-Field Comparison")
- Update README with 3-field comparison table and chart
- Update CLAUDE.md with Goldilocks/BabyBear documentation
- Tag v1.6.0 release

**Expected results (n=2^22, single NTT):**
- BLS12-381 Montgomery: ~13 ms (v1.5.0 target) or ~17 ms (v1.4.0 if radix-8 underperforms)
- Goldilocks: ~1.0–1.5 ms (10–15× faster — 8B elements, trivial arithmetic)
- BabyBear: ~0.3–0.5 ms (35–50× faster — 4B elements, ~5 instructions per mul)

**Deliverables:**
- 3-field benchmark data: `results/data/bench_v160.json`
- Comparison charts in `results/charts/`
- Updated README with multi-field performance table
- All tests green (target: ~340 total)
- Git tag v1.6.0

---

## v1.7.0 — Plantard Reduction for BLS12-381

**Goal:** Implement Plantard modular reduction as an alternative to Montgomery/Barrett for
the compute-bound fused inner kernel. Plantard eliminates one big-integer multiplication
per twiddle multiply by precomputing twiddle-specific Plantard constants.

**Expected improvement:** 5–15% on fused inner kernel → 2–5% total NTT improvement.
This is a targeted optimization for the inner kernel (now ~35% of total time post-radix-8).

**Rationale:** Unlike MoMA (which requires SPIRAL code generation and has 381-bit efficiency
problems), Plantard is a clean mathematical technique with proven GPU results. Özcan & Savaş
(IEEE Access 2025) benchmark Plantard vs Montgomery on RTX 3060Ti, showing 8–15% improvement
on compute-bound NTT inner stages for 60-bit primes. For 255-bit BLS12-381, the saving is
less dramatic but still saves one 8-limb multiplication per butterfly.

**References:**
- Plantard, "Efficient Word Size Modular Arithmetic", IEEE TETC 2021
- Özcan & Savaş (IEEE Access 2025) §V-C: Plantard benchmark on RTX GPUs

---

### Session 19 — Plantard Arithmetic + Twiddle Precomputation

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

### Session 20 — Plantard NTT Integration + Benchmark + Release v1.7.0

**Objective:** Integrate Plantard into the fused inner kernel, benchmark, release.

**Tasks:**
- Add `NTTMode::PLANTARD` to the NTT API
- Integrate Plantard into fused K=10 kernel:
  - New fused kernel variant using `ff_mul_plantard` for twiddle × element multiply
  - `ff_add_v2` / `ff_sub_v2` remain unchanged (Plantard outputs standard-form)
  - Add to `ntt_fused_kernels.cu` (separate instantiation, same TU pattern)
- Integrate into outer stages:
  - Since outer stages are memory-bound, Plantard's compute savings may not help
  - Test both: Plantard outer stages vs Montgomery/Barrett outer stages
  - If no improvement, keep Plantard for inner kernel only + Montgomery/Barrett outer
  - This "mixed-mode" approach uses the best arithmetic for each kernel's bottleneck
- Plantard twiddle cache: precompute and cache alongside Montgomery/Barrett twiddles
- Benchmark: {Montgomery, Barrett, Plantard, Plantard+Montgomery-outer} × {2^15..2^22}
  - Focus on fused kernel time (use ncu `--kernel-id` to isolate)
  - Full NTT time for all modes
- Profile: register usage, IPC, occupancy for Plantard fused kernel vs Montgomery
- Full regression sweep
- Update analysis.md, README, CLAUDE.md
- Tag v1.7.0

**Test plan — Plantard NTT integration:**
- **Forward NTT vs CPU reference:**
  - Plantard mode: all sizes 2^10..2^22
- **Roundtrip INTT(NTT(x)) = x:**
  - Plantard mode: all sizes 2^10..2^22
- **Cross-validation (Plantard NTT vs Montgomery NTT vs Barrett NTT):**
  - Bitwise identical output at all sizes 2^10..2^22
- **Batched Plantard:**
  - B=8 vs sequential at 2^15, 2^18, 2^22
  - Batched Plantard vs batched Montgomery (bitwise identical)
- **Mixed-mode validation:**
  - Plantard-inner + Montgomery-outer vs pure Montgomery (should be identical if
    conversion between representations is correct at kernel boundary)

**Expected results (n=2^22, single NTT):**
- Pure Plantard: ~12–14 ms (inner kernel 5–15% faster, outer unchanged)
- Plantard inner + radix-8 outer: ~11–13 ms
- vs v1.5.0 baseline: 2–5% improvement

**Deliverables:**
- `NTTMode::PLANTARD` in public API
- Plantard fused kernels in `ntt_fused_kernels.cu`
- Plantard twiddle cache
- Benchmark data: `results/data/bench_v170.json`
- All tests green (target: ~370 total)
- Git tag v1.7.0

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
| 12 | v1.5.0 | L2 diagnostic + radix-8 butterfly design | Profile-guided + higher radix |
| 13 | v1.5.0 | Radix-8 cooperative kernels + correctness | ~67% DRAM traffic reduction/triple |
| 14 | v1.5.0 | On-the-fly twiddle computation | Eliminate twiddle DRAM traffic |
| 15 | v1.5.0 | Benchmark + profile + release v1.5.0 | Target: ~13 ms at 2^22 |
| 16 | v1.6.0 | Goldilocks + BabyBear field arithmetic | Multi-field ZKP comparison |
| 17 | v1.6.0 | Multi-field NTT integration + correctness | Field-specific kernel tuning |
| 18 | v1.6.0 | 3-way benchmark + charts + release v1.6.0 | Portfolio/insight feature |
| 19 | v1.7.0 | Plantard arithmetic + twiddle precompute | Eliminate 1 big-int mul/butterfly |
| 20 | v1.7.0 | Plantard NTT integration + benchmark + release v1.7.0 | ~2–5% total improvement |
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
- v1.5.0: ~13 ms (radix-8 + OTF twiddles, −24% vs v1.4.0) — **projected**
- v1.6.0: ~13 ms BLS / ~1.2 ms Goldilocks / ~0.4 ms BabyBear — **multi-field** (projected)
- v1.7.0: ~12.5 ms (Plantard inner kernel, −4% vs v1.5.0) — **projected**
- v1.8.0: ~10 ms (Stockham coalesced outer, −20% vs v1.5.0, conditional) — **projected**

**Batch throughput (8× 2^22 NTTs):**
- v1.1.0: 8 × 25.1 ms = ~201 ms (sequential)
- **v1.2.0: 219 ms batched (~1.0x) / 216 ms sequential** — GPU saturated, no gain at 2^22.
  Batching helps at 2^15 (1.52x) where GPU is underutilized.
- **v1.3.0: 241 ms (4-step batched) vs 199 ms (Barrett batched)** — 4-step +21% slower.
  4-step benefits more from batching (1.16x vs 1.05x) but still slower in absolute terms.
  Best batch mode remains Barrett at 199 ms.
- **v1.4.0: ~150 ms (Montgomery batched) / ~159 ms (Barrett batched)** — radix-4 outer stages
  reduce batch time by ~20%. Montgomery faster for batched (fewer instructions per ff_mul).
- v1.5.0: ~100–110 ms (radix-8 batched, projected)

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

**Why OTF twiddles are bundled with v1.5.0 (not standalone):**
The retrospective notes: "Standalone: too small to measure above noise. Best deployed as
part of a combined radix-8 + OTF outer-stage rewrite." OTF saves ~1 ms at n=2^22 — within
measurement noise as a standalone change, but measurable when combined with radix-8.

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

## References

- **MoMA**: Zhang & Franchetti, "Code Generation for Cryptographic Kernels using Multi-word
  Modular Arithmetic on GPU", CGO 2025. [arXiv:2501.07535](https://arxiv.org/html/2501.07535),
  [SPIRAL pub](https://spiral.ece.cmu.edu/pub-spiral/abstract.jsp?id=376)
- **ZKProphet**: Verma et al., IEEE IISWC 2025 — §V-B: open optimization targets
- **cuZK**: Lu et al., TCHES 2023 — async pipeline methodology
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
