# NTT Optimization Roadmap — Future Releases

## Current State (v1.1.0)

**Benchmark (RTX 3060 Laptop, n=2^22):** 25.2 ms (single NTT, compute only)
**Target:** ≤20 ms (not met — outer stages are memory-bandwidth-limited)

### Where Time Goes (n=2^22, 4 kernel launches)

| Phase | Duration | % of Total | Bottleneck |
|---|---|---|---|
| Bit-reverse permutation | ~0.3 ms | 1% | Memory (scatter) |
| Fused K=10 (stages 0-9) | ~2.5 ms | 10% | **Compute** (69%, IPC 2.41) |
| Cooperative outer (stages 10-21) | ~19.4 ms | 77% | **Memory** (DRAM R-M-W) |
| Montgomery conversion (to/from) | ~3.0 ms | 12% | Memory |

**Root cause**: Each outer-stage butterfly reads 2 FpElements (64 B) + 1 twiddle (32 B)
from DRAM, performs 1 ff_mul + 1 ff_add + 1 ff_sub, then writes 2 FpElements back.
With stride doubling each stage, data quickly exceeds L2 capacity (4 MB on RTX 3060),
forcing every access to hit DRAM.

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
- **4-step skeleton**: forward + inverse Barrett paths. Currently uses 4 transposes
  (will optimize to 2 in Session 6). Reuses existing batched sub-NTT infrastructure.
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

### Session 6 — 4-Step NTT: Sub-NTT Integration

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

**Deliverables:**
- Complete `ntt_4step_forward` and `ntt_4step_inverse` implementation
- Integration with NTT dispatch (new `NTTMode::FOUR_STEP`)
- Intermediate correctness checks (column NTTs alone, then full pipeline)
- Batched 4-step path

---

### Session 7 — 4-Step NTT: Correctness + Edge Cases

**Objective:** Exhaustive correctness testing and edge case handling.

**Tasks:**
- Test 4-step NTT against CPU reference for all sizes 2^10 through 2^22
- Test roundtrip: INTT(NTT(x)) = x for all sizes
- Test with known vectors (all-zeros, all-ones, single-element, random)
- Test batched 4-step: batch of 8 vs 8 sequential single 4-step NTTs
- Handle sizes too small for 4-step (n < n1×n2 minimum) — fallback to fused+cooperative
- Handle sizes where √n is not integer (odd log_n): use n1=2^(k/2), n2=2^((k+1)/2)
- Verify inverse NTT uses correct inverse twiddles and n^{-1} scaling

**Deliverables:**
- Full test coverage for 4-step path (single + batched)
- Fallback logic for small sizes
- All existing tests still pass (no regressions)

---

### Session 8 — 4-Step NTT: Benchmark, Profile, Release v1.3.0

**Objective:** Quantify the 4-step improvement and release.

**Tasks:**
- Benchmark 4-step vs cooperative (v1.1) for all sizes (single + batched)
- Profile with ncu: capture roofline for sub-NTT kernels, transpose, twiddle multiply
  - Expect sub-NTTs to be compute-bound (like current fused kernel)
  - Expect transpose to be memory-bound but with coalesced access (high DRAM efficiency)
  - Expect twiddle multiply to be compute-bound (embarrassingly parallel ff_mul)
- Measure total DRAM traffic: 4-step should be ~3 full passes (load + twiddle + store)
  vs current 12+ passes (one per outer stage)
- Capture screenshots: roofline for 4-step sub-NTT, transpose, and twiddle kernels
- Update analysis.md, README, CLAUDE.md
- Update plot_benchmarks.py with new data
- Tag v1.3.0 release

**Deliverables:**
- Updated benchmark tables (single + batch throughput)
- New ncu screenshots and analysis
- Git tag v1.3.0

---

## v1.4.0 — MoMA-Inspired Register Optimization + Phase-Aware Pipeline

**Goal:** Push arithmetic performance closer to MoMA's auto-generated code by adopting
register-centric data flow patterns, and exploit the 4-step NTT's phase structure for
optimal DMA overlap.

**Expected improvement:** 10-20% single-NTT compute; 20-40% end-to-end pipeline

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

### Session 10 — Phase-Aware Async Pipeline

**Objective:** Exploit the 4-step NTT's distinct compute/memory phases for optimal
DMA overlap.

The 4-step NTT has a clear phase structure:
- transpose (memory-bound) → sub-NTTs (compute-bound) → twiddle (compute-bound) →
  transpose (memory-bound) → sub-NTTs (compute-bound)

DMA transfers can overlap with compute phases without memory controller contention —
solving the v1.1 pipeline's DMA interference problem at n=2^22.

**Tasks:**
- Implement phase-aware overlap strategy:
  - During compute phases: issue H2D for next batch + D2H for previous batch
  - During memory phases: pause DMA, let NTT have full DRAM bandwidth
  - Use `cudaStreamWaitEvent` to gate DMA streams on phase boundaries
- Extend `AsyncNTTPipeline` with 4-step NTT support
- Implement adaptive dispatch:
  - n < 2^8: naive radix-2
  - 2^8 ≤ n < 2^20: fused K=10 + cooperative outer
  - n ≥ 2^20: 4-step NTT
  - Batch mode: always 4-step (sub-NTTs provide natural parallelism)
- Benchmark: phase-aware pipeline vs naive overlap vs sequential at 2^22

**Deliverables:**
- Phase-aware `AsyncNTTPipeline`
- Adaptive dispatch logic
- Benchmark comparison + nsys timeline visualization

---

### Session 11 — CUDA Graphs + Final Polish + Release v1.4.0

**Objective:** Wrap all launch sequences in CUDA Graphs, final benchmarking, release.

**Tasks:**
- Implement CUDA Graph capture for:
  - Single NTT (4-step sequence: transpose → sub-NTTs → twiddle → transpose → sub-NTTs)
  - Batched NTT (same structure, more blocks)
  - Graph instantiation caching by (n, batch_size) key
- Final benchmark suite: all modes × all sizes × single/batch
- Final ncu profiling: capture definitive roofline and warp stall screenshots
- nsys timeline: visualize 4-step phase-aware pipeline overlap
- Update all documentation: README, analysis.md, GUIDE.md, CLAUDE.md, profiling/README.md
- Generate final charts
- Tag v1.4.0 release

**Deliverables:**
- CUDA Graph integration
- Complete benchmark suite
- Full documentation update
- Git tag v1.4.0

---

## Stretch Goals (v1.5.0+)

### SPIRAL/MoMA Code Generation Study
- Attempt to use SPIRAL (open-source) to generate BLS12-381 NTT kernels
- Compare auto-generated vs hand-tuned code quality (SASS instruction count, throughput)
- If SPIRAL-generated code is superior, adopt as primary implementation

### On-the-Fly Twiddle Computation
- Compute twiddles during butterfly stages instead of precomputing and loading from memory
- Trades memory bandwidth for compute (beneficial when compute-bound)
- Uses fast modular exponentiation: ω^k = ω^(k-1) · ω (sequential chain per thread)

### Mixed-Radix Outer Stages
- Use radix-4 butterflies for outer stages (2 stages per kernel, halving launch count)
- Higher arithmetic intensity per memory access → better compute/memory ratio

### Multi-GPU NTT
- Split n-point NTT across 2+ GPUs using the 4-step decomposition
- Column NTTs on GPU 0, row NTTs on GPU 1 (or split evenly)
- Requires NVLink or PCIe peer-to-peer for transpose step

---

## Summary: Session Plan

| Session | Release | Focus | Key Technique |
|---|---|---|---|
| 1 | v1.2.0 | Barrett reduction for BLS12-381 | MoMA-inspired arithmetic |
| 2 | v1.2.0 | Barrett NTT integration + benchmarking | Eliminate Montgomery overhead |
| 3 | v1.2.0 | Batched NTT kernel | MoMA batch processing pattern |
| 4 | v1.2.0 | Benchmark, profile, release | — |
| **5** | **v1.3.0** | **4-step NTT: transpose kernel + architecture** ✅ | **Bailey's algorithm** |
| 6 | v1.3.0 | 4-step NTT: sub-NTT integration (+ batch) | Sub-NTTs fit in shmem |
| 7 | v1.3.0 | 4-step NTT: correctness + edge cases | — |
| 8 | v1.3.0 | Benchmark, profile, release | — |
| 9 | v1.4.0 | Register-centric butterfly + carry chains | MoMA register optimization |
| 10 | v1.4.0 | Phase-aware async pipeline | Compute/memory phase separation |
| 11 | v1.4.0 | CUDA Graphs + final polish, release | Launch overhead elimination |

**Cumulative results (n=2^22, single NTT, 5-rep median):**
- v1.1.0: 25.1 ms (Montgomery, fused K=10 + cooperative outer)
- **v1.2.0: 24.9 ms (Barrett, -0.8%)** — minimal single-NTT gain; outer stages still dominate
- v1.3.0 target: ~13-16 ms (4-step eliminates outer-stage DRAM passes)
- v1.4.0 target: ~10-14 ms (register optimization + CUDA Graphs)

**Batch throughput (8× 2^22 NTTs):**
- v1.1.0: 8 × 25.1 ms = ~201 ms (sequential)
- **v1.2.0: 219 ms batched (~1.0x) / 216 ms sequential** — GPU saturated, no gain at 2^22.
  Batching helps at 2^15 (1.52x) where GPU is underutilized.
- v1.3.0 target: ~80-100 ms (4-step batched, all sub-NTTs in shmem)
- v1.4.0 target: ~60-80 ms (register opt + CUDA Graphs + phase-aware pipeline)

---

## References

- **MoMA**: Zhang & Franchetti, "Code Generation for Cryptographic Kernels using Multi-word
  Modular Arithmetic on GPU", CGO 2025. [arXiv:2501.07535](https://arxiv.org/html/2501.07535),
  [SPIRAL pub](https://spiral.ece.cmu.edu/pub-spiral/abstract.jsp?id=376)
- **ZKProphet**: Verma et al., IEEE IISWC 2025 — §V-B: open optimization targets
- **cuZK**: Lu et al., TCHES 2023 — async pipeline methodology
- **Bailey**: D.H. Bailey (1990) — "FFTs in External or Hierarchical Memory" (4-step FFT)
- **ICICLE**: Ingonyama, GPU acceleration library for ZKP — [github](https://github.com/ingonyama-zk/icicle)
- NVIDIA CUDA Programming Guide — §3.2.11 CUDA Graphs, §5.2 Memory Hierarchy
- Nsight Compute Documentation — L2 cache metrics, roofline analysis
