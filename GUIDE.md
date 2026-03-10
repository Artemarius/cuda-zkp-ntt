# GUIDE.md — Technical Deep Dive
# cuda-zkp-ntt: NTT, Finite Fields, and GPU Optimization for ZKP

---

## Table of Contents

1. [Zero-Knowledge Proofs — The Big Picture](#1-zero-knowledge-proofs)
2. [Groth16 Protocol and Why NTT Matters](#2-groth16-and-ntt)
3. [Number-Theoretic Transform — Mathematical Foundation](#3-ntt-mathematics)
4. [Finite Fields and BLS12-381](#4-finite-fields)
5. [Montgomery Multiplication — The Core Primitive](#5-montgomery-multiplication)
6. [GPU Architecture for Integer Workloads](#6-gpu-architecture)
7. [IMAD vs IADD3 — The Instruction-Level Opportunity](#7-imad-vs-iadd3)
8. [Async Pipelines and Double Buffering in CUDA](#8-async-pipeline)
9. [Nsight Compute — Profiling Methodology](#9-profiling)
10. [Implementation Walkthrough](#10-implementation)

---

## 1. Zero-Knowledge Proofs

### What ZKPs Do

A Zero-Knowledge Proof is a cryptographic protocol where one party (the **Prover**) convinces another party (the **Verifier**) that a computation `f(x, w) = y` was performed correctly, where:
- `f` is a public function
- `x` is a public input
- `w` is the **witness** — a secret known only to the Prover
- `y` is the public output

The proof `π` reveals **nothing** about `w` beyond the fact that the Prover knows it. This is the "zero-knowledge" property.

### Why This Matters

ZKPs enable:
- **Private cryptocurrencies**: Zcash uses ZKPs so transaction amounts are hidden but balances remain valid
- **Blockchain scaling**: Ethereum rollups (zkSync, StarkNet, Polygon zkEVM) batch thousands of transactions into a single proof verified on-chain
- **Verifiable ML**: Prove that a model produced an output without revealing model weights
- **Identity**: Prove you are over 18 without revealing your birthdate

### ZK-SNARKs

The practical variant used in production is **zkSNARK**: zero-knowledge **Succinct Non-interactive ARgument of Knowledge**.

Properties:
- **Succinct**: proof size is tiny (< 200 bytes for Groth16), independent of computation size
- **Non-interactive**: no back-and-forth; Prover sends one message
- **Argument of Knowledge**: cannot fake proof without actually knowing `w`

The tradeoff: proof **generation** is expensive. The Prover must perform massive polynomial arithmetic. This is the bottleneck this project targets.

---

## 2. Groth16 and Why NTT Matters

### The Groth16 Protocol

Groth16 (Groth, 2016) is the dominant zkSNARK scheme. Its proofs are constant-size (~128 bytes) and verify in ~1ms. Almost every production ZKP system uses Groth16 or a variant.

The protocol encodes the computation `f` as a **Rank-1 Constraint System (R1CS)**: a system of quadratic equations. The Prover must then:

1. Compute **polynomial evaluations** via matrix-vector multiplication (MUL)
2. Convert polynomial representations via **NTT / INTT** (Number-Theoretic Transform)
3. Commit to polynomials via **MSM** (Multi-Scalar Multiplication on elliptic curves)

The proof `π` consists of three elliptic curve points.

### Workload Breakdown

From ZKProphet (2025) and cuZK (2023), the Prover time breaks down as:

| Operation | % of Prover time (pre-MSM optimization) | % (post-MSM optimization) |
|-----------|------------------------------------------|---------------------------|
| MSM (G1)  | ~70%                                     | ~9%                       |
| NTT/INTT  | ~25%                                     | **~91%**                  |
| MUL       | ~4%                                      | <1%                       |
| Other     | ~1%                                      | <1%                       |

**This is the opportunity**: as MSM has been beaten to ~800× speedup, NTT is now the wall. Current best NTT GPU implementations achieve only ~50× over CPU. This project attacks the NTT bottleneck directly.

### Why NTT Lags

Two root causes identified by ZKProphet:

1. **Transfer-dominated**: NTT spends disproportionate time on CPU→GPU data transfer. The on-device butterfly compute is fast; getting the data there is not. MSM implementations use async streams to overlap transfers; NTT implementations do not.

2. **Suboptimal arithmetic**: NTT's butterfly operation is dominated by FF_mul (modular multiplication). The GPU implementation uses IMAD instructions almost exclusively, but there are lower-latency instruction sequences available.

---

## 3. NTT — Mathematical Foundation

### What NTT Is

NTT is the **Discrete Fourier Transform over a finite field**. Instead of complex roots of unity, we use **roots of unity in a prime field**.

Formally, given a vector `a = [a₀, a₁, ..., aₙ₋₁]`, the NTT computes:

```
A[i] = Σⱼ₌₀ⁿ⁻¹  a[j] · ωⁱʲ    (mod p)
```

where `ω` is the **primitive n-th root of unity** in `Fₚ`, i.e., `ωⁿ ≡ 1 (mod p)` and `ωᵏ ≢ 1 (mod p)` for `0 < k < n`.

The different powers `ω⁰, ω¹, ..., ωⁿ⁻¹` are the **twiddle factors**.

### Why Polynomials Need NTT

In Groth16, the Prover must multiply large polynomials. Naïve polynomial multiplication is O(n²). NTT enables O(n log n) multiplication:

```
1. Compute NTT(a) and NTT(b)         — evaluate polynomials at n points
2. Pointwise multiply: C[i] = A[i]·B[i]    — O(n) in the evaluation domain
3. Compute INTT(C)                    — convert back to coefficient form
```

The NTT is invertible: `INTT(NTT(a)) = a`, implemented with different twiddle factors and a final scaling by n⁻¹ mod p.

### The Cooley-Tukey Algorithm

The Fast NTT algorithm uses the Cooley-Tukey butterfly decomposition. For n = 2^k:

**Radix-2 Butterfly Operation**:
```
Given elements A[i] and A[j] with twiddle factor ω:
    t = ω · A[j]
    A[j] = A[i] - t
    A[i] = A[i] + t
```

An n-point NTT requires:
- `log₂(n)` stages
- `n/2` butterfly operations per stage
- Total: `(n/2) · log₂(n)` butterfly operations

**Example for n=8** (3 stages):
```
Stage 1: butterfly(A[0],A[4]), butterfly(A[1],A[5]), butterfly(A[2],A[6]), butterfly(A[3],A[7])
Stage 2: butterfly(A[0],A[2]), butterfly(A[1],A[3]), butterfly(A[4],A[6]), butterfly(A[5],A[7])
Stage 3: butterfly(A[0],A[1]), butterfly(A[2],A[3]), butterfly(A[4],A[5]), butterfly(A[6],A[7])
```

Between stages, elements are shuffled (bit-reversal permutation).

### Radix-256 Optimization

Instead of processing stages one at a time, **radix-256** combines 8 stages into a single kernel using shared memory. This reduces:
- Number of kernel launches from `log₂(n)` to `log₂(n)/8`
- Global memory round-trips per stage
- Total memory bandwidth pressure

bellperson implements radix-256 NTT. For n = 2²⁶, this means 4 kernel launches instead of 26.

### GPU Parallelization

Each butterfly operates on a pair of elements. Since all butterflies within a stage are **independent**, we assign one thread per butterfly:

```
blockDim = 256 threads
gridDim  = n / (2 * blockDim)
```

Each thread:
1. Loads two elements from global memory (or shared memory in radix-256)
2. Loads the appropriate twiddle factor
3. Performs one butterfly (2 FF_adds + 1 FF_mul)
4. Stores results

The critical observation: **each butterfly does 3 finite-field operations**. FF_mul dominates.

---

## 4. Finite Fields and BLS12-381

### What a Finite Field Is

A **finite field** `Fₚ` (also written GF(p)) is the set `{0, 1, ..., p-1}` with addition and multiplication defined modulo a prime `p`.

All arithmetic results are reduced modulo `p`, so they always stay in `[0, p)`.

Key operations:
- **Addition**: `(a + b) mod p`
- **Subtraction**: `(a - b + p) mod p` (add p to avoid negative)
- **Multiplication**: `(a · b) mod p`
- **Inverse**: `a⁻¹` such that `a · a⁻¹ ≡ 1 (mod p)` — computed via extended Euclidean algorithm

### BLS12-381 Scalar Field

The BLS12-381 elliptic curve is used in Ethereum, Zcash, Filecoin, and other production systems for its favorable security/performance tradeoffs.

It has two relevant fields:
- **Scalar field** `Fr`: used for NTT coefficients and scalars in MSM
- **Base field** `Fq`: used for elliptic curve point coordinates in MSM

For NTT, we work in `Fr`:

```
r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
```

This is a **255-bit prime**. In decimal:
```
r = 52435875175126190479447740508185965837690552500527637822603658699938581184513
```

Properties important for NTT:
- `r - 1 = 2³² · 3 · ...` — meaning `r` has a large power-of-2 factor, so NTTs of size up to 2³² exist in `Fr`
- We work with sizes up to 2²⁶ in practice

### Representing 255-bit Numbers on GPU

GPUs have 32-bit integer units. A 255-bit number needs **8 × 32-bit limbs**:

```c
struct FpElement {
    uint32_t limbs[8];  // little-endian: limbs[0] is least significant
};
```

Operations on this type require multi-precision arithmetic using carry propagation.

---

## 5. Montgomery Multiplication

### The Problem with Modular Multiplication

Naïve modular multiplication: `(a · b) mod p`

This requires:
1. Compute `a · b` — a 255×255-bit multiplication → 510-bit result
2. Compute `(510-bit result) mod p` — requires a 510-bit division by p

Division is extremely expensive on both CPU and GPU. Montgomery form avoids it.

### Montgomery Form

Instead of storing `a`, we store `ã = a · R mod p` where `R = 2^256` (chosen because it's a power of 2, making `R mod p` operations cheap via shifts).

In Montgomery form:
- A field element `a` is represented as `ã = a · R mod p`
- The identity element (1) is stored as `R mod p`
- The zero element (0) is stored as 0

### Montgomery Multiplication (MonMul)

Given `ã = a·R mod p` and `b̃ = b·R mod p`, compute `ã·b̃ · R⁻¹ mod p = (a·b) · R mod p`:

```
MonMul(ã, b̃):
    T = ã · b̃                    // 512-bit product
    m = T · p⁻¹ mod R            // reduction factor (cheap: mod R is masking)
    T = (T + m · p) / R          // right-shift by 256 bits (cheap)
    if T ≥ p: T = T - p          // conditional reduction
    return T
```

The key insight: `/ R` is a right-shift (free), and `mod R` is masking (free). The expensive division by `p` is eliminated, replaced by one additional multiplication by `m`.

### CIOS Algorithm

CIOS (Coarsely Integrated Operand Scanning) performs Montgomery multiplication in a single pass through the limbs, computing the product and reduction simultaneously:

```
for i = 0 to 7:
    C = 0
    for j = 0 to 7:
        (C, S) = T[j] + A[j] * B[i] + C
        T[j] = S
    T[8] += C
    m = T[0] * p⁻¹[0]        // 32-bit, cheap
    (C, _) = T[0] + m * p[0]
    for j = 1 to 7:
        (C, S) = T[j] + m * p[j] + C
        T[j-1] = S
    T[7] = T[8] + C
    T[8] = 0
```

The inner `A[j] * B[i]` is a 32×32→64-bit multiply. On GPU, this maps to `mad.lo.cc` + `mad.hi.cc` PTX instructions.

### Converting to Montgomery Form

Before NTT: multiply each input element by `R² mod p` using MonMul.
After NTT: multiply each output element by 1 using MonMul (which extracts from Montgomery form).

In practice, inputs are converted once before computation and outputs once after — the cost is amortized.

---

## 6. GPU Architecture for Integer Workloads

### SM Structure (Ampere / RTX 3060)

The RTX 3060 Laptop GPU has 30 Streaming Multiprocessors (SMs). Each SM contains:
- 4 Sub-Partitions (SMSPs)
- Each SMSP: 16 CUDA cores (INT32 units), 16 FP32 units, 8 FP64 units, 1 Tensor Core
- Warp scheduler: selects one warp per cycle to issue from
- 256KB register file per SM
- 128KB shared memory / L1 cache per SM (configurable)

Total RTX 3060 integer throughput: `30 SMs × 4 SMSPs × 16 INT32 = 1,920 INT32 ops/cycle × 1.78 GHz ≈ 3.42 TOPS (INT32)`

### Why ZKP Hits Integer Throughput

All finite-field operations in NTT and MSM use **only integer instructions** — no floating point. The butterfly operation chain is:

```
FF_mul → IMAD chain (8×8 limb cross-products) → conditional reduction (IADD3 + setp)
FF_add → IADD3 chain + conditional reduction
```

ZKProphet §IV-C shows that:
- GPU schedulers issue new instructions every **3.2 cycles** instead of every cycle
- 67.5% of cycles see **no eligible warps** to issue from
- The INT32 pipeline is **throttled** — adding more warps makes it worse, not better

### Warp Execution Model

A **warp** is 32 threads executing in lockstep (SIMT). All 32 threads execute the same instruction on different data.

**Warp stall sources** (from ZKProphet Fig. 10):
- `Stall_Wait`: fixed-latency dependency — a new IMAD can't issue for 4 cycles after the previous one it depends on
- `Stall_Math_Pipe_Throttle`: INT32 pipeline is oversubscribed — all active warps want INT32, but only one SMSP can issue per cycle
- `Stall_Not_Selected`: warp is ready but scheduler picked another
- `Selected`: 1-cycle overhead of the issue itself

For FF_mul with 2 warps per SMSP (typical MSM config), ZKProphet measures total stall latency of **6.2 cycles** per IMAD instruction.

### Roofline Model for ZKP

The roofline model plots performance (GOPS) vs arithmetic intensity (ops/byte) with two ceilings:
- **Compute bound** (horizontal): limited by INT32 throughput
- **Memory bound** (diagonal): limited by DRAM/L2/L1 bandwidth

ZKProphet (Fig. 9) shows:
- FF_mul and FF_sqr: high arithmetic intensity, reach **60%** of INT32 ceiling
- FF_add, FF_sub: lower arithmetic intensity, reach **40%** of INT32 ceiling

The gap from 60% to 100% is the optimization opportunity. Changing IMAD to IADD3 (halving issue latency) would theoretically double throughput if supply-demand were the bottleneck — in practice, the improvement is partial due to instruction dependencies, but measurable.

---

## 7. IMAD vs IADD3 — The Instruction-Level Opportunity

### The Problem

Montgomery multiplication's inner loop computes:

```c
(carry, result) = A[j] * B[i] + T[j] + carry
```

This is a **multiply-accumulate** — exactly what IMAD does:
```
IMAD.LO.CC  Rd, Ra, Rb, Rc    // Rd = Ra * Rb + Rc (low 32 bits), sets carry flag
IMAD.HI.CC  Re, Ra, Rb, Rc    // Re = (Ra * Rb + Rc) >> 32 (high 32 bits)
```

IMAD has a **4-cycle issue latency** (Ampere and earlier architectures).

### The IADD3 Alternative

IADD3 performs a 3-input integer add:
```
IADD3  Rd, Ra, Rb, Rc    // Rd = Ra + Rb + Rc
```

IADD3 has a **2-cycle issue latency** — half of IMAD.

For operations that are purely additions (no multiply), switching to IADD3 immediately halves the instruction latency. Even in multiplies, the **reduction step** (adding `m·p` to the partial products) involves only accumulation — this accumulation chain can be restructured to use IADD3 with carry bits managed via `IADD3.X` (with carry-in).

### The Technique

Prior work on elliptic curve signatures (cited in ZKProphet §IV-B2) converts IMAD-dominated reduction code to IADD3 by:

1. Precomputing partial products separately
2. Using IADD3 tree-reduction to accumulate them
3. Managing carry propagation explicitly with IADD3.X (carry-extended add)

The PTX code pattern:
```ptx
// IMAD-based accumulation (slow):
mad.lo.u32      carry_lo, a, b, c;
madc.hi.u32     carry_hi, a, b, 0;

// IADD3-based accumulation (faster for pure addition chains):
add.cc.u32      r0, r0, addend;
addc.cc.u32     r1, r1, 0;
addc.u32        r2, r2, 0;
```

In practice, the conditional reduction step in FF_add/FF_sub contains **no multiply** — it is pure addition with carry. This is the clearest target for IADD3.

### Branch Efficiency in Conditional Reduction

ZKProphet Table VI shows:
- FF_add branch efficiency: **52.5%** (almost half the threads take a different branch)
- FF_mul branch efficiency: **84.0%**

The 52.5% efficiency for FF_add comes from the limb-by-limb comparison to determine if result ≥ p. With 8 limbs and 32 threads in a warp, the comparison result varies across threads, causing warp divergence.

**Branchless reduction**: use PTX `setp` (set predicate) and `selp` (select by predicate) to eliminate the branch:

```ptx
// Branchless conditional reduction: result = (T >= p) ? T - p : T
setp.ge.u32     pred, T[7], p[7];   // set predicate if T[7] >= p[7]
selp.u32        out[0], T_minus_p[0], T[0], pred;  // select based on predicate
// ... repeat for each limb
```

This replaces a conditional branch with predicated instructions — all 32 threads execute both paths but only write one result, eliminating divergence.

---

## 8. Async Pipelines and Double Buffering in CUDA

### The Transfer Bottleneck

ZKProphet Fig. 7 shows that NTT's GPU compute time is **dwarfed by CPU-GPU transfer time**, while MSM implementations use async transfers to hide this cost. The pattern:

```
NTT (bellperson):     [Transfer] → [Compute] → [Transfer] → [Compute] → ...
MSM (ymc, yrrid):     [Transfer] overlaps with [Compute]
```

### CUDA Streams

A **CUDA stream** is a queue of GPU operations that execute in order. Operations in **different streams** can execute concurrently if hardware resources allow.

```cuda
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

// These can overlap:
cudaMemcpyAsync(d_buf0, h_buf0, size, cudaMemcpyHostToDevice, stream0);
ntt_kernel<<<grid, block, 0, stream1>>>(d_input1, d_output1, n);
```

For overlap to happen:
1. The device must have a dedicated DMA engine (RTX 3060 does)
2. Host memory must be **pinned** (page-locked) — use `cudaMallocHost`
3. Operations must be in different streams
4. The kernel and copy must not depend on each other

### Double Buffer Pattern

The double buffer pattern uses two buffers on device and two on host:

```
Iteration k:    [H2D: buf_A] + [Compute: buf_B from iteration k-1]
Iteration k+1:  [H2D: buf_B] + [Compute: buf_A from iteration k]
```

Implementation sketch:
```cuda
// Allocate pinned host memory
float *h_buf[2];
cudaMallocHost(&h_buf[0], batch_size * sizeof(FpElement));
cudaMallocHost(&h_buf[1], batch_size * sizeof(FpElement));

// Allocate device double buffers
FpElement *d_buf[2];
cudaMalloc(&d_buf[0], batch_size * sizeof(FpElement));
cudaMalloc(&d_buf[1], batch_size * sizeof(FpElement));

// Kickstart: send first batch
cudaMemcpyAsync(d_buf[0], h_buf[0], ..., cudaMemcpyH2D, stream[0]);

for (int batch = 0; batch < num_batches; ++batch) {
    int cur = batch % 2;
    int nxt = (batch + 1) % 2;
    
    // Compute on current buffer
    ntt_kernel<<<..., stream[cur]>>>(d_buf[cur], ...);
    
    // Transfer next buffer concurrently
    if (batch + 1 < num_batches) {
        cudaMemcpyAsync(d_buf[nxt], h_buf[nxt], ..., cudaMemcpyH2D, stream[nxt]);
    }
    
    // Sync: current kernel must finish before we reuse stream
    cudaEventRecord(event[cur], stream[cur]);
    cudaStreamWaitEvent(stream[cur], event[nxt], 0);
}
```

### Synchronization with cudaEvent_t

`cudaEvent_t` provides fine-grained synchronization:
- `cudaEventRecord(event, stream)`: marks a point in stream's execution
- `cudaStreamWaitEvent(stream, event, 0)`: makes stream wait until event is recorded

This avoids the global `cudaDeviceSynchronize()` which would destroy pipeline overlap.

### Expected Benefit

For scale 2²²:
- NTT compute time: ~X ms
- CPU→GPU transfer time: ~Y ms (where Y > X per ZKProphet Fig. 7)
- With pipeline: effective time ≈ max(X, Y) per batch instead of (X + Y)
- Theoretical speedup: (X + Y) / max(X, Y) — can be 1.5× to 2× depending on ratio

On RTX 3060 (PCIe 4.0 × 16 → 32 GB/s peak transfer rate), BLS12-381 elements are 32 bytes each. For n = 2²², transfer volume = 2²² × 32 = 128 MB → ~4ms at peak PCIe bandwidth. Compare to NTT compute time at the same scale.

---

## 9. Nsight Compute — Profiling Methodology

### Key Metrics to Collect

Replicating ZKProphet analysis:

**Roofline** (replicate Fig. 9):
```bash
ncu --metrics \
  sm__inst_executed_pipe_alu.sum,\
  sm__inst_executed_pipe_fma.sum,\
  l1tex__t_bytes.sum,\
  lts__t_bytes.sum,\
  dram__bytes.sum \
  ./ntt_profile
```

**Warp Stall Breakdown** (replicate Fig. 10):
```bash
ncu --metrics \
  smsp__average_warp_latency_per_inst_issued.ratio,\
  smsp__warp_cycles_per_issue_stall_wait.avg,\
  smsp__warp_cycles_per_issue_stall_math_throttle.avg,\
  smsp__warp_cycles_per_issue_stall_not_selected.avg \
  ./ntt_profile
```

**Instruction Mix**:
```bash
ncu --metrics \
  smsp__inst_executed_op_integer.sum,\
  smsp__inst_executed_op_fp32.sum,\
  sass__inst_executed_op_imad.sum,\
  sass__inst_executed_op_iadd.sum \
  ./ntt_profile
```

**Branch Efficiency**:
```bash
ncu --metrics \
  smsp__sass_average_branch_targets_threads_uniform.pct \
  ./ntt_profile
```

**Occupancy**:
```bash
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active \
  ./ntt_profile
```

### Interpreting Results

**Roofline position**:
- X-axis: `arithmetic_intensity = (IMAD_ops × 2 + other_ops × 1) / bytes_accessed`
- Y-axis: `throughput = total_weighted_ops / kernel_time`
- Plot against RTX 3060 ceilings: INT32 = ~3.42 TOPS, DRAM = ~360 GB/s, L2 = ~1.5 TB/s

**Warp stall interpretation**:
- High `Stall_Wait` → instruction latency (unavoidable, instruction dependency chain)
- High `Stall_Math_Pipe_Throttle` → INT32 pipeline is the bottleneck (adding warps won't help)
- High `Stall_Not_Selected` → too many active warps competing; reduce occupancy
- `Selected` → issue overhead (1 cycle, baseline)

**Branch efficiency**:
- 100% = no divergence (all threads take same path)
- 52.5% (FF_add baseline) = massive divergence from conditional reduction

### Nsight Systems for Pipeline Visualization

```bash
nsys profile --trace=cuda,nvtx \
  --output=results/data/async_pipeline \
  ./ntt_profile --mode=async
nsys-ui results/data/async_pipeline.nsys-rep
```

In Nsight Systems UI:
- Look for H2D copy (purple) and Compute (green) rows in the CUDA timeline
- Overlap = compute row and copy row are active at the same time
- Take screenshot of the timeline as `results/screenshots/nsys_pipeline_overlap.png`

---

## 10. Implementation Walkthrough

### FpElement Type

```cuda
// include/ff_arithmetic.cuh
struct __align__(32) FpElement {
    uint32_t limbs[8];  // 255-bit value, 8 × 32-bit limbs, little-endian
    
    __device__ __forceinline__ 
    static FpElement zero() { FpElement r; memset(r.limbs, 0, sizeof(r.limbs)); return r; }
    
    __device__ __forceinline__ 
    static FpElement one() { /* = R mod p = Montgomery representation of 1 */ }
};

// BLS12-381 scalar modulus
__constant__ uint32_t BLS12_381_R[8] = {
    0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
};
```

### Montgomery Multiplication Kernel

```cuda
// Core inner loop — one limb iteration of CIOS MonMul
__device__ __forceinline__
uint32_t monmul_limb(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t* __restrict__ t,  // accumulator
    uint32_t b_i,              // current scalar limb
    const uint32_t* __restrict__ p  // modulus
) {
    uint32_t C = 0;
    // Compute t += a * b[i], propagate carry
    // Using PTX mad.lo.cc / madc.hi.cc for fused multiply-accumulate
    uint32_t lo, hi;
    asm volatile (
        "mad.lo.cc.u32  %0, %2, %3, %4; \n\t"
        "madc.hi.u32    %1, %2, %3,  0; \n\t"
        : "=r"(lo), "=r"(hi)
        : "r"(a[0]), "r"(b_i), "r"(t[0])
    );
    // ... continue for all 8 limbs
    return C;
}
```

### NTT Kernel Structure

```cuda
__global__ void ntt_butterfly_kernel(
    FpElement* __restrict__ data,
    const FpElement* __restrict__ twiddles,
    uint32_t n,
    uint32_t stage
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    
    // Determine butterfly pair indices
    uint32_t half_block = 1u << stage;
    uint32_t block_start = (tid / half_block) * (2 * half_block);
    uint32_t offset = tid % half_block;
    uint32_t i = block_start + offset;
    uint32_t j = i + half_block;
    
    // Load twiddle factor
    FpElement w = twiddles[half_block + offset];  // precomputed
    
    // Butterfly: t = w * data[j]; data[j] = data[i] - t; data[i] = data[i] + t
    FpElement t = ff_mul(data[j], w);
    data[j] = ff_sub(data[i], t);
    data[i] = ff_add(data[i], t);
}
```

### Async Pipeline Class

```cuda
// include/pipeline.cuh
class AsyncNTTPipeline {
    static constexpr int NUM_BUFFERS = 2;
    
    FpElement*    d_buf[NUM_BUFFERS];  // device double buffers
    FpElement*    h_buf[NUM_BUFFERS];  // pinned host buffers
    cudaStream_t  streams[NUM_BUFFERS];
    cudaEvent_t   events[NUM_BUFFERS];
    size_t        batch_size;
    
public:
    AsyncNTTPipeline(size_t n_per_batch);
    ~AsyncNTTPipeline();
    
    // Process multiple NTT batches with compute/transfer overlap
    void process(
        const FpElement* h_input,   // all batches, host memory
        FpElement*       h_output,  // all batches, host memory
        size_t           total_n,   // total elements
        size_t           ntt_size   // size of each NTT
    );
};
```

---

## Key References

1. **ZKProphet** (Verma et al., 2025): [arXiv:2509.22684](https://arxiv.org/abs/2509.22684)
   - The primary motivation paper. §IV-C (microarchitecture), §V-B (recommendations) are most relevant.

2. **cuZK** (Lu et al., TCHES 2023): [DOI:10.46586/tches.v2023.i3.194-220](https://doi.org/10.46586/tches.v2023.i3.194-220)
   - Multi-stream async pattern in §4.4 is the template for Direction A.

3. **Pippenger's Algorithm** (1976): Original MSM algorithm, background context for why we focus on NTT.

4. **Emmart et al.** (2016): "Optimizing modular multiplication for NVIDIA's Maxwell GPUs" — direct reference for IMAD→IADD3 conversion technique (cited in ZKProphet §IV-B2 as [17]).

5. **NVIDIA Nsight Compute Documentation**: Metric definitions and roofline methodology.

6. **BLS12-381 specification**: https://hackmd.io/@benjaminion/bls12-381

7. **Montgomery Multiplication**: Acar (1996), "The Montgomery Modular Inverse" — canonical reference for CIOS algorithm.
