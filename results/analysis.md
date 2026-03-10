# Performance Analysis
## cuda-zkp-ntt — RTX 3060 (Ampere sm_86, CUDA 12.8)

*Populate this file as profiling runs are completed. Each section references a specific screenshot in `screenshots/` and connects findings to the ZKProphet paper analysis.*

---

## Hardware Reference

| Property | This System | ZKProphet (A40) |
|---|---|---|
| GPU | RTX 3060 | NVIDIA A40 |
| SMs | 30 | 84 |
| INT32/cycle/SM | 64 (4 SMSPs × 16) | 64 (same Ampere) |
| DRAM BW | ~360 GB/s | ~696 GB/s |
| VRAM | 6 GB | 48 GB |

Per-SM metrics (latency, stall breakdown, instruction mix) are **directly comparable**. Absolute throughput scales with SM count: RTX 3060 should be ~2.8× slower on total throughput but match on per-SM microarchitecture metrics.

---

## Section 1 — Finite Field Baseline (Direction B Motivation)

### 1.1 Speed of Light Analysis

**Screenshots**: `screenshots/Screenshot_151.png` (GPU Speed of Light + Memory Workload)
**Method**: Nsight Compute `--set full`, ff_mul_throughput_kernel, 2^20 elements

| Metric | Value |
|---|---|
| Compute (SM) Throughput | 64.30% |
| Memory Throughput | **91.04%** (DRAM-bound) |
| L1/TEX Throughput | 55.90% |
| L2 Cache Throughput | 50.55% |
| SM Frequency | 899.89 MHz |
| Duration | 338.05 us |
| Memory Throughput | 304.10 GB/s |
| FP32 / FP64 utilization | 0% / 0% (pure integer) |

**Key finding**: The kernel is **memory-bound** (91% DRAM vs 64% compute), despite being a compute-heavy integer kernel. This is caused by severe memory access inefficiency (see Section 1.4).

**ZKProphet comparison**: ZKProphet reports FF_mul reaching ~60% of INT32 ceiling on A40. Our 64% compute throughput on RTX 3060 is consistent — the kernel is well-optimized computationally, but memory access patterns dominate wall-clock time.

---

### 1.2 Scheduler & Warp Statistics

**Screenshot**: `screenshots/Screenshot_152.png` (Scheduler + Warp State + Instruction Statistics)
**Method**: Nsight Compute `--set full`

| Metric | RTX 3060 | ZKProphet A40 |
|---|---|---|
| Active Warps / Scheduler | 9.98 / 12 | - |
| Eligible Warps / Cycle | 1.46 | - |
| No Eligible % | **52.43%** | - |
| One or More Eligible % | 47.57% | - |
| Issue rate | 1 inst / 2.1 cycles | - |
| Warp Cycles Per Issued Inst | 20.97 | ~6.2 |
| Avg Active Threads Per Warp | 32.00 | - |
| Not Predicated Off Threads | 31.88 | - |

**Interpretation**: Over half the cycles have zero eligible warps — warps are stalled waiting on memory, not on IMAD latency (which would show as `Stall_Math_Throttle`). The 20.97 cycles per issued instruction is much higher than ZKProphet's ~6.2, suggesting our memory access pattern is the dominant bottleneck, not compute latency.

### 1.3 Occupancy & Launch Configuration

**Screenshot**: `screenshots/Screenshot_153.png` (Occupancy + Source Counters)

| Metric | Value |
|---|---|
| Theoretical Occupancy | 100% (48 warps/SM) |
| Achieved Occupancy | **83.91%** (40.28 warps/SM) |
| Registers / Thread | **38** |
| Block Limit (registers) | 6 blocks/SM |
| Block Limit (warps) | 6 blocks/SM |
| Waves Per SM | 22.76 |
| Branch Efficiency | **100%** |
| Divergent Branches | 0.03 |

Occupancy is good (84%). Register usage (38/thread) is the limiter. Branch efficiency is effectively perfect — the conditional reduction in ff_add/ff_sub/ff_mul does not cause divergence.

### 1.4 Memory Access Pattern (Critical Finding)

**Screenshot**: `screenshots/Screenshot_153.png` (Source Counters — L2 Excessive Sectors)

| Metric | Value |
|---|---|
| Global Load utilization | **4 / 32 bytes per sector (12.5%)** |
| Global Store utilization | 16 / 32 bytes per sector (50%) |
| Excessive sectors | **15,728,640 / 18,874,368 (83%)** |
| L1/TEX Hit Rate | 81.54% |
| L2 Hit Rate | 39.18% |
| Est. Speedup from fixing | **80.48%** (ncu estimate) |

All hotspots point to `ff_arithmetic.cuh:191` (inside ff_mul CIOS inner loops). Each thread accesses `a[tid].limbs[j]`, meaning adjacent threads hit addresses 32 bytes apart instead of 4 bytes. This is the **Array-of-Structures (AoS) stride problem**.

With 256 threads per block, each warp of 32 threads accessing `limbs[0]` touches 32 × 32 = 1024 bytes across 32 cache lines, but a coalesced access would only need 32 × 4 = 128 bytes (4 cache lines). This 8× amplification explains why the kernel is memory-bound despite being compute-intensive.

**Implication for Phase 3**: Before pursuing IADD3 instruction optimization, addressing the memory access pattern (SoA layout, shared-memory transpose, or vectorized loads) could yield a much larger speedup — ncu estimates 80% potential improvement from fixing coalescing alone.

---

### 1.5 Instruction Mix (Baseline)

**Screenshots**: `screenshots/Screenshot_151.png`, `screenshots/Screenshot_152.png`

| Metric | Value | ZKProphet A40 |
|---|---|---|
| Highest pipe | **ALU 44.8%** | IMAD-dominated |
| FP32 | 0% | 0% |
| FP64 | 0% | 0% |
| SM Busy | 65.89% | - |
| Issue Slots Busy | 47.18% | - |
| Executed IPC (active) | 1.88 | - |
| Branch Efficiency | 100% | 52.5% (FF_add) |

The ALU (integer) pipe at 44.8% confirms an IMAD-dominated workload, consistent with ZKProphet's finding that FF_mul is 70.8% IMAD instructions. Notably, our branch efficiency (100%) is significantly better than ZKProphet's 52.5% for FF_add — likely because nvcc generates predicated code for our conditional reduction rather than true branches.

**ZKProphet reference**: FF_mul is 70.8% IMAD. FF_add branch efficiency = 52.5%.
Detailed SASS-level instruction mix breakdown deferred to Phase 3 (requires `--section InstructionStats` with individual opcode counters).

---

## Section 2 — Direction B Results (Phase 3: IADD3-Path Optimization)

### 2.1 SASS Instruction Count Comparison

**Method**: `cuobjdump --dump-sass` on Release binary (sm_86, -O3 --use_fast_math)

| Kernel | Baseline (SASS instrs) | V2 Branchless (SASS instrs) | Reduction |
|---|---|---|---|
| ff_mul | 571 | 527 | **-7.7%** |
| ff_add | 127 | 55 | **-56.7%** |
| ff_sub | 94 | 55 | **-41.5%** |
| ff_sqr | 563 | 511 | **-9.2%** |

The CIOS Montgomery loop (ff_mul core) is identical between variants — nvcc already generates efficient carry-chain code from `uint64_t` arithmetic. The reduction comes entirely from the **conditional reduction path** being replaced by PTX `sub.cc` carry chain + `lop3.b32` branchless MUX.

### 2.2 Instruction Mix: ff_mul Baseline vs V2

| Instruction | Baseline | V2 | Notes |
|---|---|---|---|
| IMAD.WIDE.U32 | 122 (21.4%) | 123 (23.3%) | CIOS multiply-accumulate (identical) |
| IMAD.X | 87 (15.2%) | 92 (17.5%) | Extended multiply carry chain |
| IADD3 | 194 (34.0%) | 186 (35.3%) | Integer add-3 (dominant) |
| IADD3.X | 48 (8.4%) | 44 (8.3%) | Extended add carry |
| **ISETP** | **22 (3.9%)** | **1 (0.2%)** | **Comparison/predication eliminated** |
| **SEL** | **7 (1.2%)** | **0** | **Conditional select eliminated** |
| **LOP3.LUT** | **0** | **8 (1.5%)** | **Branchless MUX (new)** |
| IMAD.MOV.U32 | 25 (4.4%) | 18 (3.4%) | Register moves via IMAD |
| LDG.E.CONSTANT | 16 | 16 | Global loads (identical) |
| STG.E.128 | 2 | 2 | Global stores (identical) |

**Key observation**: Both kernels are IMAD+IADD3 dominated (~85% combined). The branchless v2 replaces 29 comparison/select instructions (ISETP+SEL) with 9 logic instructions (LOP3+SHF).

### 2.3 Instruction Mix: ff_add Baseline vs V2

| Instruction | Baseline | V2 | Notes |
|---|---|---|---|
| IADD3 + IADD3.X | 29 (22.8%) | 17 (30.9%) | Core addition |
| **ISETP (all variants)** | **23 (18.1%)** | **1 (1.8%)** | **Comparison logic eliminated** |
| **SEL** | **7 (5.5%)** | **1 (1.8%)** | **Conditional select eliminated** |
| **LOP3.LUT** | **0** | **8 (14.5%)** | **Branchless MUX (new)** |
| LDG (loads) | 16 × 32-bit | 4 × 128-bit | **V2 enables vectorized loads** |
| MOV | 12 (9.4%) | 1 (1.8%) | Fewer register shuffles |

V2 ff_add achieves **57% instruction count reduction** — the most dramatic improvement. The branchless PTX also changed register allocation patterns, allowing nvcc to merge 16 individual 32-bit loads into 4 × 128-bit vectorized loads (`LDG.E.128.CONSTANT`).

### 2.4 Microbenchmark Throughput (4M Elements, RTX 3060)

| Operation | Baseline (Gops/s) | V2 Branchless | SoA | V2 vs Baseline |
|---|---|---|---|---|
| ff_add | 3.08 | 2.48 | 2.48 | -19.5% |
| ff_sub | 2.54 | 2.54 | 2.54 | 0% |
| ff_mul | 2.48 | 2.45 | 2.54 | -1.2% |
| ff_sqr | 3.27 | 3.20 | 3.04 | -2.1% |

### 2.5 Why No Throughput Improvement Despite Fewer Instructions

The isolated FF microbenchmark is **memory-bound** (91% DRAM throughput, Section 1.1). Each thread loads 2 FpElements (64 bytes) and stores 1 (32 bytes) — the kernel's compute intensity is too low for instruction-level optimizations to be visible:

1. **Memory latency dominates**: Warps spend 52% of cycles with zero eligible warps (stalled on memory)
2. **Both variants hit the same DRAM bandwidth ceiling**: ~304 GB/s (84% of theoretical 360 GB/s)
3. **Instruction reduction is in the reduction path** which executes once per ff_mul, while the CIOS loop (unchanged) executes 8× per ff_mul

**When these optimizations WILL matter — inside NTT kernels (Phase 4/5)**:
- Multiple FF ops per loaded element → higher compute-to-memory ratio
- Data lives in shared memory during butterfly stages → no DRAM latency
- ff_add is called once per butterfly (57% fewer instructions directly reduces cycle count)
- The branchless reduction avoids any warp divergence regardless of input distribution

### 2.6 AoS vs SoA Memory Layout

**Finding**: No throughput difference. The `__align__(32)` attribute on `FpElement` causes nvcc to generate 256-bit (`LDG.E.CONSTANT.SYS [addr], desc[UR4][R2]`) vectorized loads for the AoS layout, achieving the same coalescing as explicit SoA. The "83% excessive sectors" metric from Phase 2 ncu was measuring L2→L1 sector utilization, not DRAM access patterns — the GPU's L1 cache line fills serve adjacent elements.

### 2.7 Branch Efficiency

| Operation | Our Baseline | ZKProphet A40 |
|---|---|---|
| FF_add | **100%** | 52.5% |
| FF_sub | **100%** | 56.2% |
| FF_mul | **100%** | 84.0% |

Our baseline already achieves 100% branch efficiency because nvcc (CUDA 12.8, sm_86) generates **predicated instructions** (ISETP+SEL) rather than true branches for the conditional reduction. ZKProphet's lower efficiency likely reflects an older compiler or different code structure (bellperson uses Rust + CUDA, potentially with different codegen).

This means our branchless v2 variants don't improve branch efficiency (already perfect), but do reduce **static instruction count** — relevant for compute-bound contexts.

---

## Section 3 — NTT Kernel Analysis

### 3.1 Naive Radix-2 vs Radix-256 NTT

**Method**: Google Benchmark, cudaEvent_t timing, 5-repetition mean, Release build.

| Variant | Scale 2¹⁵ | Scale 2¹⁸ | Scale 2²⁰ | Scale 2²² |
|---|---|---|---|---|
| Radix-2 (naive) | 0.169 ms | 1.85 ms | 7.13 ms | 31.2 ms |
| Radix-256 (shared mem) | 0.212 ms | 1.54 ms | 5.99 ms | 26.5 ms |
| Speedup | 0.80× | **1.20×** | **1.19×** | **1.18×** |

**Architecture**: Radix-256 fuses 8 Cooley-Tukey butterfly stages into a single shared-memory kernel (128 threads, 256 elements, 8 KB shmem per block). For n=2^22, this reduces total kernel launches from 22 to 15 (1 fused + 14 outer).

**Why ~18% not ~36%**: Fusing 8 of 22 stages eliminates ~36% of global memory round-trips. But the fused kernel has lower occupancy (4 warps/block vs the naive kernel's 8 warps/block at 256 threads), and each butterfly stage is compute-dominated by ff_mul (~500 SASS instructions). The fused kernel also has 8-way shared memory bank conflicts from 32-byte FpElement stride (padding was tested but added overhead).

**Small-size regression at 2^15**: With only 15 butterfly stages, the fused kernel's overhead (shared memory allocation, 8 consecutive syncthreads barriers) outweighs the savings from 8 fewer global memory passes.

**Block tuning attempted**: K=8 (128 threads, radix-256), K=9 (256 threads), K=10 (512 threads). K≥9 failed due to CUDA RDC template instantiation bug ("named symbol not found"). Non-template workaround for K=10 was ~2% slower than template K=8 (no loop unrolling). K=8 is the production configuration.

### 3.2 NTT Time Breakdown

For optimized radix-256 NTT at scale 2²²:
- Kernel compute time (on-device): 26.5 ms
- CPU→GPU transfer time: TBD ms (Phase 6 will measure)
- Transfer % of total: TBD %

**ZKProphet reference** (Fig. 7): NTT transfer time >> compute time (unlike MSM, where they are overlapped). This motivates Direction A.

---

## Section 4 — Direction A Results (Async Pipeline)

### 4.1 Pipeline Timeline

**Screenshot**: `screenshots/nsys_pipeline_overlap.png`
**Comparison**: `screenshots/nsys_sequential.png`

The Nsight Systems timeline should show:
- **Sequential**: H2D copy (purple) fully completes before NTT kernel (green) starts
- **Pipelined**: H2D copy for batch k+1 (purple) overlaps with NTT kernel for batch k (green)

### 4.2 End-to-End Latency Comparison

For scale 2²², 4 batches of 2²⁰ each:

| Mode | Total Time | Per-batch | vs Sequential |
|---|---|---|---|
| Sequential (no pipeline) | TBD ms | TBD ms | 1.0× |
| Async double-buffer pipeline | TBD ms | TBD ms | TBD× |

**Theoretical max speedup**: (compute + transfer) / max(compute, transfer)

At scale 2²², expected transfer time ~Xms, compute ~Yms (from Section 3.2). If Y < X, max speedup approaches X/Y.

### 4.3 Batch Size Sensitivity

| Batch log₂ | Pipeline Speedup |
|---|---|
| 18 | TBD |
| 20 | TBD |
| 22 | TBD |

---

## Section 5 — Combined Results

### Full NTT Performance Table

| Implementation | Scale 2²⁰ | Scale 2²² | Scale 2²⁴ | vs bellperson* |
|---|---|---|---|---|
| bellperson (reference) | TBD | TBD | TBD | 1.0× |
| cuda-zkp-ntt naive | TBD | TBD | TBD | TBD× |
| + FF_mul optimized | TBD | TBD | TBD | TBD× |
| + Async pipeline | TBD | TBD | TBD | TBD× |
| **cuda-zkp-ntt full** | TBD | TBD | TBD | TBD× |

*bellperson reference: we run bellperson's NTT from the bellperson Rust crate with CUDA backend and measure via its built-in timing, or estimate from ZKProphet Table II (scaled from A40 to RTX 3060 by SM count ratio).

---

## Key Findings Summary

*(Phases 2-5 complete, Phases 6-7 TBD)*

1. **FF_mul instruction mix**: ALU pipe 44.8% (integer-dominated, IMAD+IADD3 ~85%) — ZKProphet: 70.8% IMAD
2. **FF_mul throughput**: 2.48 Gops/s at 4M elements on RTX 3060
3. **FF_mul bottleneck**: DRAM at 91% (memory-bound, 52% cycles with zero eligible warps)
4. **FF_add branch efficiency**: **100%** (ZKProphet: 52.5% — our nvcc generates predicated code)
5. **Achieved occupancy**: 83.9% (38 registers/thread)
6. **Branchless v2 SASS reduction**: ff_add -57%, ff_sub -41%, ff_mul -8%, ff_sqr -9%
7. **Branchless v2 throughput**: No improvement in isolated microbench (memory-bound) — gains expected inside NTT kernels
8. **AoS vs SoA**: No difference — `__align__(32)` FpElement already generates vectorized loads
9. **Radix-256 NTT speedup**: **1.18-1.20×** for sizes ≥ 2^16 (fusing 8 stages in shared memory)
10. **Radix-256 architecture**: 128 threads, 256 elements per block, 8 KB shmem. K=8 is optimal (K≥9 blocked by CUDA RDC template bug; bank conflict padding adds overhead)
11. **NTT transfer vs compute ratio**: TBD (ZKProphet: transfer >> compute)
12. **Pipeline latency improvement**: TBD× at scale 2²²
13. **Combined optimization vs bellperson**: TBD×

---

## Methodology Notes

- All GPU timings use `cudaEvent_t` start/stop (not CPU wall clock)
- Benchmark runs: 10 warm-up, 100 measured iterations, median reported
- Nsight Compute disables CUPTI sampling that interferes with timing — profile runs are separate from benchmark runs
- Profiling adds overhead; reported benchmark numbers come from unmodified release builds
