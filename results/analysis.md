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

## Section 2 — Direction B Results (IADD3-Path Optimization)

### 2.1 Cycles per FF_mul Comparison

| Implementation | Cycles/op | vs Baseline |
|---|---|---|
| Naive CUDA (auto-vectorized) | TBD | 1.0× |
| CIOS with PTX mad.lo.cc | TBD | TBD |
| + Branchless conditional reduction | TBD | TBD |

**ZKProphet baseline**: 2656 cycles per FF_mul on A40.

**Target**: ≥10% cycle reduction. If latency is IMAD-dominated (4-cycle stalls), then reducing IMAD% in the conditional reduction path (which has no multiply) to IADD3 addresses the non-multiply portion of the computation.

### 2.2 Branch Efficiency Improvement

**Screenshot**: `screenshots/branch_efficiency_ff_ops.png`

| Operation | Before | After (branchless) | ZKProphet Baseline |
|---|---|---|---|
| FF_add | TBD | TBD | 52.5% |
| FF_sub | TBD | TBD | 56.2% |
| FF_mul | TBD | TBD | 84.0% |

**Key insight from ZKProphet Table VI**: FF_add/FF_sub branch divergence causes a 2.4× increase in execution cycles (72→244 cycles). Making reduction branchless should eliminate most of this overhead.

---

## Section 3 — NTT Kernel Analysis

### 3.1 Naive Radix-2 vs Radix-256 NTT

| Variant | Scale 2²⁰ | Scale 2²² | Scale 2²⁴ |
|---|---|---|---|
| Radix-2 (naive) | TBD ms | TBD ms | TBD ms |
| Radix-256 (shared mem) | TBD ms | TBD ms | TBD ms |
| Speedup | TBD× | TBD× | TBD× |

**Expected finding**: Radix-256 reduces global memory round-trips, improving L1 hit rate and reducing bandwidth pressure. Should be faster for all scales.

### 3.2 NTT Time Breakdown

For optimized radix-256 NTT at scale 2²²:
- Kernel compute time: TBD ms
- CPU→GPU transfer time: TBD ms
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

*(Partially populated — Phase 2 baseline complete, Phases 3-7 TBD)*

1. **FF_mul instruction mix**: ALU pipe 44.8% (integer-dominated, IMAD) — ZKProphet: 70.8% IMAD
2. **FF_mul throughput**: 2.9 Gops/s at 4M elements on RTX 3060
3. **FF_mul bottleneck**: DRAM at 91% (memory-bound due to 83% excessive sectors from AoS access)
4. **FF_add branch efficiency**: **100%** (ZKProphet: 52.5% — our nvcc generates predicated code)
5. **Achieved occupancy**: 83.9% (38 registers/thread)
6. **NTT transfer vs compute ratio**: TBD (ZKProphet: transfer >> compute)
7. **Pipeline latency improvement**: TBD× at scale 2²²
8. **Combined optimization vs bellperson**: TBD×

---

## Methodology Notes

- All GPU timings use `cudaEvent_t` start/stop (not CPU wall clock)
- Benchmark runs: 10 warm-up, 100 measured iterations, median reported
- Nsight Compute disables CUPTI sampling that interferes with timing — profile runs are separate from benchmark runs
- Profiling adds overhead; reported benchmark numbers come from unmodified release builds
