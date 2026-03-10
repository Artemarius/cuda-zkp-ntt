# Performance Analysis
## cuda-zkp-ntt — RTX 3060 (Ampere sm_86, CUDA 12.8)

*Populate this file as profiling runs are completed. Each section references a specific screenshot in `screenshots/` and connects findings to the ZKProphet paper analysis.*

---

## Hardware Reference

| Property | This System | ZKProphet (A40) |
|---|---|---|
| GPU | RTX 3060 | NVIDIA A40 |
| SMs | 28 | 84 |
| INT32/cycle/SM | 64 (4 SMSPs × 16) | 64 (same Ampere) |
| DRAM BW | ~360 GB/s | ~696 GB/s |
| VRAM | 6 GB | 48 GB |

Per-SM metrics (latency, stall breakdown, instruction mix) are **directly comparable**. Absolute throughput scales with SM count: RTX 3060 should be ~3× slower on total throughput but match on per-SM microarchitecture metrics.

---

## Section 1 — Finite Field Baseline (Direction B Motivation)

### 1.1 Roofline Analysis

**Screenshot**: `screenshots/roofline_ff_baseline.png`
**Method**: Nsight Compute roofline section, integer instruction weighting (IMAD = 2, other = 1)

| Kernel | Arithmetic Intensity | Performance | % of INT32 Ceiling |
|---|---|---|---|
| ff_add_kernel | TBD | TBD | TBD |
| ff_mul_kernel (naive) | TBD | TBD | TBD |
| ff_mul_kernel (CIOS) | TBD | TBD | TBD |

**ZKProphet reference** (A40): FF_mul reaches 60% of INT32 ceiling, FF_add reaches 40%.

**Expected finding**: Our RTX 3060 numbers should closely match, as roofline % is independent of SM count.

---

### 1.2 Warp Stall Breakdown (FF_mul Baseline)

**Screenshot**: `screenshots/warp_stalls_ff_mul.png`
**Method**: `collect_metrics.sh` → warp stall metrics for naive FF_mul kernel

| Stall Source | Cycles (RTX 3060) | ZKProphet A40 |
|---|---|---|
| Stall Wait | TBD | ~4 cycles |
| Selected | TBD | ~1 cycle |
| Stall Math Throttle | TBD | ~0.5 cycles (small at 2 warps/SMSP) |
| Stall Not Selected | TBD | ~0.7 cycles |
| Stall Other | TBD | small |
| **Total** | TBD | **~6.2 cycles** |

**Interpretation**: The dominant stall is `Stall_Wait` — the 4-cycle IMAD instruction latency. This is a fixed architectural constant; it cannot be reduced by adding more warps. The latency per FF_mul in cycles should be ~2660 (matching ZKProphet Table IV for GPU).

---

### 1.3 Instruction Mix

**Screenshot**: `screenshots/inst_mix_before_after.png`

| Kernel | IMAD % | IADD3 % | SHF % | Branch Eff. |
|---|---|---|---|---|
| ff_mul naive | TBD | TBD | TBD | TBD |
| ff_mul CIOS PTX | TBD | TBD | TBD | TBD |
| ff_add naive | TBD | TBD | TBD | TBD |
| ff_add branchless | TBD | TBD | TBD | TBD |

**ZKProphet reference**: FF_mul is 70.8% IMAD. FF_add branch efficiency = 52.5%.

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

*(Populate after profiling is complete)*

1. **FF_mul instruction mix**: TBD% IMAD (ZKProphet: 70.8%)
2. **Cycles per FF_mul**: TBD (ZKProphet: 2656 on A40)
3. **FF_add branch efficiency**: TBD% (ZKProphet: 52.5%)
4. **Branchless FF_add improvement**: TBDcycles → TBD cycles
5. **NTT transfer vs compute ratio**: TBD (ZKProphet: transfer >> compute)
6. **Pipeline latency improvement**: TBD× at scale 2²²
7. **Combined optimization vs bellperson**: TBD×

---

## Methodology Notes

- All GPU timings use `cudaEvent_t` start/stop (not CPU wall clock)
- Benchmark runs: 10 warm-up, 100 measured iterations, median reported
- Nsight Compute disables CUPTI sampling that interferes with timing — profile runs are separate from benchmark runs
- Profiling adds overhead; reported benchmark numbers come from unmodified release builds
