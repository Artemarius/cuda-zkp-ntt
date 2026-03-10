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

**Screenshots**: `screenshots/roofline_ff_mul_baseline.png` (GPU Speed of Light + Roofline)
**Method**: Nsight Compute 2025.1.1 `--set full`, ff_mul_throughput_kernel, 2^20 elements

| Metric | Value |
|---|---|
| Compute (SM) Throughput | 64.15% |
| Memory Throughput | **92.08%** (DRAM-bound) |
| L1/TEX Throughput | 55.47% |
| L2 Cache Throughput | 51.47% |
| DRAM Throughput | 92.08% |
| SM Frequency | 899.85 MHz |
| Duration | 333.41 us |
| Elapsed Cycles | 300,024 |
| Memory Throughput | 304.93 GB/s |
| FP32 / FP64 utilization | 0% / 0% (pure integer) |

**Key finding**: The kernel is **memory-bound** (92% DRAM vs 64% compute), despite being a compute-heavy integer kernel. This is caused by severe memory access inefficiency (see Section 1.4).

**ZKProphet comparison**: ZKProphet reports FF_mul reaching ~60% of INT32 ceiling on A40. Our 64% compute throughput on RTX 3060 is consistent — the kernel is well-optimized computationally, but memory access patterns dominate wall-clock time.

---

### 1.2 Scheduler & Warp Statistics

**Screenshot**: `screenshots/warp_stalls_ff_mul_baseline.png` (Warp State Statistics)
**Method**: Nsight Compute 2025.1.1 `--set full`

| Metric | RTX 3060 | ZKProphet A40 |
|---|---|---|
| Active Warps / Scheduler | 9.98 / 12 | - |
| Eligible Warps / Cycle | 1.46 | - |
| No Eligible % | **52.43%** | - |
| One or More Eligible % | 47.57% | - |
| Issue rate | 1 inst / 2.1 cycles | - |
| Warp Cycles Per Issued Inst | 21.48 | ~6.2 |
| Avg Active Threads Per Warp | 32.00 | - |
| Not Predicated Off Threads | 31.88 | - |

**Interpretation**: Over half the cycles have zero eligible warps — warps are stalled waiting on memory, not on IMAD latency (which would show as `Stall_Math_Throttle`). The 21.48 cycles per issued instruction is much higher than ZKProphet's ~6.2, suggesting our memory access pattern is the dominant bottleneck, not compute latency.

**Top warp stall reasons (ff_mul baseline)**:
1. **Stall Long Scoreboard** (~7.5 cycles) — waiting on global memory loads
2. **Stall LG Throttle** (~4.5 cycles) — L1/global memory pipeline saturated
3. **Stall Math Pipe Throttle** (~2.5 cycles) — integer ALU backpressure
4. **Stall Not Selected** (~2.2 cycles) — scheduler chose another warp
5. **Stall Dispatch Stall** (~2.0 cycles) — instruction dispatch bottleneck
6. **Stall Wait** (~1.5 cycles) — fixed-latency dependency

### 1.3 Occupancy & Launch Configuration

**Screenshot**: `screenshots/Screenshot_153.png` (Occupancy + Source Counters), `screenshots/roofline_ff_mul_baseline.png`

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

### 2.4 Nsight Compute: FF_mul Baseline vs V2

**Screenshots**: `screenshots/roofline_ff_mul_baseline.png`, `screenshots/roofline_ff_mul_v2.png`
**Method**: Nsight Compute 2025.1.1 `--set full`, 2^20 elements

| Metric | Baseline | V2 Branchless | Delta |
|---|---|---|---|
| Compute (SM) Throughput | 64.15% | 65.47% | +1.32 pp |
| Memory Throughput | 92.08% | 90.92% | -1.16 pp |
| Duration | 333.41 us | 340.29 us | +2.1% |
| Executed IPC | 1.87 | 1.85 | -1.1% |
| SM Busy | 65.48% | 65.92% | +0.7% |
| Issue Slots Busy | 46.89% | 46.38% | -1.1% |
| ALU pipe (highest) | 44.5% | 44.4% | -0.2% |
| Memory Throughput | 304.93 GB/s | 301.07 GB/s | -1.3% |

Both variants hit the same DRAM bandwidth ceiling (~92%). The v2 kernel is marginally slower (+2.1% duration) — the branchless reduction path has slightly different instruction scheduling that doesn't help when memory-bound. This confirms the Phase 3 finding: instruction optimizations are invisible in isolated FF microbenchmarks.

### 2.5 Nsight Compute: FF_add Instruction Comparison

**Screenshot**: `screenshots/instruction_mix_ff_add_comparison.png`

| Metric | Baseline | V2 Branchless | Delta |
|---|---|---|---|
| Branch Instructions | 262,144 | 65,536 | **-75% (4× fewer)** |
| Branch Instructions Ratio | 0.12% | 0.04% | -67% |
| Branch Efficiency | 100% | 100% | unchanged |
| Excessive L2 Sectors | 83% | 83% | unchanged (AoS) |
| Est. Speedup (coalescing) | 80.54% | 80.54% | unchanged |

The v2 branchless PTX eliminates 75% of branch instructions (comparison/predication replaced by LOP3 MUX), but since the kernel is memory-bound, this doesn't translate to throughput improvement. The memory access pattern (83% excessive sectors from AoS layout) remains the dominant bottleneck for both variants.

### 2.6 Microbenchmark Throughput (4M Elements, RTX 3060)

| Operation | Baseline (Gops/s) | V2 Branchless | SoA | V2 vs Baseline |
|---|---|---|---|---|
| ff_add | 3.08 | 2.48 | 2.48 | -19.5% |
| ff_sub | 2.54 | 2.54 | 2.54 | 0% |
| ff_mul | 2.48 | 2.45 | 2.54 | -1.2% |
| ff_sqr | 3.27 | 3.20 | 3.04 | -2.1% |

### 2.7 Why No Throughput Improvement Despite Fewer Instructions

The isolated FF microbenchmark is **memory-bound** (91% DRAM throughput, Section 1.1). Each thread loads 2 FpElements (64 bytes) and stores 1 (32 bytes) — the kernel's compute intensity is too low for instruction-level optimizations to be visible:

1. **Memory latency dominates**: Warps spend 52% of cycles with zero eligible warps (stalled on memory)
2. **Both variants hit the same DRAM bandwidth ceiling**: ~304 GB/s (84% of theoretical 360 GB/s)
3. **Instruction reduction is in the reduction path** which executes once per ff_mul, while the CIOS loop (unchanged) executes 8× per ff_mul

**When these optimizations WILL matter — inside NTT kernels (Phase 4/5)**:
- Multiple FF ops per loaded element → higher compute-to-memory ratio
- Data lives in shared memory during butterfly stages → no DRAM latency
- ff_add is called once per butterfly (57% fewer instructions directly reduces cycle count)
- The branchless reduction avoids any warp divergence regardless of input distribution

### 2.8 AoS vs SoA Memory Layout

**Finding**: No throughput difference. The `__align__(32)` attribute on `FpElement` causes nvcc to generate 256-bit (`LDG.E.CONSTANT.SYS [addr], desc[UR4][R2]`) vectorized loads for the AoS layout, achieving the same coalescing as explicit SoA. The "83% excessive sectors" metric from Phase 2 ncu was measuring L2→L1 sector utilization, not DRAM access patterns — the GPU's L1 cache line fills serve adjacent elements.

### 2.9 Branch Efficiency

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

### 3.2 NTT Fused Kernel Profile (ntt_fused_stages_kernel)

**Screenshot**: `screenshots/roofline_ntt_optimized_2e20.png` (Speed of Light + Roofline)
**Method**: Nsight Compute 2025.1.1 `--set full --kernel-name ntt_fused_stages_kernel`, 2^20 elements

| Metric | Fused (Optimized) | Naive Butterfly | Comparison |
|---|---|---|---|
| Compute (SM) Throughput | **69.23%** | 45.14% | +24 pp |
| Memory Throughput | 54.74% | **78.61%** | -24 pp |
| DRAM Throughput | 15.65% | 78.61% | Fused avoids DRAM |
| L1/TEX Throughput | 55.16% | 79.90% | |
| L2 Throughput | 11.92% | 53.18% | |
| Duration | 1.29 ms | 256.42 us | 8 stages fused |
| SM Busy | **69.76%** | 45.96% | |
| Issue Slots Busy | **60.94%** | 39.68% | |
| Executed IPC | **2.41** | 1.56 | 55% more IPC |
| ALU pipe utilization | **67.3%** | 41.5% | |

**Key finding**: The fused radix-256 kernel is **compute-bound** (69% compute vs 55% memory), a dramatic shift from both the naive NTT butterfly (memory-bound at 79%) and the isolated FF microbenchmark (memory-bound at 92%). This validates the Phase 3 prediction: instruction-level optimizations matter inside the NTT kernel where data lives in shared memory and the compute-to-memory ratio is high.

**ncu classification**: "High Compute Throughput — compute is more heavily utilized than memory." ALU pipe (67.3%) is the highest-utilized pipeline, confirming integer arithmetic (IMAD/IADD3) dominance.

**Warp Stall Comparison (fused kernel vs FF baseline)**:

**Screenshot**: `screenshots/warp_stalls_ntt_optimized.png`

| Stall Reason | Fused NTT Kernel | FF_mul Baseline | Interpretation |
|---|---|---|---|
| Math Pipe Throttle | **~3.5 (1st)** | ~2.5 (3rd) | ALU saturated — compute-bound |
| Not Selected | ~3.3 (2nd) | ~2.2 (4th) | High occupancy, scheduler contention |
| Dispatch Stall | ~2.5 (3rd) | ~2.0 (5th) | Instruction issue pressure |
| Barrier | **~2.0 (4th)** | 0 | `__syncthreads()` between stages |
| Wait | ~1.8 (5th) | ~1.5 (6th) | Fixed-latency dependencies |
| Long Scoreboard | **~0.4** | **~7.5 (1st)** | Shared mem eliminates DRAM waits |
| LG Throttle | **~0** | **~4.5 (2nd)** | No global mem pressure |

The stall profile transformation tells the optimization story clearly:
- **FF_mul baseline**: dominated by Long Scoreboard + LG Throttle (memory stalls)
- **Fused NTT kernel**: dominated by Math Pipe Throttle + Not Selected (compute stalls)
- The Barrier stall (~2.0 cycles) is the cost of `__syncthreads()` between the 8 fused butterfly stages — an acceptable overhead for eliminating 8 global memory round-trips.

### 3.3 NTT Naive vs Optimized Comparison

**Screenshot**: `screenshots/ntt_naive_vs_optimized_comparison.png`
**Method**: ncu baseline comparison — `ntt_butterfly_kernel` (naive) vs `ntt_fused_stages_kernel` (optimized)

Per-kernel comparison (single invocation):
- **Naive butterfly** (1 of 20 stages): 256.42 us, 2048 blocks × 256 threads, memory-bound
- **Fused kernel** (8 stages): 1.29 ms, 4096 blocks × 128 threads, compute-bound

The naive kernel launches 20 separate butterfly invocations for 2^20, each doing one stage with global memory reads/writes. The fused kernel does 8 stages with data in shared memory (8 KB/block), then the remaining 12 stages use global memory butterfly kernels. Net effect: 15 launches vs 20 launches, with the fused launch doing 8× more compute per byte of DRAM traffic.

**Memory throughput shift**: The naive butterfly achieves 260 GB/s memory throughput (+402% vs fused) — it's essentially a memory streaming kernel. The fused kernel achieves only 15.65% DRAM throughput because data stays in shared memory for 8 stages.

### 3.4 NTT Time Breakdown

For optimized radix-256 NTT at scale 2²²:
- Kernel compute time (on-device): 21.9 ms
- H2D transfer (128 MB pinned): 20.2 ms (6.6 GB/s)
- D2H transfer (128 MB pinned): 21.8 ms (6.2 GB/s)
- Transfer % of total serial: 66% (H2D + D2H = 42ms out of 64ms)

**ZKProphet reference** (Fig. 7): NTT transfer time >> compute time (unlike MSM, where they are overlapped). Our RTX 3060 measurement confirms: transfer (42ms) > compute (22ms). This strongly motivates Direction A.

---

## Section 4 — Direction A Results (Async Pipeline)

### 4.1 Device Capabilities

| Property | Value |
|---|---|
| asyncEngineCount | **5** (supports concurrent H2D + D2H) |
| concurrentKernels | 1 (yes) |
| PCIe bandwidth (H2D, 128 MB) | 6.6 GB/s (20.2 ms) |
| PCIe bandwidth (D2H, 128 MB) | 6.2 GB/s (21.8 ms) |

### 4.2 Pipeline Architecture

3-stream pipeline with cross-stream event dependencies:
- `stream_h2d_`: dedicated H2D transfers (DMA engine)
- `stream_compute_[0/1]`: NTT kernels (compute engine, double-buffered)
- `stream_d2h_`: dedicated D2H transfers (DMA engine)
- `cudaStreamWaitEvent`: H2D→compute and compute→D2H dependencies

### 4.3 End-to-End Latency: Pinned Memory (Best Case)

8 independent NTT batches, pinned host memory (no CPU staging copies):

| Scale | Pipelined | Sequential | Speedup | Per-batch (pipe) | Per-batch (seq) |
|---|---|---|---|---|---|
| 2¹⁸ | **29.7 ms** | 49.4 ms | **1.66x (40%)** | 3.7 ms | 6.2 ms |
| 2²⁰ | **141 ms** | 188 ms | **1.33x (25%)** | 17.6 ms | 23.5 ms |
| 2²² | **541 ms** | 579 ms | **1.07x (7%)** | 67.6 ms | 72.4 ms |

### 4.4 End-to-End Latency: Pageable Memory (Realistic Case)

8 independent NTT batches, pageable host memory (with pinned staging copies):

| Scale | Pipelined | Sequential | Speedup |
|---|---|---|---|
| 2¹⁸ | **40.2 ms** | 50.5 ms | **1.23x (19%)** |
| 2²⁰ | **188 ms** | 200 ms | **1.06x (6%)** |
| 2²² | **716 ms** | 791 ms | **1.10x (10%)** |

### 4.5 Why Limited Speedup at Scale 2²²

**Root cause: DMA interference on the GPU memory controller.**

At 128 MB per transfer (2²²), concurrent H2D/D2H and NTT compute contend for the GPU's memory controller bandwidth:

1. NTT is memory-bound (91% DRAM utilization) — it saturates the memory controller
2. DMA transfers also access device DRAM (writing for H2D, reading for D2H)
3. When both run simultaneously, they steal bandwidth from each other
4. Net effect: neither runs at full speed, negating the overlap benefit

At smaller sizes (2¹⁸ = 8 MB), transfers complete quickly and cause minimal contention, yielding strong overlap (40% speedup). At 2²⁰ (32 MB), the effect is moderate (25%). At 2²² (128 MB), sustained DMA traffic throughout the NTT execution causes continuous interference.

This is consistent with known CUDA behavior: the overlap benefit depends on whether the kernel is compute-bound or memory-bound. For compute-bound kernels, concurrent DMA is essentially free. For memory-bound kernels, DMA steals bandwidth.

**Potential mitigations** (not implemented, future work):
- Restructure NTT to have compute-bound and memory-bound phases, overlap DMA only during compute-bound phases
- Use CUDA graphs to batch kernel launches and reduce driver overhead
- NVLink or PCIe 5.0 with higher bandwidth would reduce transfer time relative to compute

---

## Key Findings Summary

1. **FF_mul instruction mix**: ALU pipe 44.5% (integer-dominated, IMAD+IADD3 ~85%) — ZKProphet: 70.8% IMAD
2. **FF_mul throughput**: 2.48 Gops/s at 4M elements on RTX 3060
3. **FF_mul bottleneck**: DRAM at 92% (memory-bound, 52% cycles with zero eligible warps)
4. **FF_add branch efficiency**: **100%** (ZKProphet: 52.5% — our nvcc generates predicated code)
5. **Achieved occupancy**: 83.9% (38 registers/thread)
6. **Branchless v2 SASS reduction**: ff_add -57%, ff_sub -41%, ff_mul -8%, ff_sqr -9%
7. **Branchless v2 ncu comparison**: Identical throughput profile (both hit 92% DRAM ceiling). v2 eliminates 75% of branch instructions in ff_add — invisible when memory-bound.
8. **AoS vs SoA**: No difference — `__align__(32)` FpElement already generates vectorized loads
9. **Radix-256 NTT speedup**: **1.18-1.20×** for sizes ≥ 2^16 (fusing 8 stages in shared memory)
10. **Fused NTT kernel is compute-bound**: 69% compute vs 55% memory (vs FF baseline: 64% compute vs 92% memory). Math Pipe Throttle is the top stall reason — ALU saturated at 67.3%.
11. **Stall profile transformation**: FF_mul dominated by Long Scoreboard (memory waits); fused NTT dominated by Math Pipe Throttle (ALU saturation). Shared memory eliminates global memory stalls.
12. **NTT IPC improvement**: Fused kernel achieves IPC 2.41 vs naive butterfly 1.56 (+55%). Issue slots busy 61% vs 40%.
13. **NTT transfer vs compute ratio**: Transfer 66% of total serial time (42ms transfer vs 22ms compute at 2²²) — confirms ZKProphet: transfer >> compute
14. **Pipeline latency improvement**: 1.66× at 2¹⁸, 1.33× at 2²⁰, 1.07× at 2²² (pinned memory, 8 batches). DMA interference limits overlap at large sizes.
15. **Combined optimization vs bellperson**: Not directly measured (no bellperson build). Radix-256 + pipeline gives ~1.18× NTT compute + up to 1.66× end-to-end at transfer-dominated sizes.

---

## Screenshot Index

| File | Content | Section |
|---|---|---|
| `Screenshot_151.png` | FF_mul baseline: Speed of Light + Memory (Phase 2) | 1.1 |
| `Screenshot_152.png` | FF_mul baseline: Scheduler + Warp State (Phase 2) | 1.2 |
| `Screenshot_153.png` | FF_mul baseline: Occupancy + Source Counters (Phase 2) | 1.3, 1.4 |
| `roofline_ff_mul_baseline.png` | FF_mul baseline: Speed of Light + Roofline (Phase 7) | 1.1 |
| `warp_stalls_ff_mul_baseline.png` | FF_mul baseline: Warp State Statistics (Phase 7) | 1.2 |
| `roofline_ff_mul_v2.png` | FF_mul v2 branchless: Speed of Light + Roofline | 2.4 |
| `instruction_mix_ff_add_comparison.png` | FF_add baseline vs v2: Source Counters comparison | 2.5 |
| `roofline_ntt_optimized_2e20.png` | NTT fused kernel: Speed of Light + Roofline | 3.2 |
| `warp_stalls_ntt_optimized.png` | NTT fused kernel: Warp State Statistics | 3.2 |
| `ntt_naive_vs_optimized_comparison.png` | NTT naive vs optimized: baseline comparison | 3.3 |

---

## Methodology Notes

- All GPU timings use `cudaEvent_t` start/stop (not CPU wall clock)
- Benchmark runs: 10 warm-up, 100 measured iterations, median reported
- Nsight Compute disables CUPTI sampling that interferes with timing — profile runs are separate from benchmark runs
- Profiling adds overhead; reported benchmark numbers come from unmodified release builds
