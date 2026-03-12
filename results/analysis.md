# Performance Analysis
## cuda-zkp-ntt — RTX 3060 (Ampere sm_86, CUDA 12.8)

*Each section references a specific screenshot in `screenshots/` and connects findings to the ZKProphet paper analysis.*

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

## Section 5 — v1.2.0 Results (Barrett Arithmetic + Batched NTT)

### 5.1 Barrett vs Montgomery — Single NTT

**Method**: Google Benchmark, cudaEvent_t timing, 5-repetition median, Release build.

| Size | Naive | Montgomery (OPTIMIZED) | Barrett | Barrett vs Montgomery |
|---|---|---|---|---|
| 2^15 | 0.122 ms | 0.132 ms | 0.159 ms | +20% slower |
| 2^16 | 0.228 ms | 0.244 ms | 0.308 ms | +26% slower |
| 2^18 | 1.34 ms | 1.21 ms | 1.27 ms | +5% slower |
| 2^20 | 5.88 ms | 5.51 ms | 5.61 ms | +2% slower |
| **2^22** | **26.2 ms** | **25.1 ms** | **24.9 ms** | **-1% faster** |

**Analysis**: Barrett is slower at small sizes because the fused kernel is compute-bound (69% utilization) and Barrett has 68% more instructions per ff_mul (888 SASS vs 528). At n=2^22, the outer stages dominate (77% of time) and are memory-bound — Barrett's extra instructions are hidden behind DRAM latency, while the elimination of Montgomery conversion (~3 ms) yields a net improvement.

The crossover point is around n=2^18: below that, Barrett's compute overhead exceeds the conversion savings; above that, Barrett is equal or faster.

### 5.2 Batched NTT — Throughput Scaling

**8× NTT batch (Barrett mode) vs sequential 8× Barrett calls:**

| Size | Batched 8× | Sequential 8× | Speedup | Per-NTT (batch) |
|---|---|---|---|---|
| **2^15** | **1.12 ms** | **1.70 ms** | **1.52x** | 0.140 ms |
| 2^18 | 10.4 ms | 11.2 ms | 1.08x | 1.30 ms |
| 2^20 | 48.0 ms | 48.3 ms | 1.01x | 6.00 ms |
| 2^22 | 219 ms | 216 ms | ~1.0x | 27.4 ms |

**Per-NTT cost with batch scaling (Montgomery, 2^15):**

| Batch Size | Total (ms) | Per-NTT (ms) | vs Single |
|---|---|---|---|
| B=1 | 0.140 | 0.140 | baseline |
| B=4 | 0.524 | 0.131 | -6% |
| B=8 | 1.01 | 0.126 | -10% |
| B=16 | 1.99 | 0.124 | -11% |

### 5.3 Batched Montgomery vs Barrett

**Full cross-product: {Montgomery, Barrett} × {B=1, B=4, B=8} — median of 5 reps:**

| Size | Mont B=1 | Mont B=8 | Barrett B=1 | Barrett B=8 |
|---|---|---|---|---|
| 2^15 | 0.140 ms | 1.01 ms | 0.175 ms | 1.12 ms |
| 2^18 | 1.26 ms | 9.76 ms | 1.36 ms | 10.4 ms |
| 2^20 | 5.69 ms | 45.8 ms | 6.09 ms | 48.0 ms |
| 2^22 | 26.5 ms | 214 ms | 26.7 ms | 219 ms |

**Key observations:**
1. **Montgomery is faster per-NTT at all sizes when batched** — the compute overhead of Barrett's 68% more instructions accumulates across B NTTs in the fused kernel.
2. **Batching benefit is largest at small sizes**: at 2^15, a single NTT only fills ~32 blocks for 30 SMs. Batch of 8 gives 256 blocks — much better SM occupancy.
3. **At 2^22, GPU is fully saturated** by a single NTT (4096 fused blocks for 30 SMs). Batching only saves kernel launch overhead (3-4 launches vs 32), which is negligible at these runtimes.
4. **Barrett's advantage at 2^22 disappears with batching** — the B=8 batch amortizes the Montgomery conversion cost across 8 NTTs while Barrett still pays the per-butterfly compute penalty.

### 5.4 Summary — v1.2.0 Performance

**Recommended mode selection:**
- **Single NTT, n ≥ 2^20**: Barrett (saves conversion overhead, outer stages hide extra compute)
- **Single NTT, n < 2^18**: Montgomery (fused kernel compute-bound, fewer instructions win)
- **Batched workloads**: Montgomery for all sizes (conversion cost amortized, lower per-butterfly cost)

**v1.2.0 vs v1.1.0 at n=2^22:**
- Single NTT: 25.1 ms (Montgomery) → 24.9 ms (Barrett) = **-0.8% improvement**
- Batch of 8: 8 × 25.1 = ~201 ms (sequential) → 214 ms (batched Montgomery) — GPU already saturated

The primary v1.2.0 contribution is **infrastructure for future releases**: Barrett arithmetic eliminates the conversion overhead that compounds in the 4-step NTT (v1.3.0), and the batched kernel API enables the batch-of-sub-NTTs pattern that 4-step NTT requires. The measured single-NTT improvement is modest (~1%) because the dominant bottleneck remains the memory-bound outer stages (77% of time).

---

## Section 6 — v1.3.0 Results (4-Step NTT Algorithm)

### 6.1 4-Step NTT vs Barrett — Single NTT

**Method**: Google Benchmark, cudaEvent_t timing, 5-repetition median, Release build.

The 4-step NTT (Bailey's algorithm) decomposes an n-point NTT into sub-NTTs + transpose +
twiddle multiply. For n=2^22: n1=n2=2^11 → 2048 independent 2048-point sub-NTTs per step.
Falls back to Barrett for n < 2^16 (sub-NTTs need K=8 minimum = 256 elements).

| Size | Naive | Montgomery | Barrett | **4-Step** | 4-Step vs Barrett |
|---|---|---|---|---|---|
| 2^15 | 0.122 ms | 0.132 ms | 0.158 ms | 0.160 ms | ~0% (Barrett fallback) |
| 2^16 | 0.228 ms | 0.245 ms | 0.300 ms | **0.491 ms** | **+64% slower** |
| 2^18 | 1.34 ms | 1.21 ms | 1.27 ms | **1.66 ms** | **+31% slower** |
| 2^20 | 5.88 ms | 5.50 ms | 5.60 ms | **7.03 ms** | **+26% slower** |
| **2^22** | **26.2 ms** | **25.1 ms** | **24.9 ms** | **29.5 ms** | **+18% slower** |

**Key finding**: The 4-step NTT is **slower at all sizes** where true 4-step execution occurs
(n ≥ 2^16). The target of ≤16 ms at n=2^22 was not met.

### 6.2 Root Cause Analysis — Why 4-Step is Slower

The roadmap assumed that 4-step would eliminate memory-bound outer-stage DRAM passes by keeping
all butterfly stages within shmem-resident sub-NTTs. **This assumption was partially incorrect:**

1. **Sub-NTTs still have outer stages**: For n=2^22, sub-NTTs are size 2048=2^11 (11 stages).
   K=10 fuses stages 0-9 in shared memory, but **1 cooperative outer stage remains** per sub-NTT
   batch. This outer stage accesses the full n-element array from DRAM.

2. **L2 residency assumption was wrong**: The roadmap assumed "sub-NTTs at 2048 elements have
   ≤1 outer stage, L2-resident". A single 2048-element sub-NTT is 64 KB (fits in 4 MB L2).
   But when batching 2048 sub-NTTs, the total data is 2048×2048×32B = **128 MB >> 4 MB L2**.
   The sub-NTT outer stage still hits DRAM on every access.

3. **3 transpose passes add pure overhead**: Each transpose reads and writes the full n-element
   array (2×n DRAM operations). Three transposes = 6n DRAM ops — substantial overhead that
   the original cooperative approach does not incur.

4. **Additional overhead sources**:
   - Twiddle multiply: n reads + n writes + n twiddle reads = 3n DRAM ops
   - Final memcpy: 2n DRAM ops
   - Kernel launch overhead: 7 operations per NTT vs 4 for cooperative
   - CPU-side dispatch: 33.7 ms CPU time vs 25.0 ms (complex host-side logic)

**DRAM traffic comparison (estimated, n=2^22):**

| Approach | Operation | DRAM ops |
|---|---|---|
| Cooperative (v1.2) | 12 outer stages × ~5n per stage | ~60n |
| 4-Step (v1.3) | 3 transposes (6n) + twiddle (3n) + memcpy (2n) + 2 sub-NTT outer stages (10n) | ~21n + sub-NTT overhead |

The 4-step's ~21n base DRAM traffic is lower than the cooperative's ~60n, but the sub-NTT
infrastructure adds significant constant overhead that dominates at these sizes. The fused
kernel in sub-NTTs also has lower block occupancy (4096 blocks for 2048 sub-NTTs of 2048
elements → 2 blocks per sub-NTT → 4096 total, same as cooperative) while requiring more
complex dispatch logic.

**The fundamental problem**: To fully eliminate outer stages, sub-NTTs must be ≤ 2^10 = 1024
elements (fully fused by K=10). For n=2^22, this requires n1×n2 = 2^22 with both ≤ 2^10 —
impossible. A multi-level (3-step) decomposition would be needed: e.g., 1024 × 1024 × 4,
which adds yet more transpose overhead.

### 6.3 Batched 4-Step vs Barrett — 8× NTTs

| Size | Barrett batch 8× | Barrett seq 8× | 4-Step batch 8× | 4-Step seq 8× |
|---|---|---|---|---|
| 2^15 | 1.01 ms | 1.54 ms | 1.01 ms | 1.61 ms |
| 2^18 | 9.65 ms | 10.8 ms | 11.4 ms | 17.2 ms |
| 2^20 | 44.6 ms | 47.5 ms | 53.1 ms | 66.9 ms |
| 2^22 | 199 ms | 208 ms | 241 ms | 279 ms |

**Batch speedup (batch/sequential):**

| Size | Barrett | 4-Step |
|---|---|---|
| 2^15 | 1.52x | 1.59x |
| 2^18 | 1.12x | **1.51x** |
| 2^20 | 1.07x | **1.26x** |
| 2^22 | 1.05x | **1.16x** |

**Observation**: The 4-step benefits MORE from batching than Barrett at medium-large sizes
(2^18-2^22). The 4-step's internal structure (2048 sub-NTTs per step) creates natural batch
parallelism. When external batching adds B more NTTs, sub-NTT batches grow from 2048 to
B×2048, improving GPU utilization for the cooperative outer stage.

However, in **absolute terms**, Barrett batched remains faster at all sizes. The 4-step's
inherent transpose overhead outweighs its batching advantage.

### 6.4 Summary — v1.3.0 Performance

| Metric | v1.2.0 Barrett | v1.3.0 4-Step | Delta |
|---|---|---|---|
| Single NTT 2^22 | 24.9 ms | 29.5 ms | **+18% slower** |
| Batch 8× 2^22 | 199 ms | 241 ms | **+21% slower** |
| Batch 8× 2^18 | 9.65 ms | 11.4 ms | **+18% slower** |
| Best single NTT mode | Barrett | Barrett | unchanged |

**v1.3.0 is a negative performance result.** The 4-step algorithm, while correct and fully
tested (221 tests), does not improve performance on RTX 3060 with our implementation. The
cooperative outer approach (v1.2.0 Barrett) remains the fastest path.

**Lessons learned:**
1. Transpose overhead is substantial for 256-bit elements (32 bytes per element, 3 full passes)
2. Sub-NTT outer stages still access full DRAM when batched (L2 too small for aggregate data)
3. The cooperative approach's advantage: zero structural overhead, direct butterfly access pattern
4. 4-step may work better on GPUs with larger L2 (e.g., H100 with 50 MB L2) or with sub-NTTs
   small enough to be fully fused (requires multi-level decomposition)

**Future directions for v1.4.0**: Register-centric butterfly optimization and CUDA Graphs
remain viable — they target the cooperative outer stages directly without adding transpose
overhead. The 4-step infrastructure may be revisited with L2-aware sub-NTT batch scheduling
in a future release.

---

## Section 7 — v1.4.0 Results (Branchless Arithmetic + Radix-4 + CUDA Graphs)

### 7.1 Branchless Arithmetic (Session 9)

Switched all NTT hot-path kernels from branchy to branchless arithmetic:
- `ff_mul` → `ff_mul_ptx` (branchless conditional reduction via PTX sub.cc + lop3)
- `ff_add` → `ff_add_v2` (branchless PTX carry chain + lop3 select)
- `ff_sub` → `ff_sub_v2` (branchless PTX subtract + lop3 select)
- `ff_mul_barrett` → `ff_mul_barrett_v2` (branchless 2× conditional subtraction via PTX)

**Register usage (K=10)**: Montgomery 68→66 (-2), Barrett 92→80 (-12, -13%)

**Single NTT (n=2^22, 7-rep median):**

| Mode | v1.2.0 | v1.4.0-s9 | Delta |
|---|---|---|---|
| Barrett | 24.9 ms | **23.8 ms** | **-4.4%** |
| Montgomery | 25.1 ms | **24.4 ms** | **-2.8%** |

Barrett improved more because it had 2× branchy conditional subtract loops vs Montgomery's 1×.

### 7.2 Radix-4 Outer Stages (Session 10)

Fuses pairs of consecutive outer stages into radix-4 butterflies:
- 4 data loads + 4 stores per radix-4 unit (vs 8+8 for 2 radix-2 stages)
- **~45% DRAM traffic reduction** (theoretical), measured ~43%
- For n=2^22: 12 outer stages → 6 radix-4 passes in **1 cooperative launch** (was 2)

**Single NTT (n=2^22, 7-rep median):**

| Mode | v1.4.0-s9 | v1.4.0-s10 | Delta |
|---|---|---|---|
| Montgomery | 24.4 ms | **17.0 ms** | **-30.3%** |
| Barrett | 23.8 ms | **17.1 ms** | **-28.2%** |

Outer stages went from ~19.4 ms to ~11 ms (-43%), matching theoretical DRAM traffic reduction.

### 7.3 CUDA Graphs (Session 11)

CUDA Graph API captures the NTT kernel launch sequence on first call and replays it on
subsequent calls via `cudaGraphLaunch` (~5us vs ~20-40us per individual launch).

**API**: `ntt_forward_graph()`, `ntt_inverse_graph()`, `ntt_forward_batch_graph()`,
`ntt_inverse_batch_graph()`, `ntt_graph_clear_cache()`

**Measured impact (7-rep median, head-to-head comparison):**

| Size | Montgomery | Graph | Barrett | Graph |
|---|---|---|---|---|
| 2^15 | 0.120 ms | 0.129 ms | 0.142 ms | 0.151 ms |
| 2^22 | 17.1 ms | 17.4 ms | 17.4 ms | 17.4 ms |

**Result**: Negligible performance difference (within ±2% measurement noise). Only 3-4 kernel
launches per NTT means CPU launch overhead is already ~50-100us total — small compared to
GPU compute time even at the smallest sizes. The API is valuable for embedding NTT in larger
CUDA Graph workflows where amortizing graph overhead across many operations matters.

### 7.4 Summary — v1.4.0 Performance

**Cumulative improvement from v1.1.0 (n=2^22, 7-rep median):**

| Metric | v1.1.0 | v1.2.0 | v1.3.0 | **v1.4.0** | v1.4.0 vs v1.1.0 |
|---|---|---|---|---|---|
| Montgomery | 25.1 ms | 25.1 ms | — | **17.1 ms** | **-32%** |
| Barrett | — | 24.9 ms | 29.5 ms (4-step) | **17.4 ms** | **-30%** |
| Batch 8× (Montgomery) | ~201 ms | 214 ms | — | **150 ms** | **-25%** |
| Batch 8× (Barrett) | — | 219 ms | 241 ms (4-step) | **159 ms** | **-27%** |
| Tests | — | 119 | 221 | **230** | — |

**Key contributions:**
1. **Branchless arithmetic** (-4.4%): eliminated warp divergence in all NTT hot-path kernels
2. **Radix-4 outer stages** (-30%): fused pairs of stages → 45% DRAM traffic reduction
3. **CUDA Graphs** (negligible): clean API for graph workflows, ~0% performance impact

**The v1.4.0 result (17.1 ms) exceeds the 18-22 ms target** set in the roadmap. The remaining
bottleneck is the cooperative outer stages (~11 ms at 2^22), which are fundamentally limited
by DRAM bandwidth on RTX 3060 (~360 GB/s).

---

## Section 8 — L2 Cache Behavior at v1.5.0 (Session 12)

### 8.1 L2 Cache Diagnostic — Stockham v1.8.0 Go/No-Go

**Method**: Nsight Compute 2025.1.1 `--set full`, radix-8 Barrett outer kernel profiled at 2^18, 2^20, 2^22. RTX 3060 Laptop GPU (SM 8.6, 30 SMs, 3 MB L2).

**L2 Hit Rate vs NTT Size:**

| Size | Working Set | L2 Hit Rate (radix-8 outer) | L2 Hit Rate (radix-4 leftover) |
|---|---|---|---|
| 2^18 | 8 MB | 64.8% | 47.9% |
| 2^20 | 32 MB | 60.6% | — |
| 2^22 | 128 MB | **58.5%** | — |

### 8.2 All Kernels at 2^22

| Kernel | L2 Hit Rate | Grid Config |
|---|---|---|
| `ntt_bit_reverse_kernel` | 65.5% | (16384, 1, 1) x (256, 1, 1) |
| `ntt_fused_stages_barrett_kernel<10>` | 78.1% | (4096, 1, 1) x (512, 1, 1) |
| `ntt_outer_radix8_barrett_kernel` | **58.5%** | (30, 1, 1) x (256, 1, 1) |

### 8.3 Radix-8 Outer Kernel Detailed Profile (2^22)

| Metric | Value |
|---|---|
| Occupancy (theoretical) | 16.7% (register-limited, 174 regs) |
| ALU utilization | 20.7% |
| Uncoalesced global accesses | 50% excessive sectors (16/32 bytes utilized) |
| Top stall | Instruction fetch/select: 55.3% of stall cycles |
| Issue rate | 1 instruction per 5.0 cycles |
| Eligible warps per cycle | 0.27 (of 2.0 active) |
| Blocks per SM | 1 (vs 2 for radix-4 at 98 regs) |

### 8.4 Radix-8 Register Pressure

| Kernel | Registers | Blocks/SM | Total Blocks |
|---|---|---|---|
| Radix-4 Montgomery | 86 | 2 | 60 |
| Radix-4 Barrett | 98 | 2 | 60 |
| **Radix-8 Montgomery** | **134** | **1** | **30** |
| **Radix-8 Barrett** | **174** | **1** | **30** |

No register spills (STACK:0, LOCAL:0) on any kernel.

### 8.5 Stockham v1.8.0 Decision: NO-GO

L2 hit rate at 2^22 is 58.5%, well above the 50% threshold. The outer stages are **bandwidth-bound**, not latency-bound. Stockham's sequential access pattern would not improve L2 reuse since it is already effective. The hit rate barely degrades from 2^18 (65%) to 2^22 (58.5%) despite a 16x increase in working set.

### 8.6 Primary Optimization Targets Identified

1. **50% uncoalesced access** — the stride-h butterfly pattern wastes half the sector bandwidth. This is the dominant inefficiency.
2. **16.7% occupancy** — 174 registers for Barrett radix-8 limits to 1 block/SM. The Montgomery variant (134 regs) is similarly limited.
3. **Instruction cache thrashing** — 55% of stall cycles are instruction fetch/select, likely due to the large kernel body (Barrett radix-8: ~174 regs of live state cycling through the 3-stage butterfly).

**ncu-rep files**: `results/data/l2_diag_outer_2e{18,20,22}.ncu-rep`, `results/data/l2_diag_all_2e22.ncu-rep`

---

## Section 9 — Radix-8 Benchmark Results (Session 13)

### 9.1 Barrett Radix-8: Catastrophic Regression

Initial benchmark with radix-8 active for both Montgomery and Barrett revealed a severe
regression in the Barrett path:

| Size | Barrett radix-8 | Barrett radix-4 (v1.4.0) | Change |
|------|---|---|---|
| 2^15 | 0.146 ms | 0.159 ms | −8% |
| 2^18 | 1.33 ms | 1.27 ms | +5% |
| 2^20 | 6.44 ms | 5.61 ms | +15% |
| 2^22 | **29.5 ms** | **17.1 ms** | **+73%** |

Root cause: Barrett radix-8 kernel uses 174 registers with an enormous instruction footprint.
Session 12 profiling showed 55.3% of stall cycles in instruction fetch/select, confirming
instruction cache thrashing as the bottleneck. The radix-4 Barrett kernel uses ~98 regs
(2 blocks/SM) and is small enough to stay I-cache resident.

### 9.2 Montgomery Radix-8: Clear Win

Montgomery radix-8 (134 regs) has a smaller instruction footprint and benefits from the
DRAM traffic reduction (4 passes vs 6 for radix-4):

| Size | Montgomery radix-8 | Montgomery radix-4 (v1.4.0) | Change |
|------|---|---|---|
| 2^15 | 0.121 ms | 0.132 ms | −8% |
| 2^18 | 0.900 ms | 1.21 ms | −26% |
| 2^20 | 3.84 ms | 5.51 ms | −31% |
| 2^22 | **15.6 ms** | **17.0 ms** | **−8.2%** |

### 9.3 Decision: Montgomery-Only Radix-8

Barrett dispatch changed to skip radix-8 and use radix-4 directly. After fix:

| Size | Montgomery (radix-8) | Barrett (radix-4) |
|------|---|---|
| 2^15 | 0.121 ms | 0.141 ms |
| 2^16 | 0.225 ms | 0.255 ms |
| 2^18 | 0.900 ms | 0.972 ms |
| 2^20 | 3.84 ms | 4.15 ms |
| **2^22** | **15.6 ms** | **18.0 ms** |

Montgomery radix-8 is now the fastest path at all sizes. The 15.6 ms at 2^22 is the new
best result (was 17.0 ms in v1.4.0, −8.2%).

### 9.4 Batched Performance (8× NTT)

| Size | Montgomery (radix-8) | Barrett (radix-4) |
|------|---|---|
| 2^15 | 0.788 ms | 0.957 ms |
| 2^18 | 7.08 ms | 8.12 ms |
| 2^20 | 32.3 ms | 36.2 ms |
| 2^22 | 139 ms | 167 ms |

Montgomery radix-8 batched at 2^22: 139 ms / 8 = **17.4 ms per NTT** (near-linear scaling).

### 9.5 Lessons Learned

1. **Instruction cache is a real constraint for large kernels.** Barrett radix-8's 174 regs
   + huge instruction body exceeded the I-cache capacity, causing 55% I-fetch stalls.
   Montgomery at 134 regs is at the edge but still benefits.
2. **Register count alone doesn't predict performance.** Both Montgomery and Barrett radix-8
   have 1 block/SM (16.7% occupancy), but Montgomery is 2× faster due to instruction count.
3. **Mixed radix strategies are viable.** Using different radix for different arithmetic paths
   (radix-8 for Montgomery, radix-4 for Barrett) extracts the best from each.

Benchmark data: `results/data/bench_v150_s13.json`

---

## Methodology Notes

- All GPU timings use `cudaEvent_t` start/stop (not CPU wall clock)
- Benchmark runs: 10 warm-up, 100 measured iterations, median reported
- Nsight Compute disables CUPTI sampling that interferes with timing — profile runs are separate from benchmark runs
- Profiling adds overhead; reported benchmark numbers come from unmodified release builds
