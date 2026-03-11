# CLAUDE.md — cuda-zkp-ntt

## Project Identity

GPU-accelerated Number-Theoretic Transform for Zero-Knowledge Proofs.
Two complementary optimization directions targeting BLS12-381 ZKP proof generation on NVIDIA GPUs.

- **Direction A**: Async double-buffered NTT pipeline (eliminate CPU-GPU transfer bottleneck)
- **Direction B**: Optimized Montgomery finite-field multiplication (IADD3 instruction path)

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
- `test_correctness` — validates NTT output against CPU reference
- `bench_ntt` — Google Benchmark harness, outputs CSV
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
- CPU reference implementation in `tests/ff_reference.h` (plain C++, no CUDA)

### NTT
- All NTT operate on BLS12-381 scalar field elements
- Input/output in standard (non-Montgomery) form
- Two arithmetic paths: `NTTMode::OPTIMIZED` (Montgomery internal, conversion overhead)
  and `NTTMode::BARRETT` (standard-form throughout, no conversion)
- Batched NTT: `ntt_forward_batch()` / `ntt_inverse_batch()` process B independent NTTs
  in a single set of kernel launches. Contiguous memory layout: `d_data[b*n..(b+1)*n-1]`
- Twiddle factors precomputed on device: Montgomery twiddles + separate standard-form cache
  Shared across all NTTs in a batch (single precomputation, B× reuse)
- NTT size must be power of 2, range [2^8 .. 2^26]
- Butterfly operation: `A[i] += w * A[j]; A[j] = A[i] - 2*w*A[j]` (in-place Cooley-Tukey)

---

## Project File Map

```
include/
  cuda_utils.cuh      — CUDA_CHECK macro, timing utilities
  ff_arithmetic.cuh   — Fp element type, Montgomery mul, add, sub, inv
  ff_barrett.cuh      — Barrett modular multiplication (standard-form, no Montgomery)
  ntt.cuh             — NTT public interface (single + batched, NTTMode: NAIVE, OPTIMIZED, BARRETT, ASYNC)
  pipeline.cuh        — AsyncNTTPipeline class interface

src/
  ff_mul.cu           — FF kernels: baseline AoS, v2 branchless (PTX), SoA variants
  ntt_naive.cu        — Radix-2 NTT baseline + public API dispatch + twiddle caches
  ntt_optimized.cu    — NTT host dispatch: K selection, cooperative outer (Montgomery + Barrett)
  ntt_fused_kernels.cu — Fused warp-shuffle + shmem kernel (K=8/9/10, Montgomery + Barrett, no-RDC TU)
  ntt_async.cu        — Double-buffered async pipeline NTT
  benchmark.cu        — Main benchmark entry point

tests/
  test_correctness.cu — Validates all NTT variants agree with CPU DFT reference
  ff_reference.h      — CPU-only finite field + NTT reference (test oracle)

benchmarks/
  bench_ntt.cu        — Google Benchmark: latency vs scale for all variants
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
  data/               — Benchmark CSV files
  analysis.md         — Annotated interpretation of profiling results

scripts/
  plot_benchmarks.py  — Generate benchmark bar charts from measured data

.github/
  workflows/
    build.yml          — CI: compile on Linux (CUDA 12.8/12.6) + Windows (MSVC)

GUIDE.md               — Deep-dive: ZKP math, NTT, finite fields, GPU optimization
NTT_OPTIMIZATION_ROADMAP.md — Future release plans (v1.2.0-v1.4.0) with session breakdown
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

### NTT Time Breakdown (n=2^22, v1.1.0)
- Bit-reverse: ~0.3 ms (1%), Montgomery conversions: ~3.0 ms (12%)
- Fused K=10 (stages 0-9): ~2.5 ms (10%) — **compute-bound** (69%, IPC 2.41)
- Cooperative outer (stages 10-21): ~19.4 ms (77%) — **memory-bound** (DRAM R-M-W)
- Outer stages are the dominant bottleneck; see NTT_OPTIMIZATION_ROADMAP.md

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

### 4-Step NTT (planned, v1.3.0)
- Bailey's algorithm: decompose n = n1 × n2 into sub-NTTs + transpose + twiddle multiply
- For n=2^22 with n1=n2=2^11: each sub-NTT is 2048 elements → fully fused in shmem
- Eliminates outer-stage DRAM passes entirely (~3 DRAM passes vs current 12+)
- Requires efficient FpElement matrix transpose kernel (shared-memory tiled, coalesced)
- Natural synergy with batching: B full NTTs = B×n2 sub-NTTs in step 1

---

## Phase Status

See PROJECT.md (gitignored) for full phase roadmap and strategic context.
See `NTT_OPTIMIZATION_ROADMAP.md` for future release plans (v1.2.0-v1.4.0).

Phases 1-8 complete. Current version: **v1.1.0** (v1.0.0 tagged and [released on GitHub](https://github.com/Artemarius/cuda-zkp-ntt/releases/tag/v1.0.0)).

**v1.2.0 progress:** Sessions 1-3 (Barrett arithmetic + NTT integration + batched NTT) complete. Session 4 remaining.

### Future Releases
- **v1.2.0** — MoMA-inspired Barrett arithmetic + batched NTT (target: ~22 ms single, better batch throughput)
- **v1.3.0** — 4-Step NTT algorithm (target: ≤16 ms single, ~80-100 ms batch-of-8)
- **v1.4.0** — Register optimization + phase-aware pipeline + CUDA Graphs (target: ~10-14 ms)

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
