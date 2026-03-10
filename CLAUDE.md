# CLAUDE.md — cuda-zkp-ntt

## Project Identity

GPU-accelerated Number-Theoretic Transform for Zero-Knowledge Proofs.
Two complementary optimization directions targeting BLS12-381 ZKP proof generation on NVIDIA GPUs.

- **Direction A**: Async double-buffered NTT pipeline (eliminate CPU-GPU transfer bottleneck)
- **Direction B**: Optimized Montgomery finite-field multiplication (IADD3 instruction path)

---

## Development Environment

- **OS**: Windows 10 Pro (primary build target); WSL2 Ubuntu 22.04 available
- **GPU**: NVIDIA RTX 3060 Laptop GPU, 6GB VRAM, Ampere (sm_86), 30 SMs
- **CUDA**: 12.8
- **Compiler**: MSVC 2022 (primary) / GCC 11 (WSL2)
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
- Representation: 8 × uint32_t limbs in Montgomery form
- Montgomery constant: R = 2^256 mod r
- All finite-field types live in `include/ff_arithmetic.cuh`
- CPU reference implementation in `tests/ff_reference.h` (plain C++, no CUDA)

### NTT
- All NTT operate on BLS12-381 scalar field elements
- Input/output in standard (non-Montgomery) form; Montgomery conversion internal
- Twiddle factors precomputed on device, stored in constant or global memory
- NTT size must be power of 2, range [2^15 .. 2^26]
- Butterfly operation: `A[i] += w * A[j]; A[j] = A[i] - 2*w*A[j]` (in-place Cooley-Tukey)

---

## Project File Map

```
include/
  cuda_utils.cuh      — CUDA_CHECK macro, timing utilities
  ff_arithmetic.cuh   — Fp element type, Montgomery mul, add, sub, inv
  ntt.cuh             — NTT public interface (all variants)
  pipeline.cuh        — AsyncNTTPipeline class interface

src/
  ff_mul.cu           — FF kernels: baseline AoS, v2 branchless (PTX), SoA variants
  ntt_naive.cu        — Radix-2 NTT, no optimization (correctness baseline)
  ntt_optimized.cu    — Radix-256 shared-memory NTT
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
  data/               — Benchmark CSV files
  analysis.md         — Annotated interpretation of profiling results
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
- Two CUDA streams: stream[0] and stream[1] alternate
- cudaEvent_t for synchronization between stages
- Batch size: tunable, default = 2^20 elements
- Ordering: H2D(k+1) overlaps with NTT_compute(k)

### NTT (Cooley-Tukey, radix-256)
- Combines 8 radix-2 stages into one shared-memory kernel launch
- Twiddle factors: precomputed table in global memory, cached in L1
- Thread mapping: 1 thread per butterfly pair within a block
- Block size: 128 or 256 threads (tune per SM occupancy)
- Shared memory usage: 2 × blockDim elements × sizeof(FpElement)

---

## Phase Status

See PROJECT.md (gitignored) for full phase roadmap and strategic context.

Current phase: **Phase 4 — Naive GPU NTT** (next up; Phase 3 complete)

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
