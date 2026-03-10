# Profiling Methodology

Replicating the ZKProphet (IISWC 2025) analysis framework on NVIDIA RTX 3060.

## Hardware

- NVIDIA RTX 3060 Laptop GPU (Ampere, sm_86, 30 SMs, 6GB GDDR6)
- PCIe 3.0 x8 (~6.5 GB/s measured H2D, ~6.2 GB/s D2H)
- INT32 throughput: ~3.42 TOPS (30 SMs x 4 SMSPs x 16 x 1.78 GHz)
- asyncEngineCount: 5 (supports concurrent H2D + D2H)

## Tools

- **Nsight Compute (ncu) 2025.1.1**: kernel-level profiling — roofline, instruction mix, warp stalls
- **cuobjdump**: SASS instruction disassembly for instruction mix analysis

## Profiling Commands

All ncu commands require **administrator privileges** on Windows (ERR_NVGPUCTRPERM otherwise).

### FF Kernel Profiles
```bash
# FF_mul baseline (2^20 elements, skip warmup kernel, capture profiled kernel)
ncu --set full --launch-count 1 --launch-skip 1 \
    --export results/data/ncu_ff_mul_2e20 --force-overwrite \
    build/Release/ntt_profile.exe --mode ff_mul --size 20

# FF_mul v2 branchless
ncu --set full --launch-count 1 --launch-skip 1 \
    --export results/data/ncu_ff_mul_v2_2e20 --force-overwrite \
    build/Release/ntt_profile.exe --mode ff_mul_v2 --size 20

# FF_add baseline and v2 (for branch instruction comparison)
ncu --set full --launch-count 1 --launch-skip 1 \
    --export results/data/ncu_ff_add_2e20 --force-overwrite \
    build/Release/ntt_profile.exe --mode ff_add --size 20

ncu --set full --launch-count 1 --launch-skip 1 \
    --export results/data/ncu_ff_add_v2_2e20 --force-overwrite \
    build/Release/ntt_profile.exe --mode ff_add_v2 --size 20
```

### NTT Kernel Profiles

NTT launches multiple kernels per call (bit-reverse, butterfly stages, Montgomery conversion).
Use `--kernel-name` to select the butterfly kernel and `--launch-skip` to skip warmup instances.

```bash
# NTT naive butterfly kernel (radix-2, skip warmup's 20 butterfly launches)
ncu --set full --kernel-name ntt_butterfly_kernel \
    --launch-skip 20 --launch-count 1 \
    --export results/data/ncu_ntt_naive_2e20 --force-overwrite \
    build/Release/ntt_profile.exe --mode naive --size 20

# NTT optimized fused kernel (radix-256, skip warmup's 1 fused launch)
ncu --set full --kernel-name ntt_fused_stages_kernel \
    --launch-skip 1 --launch-count 1 \
    --export results/data/ncu_ntt_optimized_2e20 --force-overwrite \
    build/Release/ntt_profile.exe --mode optimized --size 20

# Same for 2^22 scale
ncu --set full --kernel-name ntt_butterfly_kernel \
    --launch-skip 22 --launch-count 1 \
    --export results/data/ncu_ntt_naive_2e22 --force-overwrite \
    build/Release/ntt_profile.exe --mode naive --size 22

ncu --set full --kernel-name ntt_fused_stages_kernel \
    --launch-skip 1 --launch-count 1 \
    --export results/data/ncu_ntt_optimized_2e22 --force-overwrite \
    build/Release/ntt_profile.exe --mode optimized --size 22
```

### SASS Instruction Disassembly
```bash
cuobjdump --dump-sass build/Release/zkp_ntt_core.lib | grep -c "INSTRUCTION" # count per kernel
```

## Automation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/profile_ntt.sh` | Full ncu profile, exports `.ncu-rep` |
| `scripts/collect_metrics.sh` | Targeted metric CSV: SM throughput, warp stalls, roofline data |
| `scripts/nsys_timeline.sh` | nsys timeline for async pipeline, exports `.nsys-rep` |

## Key Metrics

### Roofline (replicating ZKProphet Fig. 9)
- Arithmetic intensity: `weighted_ops / bytes_accessed`
- Ceilings: INT32 = ~3.42 TOPS, DRAM = ~360 GB/s, L2 = ~1.5 TB/s
- FF_mul baseline: memory-bound (92% DRAM, 64% compute)
- NTT fused kernel: compute-bound (69% compute, 55% memory)

### Warp Stall Breakdown (replicating ZKProphet Fig. 10)
- `Stall_Long_Scoreboard`: global memory load latency (dominant in FF_mul baseline)
- `Stall_LG_Throttle`: L1/global memory pipeline saturated
- `Stall_Math_Pipe_Throttle`: INT32 pipeline fully subscribed (dominant in fused NTT)
- `Stall_Not_Selected`: scheduler chose another warp
- `Stall_Barrier`: `__syncthreads()` synchronization (fused NTT only)

### Instruction Mix
- IMAD %: CIOS Montgomery multiply-accumulate (4-cycle issue latency)
- IADD3 %: integer add-3 accumulation (2-cycle issue latency)
- LOP3.LUT: branchless MUX replacement for ISETP+SEL (Direction B optimization)

## Output

- `.ncu-rep` files: gitignored (large binaries), stored in `results/data/`
- PNG screenshots: `results/screenshots/` (11 total)
- CSV data: `results/data/`
- Analysis: `results/analysis.md` (annotated findings with ZKProphet references)
