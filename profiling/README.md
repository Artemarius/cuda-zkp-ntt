# Profiling Methodology

Replicating the ZKProphet (IISWC 2025) analysis framework on NVIDIA RTX 3060.

## Hardware

- NVIDIA RTX 3060 Laptop GPU (Ampere, sm_86, 30 SMs, 6GB GDDR6)
- PCIe 4.0 x16 (~32 GB/s peak)
- INT32 throughput: ~3.19 TOPS

## Tools

- **Nsight Compute (ncu)**: kernel-level profiling — roofline, instruction mix, warp stalls
- **Nsight Systems (nsys)**: timeline profiling — async pipeline overlap visualization

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/profile_ntt.sh` | Full ncu profile, exports `.ncu-rep` |
| `scripts/collect_metrics.sh` | Targeted metric CSV: SM throughput, warp stalls, roofline data |
| `scripts/nsys_timeline.sh` | nsys timeline for async pipeline, exports `.nsys-rep` |

## Key Metrics

### Roofline (replicating ZKProphet Fig. 9)
- Arithmetic intensity: `weighted_ops / bytes_accessed`
- Ceilings: INT32 = 3.19 TOPS, DRAM = 360 GB/s, L2 = ~1.5 TB/s

### Warp Stall Breakdown (replicating ZKProphet Fig. 10)
- `Stall_Wait`: fixed-latency instruction dependency
- `Stall_Math_Pipe_Throttle`: INT32 pipeline fully subscribed
- `Stall_Not_Selected`: scheduler chose another warp

### Instruction Mix
- IMAD %: target reduction via IADD3 path (Direction B)
- Branch efficiency: target improvement via branchless reduction

## Output

- `.ncu-rep` files: gitignored (large binaries)
- PNG screenshots: `results/screenshots/`
- CSV data: `results/data/`
