#!/bin/bash
# l2_diagnostic.sh — L2 cache diagnostic for radix-4 outer kernel (v1.5.0 Session 12, Part A)
#
# Profiles the radix-4 cooperative outer kernel with Nsight Compute to measure
# L2 cache behavior at three NTT sizes. The L2 hit rate at 2^22 determines
# whether Stockham outer stages (v1.8.0) are worth pursuing:
#
#   L2 hit rate < 20% at 2^22 --> latency-bound --> Stockham GO
#     (data spills to DRAM each pass; Stockham's sequential access helps)
#   L2 hit rate > 50% at 2^22 --> bandwidth-bound --> Stockham NO-GO
#     (L2 already capturing reuse; Stockham cannot improve further)
#
# Usage: bash l2_diagnostic.sh
# Requires: ncu (Nsight Compute CLI) in PATH
# Binary:   build/Release/ntt_profile.exe (Windows MSVC build)
#
# Outputs results to console (no .ncu-rep files saved).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../../build/Release/ntt_profile.exe"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found."
    echo "Build with: cmake --build build --config Release --target ntt_profile"
    exit 1
fi

# ─── Metric list ─────────────────────────────────────────────────────────────
# L1/L2 cache hit rates
METRICS="l1tex__t_sector_hit_rate.pct"
METRICS+=",lts__t_sector_hit_rate.pct"
METRICS+=",lts__t_sector_op_read_hit_rate.pct"
# DRAM traffic
METRICS+=",dram__bytes_read.sum"
METRICS+=",dram__bytes_write.sum"
# L2 sector demand from texture/L1
METRICS+=",lts__t_sectors_srcunit_tex_op_read.sum"
# Occupancy
METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"
# Integer instruction count
METRICS+=",smsp__sass_thread_inst_executed_op_integer_pred_on.sum"

SIZES=(18 20 22)

# ═════════════════════════════════════════════════════════════════════════════
# Part 1: Radix-4 outer kernel only (filtered by kernel name)
# ═════════════════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  L2 Cache Diagnostic — Radix-4 Outer Kernel (Barrett)             ║"
echo "║  v1.5.0 Session 12, Part A                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Binary: $BINARY"
echo "Mode:   barrett"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SECTION 1: Radix-4 outer kernel metrics (kernel-name filter)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for SIZE in "${SIZES[@]}"; do
    N=$((1 << SIZE))
    BYTES=$((N * 32))  # 8 x uint32_t = 32 bytes per element

    echo "┌─────────────────────────────────────────────────────"
    echo "│ NTT size: 2^${SIZE} = ${N} elements (${BYTES} bytes = $((BYTES / 1024 / 1024)) MB)"
    echo "│ Working set: $((BYTES * 2 / 1024 / 1024)) MB (data + twiddles)"
    echo "│ RTX 3060 L2: 3 MB"
    echo "└─────────────────────────────────────────────────────"
    echo ""

    ncu \
        --metrics "$METRICS" \
        --kernel-name-base function \
        --kernel-name "regex:radix[48]" \
        --target-processes all \
        "$BINARY" --mode barrett --size "$SIZE" \
        2>&1 | grep -v "^==" || true

    echo ""
    echo "---"
    echo ""
done

# ═════════════════════════════════════════════════════════════════════════════
# Part 2: ALL kernels — total DRAM traffic breakdown per kernel
# ═════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SECTION 2: All kernels — traffic breakdown (2^22 only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Profiling all kernels at 2^22 (no kernel-name filter)..."
echo ""

ncu \
    --metrics "$METRICS" \
    --kernel-name-base function \
    --target-processes all \
    "$BINARY" --mode barrett --size 22 \
    2>&1 | grep -v "^==" || true

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# Decision-point summary
# ═════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " DECISION FRAMEWORK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Read lts__t_sector_hit_rate.pct for the radix-4 kernel at 2^22 above."
echo ""
echo "  L2 hit rate < 20%  -->  LATENCY-BOUND (data spills to DRAM each pass)"
echo "    => Stockham outer stages GO (v1.8.0)"
echo "       Sequential access pattern can recover L2 locality."
echo "       Expected: outer stages -20 to -30% from improved L2 reuse."
echo ""
echo "  L2 hit rate 20-50% -->  MIXED REGIME"
echo "    => Stockham CONDITIONAL. Need per-stage L2 miss analysis."
echo "       Consider radix-8 first (v1.5.0) to reduce pass count."
echo ""
echo "  L2 hit rate > 50%  -->  BANDWIDTH-BOUND (L2 already capturing reuse)"
echo "    => Stockham NO-GO"
echo "       L2 is already effective; access pattern change won't help."
echo "       Focus on reducing arithmetic cost (Plantard, v1.7.0) instead."
echo ""
echo "Also compare DRAM traffic across sizes:"
echo "  2^18: $((1 << 18)) elems x 32B = $((1 << 18 << 5)) B = $(( (1 << 18 << 5) / 1024 / 1024 )) MB"
echo "           (fits in 3 MB L2 --> expect high hit rate)"
echo "  2^20: $((1 << 20)) elems x 32B = $((1 << 20 << 5)) B = $(( (1 << 20 << 5) / 1024 / 1024 )) MB"
echo "           (exceeds 3 MB L2 --> transition point)"
echo "  2^22: $((1 << 22)) elems x 32B = $((1 << 22 << 5)) B = $(( (1 << 22 << 5) / 1024 / 1024 )) MB"
echo "           (far exceeds L2 --> expect low hit rate)"
echo ""
echo "Done."
