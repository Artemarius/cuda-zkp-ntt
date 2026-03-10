#!/bin/bash
# profile_ntt.sh — Full Nsight Compute profile for NTT kernels
# Usage: bash profile_ntt.sh [naive|optimized|async] [log2_size]
#
# Requires: ncu (Nsight Compute) in PATH
# Output:   ../results/data/ncu_*.ncu-rep files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../../build"
RESULTS_DIR="$SCRIPT_DIR/../../results/data"

MODE="${1:-naive}"
LOG2_SIZE="${2:-22}"

mkdir -p "$RESULTS_DIR"

BINARY="$BUILD_DIR/ntt_profile"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run cmake --build build first."
    exit 1
fi

echo "=== Nsight Compute Profile ==="
echo "Mode:    $MODE"
echo "Scale:   2^$LOG2_SIZE"
echo "Output:  $RESULTS_DIR/ncu_${MODE}_2e${LOG2_SIZE}.ncu-rep"
echo ""

# Full profile with all sections
ncu \
    --set full \
    --target-processes all \
    --export "$RESULTS_DIR/ncu_${MODE}_2e${LOG2_SIZE}" \
    --force-overwrite \
    "$BINARY" --mode "$MODE" --log2-size "$LOG2_SIZE"

echo ""
echo "Profile saved. Open with:"
echo "  ncu-ui $RESULTS_DIR/ncu_${MODE}_2e${LOG2_SIZE}.ncu-rep"
