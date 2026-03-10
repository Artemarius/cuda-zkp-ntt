#!/bin/bash
# nsys_timeline.sh — Nsight Systems timeline for async pipeline visualization
# Captures GPU timeline showing H2D transfer and compute overlap.
#
# Usage: bash nsys_timeline.sh
# Output: ../results/data/nsys_*.nsys-rep (open in Nsight Systems UI for screenshot)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../../build"
RESULTS_DIR="$SCRIPT_DIR/../../results/data"

mkdir -p "$RESULTS_DIR"

BINARY="$BUILD_DIR/ntt_profile"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found."
    exit 1
fi

echo "=== Nsight Systems Timeline Capture ==="
echo ""

# 1. Non-pipelined (sequential H2D → compute) — for baseline comparison
echo "Capturing non-pipelined baseline..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$RESULTS_DIR/nsys_sequential" \
    --force-overwrite true \
    "$BINARY" --mode optimized --log2-size 22

echo "  Saved: $RESULTS_DIR/nsys_sequential.nsys-rep"
echo ""

# 2. Pipelined (async H2D + compute overlap)
echo "Capturing async pipeline..."
nsys profile \
    --trace=cuda,nvtx \
    --output="$RESULTS_DIR/nsys_async_pipeline" \
    --force-overwrite true \
    "$BINARY" --mode async --log2-size 22

echo "  Saved: $RESULTS_DIR/nsys_async_pipeline.nsys-rep"
echo ""

echo "=== Done ==="
echo ""
echo "Open in Nsight Systems:"
echo "  nsys-ui $RESULTS_DIR/nsys_sequential.nsys-rep"
echo "  nsys-ui $RESULTS_DIR/nsys_async_pipeline.nsys-rep"
echo ""
echo "In the GUI:"
echo "  1. Expand the CUDA row to show both H2D copy and kernel rows"
echo "  2. For async: verify that the copy (purple) and compute (green) bars overlap"
echo "  3. Take a screenshot of the timeline region and save to:"
echo "     ../../results/screenshots/nsys_sequential.png"
echo "     ../../results/screenshots/nsys_pipeline_overlap.png"
