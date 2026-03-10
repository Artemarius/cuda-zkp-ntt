#!/bin/bash
# collect_metrics.sh — Lightweight metric collection for all NTT variants
# Outputs a CSV summary for each variant and scale.
#
# Usage: bash collect_metrics.sh
# Output: ../results/data/metrics_summary.csv

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

OUTPUT="$RESULTS_DIR/metrics_summary.csv"

# CSV header
echo "mode,log2_size,stall_wait_avg,stall_math_throttle_avg,stall_not_selected_avg,branch_efficiency_pct,imad_instructions,iadd_instructions,achieved_occupancy_pct" > "$OUTPUT"

MODES=("naive" "optimized")
SIZES=(20 22 24)

for MODE in "${MODES[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        echo "Profiling: mode=$MODE size=2^$SIZE ..."

        # Collect key metrics in one ncu invocation
        METRICS="smsp__warp_cycles_per_issue_stall_wait.avg"
        METRICS+=",smsp__warp_cycles_per_issue_stall_math_throttle.avg"
        METRICS+=",smsp__warp_cycles_per_issue_stall_not_selected.avg"
        METRICS+=",smsp__sass_average_branch_targets_threads_uniform.pct"
        METRICS+=",sass__inst_executed_op_imad.sum"
        METRICS+=",sass__inst_executed_op_iadd.sum"
        METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"

        RAW=$(ncu \
            --metrics "$METRICS" \
            --csv \
            --target-processes all \
            "$BINARY" --mode "$MODE" --log2-size "$SIZE" 2>/dev/null | \
            grep -v "^==" | tail -n 1)

        # Parse values from ncu CSV output
        STALL_WAIT=$(echo "$RAW"       | awk -F',' '{print $5}')
        STALL_MATH=$(echo "$RAW"       | awk -F',' '{print $6}')
        STALL_NOTSEL=$(echo "$RAW"     | awk -F',' '{print $7}')
        BRANCH_EFF=$(echo "$RAW"       | awk -F',' '{print $8}')
        IMAD=$(echo "$RAW"             | awk -F',' '{print $9}')
        IADD=$(echo "$RAW"             | awk -F',' '{print $10}')
        OCCUPANCY=$(echo "$RAW"        | awk -F',' '{print $11}')

        echo "$MODE,$SIZE,$STALL_WAIT,$STALL_MATH,$STALL_NOTSEL,$BRANCH_EFF,$IMAD,$IADD,$OCCUPANCY" >> "$OUTPUT"
    done
done

echo ""
echo "Metrics saved to: $OUTPUT"
echo ""
echo "Summary:"
column -t -s',' "$OUTPUT"
