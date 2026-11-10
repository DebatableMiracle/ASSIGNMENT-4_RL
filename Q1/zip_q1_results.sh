#!/bin/bash
# Zip all Q1 results, checkpoints, logs, and plots
cd "$(dirname "$0")"

OUT_ZIP="../q1_results.zip"

echo "Creating $OUT_ZIP..."
zip -r "$OUT_ZIP" \
    results/ \
    checkpoints/ \
    logs/ \
    *.png \
    *.gif \
    *.json \
    -x "*.git*" 2>/dev/null

echo "Done! Archive saved at $(realpath "$OUT_ZIP")"

