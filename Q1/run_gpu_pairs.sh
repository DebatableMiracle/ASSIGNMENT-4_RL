#!/bin/bash
# Run all 12 training runs on GPU in pairs (2 at a time)
# GPU speed: ~1300 steps/sec solo, ~800 each in pairs → ~21 min/run → ~2h total
# This is 8x faster than the 12-CPU-with-set_num_threads(1) approach (5.5h each)
mkdir -p logs results checkpoints

STEPS=1000000
DEV=cuda

# All 12 runs as "algo rw seed" tuples
RUNS=(
  "train_iql sparse 0"
  "train_iql sparse 1"
  "train_iql sparse 2"
  "train_iql shaped 0"
  "train_iql shaped 1"
  "train_iql shaped 2"
  "train_ctde sparse 0"
  "train_ctde sparse 1"
  "train_ctde sparse 2"
  "train_ctde shaped 0"
  "train_ctde shaped 1"
  "train_ctde shaped 2"
)

total=${#RUNS[@]}
echo "[$(date +%H:%M:%S)] Starting $total runs in pairs on $DEV"

for ((i=0; i<total; i+=2)); do
  r1="${RUNS[$i]}"
  r2="${RUNS[$((i+1))]}"
  read algo1 rw1 s1 <<< "$r1"
  read algo2 rw2 s2 <<< "$r2"

  log1="logs/${algo1}_${rw1}_${s1}.txt"
  log2="logs/${algo2}_${rw2}_${s2}.txt"

  echo "[$(date +%H:%M:%S)] Pair $((i/2+1))/6: ${algo1}_${rw1}_s${s1} & ${algo2}_${rw2}_s${s2}"
  python3 ${algo1}.py --reward $rw1 --seed $s1 --device $DEV --steps $STEPS > $log1 2>&1 &
  PID1=$!
  python3 ${algo2}.py --reward $rw2 --seed $s2 --device $DEV --steps $STEPS > $log2 2>&1 &
  PID2=$!

  wait $PID1 && echo "  ✓ ${algo1}_${rw1}_s${s1} done" || echo "  ✗ ${algo1}_${rw1}_s${s1} FAILED"
  wait $PID2 && echo "  ✓ ${algo2}_${rw2}_s${s2} done" || echo "  ✗ ${algo2}_${rw2}_s${s2} FAILED"
done

echo "[$(date +%H:%M:%S)] ALL 12 DONE"

