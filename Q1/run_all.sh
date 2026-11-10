#!/bin/bash
# 2 algos x 2 reward variants x 3 seeds = 12 runs, 1e6 joint steps each, all on one GPU, N_PAR at a time (env is CPU-bound -> match your cores)
N_PAR=${N_PAR:-6}
mkdir -p logs
for algo in train_iql train_ctde; do for rw in sparse shaped; do for s in 0 1 2; do
  echo "python $algo.py --reward $rw --seed $s --device cuda > logs/${algo}_${rw}_$s.txt 2>&1"
done; done; done | xargs -P $N_PAR -I{} bash -c "{}"
