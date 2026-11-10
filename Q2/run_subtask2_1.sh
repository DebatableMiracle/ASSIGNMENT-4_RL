#!/bin/bash
# Subtask 2.1: Run all 3x3 decoupled scaling matrix combinations
# 3 algorithms x 2 datasets x 9 matrix entries = 54 policy extractions
# (reusing 18 trained critics across policy fractions)

ALGOS=("td3bc" "iql_awr" "cql_awr")
DATASETS=("hopper" "kitchen")
FRACS=("0.1" "0.5" "1.0")

mkdir -p results checkpoints logs

for ds in "${DATASETS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for c_frac in "${FRACS[@]}"; do
      for p_frac in "${FRACS[@]}"; do
        echo "=========================================================="
        echo "Running: Algo=$algo | Dataset=$ds | CriticFrac=$c_frac | PolicyFrac=$p_frac"
        echo "=========================================================="
        python3 train_decoupled.py \
          --algo "$algo" \
          --dataset "$ds" \
          --critic_frac "$c_frac" \
          --policy_frac "$p_frac" \
          --device cuda \
          --reuse_critic
      done
    done
  done
done

echo "All Subtask 2.1 Decoupled Scaling Matrix runs completed!"

