# AIL722 / AIL7022: Reinforcement Learning — Assignment 4

**Student Submission & Execution Guide**

This codebase covers:
1. **Problem 1 (Dec-POMDP MAPF)**: Decentralized path finding for $N=5$ agents on an $11\times11$ grid with partial observability (5x5 FOV), slip noise ($\varepsilon=0.1$), and absorbing goals.
2. **Problem 2 (Offline RL & Decoupled Learning)**: Decoupled Value & Policy learning across TD3+BC, IQL+AWR, and CQL+AWR on Minari datasets, and from-scratch IQL implementation.

---

## 1. Environment Setup

```bash
# Core requirements
pip install torch numpy matplotlib pillow gymnasium pettingzoo minari "gymnasium[mujoco]" "gymnasium-robotics>=1.2.4" reportlab
```

---

## 2. Problem 1 (`Q1/`)

Navigate to `Q1`:
```bash
cd Q1
```

### Subtasks & Execution

- **Subtask 1.2 — Unit Tests**:
  Verify the environment rules (wall blocks, vertex conflicts, swap conflicts, following, cascade resolution, rotation, absorbing goals, reward structures):
  ```bash
  python3 tests.py
  ```

- **Subtasks 1.3 & 1.4 — Training (IQL & QMIX across Sparse & Shaped Rewards, 3 Seeds each)**:
  - Run all 12 jobs in parallel (recommended on an RTX 5060, ~4 parallel streams):
    ```bash
    N_PAR=4 bash run_all.sh
    ```
  - Or run jobs sequentially in GPU pairs:
    ```bash
    bash run_gpu_pairs.sh
    ```
  - Monitor training live:
    ```bash
    python3 status.py
    ```

- **Subtask 1.5 — Plots, Metrics Table & GIF Generation**:
  Once training completes:
  ```bash
  python3 make_plots.py
  ```
  This creates:
  - `results/learning_curves_success.png` & `results/learning_curves_makespan.png`
  - `results/metrics_table.json`
  - `final_policy.gif` (demonstrating coordinated yielding)
  - `failure_table.json` & failure GIFs

---

## 3. Problem 2 (`Q2/`)

Navigate to `Q2`:
```bash
cd Q2
```

### Subtasks & Execution

- **Subtask 2.1 — Decoupled Value & Policy Learning**:
  Trains decoupled critics on $\{10\%, 50\%, 100\%\}$ data and extracts policies on $\{10\%, 50\%, 100\%\}$ data:
  ```bash
  # Automatically runs all 3x3 scaling matrix combinations:
  bash run_subtask2_1.sh
  ```
  *(Reuses trained critic checkpoints across policy fractions to eliminate redundant compute).*

- **Subtask 2.2 — Implicit Q-Learning (IQL) From Scratch**:
  Pure PyTorch joint training for $10^6$ gradient steps across 5 seeds:
  ```bash
  python3 train_iql_offline.py --seeds 0 1 2 3 4 --total_steps 1000000 --device cuda
  ```
  Outputs generated:
  - Checkpoints: `checkpoints/iql_seed{0..4}.pt`
  - Visualizations: `results/iql_seed{0..4}.gif`
  - Eval Curves: `results/iql_seed{0..4}/eval_curve.json`

- **Visualizations & Heatmaps**:
  ```bash
  python3 analyze_q2.py
  ```
  Produces `results/scaling_matrices.png` and `results/iql_learning_curve.png`.

---

## 4. Master Report Compilation

From the root directory:
```bash
python3 generate_pdf.py
```
Compiles and generates the final submission document: **`report.pdf`**.
