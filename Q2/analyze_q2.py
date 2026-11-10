"""Analysis & Visualization utility for Problem 2.
Generates:
  - 3x3 scaling matrix heatmaps / tables for TD3+BC, IQL+AWR, CQL+AWR on Hopper & Kitchen.
  - Learning curves (mean ± std) across training steps.
  - Formats LaTeX tables and summary statistics for report.pdf.
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = ["td3bc", "iql_awr", "cql_awr"]
DATASETS = ["hopper", "kitchen"]
FRACS = [0.1, 0.5, 1.0]


def collect_scaling_matrix(algo, dataset, base_dir="results"):
    """Collects 3x3 scaling matrix: matrix[i, j] = return for critic_frac i, policy_frac j."""
    matrix = np.full((len(FRACS), len(FRACS)), np.nan)
    for i, c_frac in enumerate(FRACS):
        for j, p_frac in enumerate(FRACS):
            pattern = os.path.join(base_dir, f"{algo}_{dataset}_c{c_frac}_p{p_frac}*", "final.json")
            files = glob.glob(pattern)
            if files:
                try:
                    with open(files[0]) as f:
                        data = json.load(f)
                        matrix[i, j] = data.get("mean_return", np.nan)
                except Exception:
                    pass
    return matrix


def plot_scaling_heatmaps(out_path="results/scaling_matrices.png"):
    fig, axes = plt.subplots(len(DATASETS), len(ALGOS), figsize=(12, 7))
    for r, ds in enumerate(DATASETS):
        for c, algo in enumerate(ALGOS):
            ax = axes[r, c] if len(DATASETS) > 1 else axes[c]
            mat = collect_scaling_matrix(algo, ds)
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(FRACS)))
            ax.set_yticks(range(len(FRACS)))
            ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS])
            ax.set_yticklabels([f"{int(f*100)}%" for f in FRACS])
            ax.set_xlabel("Policy Data %")
            ax.set_ylabel("Critic Data %")
            ax.set_title(f"{algo.upper()} on {ds.capitalize()}")

            # Annotate numbers
            for i in range(len(FRACS)):
                for j in range(len(FRACS)):
                    val = mat[i, j]
                    txt = f"{val:.1f}" if not np.isnan(val) else "N/A"
                    ax.text(j, i, txt, ha="center", va="center", color="white" if val < np.nanmax(mat)/2 else "black", weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved scaling matrices heatmap: {out_path}")


def plot_iql_scratch_curves(base_dir="results", out_path="results/iql_learning_curve.png"):
    """Plots evaluation return curve across 5 seeds for Subtask 2.2."""
    seed_curves = []
    for s in range(5):
        curve_file = os.path.join(base_dir, f"iql_seed{s}", "eval_curve.json")
        if os.path.exists(curve_file):
            with open(curve_file) as f:
                seed_curves.append(json.load(f))

    if not seed_curves:
        print("No IQL eval curves found yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_steps = sorted(list({entry["step"] for curve in seed_curves for entry in curve}))
    step_returns = {step: [] for step in all_steps}

    for curve in seed_curves:
        for entry in curve:
            step_returns[entry["step"]].append(entry["return_mean"])

    means = [np.mean(step_returns[s]) for s in all_steps]
    stds = [np.std(step_returns[s]) for s in all_steps]

    ax.plot(all_steps, means, label="IQL (Scratch) Mean Return", color="#2CA02C", lw=2.5)
    ax.fill_between(all_steps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.25, color="#2CA02C")
    ax.set_xlabel("Gradient Steps", fontsize=12)
    ax.set_ylabel("Average Episodic Return", fontsize=12)
    ax.set_title("Subtask 2.2: IQL on Hopper-v4 (5 Seeds, Mean ± Std)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved IQL curve plot: {out_path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_scaling_heatmaps()
    plot_iql_scratch_curves()

