"""Generate all plots and tables for the Problem 1 report.
Run after training is complete:
    python make_plots.py
Outputs: results/learning_curves.png, results/metrics_table.json, env_visualization.png,
         final_policy.gif, failure_*.gif, failure_table.json
"""
import json, os, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from common import AgentNet, greedy
from visualize import save_static, save_gif
from fast_env import FastEnv
from analyze import rollout, classify, yields

COLORS = {"iql": {"sparse": "#1F77B4", "shaped": "#FF7F0E"},
          "qmix": {"sparse": "#2CA02C", "shaped": "#D62728"}}
LABELS = {"iql_sparse": "IQL sparse", "iql_shaped": "IQL shaped",
          "qmix_sparse": "QMIX sparse", "qmix_shaped": "QMIX shaped"}

SEEDS = [0, 1, 2]
CONFIGS = ["iql_sparse", "iql_shaped", "qmix_sparse", "qmix_shaped"]
METRICS = ["success_rate", "makespan_succ", "makespan_all", "sum_costs", "collisions", "norm_makespan_succ"]


def load_curves(configs=CONFIGS, seeds=SEEDS):
    """Returns dict config -> list-of-curves (one per seed), each curve is list of dicts."""
    data = {}
    for cfg in configs:
        algo, rw = cfg.split("_", 1)
        data[cfg] = []
        for s in seeds:
            path = f"results/{algo}_{rw}_s{s}/curve.json"
            if os.path.exists(path):
                data[cfg].append(json.load(open(path)))
            else:
                data[cfg].append(None)
    return data


def load_finals(configs=CONFIGS, seeds=SEEDS):
    data = {}
    for cfg in configs:
        algo, rw = cfg.split("_", 1)
        data[cfg] = []
        for s in seeds:
            path = f"results/{algo}_{rw}_s{s}/final.json"
            if os.path.exists(path):
                data[cfg].append(json.load(open(path)))
            else:
                data[cfg].append(None)
    return data


def plot_learning_curves(data, metric="success_rate", out="results/learning_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg in CONFIGS:
        algo, rw = cfg.split("_", 1)
        col = COLORS[algo][rw]
        label = LABELS[cfg]
        curves = [c for c in data[cfg] if c is not None]
        if not curves:
            continue
        # Align steps
        steps_set = set()
        for c in curves:
            for pt in c:
                steps_set.add(pt["step"])
        steps = sorted(steps_set)
        mat = []
        for c in curves:
            step2val = {pt["step"]: pt[metric] for pt in c}
            mat.append([step2val.get(s, float("nan")) for s in steps])
        mat = np.array(mat)
        mean = np.nanmean(mat, 0)
        std = np.nanstd(mat, 0)
        ax.plot(steps, mean, label=label, color=col, lw=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=col)
    ax.set_xlabel("Joint environment steps", fontsize=12)
    metric_name = metric.replace("_", " ").title()
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Learning curves — {metric_name} (mean ± std, 3 seeds)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def build_metrics_table(finals):
    """Print a LaTeX-style table and return as dict."""
    rows = []
    for cfg in CONFIGS:
        vals = [f for f in finals[cfg] if f is not None]
        if not vals:
            continue
        row = {"config": cfg}
        for m in METRICS:
            v = [x[m] for x in vals if not (isinstance(x[m], float) and x[m] != x[m])]
            if v:
                row[m] = (np.mean(v), np.std(v))
            else:
                row[m] = (float("nan"), float("nan"))
        rows.append(row)
    return rows


def print_table(rows):
    hdr = f"{'Config':<15} {'Success':>10} {'Makespan(S)':>13} {'Makespan(A)':>13} {'SumCosts':>12} {'Collisions':>12}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        def fmt(k):
            m, s = row[k]
            return f"{m:.3f}±{s:.3f}"
        print(f"{row['config']:<15} {fmt('success_rate'):>10} {fmt('makespan_succ'):>13} "
              f"{fmt('makespan_all'):>13} {fmt('sum_costs'):>12} {fmt('collisions'):>12}")


def run_analysis(best_ckpt, slip=0.1, n_episodes=100, n_fail_gifs=5, reward="sparse"):
    """Run subtask 1.5: static map, GIF, failure analysis."""
    save_static("env_visualization.png")
    q = AgentNet(5)
    ckpt = torch.load(best_ckpt, map_location="cpu")
    q.load_state_dict(ckpt["q"])
    pol = greedy(q)
    
    data = []
    for s in range(10000, 10000 + n_episodes):
        env, P, Fn, frames = rollout(pol, s, slip, reward, render=False)
        data.append((s, env, P, Fn))
    
    ok = [(s, env, P, Fn) for s, env, P, Fn in data if all(Fn[-1])]
    bad = [(s, env, P, Fn) for s, env, P, Fn in data if not all(Fn[-1])]
    
    # Best coordinated episode
    if ok:
        best = max(ok, key=lambda x: yields(x[1], x[2], x[3]) - 0.01 * len(x[2]))
        s, env, P, Fn = best
        _, _, _, frames = rollout(pol, s, slip, reward, render=True)
        save_gif(frames, "final_policy.gif")
        print(f"Coordinated episode: seed={s}, steps={len(P)-1}, yields={yields(env, P, Fn)}")
    
    # Failure analysis
    cnt, env0 = collections.Counter(), FastEnv()
    for k, (s, env, P, Fn) in enumerate(bad):
        c = classify(env0, P, Fn)
        cnt[c] += 1
        if k < n_fail_gifs:
            _, _, _, frames = rollout(pol, s, slip, reward, render=True)
            save_gif(frames, f"failure_{k}_{c}_seed{s}.gif")
    
    print(f"slip={slip}: {len(bad)}/{len(data)} failures", dict(cnt))
    result = {"n_fail": len(bad), "n_total": len(data),
              "fractions": {k: v / max(1, len(bad)) for k, v in cnt.items()}}
    json.dump(result, open("failure_table.json", "w"), indent=1)
    return result


def find_best_checkpoint():
    """Find the checkpoint with highest final success rate."""
    best_ckpt, best_succ = None, -1
    for path in glob.glob("results/*/final.json"):
        name = path.split("/")[1]
        d = json.load(open(path))
        if d["success_rate"] > best_succ:
            best_succ = d["success_rate"]
            best_ckpt = f"checkpoints/{name}.pt"
    return best_ckpt, best_succ


if __name__ == "__main__":
    print("=== Generating all plots and analysis ===")
    
    data = load_curves()
    finals = load_finals()
    
    # Plot all 4 configs on one figure
    plot_learning_curves(data, "success_rate", "results/learning_curves_success.png")
    plot_learning_curves(data, "makespan_all", "results/learning_curves_makespan.png")
    
    # Metrics table
    rows = build_metrics_table(finals)
    print("\n=== Final Metrics Table ===")
    print_table(rows)
    json.dump(rows, open("results/metrics_table.json", "w"), indent=2, default=str)
    
    # Static visualization
    save_static("env_visualization.png")
    
    # 1.5 Analysis - needs best checkpoint
    best_ckpt, best_succ = find_best_checkpoint()
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\nBest checkpoint: {best_ckpt} (success={best_succ:.2f})")
        algo = best_ckpt.split("/")[1].split("_")[0]  # iql or qmix
        rw = best_ckpt.split("/")[1].split("_")[1]
        result = run_analysis(best_ckpt, reward=rw)
        print("Failure analysis:", result)
    else:
        print("No checkpoint found yet - run after training completes")
    
    print("\nDone!")

