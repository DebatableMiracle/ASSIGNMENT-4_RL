"""Subtask 2.2: Implicit Q-Learning (IQL) from scratch on mujoco/hopper/medium-v0.

Reference: Kostrikov et al., ICLR 2022.
Implements:
  - Twin Q-networks with target networks (Min-Q)
  - Expectile Value regression: L_V(ψ) = E |τ - 1(Q - V < 0)| (Q_min(s, a) - V(s))^2
  - Advantage-weighted policy extraction (AWR): L_π(θ) = -E [ exp(β * (Q - V)) * log π(a|s) ]
  - Joint training for 1,000,000 gradient steps per seed (5 seeds required: 0..4).
  - Periodic evaluation, logging losses, final return table, and visualization rollout GIF.
"""

import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import time
import json
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import minari
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (Kostrikov et al., 2022 benchmark defaults for Hopper)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_EXPECTILE = 0.7       # tau for V-loss
DEFAULT_TEMPERATURE = 3.0     # beta for policy extraction
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_VALUE = 3e-4
BATCH_SIZE = 256
HIDDEN_DIM = 256
GAMMA = 0.99
TAU = 0.005                   # Soft target update rate
CLIP_MAX_ADV = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Neural Networks
# ─────────────────────────────────────────────────────────────────────────────
def mlp(input_dim, output_dim, hidden_dim=HIDDEN_DIM, num_layers=2, act_fn=nn.ReLU):
    layers = []
    in_d = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_d, hidden_dim))
        layers.append(act_fn())
        in_d = hidden_dim
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    """V(s) state-value network."""
    def __init__(self, obs_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.v = mlp(obs_dim, 1, hidden_dim=hidden_dim, num_layers=2)

    def forward(self, obs):
        return self.v(obs)


class TwinQNetwork(nn.Module):
    """Twin Q-networks Q1(s, a) and Q2(s, a)."""
    def __init__(self, obs_dim, act_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden_dim=hidden_dim, num_layers=2)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden_dim=hidden_dim, num_layers=2)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian Actor with learnable independent log_std."""
    def __init__(self, obs_dim, act_dim, max_action=1.0, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.max_action = float(max_action)
        self.backbone = mlp(obs_dim, hidden_dim, hidden_dim=hidden_dim, num_layers=2)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        feat = F.relu(self.backbone(obs))
        mean = self.mean_head(feat)
        return mean

    def get_dist(self, obs):
        mean = self.forward(obs)
        std = self.log_std.clamp(-20, 2).exp()
        return Normal(mean, std)

    def act(self, obs, deterministic=True):
        dist = self.get_dist(obs)
        if deterministic:
            a = dist.mean
        else:
            a = dist.sample()
        return torch.clamp(a, -self.max_action, self.max_action)

    def log_prob(self, obs, action):
        dist = self.get_dist(obs)
        return dist.log_prob(action).sum(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# Offline Replay Dataset
# ─────────────────────────────────────────────────────────────────────────────
class OfflineDataset:
    def __init__(self, dataset_id="mujoco/hopper/medium-v0", device="cpu"):
        print(f"Loading Minari dataset: {dataset_id}...")
        ds = minari.load_dataset(dataset_id, download=True)
        obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []

        for ep in ds.iterate_episodes():
            # Observations have length T + 1
            o = ep.observations[:-1]
            no = ep.observations[1:]
            a = ep.actions
            r = ep.rewards
            # For IQL: only use true environment terminations for bootstrap mask
            # terminations indicates absorbing terminal, truncations is timeout
            term = ep.terminations

            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            next_obs_list.append(no)
            term_list.append(term)

        self.obs = torch.as_tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(np.concatenate(act_list, axis=0), dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(np.concatenate(rew_list, axis=0), dtype=torch.float32, device=device).unsqueeze(1)
        self.next_obs = torch.as_tensor(np.concatenate(next_obs_list, axis=0), dtype=torch.float32, device=device)
        self.terminals = torch.as_tensor(np.concatenate(term_list, axis=0), dtype=torch.float32, device=device).unsqueeze(1)

        self.size = len(self.obs)
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.actions.shape[1]
        print(f"Loaded {self.size:,} transitions (obs_dim={self.obs_dim}, act_dim={self.act_dim})")

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        idx_t = torch.as_tensor(idx, device=self.obs.device, dtype=torch.long)
        return (
            self.obs[idx_t],
            self.actions[idx_t],
            self.rewards[idx_t],
            self.next_obs[idx_t],
            self.terminals[idx_t],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Expectile Loss
# ─────────────────────────────────────────────────────────────────────────────
def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, (1.0 - expectile))
    return torch.mean(weight * (diff ** 2))


def soft_update(source, target, tau):
    for s_param, t_param in zip(source.parameters(), target.parameters()):
        t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation and Visualization
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(env_id, policy, n_episodes=10, seed=1000, device="cpu"):
    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act = policy.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_ret += float(r)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def record_policy_gif(env_id, policy, out_path, seed=42, max_steps=1000, device="cpu", fps=30):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    step = 0
    while not done and step < max_steps:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act = policy.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        frames.append(env.render())
        step += 1
    env.close()

    if frames and frames[0] is not None:
        pil_imgs = [Image.fromarray(f) for f in frames]
        pil_imgs[0].save(
            out_path,
            save_all=True,
            append_images=pil_imgs[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved GIF: {out_path} ({len(frames)} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# Main Joint Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_iql(seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    results_dir = os.path.join(args.output_dir, f"iql_seed{seed}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    dataset = OfflineDataset(dataset_id=args.dataset_id, device=device)
    env_id = args.env_id

    # Networks
    v_net = ValueNetwork(dataset.obs_dim, hidden_dim=args.hidden_dim).to(device)
    q_net = TwinQNetwork(dataset.obs_dim, dataset.act_dim, hidden_dim=args.hidden_dim).to(device)
    q_target = TwinQNetwork(dataset.obs_dim, dataset.act_dim, hidden_dim=args.hidden_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    actor = GaussianPolicy(dataset.obs_dim, dataset.act_dim, max_action=args.max_action, hidden_dim=args.hidden_dim).to(device)

    opt_v = torch.optim.Adam(v_net.parameters(), lr=args.lr_v)
    opt_q = torch.optim.Adam(q_net.parameters(), lr=args.lr_q)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)

    lr_scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(opt_actor, T_max=args.total_steps)

    print(f"\n[Seed {seed}] Training IQL for {args.total_steps:,} steps on {device}...")
    curve = []
    loss_log = open(os.path.join(results_dir, "losses.jsonl"), "w")

    t0 = time.time()
    accum_v_loss, accum_q_loss, accum_pi_loss, accum_adv = 0.0, 0.0, 0.0, 0.0
    log_interval = args.log_every

    for step in range(1, args.total_steps + 1):
        obs, actions, rewards, next_obs, terminals = dataset.sample(args.batch_size, rng)

        # ── 1. Value Network Update (Expectile Regression) ────────────────────
        with torch.no_grad():
            t_q1, t_q2 = q_target(obs, actions)
            target_q = torch.min(t_q1, t_q2)

        v = v_net(obs)
        v_loss = expectile_loss(target_q - v, args.expectile)

        opt_v.zero_grad()
        v_loss.backward()
        opt_v.step()

        # ── 2. Twin Q Update (Fitted Q-iteration against V) ───────────────────
        with torch.no_grad():
            next_v = v_net(next_obs)
            target_y = rewards + (1.0 - terminals) * args.gamma * next_v

        q1, q2 = q_net(obs, actions)
        q_loss = F.mse_loss(q1, target_y) + F.mse_loss(q2, target_y)

        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        # Soft update target Q
        soft_update(q_net, q_target, args.tau)

        # ── 3. Policy Update (Advantage Weighted Regression) ──────────────────
        with torch.no_grad():
            adv = target_q - v
            exp_adv = torch.exp(adv / args.temperature).clamp(max=CLIP_MAX_ADV)

        log_probs = actor.log_prob(obs, actions)
        pi_loss = -torch.mean(exp_adv * log_probs)

        opt_actor.zero_grad()
        pi_loss.backward()
        opt_actor.step()
        lr_scheduler_actor.step()

        accum_v_loss += v_loss.item()
        accum_q_loss += q_loss.item()
        accum_pi_loss += pi_loss.item()
        accum_adv += adv.mean().item()

        # Periodic logging
        if step % log_interval == 0:
            loss_entry = {
                "step": step,
                "v_loss": accum_v_loss / log_interval,
                "q_loss": accum_q_loss / log_interval,
                "pi_loss": accum_pi_loss / log_interval,
                "mean_adv": accum_adv / log_interval,
            }
            loss_log.write(json.dumps(loss_entry) + "\n")
            loss_log.flush()
            accum_v_loss, accum_q_loss, accum_pi_loss, accum_adv = 0.0, 0.0, 0.0, 0.0

        # Periodic evaluation
        if step % args.eval_every == 0 or step == args.total_steps:
            ret_mean, ret_std = evaluate_policy(env_id, actor, n_episodes=args.eval_episodes, seed=1000, device=device)
            elapsed = time.time() - t0
            steps_per_sec = step / max(1e-5, elapsed)
            print(f"[Seed {seed}] Step {step:7d}/{args.total_steps} | Return: {ret_mean:7.2f} ± {ret_std:6.2f} | Speed: {steps_per_sec:5.1f} steps/s")
            curve.append({"step": step, "return_mean": ret_mean, "return_std": ret_std})
            with open(os.path.join(results_dir, "eval_curve.json"), "w") as f:
                json.dump(curve, f, indent=2)

    loss_log.close()

    # Final evaluation on 20 episodes per instructions
    final_mean, final_std = evaluate_policy(env_id, actor, n_episodes=20, seed=5000, device=device)
    print(f"\n[Seed {seed}] Final Return (20 eps): {final_mean:.2f} ± {final_std:.2f}")

    # Save checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, f"iql_seed{seed}.pt")
    torch.save({
        "actor": actor.state_dict(),
        "q_net": q_net.state_dict(),
        "v_net": v_net.state_dict(),
        "final_return_mean": final_mean,
        "final_return_std": final_std,
    }, ckpt_path)

    # Save rollout GIF: iql_seed{0..4}.gif
    gif_path = os.path.join(args.output_dir, f"iql_seed{seed}.gif")
    record_policy_gif(env_id, actor, gif_path, seed=42 + seed, device=device)

    summary = {
        "seed": seed,
        "final_return_mean": final_mean,
        "final_return_std": final_std,
        "checkpoint": ckpt_path,
        "gif": gif_path,
    }
    with open(os.path.join(results_dir, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Subtask 2.2: IQL from scratch")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seeds (requires 5)")
    parser.add_argument("--dataset_id", type=str, default="mujoco/hopper/medium-v0")
    parser.add_argument("--env_id", type=str, default="Hopper-v4")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--expectile", type=float, default=DEFAULT_EXPECTILE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--lr_v", type=float, default=LR_VALUE)
    parser.add_argument("--lr_q", type=float, default=LR_CRITIC)
    parser.add_argument("--lr_actor", type=float, default=LR_ACTOR)
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=50_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1_000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    all_finals = []
    for s in args.seeds:
        res = train_iql(s, args)
        all_finals.append(res)

    print("\n================== Final Return Summary across 5 Seeds ==================")
    for r in all_finals:
        print(f"Seed {r['seed']}: Return = {r['final_return_mean']:.2f} ± {r['final_return_std']:.2f}")

    means = [r["final_return_mean"] for r in all_finals]
    print(f"Overall Mean ± Std: {np.mean(means):.2f} ± {np.std(means):.2f}")


if __name__ == "__main__":
    main()

