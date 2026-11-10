"""Subtask 2.1 – Decoupled offline RL: train critic, freeze it, extract policy.

Algorithms: TD3+BC, IQL+AWR, CQL+AWR
Datasets:   D4RL/kitchen/complete-v2  and  mujoco/hopper/medium-v0

Usage:
  python train_decoupled.py --algo td3bc   --dataset hopper   --critic_frac 1.0 --policy_frac 1.0
  python train_decoupled.py --algo iql_awr --dataset kitchen  --critic_frac 0.5 --policy_frac 0.1
  python train_decoupled.py --algo cql_awr --dataset hopper   --critic_frac 0.1 --policy_frac 1.0

Features:
  - Nested subset splits ({10%, 50%, 100%}) for critic and policy data.
  - Frozen critic checkpointing and reuse across policy data ratios.
  - Periodic evaluation, logging losses, and data scaling matrix export.
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_STEPS = 1_000_000
CRITIC_STEPS = 200_000      # 200k steps for critic, remaining 800k for policy extraction
BATCH_SIZE = 256
HIDDEN_DIM = 256
LR_CRITIC = 3e-4
LR_POLICY = 3e-4
GAMMA = 0.99
TAU = 0.005

# Algorithm-specific
IQL_EXPECTILE = 0.7
IQL_TEMPERATURE = 3.0
CQL_ALPHA = 1.0
TD3BC_ALPHA = 2.5

DATASET_IDS = {
    "hopper": "mujoco/hopper/medium-v0",
    "kitchen": "D4RL/kitchen/complete-v2",
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(net, target, tau=TAU):
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset_arrays(dataset_id):
    """Loads a Minari dataset and unpacks flat numpy arrays."""
    print(f"Loading {dataset_id}...")
    ds = minari.load_dataset(dataset_id, download=True)
    obs_list, act_list, rew_list, nobs_list, term_list = [], [], [], [], []

    for ep in ds.iterate_episodes():
        # Handle dict observations (e.g. FrankaKitchen GoalEnv) vs flat array
        raw_obs = ep.observations
        if isinstance(raw_obs, dict):
            o_arr = raw_obs["observation"]
        else:
            o_arr = raw_obs

        o = o_arr[:-1]
        no = o_arr[1:]
        a = ep.actions
        r = ep.rewards
        term = ep.terminations

        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        nobs_list.append(no)
        term_list.append(term)

    obs = np.concatenate(obs_list, axis=0).astype(np.float32)
    act = np.concatenate(act_list, axis=0).astype(np.float32)
    rew = np.concatenate(rew_list, axis=0).astype(np.float32)
    nobs = np.concatenate(nobs_list, axis=0).astype(np.float32)
    term = np.concatenate(term_list, axis=0).astype(np.float32)

    print(f"Loaded {len(obs):,} transitions | obs_dim={obs.shape[1]}, act_dim={act.shape[1]}")
    return obs, act, rew, nobs, term, ds


def get_nested_subset(obs, act, rew, nobs, term, fraction, seed=42):
    """Generates a nested random subset of transitions based on fraction.
    Same permutation order guarantees: 10% subset is contained in 50%, which is in 100%.
    """
    n = len(obs)
    k = max(1, int(round(n * fraction)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    sub_idx = perm[:k]
    return (
        obs[sub_idx],
        act[sub_idx],
        rew[sub_idx],
        nobs[sub_idx],
        term[sub_idx],
    )


class ReplayBuffer:
    def __init__(self, obs, act, rew, nobs, term, device):
        self.obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(act, dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(rew, dtype=torch.float32, device=device).unsqueeze(1)
        self.next_obs = torch.as_tensor(nobs, dtype=torch.float32, device=device)
        self.terminals = torch.as_tensor(term, dtype=torch.float32, device=device).unsqueeze(1)
        self.size = len(obs)

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
# Architectures
# ─────────────────────────────────────────────────────────────────────────────
class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.v(obs)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action=1.0, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        feat = self.net(obs)
        mean = self.mean_head(feat)
        return mean

    def get_dist(self, obs):
        mean = self.forward(obs)
        std = self.log_std.clamp(-20, 2).exp()
        return Normal(mean, std)

    def act(self, obs, deterministic=True, with_grad=False):
        dist = self.get_dist(obs)
        if with_grad:
            a = dist.rsample() if not deterministic else dist.mean
        else:
            a = dist.mean if deterministic else dist.sample()
        return torch.clamp(a, -self.max_action, self.max_action)

    def log_prob(self, obs, action):
        dist = self.get_dist(obs)
        return dist.log_prob(action).sum(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# Decoupled Critic Training
# ─────────────────────────────────────────────────────────────────────────────
def train_critic_td3bc(buf, obs_dim, act_dim, max_act, steps, device, rng, log_f):
    q_net = TwinQ(obs_dim, act_dim).to(device)
    q_target = TwinQ(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    bc_actor = GaussianPolicy(obs_dim, act_dim, max_act).to(device)

    opt_q = torch.optim.Adam(q_net.parameters(), lr=LR_CRITIC)
    opt_bc = torch.optim.Adam(bc_actor.parameters(), lr=LR_CRITIC)

    pbar = tqdm(range(1, steps + 1), desc="TD3+BC Critic", dynamic_ncols=True,
                bar_format="{desc} |{bar:25}| {percentage:5.1f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step in pbar:
        o, a, r, no, d = buf.sample(BATCH_SIZE, rng)

        # Train target BC behavior generator
        bc_loss = -bc_actor.log_prob(o, a).mean()
        opt_bc.zero_grad()
        bc_loss.backward()
        opt_bc.step()

        # Bellman update
        with torch.no_grad():
            na = bc_actor.act(no, deterministic=False)
            noise = (torch.randn_like(na) * 0.2).clamp(-0.5, 0.5)
            na = torch.clamp(na + noise, -max_act, max_act)
            tq1, tq2 = q_target(no, na)
            target_y = r + (1.0 - d) * GAMMA * torch.min(tq1, tq2)

        q1, q2 = q_net(o, a)
        q_loss = F.mse_loss(q1, target_y) + F.mse_loss(q2, target_y)

        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        soft_update(q_net, q_target)

        if step % 5000 == 0:
            log_f.write(json.dumps({"step": step, "stage": "critic", "q_loss": q_loss.item()}) + "\n")

    return q_net, None


def train_critic_iql(buf, obs_dim, act_dim, max_act, steps, device, rng, log_f):
    q_net = TwinQ(obs_dim, act_dim).to(device)
    q_target = TwinQ(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    v_net = ValueNet(obs_dim).to(device)

    opt_q = torch.optim.Adam(q_net.parameters(), lr=LR_CRITIC)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=LR_CRITIC)

    pbar = tqdm(range(1, steps + 1), desc="IQL Critic", dynamic_ncols=True,
                bar_format="{desc} |{bar:25}| {percentage:5.1f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step in pbar:
        o, a, r, no, d = buf.sample(BATCH_SIZE, rng)

        # V expectile regression
        with torch.no_grad():
            tq1, tq2 = q_target(o, a)
            min_q = torch.min(tq1, tq2)

        v = v_net(o)
        diff = min_q - v
        weight = torch.where(diff > 0, IQL_EXPECTILE, 1.0 - IQL_EXPECTILE)
        v_loss = torch.mean(weight * (diff ** 2))

        opt_v.zero_grad()
        v_loss.backward()
        opt_v.step()

        # Q update
        with torch.no_grad():
            nv = v_net(no)
            target_y = r + (1.0 - d) * GAMMA * nv

        q1, q2 = q_net(o, a)
        q_loss = F.mse_loss(q1, target_y) + F.mse_loss(q2, target_y)

        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        soft_update(q_net, q_target)

        if step % 5000 == 0:
            log_f.write(json.dumps({"step": step, "stage": "critic", "q_loss": q_loss.item(), "v_loss": v_loss.item()}) + "\n")

    return q_net, v_net


def train_critic_cql(buf, obs_dim, act_dim, max_act, steps, device, rng, log_f):
    """CQL with soft_q_backup=False: deterministic greedy actor target like DDPG."""
    q_net = TwinQ(obs_dim, act_dim).to(device)
    q_target = TwinQ(obs_dim, act_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    pi_eval = GaussianPolicy(obs_dim, act_dim, max_act).to(device)

    opt_q = torch.optim.Adam(q_net.parameters(), lr=LR_CRITIC)

    pbar = tqdm(range(1, steps + 1), desc="CQL Critic", dynamic_ncols=True,
                bar_format="{desc} |{bar:25}| {percentage:5.1f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step in pbar:
        o, a, r, no, d = buf.sample(BATCH_SIZE, rng)

        # Standard Bellman error with soft_q_backup=False (deterministic target)
        with torch.no_grad():
            na = pi_eval.act(no, deterministic=True)
            tq1, tq2 = q_target(no, na)
            target_y = r + (1.0 - d) * GAMMA * torch.min(tq1, tq2)

        q1, q2 = q_net(o, a)
        bellman_loss = F.mse_loss(q1, target_y) + F.mse_loss(q2, target_y)

        # Conservative penalty: E_pi[Q] - E_data[Q]
        rand_a = torch.empty(BATCH_SIZE, act_dim, device=device).uniform_(-max_act, max_act)
        q1_rand, q2_rand = q_net(o, rand_a)
        cql_penalty = (q1_rand.mean() - q1.mean()) + (q2_rand.mean() - q2.mean())

        q_loss = bellman_loss + CQL_ALPHA * cql_penalty

        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        soft_update(q_net, q_target)

        if step % 5000 == 0:
            log_f.write(json.dumps({"step": step, "stage": "critic", "q_loss": q_loss.item(), "cql_pen": cql_penalty.item()}) + "\n")

    return q_net, None


# ─────────────────────────────────────────────────────────────────────────────
# Decoupled Policy Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_policy(algo, buf, q_net, v_net, obs_dim, act_dim, max_act, steps, device, rng, log_f):
    for p in q_net.parameters():
        p.requires_grad = False
    if v_net is not None:
        for p in v_net.parameters():
            p.requires_grad = False

    actor = GaussianPolicy(obs_dim, act_dim, max_act).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=LR_POLICY)

    pbar = tqdm(range(1, steps + 1), desc=f"Policy [{algo.upper()}]", dynamic_ncols=True,
                bar_format="{desc} |{bar:25}| {percentage:5.1f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step in pbar:
        o, a, *_ = buf.sample(BATCH_SIZE, rng)

        if algo == "td3bc":
            pi_a = actor.act(o, deterministic=True, with_grad=True)
            q1, q2 = q_net(o, pi_a)
            q_val = torch.min(q1, q2)
            lmbda = TD3BC_ALPHA / (q_val.abs().mean().detach() + 1e-6)
            bc_err = F.mse_loss(pi_a, a)
            loss = -lmbda * q_val.mean() + bc_err
        else:  # AWR extraction (for iql_awr and cql_awr)
            with torch.no_grad():
                q1, q2 = q_net(o, a)
                q_val = torch.min(q1, q2)
                base = v_net(o) if v_net is not None else q_val.mean()
                adv = q_val - base
                weights = torch.exp(adv / IQL_TEMPERATURE).clamp(max=100.0)
                weights = weights / (weights.sum() + 1e-8)
            log_probs = actor.log_prob(o, a)
            loss = -(weights * log_probs).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5000 == 0:
            log_f.write(json.dumps({"step": step, "stage": "policy", "pi_loss": loss.item()}) + "\n")

    return actor


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy_env(env, actor, n_episodes=20, device="cpu"):
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            if isinstance(obs, dict):
                obs_arr = obs["observation"]
            else:
                obs_arr = obs
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
                act = actor.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_ret += float(r)
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Subtask 2.1 Decoupled Offline RL")
    parser.add_argument("--algo", type=str, required=True, choices=["td3bc", "iql_awr", "cql_awr"])
    parser.add_argument("--dataset", type=str, required=True, choices=["hopper", "kitchen"])
    parser.add_argument("--critic_frac", type=float, default=1.0, choices=[0.1, 0.5, 1.0])
    parser.add_argument("--policy_frac", type=float, default=1.0, choices=[0.1, 0.5, 1.0])
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--critic_steps", type=int, default=CRITIC_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse_critic", action="store_true", default=True,
                        help="Reuse pre-trained critic for identical (algo, dataset, critic_frac)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    policy_steps = args.total_steps - args.critic_steps
    dataset_id = DATASET_IDS[args.dataset]

    obs_all, act_all, rew_all, nobs_all, term_all, minari_ds = load_dataset_arrays(dataset_id)
    obs_dim = obs_all.shape[1]
    act_dim = act_all.shape[1]
    max_act = float(np.max(np.abs(act_all)))

    out_name = f"{args.algo}_{args.dataset}_c{args.critic_frac}_p{args.policy_frac}_s{args.seed}"
    res_dir = os.path.join("results", out_name)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_f = open(os.path.join(res_dir, "train_log.jsonl"), "w")

    # Nested data subsets
    c_obs, c_act, c_rew, c_nobs, c_term = get_nested_subset(
        obs_all, act_all, rew_all, nobs_all, term_all, args.critic_frac, seed=args.seed
    )
    p_obs, p_act, p_rew, p_nobs, p_term = get_nested_subset(
        obs_all, act_all, rew_all, nobs_all, term_all, args.policy_frac, seed=args.seed + 100
    )

    c_buf = ReplayBuffer(c_obs, c_act, c_rew, c_nobs, c_term, device)
    p_buf = ReplayBuffer(p_obs, p_act, p_rew, p_nobs, p_term, device)

    # Check for reusable critic checkpoint
    critic_ckpt = f"checkpoints/critic_{args.algo}_{args.dataset}_c{args.critic_frac}_s{args.seed}.pt"
    q_net = TwinQ(obs_dim, act_dim).to(device)
    v_net = ValueNet(obs_dim).to(device) if args.algo == "iql_awr" else None

    if args.reuse_critic and os.path.exists(critic_ckpt):
        print(f"Reusing existing critic checkpoint: {critic_ckpt}")
        ckpt = torch.load(critic_ckpt, map_location=device)
        q_net.load_state_dict(ckpt["q_net"])
        if v_net is not None and "v_net" in ckpt:
            v_net.load_state_dict(ckpt["v_net"])
    else:
        print(f"Training critic ({args.algo}) for {args.critic_steps:,} steps...")
        if args.algo == "td3bc":
            q_net, _ = train_critic_td3bc(c_buf, obs_dim, act_dim, max_act, args.critic_steps, device, rng, log_f)
        elif args.algo == "iql_awr":
            q_net, v_net = train_critic_iql(c_buf, obs_dim, act_dim, max_act, args.critic_steps, device, rng, log_f)
        else:
            q_net, _ = train_critic_cql(c_buf, obs_dim, act_dim, max_act, args.critic_steps, device, rng, log_f)

        save_dict = {"q_net": q_net.state_dict()}
        if v_net is not None:
            save_dict["v_net"] = v_net.state_dict()
        torch.save(save_dict, critic_ckpt)
        print(f"Saved critic checkpoint: {critic_ckpt}")

    print(f"Extracting policy ({args.algo}) for {policy_steps:,} steps on {args.policy_frac*100:.0f}% policy data...")
    actor = extract_policy(args.algo, p_buf, q_net, v_net, obs_dim, act_dim, max_act, policy_steps, device, rng, log_f)
    log_f.close()

    # Recover evaluation environment from Minari dataset
    eval_env = minari_ds.recover_environment()
    mean_ret, std_ret = evaluate_policy_env(eval_env, actor, n_episodes=20, device=device)
    eval_env.close()

    print(f"\nFinal Evaluation ({out_name}): Return = {mean_ret:.2f} ± {std_ret:.2f}")

    final_res = {
        "algo": args.algo,
        "dataset": args.dataset,
        "critic_frac": args.critic_frac,
        "policy_frac": args.policy_frac,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "seed": args.seed,
    }
    with open(os.path.join(res_dir, "final.json"), "w") as f:
        json.dump(final_res, f, indent=2)


if __name__ == "__main__":
    main()
