"""Shared pieces: shared utility network, replay buffer, evaluation/metrics, generic off-policy training loop (GPU-ready)."""
import os, sys, json, time, logging, argparse, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from mapf_env import R_TEAM, GAMMA
from fast_env import FastEnv, OBS_DIM, N_ACT


class AgentNet(nn.Module):
    """Shared Q_theta(o_i, id_i, .). Input is ONLY the local observation + agent index (no global state)."""
    def __init__(self, n_agents=5, hid=256):
        super().__init__()
        self.n = n_agents
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(32 * 25 + 8 + n_agents, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, N_ACT))

    def forward(self, obs, idx):
        assert obs.shape[-1] == OBS_DIM
        fov = obs[:, :100].reshape(-1, 4, 5, 5)
        return self.mlp(torch.cat([self.conv(fov), obs[:, 100:], F.one_hot(idx, self.n).float()], 1))


class Replay:
    """Joint-transition buffer (float16 obs to save RAM); sample() returns tensors on `device`."""
    def __init__(self, cap, N, sdim, device):
        self.cap, self.i, self.n, self.dev, h = cap, 0, 0, device, np.float16
        self.d = dict(o=np.zeros((cap, N, OBS_DIM), h), no=np.zeros((cap, N, OBS_DIM), h), a=np.zeros((cap, N), np.int64),
                      r=np.zeros((cap, N), np.float32), tr=np.zeros(cap, np.float32), act=np.zeros((cap, N), bool),
                      nact=np.zeros((cap, N), bool), term=np.zeros(cap, np.float32), s=np.zeros((cap, sdim), h), ns=np.zeros((cap, sdim), h))

    def add(self, **kw):
        for k, v in kw.items(): self.d[k][self.i] = v
        self.i, self.n = (self.i + 1) % self.cap, min(self.n + 1, self.cap)

    def sample(self, B, rng):
        ix = np.sort(rng.integers(0, self.n, B))
        return {k: torch.from_numpy(v[ix].astype(np.float32) if v.dtype == np.float16 else v[ix]).to(self.dev, non_blocking=True) for k, v in self.d.items()}


def episode_metrics(env):
    ft = [int(f) if f >= 0 else env.T for f in env.finish_step]
    maxd = max(env.dist[i][tuple(env.starts[i])] for i in range(env.N))
    return dict(success=float((env.finish_step >= 0).all()), makespan=max(ft), sum_costs=sum(ft), collisions=env.collisions, norm_makespan=max(ft) / maxd)


def aggregate(rows):
    ok = [r["success"] == 1 for r in rows]
    g = lambda k, m=None: float(np.mean([r[k] for r, o in zip(rows, m or [True] * len(rows)) if o])) if (m is None or any(m)) else float("nan")
    return dict(success_rate=g("success"), makespan_succ=g("makespan", ok), makespan_all=g("makespan"), sum_costs=g("sum_costs"),
                collisions=g("collisions"), norm_makespan_succ=g("norm_makespan", ok), n_episodes=len(rows))


def evaluate(policy, seeds, reward="sparse", slip=0.1):
    """policy: obs (N,108) np -> actions (N,). Greedy, slip on."""
    env, rows = FastEnv(slip_prob=slip, reward_mode=reward), []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = trunc = False
        while not (done or trunc):
            obs, _, _, _, done, trunc, _ = env.step_arrays(policy(obs))
        rows.append(episode_metrics(env))
    return aggregate(rows)


def greedy(q):
    dev = next(q.parameters()).device; ids = torch.arange(q.n, device=dev)
    def pol(obs):
        with torch.no_grad(): return q(torch.from_numpy(obs).to(dev), ids).argmax(1).cpu().numpy()
    return pol


def get_args(extra=None):
    p = argparse.ArgumentParser()
    p.add_argument("--reward", default="sparse", choices=["sparse", "shaped"]); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=1_000_000); p.add_argument("--eval_every", type=int, default=50_000)
    p.add_argument("--log_every", type=int, default=10_000); p.add_argument("--eval_eps", type=int, default=50); p.add_argument("--final_eps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--batch", type=int, default=128)
    p.add_argument("--train_every", type=int, default=4); p.add_argument("--buf", type=int, default=100_000)
    p.add_argument("--learn_start", type=int, default=5000); p.add_argument("--target_every", type=int, default=1000)
    p.add_argument("--eps_end", type=float, default=0.05); p.add_argument("--eps_decay", type=int, default=300_000)
    for a in (extra or []): p.add_argument(*a[0], **a[1])
    return p.parse_args()


def make_logger(name):
    log = logging.getLogger(name); log.setLevel(logging.INFO); log.handlers.clear(); log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(f"results/{name}/run.log", mode="w"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    return log


def fmt_t(sec):
    sec = int(sec); return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def run(args, name, q, update_fn, save_extra=None):
    """Generic off-policy loop. update_fn(batch, n_updates) -> loss dict. Acting only calls q(obs, id): no global state."""
    torch.set_num_threads(1); torch.manual_seed(args.seed); torch.backends.cudnn.benchmark = True
    rng, dev = np.random.default_rng(args.seed), args.device
    os.makedirs(f"results/{name}", exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
    json.dump(vars(args), open(f"results/{name}/config.json", "w"), indent=1)
    log = make_logger(name)
    env = FastEnv(reward_mode=args.reward); N = env.N
    buf, ids = Replay(args.buf, N, env.state_dim, dev), torch.arange(N, device=dev)
    n_params = sum(p.numel() for p in q.parameters())
    log.info(f"START {name} | device={dev} reward={args.reward} seed={args.seed} steps={args.steps:,} params(Q)={n_params:,}")
    log.info(f"CONFIG {json.dumps(vars(args))}")

    loss_file, curve = open(f"results/{name}/train_log.jsonl", "w"), []
    def new_ep():  # training seeds from [1e6,1e9): never the eval seeds 10000-10099
        obs, _ = env.reset(seed=int(rng.integers(10**6, 10**9))); return obs, env.state_vec()
    obs, st = new_ep()
    n_upd, t0, ep_ret, ep_len = 0, time.time(), 0.0, 0
    ep_hist = collections.deque(maxlen=100)          # (team return, length, success) of last 100 training episodes
    acc, n_ep_total, last_eval = collections.defaultdict(float), 0, float("nan")
    bar = tqdm(total=args.steps, desc=f"{name:<16}", unit="step", dynamic_ncols=True, smoothing=0.05,
               bar_format="{desc} |{bar:25}| {percentage:5.1f}% [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for t in range(1, args.steps + 1):
        eps = max(args.eps_end, 1 - (1 - args.eps_end) * t / args.eps_decay)
        with torch.no_grad(): a = q(torch.from_numpy(obs).to(dev), ids).argmax(1).cpu().numpy()
        rnd = rng.random(N) < eps; a[rnd] = rng.integers(0, N_ACT, int(rnd.sum()))
        nobs, r, bonus, team, done, trunc, info = env.step_arrays(a)
        nst = env.state_vec()
        buf.add(o=obs, no=nobs, a=a, r=r, tr=team, act=info["active"], nact=~env.finished, term=float(done), s=st, ns=nst)
        obs, st, ep_ret, ep_len = nobs, nst, ep_ret + team, ep_len + 1
        if done or trunc:
            ep_hist.append((ep_ret, ep_len, float(done))); n_ep_total += 1; ep_ret, ep_len = 0.0, 0
            obs, st = new_ep()

        if t >= args.learn_start and t % args.train_every == 0:
            losses = update_fn(buf.sample(args.batch, rng), n_upd); n_upd += 1
            for k, v in losses.items(): acc[k] += v
            if n_upd % 500 == 0:
                loss_file.write(json.dumps(dict(step=t, eps=eps, **{k: v / 500 for k, v in acc.items()})) + "\n"); loss_file.flush()
                bar.set_postfix(eps=f"{eps:.2f}", loss=f"{acc['loss'] / 500:.3f}", train_succ=f"{np.mean([h[2] for h in ep_hist]):.2f}", eval_succ=f"{last_eval:.2f}", refresh=False)
                acc.clear()
        if t % 500 == 0: bar.update(500)

        if t % args.log_every == 0:
            el, h = time.time() - t0, np.array(ep_hist) if ep_hist else np.zeros((1, 3))
            log.info(f"TRAIN t={t:,} ({100 * t / args.steps:.1f}%) | eps={eps:.3f} updates={n_upd:,} episodes={n_ep_total:,} | "
                     f"last100: team_ret={h[:, 0].mean():.2f} len={h[:, 1].mean():.1f} succ={h[:, 2].mean():.2f} | "
                     f"{t / el:.0f} steps/s elapsed={fmt_t(el)} eta={fmt_t(el / t * (args.steps - t))}")
            json.dump(dict(step=t, total=args.steps, pct=100 * t / args.steps, eta_s=el / t * (args.steps - t), last_eval_succ=last_eval),
                      open(f"results/{name}/progress.json", "w"))

        if t % args.eval_every == 0 or t == args.steps:
            m = evaluate(greedy(q), range(10000, 10000 + args.eval_eps), args.reward); m["step"] = t; curve.append(m)
            json.dump(curve, open(f"results/{name}/curve.json", "w")); last_eval = m["success_rate"]
            log.info(f"EVAL  t={t:,} ({args.eval_eps} eps) | success={m['success_rate']:.2f} makespan(succ)={m['makespan_succ']:.1f} "
                     f"makespan(all)={m['makespan_all']:.1f} sum_costs={m['sum_costs']:.1f} collisions={m['collisions']:.2f} "
                     f"norm_makespan={m['norm_makespan_succ']:.2f}")
    bar.close()

    final = evaluate(greedy(q), range(10000, 10000 + args.final_eps), args.reward)
    json.dump(final, open(f"results/{name}/final.json", "w"), indent=1)
    torch.save({"q": q.state_dict(), **(save_extra() if save_extra else {})}, f"checkpoints/{name}.pt")
    log.info(f"FINAL ({args.final_eps} eps) | " + " ".join(f"{k}={v:.3f}" for k, v in final.items()))
    log.info(f"DONE {name} in {fmt_t(time.time() - t0)} | checkpoint: checkpoints/{name}.pt")