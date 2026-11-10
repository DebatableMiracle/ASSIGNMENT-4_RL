"""Subtask 1.5: env_visualization.png, final_policy.gif (coordinated episode), failure GIFs + failure classification.
python analyze.py --ckpt checkpoints/qmix_shaped_s0.pt --slip 0.1 --episodes 100 [--reward shaped]
(Analysis only: eval seeds 10000+ are used to *inspect* a finished policy, never for training/selection.)"""
import argparse, collections, json, numpy as np, torch
from mapf_env import DR, DC
from fast_env import FastEnv
from common import AgentNet, greedy
from visualize import save_static, save_gif

def rollout(pol, seed, slip, reward, render=False):
    env = FastEnv(slip_prob=slip, reward_mode=reward, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed); done = trunc = False
    P, Fn, frames = [[tuple(p) for p in env.pos]], [list(env.finished)], [env.render()] if render else []
    while not (done or trunc):
        obs, _, _, _, done, trunc, _ = env.step_arrays(pol(obs)); P.append([tuple(p) for p in env.pos]); Fn.append(list(env.finished))
        if render: frames.append(env.render())
    env.close(); return env, P, Fn, frames

def free_path(env, s, g, blocked):  # analysis-only BFS (finished agents = walls); never used by a policy
    q, seen = collections.deque([s]), {s}
    while q:
        p = q.popleft()
        if p == g: return True
        for a in range(1, 5):
            n = (p[0] + int(DR[a]), p[1] + int(DC[a]))
            if 0 <= n[0] < env.H and 0 <= n[1] < env.W and env.grid[n] == 0 and n not in seen and n not in blocked: seen.add(n); q.append(n)
    return False

def classify(env, P, Fn):
    """goal_blocking > deadlock > livelock > timeout. goal blocking: final state; deadlock/livelock: anywhere in the episode."""
    N, L = env.N, len(P) - 1
    blocked = {P[-1][i] for i in range(N) if Fn[-1][i]}
    if any(not free_path(env, P[-1][i], tuple(env.goals[i]), blocked) for i in range(N) if not Fn[-1][i]): return "goal_blocking"
    run = 0
    for t in range(1, L + 1):
        unf = [i for i in range(N) if not Fn[t - 1][i]]
        run = 0 if any(P[t][i] != P[t - 1][i] for i in unf) else run + 1
        if run >= 10: return "deadlock"
    last, seen = 0, set()
    for t in range(L + 1):
        if t and Fn[t] != Fn[t - 1]: last, seen = t, set()
        cfg = tuple(P[t][i] for i in range(N) if not Fn[t][i])
        if cfg in seen and t - last >= 20: return "livelock"
        seen.add(cfg)
    return "timeout"

def yields(env, P, Fn):  # coordination score: unfinished agent waits while another unfinished agent is within 2 cells
    s = 0
    for t in range(1, len(P)):
        for i in range(env.N):
            if not Fn[t - 1][i] and P[t][i] == P[t - 1][i] and any(not Fn[t - 1][j] and abs(P[t][i][0] - P[t][j][0]) + abs(P[t][i][1] - P[t][j][1]) <= 2 for j in range(env.N) if j != i): s += 1
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--slip", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=100); ap.add_argument("--reward", default="sparse"); ap.add_argument("--nfail_gifs", type=int, default=5); a = ap.parse_args()
    q = AgentNet(5); q.load_state_dict(torch.load(a.ckpt, map_location="cpu")["q"]); pol = greedy(q)
    save_static("env_visualization.png")
    data = [(s, *rollout(pol, s, a.slip, a.reward)[:3]) for s in range(10000, 10000 + a.episodes)]
    ok = [d for d in data if all(d[3][-1])]; bad = [d for d in data if not all(d[3][-1])]
    if ok:
        s, P, Fn = max(((d[0], d[2], d[3]) for d in ok), key=lambda x: yields(FastEnv(), x[1], x[2]) - 0.01 * len(x[1]))
        save_gif(rollout(pol, s, a.slip, a.reward, render=True)[3], "final_policy.gif")
        print(f"coordinated episode: seed={s}, steps 0-{len(P)-1}, yield-steps={yields(FastEnv(), P, Fn)}")
    cnt, env0 = collections.Counter(), FastEnv()
    for k, (s, _, P, Fn) in enumerate(bad):
        c = classify(env0, P, Fn); cnt[c] += 1
        if k < a.nfail_gifs: save_gif(rollout(pol, s, a.slip, a.reward, render=True)[3], f"failure_{k}_{c}_seed{s}.gif")
    print(f"slip={a.slip}: {len(bad)}/{len(data)} failures", dict(cnt))
    json.dump({"n_fail": len(bad), "fractions": {k: v / max(1, len(bad)) for k, v in cnt.items()}}, open("failure_table.json", "w"), indent=1)
