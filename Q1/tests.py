"""python tests.py -- hand-made configurations on the starter env, slip=0 unless stated."""
import numpy as np
from mapf_env import CustomMAPFParallelEnv as Env, R_GOAL, C_STEP, C_COL, R_TEAM, BETA, GAMMA
from fast_env import FastEnv, OBS_DIM

S, U, D, L, R = 0, 1, 2, 3, 4
def mk(grid, starts, goals, slip=0.0, mode="sparse", cls=Env):
    g = np.array([[1 if ch == "#" else 0 for ch in row] for row in grid])
    e = cls(slip_prob=slip, reward_mode=mode, layout=(g, starts, goals)); e.reset(seed=0); return e
def st(e, *acts):
    o, r, te, tr, info = e.step({f"agent_{i}": a for i, a in enumerate(acts)})
    return np.array(list(r.values())), np.array([info[n]["blocked"] for n in e.possible_agents]), te, info
def P(e): return [tuple(p) for p in e.pos]
def close(a, b): return abs(a - b) < 1e-5

def test_wall():  # (i) wall and outside grid: blocked + penalised
    e = mk(["..#"], [(0, 1)], [(0, 0)])
    r, b, *_ = st(e, R); assert P(e) == [(0, 1)] and b[0] and close(r[0], C_STEP + C_COL)
    r, b, *_ = st(e, U); assert P(e) == [(0, 1)] and b[0]
def test_vertex():  # (ii) same target -> both blocked; blocked by a standing agent -> only the mover counted
    e = mk(["..."], [(0, 0), (0, 2)], [(0, 2), (0, 0)])
    r, b, *_ = st(e, R, L); assert P(e) == [(0, 0), (0, 2)] and b.all()
    e = mk(["..."], [(0, 0), (0, 1)], [(0, 2), (0, 0)])
    r, b, *_ = st(e, R, S); assert P(e) == [(0, 0), (0, 1)] and list(b) == [True, False]
def test_swap():  # (iii)
    e = mk(["..."], [(0, 0), (0, 1)], [(0, 2), (0, 0)])
    r, b, *_ = st(e, R, L); assert P(e) == [(0, 0), (0, 1)] and b.all()
def test_following():  # (iv) enter a cell vacated in the same step
    e = mk(["....."], [(0, 0), (0, 1)], [(0, 4), (0, 3)])
    r, b, *_ = st(e, R, R); assert P(e) == [(0, 1), (0, 2)] and not b.any()
def test_cascade():  # (v) wall stops front agent -> blocking propagates along the chain of three
    e = mk(["...#.", "....."], [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)])
    r, b, *_ = st(e, R, R, R); assert P(e) == [(0, 0), (0, 1), (0, 2)] and b.all()
def test_rotation():  # (vi) 4 agents rotate on a 2x2 block
    e = mk(["..", ".."], [(0, 0), (0, 1), (1, 1), (1, 0)], [(1, 1), (1, 0), (0, 0), (0, 1)])
    r, b, *_ = st(e, R, D, L, U); assert P(e) == [(0, 1), (1, 1), (1, 0), (0, 0)] and not b.any()
def test_finished_blocks():  # (vii) finished agent stays, gets 0 reward, blocks others, is never counted
    e = mk(["..."], [(0, 0), (0, 2)], [(0, 1), (0, 0)])
    r, b, *_ = st(e, R, S); assert e.finished[0] and close(r[0], R_GOAL)
    r, b, *_ = st(e, R, L); assert P(e) == [(0, 1), (0, 2)] and list(b) == [False, True] and r[0] == 0
def test_full_slip():  # (viii) slip=1: nobody moves, nobody penalised for collisions
    e = Env(slip_prob=1.0); e.reset(seed=3); starts = P(e)
    for _ in range(5):
        r, b, *_ = st(e, *np.random.default_rng(0).integers(1, 5, 5)); assert P(e) == starts and not b.any() and np.allclose(r, C_STEP)
def test_rewards():  # (ix) goal, step, collision, team bonus, shaping identity
    for mode in ("sparse", "shaped"):
        e, sh = mk(["....."], [(0, 0)], [(0, 2)], mode=mode), mode == "shaped"
        r, b, te, info = st(e, L); assert close(r[0], C_STEP + C_COL + (BETA * (GAMMA * -2 + 2) if sh else 0)) and not te["agent_0"]
        r, *_ = st(e, R); assert close(r[0], C_STEP + (BETA * (GAMMA * -1 + 2) if sh else 0))
        r, b, te, info = st(e, R); g = R_GOAL + (BETA * 1 if sh else 0)
        assert te["agent_0"] and close(r[0], g) and info["agent_0"]["team_bonus"] == R_TEAM and close(info["agent_0"]["team_reward"], g + R_TEAM)
def test_determinism():  # (x) equal seed + actions -> identical trajectories
    outs = []
    for _ in range(2):
        e, rng, tr = Env(slip_prob=0.3), np.random.default_rng(1), []; e.reset(seed=7)
        for _ in range(40):
            o, r, *_ = e.step({n: int(rng.integers(5)) for n in e.possible_agents}); tr.append((o["agent_0"]["fov"].copy(), tuple(r.values()), P(e)))
        outs.append(tr)
    assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2] for a, b in zip(*outs))
def test_fast_obs_matches_starter():  # FastEnv flat obs == starter dict obs (flattened) along a random slip rollout
    f, s, rng = FastEnv(slip_prob=0.2), Env(slip_prob=0.2), np.random.default_rng(0)
    of, _ = f.reset(seed=5); os_, _ = s.reset(seed=5)
    for _ in range(60):
        flat = lambda o: np.concatenate([o["fov"].ravel(), o["delta"], o["time"], o["prev_action"]])
        assert all(np.array_equal(of[i], flat(os_[n])) for i, n in enumerate(s.possible_agents))
        a = rng.integers(0, 5, 5); of, *_ = f.step_arrays(a); os_, *_ = s.step({n: int(a[i]) for i, n in enumerate(s.possible_agents)})
        assert f.pos.tolist() == s.pos.tolist()

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns: fn(); print("ok", fn.__name__)
    print(f"all {len(fns)} tests passed")
