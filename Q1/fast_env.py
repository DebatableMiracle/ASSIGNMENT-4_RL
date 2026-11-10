"""Speed-up wrapper around the starter env. Dynamics are INHERITED UNCHANGED; this only adds a flat-array observation path
(obs layout per agent, 108 = fov 4x5x5 | delta 2 | t/T 1 | prev_action 5, i.e. exactly the dict obs flattened)."""
import numpy as np
from mapf_env import CustomMAPFParallelEnv

OBS_DIM, N_ACT = 108, 5

class FastEnv(CustomMAPFParallelEnv):
    def __init__(self, *a, fast=True, **kw):
        super().__init__(*a, **kw)
        self.fast, self.collisions = fast, 0
        self._gp = np.pad(self.grid.astype(np.float32), 2, constant_values=1)  # outside grid = wall

    def reset(self, seed=None, options=None):
        out = super().reset(seed, options); self.collisions = 0
        return out

    def _all_obs(self):
        return self.obs_array() if self.fast else super()._all_obs()

    def obs_array(self):
        N, M = self.N, max(self.H, self.W)
        P, G = self.pos.tolist(), self.goals.tolist()
        out = np.zeros((N, OBS_DIM), np.float32)
        for i in range(N):
            r, c = P[i]
            out[i, :25] = self._gp[r:r + 5, c:c + 5].ravel()
            f = out[i, :100].reshape(4, 5, 5)
            for j in range(N):
                if j != i:
                    a, b = P[j][0] - r + 2, P[j][1] - c + 2
                    if 0 <= a < 5 and 0 <= b < 5: f[1, a, b] = 1
                a, b = G[j][0] - r + 2, G[j][1] - c + 2
                if 0 <= a < 5 and 0 <= b < 5: f[2 if j != i else 3, a, b] = 1
            out[i, 100], out[i, 101] = (G[i][0] - r) / M, (G[i][1] - c) / M
            out[i, 102] = self.t / self.T
            out[i, 103:] = self.prev_action[i]
        return out

    def step_arrays(self, a):
        """a: (N,) ints. Returns obs (N,108), r_i (N,) [no team bonus], bonus, team_reward, done, truncated, info."""
        active = ~self.finished
        obs, rew, term, trunc, info = self.step({n: int(a[i]) for i, n in enumerate(self.possible_agents)})
        n0 = self.possible_agents[0]
        blocked = np.array([info[n]["blocked"] for n in self.possible_agents])
        self.collisions += int(blocked.sum())
        r = np.array([rew[n] for n in self.possible_agents], np.float32)
        return obs, r, info[n0]["team_bonus"], info[n0]["team_reward"], info[n0]["success"], trunc[n0], {"active": active, "blocked": blocked}

    def state_vec(self):
        g = self.get_global_state()  # centralised training only
        return np.concatenate([g["walls"].ravel(), g["positions"].ravel() / 10, g["goals"].ravel() / 10,
                               g["finished"], [g["t"] / self.T]]).astype(np.float32)

    @property
    def state_dim(self): return self.H * self.W + 5 * self.N + 1
