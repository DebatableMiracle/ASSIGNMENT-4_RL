# AIL722 A4 – Problem 1
Files: mapf_env.py + visualize.py (starter, unchanged), fast_env.py (subclass: same dynamics, flat-array obs for speed), tests.py,
common.py, train_iql.py, train_ctde.py (QMIX), analyze.py (1.5), run_all.sh, report_1_1.md (draft of 1.1).
```
pip install torch numpy matplotlib pillow pettingzoo gymnasium
python tests.py                                    # 11 tests (incl. FastEnv obs == starter obs)
N_PAR=6 bash run_all.sh                            # all 12 runs on GPU, logs in logs/
python train_iql.py  --reward sparse|shaped --seed 0|1|2 --device cuda    # results/iql_<rw>_s<seed>/, checkpoints/iql_<rw>_s<seed>.pt
python train_ctde.py --reward sparse|shaped --seed 0|1|2 --device cuda    # results/qmix_<rw>_s<seed>/, checkpoints/qmix_...
python analyze.py --ckpt checkpoints/qmix_shaped_s0.pt --slip 0.1 --reward shaped   # add --slip 0.3 if it rarely fails
```
Each run: curve.json (every 50k steps, 50 eps, seeds 10000-10049), final.json (100 eps, seeds 10000-10099), train_log.jsonl (losses).
Training episodes use seeds from [1e6,1e9); final checkpoint evaluated (no model selection).
Hyper-params (both): Adam lr 5e-4, batch 128 joint transitions (640 agent samples), 1 update / 4 env steps, buffer 100k, learn start 5k,
hard target update / 1000 updates, eps 1->0.05 over 300k steps, Huber, grad-clip 10, gamma 0.99, Double-Q targets.
Net: Conv(4,32,3)-Conv(32,32,3) on the 5x5 FOV, concat [delta, t/T, prev-a, one-hot id] -> MLP 256-256 -> 5.
QMIX mixer: emb 32, hypernet hidden 64, abs() weights, input = global state (walls, positions, goals, finished, t) – training only.
Conventions: IQL reward = r_i + team_bonus (from infos); finished agents are masked from losses (QMIX: utility 0 in mixer).
# ASSIGNMENT-4_RL
