# Problem 2 Report: Offline Reinforcement Learning

## Subtask 2.1: Data Scaling with Decoupled Value and Policy Learning

### 1. Implementation Details (`train_decoupled.py`)
- **Decoupled Workflow**:
  - The total budget is $10^6$ gradient steps, split into **$200{,}000$ steps for critic training** and **$800{,}000$ steps for policy extraction**. This split reflects the finding of Park et al. (2024) that value learning converges relatively early, whereas policy extraction with conservative/advantage constraints requires substantial step iterations to distill a robust policy.
  - Critic parameters are frozen ($\nabla_\theta Q = 0$) during policy extraction.
  - **Critic Reuse**: Because the critic objective depends only on the dataset and the critic data fraction, a trained critic checkpoint (`critic_{algo}_{dataset}_c{frac}.pt`) is reused across the 3 policy fractions $\{10\%, 50\%, 100\%\}$. This avoids redundant training, reducing the required critic runs from $54$ to $18$.
- **Data Splitting & Nested Subsets**:
  - Subsets are drawn transition-wise with a deterministic pseudo-random permutation.
  - **Nested property**: Under the same random seed, index slicing ensures that $D_{10\%} \subset D_{50\%} \subset D_{100\%}$, isolating data volume effects from sample variance.
- **Algorithms**:
  1. **TD3+BC**: Critic trained with clipped double Q-learning and target policy smoothing. Extraction maximizes $\lambda Q(s, \pi(s)) - (\pi(s) - a)^2$ where $\lambda = \frac{\alpha}{\frac{1}{N}\sum |Q(s_i, a_i)|}$.
  2. **IQL+AWR**: Critic trained with expectile regression ($\tau = 0.7$). Policy extracted via advantage-weighted regression $\exp(\frac{Q(s,a) - V(s)}{\beta}) \nabla_\theta \log \pi(a|s)$ with temperature $\beta = 3.0$.
  3. **CQL+AWR**: Conservative Q-learning critic with `soft_q_backup=False` (resembling DDPG with deterministic policy targets). Policy extracted via advantage weighting against mean batch value.

### 2. Analysis & Park et al. Comparison
- **Critic vs. Policy Data Sensitivity**:
  - In our evaluations, varying the **critic data fraction** along columns shows that having even $10\%$ of critic data is often sufficient to learn a value function that can rank in-distribution actions accurately.
  - In contrast, varying the **policy data fraction** along rows produces a more pronounced impact on final return: with only $10\%$ policy data, behavioral coverage is limited and advantage-weighted extraction frequently collapses or clones suboptimal trajectories.
- **Agreement with Park et al.**:
  - These results align directly with the central claim of Park et al. (2024) (*"Is Value Learning Really the Main Bottleneck in Offline RL?"*): **policy extraction is often the true bottleneck**, not value learning. Once the critic captures general value trends, performance scales more strongly with policy data coverage.
- **Potential Confounds**:
  1. *Single Evaluation Seed*: Evaluation return in MuJoCo can exhibit high episode-to-episode variance depending on initial state distribution.
  2. *Step Budget Allocation*: Allocating fixed $200\text{k} / 800\text{k}$ gradient steps may favor policy extraction over critic training if certain algorithms require longer critic burn-in.

---

## Subtask 2.2: Implicit Q-Learning (IQL) from Scratch

### 1. Implementation Architecture (`train_iql_offline.py`)
- **Twin Q-Networks**: $Q_1(s, a), Q_2(s, a)$ parameterized as 2-layer MLPs (hidden dimension 256, ReLU activations).
- **Target Networks**: Soft Polyak target update with $\tau_{\text{soft}} = 0.005$.
- **Expectile Value Network**: $V(s)$ updated using asymmetric expectile squared loss:
  $$L_V(\psi) = \mathbb{E}_{(s, a) \sim D} \left[ |\tau - \mathbf{1}(Q_{\text{target}}(s, a) - V(s) < 0)| \cdot (Q_{\text{target}}(s, a) - V(s))^2 \right]$$
  with expectile parameter $\tau = 0.7$.
- **Policy Extraction (AWR)**: Diagonal Gaussian actor $\pi_\theta(a|s)$ trained with exponentiated advantages:
  $$L_\pi(\theta) = -\mathbb{E}_{(s, a) \sim D} \left[ \exp\left(\frac{Q_{\text{target}}(s, a) - V(s)}{\beta}\right) \log \pi_\theta(a|s) \right]$$
  with temperature $\beta = 3.0$ and advantage weights clipped to $100.0$ to prevent numerical instability.
- **Joint Training**: All networks trained jointly for $10^6$ gradient steps across 5 seeds ($0, 1, 2, 3, 4$).

### 2. Hyperparameter Summary
| Hyperparameter | Value |
|---|---|
| Expectile ($\tau$) | $0.7$ |
| Temperature ($\beta$) | $3.0$ |
| Batch Size | $256$ |
| Learning Rate (Actor, Critic, Value) | $3 \times 10^{-4}$ (Adam) |
| Actor LR Schedule | Cosine Annealing to 0 |
| Discount ($\gamma$) | $0.99$ |
| Soft Target Update Rate ($\tau_{\text{soft}}$) | $0.005$ |
| Hidden Dimensions | $256 \times 256$ (ReLU) |

### 3. Challenges & Resolutions
- **Advantage Explosion**: Unconstrained exponential weights $\exp((Q - V)/\beta)$ can cause extreme gradients. Resolved by clipping advantage weights to $\le 100.0$.
- **Headless MuJoCo Rendering**: Rendering rollout GIFs in headless server environments failed initially with GLFW/OpenGL errors. Resolved by setting `os.environ["MUJOCO_GL"] = "egl"`.
- **Termination vs. Truncation**: In offline datasets, bootstrapping on timeout truncations damages value estimation. Resolved by masking out terminals strictly using `ep.terminations` and bootstrapping through `ep.truncations`.

