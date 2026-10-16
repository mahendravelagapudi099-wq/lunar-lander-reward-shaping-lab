# Empirical Evaluation of Reward Shaping in PPO: Benefits, Policy Invariance, and Deliberate Reward Hacking

**Environment:** Gymnasium `LunarLander-v3` (Box2D)  
**Algorithm:** Proximal Policy Optimization (PPO, Stable-Baselines3 `MlpPolicy`)  
**Evaluation:** 300 deterministic paired episodes per mode on the unmodified environment  
**Author:** Mahendra Velagapudi  

---

## Executive Summary

Reward shaping is frequently used in reinforcement learning to alleviate sparse feedback and accelerate policy exploration. However, poorly specified shaping terms risk Goodhart's law—creating perverse incentives that policies exploit rather than solving the target objective.

This laboratory presents a controlled empirical study comparing four PPO configurations trained under identical hyperparameters:
1. **`baseline` (Reference):** Built-in Gymnasium reward. Reaches **+254.6 ± 3.5** points (99% landings, 1% timeouts).
2. **`time_penalty` (−0.05/frame):** Achieves statistically significant performance improvements: **+262.4 ± 0.9** points (100% landings, 0% timeouts, paired gain of **+7.8**, 95% bootstrap CI `[+6.1, +9.6]`). The penalty reduces flight duration by 15.3% and fuel expenditure by 13.0%.
3. **`hover_bonus` (+1.0/frame):** Deliberately exploitable bonus triggering catastrophic reward hacking. Training scores climb continuously, yet true evaluation drops to **−4.4 ± 51.9** points (85% timeout rate, hovering in place for 700+ consecutive frames).
4. **`potential_shaped` ($\gamma \Phi(s') - \Phi(s)$):** Theory-grounded potential shaping (Ng et al., 1999). Yields **+256.3 ± 16.4** points (paired difference of **+1.8**, 95% bootstrap CI `[−2.1, +5.3]`, straddling zero), confirming mathematical invariance on the optimal policy.

---

## Problem Formulation & Reward Mechanics

All experiments run in Gymnasium's continuous-state, discrete-action `LunarLander-v3`. The observation space $s \in \mathbb{R}^8$ captures position $(x, y)$, linear velocity $(v_x, v_y)$, orientation angle $\theta$, angular velocity $\dot{\theta}$, and two ground-contact booleans. The action space $a \in \{0, 1, 2, 3\}$ controls the main and lateral thrusters.

### Reward Formulations

| Mode | Mathematical Formulation | Design Intent & Mechanism |
| :--- | :--- | :--- |
| `baseline` | $R(s, a)$ | Standard environment reward (distance reduction, touchdown points, leg contact bonuses, fuel costs). |
| `time_penalty` | $R'(s, a) = R(s, a) - 0.05$ | Constant deadline penalty (−50 maximum per 1000-step episode). Penalizes idling; favors decisive trajectories. |
| `hover_bonus` | $R'(s, a) = R(s, a) + 1.0 \cdot \mathbb{I}(\text{hover})$ | **Deliberately exploitable:** Pays +1.0 whenever $|\theta| < 0.2\text{ rad}$ and $\|(v_x, v_y)\| < 0.3$. Ignores altitude $y$. Max payout is +1000 (10× the +100 touchdown reward). |
| `potential_shaped` | $R'(s, a, s') = R(s, a) + \gamma \Phi(s') - \Phi(s) - 0.01$ | **Telescoping potential:** $\Phi(s) = -(1.0 \|(x,y)\| + 2.0 \|(v_x, v_y)\| + 1.0 |\theta|)$. Rewards trajectory improvement; holding state yields near-zero return. |

### Mathematical Invariance of Potential Shaping

For a discount factor $\gamma \in (0, 1)$ and state potential function $\Phi: \mathcal{S} \to \mathbb{R}$, the cumulative shaping return along an episode of length $T$ telescopes:

$$\sum_{t=0}^{T-1} \gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^{T-1} \gamma^t (\gamma \Phi(s_{t+1}) - \Phi(s_t)) = \gamma^T \Phi(s_T) - \Phi(s_0)$$

Because the total shaping bonus depends strictly on boundary conditions rather than the intermediate path length, hovering indefinitely yields at most $(1 - \gamma) T |\Phi| \approx 2$ points over 1,000 steps at $\gamma = 0.999$, making loitering mathematically unprofitable.

---

## Experimental Setup & Benchmark Protocol

### Hyperparameter Controls

To isolate the causal effect of reward shaping, all training hyperparameters are strictly pinned using the tuned RL Baselines3 Zoo configuration for `LunarLander-v3`. An automated regression test (`test_all_reward_configs_share_identical_hyperparameters`) asserts that non-reward configuration values match across modes.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Algorithm** | PPO (`MlpPolicy`) | Two hidden layers of 64 units with tanh activations |
| **Learning Rate / Clip** | $3 \times 10^{-4}$ / 0.2 | Standard Adam optimizer schedule |
| **Rollout Buffer** | $1024 \text{ steps} \times 16 \text{ envs}$ | $16,384$ transitions per PPO update iteration |
| **Optimization** | Batch size 64, 4 epochs | Minibatch sampling with shuffle per update |
| **GAE ($\lambda$) / Discount ($\gamma$)** | 0.98 / 0.999 | Long-horizon credit assignment |
| **Entropy Coefficient** | 0.01 | Exploration bonus |
| **Training Budget** | 1,015,808 steps | 3 seeds per mode ($3 \times 4 = 12$ campaigns, ~25 min/seed on CPU) |

### Evaluation Methodology

1. **Unwrapped Original Reward:** Evaluation strictly bypasses all reward wrappers, testing agents against the ground-truth environment reward.
2. **Paired Initial Conditions:** Each model is evaluated across 100 deterministic episodes. All models evaluate against the exact same initial pseudo-random seeds ($[0, 99]$), yielding 300 paired comparisons per mode that isolate policy variance from environmental spawn luck.
3. **Statistical Inference:** Paired differences are analyzed using 10,000-resample non-parametric bootstrap confidence intervals (95% CI).

---

## Headline Results & Comparative Analysis

### Primary Benchmark Results (300 Paired Episodes per Mode)

| Mode | Mean Reward (Seed Spread) | Landed % | Timeout % | Crash % | Mean Length | Fuel Used | Solved ($\ge 200$) % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | $+254.6 \pm 3.5$ | 99.0% | 1.0% | 0.0% | 366.3 | 53.7 | 99.0% |
| **`time_penalty`** | **+262.4 ± 0.9** | **100.0%** | **0.0%** | 0.0% | **310.2** | **46.7** | **100.0%** |
| **`hover_bonus`** | **−4.4 ± 51.9** | **15.0%** | **85.0%** | 0.0% | **984.7** | **179.3** | **1.0%** |
| **`potential_shaped`** | $+256.3 \pm 16.4$ | 97.0% | 3.0% | 0.0% | 332.1 | 44.9 | 96.0% |

*Note: Seed spread represents the range across the 3 seeds (the replication unit). Fuel consumption measures active engine firing frames ($0.3 \cdot \text{main} + 0.03 \cdot \text{side}$).*

### Paired Differences vs. Baseline (300 Paired Episodes)

| Mode | Paired Mean Difference | Bootstrap 95% CI | Statistical Verdict |
| :--- | :---: | :---: | :--- |
| **`time_penalty`** | **+7.8** | **[+6.1, +9.6]** | **Statistically Significant Gain** |
| **`hover_bonus`** | **−259.0** | **[−267.2, −250.7]** | **Statistically Significant Collapse** |
| **`potential_shaped`** | **+1.8** | **[−2.1, +5.3]** | **Within Noise (Invariant)** |

### Hover Diagnostics (20 Raw-Environment Telemetry Runs)

Using `scripts/inspect_hover.py`, agent telemetry was analyzed to measure loitering behavior:

| Mode | Hover State Frame % | Mean Consecutive Hover Run | Longest Observed Hover Run |
| :--- | :---: | :---: | :---: |
| **`baseline`** | 88.0% | 239.1 frames | 358 frames |
| **`time_penalty`** | 76.5% | 42.8 frames | 84 frames |
| **`hover_bonus`** | **95.6%** | **758.0 frames** | **1,000 frames (Full Episode)** |
| **`potential_shaped`** | 68.2% | 51.4 frames | 98 frames |

---

## Detailed Empirical Breakdown

### 1. Baseline Policy (Control Arm)
The baseline PPO agent establishes a reliable benchmark (+254.6 points, 99% landings). Telemetry analysis reveals that nominal successful descents naturally satisfy the hover criteria ($|\theta| < 0.2$, speed $< 0.3$) for ~88% of their trajectory. This demonstrates that the hover condition is an intrinsic feature of successful flight, explaining why agents discover it immediately.

### 2. Time Penalty (−0.05/frame): Decisive Optimization
The time penalty variant delivered the best overall score (+262.4) with minimal variance between seeds (spread of ±0.9). 
- **Mechanism:** The per-frame cost discourages prolonged stabilization and hunting near the pad. 
- **Efficiency:** The mean episode length dropped from 366 to 310 steps, driving a 13.0% reduction in fuel consumption (46.7 vs. 53.7).
- **Log Validation:** Training Monitor CSVs show an average shaped penalty delta of −16.34 per episode, aligning with $0.05 \times 306 \text{ frames} = 15.3$ expected penalty.

### 3. Hover Bonus (+1.0/frame): Anatomy of Reward Hacking
The hover bonus presents a textbook manifestation of Goodhart's law:
- **Divergent Signals:** Training curves show massive apparent success (training reward steadily exceeding +280), but true evaluation scores collapse to −4.4.
- **Behavioral Shift:** Rather than landing to claim the one-time +100 touchdown reward, agents loiter above the pad, timing out at the 1,000-step ceiling in 85% of episodes while banking up to +1,000 in shaping rewards.
- **Bifurcated Strategies:** Seed 0 discovered a mixed policy (39% landings, +50.2 mean score) hovering temporarily before committing to a late landing. Seeds 1 and 2 converged entirely to pure loitering (99% timeouts, −53.1 and −10.4 scores).

### 4. Potential-Based Shaping: Verified Policy Invariance
Potential shaping preserves the asymptotic policy characteristics of the baseline:
- **Empirical Invariance:** The paired score difference (+1.8, 95% CI `[−2.1, +5.3]`) straddles zero, verifying that potential shaping does not alter the optimal trajectory.
- **Anti-Farming Guarantee:** Training Monitor logs report an average per-episode shaping delta of only −1.94 (due to the minor 0.01 frame penalty), preventing the loitering collapse seen in the naive bonus.

---

## Engineering Postmortem: The Bitwise-Identical Weights Incident

During Phase 2 implementation, a critical silent configuration bug was diagnosed and corrected:

### Incident Description
The initial training run for `hover_bonus` completed without error, reporting healthy convergence. However, evaluation outputs appeared suspiciously close to the baseline.

### Detection & Root Cause Analysis
1. **Weight Hashing:** Inspection revealed that the trained model weights for `hover_bonus` were **bitwise-identical** to the baseline model trained with the same seed.
2. **Monitor Telemetry:** The per-episode shaping column (`episode_shaping`) in the Monitor logs reported `0.0` across all 2,406 training episodes, whereas a functioning wrapper generates distinct non-zero deltas.
3. **Root Cause:** In `scripts/train.py`, the CLI flag `--mode hover_bonus` dynamically updated the output directory and logging banner, but the reward wrapper initialization resolved mode configuration strictly through `config.get("reward")`. Because `ppo_baseline.yaml` lacked a reward block, the code silently defaulted to the unwrapped environment.

### Remediation & Permanent Controls
- **Refactoring:** Consolidated configuration resolution into a centralized, deterministic function: `resolve_reward_cfg(config, cli_mode)`.
- **Regression Tests:** Added `tests/test_train_helpers.py` asserting that CLI `--mode` arguments override and populate wrapper configs.
- **Standing Bring-Up Check:** Added automated validation of `episode_shaping` deltas in the first monitor batch before completing training runs.
- **Data Hygiene:** All corrupted model checkpoints and evaluation logs were purged, and campaigns were retrained from scratch.

---

## Technical Discussion & Practical Takeaways

### 1. Payoff Asymmetry Dictates Exploitability
Reward hacking is governed by the relative payoff ratio between the primary goal and the shaping heuristic. In this environment, landing pays a fixed +100 bonus, while loitering pays +1.0 per step up to a 1,000-step horizon (+1000 total). Because the shaping bonus outweighs the true objective by 10:1, the discount horizon ($\gamma = 0.999$) naturally prioritizes sustained loitering over terminal landing.

### 2. Time Penalties as Implicit Regularizers
A small negative step penalty acts as a simple yet effective regularizer in goal-directed RL. By imposing an opportunity cost on time, it penalizes low-value dithering and incentivizes minimal-length trajectories, yielding decisive controllers that conserve fuel.

### 3. Theory vs. Implementation Overhead
While potential-based reward shaping theoretically guarantees policy invariance, constructing effective potential functions $\Phi(s)$ requires domain expertise and careful state-space normalization. In benchmark environments with dense base feedback, the empirical benefits of potential shaping may not justify the engineering overhead compared to simple deadline penalties.

### 4. RL Pipeline Observability
Standard convergence metrics (episode length, loss values, training reward) are insufficient to verify agent alignment. Robust RL development demands:
- Continuous tracking of both **shaped training reward** and **ground-truth task reward**.
- Routine verification of weight hashes across different experiment arms.
- Telemetry inspections of intermediate state distributions.

---

## Reproduction Guide

All results, tables, and figures can be reproduced directly on a standard multi-core CPU machine:

```powershell
# 1. Train models (repeat for modes: baseline, time_penalty, hover_bonus, potential_shaped)
python scripts/train.py --config configs/ppo_time_penalty.yaml --seed 0
python scripts/train.py --config configs/ppo_time_penalty.yaml --seed 1
python scripts/train.py --config configs/ppo_time_penalty.yaml --seed 2

# 2. Evaluate models on the unmodified reward (100 deterministic episodes per seed)
python scripts/evaluate.py --mode time_penalty --episodes 100

# 3. Aggregate statistics and run paired bootstrap comparisons
python scripts/aggregate.py
python scripts/compare.py

# 4. Generate hover-farming telemetry diagnostics
python scripts/inspect_hover.py --episodes 20 `
  --model results/runs/hover_bonus/seed_0/models/ppo_lander_seed0.zip

# 5. Run test suite (51 passing unit and regression tests)
pytest tests/ -q
```
