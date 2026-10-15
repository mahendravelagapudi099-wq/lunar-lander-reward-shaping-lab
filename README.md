# Lunar Lander Reward Shaping Lab

A reinforcement-learning portfolio project that trains a PPO agent on
Gymnasium's `LunarLander-v3` and analyses its behaviour. **Phase 1** (this
codebase) sets up the environment, trains a solid baseline with the built-in
reward using tuned hyperparameters from the RL Baselines3 Zoo, and builds
evaluation, plotting and video tooling on top of the *original, unmodified*
environment reward. Phase 2 will add custom reward shaping variants and
compare them against this baseline.

## The environment

LunarLander-v3 is a 2D Box2D rocket-landing problem. The lander starts
airborne and must touch down on the landing pad at coordinates (0, 0).

**Observation (8 values):**

| # | Value | Meaning |
|---|-------|---------|
| 0 | x position | horizontal position, pad is at 0 (normalised units) |
| 1 | y position | height above the pad |
| 2 | x velocity | horizontal velocity |
| 3 | y velocity | vertical velocity |
| 4 | angle | lander tilt in radians (0 = upright) |
| 5 | angular velocity | rotation rate |
| 6 | left leg contact | 1 if the left leg touches the ground, else 0 |
| 7 | right leg contact | 1 if the right leg touches the ground, else 0 |

**Actions (discrete, 4):**

| Action | Effect |
|--------|--------|
| 0 | do nothing |
| 1 | fire the left orientation engine |
| 2 | fire the main (bottom) engine |
| 3 | fire the right orientation engine |

**Reward:** per-step shaping rewards getting closer to the pad, moving slowly
and staying level; +10 for each leg touching ground; -0.03 per frame a side
engine fires and -0.3 per frame the main engine fires (fuel cost). A safe
landing ends the episode with **+100**; a crash (hull contact) or coming to
rest off the pad ends it with **-100**. Episodes truncate after 1000 steps.
The task is considered **solved at 200+ points**.

## Installation (Windows, CPU only)

1. Install [Python 3.11](https://www.python.org/downloads/) (3.10–3.13 also
   work; Python 3.14 is not supported by stable-baselines3 2.9.0).
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the CPU-only PyTorch wheel (avoids the ~2.5 GB CUDA wheel):

   ```powershell
   pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

4. Install the remaining dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

All commands run from the repository root with the virtual environment
activated (`.venv\Scripts\activate`).

```powershell
# 1. Sanity check: 5 random episodes, prints observations, saves
#    videos/random_agent.mp4
python scripts/random_agent.py

# 2. Throughput benchmark (1, 8 and 16 vec envs, 50k steps each)
python scripts/timing_check.py

# 3. Train the baseline: one process per seed, run the three in parallel
python -u scripts/train.py --seed 0
python -u scripts/train.py --seed 1
python -u scripts/train.py --seed 2
#    ...or a single seed to completion; --resume continues from the latest
#    checkpoint if a run is interrupted:
python -u scripts/train.py --seed 0 --resume

# 4. Evaluate: 100 deterministic episodes per seed on the original reward
#    (shared episode-seed sequence -> comparable across models)
python scripts/evaluate.py --seeds 0 1 2 --episodes 100

# 5. Aggregate the evaluation JSONs into results/evaluations/summary.{json,md}
python scripts/aggregate.py

# 6. Learning curves -> results/figures/learning_curves.png
python scripts/plot_curves.py

# 7. Landing videos -> videos/ppo_lander_seed{0,1,2}.mp4
python scripts/record_video.py --seeds 0 1 2

# 8. Test suite
python -m pytest tests/ -q
```

Everything is config-driven from `configs/ppo_baseline.yaml` (RL Baselines3
Zoo recipe for LunarLander-v3: `n_envs: 16, n_steps: 1024, batch_size: 64,
gae_lambda: 0.98, gamma: 0.999, n_epochs: 4, ent_coef: 0.01`).

## Results

**Training** — 3 seeds × 1M steps (runs overshoot to 1,015,808 steps because
PPO finishes the current `n_envs × n_steps` rollout), `DummyVecEnv` × 16,
single-threaded torch, on an i5-1235U laptop (12 logical cores, 15.7 GB RAM):

| seed | wall time | throughput | checkpoints |
|------|-----------|------------|-------------|
| 0    | 23.1 min  | 733 steps/s | every 50k → 1M |
| 1    | 23.8 min  | 712 steps/s | every 50k → 1M |
| 2    | 24.4 min  | 693 steps/s | every 50k → 1M |

The three seeds trained as parallel background processes, so the whole
baseline took ~25 minutes of wall time. Deterministic evaluation during
training (10 episodes every 50k steps) reached +256.1 / +272.6 / +255.3 at
1M steps — the final model is also the best model for every seed.

**Evaluation** — 100 deterministic episodes per seed on the *unmodified*
built-in reward (300 episodes total), from
`results/evaluations/summary.md`:

| seed | mean reward | SEM | min | max | landed | crash | timeout | fuel |
|------|------------|-----|-----|-----|--------|-------|---------|------|
| 0    | +250.5     | 2.7 | +124.1 | +296.1 | 98%  | 0% | 2% | 57.0 |
| 1    | +257.1     | 2.1 | +209.3 | +300.4 | 100% | 0% | 0% | 49.9 |
| 2    | +256.0     | 2.0 | +209.5 | +293.1 | 100% | 0% | 0% | 54.2 |
| pooled | +254.6   | —   | +124.1 | +300.4 | 99%  | 0% | 1% | 53.7 |

**Baseline: +254.6 ± 3.5 points across seeds** — comfortably above the
200-point "solved" bar, with a **99% landing rate and zero crashes** over 300
episodes. `fuel cost` is the engine-firing proxy `0.3·main + 0.03·side`
frames. Learning curves (all three seeds crossing 200 well before 1M steps)
are in `results/figures/learning_curves.png`; example landings are in
`videos/ppo_lander_seed{0,1,2}.mp4`.

Every number above is read from JSON/NPZ artifacts under `results/`
produced by real runs (`results/runs/seed_*/summary.json`,
`results/evaluations/*.json`) — nothing is hand-written or estimated.

## Notes for this machine (CPU-only laptops)

- `torch.set_num_threads(1)` is essential: the default 12-thread intra-op
  parallelism is 5–10× *slower* for the tiny (64, 64) policy — thread-sync
  overhead dominates. See `results/timing/` and `progress.md`.
- `SubprocVecEnv` is ~2.4× faster than `DummyVecEnv` per single run, but
  stacking it (16 workers × 3 parallel seeds) spawns ~50 processes that each
  re-import torch, which exceeded the machine's 15.7 GB RAM (spawn-wave OOM,
  `OMP: Error #111`). The baseline therefore uses `DummyVecEnv`, trading
  throughput for a stable ~240 MB per seed.

## Project layout

```
configs/     # YAML configs (ppo_baseline.yaml: Zoo-tuned PPO recipe)
scripts/     # entry points: random_agent, timing_check, train, evaluate,
             # aggregate, plot_curves, record_video
tests/       # pytest suite (helpers + episode/episode-JSON round trip)
results/
  runs/seed_*/        # models/, eval/, best/, monitor/, summary.json
  evaluations/        # per-seed episode JSONs + summary.{json,md}
  figures/            # learning_curves.png
  timing/             # throughput benchmark CSVs
  tb/                 # TensorBoard event files
videos/     # recorded episode clips (random agent + trained agents)
```
