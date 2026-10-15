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

> The exact commands for every script, and the baseline results, are added in
> Step 6 of the build, once the training runs are complete. Commands below are
> provisional until then.

```powershell
# Sanity check: 5 random episodes, saves videos/random_agent.mp4
python scripts/random_agent.py
```

## Results

Filled in Step 6 — every number below comes from JSON/CSV files under
`results/` produced by real runs (see `scripts/evaluate.py` and
`scripts/aggregate.py`). No value is hand-written or estimated.

## Project layout

```
configs/    # YAML hyperparameter files (ppo_baseline.yaml)
scripts/    # runnable entry points (random_agent, train, evaluate, ...)
src/        # shared helper modules
tests/      # pytest suite
results/    # per-run outputs: models, monitor CSVs, tensorboard logs, evaluation JSONs
videos/     # recorded episode clips (mp4/gif)
models/     # saved SB3 checkpoints
```
