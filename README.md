# Lunar Lander Reward Shaping Lab

A reinforcement-learning portfolio project that trains a PPO agent on
Gymnasium's `LunarLander-v3` and analyses its behaviour. **Phase 1** sets up
the environment, trains a solid baseline with the built-in reward using
tuned hyperparameters from the RL Baselines3 Zoo, and builds evaluation,
plotting and video tooling on top of the *original, unmodified* environment
reward. **Phase 2** adds three custom reward-shaping variants — a per-frame
time penalty, a deliberately exploitable hover bonus, and a potential-based
"unfarmable" shaping — and compares all of them against the baseline on the
original reward. Full Phase 2 write-up:
[`reports/results.md`](reports/results.md).

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

## What is reward shaping? (for beginners)

The reward is the score the learning algorithm is trying to maximise.
**Reward shaping** means tweaking that score during *training* to steer the
agent toward behaviour you want — "add 1 point for every second you stay
upright" — while keeping *evaluation* on the real, unmodified score, so you
can measure whether the steering actually helped.

Shaping is famously double-edged:

- **Helpful shaping** nudges the agent past slow early exploration: a small
  per-frame cost for dawdling turned a dithering lander into a decisive
  one (see `time_penalty` below: the best mode in the project).
- **Reward hacking** is what happens when the shaping is easier to exploit
  than the task itself — the agent maximises *your score* without ever
  solving *your problem*. This lab reproduces it on purpose: the
  `hover_bonus` variant pays +1 per frame for being upright and slow
  *anywhere above the pad*, up to +1000 for hovering in place for a whole
  episode, versus +100 for actually landing. The trained agents do the
  economical thing: they hover until the clock runs out (85% of evaluation
  episodes time out, mean original-reward score −4.4). Goodhart's law —
  "when a measure becomes a target, it ceases to be a good measure" — in a
  100-line wrapper.

The class of shaping that theory says is safe is *potential-based*
(`gamma·Φ(s') − Φ(s)`), whose terms **telescope**: hovering in a good state
earns almost nothing, only *improving* pays, so it cannot be farmed. The
project's `potential_shaped` variant implements exactly that and stays
neutral on the true task. All four modes are trained with identical
hyperparameters and graded on the original reward.

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

### Phase 2: reward-shaped variants (per-mode configs under `configs/`)

```powershell
# Train a shaped campaign (3 seeds; per-mode configs pin the reward params)
python -u scripts/train.py --config configs/ppo_time_penalty.yaml --seed 0
python -u scripts/train.py --config configs/ppo_hover_bonus.yaml --seed 1
python -u scripts/train.py --config configs/ppo_potential_shaped.yaml --seed 2
#    --mode <mode> selects the same reward through the CLI (defaults from
#    src/reward_wrappers.py); run dirs become results/runs/<mode>/seed_{S}/

# Evaluate a mode (100 deterministic episodes per seed, original reward)
python scripts/evaluate.py --mode time_penalty --episodes 100

# Hover-farming diagnostics (timeouts, hover-state %, hover run lengths)
python scripts/inspect_hover.py --episodes 20 `
  --model results/runs/hover_bonus/seed_0/models/ppo_lander_seed0.zip

# Cross-mode comparison: table, paired bootstrap CIs, plots
python scripts/aggregate.py
python scripts/compare.py

# Videos for a mode + hover-vs-potential side-by-side clip
python scripts/record_video.py --mode hover_bonus --seeds 0 1 2
python scripts/side_by_side.py --seed 0
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

**Phase 2 — reward shaping comparison** (identical recipe, original reward,
300 paired episodes per mode; from `results/evaluations/comparison.md` and
`paired_differences.md`, 95% bootstrap CIs):

| mode | mean across seeds | landed | timeout | mean len | fuel | paired diff vs baseline |
|---|---|---|---|---|---|---|
| baseline | +254.6 ± 3.5 | 99% | 1% | 366 | 53.7 | — |
| time_penalty | **+262.4 ± 0.9** | 100% | 0% | 310 | 46.7 | **+7.8 [+6.1, +9.6]** |
| hover_bonus | **−4.4 ± 51.9** | 15% | **85%** | 985 | 179.3 | **−259.0 [−267.2, −250.7]** |
| potential_shaped | +256.3 ± 16.4 | 97% | 3% | 332 | 44.9 | +1.8 [−2.1, +5.3] (noise) |

The deliberately exploitable hover bonus **reward-hacked**: agents park
upright and slow above the pad and stay there — 95–96% of frames in the
hover state, hover *runs* of 684–799 consecutive frames — until the
1000-step truncation (85% of episodes; 99% on seed 1), burning 3.3× the
baseline's fuel. The time penalty produced the *best and most decisive*
landings (100% landed, zero timeouts, lowest fuel). The potential-based
form, which telescopes and cannot be farmed, stayed neutral. Videos:
`videos/<mode>_seed{0,1,2}.mp4` and
`videos/side_by_side_hover_bonus_vs_potential_shaped_seed0.mp4`; full
write-up with failure notes and reproduction commands in
[`reports/results.md`](reports/results.md). Caveat: 3 seeds per mode — the
paired differences are solid *for these 300 shared initial states* and thin
as claims about the recipes in general.

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
configs/     # YAML configs: ppo_baseline.yaml + per-mode ppo_<mode>.yaml
src/         # reward_wrappers.py (time_penalty / hover_bonus / potential)
scripts/     # entry points: random_agent, timing_check, train, evaluate,
             # inspect_hover, aggregate, compare, plot_curves,
             # record_video, side_by_side
tests/       # pytest suite (51 tests: wrappers, helpers, episode/JSON
             # round trips, reward-mode resolution)
docs/        # resume_bullets.md (job-search material)
reports/     # results.md (Phase 2 write-up)
results/
  runs/seed_*/          # baseline: models/, eval/, best/, monitor/, summary.json
  runs/<mode>/seed_*/   # shaped campaigns (same layout per seed)
  evaluations/          # per-seed episode JSONs, summary, comparison, plots
  figures/              # learning_curves.png
  timing/               # throughput benchmark CSVs
  tb/                   # TensorBoard event files
videos/     # recorded episode clips (random agent + trained agents)
```
