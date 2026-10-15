# Progress — Lunar Lander Reward Shaping Lab (Phase 1)

Project: train a baseline PPO agent on Gymnasium's `LunarLander-v3` with the
built-in reward, and build the evaluation tooling. Phase 2 will add custom
reward shaping on top of this lab.

## Step log

- [x] **Step 0 — Repo init.** `git init` in `d:\Projects\LunarLander`.
- [x] **Step 1 — Setup & sanity check.** Python 3.11.17 venv at `.venv`
  (via uv; only Python 3.14 existed on the machine, but SB3 2.9.0 supports
  3.10–3.13 and gymnasium's Box2D extra ships prebuilt wheels below 3.14).
  Installed: gymnasium[box2d]==1.4.0, stable-baselines3==2.9.0,
  torch==2.14.1+cpu (PyTorch CPU index), numpy 2.4.6, pandas 3.0.6,
  matplotlib 3.11.2, moviepy 2.2.1, imageio 2.38.1, imageio-ffmpeg 0.6.0,
  tensorboard 2.21.0, PyYAML 6.0.3, pytest 9.1.1.
  `scripts/random_agent.py` runs 5 random episodes, prints labelled
  first-frame observations and per-episode reward/length, and saves
  `videos/random_agent.mp4`. All imports verified; environment steps fine.
- [ ] **Step 2 — Timing check.** PPO 50k steps with 1 / 8 / 16 vec envs;
  measure steps/sec; report RL-Baselines3-Zoo hyperparameters for
  LunarLander-v3; estimate time for 1M steps. **STOP here for approval.**
- [ ] **Step 3 — Baseline training.** `scripts/train.py` + tuned config;
  3 seeds x 1,000,000 steps; resumable.
- [ ] **Step 4 — Evaluation tool.** `scripts/evaluate.py` (100 episodes on
  the original reward) + `scripts/aggregate.py`.
- [ ] **Step 5 — Video & plots.** `scripts/record_video.py`,
  `scripts/plot_curves.py`.
- [ ] **Step 6 — Tests & README.** pytest suite + README with real numbers.

## Key facts (verified against installed sources)

- Environment: `LunarLander-v3` (Gymnasium 1.4.0). Observation = 8-dim:
  [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact].
  Actions: 0 do nothing, 1 fire left engine, 2 fire main engine,
  3 fire right engine.
- Reward (from `gymnasium/envs/box2d/lunar_lander.py`): per-step shaping
  based on distance to pad, speed and tilt; +10 per leg in ground contact;
  -0.03 per frame a side engine fires; -0.30 per frame the main engine fires;
  termination step gives exactly **+100** when the lander comes to rest
  (landed) or **-100** when the hull touches the ground or the lander comes
  to rest off-pad (|x| >= 1.0). Episodes truncate at 1000 steps.
- Engine costs => fuel proxy: `0.3 * main_firings + 0.03 * side_firings`.
