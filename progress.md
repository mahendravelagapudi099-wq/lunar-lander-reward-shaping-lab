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
- [x] **Step 2 — Timing check.** `scripts/timing_check.py`, 50,000 PPO steps
  per configuration with the RL-Baselines3-Zoo hyperparameters for
  LunarLander-v3, single-threaded torch, on this machine
  (i5-1235U, 10 cores/12 threads, big.LITTLE):

  | n_envs | wall (50k) | steps/sec | est. 1M steps |
  |--------|-----------|-----------|----------------|
  | 1      | 2m 50s     | 294.8     | ~56 min        |
  | 8      | 1m 44s     | 482.1     | ~35 min        |
  | 16     | 2m 13s     | 376.6     | ~44 min        |

  Notes: the default 12-thread torch was ~5-10x slower for the tiny (64, 64)
  policy — thread-sync overhead dominates — so both the benchmark and
  training pin `torch.set_num_threads(1)`. `DummyVecEnv` is single-process,
  so more envs do not give real parallelism here; n_envs=8 was fastest but
  the baseline keeps the Zoo's n_envs=16 for fidelity. Plan: 3 seeds x 1M
  steps at n_envs=16 = ~2h15m total. Results also in
  `results/timing/timing_results.csv`. **STOPPING here for approval.**
- [x] **Step 3 — Baseline training.** `scripts/train.py` (config-driven,
  3 seeds × 1,000,000 steps, resumable via `--resume`):
  - `configs/ppo_baseline.yaml`: RL-Baselines3-Zoo hyperparameters for
    LunarLander-v3 (`n_envs: 16, n_steps: 1024, batch_size: 64,
    gae_lambda: 0.98, gamma: 0.999, n_epochs: 4, ent_coef: 0.01`),
    `vec_env: dummy`, `torch_threads: 1`, eval every 50k steps /
    10 episodes (no render), checkpoint every 50k, Monitor CSVs per env.
  - Per-seed layout under `results/runs/seed_{S}/`: `models/`
    (`ppo_lander_seed{S}.zip` + `ppo_lander_seed{S}_{steps}_steps.zip`),
    `eval/evaluations.npz`, `best/best_model.zip` (EvalCallback on
    deterministic eval reward), `monitor/*.monitor.csv`, plus a summary
    JSON with wall time and steps/s. TensorBoard events under
    `results/tb/ppo_seed{S}_1/`.
  - Env choice: a single-run 50k-step comparison favoured `SubprocVecEnv`
    (885.7 vs 376.6 steps/s), but stacking subproc ×16 on top of 3 parallel
    seeds needs ~50 processes each importing torch on a 15.7 GB machine —
    the spawn wave hit `OMP: Error #111` and workers died importing Box2D.
    Reverted to the single-process `DummyVecEnv` (the pre-approved setup):
    3 parallel processes, ~1–2 GB RAM total.
  - Reality check: the live DummyVecEnv runs sustained ~1,600–1,800 steps/s
    per seed (far above the 376.6 steps/s benchmark figure, which was
    measured during a busy period and included start-up warm-up), so the
    full 3×1M baseline finished in ~11 minutes, not ~2 h 15 m. Timing CSVs
    kept as-is for provenance; the summary JSONs hold the real wall times.
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
