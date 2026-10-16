# Progress — Lunar Lander Reward Shaping Lab (Phase 1)

Project: train a baseline PPO agent on Gymnasium's `LunarLander-v3` with the
built-in reward, and build the evaluation tooling. Phase 2 will add custom
reward shaping on top of this lab.

## Step log

- [x] **Step 0 — Repo init.** `git init` in `d:\Projects\LunarLander`.

**Phase 1 status: COMPLETE.** All six steps done; working tree clean on
`master` (9 commits). Ready for Phase 2 (custom reward shaping variants,
compared against this baseline with the same evaluation tooling).
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
    (`ppo_lander_seed{S}.zip` final + `..._{steps}_steps.zip` every 50k),
    `eval/evaluations.npz` (EvalCallback, deterministic, 10 episodes per eval),
    `best/best_model.zip`, `monitor/*.monitor.csv` (16 per env), and
    `summary.json` run metadata. TensorBoard events under
    `results/tb/ppo_seed{S}_1/` (results/tb is gitignored).
  - Env choice: a single-run 50k-step comparison favoured `SubprocVecEnv`
    (885.7 vs 376.6 steps/s), but stacking subproc ×16 on top of 3 parallel
    seeds means ~50 processes that each re-import torch (Windows spawn
    re-runs the main module), which exceeded memory on this 15.7 GB machine —
    the spawn wave hit `OMP: Error #111` and workers died importing Box2D.
    Reverted to the single-process `DummyVecEnv` (the pre-approved setup):
    3 parallel processes, ~240 MB each, ~1-2 GB total.
  - Training launched as 3 parallel background processes
    (`--seed 0|1|2`, logs in `results/logs/`, gitignored). Each run overshot
    1M to 1,015,808 steps because PPO finishes the current 16×1024 rollout.
    Real wall times and throughput (from each run's own summary line):

    | seed | wall (min) | steps/s | best eval reward | mean of last 10 evals |
    |------|-----------|---------|------------------|-----------------------|
    | 0    | 23.1      | 733.4   | 256.1            | 245.9                 |
    | 1    | 23.8      | 711.5   | 272.6            | 224.5                 |
    | 2    | 24.4      | 692.7   | 255.3            | 214.5                 |

    All three converged to solid landing policies (the "solved" bar is ~200)
    and for every seed the final model is also the best-eval model.
    The three summaries were backfilled from the run logs
    (`backfilled_from` key) because `train.py` only gained `summary.json`
    persistence after the runs finished; all three final/best models were
    re-loaded successfully as an integrity check.
- [x] **Step 4 — Evaluation tool.** `scripts/evaluate.py` (100 deterministic
  episodes per seed on the *unmodified* built-in reward; fixed episode-seed
  sequence so every model sees the same 100 initial states) and
  `scripts/aggregate.py` (writes `results/evaluations/summary.json` +
  `summary.md`). Per-episode JSON records reward, length, outcome
  (landed/crashed/timeout, from the sign of the ±100 termination reward) and
  the fuel proxy `0.3*main + 0.03*side` firings. Results over 300 episodes:

  | seed | mean reward | SEM | landed | crash | timeout | fuel |
  |------|------------|-----|--------|-------|---------|------|
  | 0    | +250.5     | 2.7 | 98%    | 0%    | 2%      | 57.0 |
  | 1    | +257.1     | 2.1 | 100%   | 0%    | 0%      | 49.9 |
  | 2    | +256.0     | 2.0 | 100%   | 0%    | 0%      | 54.2 |
  | pooled | +254.6   | —   | 99%    | 0%    | 1%      | 53.7 |

  Baseline: **+254.6 ± 3.5 across seeds** (99% of episodes ≥ 200, the classic
  "solved" bar; zero crashes). Seed 0 had one low outlier (+124.1: landed
  outside the pad at |x| < 1, which still pays +100 minus shaping/fuel).
- [x] **Step 5 — Video & plots.** `scripts/record_video.py` (one deterministic
  episode per seed, same initial state as evaluation episode 0, saved to
  `videos/ppo_lander_seed{0,1,2}.mp4`; rewards +258.2 / +278.0 / +270.5) and
  `scripts/plot_curves.py` (`results/figures/learning_curves.png`, raw +
  smoothed eval curves per seed with the 200-point solved bar). The videos
  themselves are gitignored as regenerable artifacts.
- [x] **Step 6 — Tests & README.** `tests/` — 18 pytest tests over all script
  helpers (config loading, checkpoint resume logic, episode outcome/fuel
  accounting and determinism, evaluate_seed JSON round trip, pooling and
  seed-level spread, smoothing, curve loading); 18 passed in 6.6 s. README
  rewritten with the full command list and the real training/evaluation
  numbers above.

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



---

# Phase 2 — Reward shaping lab (planned)

Add reward wrappers (time penalty, a deliberately exploitable hover bonus,
and a potential-based fix), train each with the *identical* PPO recipe,
then evaluate every model with the **original unmodified reward** and
compare honestly. Nothing is evaluated through a shaping wrapper; every
mode differs from the baseline only in the reward signal.

## Phase 2 step log

- [x] **P2-1 — Reward wrapper (`src/reward_wrappers.py`).** One
  `ShapingRewardWrapper(gym.Wrapper)`, mode selected in config:
  `builtin` (identity control), `time_penalty` (small configurable per-frame
  cost, start 0.05), `hover_bonus` (+1/frame when the lander is upright and
  slow — absolute angle and |vx|,|vy| below thresholds measured from real
  baseline landings; deliberately exploitable because it ignores position),
  `potential_shaped` (potential Phi = weighted sum of distance to pad, speed
  and tilt; shaping term `gamma * Phi(s') - Phi(s)` with PPO's gamma, plus a
  small per-frame cost — telescoping, so it cannot be farmed by hovering).
  Wrapper records the original reward and the shaping term per step and
  accumulates both per episode in `info`, so the original reward can always
  be logged/evaluated.
  **Wrapper order (verified empirically, 2-episode smoke run):**
  `make_vec_env` applies `Monitor` *first* and `wrapper_class` on top
  (verified in `stable_baselines3/common/env_util.py`), so passing the
  wrapper as `wrapper_class` makes Monitor log the ORIGINAL reward. Instead
  we build the stack explicitly as
  `Monitor(ShapingRewardWrapper(gym.make(...)))` with
  `info_keywords=("episode_original", "episode_shaping")`: smoke-run output
  confirmed CSV columns `['r', 'l', 't', 'episode_original',
  'episode_shaping']` where `r` is the SHAPED reward and the keyword columns
  are the per-episode originals — both reward streams logged, clearly
  labelled.
  Unit tests: builtin-identical to the raw env; potential term telescopes to
  ~0 over a hover-in-place sequence; hover bonus fires exactly when the
  conditions hold (boundary cases included).
- [x] **P2-2 — Training runs.** `train.py` extended with `--mode`
  (run dirs `results/runs/<mode>/seed_{S}/`; baseline stays at
  `results/runs/seed_{S}`), reward block in the YAML configs (one config
  per mode, every attempted config kept). Same total timesteps, network and
  hyperparameters for every mode — only the reward differs. `EvalCallback`
  always scores the original reward (eval env is never wrapped). Runs are
  background, resumable, with a printed estimated wall time. Hover diagnostics
  (episode length, timeout flag, landing rate, fraction of frames in the hover
  state) via `scripts/inspect_hover.py` on the raw env. **Each launch waits
  for explicit go-ahead.** Sequence: time_penalty x3 -> hover_bonus v1 (+1)
  x3 -> potential_shaped x3; if v1 still lands, hover v2 (+2/+3 per frame,
  larger budget), each attempt reported honestly.
  - **Tooling DONE (uncommitted→committed):** `train.py` builds the training
    env as `Monitor(ShapingRewardWrapper(gym.make(...)))` with seeding that
    mirrors SB3's `make_vec_env` exactly (verified against
    `stable_baselines3/common/env_util.py`: `action_space.seed(seed + rank)`
    per env, then `vec_env.seed(seed)`); eval env stays unwrapped.
    Smoke-run output in `time_penalty` mode confirmed Monitor CSV header
    `r,l,t,episode_original,episode_shaping` with `r` = shaped reward and
    `-2.95 = 59 × -0.05` for `episode_shaping`. `--resume` verified against
    the new per-mode dirs. Configs `configs/ppo_{time_penalty,hover_bonus,
    potential_shaped}.yaml` carry the *identical* Zoo hyperparameters —
    enforced by a new test (`test_all_reward_configs_share_identical_
    hyperparameters`). `scripts/inspect_hover.py` verified on the baseline
    model (20 episodes: +249.9 ± 7.9, 95% landed, 5% timeout, **88% of
    frames in the hover state**, mean hover run 239 frames — matches the
    87% threshold measurement, so the hover criterion describes what a good
    landing policy already does). 29 tests pass.
  - **time_penalty runs DONE (2026-10-16):** 3 seeds × 1,000,000 steps
    (1,015,808 actual, matching Phase 1), three concurrent background
    processes, 34.9 min/seed at 485.6 steps/s. Per-seed `summary.json`
    confirms `reward.mode = time_penalty`, `time_penalty: 0.05`, identical
    hyperparameters, zero error-log bytes. Final EvalCallback (original
    reward, deterministic): seed 0 +266.5 / seed 1 +263.2 / seed 2 +267.6
    (best-ever +267.7 / +275.1 / +267.6). Learning curve compared to the
    baseline at matched 50k-step evals: roughly equal shape, with the
    time_penalty policy slightly *ahead* at the intermediate 500k evals on
    seeds 1 and 2 — the penalty is not slowing learning here.
    `inspect_hover.py` on the three final models (20 episodes each, original
    reward): +263.4 / +265.6 / +266.4, **100%/100%/100% landed, 0 timeouts**,
    mean hover run **35/50/48 frames** vs the baseline's 239, mean episode
    length 305-321 (max 347) vs the baseline's timeout-containing 5%.
    The time penalty produced short, decisive landings — the opposite of
    hover-farming.
  - **hover_bonus + potential_shaped first attempts INVALIDATED and
    retrained (2026-10-16):** the first 3 x hover_bonus campaign (and the
    first ~20 min of potential_shaped, killed) was silently trained on the
    *builtin* reward. Cause: `train.py` read the reward mode from the YAML's
    `reward` block (the baseline config has none -> `builtin`) while
    `--mode` only picked the output directory — banner mode and wrapper mode
    disagreed. Detection: the hover 100-episode eval produced episodes
    byte-identical to baseline's, its model's policy weights are
    bitwise-identical to baseline's (same-seed deterministic replay of the
    builtin training), and every one of its 2,406 monitor episodes logged
    `episode_shaping = 0.0` (the healthy time_penalty run showed -16.34).
    Fix: new `resolve_reward_cfg()` routes the effective mode (CLI `--mode`,
    else the config's `reward.mode`) into the config that builds the
    wrapper, + 4 regression tests (51 pass). Verified with a 16,384-step
    smoke run (`--mode hover_bonus` + baseline config): mean
    `episode_shaping = +7.07`. All invalid artifacts deleted (run dirs, eval
    JSONs, TensorBoard logs); both campaigns relaunched with their per-mode
    configs (`configs/ppo_{hover_bonus,potential_shaped}.yaml`, params
    pinned). Monitors after 90 s: hover shaping +9.2/+14.3/+10.0 per
    episode, potential shaping nonzero in 100% of episodes — the wrappers
    are on. Lesson: every campaign launch records the config file it used,
    and `episode_shaping != 0` in the first monitor batch is the bring-up
    check for any shaped run.
  - **hover_bonus + potential_shaped retrained (2026-10-16):** 3 seeds each
    x 1,000,000 steps (1,015,808 actual), all six as one concurrent batch,
    25.0-26.6 min/seed for hover (637-678 steps/s) and 19.5-22.0 min/seed
    for potential (770-869 steps/s) — six DummyVecEnv runs fit the 12
    threads with room to spare. Per-seed summary.json confirms the correct
    `reward.mode` and params. Training-time accounting (monitor CSVs):
    hover_bonus mean episode shaping +254.9/+307.7/+279.0 with original
    reward -41.6/-44.0/-49.5 (the agent maximizes the bonus, not the
    landing); potential_shaped mean shaping -1.97/-1.95/-1.91 — the
    telescoping potential stays small by construction, exactly as designed.
    100-episode formal evaluations (original reward): hover_bonus
    +50.3/-53.1/-10.4 per seed (61%/99%/94% timeouts, landed 39%/1%/4%,
    mean length 963-998 — hovering at the cap); potential_shaped
    +264.9/+266.6/+237.4 (95-100% landed, 0-4% timeouts, solved 96%).
    `inspect_hover.py` (20 episodes, original reward): hover_bonus models
    spend 95-96% of frames in the hover state with mean hover RUN 684-799
    consecutive frames (vs the baseline's 239 and time_penalty's 35-50) and
    land 0-30% — full reward hacking; potential_shaped passes through the
    hover state in 28-92-frame bursts mid-descent (50-80% of frames) and
    lands 95-100%. Final comparison (`compare.py`, 300 paired episodes vs
    baseline): baseline +254.6 +/- 3.5; time_penalty +262.4 +/- 0.9 (paired
    +7.8, CI [+6.1, +9.6] -> SIGNIFICANT); hover_bonus -4.4 +/- 51.9
    (paired -259.0, CI [-267.2, -250.7] -> SIGNIFICANT, catastrophic);
    potential_shaped +256.3 +/- 16.4 (paired +1.8, CI [-2.1, +5.3] -> WITHIN
    NOISE, the theory-correct form is neutral on the true objective here).
  - **Smart App Control blocker RESOLVED:** on the retry signal (~19:20
    local) `import torch`/`stable_baselines3` were healthy again and all 29
    tests passed before relaunch; the policy flip was transient-enforcement
    timing, and no exemption was needed. (Original blocker notes kept below
    for the record.)
  - **BLOCKED (2026-10-16, ~18:55 local): Smart App Control flipped from
    evaluation to enforcement mid-session.** Windows Code Integrity now
    blocks `torch_python.dll` (unsigned) sandbox-wide:
    `OSError: [WinError 4551] An Application Control policy has blocked this
    file` on *any* `import torch`/`stable_baselines3`, in the foreground and
    in spawned processes. Evidence: CodeIntegrity/Operational event IDs
    3033/3077 ("did not meet the Enterprise signing level requirements",
    Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6}) plus 3118 ("Smart App
    Control Block"); `Get-AuthenticodeSignature` → NotSigned. numpy/gymnasium
    still import and the pure-wrapper tests pass (7/7, no torch needed), so
    only torch-backed work is affected. This happened *after* all Phase 1
    training and *after* today's smoke runs — torch loaded fine earlier
    (SAC evaluation mode allows unsigned files; enforcement blocks them —
    a one-way, machine-level change). Verified: retries fail identically,
    network is up (rules out a reputation-lookup outage), so this is a hard
    blocker on this machine, not a transient. All P2-2 tooling above stays
    committed and ready; the training/eval/compare steps need the policy
    lifted (SAC off in Windows Security — note Microsoft marks that
    permanent for this install — or an org WDAC exemption) or a move to a
    machine without enforcement.
- [x] **P2-3 — Honest evaluation.** `evaluate.py --mode` (100 deterministic
  episodes/seed on the original reward, wrapper never used for evaluation;
  files `ppo_<mode>_seed_<S>_<src>.json`, baseline keeps the legacy
  `ppo_seed_<S>_<src>.json` name); `aggregate.py` groups by
  `<reward_mode>_<model_source>` (legacy JSONs default to baseline) and its
  markdown calls out `timeout %` as the hover-farming signal. New
  `scripts/compare.py` writes, under `results/evaluations/`:
  `comparison.md` / `csv` / `json` (per-mode mean across seeds +/- seed
  spread, pooled mean, landed/crash/timeout rates, mean length, fuel,
  solved>=200%), `comparison.png` (bar chart with seed-level error bars),
  `paired_differences.md` (paired mean difference vs baseline over the
  shared initial states + bootstrap 95% CI, "WITHIN NOISE" stated plainly)
  and `learning_curves.png` (per-mode Monitor `r` vs `episode_original`
  overlay). Monitor accounting spot-checked: time_penalty training gap
  between original and shaped rewards = 15.3 = 0.05 x mean episode length
  306, exactly as designed. **time_penalty vs baseline (300 paired
  episodes): mean +262.4 +/- 0.9 across seeds vs +254.6 +/- 3.5; paired
  difference +7.8, 95% CI [+6.1, +9.6] -> SIGNIFICANT on these initial
  states (caveat: 3 seeds is thin for a claim about the recipe itself).**
  39 tests pass.
- [x] **P2-4 — Videos.** `record_video.py --mode` (mode-aware model
  discovery mirroring `evaluate.py`; uniform naming
  `videos/<mode>_seed{S}.mp4` for baseline / time_penalty / hover_bonus /
  potential_shaped; the Phase 1 legacy `ppo_lander_seed{S}.mp4` files stay
  untouched). New `scripts/side_by_side.py` (moviepy) composes one clip with
  the hover-bonus episode on the left and the potential-shaped episode on
  the right (shorter side padded with black). 8 new tests (47 total).
  Recorded so far: `baseline_seed{0,1,2}.mp4` (+258.2 / +278.0 / +270.5 —
  byte-identical to the Phase 1 legacy recordings, i.e. reproducible) and
  `time_penalty_seed{0,1,2}.mp4` (+280.7 / +283.2 / +281.8, lengths
  332/313/320 — no hover-farming visible: short decisive landings). After
  the retrain: `hover_bonus_seed{0,1,2}.mp4` (+168.9 len 902 / -60.9 len
  1000 / -21.8 len 1000 — two episodes never land, one lands after a long
  hover) and `potential_shaped_seed{0,1,2}.mp4` (+283.4 len 324 / +296.0
  len 248 / +256.0 len 415 — clean decisive landings), plus
  `side_by_side_hover_bonus_vs_potential_shaped_seed0.mp4` (1200x400,
  hover left / potential right, 18 s). Videos stay gitignored as
  regenerable artifacts.
- [ ] **P2-5 — Write-up.** `reports/results.md` (comparison table, plots,
  factual per-mode sections, failures and surprises), README update with a
  beginner-level explanation of reward shaping / reward hacking and
  reproduction commands, `docs/resume_bullets.md` (2 resume bullets + 1
  LinkedIn post draft, verified numbers only, unconfirmed claims flagged),
  and a "What I learned" section with prompts for the human to answer
  (not answered by the agent).
- [ ] **P2-6 — Verify & report.** Full `pytest` run; honest completion
  report: which tests passed, which runs finished/failed, how many
  hover-bonus attempts were needed, whether reward hacking actually
  appeared, and what the human must still do manually (watch the videos,
  write their reflections, publish the repo).

**Rules for Phase 2:** no invented results; evaluation always uses the
original reward; every mode uses identical hyperparameters except the
reward; no unrequested features.
