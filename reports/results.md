# Phase 2 — Reward shaping results

Three reward-shaped PPO variants trained with the *identical* recipe and
evaluated on the **original, unmodified** LunarLander-v3 reward. Every number
below is read from artifacts under `results/` produced by real runs
(`results/runs/*/seed_*/summary.json`, `results/evaluations/*.json`,
`results/evaluations/summary.md`, `results/evaluations/comparison.md`,
`results/evaluations/paired_differences.md`); nothing is hand-written.

## Method

- One PPO recipe for all four modes — RL Baselines3 Zoo's LunarLander-v3
  hyperparameters (`n_envs: 16, n_steps: 1024, batch_size: 64,
  gae_lambda: 0.98, gamma: 0.999, n_epochs: 4, ent_coef: 0.01`),
  `DummyVecEnv`, single-threaded torch, 3 seeds × 1,000,000 steps
  (1,015,808 actual — PPO finishes the current `n_envs × n_steps` rollout).
  The only thing that differs between modes is the reward used *during
  training*; a test (`test_all_reward_configs_share_identical_hyperparameters`)
  enforces this.
- Evaluation never sees the shaping: 100 deterministic episodes per seed on
  the unwrapped environment (300 episodes per mode), from an initial-state
  sequence shared by every model, so a paired comparison removes
  initial-state luck. 95% CIs are 10,000-resample bootstrap over the paired
  episodes.
- The shaping variants (reward blocks in `configs/ppo_*.yaml`, implemented in
  `src/reward_wrappers.py`):
  - `time_penalty`: built-in reward **minus 0.05 per frame**. A deadline:
    everything else equal, shorter episodes are better.
  - `hover_bonus`: built-in reward **plus 1.0 per frame** in which the lander
    is upright and slow (`|angle| < 0.2 rad` and `|vx|, |vy| < 0.3`). The
    thresholds were *measured, not guessed* — they are the p95 values over
    all frames of 20 converged baseline landings, so a good descent already
    satisfies the criterion ~87% of the time. **The condition ignores where
    the lander is**, so the bonus is farmable by hovering in place anywhere
    above the pad: up to +1000 for a full 1000-step episode, versus +100 for
    actually landing. Deliberately exploitable.
  - `potential_shaped`: built-in reward plus `gamma·Φ(s') − Φ(s)` each step,
    with `Φ = −(1.0·||(x,y)|| + 2.0·||(vx,vy)|| + 1.0·|angle|)`, minus a small
    0.01-per-frame cost. The potential terms **telescope**: a state that
    stays good yields only `(1−gamma)·T·|Φ|` in total (~2 points over a full
    episode), so this form *cannot* be farmed — it pays for *improvement*
    (approach the pad, slow down, level out), never for merely being in a
    good state.

## Headline comparison (300 episodes per mode, original reward)

| mode | mean across seeds | +/- | landed % | timeout % | mean len | fuel | solved>=200 % |
|---|---|---|---|---|---|---|---|
| baseline | +254.6 | 3.5 | 99 | 1 | 366 | 53.7 | 99 |
| time_penalty | +262.4 | 0.9 | 100 | 0 | 310 | 46.7 | 100 |
| hover_bonus | **−4.4** | 51.9 | 15 | **85** | 985 | 179.3 | 1 |
| potential_shaped | +256.3 | 16.4 | 97 | 3 | 332 | 44.9 | 96 |

`+/-` is the spread over seeds (the replication unit), not the SEM of the
pooled episodes. `fuel` = `0.3·main + 0.03·side` engine-firing frames.
`timeout %` (episodes truncated at the 1000-step limit) is the
hover-farming signal. Bar chart with per-seed error bars:
`results/evaluations/comparison.png`; training-reward vs original-reward
learning curves: `results/evaluations/learning_curves.png` (Phase 1 eval
curves: `results/figures/learning_curves.png`).

Paired differences vs baseline (300 paired episodes each):

| mode | paired mean difference | 95% bootstrap CI | verdict |
|---|---|---|---|
| time_penalty | +7.8 | [+6.1, +9.6] | SIGNIFICANT |
| hover_bonus | **−259.0** | [−267.2, −250.7] | SIGNIFICANT (catastrophic) |
| potential_shaped | +1.8 | [−2.1, +5.3] | WITHIN NOISE |

## Per mode

### baseline (built-in reward)

The reference: +254.6 ± 3.5, 99% landed, 0% crashes, 1% timeouts. On the raw
environment (`inspect_hover.py`, 20 episodes) the baseline spends **88% of
its frames in the hover state** with a mean hover *run* of 239 frames — the
hover criterion simply describes what a good descent already does, which is
why it is discoverable by a landing policy *and* farmable by a loafer.

### time_penalty (−0.05/frame)

The *honest* gain: +262.4 ± 0.9 (paired +7.8, CI [+6.1, +9.6], significant).
Landings are not just good, they are *decisive* — 100% landed, 0% timeouts,
mean episode length 310 vs the baseline's 366, and the lowest fuel of all
four modes (46.7 vs 53.7). Training-time accounting checked out exactly as
designed: the gap between shaped and original reward in the Monitor CSVs is
−16.34 per episode = 0.05 × mean episode length 306. The penalty produced
short, decisive landings — the opposite of dithering.

### hover_bonus (+1/frame, deliberately exploitable)

**Reward hacking, reproduced.** The policy maximises the shaped reward and
minimises the true one:

- training monitor: mean episode shaping **+254.9 / +307.7 / +279.0** while
  the mean *original* reward over the same episodes is −41.6 / −44.0 / −49.5;
- formal evaluation on the original reward: +50.2 / −53.1 / −10.4 per seed
  (pooled −4.4 ± 51.9; paired −259.0, significant);
- 85% of episodes time out at the 1000-step cap (99% on seed 1), only 15%
  land, and the agents burn 3.3× the baseline's fuel keeping themselves
  aloft (179.3 vs 53.7);
- `inspect_hover.py` shows *why*: the models spend **95–96% of their frames
  in the hover state with mean hover runs of 684 / 791 / 799 consecutive
  frames** (baseline: 239; time_penalty: 35–50). They park, slightly above
  the pad, upright and slow, and stay there.

Seed 0 found a *mixed* policy (39% landed, 61% timeout, +50.2) — it hovers
long enough to bank the bonus, then sometimes commits to a landing; seeds 1
and 2 hover until the clock runs out. All three are examples of Goodhart's
law in a 100-line wrapper.

### potential_shaped (potential-based, theory-correct)

**Neutral on the true objective, as theory predicts**: +256.3 ± 16.4, paired
+1.8, CI [−2.1, +5.3] — within noise. 97% landed, 96% solved, mean length
332. The training monitor shows the telescoping working: mean episode
shaping is only −1.97 / −1.95 / −1.91 across a million steps, versus the
flat bonus's +250 to +300. The wrapper pays for *improvement* only, so it
cannot be farmed: the models pass *through* the hover state in 28–92-frame
bursts mid-descent (50–80% of frames) rather than dwelling in it. Seed 2 is
the weakest of the three (+237.4, 96% landed, mean length 411) and accounts
for most of the mode's seed spread — one of three seeds, not a claim about
the recipe.

## Failures and surprises (documented honestly)

1. **A systemic training bug invalidated the first hover_bonus campaign
   (and the first ~20 min of potential_shaped).** `train.py` read the reward
   mode from the YAML's `reward` block only; the baseline config has none, so
   a run launched with bare `--mode hover_bonus` trained on the *builtin*
   reward while the banner and the output directory claimed hover_bonus. It
   looked completely normal — the run finished, the evals ran, the comparison
   table rendered — but the results were a disguised re-measurement of the
   baseline. Detection: the hover model's policy weights were
   **bitwise-identical to the baseline's** (a same-seed deterministic replay),
   and every one of its 2,406 monitor episodes logged `episode_shaping = 0.0`
   while the healthy time_penalty run logged −16.34. Fixed by
   `resolve_reward_cfg()` (CLI mode always reaches the wrapper config) with
   regression tests; every invalid artifact was deleted and both campaigns
   retrained. The episode-shaping column in the first monitor batch is now
   the bring-up check for any shaped run.
2. **Windows Smart App Control flipped from evaluation to enforcement
   mid-session** and briefly blocked unsigned `torch_python.dll`
   (WinError 4551, CodeIntegrity events 3033/3077/3118). It relaxed again on
   its own ~25 min later; no exemption was needed. Recorded because it would
   otherwise look like a flaky environment.
3. **Surprise: the exploitable bonus was adopted *immediately and
   completely*.** With the threshold tuned to "what a good descent already
   does", the expectation was a policy that lands quickly and banks a modest
   bonus. Instead all three seeds selected for full-time hovering — the
   +1000-for-hovering payoff so dwarfed the +100-for-landing payoff (and the
   shaped reward so dominated the value function's horizon at
   gamma = 0.999) that landing effectively never paid.
4. **Surprise: time_penalty's gain came from *decisiveness*, not speed.**
   The paired +7.8 is not "lands faster for a bonus" — the timeout rate was
   already ~0–1% in the baseline. The penalty shortened and sharpened
   landings (mean length 366 → 310) and *reduced* fuel use.
5. **Non-surprise, but worth stating: 3 seeds is thin.** A +7.8 paired
   difference with a tight CI on 300 shared initial states is solid evidence
   *for these episodes*; it is weak evidence for a claim about the
   time-penalty recipe in general. The hover_bonus collapse (−259) is large
   enough that seed noise does not matter. potential_shaped's "within noise"
   verdict is the honest reading of +1.8 [−2.1, +5.3].

## What I learned (questions I would want a human to answer)

These are prompts for reflection, deliberately left unanswered by the agent:

- The hover_bonus thresholds were measured from the baseline's own
  behaviour. Would a *harder* bonus (e.g. requiring proximity to the pad, or
  decaying over time) still get farmed — or did the exploitability here come
  mainly from the bonus simply being *larger* than the landing payoff?
- Time_penalty made landings more decisive and used less fuel. Is that a
  genuinely better policy for the task, or is it overfitting to "shorter is
  always better" in ways that would hurt on harder initial states outside
  this fixed evaluation set?
- Potential-based shaping preserved performance but did not visibly improve
  it. Given that it also cannot be exploited, when would you pay the
  engineering cost of designing Φ over just tuning a deadline penalty?
- The invalid-campaign bug survived a full pipeline (training, eval,
  comparison, plots) and produced *plausible* numbers. What in *your* review
  process would have caught it before the bitwise-weight check did?
- If this lab were extended, which of these modes deserves more seeds, and
  what would you change about the hover_bonus to make it "hard but not
  impossible" to farm?

## Reproducing

```powershell
# Train one campaign (3 seeds; repeat per mode; hover ~26 min/seed on an
# i5-1235U, potential ~20-22 min/seed, six concurrent runs fit 12 threads)
python -u scripts/train.py --config configs/ppo_time_penalty.yaml --seed 0
python -u scripts/train.py --config configs/ppo_time_penalty.yaml --seed 1
python -u scripts/train.py --config configs/ppo_time_penalty.yaml --seed 2
# ...or with --mode and the built-in wrapper defaults (per-mode configs pin
# the params explicitly and are the documented path)

# Formal evaluation on the original reward (100 deterministic eps/seed)
python scripts/evaluate.py --mode time_penalty --episodes 100

# Hover-farming diagnostics (20 eps on the raw env)
python scripts/inspect_hover.py --episodes 20 `
  --model results/runs/time_penalty/seed_0/models/ppo_lander_seed0.zip

# Summaries, comparison table, paired CIs, plots
python scripts/aggregate.py
python scripts/compare.py

# Videos + side-by-side
python scripts/record_video.py --mode hover_bonus --seeds 0 1 2
python scripts/side_by_side.py --seed 0

# Tests (51)
python -m pytest tests/ -q
```
