# Resume bullets and LinkedIn post — Lunar Lander Reward Shaping Lab

All numbers below are read from artifacts in this repository
(`results/evaluations/summary.md`, `comparison.md`,
`paired_differences.md`) and the training summaries under `results/runs/`.
Everything is reproducible on a CPU-only laptop (~25 min per seed).

## Resume bullets (2)

- Trained a PPO baseline on Gymnasium's `LunarLander-v3` (3 seeds × 1M
  steps, +254.6 ± 3.5 points, 99% landing rate over 300 deterministic
  episodes), then built and compared three reward-shaping variants under
  identical hyperparameters; a deliberately exploitable "+1 per frame of
  hovering" bonus collapsed performance to −4.4 (85% episode timeouts,
  700+ frame hover runs — a live demonstration of reward hacking), while a
  potential-based (telescoping) shaping stayed neutral (+256.3, within
  noise) and a per-frame time penalty improved it (+262.4; paired +7.8,
  bootstrap 95% CI [+6.1, +9.6] over 300 paired initial states).
- Diagnosed and fixed a silent training-pipeline bug in which a CLI
  `--mode` flag selected only the output directory, not the reward, so an
  entire campaign trained on the wrong reward while producing
  plausible-looking results — caught it via per-episode reward-shaping
  accounting in the Monitor logs and bitwise-identical policy weights, added
  a regression test, and re-ran the affected campaigns; documented the
  incident and its detection method in the project log.

## LinkedIn post draft

**I taught a rocket to hover forever — on purpose.** 🚀

My latest RL lab: a PPO agent on LunarLander (solid baseline: +254.6 points,
99% landings), plus three hand-written reward-shaping variants trained with
identical hyperparameters and graded on the *original* reward:

- A time penalty (−0.05/frame) made landings more decisive: +262.4, 100%
  landings, less fuel burned.
- A "hover bonus" (+1/frame for being upright and slow) — deliberately
  exploitable — broke the agent completely: 85% of episodes time out at the
  1000-step cap with the lander parked above the pad. It maximises my score
  and never lands. Goodhart's law, in a 100-line wrapper.
- A potential-based reward (gamma·Φ(s') − Φ(s), which telescopes and so
  can't be farmed) kept performance intact: +256.3, statistically
  indistinguishable from baseline.

The best part? The exploitable variant looked completely successful in every
metric — until I noticed its trained weights were byte-identical to the
baseline's. A CLI flag had renamed the output folder but not the reward, so
the whole "shaped" campaign had trained on the default reward. Lesson
installed: check what the reward *actually was*, not what the banner says;
I now sanity-check per-episode shaping deltas before trusting any run.

Code, training curves, paired-difference CIs, and honest failure notes:
[repo link]

# Claims status

- Verified (artifact-backed): every number in the bullets and post —
  baseline +254.6 ± 3.5 / 99% landed; time_penalty +262.4 ± 0.9, 100%
  landed, paired +7.8 CI [+6.1, +9.6]; hover_bonus −4.4, 85% timeouts,
  hover runs 684–799 frames; potential_shaped +256.3 ± 16.4, paired +1.8
  CI [−2.1, +5.3]; the bit-identical-weights bug detection and fix.
- Unconfirmed (do not state as fact without more seeds): that the
  time_penalty *recipe* generalises (3 seeds, one environment); that
  potential shaping could never help (neutral here, not proven optimal);
  any claim about sample efficiency or wall-clock advantage vs other
  algorithms; the "~25 min per seed" figure is machine-specific
  (i5-1235U laptop) and not a portable benchmark claim.
