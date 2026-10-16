# Paired differences vs baseline (original reward)
Episodes are paired by initial state: every model was evaluated on the same episode seeds, so the difference removes initial-state luck. 95% CIs are bootstrap (10,000 resamples of the paired episodes).

## time_penalty

- paired episodes: 300 (seeds [0, 1, 2])
- mean difference vs baseline: +7.8
- 95% bootstrap CI: [+6.1, +9.6]
- verdict: **SIGNIFICANT** (a CI straddling 0 means the difference is not distinguishable from noise at this episode count)
