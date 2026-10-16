# PPO evaluation on the original reward (by reward mode)

### Model: `baseline_final` on LunarLander-v3 (original reward)

| seed | episodes | mean reward | SEM | min | max | landed % | crash % | timeout % | mean len | fuel cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | +250.5 | 2.7 | +124.1 | +296.1 | 98 | 0 | 2 | 384 | 57.0 |
| 1 | 100 | +257.1 | 2.1 | +209.3 | +300.4 | 100 | 0 | 0 | 350 | 49.9 |
| 2 | 100 | +256.0 | 2.0 | +209.5 | +293.1 | 100 | 0 | 0 | 363 | 54.2 |
| *pooled* | 300 | +254.6 | - | +124.1 | +300.4 | 99 | 0 | 1 | 366 | 53.7 |

### Model: `time_penalty_final` on LunarLander-v3 (original reward)

| seed | episodes | mean reward | SEM | min | max | landed % | crash % | timeout % | mean len | fuel cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | +261.4 | 2.0 | +217.2 | +304.7 | 100 | 0 | 0 | 317 | 48.4 |
| 1 | 100 | +263.2 | 2.0 | +218.1 | +304.7 | 100 | 0 | 0 | 304 | 46.5 |
| 2 | 100 | +262.6 | 2.5 | +112.5 | +303.7 | 99 | 0 | 1 | 309 | 45.1 |
| *pooled* | 300 | +262.4 | - | +112.5 | +304.7 | 100 | 0 | 0 | 310 | 46.7 |

`SEM` = standard error of the per-seed mean over its episodes; `fuel cost` = 0.3 * main + 0.03 * side engine firings; `timeout %` is the hover-farming signal. Sections are grouped as `<reward_mode>_<model_source>`; files written before the `--mode` flag default to `baseline`.
