# PPO baseline evaluation on the original reward

### Model: `final` on LunarLander-v3 (original reward)

| seed | episodes | mean reward | SEM | min | max | landed % | crash % | timeout % | mean len | fuel cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | +250.5 | 2.7 | +124.1 | +296.1 | 98 | 0 | 2 | 384 | 57.0 |
| 1 | 100 | +257.1 | 2.1 | +209.3 | +300.4 | 100 | 0 | 0 | 350 | 49.9 |
| 2 | 100 | +256.0 | 2.0 | +209.5 | +293.1 | 100 | 0 | 0 | 363 | 54.2 |
| *pooled* | 300 | +254.6 | - | +124.1 | +300.4 | 99 | 0 | 1 | 366 | 53.7 |

`SEM` = standard error of the per-seed mean over its episodes; `fuel cost` = 0.3 * main + 0.03 * side engine firings.
