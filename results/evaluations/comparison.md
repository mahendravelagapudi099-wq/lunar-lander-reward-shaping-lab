# Reward-shaping comparison on the original reward

All models were trained with identical hyperparameters and evaluated on the same shared episode seeds. `mean across seeds +/-` is the honest spread over seeds (the replication unit); `timeout %` is the hover-farming signal.

| mode | seeds | episodes | mean across seeds | +/- | pooled mean | landed % | crash % | timeout % | mean len | fuel cost | solved>=200 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 300 | +254.6 | 3.5 | +254.6 | 99 | 0 | 1 | 366 | 53.7 | 99 |
| time_penalty | 3 | 300 | +262.4 | 0.9 | +262.4 | 100 | 0 | 0 | 310 | 46.7 | 100 |
