# Pooled causality statistics

2 seeds x folds = **10 paired measurements**. A negative difference means the causal model scores lower.

| Seed | Folds | Causal | Non-causal | Difference | Causal loses |
|---|---:|---|---|---|---|
| 42 | 5 | 0.6634 | 0.6912 | -0.0278 | 5/5 |
| 43 | 5 | 0.6649 | 0.6893 | -0.0244 | 5/5 |
| **Pooled** | 10 | **0.6642** | **0.6903** | **-0.0261** | **10/10** |

- Paired t(9) = -6.11, p = 0.000176
- Wilcoxon signed-rank: p = 0.0020 (floor at n = 10 is 0.0020 when every difference shares a sign)
- Bootstrap 95% CI on the mean difference: [-0.0337, -0.0179]
- Cohen's d = -1.93

- Seed-to-seed spread of the effect: -0.0278 to -0.0244 (sd 0.0024)

The bootstrap interval is the more reliable summary; Cohen's d is inflated at small paired sample sizes.
