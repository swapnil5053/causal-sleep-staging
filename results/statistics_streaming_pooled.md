# Pooled causality statistics

3 seeds x folds = **15 paired measurements**. A negative difference means the causal model scores lower.

| Seed | Folds | Causal | Non-causal | Difference | Causal loses |
|---|---:|---|---|---|---|
| 42 | 5 | 0.6634 | 0.6912 | -0.0278 | 5/5 |
| 43 | 5 | 0.6649 | 0.6893 | -0.0244 | 5/5 |
| 44 | 5 | 0.6664 | 0.6888 | -0.0224 | 4/5 |
| **Pooled** | 15 | **0.6649** | **0.6898** | **-0.0249** | **14/15** |

- Paired t(14) = -7.33, p = 3.75e-06
- Wilcoxon signed-rank: p = 0.0001
- Bootstrap 95% CI on the mean difference: [-0.0310, -0.0183]
- Cohen's d = -1.89

- Seed-to-seed spread of the effect: -0.0278 to -0.0224 (sd 0.0028)

The bootstrap interval is the more reliable summary; Cohen's d is inflated at small paired sample sizes.
