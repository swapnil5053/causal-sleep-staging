# Statistical analysis of the causality ablation

Per-fold Cohen's kappa, causal model versus the same architecture with the causal
constraint removed. Identical parameters, data and folds; the only difference is access
to future signal. Paired tests, since folds are matched.

## Summary

| Dataset | Causal | Non-causal | Difference | 95% CI | t | p | Cohen's d | Folds causal loses |
|---|---|---|---|---|---|---|---|---|
| Sleep-EDF-78 streaming | 0.6634 | 0.6912 | -0.0278 | [-0.0372, -0.0178] | -4.90 | 0.0080 | -2.19 | 5/5 |

## Sleep-EDF-78 streaming

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.6237 | 0.6652 | -0.0415 |
| 1 | 0.6444 | 0.6641 | -0.0197 |
| 2 | 0.6713 | 0.7063 | -0.0350 |
| 3 | 0.7116 | 0.7444 | -0.0328 |
| 4 | 0.6661 | 0.6762 | -0.0101 |
| Mean | 0.6634 | 0.6912 | -0.0278 |

- Paired t-test: t(4) = -4.90, p = 0.0080
- Wilcoxon signed-rank: p = 0.0625
- Bootstrap 95% CI on the mean difference: [-0.0372, -0.0178]
- Cohen's d = -2.19
- Fold-to-fold kappa sd: causal 0.0329, non-causal 0.0343
- Folds needed for 80% power at this effect size: 2
- Folds needed for 90% power: 3

