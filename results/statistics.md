# Statistical analysis of the causality ablation

Per-fold Cohen's kappa, causal model versus the same architecture with the causal
constraint removed. Identical parameters, data and folds; the only difference is access
to future signal. Paired tests, since folds are matched.

## Summary

| Dataset | Causal | Non-causal | Difference | 95% CI | t | p | Cohen's d | Folds causal loses |
|---|---|---|---|---|---|---|---|---|
| Sleep-EDF-20 | 0.6625 | 0.6482 | +0.0143 | [-0.0052, +0.0343] | 1.26 | 0.2762 | 0.56 | 2/5 |
| Sleep-EDF-78 | 0.6335 | 0.6570 | -0.0235 | [-0.0322, -0.0129] | -4.24 | 0.0133 | -1.89 | 5/5 |

## Sleep-EDF-20

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.6854 | 0.6904 | -0.0050 |
| 1 | 0.7343 | 0.6867 | +0.0476 |
| 2 | 0.6010 | 0.6122 | -0.0112 |
| 3 | 0.5363 | 0.5300 | +0.0063 |
| 4 | 0.7553 | 0.7215 | +0.0338 |
| Mean | 0.6625 | 0.6482 | +0.0143 |

- Paired t-test: t(4) = 1.26, p = 0.2762
- Wilcoxon signed-rank: p = 0.4375
- Bootstrap 95% CI on the mean difference: [-0.0052, +0.0343]
- Cohen's d = 0.56
- Fold-to-fold kappa sd: causal 0.0922, non-causal 0.0773
- Folds needed for 80% power at this effect size: 25
- Folds needed for 90% power: 34

## Sleep-EDF-78

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.6144 | 0.6363 | -0.0219 |
| 1 | 0.6366 | 0.6408 | -0.0042 |
| 2 | 0.6250 | 0.6549 | -0.0299 |
| 3 | 0.6830 | 0.7070 | -0.0240 |
| 4 | 0.6085 | 0.6462 | -0.0377 |
| Mean | 0.6335 | 0.6570 | -0.0235 |

- Paired t-test: t(4) = -4.24, p = 0.0133
- Wilcoxon signed-rank: p = 0.0625
- Bootstrap 95% CI on the mean difference: [-0.0322, -0.0129]
- Cohen's d = -1.89
- Fold-to-fold kappa sd: causal 0.0297, non-causal 0.0288
- Folds needed for 80% power at this effect size: 3
- Folds needed for 90% power: 3

## Interpretation

On Sleep-EDF-78 the causal model is worse in 5 of 5 folds and the
difference is significant (p = 0.0133). The confidence interval [-0.0322, -0.0129] excludes
zero.

On Sleep-EDF-20 the same comparison gives +0.0143 with p = 0.2762 and a
confidence interval [-0.0052, +0.0343] that spans zero. Fold-to-fold
variance is 3.1 times larger there (kappa sd 0.0922 against
0.0297), which is enough to hide an effect of this size and even flip its sign.

The practical conclusion is that causality penalties reported on 20-subject splits are
unreliable, and that the true cost of causal operation for this architecture is around
0.024 kappa.
