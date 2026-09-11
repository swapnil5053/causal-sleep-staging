# Pooled causality statistics

3 seeds x 5 folds = **15 paired measurements**. A negative difference means the causal model scores lower.

Per-seed results use the naive paired t-test (5 folds within one seed). The pooled figure across seeds uses the Nadeau-Bengio correction, since folds pooled across seeds still share overlapping training subjects and the naive pooled test overstates significance.

## Per-seed results (lead with these)

| Seed | Folds | Causal | Non-causal | Difference | Causal loses | t | p |
|---|---:|---|---|---|---|---|---|
| 42 | 5 | 0.6634 | 0.6912 | -0.0278 | 5/5 | -4.90 | 0.0080 |
| 43 | 5 | 0.6649 | 0.6893 | -0.0244 | 5/5 | -3.50 | 0.0248 |
| 44 | 5 | 0.6664 | 0.6888 | -0.0224 | 4/5 | -3.71 | 0.0207 |

## Pooled

- Causal: 0.6649, Non-causal: 0.6898, Difference: -0.0249, Causal loses: 14/15
- **Nadeau-Bengio corrected: t(14) = -3.36, p = 0.004652**
- Naive pooled t-test (folds treated as fully independent, for comparison): t(14) = -7.33, p = 3.75e-06
- Wilcoxon signed-rank: p = 0.0001
- Bootstrap 95% CI on the mean difference: [-0.0310, -0.0183]
- Cohen's d = -1.89

- Seed-to-seed spread of the effect: -0.0278 to -0.0224 (sd 0.0028)

The corrected pooled test is more conservative because it accounts for shared training subjects across folds. The bootstrap interval and per-seed results are the more reliable summary; Cohen's d is inflated at small paired sample sizes.
