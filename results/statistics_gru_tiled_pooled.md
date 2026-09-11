# Pooled causality statistics

3 seeds x 5 folds = **15 paired measurements**. A negative difference means the causal model scores lower.

Per-seed results use the naive paired t-test (5 folds within one seed). The pooled figure across seeds uses the Nadeau-Bengio correction, since folds pooled across seeds still share overlapping training subjects and the naive pooled test overstates significance.

## Per-seed results (lead with these)

| Seed | Folds | Causal | Non-causal | Difference | Causal loses | t | p |
|---|---:|---|---|---|---|---|---|
| 42 | 5 | 0.6601 | 0.6964 | -0.0363 | 5/5 | -8.98 | 0.0009 |
| 43 | 5 | 0.6637 | 0.7053 | -0.0416 | 5/5 | -8.83 | 0.0009 |
| 44 | 5 | 0.6677 | 0.7048 | -0.0371 | 5/5 | -10.01 | 0.0006 |

## Pooled

- Causal: 0.6638, Non-causal: 0.7022, Difference: -0.0384, Causal loses: 15/15
- **Nadeau-Bengio corrected: t(14) = -7.60, p = 2.485e-06**
- Naive pooled t-test (folds treated as fully independent, for comparison): t(14) = -16.56, p = 1.37e-10
- Wilcoxon signed-rank: p = 0.0007 (floor at n = 15 is 0.0001 when every difference shares a sign)
- Bootstrap 95% CI on the mean difference: [-0.0427, -0.0340]
- Cohen's d = -4.28

- Seed-to-seed spread of the effect: -0.0416 to -0.0363 (sd 0.0028)

The corrected pooled test is more conservative because it accounts for shared training subjects across folds. The bootstrap interval and per-seed results are the more reliable summary; Cohen's d is inflated at small paired sample sizes.
