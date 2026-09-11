# Pooled causality statistics

3 seeds x 5 folds = **15 paired measurements**. A negative difference means the causal model scores lower.

Per-seed results use the naive paired t-test (5 folds within one seed). The pooled figure across seeds uses the Nadeau-Bengio correction, since folds pooled across seeds still share overlapping training subjects and the naive pooled test overstates significance.

## Per-seed results (lead with these)

| Seed | Folds | Causal | Non-causal | Difference | Causal loses | t | p |
|---|---:|---|---|---|---|---|---|
| 42 | 5 | 0.6567 | 0.6922 | -0.0355 | 5/5 | -2.46 | 0.0694 |
| 43 | 5 | 0.6235 | 0.6412 | -0.0176 | 1/5 | -0.65 | 0.5495 |
| 44 | 5 | 0.6185 | 0.6643 | -0.0457 | 4/5 | -2.18 | 0.0947 |

## Pooled

- Causal: 0.6329, Non-causal: 0.6659, Difference: -0.0329, Causal loses: 10/15
- **Nadeau-Bengio corrected: t(14) = -1.27, p = 0.2233**
- Naive pooled t-test (folds treated as fully independent, for comparison): t(14) = -2.78, p = 0.0148
- Wilcoxon signed-rank: p = 0.0215
- Bootstrap 95% CI on the mean difference: [-0.0565, -0.0116]
- Cohen's d = -0.72

- Seed-to-seed spread of the effect: -0.0457 to -0.0176 (sd 0.0142)

The corrected pooled test is more conservative because it accounts for shared training subjects across folds. The bootstrap interval and per-seed results are the more reliable summary; Cohen's d is inflated at small paired sample sizes.
