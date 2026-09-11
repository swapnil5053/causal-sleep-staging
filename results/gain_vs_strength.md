# Protocol gain against absolute score

Each row is one arm: one set of trained weights, scored on its own held-out
seconds under both protocols. The gain is within-arm, so arms that are not
comparable to each other still contribute comparable gains.

| Arm | Kind | Folds | Tiled | Streaming | Gain |
|---|---|---:|---|---|---|
| GRU causal, Sleep-EDF-78 | causal | 5 | 0.6601 | 0.6942 | +0.0342 |
| TCN causal, Sleep-EDF-78 | causal | 5 | 0.6634 | 0.6982 | +0.0348 |
| TCN causal wide, Sleep-EDF-78 | causal | 5 | 0.6535 | 0.6901 | +0.0365 |
| TCN causal, DOD-H | causal | 5 | 0.6567 | 0.6942 | +0.0374 |
| TCN control, Sleep-EDF-78 | control | 5 | 0.6912 | 0.6951 | +0.0039 |
| TCN control, DOD-H | control | 5 | 0.6922 | 0.7002 | +0.0080 |
| GRU control, Sleep-EDF-78 | control | 5 | 0.6964 | 0.7045 | +0.0081 |

- 4 causal arms, gains +0.0342 to +0.0374, spread 0.0033
- 3 control arms, gains +0.0039 to +0.0081
- Absolute score across the causal arms spans 0.0081 kappa, 0.6901 to 0.6982

- Across causal arms: Pearson r = -0.474 (p = 0.526), Spearman rho = -0.600 (p = 0.4)

**The correlation above resolves nothing and should not be quoted.** The
causal arms differ by less than 0.02 kappa in absolute score, so there is
no variation in strength to correlate a gain against. What the table does
support is the narrower statement: across these arms, spanning two
datasets, two architecture families and a range of capacities, the
protocol gain is nearly constant.

Control arms are excluded from the correlation deliberately. A control scores
higher and gains less by construction, so pooling the two kinds would restate the
causal and control difference as though it were a relationship with strength.
