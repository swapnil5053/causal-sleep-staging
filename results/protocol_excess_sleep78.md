# Protocol gain by arm: Sleep-EDF-78, TCN

Streaming stride 30 s against the tiled protocol. 3 seeds x 5 folds = **15 paired measurements**. A positive excess means the protocol change is worth more to the causal arm.

| Seed | Causal arm gains | Non-causal arm gains | Excess | Folds positive |
|---|---|---|---|---|
| 42 | +0.0348 | +0.0039 | +0.0310 | 5/5 |
| 43 | +0.0345 | +0.0026 | +0.0319 | 5/5 |
| 44 | +0.0355 | +0.0043 | +0.0312 | 5/5 |
| **Pooled** | **+0.0350** | **+0.0036** | **+0.0314** | **15/15** |

- Nadeau-Bengio corrected: t(14) = 12.69, p = 4.56e-09 (corrected SE 0.00247, factor 1/15 + 1/4)
- Bootstrap 95% CI on the mean excess: [+0.0291, +0.0334]
- Smallest single-fold excess: +0.0221
- Per-seed range: +0.0310 to +0.0319

Each gain is a within-arm quantity: the same trained weights scored on the same held-out seconds under two protocols. The excess therefore does not depend on the two arms being comparable to each other.
