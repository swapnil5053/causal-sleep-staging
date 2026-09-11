# Protocol gain by arm: Sleep-EDF-78, GRU, three seeds

Streaming stride 30 s against the tiled protocol. 3 seeds x 5 folds = **15 paired measurements**. A positive excess means the protocol change is worth more to the causal arm.

| Seed | Causal arm gains | Non-causal arm gains | Excess | Folds positive |
|---|---|---|---|---|
| 42 | +0.0342 | +0.0081 | +0.0260 | 5/5 |
| 43 | +0.0359 | +0.0021 | +0.0338 | 5/5 |
| 44 | +0.0359 | +0.0063 | +0.0296 | 5/5 |
| **Pooled** | **+0.0353** | **+0.0055** | **+0.0298** | **15/15** |

- Nadeau-Bengio corrected: t(14) = 7.99, p = 1.39e-06 (corrected SE 0.00373, factor 1/15 + 1/4)
- Bootstrap 95% CI on the mean excess: [+0.0265, +0.0330]
- Smallest single-fold excess: +0.0152
- Per-seed range: +0.0260 to +0.0338

Each gain is a within-arm quantity: the same trained weights scored on the same held-out seconds under two protocols. The excess therefore does not depend on the two arms being comparable to each other.
