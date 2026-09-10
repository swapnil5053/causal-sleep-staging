# Protocol gain by arm: DOD-H, TCN

Streaming stride 30 s against the tiled protocol. 3 seeds x 5 folds = **15 paired measurements**. A positive excess means the protocol change is worth more to the causal arm.

| Seed | Causal arm gains | Non-causal arm gains | Excess | Folds positive |
|---|---|---|---|---|
| 42 | +0.0374 | +0.0080 | +0.0294 | 5/5 |
| 43 | +0.0299 | +0.0014 | +0.0285 | 5/5 |
| 44 | +0.0361 | +0.0064 | +0.0297 | 5/5 |
| **Pooled** | **+0.0345** | **+0.0052** | **+0.0292** | **15/15** |

- Nadeau-Bengio corrected: t(14) = 6.48, p = 1.44e-05 (corrected SE 0.00451, factor 1/15 + 1/4)
- Bootstrap 95% CI on the mean excess: [+0.0253, +0.0332]
- Smallest single-fold excess: +0.0173
- Per-seed range: +0.0285 to +0.0297

Each gain is a within-arm quantity: the same trained weights scored on the same held-out seconds under two protocols. The excess therefore does not depend on the two arms being comparable to each other.
