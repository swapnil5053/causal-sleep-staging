# Accuracy versus permitted latency

Model: `configs/dodh/causal_s42.yaml` (`causal: True`). Buffer 120 s (trained at 120 s), window stride 5 s, folds [0, 1, 2, 3, 4].

Buffer equals the trained sequence length.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 120 | 147,397 | 0.7785 | 0.6922 | 0.7278 |
| 1 | 119 | 147,397 | 0.7774 | 0.6907 | 0.7266 |
| 2 | 118 | 147,397 | 0.7764 | 0.6894 | 0.7253 |
| 5 | 115 | 147,397 | 0.7788 | 0.6925 | 0.7280 |
| 10 | 110 | 147,397 | 0.7788 | 0.6926 | 0.7279 |
| 15 ** | 105 | 147,397 | 0.7793 | 0.6932 | 0.7281 |
| 20 | 100 | 147,397 | 0.7791 | 0.6928 | 0.7275 |
| 30 | 90 | 147,397 | 0.7790 | 0.6927 | 0.7272 |
| 45 | 75 | 147,397 | 0.7777 | 0.6909 | 0.7249 |
| 60 | 60 | 147,397 | 0.7751 | 0.6873 | 0.7212 |
| 90 | 30 | 147,397 | 0.7513 | 0.6559 | 0.6967 |
| 119 | 1 | 147,397 | 0.4247 | 0.2524 | 0.3931 |

At zero latency kappa is **0.6922**. The best latency is **15 s** at kappa **0.6932** (+0.0010).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.7772 | 0.4036 | 0.8332 | 0.8484 | 0.7765 |
| 1 | 0.7761 | 0.4020 | 0.8321 | 0.8469 | 0.7758 |
| 2 | 0.7743 | 0.3992 | 0.8313 | 0.8467 | 0.7748 |
| 5 | 0.7783 | 0.4049 | 0.8337 | 0.8482 | 0.7750 |
| 10 | 0.7787 | 0.4055 | 0.8339 | 0.8475 | 0.7740 |
| 15 | 0.7791 | 0.4064 | 0.8347 | 0.8473 | 0.7731 |
| 20 | 0.7797 | 0.4047 | 0.8347 | 0.8467 | 0.7719 |
| 30 | 0.7815 | 0.4049 | 0.8351 | 0.8459 | 0.7688 |
| 45 | 0.7839 | 0.3986 | 0.8346 | 0.8436 | 0.7638 |
| 60 | 0.7861 | 0.3887 | 0.8337 | 0.8411 | 0.7562 |
| 90 | 0.7810 | 0.3440 | 0.8162 | 0.8243 | 0.7181 |
| 119 | 0.4627 | 0.1711 | 0.4709 | 0.5028 | 0.3580 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_dodh_causal.csv`.
