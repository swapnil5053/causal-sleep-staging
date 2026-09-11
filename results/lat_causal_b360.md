# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_causal.yaml` (`causal: True`). Buffer 360 s (trained at 120 s), window stride 30 s, folds [0, 1, 2, 3, 4].

Buffer differs from the trained sequence length, so this is a train/test length mismatch. The control is latency 240 s, at which the causal arm sees exactly 120 s of history with its future masked, as in training: if its score there matches the trained-length run, length extrapolation is not affecting the comparison.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 360 | 194,611 | 0.7627 | 0.6825 | 0.7196 |
| 1 | 359 | 194,611 | 0.7632 | 0.6832 | 0.7205 |
| 2 | 358 | 194,611 | 0.7635 | 0.6836 | 0.7207 |
| 5 | 355 | 194,611 | 0.7637 | 0.6839 | 0.7211 |
| 10 | 350 | 194,611 | 0.7641 | 0.6844 | 0.7215 |
| 15 | 345 | 194,611 | 0.7630 | 0.6829 | 0.7201 |
| 20 | 340 | 194,611 | 0.7603 | 0.6793 | 0.7171 |
| 30 | 330 | 194,611 | 0.7660 | 0.6869 | 0.7231 |
| 45 | 315 | 194,611 | 0.7655 | 0.6863 | 0.7226 |
| 60 | 300 | 194,611 | 0.7686 | 0.6903 | 0.7253 |
| 90 | 270 | 194,611 | 0.7716 | 0.6943 | 0.7282 |
| 119 | 241 | 194,611 | 0.7627 | 0.6824 | 0.7179 |
| 120 | 240 | 194,611 | 0.7743 | 0.6979 | 0.7303 |
| 150 | 210 | 194,611 | 0.7769 | 0.7013 | 0.7325 |
| 180 | 180 | 194,611 | 0.7785 | 0.7032 | 0.7333 |
| 240 | 120 | 194,611 | 0.7790 | 0.7038 | 0.7317 |
| 300 | 60 | 194,611 | 0.7688 | 0.6902 | 0.7161 |
| 359 | 1 | 194,611 | 0.4687 | 0.3120 | 0.4043 |

At zero latency kappa is **0.6825**. The best latency is **218 s** at kappa **0.7049** (+0.0224).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.8915 | 0.4376 | 0.7872 | 0.7176 | 0.7639 |
| 1 | 0.8919 | 0.4389 | 0.7873 | 0.7200 | 0.7645 |
| 2 | 0.8920 | 0.4389 | 0.7878 | 0.7200 | 0.7649 |
| 5 | 0.8924 | 0.4422 | 0.7873 | 0.7200 | 0.7635 |
| 10 | 0.8923 | 0.4439 | 0.7879 | 0.7206 | 0.7628 |
| 15 | 0.8915 | 0.4434 | 0.7863 | 0.7179 | 0.7614 |
| 20 | 0.8903 | 0.4419 | 0.7825 | 0.7126 | 0.7581 |
| 30 | 0.8923 | 0.4432 | 0.7908 | 0.7207 | 0.7685 |
| 45 | 0.8922 | 0.4466 | 0.7892 | 0.7194 | 0.7658 |
| 60 | 0.8934 | 0.4469 | 0.7936 | 0.7203 | 0.7724 |
| 90 | 0.8940 | 0.4506 | 0.7973 | 0.7233 | 0.7756 |
| 119 | 0.8887 | 0.4446 | 0.7857 | 0.7049 | 0.7657 |
| 120 | 0.8953 | 0.4555 | 0.8002 | 0.7220 | 0.7786 |
| 150 | 0.8962 | 0.4594 | 0.8035 | 0.7228 | 0.7806 |
| 180 | 0.8972 | 0.4614 | 0.8054 | 0.7221 | 0.7804 |
| 240 | 0.8991 | 0.4593 | 0.8064 | 0.7199 | 0.7738 |
| 300 | 0.8993 | 0.4368 | 0.7981 | 0.7073 | 0.7392 |
| 359 | 0.6590 | 0.1868 | 0.4878 | 0.3795 | 0.3083 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-359: `lat_causal_b360.csv`.
