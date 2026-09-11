# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_noncausal.yaml` (`causal: False`). Buffer 360 s (trained at 120 s), window stride 30 s, folds [0, 1, 2, 3, 4].

Buffer differs from the trained sequence length, so this is a train/test length mismatch. The control is latency 240 s, at which the causal arm sees exactly 120 s of history with its future masked, as in training: if its score there matches the trained-length run, length extrapolation is not affecting the comparison.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 360 | 194,611 | 0.7439 | 0.6593 | 0.6957 |
| 1 | 359 | 194,611 | 0.7455 | 0.6615 | 0.6961 |
| 2 | 358 | 194,611 | 0.7471 | 0.6635 | 0.6971 |
| 5 | 355 | 194,611 | 0.7469 | 0.6635 | 0.6976 |
| 10 | 350 | 194,611 | 0.7520 | 0.6699 | 0.7013 |
| 15 | 345 | 194,611 | 0.7532 | 0.6714 | 0.7020 |
| 20 | 340 | 194,611 | 0.7517 | 0.6696 | 0.7008 |
| 30 | 330 | 194,611 | 0.7545 | 0.6732 | 0.7032 |
| 45 | 315 | 194,611 | 0.7612 | 0.6820 | 0.7106 |
| 60 | 300 | 194,611 | 0.7602 | 0.6807 | 0.7091 |
| 90 | 270 | 194,611 | 0.7634 | 0.6849 | 0.7127 |
| 119 | 241 | 194,611 | 0.7682 | 0.6912 | 0.7186 |
| 120 | 240 | 194,611 | 0.7657 | 0.6879 | 0.7155 |
| 150 | 210 | 194,611 | 0.7658 | 0.6880 | 0.7155 |
| 180 | 180 | 194,611 | 0.7640 | 0.6857 | 0.7137 |
| 240 | 120 | 194,611 | 0.7565 | 0.6757 | 0.7055 |
| 300 | 60 | 194,611 | 0.7407 | 0.6550 | 0.6883 |
| 359 | 1 | 194,611 | 0.7288 | 0.6391 | 0.6766 |

At zero latency kappa is **0.6593**. The best latency is **137 s** at kappa **0.6984** (+0.0390).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.8954 | 0.4147 | 0.7543 | 0.6789 | 0.7351 |
| 1 | 0.8987 | 0.4165 | 0.7535 | 0.6754 | 0.7362 |
| 2 | 0.8994 | 0.4177 | 0.7556 | 0.6767 | 0.7360 |
| 5 | 0.8997 | 0.4201 | 0.7553 | 0.6756 | 0.7373 |
| 10 | 0.9026 | 0.4270 | 0.7612 | 0.6776 | 0.7382 |
| 15 | 0.9037 | 0.4305 | 0.7617 | 0.6751 | 0.7391 |
| 20 | 0.9031 | 0.4305 | 0.7594 | 0.6731 | 0.7378 |
| 30 | 0.9057 | 0.4252 | 0.7611 | 0.6741 | 0.7498 |
| 45 | 0.9080 | 0.4442 | 0.7689 | 0.6806 | 0.7514 |
| 60 | 0.9084 | 0.4343 | 0.7665 | 0.6766 | 0.7596 |
| 90 | 0.9098 | 0.4400 | 0.7690 | 0.6786 | 0.7659 |
| 119 | 0.9106 | 0.4604 | 0.7733 | 0.6806 | 0.7680 |
| 120 | 0.9103 | 0.4457 | 0.7708 | 0.6801 | 0.7705 |
| 150 | 0.9099 | 0.4471 | 0.7705 | 0.6789 | 0.7711 |
| 180 | 0.9090 | 0.4449 | 0.7683 | 0.6774 | 0.7691 |
| 240 | 0.9056 | 0.4317 | 0.7599 | 0.6717 | 0.7587 |
| 300 | 0.8995 | 0.4040 | 0.7425 | 0.6590 | 0.7365 |
| 359 | 0.8910 | 0.3846 | 0.7334 | 0.6554 | 0.7186 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-359: `lat_noncausal_b360.csv`.
