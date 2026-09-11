# Accuracy versus permitted latency

Model: `configs\sleep78_streaming_causal_wide.yaml` (`causal: True`). Buffer 120 s (trained at 120 s), window stride 5 s, folds [0, 1, 2, 3, 4].

Buffer equals the trained sequence length.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 ** | 120 | 1,171,020 | 0.7701 | 0.6923 | 0.7140 |
| 1 | 119 | 1,171,020 | 0.7695 | 0.6915 | 0.7134 |
| 2 | 118 | 1,171,020 | 0.7691 | 0.6910 | 0.7128 |
| 5 | 115 | 1,171,020 | 0.7697 | 0.6917 | 0.7133 |
| 10 | 110 | 1,171,020 | 0.7692 | 0.6911 | 0.7125 |
| 15 | 105 | 1,171,020 | 0.7688 | 0.6905 | 0.7118 |
| 20 | 100 | 1,171,020 | 0.7681 | 0.6896 | 0.7109 |
| 30 | 90 | 1,171,020 | 0.7665 | 0.6875 | 0.7087 |
| 45 | 75 | 1,171,020 | 0.7633 | 0.6832 | 0.7043 |
| 60 | 60 | 1,171,020 | 0.7589 | 0.6773 | 0.6983 |
| 90 | 30 | 1,171,020 | 0.7359 | 0.6474 | 0.6702 |
| 119 | 1 | 1,171,020 | 0.4303 | 0.2823 | 0.3736 |

At zero latency kappa is **0.6923**. The best latency is **0 s** at kappa **0.6923** (+0.0000).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.8937 | 0.4448 | 0.7992 | 0.6815 | 0.7507 |
| 1 | 0.8932 | 0.4438 | 0.7987 | 0.6808 | 0.7503 |
| 2 | 0.8930 | 0.4434 | 0.7982 | 0.6798 | 0.7498 |
| 5 | 0.8936 | 0.4431 | 0.7990 | 0.6813 | 0.7494 |
| 10 | 0.8936 | 0.4413 | 0.7987 | 0.6809 | 0.7482 |
| 15 | 0.8934 | 0.4397 | 0.7986 | 0.6809 | 0.7463 |
| 20 | 0.8932 | 0.4381 | 0.7980 | 0.6804 | 0.7445 |
| 30 | 0.8930 | 0.4339 | 0.7969 | 0.6797 | 0.7397 |
| 45 | 0.8921 | 0.4263 | 0.7945 | 0.6779 | 0.7306 |
| 60 | 0.8910 | 0.4159 | 0.7913 | 0.6760 | 0.7174 |
| 90 | 0.8840 | 0.3693 | 0.7720 | 0.6634 | 0.6621 |
| 119 | 0.6261 | 0.1379 | 0.4474 | 0.3369 | 0.3196 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_causal_wide_s5.csv`.
