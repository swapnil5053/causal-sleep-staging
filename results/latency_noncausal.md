# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_noncausal.yaml` (`causal: False`). Buffer 120 s, window stride 30 s, folds [0, 1, 2, 3, 4].

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 120 | 195,235 | 0.7706 | 0.6933 | 0.7192 |
| 1 | 119 | 195,235 | 0.7715 | 0.6945 | 0.7189 |
| 2 | 118 | 195,235 | 0.7719 | 0.6950 | 0.7188 |
| 5 | 115 | 195,235 | 0.7721 | 0.6955 | 0.7201 |
| 10 | 110 | 195,235 | 0.7737 | 0.6976 | 0.7208 |
| 15 | 105 | 195,235 | 0.7736 | 0.6974 | 0.7201 |
| 20 | 100 | 195,235 | 0.7732 | 0.6968 | 0.7196 |
| 30 | 90 | 195,235 | 0.7694 | 0.6916 | 0.7146 |
| 45 | 75 | 195,235 | 0.7765 | 0.7011 | 0.7228 |
| 60 | 60 | 195,235 | 0.7642 | 0.6847 | 0.7084 |
| 90 | 30 | 195,235 | 0.7538 | 0.6707 | 0.6962 |
| 119 | 1 | 195,235 | 0.7562 | 0.6742 | 0.7001 |

At zero latency kappa is **0.6933**. The best latency is **46 s** at kappa **0.7013** (+0.0079).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.9086 | 0.4558 | 0.7865 | 0.6965 | 0.7485 |
| 1 | 0.9108 | 0.4560 | 0.7855 | 0.6937 | 0.7484 |
| 2 | 0.9113 | 0.4554 | 0.7859 | 0.6938 | 0.7478 |
| 5 | 0.9116 | 0.4607 | 0.7852 | 0.6936 | 0.7493 |
| 10 | 0.9129 | 0.4620 | 0.7863 | 0.6923 | 0.7506 |
| 15 | 0.9138 | 0.4607 | 0.7852 | 0.6894 | 0.7513 |
| 20 | 0.9130 | 0.4604 | 0.7847 | 0.6887 | 0.7515 |
| 30 | 0.9130 | 0.4478 | 0.7797 | 0.6830 | 0.7497 |
| 45 | 0.9165 | 0.4669 | 0.7868 | 0.6903 | 0.7534 |
| 60 | 0.9110 | 0.4321 | 0.7741 | 0.6793 | 0.7453 |
| 90 | 0.9061 | 0.4074 | 0.7637 | 0.6712 | 0.7330 |
| 119 | 0.9074 | 0.4137 | 0.7674 | 0.6784 | 0.7335 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_noncausal.csv`.
