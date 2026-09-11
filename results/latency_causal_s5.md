# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_causal.yaml` (`causal: True`). Buffer 120 s, window stride 5 s, folds [0, 1, 2, 3, 4].

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 ** | 120 | 1,171,020 | 0.7768 | 0.7008 | 0.7291 |
| 1 | 119 | 1,171,020 | 0.7762 | 0.7000 | 0.7284 |
| 2 | 118 | 1,171,020 | 0.7758 | 0.6994 | 0.7278 |
| 5 | 115 | 1,171,020 | 0.7766 | 0.7004 | 0.7286 |
| 10 | 110 | 1,171,020 | 0.7762 | 0.6999 | 0.7279 |
| 15 | 105 | 1,171,020 | 0.7759 | 0.6995 | 0.7273 |
| 20 | 100 | 1,171,020 | 0.7753 | 0.6987 | 0.7263 |
| 30 | 90 | 1,171,020 | 0.7743 | 0.6973 | 0.7245 |
| 45 | 75 | 1,171,020 | 0.7716 | 0.6937 | 0.7204 |
| 60 | 60 | 1,171,020 | 0.7671 | 0.6877 | 0.7141 |
| 90 | 30 | 1,171,020 | 0.7439 | 0.6572 | 0.6840 |
| 119 | 1 | 1,171,020 | 0.4729 | 0.3168 | 0.4069 |

At zero latency kappa is **0.7008**. The best latency is **0 s** at kappa **0.7008** (+0.0000).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.8974 | 0.4594 | 0.8031 | 0.7157 | 0.7701 |
| 1 | 0.8972 | 0.4587 | 0.8026 | 0.7149 | 0.7686 |
| 2 | 0.8969 | 0.4581 | 0.8021 | 0.7135 | 0.7685 |
| 5 | 0.8976 | 0.4586 | 0.8030 | 0.7152 | 0.7686 |
| 10 | 0.8977 | 0.4575 | 0.8027 | 0.7145 | 0.7671 |
| 15 | 0.8978 | 0.4569 | 0.8027 | 0.7138 | 0.7652 |
| 20 | 0.8979 | 0.4556 | 0.8022 | 0.7124 | 0.7635 |
| 30 | 0.8980 | 0.4530 | 0.8015 | 0.7108 | 0.7592 |
| 45 | 0.8979 | 0.4467 | 0.7993 | 0.7078 | 0.7506 |
| 60 | 0.8973 | 0.4364 | 0.7956 | 0.7036 | 0.7378 |
| 90 | 0.8929 | 0.3925 | 0.7731 | 0.6812 | 0.6800 |
| 119 | 0.6645 | 0.1861 | 0.4930 | 0.3813 | 0.3097 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_causal_s5.csv`.
