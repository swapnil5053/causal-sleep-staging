# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_causal.yaml` (`causal: True`). Buffer 120 s, window stride 30 s, folds [0, 1, 2, 3, 4].

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 120 | 195,235 | 0.7795 | 0.7044 | 0.7318 |
| 1 | 119 | 195,235 | 0.7796 | 0.7045 | 0.7321 |
| 2 | 118 | 195,235 | 0.7802 | 0.7052 | 0.7324 |
| 5 | 115 | 195,235 | 0.7797 | 0.7047 | 0.7319 |
| 10 | 110 | 195,235 | 0.7790 | 0.7036 | 0.7310 |
| 15 | 105 | 195,235 | 0.7771 | 0.7012 | 0.7287 |
| 20 | 100 | 195,235 | 0.7736 | 0.6963 | 0.7243 |
| 30 | 90 | 195,235 | 0.7772 | 0.7011 | 0.7272 |
| 45 | 75 | 195,235 | 0.7728 | 0.6953 | 0.7218 |
| 60 | 60 | 195,235 | 0.7694 | 0.6908 | 0.7162 |
| 90 | 30 | 195,235 | 0.7453 | 0.6590 | 0.6847 |
| 119 | 1 | 195,235 | 0.4693 | 0.3125 | 0.4043 |

At zero latency kappa is **0.7044**. The best latency is **4 s** at kappa **0.7055** (+0.0011).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.8998 | 0.4591 | 0.8063 | 0.7199 | 0.7738 |
| 1 | 0.8996 | 0.4613 | 0.8065 | 0.7207 | 0.7723 |
| 2 | 0.9000 | 0.4626 | 0.8069 | 0.7200 | 0.7727 |
| 5 | 0.9001 | 0.4632 | 0.8065 | 0.7192 | 0.7704 |
| 10 | 0.8992 | 0.4634 | 0.8059 | 0.7178 | 0.7689 |
| 15 | 0.8980 | 0.4594 | 0.8039 | 0.7157 | 0.7664 |
| 20 | 0.8964 | 0.4551 | 0.7999 | 0.7090 | 0.7609 |
| 30 | 0.9005 | 0.4532 | 0.8046 | 0.7147 | 0.7628 |
| 45 | 0.8981 | 0.4493 | 0.8007 | 0.7094 | 0.7514 |
| 60 | 0.9000 | 0.4366 | 0.7981 | 0.7073 | 0.7391 |
| 90 | 0.8963 | 0.3863 | 0.7748 | 0.6877 | 0.6786 |
| 119 | 0.6601 | 0.1867 | 0.4876 | 0.3789 | 0.3080 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_causal.csv`.
