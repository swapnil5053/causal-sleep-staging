# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_causal.yaml` (`causal: True`). Buffer 240 s (trained at 120 s), window stride 30 s, folds [0, 1, 2, 3, 4].

Buffer differs from the trained sequence length, so this is a train/test length mismatch. The control is latency 120 s, at which the causal arm sees exactly 120 s of history with its future masked, as in training: if its score there matches the trained-length run, length extrapolation is not affecting the comparison.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 240 | 194,923 | 0.7746 | 0.6982 | 0.7304 |
| 1 | 239 | 194,923 | 0.7750 | 0.6987 | 0.7310 |
| 2 | 238 | 194,923 | 0.7757 | 0.6996 | 0.7316 |
| 5 | 235 | 194,923 | 0.7756 | 0.6994 | 0.7317 |
| 10 | 230 | 194,923 | 0.7752 | 0.6988 | 0.7311 |
| 15 | 225 | 194,923 | 0.7741 | 0.6975 | 0.7302 |
| 20 | 220 | 194,923 | 0.7711 | 0.6935 | 0.7267 |
| 30 | 210 | 194,923 | 0.7772 | 0.7016 | 0.7326 |
| 45 | 195 | 194,923 | 0.7760 | 0.7000 | 0.7315 |
| 60 | 180 | 194,923 | 0.7788 | 0.7036 | 0.7334 |
| 90 | 150 | 194,923 | 0.7792 | 0.7041 | 0.7328 |
| 119 | 121 | 194,923 | 0.7672 | 0.6878 | 0.7187 |
| 120 | 120 | 194,923 | 0.7792 | 0.7041 | 0.7317 |
| 150 | 90 | 194,923 | 0.7769 | 0.7008 | 0.7271 |
| 180 | 60 | 194,923 | 0.7691 | 0.6905 | 0.7162 |
| 239 | 1 | 194,923 | 0.4691 | 0.3123 | 0.4043 |

At zero latency kappa is **0.6982**. The best latency is **95 s** at kappa **0.7053** (+0.0071).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.8958 | 0.4555 | 0.8002 | 0.7220 | 0.7787 |
| 1 | 0.8957 | 0.4567 | 0.8005 | 0.7234 | 0.7790 |
| 2 | 0.8960 | 0.4576 | 0.8013 | 0.7239 | 0.7791 |
| 5 | 0.8961 | 0.4592 | 0.8007 | 0.7236 | 0.7790 |
| 10 | 0.8955 | 0.4586 | 0.8002 | 0.7240 | 0.7772 |
| 15 | 0.8948 | 0.4584 | 0.7990 | 0.7232 | 0.7757 |
| 20 | 0.8930 | 0.4561 | 0.7956 | 0.7175 | 0.7715 |
| 30 | 0.8966 | 0.4593 | 0.8035 | 0.7228 | 0.7806 |
| 45 | 0.8957 | 0.4619 | 0.8013 | 0.7219 | 0.7767 |
| 60 | 0.8976 | 0.4613 | 0.8054 | 0.7221 | 0.7804 |
| 90 | 0.8986 | 0.4618 | 0.8055 | 0.7201 | 0.7781 |
| 119 | 0.8918 | 0.4447 | 0.7919 | 0.7024 | 0.7627 |
| 120 | 0.8995 | 0.4592 | 0.8064 | 0.7199 | 0.7738 |
| 150 | 0.9002 | 0.4533 | 0.8046 | 0.7147 | 0.7628 |
| 180 | 0.8996 | 0.4367 | 0.7981 | 0.7073 | 0.7392 |
| 239 | 0.6596 | 0.1868 | 0.4877 | 0.3792 | 0.3082 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-239: `lat_causal_b240.csv`.
