# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_noncausal.yaml` (`causal: False`). Buffer 240 s (trained at 120 s), window stride 30 s, folds [0, 1, 2, 3, 4].

Buffer differs from the trained sequence length, so this is a train/test length mismatch. The control is latency 120 s, at which the causal arm sees exactly 120 s of history with its future masked, as in training: if its score there matches the trained-length run, length extrapolation is not affecting the comparison.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 240 | 194,923 | 0.7592 | 0.6791 | 0.7101 |
| 1 | 239 | 194,923 | 0.7601 | 0.6803 | 0.7098 |
| 2 | 238 | 194,923 | 0.7616 | 0.6822 | 0.7109 |
| 5 | 235 | 194,923 | 0.7613 | 0.6820 | 0.7114 |
| 10 | 230 | 194,923 | 0.7649 | 0.6865 | 0.7136 |
| 15 | 225 | 194,923 | 0.7653 | 0.6871 | 0.7137 |
| 20 | 220 | 194,923 | 0.7649 | 0.6865 | 0.7134 |
| 30 | 210 | 194,923 | 0.7658 | 0.6878 | 0.7140 |
| 45 | 195 | 194,923 | 0.7728 | 0.6969 | 0.7216 |
| 60 | 180 | 194,923 | 0.7689 | 0.6917 | 0.7167 |
| 90 | 150 | 194,923 | 0.7688 | 0.6915 | 0.7166 |
| 119 | 121 | 194,923 | 0.7732 | 0.6974 | 0.7220 |
| 120 | 120 | 194,923 | 0.7666 | 0.6886 | 0.7143 |
| 150 | 90 | 194,923 | 0.7616 | 0.6820 | 0.7088 |
| 180 | 60 | 194,923 | 0.7536 | 0.6714 | 0.6999 |
| 239 | 1 | 194,923 | 0.7438 | 0.6583 | 0.6901 |

At zero latency kappa is **0.6791**. The best latency is **107 s** at kappa **0.7022** (+0.0231).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.9023 | 0.4372 | 0.7715 | 0.6889 | 0.7508 |
| 1 | 0.9048 | 0.4370 | 0.7704 | 0.6860 | 0.7510 |
| 2 | 0.9055 | 0.4392 | 0.7723 | 0.6867 | 0.7509 |
| 5 | 0.9053 | 0.4418 | 0.7718 | 0.6861 | 0.7519 |
| 10 | 0.9080 | 0.4467 | 0.7754 | 0.6856 | 0.7526 |
| 15 | 0.9088 | 0.4486 | 0.7749 | 0.6834 | 0.7528 |
| 20 | 0.9084 | 0.4478 | 0.7742 | 0.6831 | 0.7534 |
| 30 | 0.9102 | 0.4445 | 0.7737 | 0.6804 | 0.7612 |
| 45 | 0.9133 | 0.4612 | 0.7816 | 0.6891 | 0.7627 |
| 60 | 0.9119 | 0.4483 | 0.7767 | 0.6818 | 0.7648 |
| 90 | 0.9117 | 0.4480 | 0.7762 | 0.6822 | 0.7649 |
| 119 | 0.9126 | 0.4661 | 0.7804 | 0.6837 | 0.7673 |
| 120 | 0.9108 | 0.4455 | 0.7729 | 0.6793 | 0.7629 |
| 150 | 0.9091 | 0.4356 | 0.7671 | 0.6748 | 0.7572 |
| 180 | 0.9059 | 0.4186 | 0.7585 | 0.6690 | 0.7474 |
| 239 | 0.8990 | 0.3991 | 0.7511 | 0.6678 | 0.7333 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-239: `lat_noncausal_b240.csv`.
