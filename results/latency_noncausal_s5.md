# Accuracy versus permitted latency

Model: `configs/sleep78_streaming_noncausal.yaml` (`causal: False`). Buffer 120 s, window stride 5 s, folds [0, 1, 2, 3, 4].

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 120 | 1,171,020 | 0.7658 | 0.6870 | 0.7142 |
| 1 | 119 | 1,171,020 | 0.7664 | 0.6879 | 0.7136 |
| 2 | 118 | 1,171,020 | 0.7667 | 0.6881 | 0.7136 |
| 5 | 115 | 1,171,020 | 0.7676 | 0.6896 | 0.7154 |
| 10 | 110 | 1,171,020 | 0.7704 | 0.6932 | 0.7171 |
| 15 | 105 | 1,171,020 | 0.7716 | 0.6947 | 0.7180 |
| 20 | 100 | 1,171,020 | 0.7724 | 0.6958 | 0.7188 |
| 30 | 90 | 1,171,020 | 0.7734 | 0.6971 | 0.7196 |
| 45 | 75 | 1,171,020 | 0.7731 | 0.6966 | 0.7189 |
| 60 | 60 | 1,171,020 | 0.7713 | 0.6942 | 0.7167 |
| 90 | 30 | 1,171,020 | 0.7637 | 0.6840 | 0.7079 |
| 119 | 1 | 1,171,020 | 0.7472 | 0.6621 | 0.6897 |

At zero latency kappa is **0.6870**. The best latency is **38 s** at kappa **0.6977** (+0.0107).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.9051 | 0.4496 | 0.7811 | 0.6904 | 0.7447 |
| 1 | 0.9074 | 0.4493 | 0.7798 | 0.6866 | 0.7450 |
| 2 | 0.9077 | 0.4491 | 0.7798 | 0.6866 | 0.7448 |
| 5 | 0.9088 | 0.4538 | 0.7802 | 0.6873 | 0.7469 |
| 10 | 0.9113 | 0.4554 | 0.7824 | 0.6871 | 0.7491 |
| 15 | 0.9126 | 0.4573 | 0.7830 | 0.6869 | 0.7503 |
| 20 | 0.9134 | 0.4588 | 0.7834 | 0.6873 | 0.7510 |
| 30 | 0.9141 | 0.4597 | 0.7844 | 0.6882 | 0.7517 |
| 45 | 0.9145 | 0.4578 | 0.7838 | 0.6872 | 0.7515 |
| 60 | 0.9141 | 0.4522 | 0.7816 | 0.6860 | 0.7497 |
| 90 | 0.9114 | 0.4336 | 0.7735 | 0.6807 | 0.7403 |
| 119 | 0.9027 | 0.3939 | 0.7584 | 0.6689 | 0.7247 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_noncausal_s5.csv`.
