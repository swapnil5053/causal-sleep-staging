# Accuracy versus permitted latency

Model: `configs/dodh/noncausal_s42.yaml` (`causal: False`). Buffer 120 s (trained at 120 s), window stride 5 s, folds [0, 1, 2, 3, 4].

Buffer equals the trained sequence length.

Latency L means the label for second *t* is emitted at *t + L*, so the model sees L seconds of future and `buffer - L` seconds of past. L = 0 is real time.

| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |
|---:|---:|---:|---|---|---|
| 0 | 120 | 147,397 | 0.7734 | 0.6859 | 0.7205 |
| 1 | 119 | 147,397 | 0.7734 | 0.6862 | 0.7213 |
| 2 | 118 | 147,397 | 0.7742 | 0.6872 | 0.7224 |
| 5 | 115 | 147,397 | 0.7786 | 0.6933 | 0.7273 |
| 10 | 110 | 147,397 | 0.7797 | 0.6959 | 0.7309 |
| 15 | 105 | 147,397 | 0.7819 | 0.6990 | 0.7330 |
| 20 | 100 | 147,397 | 0.7830 | 0.7004 | 0.7340 |
| 30 | 90 | 147,397 | 0.7851 | 0.7033 | 0.7356 |
| 45 | 75 | 147,397 | 0.7863 | 0.7048 | 0.7360 |
| 60 | 60 | 147,397 | 0.7837 | 0.7013 | 0.7327 |
| 90 | 30 | 147,397 | 0.7735 | 0.6870 | 0.7201 |
| 119 | 1 | 147,397 | 0.7435 | 0.6469 | 0.6865 |

At zero latency kappa is **0.6859**. The best latency is **46 s** at kappa **0.7058** (+0.0198).

## Per-class F1 by latency

The aggregate penalty under batch tiling is dominated by short-history positions. This table is what decides whether a per-class claim measured under tiling still holds at the latency a deployment actually runs at.

| Latency (s) | W | N1 | N2 | N3 | REM |
|---:|---|---|---|---|---|
| 0 | 0.7469 | 0.3711 | 0.8277 | 0.8346 | 0.8222 |
| 1 | 0.7558 | 0.3665 | 0.8265 | 0.8367 | 0.8210 |
| 2 | 0.7593 | 0.3691 | 0.8268 | 0.8363 | 0.8208 |
| 5 | 0.7639 | 0.3802 | 0.8315 | 0.8382 | 0.8226 |
| 10 | 0.7861 | 0.3816 | 0.8280 | 0.8359 | 0.8231 |
| 15 | 0.7895 | 0.3852 | 0.8302 | 0.8362 | 0.8241 |
| 20 | 0.7913 | 0.3858 | 0.8315 | 0.8367 | 0.8246 |
| 30 | 0.7931 | 0.3882 | 0.8338 | 0.8363 | 0.8269 |
| 45 | 0.7933 | 0.3879 | 0.8354 | 0.8349 | 0.8283 |
| 60 | 0.7907 | 0.3800 | 0.8335 | 0.8316 | 0.8278 |
| 90 | 0.7807 | 0.3512 | 0.8262 | 0.8228 | 0.8194 |
| 119 | 0.7456 | 0.2670 | 0.8016 | 0.8088 | 0.8092 |

For a causal model this curve should be flat or falling: it cannot read future signal, so a larger L only costs it history. For a non-causal model it should rise, and the rise is what lookahead is actually worth in seconds of delay.

Each latency is scored on a different 1-in-stride subsample of seconds, all uniform over the same recordings. Read the shape of the curve; treat small differences between adjacent latencies as sampling noise.

Full curve for every latency 0-119: `latency_dodh_noncausal.csv`.
