# Per-class causality cost

Causal-minus-non-causal F1 difference per sleep stage, pooled across 3 seeds x 5 folds (15 measurements per class) on the streaming Sleep-EDF-78 runs. A negative difference means the causal model scores lower on that class.

| Class | Causal F1 | Non-causal F1 | Difference | n |
|---|---|---|---|---:|
| REM | 0.6945 | 0.7542 | -0.0598 | 15 |
| N1 | 0.4212 | 0.4460 | -0.0248 | 15 |
| W | 0.8931 | 0.9112 | -0.0181 | 15 |
| N3 | 0.6654 | 0.6734 | -0.0080 | 15 |
| N2 | 0.7728 | 0.7752 | -0.0024 | 15 |

The largest causal cost falls on **REM** (-0.0598 F1), and the smallest on **N2** (-0.0024 F1).

The cost concentrates in REM and N1: REM is normally disambiguated using eye-movement and muscle-tone context around the epoch, and N1 is an inherently transitional stage, so both lose more when the model cannot look ahead. N2, N3 and W have locally distinctive signal (spindles/K-complexes, slow waves, clear alpha/beta) and are barely affected.
