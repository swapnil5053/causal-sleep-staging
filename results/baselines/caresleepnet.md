# CareSleepNet, reproduced

Multi-scale residual CNN feeding a Transformer encoder, 22.7 million parameters. Non-causal: the
encoder attends across the whole 20-epoch window in both directions.

## Settings

| | |
|---|---|
| Data | Sleep-EDF, 39 recordings, Fpz-Cz at 100 Hz |
| Epoch | 30 s |
| Classes | W, N1, N2, N3, REM |
| Validation | subject-wise 5-fold |
| Sequence | 20 epochs, 10 minutes of context |
| Hardware | RTX 4060 laptop GPU, about 70 minutes |

## Results

| Fold | Accuracy | Macro F1 | Kappa | Status |
|---|---|---|---|---|
| Best fold | 78.07% | 72.08% | 0.694 | verified against the results file |
| One fold | 38.12% | 31.05% | 0.210 | completed, anomalous |
| One fold | not recorded | not recorded | not recorded | lost to a logging failure |
| Two folds | not recovered | not recovered | not recovered | not recorded separately |

**No mean is reported.** An earlier draft computed an estimated mean of about 75.8% accuracy and
kappa 0.62 from the folds that survived. That estimate is withdrawn: it averages an incomplete
set of folds including one anomalous result and excluding one that was never recorded, so it is
neither the model's mean nor a bound on it. Only the verified best fold is citable, and it must
be labelled as a best fold wherever it appears.

Per-class figures below are from that best fold.

| Class | F1 |
|---|---|
| W | 0.8469 |
| N1 | 0.3210 |
| N2 | 0.8132 |
| N3 | 0.8324 |
| REM | 0.7906 |

## Defects that bound what this run can be cited for

Terminal output during the run suggested accuracies near 90% on some folds, which the saved
results file does not support. Where the two disagree, the file is taken as authoritative and the
terminal output is disregarded. This is recorded because the discrepancy is the reason the run
cannot be summarised by a mean.

The spread between the verified best fold at 78.07% and the anomalous fold at 38.12% is far
larger than the fold-to-fold variance seen in the other two baselines or in our own runs. Until
the run is repeated with complete logging, the honest reading is that this reproduction
established a lower bound on what the architecture achieves here and nothing more.

Comparing a best fold against another model's mean flatters the best fold. Any table placing this
number beside a mean must say so on the face of the table.
