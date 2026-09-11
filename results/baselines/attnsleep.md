# AttnSleep, reproduced

Multi-scale residual CNN with an attention block, about 48,000 parameters. Run at the standard
30-second epoch and at a 5-second epoch with an extended context window.

## Settings

| | |
|---|---|
| Data | Sleep-EDF, 39 recordings, Fpz-Cz at 100 Hz |
| Classes | W, N1, N2, N3, REM |
| Validation | 10-fold |
| Epoch sizes | 30 s and 5 s |
| Sequence | 20 epochs at 30 s; 20 then 120 epochs at 5 s |

## Results

| | 30 s | 5 s, 20-epoch sequence | 5 s, 120-epoch sequence |
|---|---|---|---|
| Accuracy | 80.90% | 72.57% | 81.28% |
| Macro F1 | 0.755 | 0.7297 | about 0.81 |
| Kappa | 0.7088 | 0.6262 | about 0.75 |
| Context | 600 s | 100 s | 600 s |
| Training samples | 3000 | 500 | 500 |
| Training time | 25.6 min | 20 min | about 35 min |

Per fold at 5 s with the 120-epoch sequence: 79.51, 78.04, 86.30, 79.20, 79.53, 81.70, 87.94,
78.20, 80.50, 81.80, mean 81.28%, standard deviation about 3.2 points.

Per-class figures below are from fold 2 at 81.77% accuracy, a single fold, not a mean.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| W | 0.96 | 0.93 | 0.94 |
| N1 | 0.32 | 0.41 | 0.36 |
| N2 | 0.81 | 0.83 | 0.82 |
| N3 | 0.86 | 0.89 | 0.88 |
| REM | 0.83 | 0.69 | 0.76 |

## Defects that bound what this run can be cited for

**The 5-second rows are not comparable to the 30-second row.** The 30-second baseline was trained
on 3000 samples and both 5-second variants on 500. The 8.7-point accuracy difference between the
two 5-second rows is therefore the effect of context length at a fixed sample size, which is a
valid internal comparison, but the difference between the 5-second and 30-second rows confounds
context, epoch size and training set size at once.

**Kappa and macro F1 for the 5-second 120-epoch run are approximate**, recorded as "about 0.75"
and "about 0.81" rather than computed exactly. Only the 30-second row carries an exact kappa, so
only the 30-second row should appear in a comparison table.

For that reason the comparison in `../../RESULTS.md` cites the 30-second row, 80.90% accuracy and
kappa 0.7088, and not the 5-second variants.

## Notes

The finding that a 5-second epoch matches a 30-second epoch once total context is held equal is
interesting on its own and consistent with the protocol result in this repository: what a
sequence model is scored on depends far more on how much history it carries than on how the
history is chopped up. It is reported here as an observation from a single 10-fold run at a
reduced sample size, not as a result of this work.
