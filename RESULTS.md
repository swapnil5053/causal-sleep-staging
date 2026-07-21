# Results

Subject-wise 5-fold cross-validation on Sleep-EDF-20 (20 subjects, 39 recordings), single
channel Fpz-Cz at 100 Hz. Wake trimmed to 30 minutes either side of each night's sleep period.
Checkpoints selected on validation kappa, early stopping with patience 8.

## Headline

Best causal configuration is run B (120 s context).

| Metric | Value |
|---|---|
| Accuracy | 74.4% |
| Cohen's kappa | 0.662 |
| Macro F1 | 0.688 |
| Accuracy at 30 s granularity | 76.1% |
| Kappa at 30 s granularity | 0.684 |
| Parameters | 30,757 |
| CPU inference | under 0.074 ms per second of EEG (Intel Core i9-14900HX) |

Measured latency, Intel Core i9-14900HX, 200 runs:

| Configuration | ms per 1 s of EEG | Target |
|---|---|---|
| 60 s context, 3 TCN blocks | 0.045 | 3.0 |
| 120 s context, 4 TCN blocks | 0.074 | 3.0 |

Run B (120 s, 3 blocks) sits between the two and has not been benchmarked separately; it is
bounded above by the 4-block figure.

## Per fold, run B

| Fold | Test subjects | Accuracy | Kappa | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 19, 05, 14, 04 | 0.7657 | 0.6854 | 0.7213 | 0.801 | 0.384 | 0.846 | 0.831 | 0.745 |
| 1 | 09, 13, 15, 18 | 0.7980 | 0.7343 | 0.7317 | 0.910 | 0.323 | 0.820 | 0.874 | 0.731 |
| 2 | 06, 12, 17, 10 | 0.6971 | 0.6010 | 0.6401 | 0.681 | 0.303 | 0.784 | 0.682 | 0.750 |
| 3 | 01, 11, 02, 16 | 0.6440 | 0.5363 | 0.5999 | 0.745 | 0.309 | 0.663 | 0.604 | 0.678 |
| 4 | 07, 08, 00, 03 | 0.8133 | 0.7553 | 0.7464 | 0.896 | 0.362 | 0.841 | 0.864 | 0.770 |
| Mean | | 0.7436 | 0.6625 | 0.6879 | 0.807 | 0.336 | 0.791 | 0.771 | 0.735 |

Per-class columns are F1 scores.

## Configuration ablation

| Run | Configuration | Accuracy | Kappa | Macro F1 | N1 F1 | kappa sd |
|---|---|---|---|---|---|---|
| A | 60 s context, 3 TCN blocks | 0.7297 | 0.6433 | 0.6713 | 0.324 | 0.089 |
| B | 120 s context, 3 TCN blocks | 0.7436 | 0.6625 | 0.6879 | 0.336 | 0.092 |
| C | 120 s context, 4 TCN blocks | 0.7405 | 0.6575 | 0.6851 | 0.348 | 0.085 |

Doubling the context window from 60 s to 120 s improved kappa by +0.0191.
Adding a fourth dilated block changed kappa by -0.0049, well within fold
variance, so the extra depth is not justified by these results.

## Cost of causality

Run C and run D are the same model: 37,093 parameters, identical layers, channels, data and
folds. The only difference is that D pads convolutions symmetrically and drops the attention
mask, so it can see future signal.

| Metric | Causal (C) | Non-causal (D) | Difference |
|---|---|---|---|
| Accuracy | 0.7405 | 0.7693 | -0.0287 |
| Kappa | 0.6575 | 0.6948 | -0.0373 |
| Macro F1 | 0.6851 | 0.7120 | -0.0269 |
| N1 F1 | 0.3480 | 0.3562 | -0.0081 |
| Accuracy at 30 s | 0.7577 | 0.7718 | -0.0141 |
| Kappa at 30 s | 0.6794 | 0.6980 | -0.0186 |

Fold by fold, kappa:

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.7056 | 0.7451 | -0.0395 |
| 1 | 0.7283 | 0.7622 | -0.0339 |
| 2 | 0.5424 | 0.6592 | -0.1168 |
| 3 | 0.5920 | 0.5436 | +0.0484 |
| 4 | 0.7193 | 0.7641 | -0.0448 |
| Mean | 0.6575 | 0.6948 | -0.0373 |

The causal model is worse in 4 of 5 folds and better in 1. A paired t-test over the five folds
gives t(4) = -1.42, p = 0.228, so at 20 subjects the penalty for causality cannot be
distinguished from zero. The point estimate is 0.037 kappa, falling to
0.019 when predictions are aggregated to 30 s epochs.
Resolving whether the penalty is real needs more subjects; Sleep-EDF-78 is the obvious step.

## Preprocessing ablation

An earlier configuration kept the full recordings and stacked a weighted sampler on top of the
focal loss. Untrimmed, Wake is 68% of the data and dominates accuracy; the doubled imbalance
correction also pushed N1 precision to 0.12-0.20, meaning the model labelled large amounts of
Wake and N2 as N1.

| Metric | Untrimmed, sampler on | Trimmed, sampler off | Change |
|---|---|---|---|
| Accuracy | 0.8064 | 0.7283 | -0.0781 |
| Kappa | 0.6584 | 0.6427 | -0.0157 |
| Macro F1 | 0.6322 | 0.6786 | +0.0463 |
| N1 F1 | 0.2481 | 0.3554 | +0.1073 |
| N2 F1 | 0.7045 | 0.7829 | +0.0784 |
| N3 F1 | 0.6890 | 0.7752 | +0.0863 |
| REM F1 | 0.5898 | 0.6784 | +0.0886 |
| Wake F1 | 0.9296 | 0.8010 | -0.1286 |

Every sleep stage improved. Accuracy and Wake F1 fell because the easy Wake majority that had
been carrying them was removed. Accuracy on untrimmed data should not be compared against
published Sleep-EDF results, which are all reported on trimmed recordings.

## Comparison with published baselines

| Model | Accuracy | Kappa | Params | Causal | Output rate |
|---|---|---|---|---|---|
| AttnSleep | 84.4% | 0.79 | ~48K | no | 30 s |
| DeepSleepNet | 82.2% | 0.754 | ~22M | no (Bi-LSTM) | 30 s |
| CareSleepNet | 78.1% | 0.694 | 22.7M | no (Transformer) | 30 s |
| This work (run B) | 74.4% | 0.662 | 30.7K | yes | 1 s |
| This work, 30 s aggregated | 76.1% | 0.684 | 30.7K | yes | 30 s |

All three baselines use future signal, through a bidirectional LSTM or attention across the full
sequence, so none can run in real time. The 30 s aggregated row is the closest like-for-like
comparison. N1 F1 of 0.336 is close to AttnSleep's 0.36 and above CareSleepNet's 0.32.

## Limitations

- Single dataset and single channel. Fold variance is large: kappa spans
  0.536 to 0.755 in run B.
- Supervision is 30 s labels replicated to 1 Hz, not genuine per-second scoring.
- The causality comparison is underpowered at five folds (p = 0.23).
- Validation peaks early; early stopping typically fires between epochs 10 and 20.

## Artifacts

- `results/run_a_baseline/` 60 s context
- `results/run_b_context/` 120 s context, best causal result
- `results/run_c_depth/` 120 s context with a 4th TCN block
- `results/run_d_noncausal/` causality ablation
- `results/trimmed/` and `results/baseline_untrimmed/` preprocessing ablation

Each directory holds its summary, per-fold reports, per-epoch curves and training log.

## Reproducing

```bash
python -m src.data.preprocessing --all
python -m src.train.train --config configs/run_b_context.yaml --fold -1
python -m src.eval.evaluate --config configs/run_b_context.yaml --fold 0   # repeat 0-4
python -m src.eval.evaluate --config configs/run_b_context.yaml --benchmark
```
