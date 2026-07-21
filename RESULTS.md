# Results

Subject-wise 5-fold cross-validation on Sleep-EDF-20 (20 subjects, 39 recordings), single
channel Fpz-Cz at 100 Hz. Model has 30,757 parameters and emits one prediction per second.

## Summary

| Metric | Value |
|---|---|
| Accuracy | 72.8% |
| Cohen's kappa | 0.643 |
| Macro F1 | 0.679 |
| Parameters | 30,757 |
| CPU inference | 0.045 ms per second of EEG |

## Per fold

Test subjects are held out; the model never sees them during training or validation.

| Fold | Test subjects | Accuracy | Kappa | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 19, 05, 14, 04 | 0.7726 | 0.6933 | 0.7216 | 0.812 | 0.379 | 0.851 | 0.837 | 0.730 |
| 1 | 09, 13, 15, 18 | 0.7593 | 0.6875 | 0.7028 | 0.907 | 0.322 | 0.797 | 0.838 | 0.650 |
| 2 | 06, 12, 17, 10 | 0.6872 | 0.5854 | 0.6229 | 0.665 | 0.258 | 0.790 | 0.720 | 0.682 |
| 3 | 01, 11, 02, 16 | 0.6477 | 0.5366 | 0.6139 | 0.720 | 0.407 | 0.696 | 0.654 | 0.593 |
| 4 | 07, 08, 00, 03 | 0.7745 | 0.7106 | 0.7316 | 0.901 | 0.412 | 0.781 | 0.827 | 0.738 |
| Mean | | 0.7283 | 0.6427 | 0.6786 | 0.801 | 0.355 | 0.783 | 0.775 | 0.678 |

Per-class columns are F1 scores. Kappa ranges 0.537 to 0.711 across folds, standard
deviation 0.077.

## Preprocessing ablation

An earlier run kept the full recordings and used a weighted sampler on top of the focal
loss. Both choices turned out to be wrong.

Untrimmed, Wake is 68% of the data, so accuracy mostly measures whether the model can spot
wakefulness. Stacking a weighted sampler on top of focal-loss alpha weights also made the
model over-predict N1: precision fell to 0.12-0.20 while recall sat at 0.45-0.80, meaning it
labelled large amounts of Wake and N2 as N1. That also suppressed N2 recall to 0.42-0.70.

The final configuration trims wake to 30 minutes either side of each night's sleep period
and uses focal loss alone.

| Metric | Untrimmed, sampler on | Trimmed, sampler off | Change |
|---|---|---|---|
| Accuracy | 0.8064 | 0.7283 | -0.078 |
| Kappa | 0.6584 | 0.6427 | -0.016 |
| Macro F1 | 0.6322 | 0.6786 | +0.046 |
| N1 F1 | 0.2481 | 0.3554 | +0.107 |
| N2 F1 | 0.7045 | 0.7829 | +0.078 |
| N3 F1 | 0.6890 | 0.7752 | +0.086 |
| REM F1 | 0.5898 | 0.6784 | +0.089 |
| Wake F1 | 0.9296 | 0.8010 | -0.129 |

Every sleep stage improved. Accuracy and Wake F1 dropped because the easy Wake majority that
had been carrying them was removed. Kappa is unchanged within fold noise, and fold-to-fold
variance tightened (sd 0.092 to 0.077). N1 precision recovered to 0.31-0.35.

Accuracy on the untrimmed data should not be compared against published Sleep-EDF results,
since those are all reported on trimmed recordings.

## Comparison with published baselines

| Model | Accuracy | Kappa | Params | Causal | Output rate |
|---|---|---|---|---|---|
| AttnSleep | 84.4% | 0.79 | ~48K | no | 30 s |
| DeepSleepNet | 82.2% | 0.754 | ~22M | no (Bi-LSTM) | 30 s |
| CareSleepNet | 78.1% | 0.694 | 22.7M | no (Transformer) | 30 s |
| This work | 72.8% | 0.643 | 30.7K | yes | 1 s |

The gap is expected. All three baselines use future signal, either through a bidirectional
LSTM or attention across the full sequence, so none of them can run in real time. This model
cannot see past the current second by construction, emits a stage every second instead of
every 30, and uses roughly 700x fewer parameters than the Transformer baseline. The accuracy
difference is the cost of that constraint, which is the thing the experiment set out to
measure.

N1 F1 of 0.355 is close to AttnSleep's 0.36 and above CareSleepNet's 0.32, on the stage that
is hardest for both models and human scorers.

## Limitations

- Labels are 30-second epoch labels replicated across their 30 seconds. Inference is
  continuous, supervision is not.
- Validation macro F1 peaks between epochs 5 and 35 of 50, so the later epochs are wasted and
  the model is overfitting past that point.
- Fold variance is large (kappa 0.537 to 0.711) with only 20 subjects.
- Single channel, single dataset.

## Reproducing

```bash
python -m src.data.preprocessing --all
python -m src.train.train --fold -1
python -m src.eval.evaluate --fold 0     # repeat 0-4
python -m src.eval.evaluate --benchmark
```

Archived artifacts:

- `results/trimmed/` is the final run: summary, per-fold test reports, per-epoch curves,
  training log, and the five checkpoints with their subject splits.
- `results/baseline_untrimmed/` is the earlier run kept for the ablation above.
