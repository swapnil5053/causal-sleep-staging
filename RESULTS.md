# Results

Subject-wise 5-fold cross-validation on Sleep-EDF (PhysioNet sleep-cassette), single channel
Fpz-Cz at 100 Hz, five classes. Wake trimmed to 30 minutes either side of each night's sleep
period. Checkpoints selected on validation kappa, early stopping with patience 8.

Two dataset sizes are reported: Sleep-EDF-20 (20 subjects, 39 recordings) and Sleep-EDF-78
(78 subjects, 153 recordings, 1,629 hours).

## Headline

| Metric | Sleep-EDF-20 | Sleep-EDF-78 |
|---|---|---|
| Accuracy | 0.744 | 0.723 |
| Cohen's kappa | 0.662 | 0.634 |
| Macro F1 | 0.688 | 0.662 |
| N1 F1 | 0.336 | 0.398 |
| Kappa at 30 s | 0.684 | 0.652 |
| Fold-to-fold kappa sd | 0.092 | 0.030 |
| Parameters | 30,757 | 30,757 |
| CPU inference | 0.026 ms/s | 0.026 ms/s |

Kappa is lower on the larger set, which is expected: Sleep-EDF-78 spans ages 25 to 101 and is
a harder, more representative population. Published models show the same direction (AttnSleep
reports 84.4% on Sleep-EDF-20 and 81.6% on Sleep-EDF-78).

## Cost of causality

The causal and non-causal models are identical in every respect except that the non-causal
variant pads convolutions symmetrically and drops the attention mask, so it can see future
signal. Same parameters, same data, same folds.

| Subjects | Causal | Non-causal | Difference | t(4) | p | Folds causal loses |
|---|---|---|---|---|---|---|
| 20 | 0.6625 | 0.6482 | +0.0143 | 1.26 | 0.276 | 2/5 |
| 78 | 0.6335 | 0.6570 | -0.0235 | -4.24 | 0.013 | 5/5 |

Per-fold kappa difference (causal minus non-causal):

| Fold | 20 subjects | 78 subjects |
|---|---|---|
| 0 | -0.0050 | -0.0219 |
| 1 | +0.0476 | -0.0042 |
| 2 | -0.0112 | -0.0299 |
| 3 | +0.0063 | -0.0240 |
| 4 | +0.0338 | -0.0377 |

At 78 subjects the causal model is worse in all five folds and the difference is significant
(p = 0.013). Removing the causal constraint buys 0.024 kappa.

At 20 subjects the same comparison was not significant and the sign was unstable: this
configuration gave +0.0143 while a four-block variant gave -0.0373. Fold variance at 20 subjects (sd 0.092) is three times larger than at 78 (sd 0.030),
which is enough to swamp an effect this size. Studies measuring causality penalties on 20
subjects should be treated with caution.

## Per fold, Sleep-EDF-78 causal

| Fold | Accuracy | Kappa | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7094 | 0.6144 | 0.6386 | 0.913 | 0.419 | 0.692 | 0.504 | 0.665 |
| 1 | 0.7283 | 0.6366 | 0.6721 | 0.883 | 0.413 | 0.762 | 0.674 | 0.628 |
| 2 | 0.7132 | 0.6250 | 0.6579 | 0.882 | 0.385 | 0.710 | 0.622 | 0.691 |
| 3 | 0.7662 | 0.6830 | 0.6876 | 0.914 | 0.389 | 0.763 | 0.669 | 0.703 |
| 4 | 0.6961 | 0.6085 | 0.6520 | 0.887 | 0.384 | 0.664 | 0.605 | 0.719 |
| Mean | 0.7226 | 0.6335 | 0.6616 | 0.896 | 0.398 | 0.718 | 0.615 | 0.681 |

## Configuration ablation, Sleep-EDF-20

| Run | Configuration | Accuracy | Kappa | Macro F1 | N1 F1 |
|---|---|---|---|---|---|
| A | 60 s context, 3 TCN blocks | 0.7297 | 0.6433 | 0.6713 | 0.324 |
| B | 120 s context, 3 TCN blocks | 0.7436 | 0.6625 | 0.6879 | 0.336 |
| C | 120 s context, 4 TCN blocks | 0.7405 | 0.6575 | 0.6851 | 0.348 |

Doubling context from 60 s to 120 s improved kappa by +0.0191. A fourth
dilated block changed it by -0.0049, within fold variance, so three blocks are used.

## Preprocessing ablation

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

Untrimmed, Wake is 68% of the data and inflates accuracy. Stacking a weighted sampler on top
of focal loss also pushed N1 precision to 0.12-0.20. Every sleep stage improved after both were
corrected; accuracy fell because the easy Wake majority carrying it was removed.

## Comparison with published baselines

| Model | Dataset | Accuracy | Kappa | Params | Causal | Output rate |
|---|---|---|---|---|---|---|
| AttnSleep | Sleep-EDF-20 | 84.4% | 0.79 | ~48K | no | 30 s |
| DeepSleepNet | Sleep-EDF-20 | 82.2% | 0.754 | ~22M | no (Bi-LSTM) | 30 s |
| CareSleepNet | Sleep-EDF | 78.1% | 0.694 | 22.7M | no (Transformer) | 30 s |
| This work | Sleep-EDF-20 | 74.4% | 0.662 | 30.7K | yes | 1 s |
| This work | Sleep-EDF-78 | 72.3% | 0.634 | 30.7K | yes | 1 s |
| This work, non-causal | Sleep-EDF-78 | 74.0% | 0.657 | 30.7K | no | 1 s |

Every baseline uses future signal and cannot run in real time. The final row is our own model
with the causal constraint removed, which isolates how much of the gap to published work is
attributable to causality (0.024 kappa) rather than to model size or design.

N1 F1 on Sleep-EDF-78 is 0.398, above AttnSleep's 0.36 and CareSleepNet's 0.32.

## Latency

Intel Core i9-14900HX, 200 runs each.

| Configuration | ms per 1 s of EEG | Margin vs 3 ms/s target |
|---|---|---|
| 60 s context, 3 TCN blocks | 0.045 | 67x |
| 120 s context, 3 TCN blocks | 0.026 | 114x |
| 120 s context, 4 TCN blocks | 0.074 | 41x |

## Limitations

- Single channel (Fpz-Cz) and a single dataset family.
- Supervision is 30 s labels replicated to 1 Hz, not genuine per-second scoring.
- Five folds; the causality effect is significant at 78 subjects but rests on five paired values.

## Artifacts

- `results/sleep78_causal/`, `results/sleep78_noncausal/` main result and causality ablation
- `results/run_a_baseline/`, `run_b_context/`, `run_c_depth/` configuration ablation, 20 subjects
- `results/run_d_noncausal/`, `run_d2_noncausal/` causality ablation, 20 subjects
- `results/trimmed/`, `results/baseline_untrimmed/` preprocessing ablation

## Reproducing

```bash
python -m src.data.preprocessing --all --processed_dir data/processed78
python -m src.train.train --config configs/sleep78_causal.yaml --fold -1
python -m src.eval.evaluate --config configs/sleep78_causal.yaml --fold 0   # repeat 0-4
```
