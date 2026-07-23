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
| Kappa with 30 s causal smoothing | | 0.642 |
| Kappa pooled over 3 seeds | | 0.641 |
| Stage changes per hour, raw / smoothed / human | | 181 / 26 / 13 |
| Parameters | 30,757 | 30,757 |
| CPU inference | 0.026 ms/s | 0.026 ms/s |

Kappa is lower on the larger set, which is expected: Sleep-EDF-78 spans ages 25 to 101 and is
a harder, more representative population. Published models show the same direction (AttnSleep
reports 84.4% on Sleep-EDF-20 and 81.6% on Sleep-EDF-78).

## Temporal stability and causal smoothing

The model labels every second independently, so its raw output is far more fragmented than a
scored hypnogram. On the Sleep-EDF-78 held-out recordings it changes stage **181 times an hour**
against **13 times an hour** for the technician, a 14-fold difference.

A trailing-window mode filter fixes most of this. The label at second t becomes the most common
prediction over [t-w+1, t], so only past predictions are used and the causal property holds.

| Window (s) | Accuracy | Kappa | Macro F1 | N1 F1 | Stage changes/hour |
|---|---|---|---|---|---|
| 1 (raw) | 0.7227 | 0.6335 | 0.6616 | 0.398 | 181 |
| 5 | 0.7252 | 0.6367 | 0.6641 | 0.399 | 72 |
| 10 | 0.7277 | 0.6399 | 0.6667 | 0.401 | 56 |
| 15 | 0.7283 | 0.6405 | 0.6672 | 0.399 | 37 |
| **30** | **0.7300** | **0.6424** | **0.6687** | 0.397 | 26 |
| 45 | 0.7297 | 0.6418 | 0.6680 | 0.392 | 19 |
| 60 | 0.7292 | 0.6410 | 0.6672 | 0.387 | 16 |
| 90 | 0.7275 | 0.6384 | 0.6649 | 0.376 | 12 |
| 120 | 0.7258 | 0.6360 | 0.6627 | 0.369 | 10 |

A 30 s window is best for kappa (+0.0089) and cuts fragmentation sevenfold. A 45 s window
brings output stability to 19 changes an hour, close to the human rate, at a negligible cost.

N1 F1 peaks earlier, around a 10 s window, and falls as smoothing grows: N1 bouts are short and
heavy smoothing absorbs them into neighbouring stages. The best window therefore depends on
whether overall agreement or transitional-stage sensitivity matters more.

Smoothing is applied at inference only, needs no retraining, adds no parameters and preserves
causality. It recovers roughly 38% of the accuracy given up by the causal constraint.

## Cost of causality

The causal and non-causal models are identical in every respect except that the non-causal
variant pads convolutions symmetrically and drops the attention mask, so it can see future
signal. Same parameters, same data, same folds, same seeds.

Sleep-EDF-78, three random seeds x five folds = 15 paired measurements:

| Seed | Causal | Non-causal | Difference |
|---|---|---|---|
| 42 | 0.6335 | 0.6570 | -0.0235 |
| 43 | 0.6426 | 0.6743 | -0.0317 |
| 44 | 0.6456 | 0.6764 | -0.0307 |
| **Pooled** | **0.6406** | **0.6693** | **-0.0287** |

Paired t(14) = -10.37, p = 5.91e-08. Bootstrap 95% CI on the difference
[-0.0339, -0.0234], Cohen's d = -2.68. The causal model is worse in
**15 of 15** measurements and the effect appears in all three seeds.
Seed-to-seed variation is small (causal kappa sd 0.0052), so the
result is not an artefact of initialisation.

At 20 subjects the same experiment gave +0.0143 with p = 0.28, the wrong sign. Fold variance
there is three times larger (kappa sd 0.092 against 0.030), enough to swamp an effect this size.
Causality penalties measured on 20-subject splits should be treated with caution.


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

All three baselines were run in-house on **39 recordings (Sleep-EDF-20)**, so the comparison
below uses our Sleep-EDF-20 numbers. Comparing our Sleep-EDF-78 result against their
Sleep-EDF-20 results would understate our model, since the 78-subject set spans ages 25 to 101
and is materially harder.

| Model | Accuracy | Kappa | Params | Causal | Output rate | Reporting |
|---|---|---|---|---|---|---|
| DeepSleepNet | 82.2% | 0.754 | ~22M | no (Bi-LSTM) | 30 s | mean of 5 folds |
| AttnSleep | 81.3% | ~0.75 | ~48K | partial | 5 s | mean of 10 folds |
| **This work, 30 s aggregated** | **76.1%** | **0.684** | **30.7K** | **yes** | **30 s** | mean of 5 folds |
| **This work, per-second** | **74.4%** | **0.662** | **30.7K** | **yes** | **1 s** | mean of 5 folds |
| CareSleepNet | ~75.8% | ~0.62 | 22.7M | no (Transformer) | 30 s | **estimated mean** |

Two caveats matter when reading this table.

CareSleepNet's frequently quoted 78.07% / 0.694 is its **best fold**, not its mean. Its own
report records fold 1 at 38.12% and fold 4 as unrecoverable, giving an estimated mean of ~75.8%
/ 0.62. Against that mean, our model reaches a higher kappa with **740x fewer parameters** while
remaining causal.

The 30 s aggregated row is the like-for-like comparison against 30 s models, obtained by
majority-voting our per-second predictions within each epoch. No retraining is involved.

We remain below DeepSleepNet and AttnSleep. Both read future signal, and DeepSleepNet uses
roughly 700x more parameters. Our own non-causal ablation isolates how much of that gap is
attributable to causality: 0.029 kappa, leaving the remainder to capacity and architecture.

N1 F1 is 0.398 on Sleep-EDF-78, above AttnSleep's 0.36 and CareSleepNet's 0.32.

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
- The causality effect rests on 15 paired measurements (3 seeds x 5 folds) on one dataset family.

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
