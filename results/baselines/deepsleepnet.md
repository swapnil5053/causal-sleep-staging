# DeepSleepNet, reproduced

CNN feature extractor with two filter scales feeding a bidirectional LSTM, about 22 million
parameters. Non-causal: the recurrent layer reads the whole sequence in both directions.

## Settings

| | |
|---|---|
| Data | Sleep-EDF, 39 recordings, Fpz-Cz at 100 Hz |
| Epoch | 30 s, 3000 samples per input |
| Classes | W, N1, N2, N3, REM |
| Validation | subject-wise 5-fold, no subject in both train and test |
| Optimiser | Adam, learning rate 1e-4 |
| Batch size | 16 |
| Epochs | 10, fine-tuning only |
| Class balance | oversampling of N1 |
| Sequence | 25 epochs |

The published two-step schedule pre-trains the convolutional extractors before end-to-end
fine-tuning. The pre-training phase was skipped here and only the fine-tuning phase was run,
which is a deviation from the original recipe and is the most likely reason for any gap against
the published figures. Training took about two hours.

## Results

| Fold | Accuracy | Macro F1 | Kappa |
|---|---|---|---|
| 0 | 0.8663 | 0.8306 | 0.8147 |
| 1 | 0.8470 | 0.7901 | 0.7932 |
| 2 | 0.7926 | 0.7470 | 0.7198 |
| 3 | 0.8116 | 0.6801 | 0.7318 |
| 4 | 0.7939 | 0.7167 | 0.7079 |
| **Mean** | **0.8223** | **0.7529** | **0.7535** |

Per-class figures below are from fold 0, the best fold, over its 14,000 test epochs. They are not
a mean across folds and should not be quoted as one.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| W | 0.9495 | 0.8917 | 0.9197 | 2382 |
| N1 | 0.5358 | 0.6377 | 0.5823 | 1256 |
| N2 | 0.8987 | 0.9117 | 0.9052 | 6182 |
| N3 | 0.9365 | 0.8237 | 0.8765 | 1809 |
| REM | 0.8633 | 0.8760 | 0.8696 | 2371 |

## Notes

Fold-to-fold macro F1 ranges from 0.6801 to 0.8306, a spread of 0.15, which is the usual
consequence of holding out a handful of subjects at a time on this corpus.

N1 is the weakest class at F1 0.58 even on the best fold, consistent with every model evaluated
here and with the inter-scorer disagreement reported for that stage.

The bidirectional recurrent layer processes a 25-epoch sequence step by step and cannot be
parallelised across time, which dominates training cost.
