# Causal temporal smoothing

The model labels each second independently, so its output changes stage far more often
than a scored hypnogram. A trailing-window mode filter is applied to the prediction
stream: the label at second t is the most common prediction over [t-w+1, t]. It uses only
past predictions, so the real-time property is preserved, and it costs nothing to run.

Source: archived predictions: `logs_78streaming_causal_s42`, 5 folds, smoothing applied within each recording.

| Window (s) | Accuracy | Kappa | Macro F1 | N1 F1 | Stage changes/hour |
|---|---|---|---|---|---|
| 1 | 0.7487 | 0.6634 | 0.6903 | 0.414 | 167 |
| 5 | 0.7514 | 0.6668 | 0.6931 | 0.416 | 68 |
| 10 | 0.7541 | 0.6702 | 0.6960 | 0.418 | 53 |
| 15 | 0.7548 | 0.6711 | 0.6969 | 0.417 | 35 |
| 30 ** | 0.7568 | 0.6736 | 0.6993 | 0.417 | 25 |
| 45 | 0.7567 | 0.6732 | 0.6993 | 0.414 | 18 |
| 60 | 0.7568 | 0.6732 | 0.6995 | 0.411 | 15 |
| 90 | 0.7549 | 0.6703 | 0.6970 | 0.401 | 11 |
| 120 | 0.7522 | 0.6663 | 0.6937 | 0.393 | 9 |

Human scoring in the same recordings changes stage **13 times an hour**.

Unsmoothed output changes **167 times an hour**, 13x more often than a technician.

Best window is **30 s**: kappa 0.6736 against 0.6634 unsmoothed (+0.0102), with stage changes falling from 167 to 25 per hour.

Smoothing is applied at inference only. No retraining is involved and the model is
unchanged, so this is a free post-processing gain available to any per-second model.

