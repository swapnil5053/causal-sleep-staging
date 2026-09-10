# Causal temporal smoothing

The model labels each second independently, so its output changes stage far more often
than a scored hypnogram. A trailing-window mode filter is applied to the prediction
stream: the label at second t is the most common prediction over [t-w+1, t]. It uses only
past predictions, so the real-time property is preserved, and it costs nothing to run.

Source: archived predictions: `logs_dodh_causal_s42`, 5 folds, smoothing applied within each recording.

| Window (s) | Accuracy | Kappa | Macro F1 | N1 F1 | Stage changes/hour |
|---|---|---|---|---|---|
| 1 | 0.7517 | 0.6567 | 0.6919 | 0.365 | 205 |
| 5 | 0.7543 | 0.6600 | 0.6941 | 0.364 | 83 |
| 10 ** | 0.7566 | 0.6629 | 0.6957 | 0.363 | 64 |
| 15 | 0.7563 | 0.6622 | 0.6944 | 0.357 | 44 |
| 30 | 0.7555 | 0.6607 | 0.6903 | 0.340 | 31 |
| 45 | 0.7519 | 0.6552 | 0.6831 | 0.321 | 22 |
| 60 | 0.7486 | 0.6501 | 0.6776 | 0.309 | 19 |
| 90 | 0.7423 | 0.6405 | 0.6658 | 0.280 | 14 |
| 120 | 0.7363 | 0.6313 | 0.6547 | 0.253 | 12 |

Human scoring in the same recordings changes stage **13 times an hour**.

Unsmoothed output changes **205 times an hour**, 16x more often than a technician.

Best window is **10 s**: kappa 0.6629 against 0.6567 unsmoothed (+0.0062), with stage changes falling from 205 to 64 per hour.

Smoothing is applied at inference only. No retraining is involved and the model is
unchanged, so this is a free post-processing gain available to any per-second model.

