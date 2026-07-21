# Causal temporal smoothing

The model labels each second independently, so its output changes stage far more often
than a scored hypnogram. A trailing-window mode filter is applied to the prediction
stream: the label at second t is the most common prediction over [t-w+1, t]. It uses only
past predictions, so the real-time property is preserved, and it costs nothing to run.

Config: `configs\sleep78_causal.yaml`, 5 folds, smoothing applied within each recording.

| Window (s) | Accuracy | Kappa | Macro F1 | N1 F1 | Stage changes/hour |
|---|---|---|---|---|---|
| 1 | 0.7227 | 0.6335 | 0.6616 | 0.398 | 181 |
| 5 | 0.7252 | 0.6367 | 0.6641 | 0.399 | 72 |
| 10 | 0.7277 | 0.6399 | 0.6667 | 0.401 | 56 |
| 15 | 0.7283 | 0.6405 | 0.6672 | 0.399 | 37 |
| 30 ** | 0.7300 | 0.6424 | 0.6687 | 0.397 | 26 |
| 45 | 0.7297 | 0.6418 | 0.6680 | 0.392 | 19 |
| 60 | 0.7292 | 0.6410 | 0.6672 | 0.387 | 16 |
| 90 | 0.7275 | 0.6384 | 0.6649 | 0.376 | 12 |
| 120 | 0.7258 | 0.6360 | 0.6627 | 0.369 | 10 |

Human scoring in the same recordings changes stage **13 times an hour**.

Unsmoothed output changes **181 times an hour**, 14x more often than a technician.

Best window is **30 s**: kappa 0.6424 against 0.6335 unsmoothed (+0.0089), with stage changes falling from 181 to 26 per hour.

Smoothing is applied at inference only. No retraining is involved and the model is
unchanged, so this is a free post-processing gain available to any per-second model.

