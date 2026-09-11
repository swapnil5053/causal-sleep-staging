# Prediction analysis

5 prediction file(s), 78 held-out subject records, 5,861,040 scored seconds.

Read from archived predictions; no checkpoint was loaded and nothing was retrained.

## Pooled

| Metric | Value |
|---|---|
| Accuracy | 0.7487 |
| Cohen's kappa | 0.6638 |
| Macro-F1 | 0.6920 |

| W | N1 | N2 | N3 | REM |
|---|---|---|---|---|
| 0.8889 | 0.4163 | 0.7762 | 0.6766 | 0.7018 |

## Subject-level bootstrap

Whole subjects resampled with replacement, 2,000 draws. Seconds inside a subject are autocorrelated, so the subject is the independent unit rather than the second.

| Metric | Estimate | 95% CI |
|---|---|---|
| Accuracy | 0.7487 | [0.7309, 0.7661] |
| Cohen's kappa | 0.6638 | [0.6415, 0.6858] |
| Macro-F1 | 0.6920 | [0.6730, 0.7102] |

## Per subject

| Subject | Seconds | Accuracy | Kappa | Macro-F1 |
|---|---:|---|---|---|
| 08 (logs_78streaming_causal_s42/fold_0) | 65,640 | 0.8156 | 0.7604 | 0.7640 |
| 15 (logs_78streaming_causal_s42/fold_0) | 81,360 | 0.8658 | 0.8199 | 0.7772 |
| 23 (logs_78streaming_causal_s42/fold_0) | 78,960 | 0.6906 | 0.5767 | 0.5244 |
| 33 (logs_78streaming_causal_s42/fold_0) | 96,000 | 0.8246 | 0.7399 | 0.6127 |
| 41 (logs_78streaming_causal_s42/fold_0) | 60,000 | 0.6698 | 0.5724 | 0.6611 |
| 48 (logs_78streaming_causal_s42/fold_0) | 118,080 | 0.8041 | 0.7096 | 0.7018 |
| 49 (logs_78streaming_causal_s42/fold_0) | 64,200 | 0.7452 | 0.6598 | 0.6715 |
| 51 (logs_78streaming_causal_s42/fold_0) | 61,200 | 0.6130 | 0.4750 | 0.5007 |
| 52 (logs_78streaming_causal_s42/fold_0) | 29,880 | 0.7442 | 0.6262 | 0.6034 |
| 53 (logs_78streaming_causal_s42/fold_0) | 64,560 | 0.6964 | 0.5851 | 0.6214 |
| 54 (logs_78streaming_causal_s42/fold_0) | 85,920 | 0.7858 | 0.6957 | 0.6324 |
| 57 (logs_78streaming_causal_s42/fold_0) | 69,600 | 0.6726 | 0.5601 | 0.5886 |
| 62 (logs_78streaming_causal_s42/fold_0) | 98,040 | 0.6698 | 0.5459 | 0.5232 |
| 70 (logs_78streaming_causal_s42/fold_0) | 96,960 | 0.5746 | 0.4365 | 0.4781 |
| 71 (logs_78streaming_causal_s42/fold_0) | 79,560 | 0.7113 | 0.6015 | 0.5349 |
| 77 (logs_78streaming_causal_s42/fold_0) | 79,440 | 0.5814 | 0.4368 | 0.5372 |
| 07 (logs_78streaming_causal_s42/fold_1) | 67,440 | 0.8487 | 0.8022 | 0.8011 |
| 18 (logs_78streaming_causal_s42/fold_1) | 56,520 | 0.7691 | 0.6972 | 0.7010 |
| 19 (logs_78streaming_causal_s42/fold_1) | 84,240 | 0.7790 | 0.6958 | 0.7026 |
| 20 (logs_78streaming_causal_s42/fold_1) | 61,200 | 0.7342 | 0.6223 | 0.5450 |
| 25 (logs_78streaming_causal_s42/fold_1) | 59,760 | 0.7832 | 0.6859 | 0.7059 |
| 30 (logs_78streaming_causal_s42/fold_1) | 53,400 | 0.7411 | 0.6281 | 0.6339 |
| 36 (logs_78streaming_causal_s42/fold_1) | 24,720 | 0.7435 | 0.5987 | 0.5949 |
| 43 (logs_78streaming_causal_s42/fold_1) | 49,800 | 0.7634 | 0.6795 | 0.7209 |
| 44 (logs_78streaming_causal_s42/fold_1) | 68,520 | 0.7273 | 0.6293 | 0.6979 |
| 58 (logs_78streaming_causal_s42/fold_1) | 68,040 | 0.6793 | 0.5505 | 0.5545 |
| 59 (logs_78streaming_causal_s42/fold_1) | 92,040 | 0.6309 | 0.5192 | 0.6018 |
| 61 (logs_78streaming_causal_s42/fold_1) | 81,360 | 0.5338 | 0.4109 | 0.4583 |
| 63 (logs_78streaming_causal_s42/fold_1) | 65,040 | 0.6732 | 0.5545 | 0.6015 |
| 64 (logs_78streaming_causal_s42/fold_1) | 99,600 | 0.8339 | 0.7380 | 0.5958 |
| 66 (logs_78streaming_causal_s42/fold_1) | 120,600 | 0.7626 | 0.6288 | 0.6131 |
| 67 (logs_78streaming_causal_s42/fold_1) | 89,640 | 0.7569 | 0.6558 | 0.6392 |
| 02 (logs_78streaming_causal_s42/fold_2) | 60,960 | 0.7159 | 0.6144 | 0.6869 |
| 05 (logs_78streaming_causal_s42/fold_2) | 57,480 | 0.8306 | 0.7697 | 0.7781 |
| 06 (logs_78streaming_causal_s42/fold_2) | 55,680 | 0.8585 | 0.8073 | 0.8149 |
| 09 (logs_78streaming_causal_s42/fold_2) | 67,080 | 0.8031 | 0.7196 | 0.7471 |
| 10 (logs_78streaming_causal_s42/fold_2) | 65,880 | 0.7565 | 0.6330 | 0.6139 |
| 16 (logs_78streaming_causal_s42/fold_2) | 64,320 | 0.8169 | 0.7522 | 0.7424 |
| 21 (logs_78streaming_causal_s42/fold_2) | 71,520 | 0.7934 | 0.7122 | 0.6879 |
| 22 (logs_78streaming_causal_s42/fold_2) | 66,120 | 0.7119 | 0.6140 | 0.6958 |
| 24 (logs_78streaming_causal_s42/fold_2) | 103,440 | 0.8742 | 0.8004 | 0.6895 |
| 40 (logs_78streaming_causal_s42/fold_2) | 64,080 | 0.6139 | 0.5074 | 0.6007 |
| 47 (logs_78streaming_causal_s42/fold_2) | 100,440 | 0.7070 | 0.5895 | 0.6406 |
| 50 (logs_78streaming_causal_s42/fold_2) | 72,840 | 0.7484 | 0.6573 | 0.6200 |
| 56 (logs_78streaming_causal_s42/fold_2) | 71,520 | 0.5980 | 0.4790 | 0.5771 |
| 60 (logs_78streaming_causal_s42/fold_2) | 101,760 | 0.7502 | 0.6464 | 0.6489 |
| 65 (logs_78streaming_causal_s42/fold_2) | 137,160 | 0.7428 | 0.6285 | 0.6193 |
| 74 (logs_78streaming_causal_s42/fold_2) | 98,160 | 0.7378 | 0.6255 | 0.5297 |
| 00 (logs_78streaming_causal_s42/fold_3) | 59,040 | 0.8390 | 0.7878 | 0.7683 |
| 01 (logs_78streaming_causal_s42/fold_3) | 68,640 | 0.8099 | 0.7238 | 0.7482 |
| 12 (logs_78streaming_causal_s42/fold_3) | 60,840 | 0.7576 | 0.6756 | 0.7060 |
| 26 (logs_78streaming_causal_s42/fold_3) | 77,280 | 0.7911 | 0.7227 | 0.7562 |
| 32 (logs_78streaming_causal_s42/fold_3) | 77,400 | 0.7763 | 0.6885 | 0.5808 |
| 34 (logs_78streaming_causal_s42/fold_3) | 92,400 | 0.8196 | 0.7345 | 0.5872 |
| 37 (logs_78streaming_causal_s42/fold_3) | 72,720 | 0.6822 | 0.5604 | 0.5631 |
| 38 (logs_78streaming_causal_s42/fold_3) | 109,320 | 0.8330 | 0.7460 | 0.6580 |
| 42 (logs_78streaming_causal_s42/fold_3) | 50,040 | 0.8371 | 0.7862 | 0.8151 |
| 45 (logs_78streaming_causal_s42/fold_3) | 71,160 | 0.7939 | 0.7394 | 0.7807 |
| 46 (logs_78streaming_causal_s42/fold_3) | 60,120 | 0.7510 | 0.6568 | 0.6646 |
| 75 (logs_78streaming_causal_s42/fold_3) | 92,760 | 0.8619 | 0.7607 | 0.5789 |
| 76 (logs_78streaming_causal_s42/fold_3) | 130,320 | 0.7313 | 0.4619 | 0.4427 |
| 80 (logs_78streaming_causal_s42/fold_3) | 74,040 | 0.6805 | 0.5530 | 0.5592 |
| 82 (logs_78streaming_causal_s42/fold_3) | 92,040 | 0.8576 | 0.7991 | 0.7693 |
| 03 (logs_78streaming_causal_s42/fold_4) | 55,800 | 0.7893 | 0.7086 | 0.7213 |
| 04 (logs_78streaming_causal_s42/fold_4) | 72,960 | 0.7707 | 0.6831 | 0.7287 |
| 11 (logs_78streaming_causal_s42/fold_4) | 51,840 | 0.6068 | 0.4697 | 0.5777 |
| 13 (logs_78streaming_causal_s42/fold_4) | 30,840 | 0.8252 | 0.7610 | 0.7786 |
| 14 (logs_78streaming_causal_s42/fold_4) | 58,680 | 0.8482 | 0.7898 | 0.7481 |
| 17 (logs_78streaming_causal_s42/fold_4) | 83,160 | 0.6125 | 0.5026 | 0.5383 |
| 27 (logs_78streaming_causal_s42/fold_4) | 64,200 | 0.7313 | 0.6362 | 0.6748 |
| 28 (logs_78streaming_causal_s42/fold_4) | 65,880 | 0.7569 | 0.6829 | 0.7366 |
| 29 (logs_78streaming_causal_s42/fold_4) | 82,080 | 0.8227 | 0.7606 | 0.7559 |
| 31 (logs_78streaming_causal_s42/fold_4) | 66,960 | 0.7465 | 0.6732 | 0.7093 |
| 35 (logs_78streaming_causal_s42/fold_4) | 58,080 | 0.6207 | 0.4920 | 0.5593 |
| 55 (logs_78streaming_causal_s42/fold_4) | 64,080 | 0.7713 | 0.6845 | 0.6604 |
| 72 (logs_78streaming_causal_s42/fold_4) | 64,800 | 0.6654 | 0.5330 | 0.5182 |
| 73 (logs_78streaming_causal_s42/fold_4) | 149,520 | 0.8079 | 0.6505 | 0.5047 |
| 81 (logs_78streaming_causal_s42/fold_4) | 74,280 | 0.7999 | 0.7337 | 0.7556 |

## Class-prior correction

Logits shifted by `log(target prior) - log(source prior)`, applied at inference only. No retraining, no added parameters. The source prior is a uniform prior, assumed because the prediction files carry no training class counts. That assumption holds for the class-weighted focal loss this repository trains with and not for an unweighted one; re-run evaluation against a checkpoint written by the current trainer, or pass --source-prior, to remove the guess.

| Metric | Before | After | Change |
|---|---|---|---|
| Accuracy | 0.7487 | 0.7452 | -0.0035 |
| Cohen's kappa | 0.6638 | 0.6265 | -0.0373 |
| Macro-F1 | 0.6920 | 0.5237 | -0.1682 |

| Stage | F1 before | F1 after | Change |
|---|---|---|---|
| W | 0.8889 | 0.8932 | +0.0044 |
| N1 | 0.4163 | 0.1615 | -0.2549 |
| N2 | 0.7762 | 0.7857 | +0.0095 |
| N3 | 0.6766 | 0.1132 | -0.5634 |
| REM | 0.7018 | 0.6651 | -0.0367 |

A gain here is a free post-processing improvement of the same kind as the causal mode filter. A loss is also informative: it says the reweighted loss was not the source of the imbalance.

One caveat before quoting this. The target prior used here is the class distribution of the held-out data itself, so the figure is a best case: it assumes the deployment prior is known. The honest protocol estimates the prior from the training folds, or from population statistics, and applies that instead.

