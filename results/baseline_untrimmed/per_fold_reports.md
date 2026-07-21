# Baseline run: per-fold test reports (untrimmed, weighted sampler on)

Configuration: `sequence_length: 60`, `use_weighted_sampler: true`, focal loss (gamma 2.0),
no wake trimming (Wake = 68% of all data). 50 epochs, subject-wise 5-fold CV, Sleep-EDF-20.

> Recovered from the run console output. The original `logs/fold_*_test_report.txt` files from
> this run were overwritten by the later trimmed run; `summary.csv` and `train_log.txt` are the
> originals. Retained because the precision/recall split here is the evidence that motivated
> turning the weighted sampler off.

## Headline

| Fold | Test subjects | Best epoch | Accuracy | Kappa | Macro F1 |
|------|---------------|-----------|----------|-------|----------|
| 0 | 19, 05, 14, 04 | 35 | 0.8304 | 0.7026 | 0.6887 |
| 1 | 09, 13, 15, 18 | 23 | 0.8737 | 0.7649 | 0.6982 |
| 2 | 06, 12, 17, 10 | 26 | 0.7528 | 0.5797 | 0.5663 |
| 3 | 01, 11, 02, 16 | 5  | 0.7338 | 0.5457 | 0.5523 |
| 4 | 07, 08, 00, 03 | 28 | 0.8411 | 0.6991 | 0.6556 |
| Mean | | | 0.8064 | 0.6584 | 0.6322 |

CPU inference latency: 7.043 ms per 60 s sequence = 0.1174 ms/s (target ≤ 3 ms/s).

## The N1 precision problem

Across every fold N1 shows low precision with high recall. The model labels far too much
as N1. This is the signature of stacking two imbalance corrections (focal-loss alpha giving N1
a 26x weight over Wake, *and* a weighted sampler oversampling N1 windows).

| Fold | N1 precision | N1 recall | N1 F1 |
|------|--------------|-----------|-------|
| 0 | 0.18 | 0.68 | 0.29 |
| 1 | 0.20 | 0.62 | 0.30 |
| 2 | 0.14 | 0.45 | 0.21 |
| 3 | 0.12 | 0.80 | 0.21 |
| 4 | 0.14 | 0.70 | 0.24 |

It also suppresses N2 recall (0.42–0.70), because N2 epochs are being taken by N1 and N3.

## Fold 0: acc 0.8304, kappa 0.7026

```
              precision    recall  f1-score   support
    Wake (W)       0.99      0.88      0.93    432570
          N1       0.18      0.68      0.29     21210
          N2       0.95      0.70      0.80    120720
          N3       0.64      0.96      0.77     25860
         REM       0.61      0.70      0.65     53340
    accuracy                           0.83    653700

Confusion (rows=true, cols=predicted)
           W     N1     N2     N3    REM
W     381924  38469   1600   3024   7553
N1       658  14443   1358     84   4667
N2       118  13657  84374  11157  11414
N3         1    128    768  24910     53
REM     3128  12185    836     25  37166
```

## Fold 1: acc 0.8737, kappa 0.7649

```
              precision    recall  f1-score   support
    Wake (W)       1.00      0.96      0.98    374880
          N1       0.20      0.62      0.30     12750
          N2       0.92      0.61      0.73     91200
          N3       0.64      0.97      0.77     38580
         REM       0.72      0.70      0.71     42090
    accuracy                           0.87    559500

Confusion (rows=true, cols=predicted)
           W     N1     N2     N3    REM
W     358359  14065    156   1215   1085
N1       408   7914   1044     80   3304
N2       112   9226  55609  19106   7147
N3        83    298    557  37510    132
REM      711   8843   2761    354  29421
```

## Fold 2: acc 0.7528, kappa 0.5797

```
              precision    recall  f1-score   support
    Wake (W)       1.00      0.80      0.89    450960
          N1       0.14      0.45      0.21     16860
          N2       0.87      0.57      0.69    116010
          N3       0.38      0.99      0.55     26580
         REM       0.37      0.76      0.50     47370
    accuracy                           0.75    657780

Confusion (rows=true, cols=predicted)
           W     N1     N2     N3    REM
W     358808  35640   2042   7178  47292
N1        48   7637   3454    664   5057
N2        51   6717  66459  34304   8479
N3        27     20    111  26384     38
REM      111   5580   4394   1429  35856
```

## Fold 3: acc 0.7338, kappa 0.5457

```
              precision    recall  f1-score   support
    Wake (W)       0.99      0.82      0.90    450510
          N1       0.12      0.80      0.21     18210
          N2       0.95      0.42      0.58    119220
          N3       0.40      0.92      0.55     28440
         REM       0.51      0.53      0.52     43560
    accuracy                           0.73    659940

Confusion (rows=true, cols=predicted)
           W     N1     N2     N3    REM
W     370450  56887    335  11362  11476
N1        76  14611    521    927   2075
N2      1752  31862  49736  27281   8589
N3        94   1467    530  26167    182
REM      305  18173   1328    483  23271
```

## Fold 4: acc 0.8411, kappa 0.6991

```
              precision    recall  f1-score   support
    Wake (W)       1.00      0.91      0.95    462510
          N1       0.14      0.70      0.24     15090
          N2       0.94      0.58      0.71     86820
          N3       0.69      0.97      0.80     51630
         REM       0.61      0.53      0.57     45150
    accuracy                           0.84    661200

Confusion (rows=true, cols=predicted)
           W     N1     N2     N3    REM
W     421483  31306    173   1156   8392
N1        55  10509    777    312   3437
N2        19  12377  49932  21244   3248
N3        13    515    927  50141     34
REM      174  19304   1491    103  24078
```
