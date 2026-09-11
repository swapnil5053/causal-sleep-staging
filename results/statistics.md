# Statistical analysis

Everything here compares the causal model against the same architecture with the causal
constraint removed. Identical parameters, data, folds and seeds; the only difference is
access to future signal. Tests are paired, since the folds are matched.

Two preprocessing regimes appear throughout. The **epoch-normalised** pipeline is the one
behind paper v9. The **leak-free** pipeline replaces per-epoch z-scoring with a causal
trailing-window statistic and was re-run afterwards. Both are reported so the published
numbers stay verifiable alongside the corrected ones.

---

## 1. The causality ablation

| Setting | Causal κ | Non-causal κ | Difference | 95% CI | Causal loses |
|---|---|---|---|---|---|
| Sleep-EDF-20, epoch-normalised | 0.6625 | 0.6482 | **+0.0143** | [−0.0052, +0.0343] | 2/5 |
| Sleep-EDF-78, epoch-normalised | 0.6335 | 0.6570 | **−0.0235** | [−0.0322, −0.0129] | 5/5 |
| Sleep-EDF-78, leak-free, 3 seeds | 0.6649 | 0.6898 | **−0.0249** | [−0.0310, −0.0183] | 14/15 |

The bottom row is the figure reported in the paper.

### Sleep-EDF-20, epoch-normalised

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.6854 | 0.6904 | −0.0050 |
| 1 | 0.7343 | 0.6867 | +0.0476 |
| 2 | 0.6010 | 0.6122 | −0.0112 |
| 3 | 0.5363 | 0.5300 | +0.0063 |
| 4 | 0.7553 | 0.7215 | +0.0338 |
| **Mean** | **0.6625** | **0.6482** | **+0.0143** |

Paired t(4) = 1.26, p = 0.2762. Wilcoxon p = 0.4375. Cohen's d = 0.56.
Fold-to-fold κ s.d.: causal 0.0922, non-causal 0.0773.
Folds needed for 80% power at this effect size: 25.

### Sleep-EDF-78, epoch-normalised

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.6144 | 0.6363 | −0.0219 |
| 1 | 0.6366 | 0.6408 | −0.0042 |
| 2 | 0.6250 | 0.6549 | −0.0299 |
| 3 | 0.6830 | 0.7070 | −0.0240 |
| 4 | 0.6085 | 0.6462 | −0.0377 |
| **Mean** | **0.6335** | **0.6570** | **−0.0235** |

Paired t(4) = −4.24, p = 0.0133. Wilcoxon p = 0.0625. Cohen's d = −1.89.
Fold-to-fold κ s.d.: causal 0.0297, non-causal 0.0288.
Folds needed for 80% power: 3.

### Sleep-EDF-78, leak-free, seed 42

| Fold | Causal | Non-causal | Difference |
|---|---|---|---|
| 0 | 0.6237 | 0.6652 | −0.0415 |
| 1 | 0.6444 | 0.6641 | −0.0197 |
| 2 | 0.6713 | 0.7063 | −0.0350 |
| 3 | 0.7116 | 0.7444 | −0.0328 |
| 4 | 0.6661 | 0.6762 | −0.0101 |
| **Mean** | **0.6634** | **0.6912** | **−0.0278** |

Paired t(4) = −4.90, p = 0.0080. Bootstrap 95% CI [−0.0372, −0.0178]. Cohen's d = −2.19.

### Sleep-EDF-78, leak-free, pooled over three seeds

Fifteen paired measurements. Per-seed rows use the naive paired t-test within one seed;
the pooled figure uses the Nadeau–Bengio correction, because folds pooled across seeds
still share overlapping training subjects and the naive pooled test overstates
significance.

| Seed | Folds | Causal | Non-causal | Difference | Causal loses | t | p |
|---|---:|---|---|---|---|---|---|
| 42 | 5 | 0.6634 | 0.6912 | −0.0278 | 5/5 | −4.90 | 0.0080 |
| 43 | 5 | 0.6649 | 0.6893 | −0.0244 | 5/5 | −3.50 | 0.0248 |
| 44 | 5 | 0.6664 | 0.6888 | −0.0224 | 4/5 | −3.71 | 0.0207 |
| **Pooled** | 15 | **0.6649** | **0.6898** | **−0.0249** | **14/15** | **−3.36** | **0.0047** |

- **Nadeau–Bengio corrected: t(14) = −3.36, p = 0.004652**
- Naive pooled t-test, folds treated as fully independent, for comparison: t(14) = −7.33, p = 3.75 × 10⁻⁶
- Wilcoxon signed-rank: p = 0.0001
- Bootstrap 95% CI on the mean difference: [−0.0310, −0.0183]
- Cohen's d = −1.89
- Seed-to-seed spread of the effect: −0.0278 to −0.0224 (s.d. 0.0028)

The corrected pooled test is the conservative one and is what the paper reports. The
bootstrap interval and the per-seed rows are the more reliable summary; Cohen's d is
inflated at small paired sample sizes.

---

## 2. Where the cost falls

Causal-minus-non-causal F1 per sleep stage, pooled across 3 seeds × 5 folds on the
leak-free runs. Fifteen measurements per class. Negative means the causal model scores
lower. Holm–Bonferroni correction across the five class tests.

| Class | Causal F1 | Non-causal F1 | Difference | t | p | p (Holm) | Survives |
|---|---|---|---|---|---|---|---|
| **REM** | 0.6945 | 0.7542 | **−0.0598** | −5.10 | 0.0002 | **0.0008** | yes |
| N1 | 0.4212 | 0.4460 | −0.0248 | −2.57 | 0.0223 | 0.0892 | no |
| Wake | 0.8931 | 0.9112 | −0.0181 | −2.49 | 0.0262 | 0.0892 | no |
| N3 | 0.6654 | 0.6734 | −0.0080 | −0.54 | 0.5966 | 1.0000 | no |
| N2 | 0.7728 | 0.7752 | −0.0024 | −0.18 | 0.8561 | 1.0000 | no |

Macro-F1: causal 0.6894, non-causal 0.7120, difference −0.0226.

The cost concentrates in REM and N1. REM is normally disambiguated using eye-movement and
muscle-tone context around the epoch, and N1 is inherently transitional, so both lose more
when the model cannot look ahead. N2, N3 and Wake have locally distinctive signal:
spindles and K-complexes, slow waves, clear alpha and beta, and are barely affected. REM
is the only class whose loss survives correction, at 2.4× the macro average and 25× what
N2 pays.

Per-fold values are in `per_class_breakdown.csv`.

---

## 3. Why the pooled test needs correcting

`get_cv_splits()` originally took a single `seed` argument controlling both the subject
shuffle and, elsewhere, weight initialisation. Simulating 5-fold splits over 78 subjects
at seeds 42, 43 and 44 shows how much the *test set* of each fold index moves between
seeds. If seeds only changed initialisation, overlap would be 100%.

| Fold | 42–43 | 42–44 | 43–44 | Fold size |
|---|---|---|---|---:|
| 0 | 4/16 | 2/16 | 3/16 | 16 |
| 1 | 4/16 | 2/16 | 2/16 | 16 |
| 2 | 3/16 | 3/16 | 4/16 | 16 |
| 3 | 2/15 | 2/15 | 3/15 | 15 |
| 4 | 1/15 | 4/15 | 2/15 | 15 |

Average overlap across all folds and seed pairs: **2.7 subjects** out of an average fold
size of 15.6, about **18%**.

So "fold 0" under two different seeds is not the same held-out subjects, and per-seed fold
numbers are not directly comparable across seeds. The fifteen runs are neither fully
independent nor fully dependent, which is why the pooled figure uses the Nadeau–Bengio
correction rather than a naive paired test. `split_seed` is now separated from the
training seed in the config, so future repeated-seed runs share folds and vary only
initialisation.

---

## 4. Causal temporal smoothing

The model labels each second independently, so its raw output changes stage far more often
than a scored hypnogram. A trailing-window mode filter is applied to the prediction stream:
the label at second *t* is the most common prediction over [t−w+1, t]. It uses only past
predictions, so the real-time property holds, and it costs nothing to run.

Measured on the epoch-normalised pipeline, `configs/sleep78_causal.yaml`, 5 folds, applied
within each recording.

| Window (s) | Accuracy | κ | Macro-F1 | N1 F1 | Stage changes/hour |
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

Human scoring on the same recordings changes stage **13 times an hour**. Unsmoothed output
changes **181 times an hour**, 14 times more often.

A 30 s window is best for κ (+0.0089 against unsmoothed) and cuts fragmentation sevenfold,
from 181 to 26 changes an hour. A 45 s window brings stability to 19 an hour, close to the
human rate, at negligible cost.

N1 F1 peaks earlier, around a 10 s window, and falls as smoothing grows: N1 bouts are short
and heavy smoothing absorbs them into neighbouring stages. The best window therefore
depends on whether overall agreement or transitional-stage sensitivity matters more.

Smoothing is applied at inference only. No retraining, no added parameters, causality
preserved, a free post-processing gain available to any per-second model.

This sweep has not yet been re-run on the leak-free pipeline; it needs the saved
prediction files and is outstanding.

---

## Interpretation

On Sleep-EDF-78 the causal model is worse in 5 of 5 folds on the epoch-normalised pipeline
and 14 of 15 across three seeds on the leak-free one. Both confidence intervals exclude
zero, and the effect size agrees between two independently preprocessed pipelines
(−0.0235 and −0.0249 κ). That agreement is the strongest evidence that the measurement
reflects the causal constraint rather than an artifact of preprocessing.

On Sleep-EDF-20 the same comparison gives **+0.0143** with p = 0.2762 and an interval
spanning zero, so the sign flips. Fold-to-fold variance there is 3.1 times larger (κ s.d. 0.0922
against 0.0297), which is enough to hide an effect of this size and reverse its direction.
Roughly 25 folds would be needed for 80% power at that scale, against 3 on the 78-subject
set.

The practical conclusions: the cost of causal operation for this architecture is about
**0.025 κ**; it is concentrated almost entirely in **REM**; and causality penalties
reported on 20-subject splits are unreliable.
