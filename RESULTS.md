# Results

Every number here comes from a committed per-fold CSV under `results/` and can be regenerated
without retraining by `runs/reproduce.ps1`. Nothing is typed by hand.

Two datasets, two architectures, two evaluation protocols. Three seeds by five subject-wise folds
gives fifteen paired measurements per comparison. Pooled tests use the Nadeau-Bengio correction
for repeated cross-validation, which inflates the variance estimate by `1/n + 1/(k-1)`, here
`1/15 + 1/4`. Bootstrap intervals resample whole subjects. Within a seed the two arms use
byte-identical subject lists.

## 1. The headline

Batch-tiled evaluation systematically penalises causal models. The penalty is large, it is
measurable within a single arm, and it accounts for most of what a tiled comparison reports as
the cost of causality.

| | Sleep-EDF-78, TCN | DOD-H, TCN | Sleep-EDF-78, GRU |
|---|---|---|---|
| Causal arm gains, tiled to streaming | +0.0350 | +0.0345 | +0.0353 |
| Non-causal arm gains | +0.0036 | +0.0052 | +0.0055 |
| **Excess to the causal arm** | **+0.0314** | **+0.0292** | **+0.0298** |
| Folds with positive excess | 15/15 | 15/15 | 15/15 |
| Smallest single fold | +0.0221 | +0.0173 | +0.0152 |
| Bootstrap 95% CI | [.0291, .0334] | [.0253, .0332] | [.0265, .0330] |
| Corrected t(14) | 12.69 | 6.48 | 7.99 |
| p | 4.6e-09 | 1.4e-05 | 1.4e-06 |
| Per-seed range | .0310 to .0319 | .0285 to .0297 | .0260 to .0338 |

Artifacts: `results/protocol_excess_sleep78.md`, `results/protocol_excess_dodh.md`,
`results/protocol_excess_gru_pooled.md`.

The excess is a within-arm quantity: the same trained weights, the same held-out seconds, scored
under two protocols. It does not require the two arms to be comparable to each other, which is
what makes it measurable on corpora where the between-arm comparison is not.

Two datasets differing in montage, sampling rate, scoring team, class balance, subject count and
recording length, and two architecture families sharing only the convolutional front end, agree
on the penalty to the causal arm to within 0.001 kappa.

## 2. The two protocols

**Tiled.** The recording is covered by non-overlapping context windows and every position of
every window contributes one scored prediction. Each window begins from an empty history. This is
the default in every implementation we have examined, including our own earlier work.

**Streaming.** The model recomputes every 30 s against a full buffer and only the freshest 30
predictions are kept, so after the opening window every scored second carries at least 90 seconds
of genuine history. This is what a deployed system does.

Every held-out second is scored exactly once under each protocol, from the same trained weights.
`--stream_stride 30` on `src/eval/evaluate.py` selects the second; results are written with a
`_streaming30` suffix.

## 3. The cost of causality, under each protocol

| Dataset, architecture | Protocol | Causal | Non-causal | Difference | Causal loses | Corrected t(14) | p |
|---|---|---|---|---|---|---|---|
| Sleep-EDF-78, TCN | Tiled | 0.6649 | 0.6898 | -0.0249 | 14/15 | -7.33 | 3.8e-06 |
| Sleep-EDF-78, TCN | Streaming | 0.6998 | 0.6934 | +0.0065 | 7/15 | | 0.47 |
| DOD-H, TCN | Tiled | 0.6329 | 0.6659 | -0.0329 | 10/15 | -1.27 | 0.22 |
| DOD-H, TCN | Streaming | 0.6674 | 0.6711 | -0.0037 | 6/15 | | 0.89 |
| Sleep-EDF-78, GRU | Tiled | 0.6638 | 0.7022 | -0.0384 | 15/15 | -7.60 | 2.5e-06 |
| Sleep-EDF-78, GRU | Streaming | 0.6992 | 0.7077 | -0.0085 | 11/15 | -1.57 | 0.14 |

Artifacts: `results/statistics_streaming_pooled.md`, `results/statistics_dodh.md`,
`results/statistics_gru_tiled_pooled.md`, and the per-protocol tables inside the three
`protocol_excess_*.md` files.

Under tiling the causal constraint appears to cost between 0.025 and 0.038 kappa depending on
dataset and architecture. Under the deployment protocol none of the three differences is
resolvable. The protocol accounts for essentially all of the measured penalty on Sleep-EDF-78 with
the convolutional model, about 89% of it on DOD-H, and about 78% of it with the recurrent model.

This table is the weaker of the two measurements in this document, because unlike the excess it
requires the two arms to be comparable. Read it second.

## 4. The mechanism, verified arithmetically

One forward pass over a window yields a prediction at every position, and position *p* in a
buffer of length *L*<sub>buf</sub> carries exactly *L*<sub>buf</sub> − 1 − *p* seconds of
lookahead and the rest as history. A single sweep therefore measures accuracy at every permitted
latency without retraining. `scripts/latency_sweep.py` produces it.

Averaging that curve over all 120 buffer positions reproduces the model's tiled score:

| | Measured tiled | Mean of the curve |
|---|---|---|
| Sleep-EDF-78, causal | 0.6634 | 0.6626 |
| Sleep-EDF-78, non-causal | 0.6912 | 0.6890 |
| DOD-H, causal | 0.6567 | 0.6537 |
| DOD-H, non-causal | 0.6922 | 0.6915 |

Within 0.003 kappa in all four combinations. The tiled protocol is, arithmetically, this average.

Where the penalty lives, by buffer quartile, seed 42, 120 s buffer. Positive difference means the
causal arm pays.

| History | Sleep-EDF-78 causal | non-causal | Δ | DOD-H causal | non-causal | Δ |
|---|---|---|---|---|---|---|
| 91 to 120 s | 0.6982 | 0.6930 | -0.0052 | 0.6904 | 0.6961 | +0.0057 |
| 61 to 90 s | 0.6923 | 0.6969 | +0.0046 | 0.6887 | 0.7044 | +0.0157 |
| 31 to 60 s | 0.6753 | 0.6911 | +0.0158 | 0.6737 | 0.6970 | +0.0233 |
| 1 to 30 s | 0.5847 | 0.6752 | +0.0905 | 0.5622 | 0.6685 | +0.1063 |

At positions carrying a nearly full buffer the arms are level. Across the last quartile the causal
arm loses 0.114 and 0.128 kappa against its own best quartile while the non-causal arm loses 0.018
and 0.028. That gap is the artifact. A streaming system occupies those positions once, at
start-up, and a tiled evaluation returns to them at the start of every window.

Artifacts: `results/latency_causal_s5.{csv,md}`, `results/latency_noncausal_s5.{csv,md}`,
`results/latency_dodh_causal.{csv,md}`, `results/latency_dodh_noncausal.{csv,md}`. Use the
stride-5 files for Sleep-EDF-78; the unsuffixed files are stride 30 and mixing strides across
datasets is not a fair comparison.

### Accuracy against permitted latency

The causal arm declines monotonically from zero latency: it cannot read forward, so a larger
permitted delay only costs it history, and at 119 s it collapses to kappa 0.317 and 0.252. The
non-causal arm traces an inverted U, peaking at 38 s of delay on Sleep-EDF-78 and 46 s on DOD-H.

At each arm's best operating point the two datasets disagree, and that is reported here because it
bounds the claim. On Sleep-EDF-78 the causal arm reaches 0.7008 at zero latency against a
non-causal optimum of 0.6977 at 38 s, a margin of 0.003 kappa that is not worth defending as more
than a tie. On DOD-H the causal arm reaches 0.6922 at zero latency and the non-causal optimum is
0.7058 at 46 s, exceeding it by 0.0136. What replicates is the protocol asymmetry. The stronger
claim, that a non-causal model cannot beat a causal one at any latency, holds on one dataset and
fails on the other, where roughly 0.014 kappa is available in exchange for 46 seconds of delay.

## 5. The apparent REM concentration is the same artifact

Per-class F1 difference by protocol, fifteen paired folds per column. Positive means the causal
arm pays.

| Class | Sleep-EDF-78 tiled | streaming | DOD-H tiled | streaming |
|---|---|---|---|---|
| W | +0.018 | +0.006 | +0.046 | +0.035 |
| N1 | +0.025 | -0.009 | +0.007 | -0.018 |
| N2 | +0.002 | -0.017 | +0.014 | -0.004 |
| N3 | +0.008 | -0.020 | +0.015 | -0.004 |
| REM | +0.060 | +0.002 | +0.060 | +0.023 |
| Macro | +0.023 | -0.007 | +0.028 | +0.006 |

Under tiling REM appears to pay 0.060 F1 on both datasets, four times the next-largest class on
Sleep-EDF-78 and the only class whose penalty there survives Holm correction across the five
per-class tests. Under the deployment protocol that falls to +0.002 and +0.023, and N1, N2 and N3
invert on both datasets with the causal arm ending ahead.

This is mechanistically sensible. REM is defined partly by surrounding context, which is why a
technician resolves it against neighbouring epochs, so it is the class most damaged by a nearly
empty buffer.

Two residuals remain rather than vanishing. REM still costs 0.023 F1 on DOD-H at deployment, and
Wake costs 0.035 there under both protocols. DOD-H is not Wake-trimmed, so its Wake is
sleep-adjacent and transitional throughout, which is exactly the case where reading forward helps.

Artifact: `results/predictions.md`, produced by `scripts/per_class_breakdown.py`.

## 6. Why the estimand matters at this sample size

DOD-H holds out five subjects per fold against sixteen for Sleep-EDF-78, and its fold-to-fold
kappa standard deviation is 0.098 against 0.026, four times larger.

The same fifteen runs cannot resolve the between-arm causality difference on DOD-H (p = 0.22) and
do resolve the within-arm protocol gain (p = 1.4e-05), because the latter scores identical weights
on identical seconds and removes model variance from the comparison entirely.

At the subject counts common in this literature the choice of estimand decides what is measurable.
That is the practical reason to report the protocol comparison alongside the ablation, and it is
itself a reported result rather than an aside.

## 7. What the per-second rate buys

A per-second rate is only useful if it lets a system report something sooner. Both systems are
charged for the evidence they need: a change counts as reported once the new stage has been held
for *h* consecutive seconds, and an epoch-level system cannot emit the label for epoch *k* until
epoch *k* has ended, so it carries a structural floor of 30 s on any change that begins an epoch.

At *h* = 10 s the per-second system reports first in 94.8% of transitions on Sleep-EDF-78 and
95.6% on DOD-H, with a median 19 seconds saved on both.

The advantage is conditional on *h* and the full sweep is reported: it falls to 66.5% at *h* = 20
and 29.6% at *h* = 30, where the epoch system leads, because requiring 30 seconds of confirmation
reimposes the epoch and discards the resolution.

On DOD-H, over 2,615 scored stage changes, the per-second system detected 2,314 with a median
delay of 17 s against the 30-second system's 2,265 at a median of 30 s. Excluding the 21.3% of
changes where the model was already emitting the incoming stage when the search window opened,
which are more likely false positives during the outgoing stage than genuine early detections, the
figures are 1,758 of 2,058 at a median 25 s against 1,708 of 2,058 at a median 30 s. That
conservative subset is the one to quote.

Artifacts: `results/boundary_latency.md`, `results/boundary_latency_dodh.md`.

## 8. Inference cost

| | |
|---|---|
| TCN plus attention, either arm | 30,757 parameters |
| GRU causal | 20,197 parameters |
| GRU non-causal control | 20,335 parameters |
| CPU inference | 0.026 ms per second of EEG, one thread |

Artifact: `results/verification.md`.

## 9. End-to-end causal preprocessing

Per-epoch z-scoring, near-universal in this literature, normalises an early sample using
statistics computed over the whole epoch including that sample's future. The network is causal;
the pipeline was not.

Replacing it with a trailing 30-second statistic computable one sample at a time raises kappa from
0.6406 to 0.6649 on the causal arm and 0.6693 to 0.6898 on the non-causal arm. Closing the leak is
a modelling gain, not a trade.

`scripts/verify_causality.py` checks the whole path from raw sample to logit by perturbation,
requires every earlier output to be bit-identical, and confirms the offline arrays match a
sample-at-a-time online filter exactly. It exits non-zero on failure. Configs still using
per-epoch z-scoring fail it by design.

Artifacts: `results/verification.md`, `results/causality_verification_gru.md`.

## 10. Output stability

Predicting every second independently makes the raw output far more fragmented than a scored
hypnogram: 181 stage changes an hour against 13 for the technician. A trailing-window mode filter,
which uses only past predictions and so stays causal, cuts that to 26 an hour and adds 0.009 kappa
at no training cost.

Artifact: the smoothing sweep section of `results/statistics.md`.

## 11. Comparison with published baselines

Three published models were reproduced in-house rather than quoted from their papers. Full
settings, per-fold numbers and the defects of each run are in
[`results/baselines/`](results/baselines/README.md).

All three were run on Sleep-EDF's 39-recording cassette subset, so this table uses our
Sleep-EDF-20 figures. Comparing our Sleep-EDF-78 numbers against their 20-subject numbers would
flatter us: the 78-subject set spans ages 25 to 101 and is materially harder.

| Model | Accuracy | Kappa | Parameters | Causal | Output rate | Basis |
|---|---|---|---|---|---|---|
| DeepSleepNet | 82.2% | 0.754 | ~22M | no, Bi-LSTM | 30 s | mean of 5 folds |
| AttnSleep | 80.9% | 0.709 | ~48K | no | 30 s | mean of 10 folds |
| This work, 30 s aggregated | 76.1% | 0.684 | 30.7K | yes | 30 s | mean of 5 folds |
| This work, per-second | 74.4% | 0.662 | 30.7K | yes | 1 s | mean of 5 folds |
| CareSleepNet | 78.1% | 0.694 | 22.7M | no, Transformer | 30 s | **best fold only** |

Read this table with three caveats on the face of it.

**CareSleepNet's row is a best fold, not a mean.** One fold was lost to a logging failure and
another produced 38.12%, so no mean could be recovered. A best fold compared against another
model's mean flatters the best fold. An earlier draft of this document quoted an estimated mean of
about 75.8% and kappa 0.62; that estimate is withdrawn, because averaging an incomplete set of
folds including one anomaly is neither the model's mean nor a bound on it.

**The AttnSleep row is its 30-second configuration.** Its 5-second variants reached 81.3% accuracy
but were trained on a smaller sample than its own 30-second baseline and their kappa was recorded
approximately rather than computed, so they are not citable in a comparison table. Their internal
finding, that a 5-second epoch matches a 30-second one once total context is held equal, is
reported in the baseline document and is consistent with the protocol result here.

**The 30 s aggregated row is the like-for-like comparison** against epoch-level models, obtained
by majority-voting our per-second predictions within each epoch. No retraining is involved.

We remain below DeepSleepNet. It reads future signal in both directions and uses roughly 700 times
the parameters. What this document measures is not whether a small causal model beats a large
offline one, but how much of the apparent gap between a causal model and its own matched twin is
produced by the evaluation protocol.

None of the three baselines can participate in the protocol comparison. They emit one label per
epoch, so there is no context buffer whose positions can be scored separately.

N1 F1 is 0.398 on Sleep-EDF-78, above AttnSleep's 0.36 and CareSleepNet's 0.32.

## 12. Limitations

- Both of our models are small, 20k and 31k parameters, and their absolute kappa is below
  published offline systems on the same corpora. Whether a stronger model pays the same tiling
  penalty is not tested here. Two architecture families in the same capacity class do not settle
  it.
- The streaming protocol denies the non-causal arm the lookahead that defines it, so that
  comparison is between deployment options rather than between architectures.
- The non-causal arm is our own matched twin rather than a state-of-the-art offline model, so the
  comparison bounds what causality costs within this capacity class.
- For the recurrent architecture the two arms are matched on parameter count to 0.68% rather than
  being identical, because a non-causal recurrent encoder necessarily adds a second direction.
  For the convolutional architecture the arms are literally identical.
- DOD-H cannot resolve the between-arm difference at five held-out subjects per fold.
- Supervision is 30-second labels replicated to 1 Hz. No public dataset carries genuine per-second
  expert scoring, so the inference is continuous but the supervision is not. Kappa is the more
  meaningful number.
- Both datasets are single-channel EEG scored to AASM conventions.
- Training-time BatchNorm pools over the time axis within a window. Inference uses running
  statistics, so the deployed path is causal and the perturbation test passes in evaluation mode,
  but the learned weights are not independent of within-window future samples. Both arms share
  this property.

## 13. Reproducing

```powershell
powershell -ExecutionPolicy Bypass -File runs\reproduce.ps1
```

Regenerates every table above from the committed per-fold CSVs, without retraining, into
`results/generated/`. Diff those against the committed copies in `results/`; every script here is
deterministic given the same inputs, so any disagreement is a bug rather than a rounding
difference.

`REPRODUCIBILITY.md` maps each experiment to its config, its run directory and the commands that
produced it.
