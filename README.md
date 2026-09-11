# Causal Sleep Staging

Second-by-second sleep stage classification from a single EEG channel, using only past signal,
and a controlled measurement of what that constraint costs.

The prediction at second *t* depends on the raw signal up to second *t* and on nothing after it.
That holds through the network and through preprocessing, and it is verified by perturbation
rather than asserted. Two architectures are implemented, each with a matched non-causal control,
and both are evaluated on two datasets under two evaluation protocols.

## The result

The cost of the causal constraint is mostly produced by how the models are scored, not by the
constraint.

Two protocols are used. **Tiled** covers a recording with non-overlapping context windows and
scores every position of every window, which is the default in the implementations we have
examined, including our own earlier work. Each window starts from an empty history.
**Streaming** recomputes every 30 seconds against a full buffer and keeps only the freshest 30
predictions, so after the opening window every scored second carries at least 90 seconds of
genuine history. This is what a deployed system does. Every held-out second is scored exactly
once under each protocol, from the same trained weights.

Moving from tiled to streaming lifts both arms, but it lifts the causal arm far more. That
excess is the paper's central quantity, and it is a within-arm measurement: the same weights,
the same held-out seconds, two protocols. It does not require the two arms to be comparable.

| | Sleep-EDF-78, TCN | DOD-H, TCN | Sleep-EDF-78, GRU |
|---|---|---|---|
| Causal arm gains | +0.0350 | +0.0345 | +0.0353 |
| Non-causal arm gains | +0.0036 | +0.0052 | +0.0055 |
| **Excess to the causal arm** | **+0.0314** | **+0.0292** | **+0.0298** |
| Folds with positive excess | 15/15 | 15/15 | 15/15 |
| Bootstrap 95% CI | [.0291, .0334] | [.0253, .0332] | [.0265, .0330] |
| Corrected t(14) | 12.69 | 6.48 | 7.99 |
| p | 4.6e-09 | 1.4e-05 | 1.4e-06 |

Two datasets that differ in montage, sampling rate, scoring team, class balance and subject
count, and two architecture families that share only the convolutional front end, agree on the
size of the penalty to the causal arm to within 0.001 kappa.

The aggregate cost of causality follows from that arithmetic rather than as an independent
observation.

| Dataset, architecture | Tiled | Streaming |
|---|---|---|
| Sleep-EDF-78, TCN | -0.0249 (14/15, p = 3.8e-06) | +0.0065 (7/15, p = 0.47) |
| DOD-H, TCN | -0.0329 (10/15, p = 0.22) | -0.0037 (6/15, p = 0.89) |
| Sleep-EDF-78, GRU | -0.0384 (15/15, p = 2.5e-06) | -0.0085 (11/15, p = 0.14) |

Negative means the causal arm scores lower. All figures are Cohen's kappa on held-out seconds,
pooled over three seeds and five subject-wise folds, tested with the Nadeau-Bengio correction
for repeated cross-validation.

Full per-fold numbers, per-class breakdowns, the buffer-position analysis and the latency
measurements are in [RESULTS.md](RESULTS.md).

## Models

Both share the front end: two parallel causal convolutions at 100 Hz, kernel 50 for spindles and
K-complexes and kernel 400 for slow waves, left-aligned max pooling to one feature vector per
second. They differ only in the sequence encoder.

```
raw EEG, 100 Hz
  -> MRCNN             two-branch causal convolution, pooled to 1 Hz
  -> sequence encoder  either
                         TCN + attention  three residual blocks of causal dilated
                                          convolutions (dilations 1, 2, 4) then four-head
                                          self-attention with a causal mask
                         GRU              two-layer unidirectional recurrent encoder
                                          with layer normalisation
  -> classifier        linear, five classes, one prediction per second
```

| Arm | Parameters |
|---|---:|
| TCN + attention, causal and control | 30,757 |
| GRU, causal | 20,197 |
| GRU, control | 20,335 |

Causality is enforced in two places for the convolutional model: convolutions pad only on the
left by `(kernel - 1) * dilation`, and the attention mask sets positions above the diagonal to
minus infinity. The control removes both and changes nothing else, so the two arms are identical
in capacity.

The recurrent case is different and the difference matters. A unidirectional GRU is causal by
construction and needs no mask, but its non-causal twin must add a second direction, which at
equal hidden size would carry 1.94 times the parameters. The control therefore runs at a reduced
per-direction hidden size, matching the causal arm on parameter count to 0.68% rather than being
literally identical. `scripts/make_gru_configs.py` enforces that match and refuses to write the
configs if it drifts past one percent.

## Causality is verified, not claimed

```bash
python scripts/verify_causality.py --config configs/sleep78_streaming_causal.yaml
```

Seven checks, each requiring bit-identical output rather than closeness. It perturbs a synthetic
recording from second 120 onward and requires every earlier logit to be unchanged; it checks the
offline normaliser against a sample-at-a-time online filter; it confirms the normalised signal
stays finite through the cold start; and it confirms the non-causal control does leak, so the
ablation is a real contrast rather than two identical models. It exits non-zero on failure and
can gate a run. Configs using per-epoch z-scoring fail it by design.

Per-epoch z-scoring is near-universal in this literature and it is a look-ahead leak: normalising
an early sample with statistics computed over the whole epoch reads that sample's future. It is
replaced here by a trailing 30-second statistic computable one sample at a time. Closing that leak
raised kappa rather than lowering it, so it is a modelling gain and not a trade.

## Setup

Python 3.10 or newer. Training needs a CUDA GPU in practice; a full fold takes about 16 minutes
on an RTX 4060 laptop GPU and hours on CPU.

```bash
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Data

Neither dataset is redistributed here.

**Sleep-EDF Expanded**, sleep-cassette subset, 78 subjects and 153 recordings, channel Fpz-Cz at
100 Hz, 1,629 hours after trimming each night to 30 minutes of Wake either side of the sleep
period.

```bash
pip install awscli
aws s3 sync --no-sign-request \
  s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/
```

**DOD-H** from Dreem Open Datasets, 25 subjects, 205.5 hours, channel F3-M2 resampled from 250 Hz
with a polyphase anti-aliasing filter, no Wake trimming since Wake is already 12.3% of that
corpus. See `scripts/dod_preprocessing.py` for acquisition and preparation.

Both are scored to the five AASM classes in 30-second epochs. Each epoch label is replicated
across its 30 one-second targets, so supervision is at 1 Hz but its resolution is not: no public
dataset carries genuine per-second expert scoring.

## Running

```bash
python -m src.data.preprocessing --all                                    # EDF to .npz
python -m src.train.train  --config configs/sleep78_streaming_causal.yaml --fold -1
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --fold 0
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --fold 0 --stream_stride 30
```

The first evaluation call scores under the tiled protocol, the second under streaming. Each
writes `fold_N_test_report.txt` and appends one row to `test_metrics_summary.csv`, with the
streaming results carrying a `_streaming30` suffix. Both are needed before any statistic below
can be computed.

`smoke_test.py` runs the model, losses, optimiser, checkpoint round-trip and metrics on random
batches in a few seconds, which catches shape and device errors before committing to a long run.

The PowerShell drivers under `runs/` chain training, both evaluation protocols, output checks and
the statistics for a whole experiment, and are restartable: anything already finished is skipped.

## Reproducing the reported numbers

Every figure in the tables above comes from a script in this repository operating on committed
per-fold CSVs, and can be regenerated without retraining.

```bash
python scripts/protocol_excess.py \
  --seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42 \
  --seed 43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43 \
  --seed 44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44
```

`scripts/pool_seeds.py` gives the between-arm causality cost with the same correction,
`scripts/latency_sweep.py` gives accuracy as a function of permitted latency across every buffer
position from a single sweep, `scripts/boundary_latency.py` measures how quickly each system
reports a stage change, and `scripts/per_class_breakdown.py` gives the per-class picture.
`runs/reproduce.ps1` runs all of them in order. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps
every reported experiment to its config and its run directory.

## Layout

```
src/
  data/preprocessing.py    EDF loading, wake trimming, resampling, run manifest
  data/normalization.py    trailing-window z-score, offline and online implementations
  data/dataset.py          windowing, subject-wise CV splits, preprocessing-regime guard
  model/                   causal convolution, MRCNN, TCN, attention, GRU, classifier
  train/                   focal and weighted-CE losses, cross-validation loop
  eval/evaluate.py         both evaluation protocols, metrics, CPU latency benchmark
configs/                   one file per experiment, each writing to its own run directory
scripts/                   preprocessing, verification, statistics and analysis
runs/                      PowerShell drivers for whole experiments
tests/                     unit tests, including end-to-end causality for both architectures
results/                   curated per-fold CSVs and reports for every number reported
docs/                      architecture notes for both sequence encoders
figures/                   generated figures
```

Run directories (`logs_*`, `checkpoints_*`) are not committed. The curated per-fold CSVs they
produce are, under `results/`.

## Limitations

- Both models are small, 20k and 31k parameters, and their absolute kappa is below published
  offline systems on the same corpora. Whether a stronger model pays the same tiling penalty is
  not tested here. Two architecture families in the same capacity class do not settle it.
- The streaming protocol denies the non-causal arm the lookahead that defines it, so that
  comparison is between deployment options rather than between architectures. At matched latency
  the two datasets disagree: lookahead is worth nothing on Sleep-EDF-78 and about 0.014 kappa at
  46 seconds of delay on DOD-H.
- The non-causal arm is a matched twin rather than a state-of-the-art offline model, so the
  comparison bounds what causality costs within this capacity class.
- DOD-H holds out five subjects per fold and its fold-to-fold kappa standard deviation is 0.098,
  four times Sleep-EDF-78's. It cannot resolve the between-arm difference, though it resolves the
  within-arm protocol effect decisively. That asymmetry is itself a reported result.
- Supervision is 30-second labels replicated to 1 Hz, so accuracy should be read with that in
  mind and kappa is the more meaningful number.
- Both datasets are single-channel EEG scored to AASM conventions.

## Citation

If you use this work, cite the datasets:

> Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. Analysis of a sleep-dependent
> neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on
> Biomedical Engineering 47(9):1185-1194 (2000).

> Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet:
> Components of a New Research Resource for Complex Physiologic Signals. Circulation
> 101(23):e215-e220 (2000).

> Guillot A, Sauvet F, During EH, Thorey V. Dreem Open Datasets: multi-scored sleep datasets to
> compare human and automated sleep staging. IEEE Transactions on Neural Systems and
> Rehabilitation Engineering 28(9):1955-1965 (2020).

and this repository, using the metadata in [CITATION.cff](CITATION.cff).

## License

Code is released under the MIT License, see [LICENSE](LICENSE). Sleep-EDF Expanded is covered by
the Open Data Commons Attribution License and DOD-H by its own terms; neither is redistributed
here.

## Contributors

Capstone project at PES University, PW25_BJD_21, supervised by
[Dr. Bhaskarjyoti Das](https://scholar.google.co.in/citations?user=d6gtOwwAAAAJ&hl=en).

- [Swapnil S](https://github.com/swapnil5053): the evaluation protocols and the finding built
  on them. End-to-end causal preprocessing and its streaming-equivalence proof. The DOD-H
  replication. The statistical treatment: the within-arm estimand, the Nadeau-Bengio correction,
  and the analysis scripts behind every reported number. Buffer-position, latency and
  boundary-latency measurement. Integration, matched control and experiments for the second
  architecture. The manuscripts.
- [Jahnvi R](https://github.com/jahnvi1504): the initial codebase, the convolutional architecture
  and loss design, the recurrent second-architecture variant, and publication positioning.
- [Shreeya Methuku](https://github.com/shreeya-methuku): literature review and technical writing.
- [Shreshtha Dixit](https://github.com/shreshtha-dixit): literature review and technical writing.
