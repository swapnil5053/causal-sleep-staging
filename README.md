# Causal Sleep Staging

Sleep stage classification from a single EEG channel that predicts a stage every second using
only past signal.

Dense, high-frequency staging is not itself new — U-Sleep evaluated output at up to 7,680
stages per minute back in 2021. What is different here is that the *whole pipeline* is causal,
normalisation included: the prediction at time *t* is a function of the raw signal up to *t*
alone, and that is verified rather than asserted. The model has 30,757 parameters and runs at
0.026 ms per second of EEG on an Intel Core i9-14900HX CPU, so it can keep up with a live
stream on a wearable.

The repository also contains a controlled measurement of what that constraint costs: the same
architecture, parameter for parameter, trained with and without access to future signal. See
[docs/framing_usleep.md](docs/framing_usleep.md) for how this claim is positioned against
U-Sleep and U-Time.

## Results

Subject-wise 5-fold cross-validation on Sleep-EDF, single channel Fpz-Cz at 100 Hz, five classes
(W, N1, N2, N3, REM).

| Metric | Sleep-EDF-20 | Sleep-EDF-78 | Sleep-EDF-78, streaming |
|---|---|---|---|
| Accuracy | 74.4% | 72.3% | **74.9%** |
| Cohen's kappa | 0.662 | 0.634 | **0.663** |
| Macro F1 | 0.688 | 0.662 | **0.690** |
| N1 F1 | 0.336 | 0.398 | **0.414** |
| Kappa at 30 s granularity | 0.684 | 0.652 | **0.683** |
| Fold-to-fold kappa sd | 0.092 | 0.030 | 0.033 |
| Kappa with 30 s causal smoothing | | 0.642 | |
| End-to-end causal preprocessing | no | no | **yes** |
| Parameters | 30,757 | 30,757 | 30,757 |
| CPU inference | 0.026 ms/s | 0.026 ms/s | 0.026 ms/s |

The first two columns z-score each 30-second epoch, which reads samples from later in that
epoch: the network is causal, the pipeline is not. The streaming column replaces that with a
trailing 30-second window, so no stage of the pipeline touches the future. Kappa rises by
0.030 rather than falling, so end-to-end causality costs nothing here.

Predicting every second independently makes the raw output far more fragmented than a scored
hypnogram: 181 stage changes an hour against 13 for the technician. A trailing-window mode
filter, which uses only past predictions and so stays causal, cuts that to 26 an hour and adds
0.009 kappa at no training cost. See [RESULTS.md](RESULTS.md).

### What causality costs

The same model with the causal constraint removed, holding parameters, data and folds fixed:

| Subjects | Normalization | Causal | Non-causal | Difference | p | Folds causal loses |
|---|---|---|---|---|---|---|
| 20 | epoch z-score | 0.6625 | 0.6482 | +0.014 | 0.276 | 2 of 5 |
| 78 | epoch z-score | 0.6406 | 0.6693 | -0.029 | 5.9e-08 | 15 of 15 |
| 78 | causal rolling | 0.6649 | 0.6898 | **-0.025** | **3.8e-06** | **14 of 15** |

Both 78-subject rows pool three seeds over five folds (15 paired measurements each); the
20-subject row is a single seed.

On 78 subjects, giving the model access to future signal improves kappa by roughly 0.03,
consistently across every fold. That is the measurable price of running in real time, and it
holds under both normalization regimes, each measured over three seeds and five folds: -0.029
with epoch z-scoring and -0.025 with the fully causal pipeline (paired t(14) = -7.33,
p = 3.8e-06, 95% CI [-0.031, -0.018]). The causal model loses in 14 of those 15 measurements.

The same comparison on 20 subjects is not significant and its sign is unstable, because
fold-to-fold variance there is three times larger (kappa sd 0.092 against 0.030). Causality
penalties measured on 20 subjects should be treated with caution.

Per-fold numbers, the configuration and preprocessing ablations, and the full baseline
comparison are in [RESULTS.md](RESULTS.md).

## Model

```
raw EEG, 100 Hz
  -> MRCNN        two parallel causal convolutions: kernel 50 (0.5 s, spindles and
                  K-complexes) and kernel 400 (4 s, slow waves), max-pooled to 1 Hz
  -> TCN          residual blocks of causal dilated convolutions (default 3, dilations 1, 2, 4)
  -> attention    causal masked multi-head self-attention, 4 heads
  -> classifier   linear, 5 classes, one prediction per second
```

Causality is enforced by left-padding every convolution by `(kernel - 1) * dilation` and masking
the attention above the diagonal. Perturbing the signal at second *t* leaves every output before
*t* bit-identical.

Setting `causal: false` keeps every layer, channel and parameter identical but pads convolutions
symmetrically and removes the attention mask, so the model can see the future. That is the only
difference, which makes the causal/non-causal comparison a controlled one.

## Setup

Python 3.10+. A CUDA GPU is not required but training takes hours on CPU.
For exact experiment-to-config mappings and a complete run checklist, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Data

Sleep-EDF Expanded from PhysioNet, sleep-cassette subset, subjects 00-19 (about 2 GB).

```bash
pip install awscli
aws s3 sync --no-sign-request --exclude "*" --include "SC40*" --include "SC41*" \
  s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/
```

That gives 39 `*-PSG.edf` files with their matching `*-Hypnogram.edf` files. The hypnogram
filenames carry a scorer letter that varies (`SC4001EC`, `SC4011EH`, ...); the loader matches
them by subject prefix, so nothing needs renaming.

Drop the two `--include` filters to fetch the full 78-subject cassette set instead.

## Running

```bash
python -m src.data.preprocessing --all     # EDF -> data/processed/subject_XX.npz
python -m src.train.train --fold -1        # all 5 folds
python -m src.eval.evaluate --fold 0       # repeat for folds 0-4
python -m src.eval.evaluate --benchmark    # CPU latency
```

Evaluation reports metrics twice: per-second, and aggregated to 30-second epochs by majority
vote. The second set is what compares like for like against published 30-second results.

Output lands in `logs/`: per-epoch curves in `fold_N_metrics.csv`, held-out test reports in
`fold_N_test_report.txt`, one row per fold in `test_metrics_summary.csv`, and one row per
held-out subject in `fold_N_subject_metrics.csv` and `test_subject_metrics.csv`. Checkpoints
and the subject splits used for each fold go to `checkpoints/`.

## Streaming

To watch the deployed path run, rather than reading that it exists:

```bash
python scripts/streaming_demo.py --synthetic --verify          # needs no data at all
python scripts/streaming_demo.py --edf data/raw/SC4001E0-PSG.edf \
  --hypnogram data/raw/SC4001EC-Hypnogram.edf \
  --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth
```

Raw samples go in one at a time through `StreamingZScore`, and one stage comes out per second.
`--verify` additionally re-runs the whole recording and asserts the labels are identical to
those `src/eval/evaluate.py` produces, so the reported kappa is the number this path gives and
not one obtained by batching a night. Without `--checkpoint` the weights are random: the labels
are then meaningless, but causality, throughput and the equivalence check do not depend on the
weights and still hold.

## Experiments

Each config writes to its own `log_dir` and `checkpoint_dir`, so runs never overwrite each other.

| Config | Variable under test |
|---|---|
| `configs/default.yaml` | baseline, 60 s context, 3 TCN blocks |
| `configs/run_b_context.yaml` | 120 s context |
| `configs/run_c_depth.yaml` | 120 s context plus a 4th TCN block (37,093 params) |
| `configs/run_d_noncausal.yaml` | identical to run C but `causal: false` |

```bash
python -m src.train.train --config configs/run_c_depth.yaml --fold -1
python -m src.eval.evaluate --config configs/run_c_depth.yaml --fold 0
```

Run C minus run D is the cost of causality on this architecture, holding everything else fixed.

`smoke_test.py` runs the model, losses, optimiser, checkpoint round-trip and metrics on random
batches in a few seconds, which catches shape and device errors before a long run:

```bash
python smoke_test.py configs/run_c_depth.yaml
```

## Preprocessing notes

Three decisions affect the numbers a lot.

**Normalization.** The archived runs z-scored each 30-second epoch using that epoch's own
mean and standard deviation. The network is causal with respect to its input, but that
statistic reads samples from later in the epoch, so the *pipeline* was not end-to-end causal
even though the model was. `data.normalization.method: causal_rolling` replaces it with a
trailing z-score: at sample *t* the mean and standard deviation come from
`[t - window + 1, t]` only, and the statistic resets at the start of every recording. It is
mathematically identical to what a device computes sample by sample, and
`src/data/normalization.py` ships that online filter (`StreamingZScore`) so the equivalence
is checked rather than asserted.

`python scripts/verify_causality.py` runs the whole path — raw sample, normalization, model —
perturbs the input at a future second, and requires every earlier output to be bit-identical.
It writes `results/causality_verification.md` and exits non-zero on failure, so it can gate a
run. Configs that still use `epoch_zscore` fail it by design.

Only `configs/sleep78_streaming_*.yaml` use the causal normalization; every other config keeps
`epoch_zscore` so the archived numbers stay reproducible. The two regimes write to different
`processed_dir`s and the loader refuses to mix them in one directory.

Sleep-EDF cassette recordings run about 20 hours per night, most of it awake and out of bed.
Keeping all of it makes Wake 68% of the data and inflates accuracy, and it is not what the
published baselines do. `wake_trim_minutes: 30` keeps 30 minutes of wake either side of each
night's sleep period and drops the rest, which brings Wake down to roughly 15%. Trimming runs per
night, before a subject's two nights are concatenated, so the daytime gap between the two
recordings is removed as well.

Labels are 30-second epoch labels replicated across their 30 seconds. No public dataset has
per-second expert scoring, so the inference is continuous but the supervision is not. Accuracy
should be read with that in mind, and kappa is the more meaningful number.

## Layout

```
src/
  data/preprocessing.py   EDF loading, wake trimming, normalisation, run manifest
  data/normalization.py   trailing-window z-score, offline and online implementations
  data/dataset.py         windowing, subject-wise CV splits, weighted sampler
  model/                  causal conv, MRCNN, TCN, attention, classifier
  train/                  focal and weighted-CE losses, cross-validation loop
  eval/evaluate.py        metrics and CPU latency benchmark
configs/                  default plus the ablation and streaming configs
results/                  archived runs, see RESULTS.md
scripts/verify_causality.py   end-to-end causality proof, writes a report
scripts/recover_splits.py     rebuild archived split files from the fold reports
scripts/warm_start_eval.py    scores cold and warm on identical seconds, dumps predictions
scripts/calibration.py        reliability, ECE and Brier from a prediction dump
scripts/transition_response.py  how fast the model follows a stage change
docs/paper_report_template.md   required write-up before any method change
scripts/streaming_demo.py     sample-at-a-time staging, and proof it matches evaluate.py
scripts/subject_paired_test.py  causality cost paired by subject rather than by fold
scripts/pool_seeds.py         pooled paired test across seeds
scripts/validate_results.py   structural check on archived result CSVs
docs/framing_usleep.md        how the contribution is positioned against U-Sleep
smoke_test.py             fast pre-run sanity check
tests/                    unit tests, including end-to-end causality
```

## Config notes

- `sequence_length` (60) is the context in seconds. Longer gives the attention more history and
  is the most useful knob if kappa is low. Must be a multiple of 30 for the 30-second reporting.
- `causal` (true). Set to false only for the causality ablation.
- `seed` (42) controls weight initialisation, shuffling and dropout. `split_seed` (42) controls
  the cross-validation subject partition, and is separate on purpose: a repeat under a new
  `seed` is only a replication if it keeps the same folds. Every config now pins `split_seed`
  to 42 so seed variants differ in initialisation alone.
- `wake_trim_minutes` (30). Set to `null` to keep the full recordings.
- `use_weighted_sampler` (false). Focal loss already handles the class imbalance; enabling both
  makes the model over-predict N1 badly (precision drops to 0.12-0.20).
- `checkpoint_metric` (kappa) selects which validation metric decides the saved checkpoint.
  Macro F1 and kappa peak at different epochs, so this should match whichever you report.
- `early_stopping_patience` (8). Validation plateaus by roughly epoch 10, so `epochs` is a cap
  rather than a target.
- Write `weight_decay` as `0.0001`, not `1e-4`. YAML parses the latter as a string and the
  optimiser then fails.

## Limitations

- Single dataset (Sleep-EDF-20) and single channel, so fold variance is large: kappa ranges
  0.537 to 0.711 across the five folds.
- Supervision is 30-second labels replicated to 1 Hz, not genuine per-second scoring.
- Validation peaks early and the model overfits past roughly epoch 10.
- The archived seed-43 and seed-44 runs were produced before `split_seed` was separated from
  `seed`, so they use different subject partitions rather than the same folds under a new
  initialisation. The configs now fix this going forward; the archived numbers are what they
  are, and pooling them needs a correction for the reused subject pool.
- Evaluation uses non-overlapping windows, so the causal model restarts with almost no context
  at each window boundary while the non-causal model sees the whole window. Part of the measured
  cost is therefore a boundary artefact. `scripts/warm_start_eval.py` measures how much, scoring
  the same seconds cold and warm; it needs a checkpoint to produce numbers.
- Supervision is 30-second epoch labels, so a scored stage transition is only ever located to
  the nearest epoch boundary. Response-time results are bounded by that, not by the model.
- The 30-second normalisation window is the only one tested. `sleep78_streaming_causal_w120`
  exists as the ablation config and has not been run.

## Data and citation

Sleep-EDF Expanded is distributed by PhysioNet under the Open Data Commons Attribution License
v1.0. If you use this work, cite the dataset and PhysioNet:

> Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent
> neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on
> Biomedical Engineering 47(9):1185-1194 (2000).

> Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet:
> Components of a New Research Resource for Complex Physiologic Signals. Circulation
> 101(23):e215-e220 (2000).

To cite this repository:

```bibtex
@software{causal_sleep_staging_2026,
  author = {Jahnvi R and Swapnil S and Methuku, Shreeya and Dixit, Shreshtha},
  title  = {Causal Sleep Staging: second-by-second sleep stage classification
            from single-channel EEG},
  year   = {2026},
  url    = {https://github.com/swapnil5053/causal-sleep-staging}
}
```

## License

Code released under the MIT License, see [LICENSE](LICENSE). The Sleep-EDF Expanded dataset is
covered separately by its own Open Data Commons Attribution License and is not redistributed
here.

## Credits

Capstone project at PES University (PW25_BJD_21).

- [Jahnvi R](https://github.com/jahnvi1504): initial codebase, model architecture, loss function design, and publication positioning
- [Swapnil S](https://github.com/swapnil5053): data pipeline, training and evaluation, results analysis
- [Shreeya Methuku](https://github.com/shreeya-methuku): technical writing and literature review
- [Shreshtha Dixit](https://github.com/shreshtha-dixit): technical writing and literature review

Supervised by [Dr. Bhaskarjyoti Das](https://scholar.google.co.in/citations?user=d6gtOwwAAAAJ&hl=en).
