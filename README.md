# Causal Sleep Staging

Sleep stage classification from a single EEG channel that predicts a stage every second using
only past signal.

Most sleep staging models score 30-second epochs and read the whole night at once, so they
cannot run live. This one is strictly causal: the prediction at time *t* depends only on input
up to *t*. It has 30,757 parameters and runs at 0.026 ms per second of EEG on an Intel Core
i9-14900HX CPU, so it can keep up with a live stream on a wearable.

The repository also contains a controlled measurement of what that constraint costs: the same
architecture, parameter for parameter, trained with and without access to future signal.

## Results

Subject-wise 5-fold cross-validation on Sleep-EDF, single channel Fpz-Cz at 100 Hz, five classes
(W, N1, N2, N3, REM).

| Metric | Sleep-EDF-20 | Sleep-EDF-78 |
|---|---|---|
| Accuracy | 74.4% | 72.3% |
| Cohen's kappa | 0.662 | 0.634 |
| Macro F1 | 0.688 | 0.662 |
| N1 F1 | 0.336 | 0.398 |
| Kappa at 30 s granularity | 0.684 | 0.652 |
| Fold-to-fold kappa sd | 0.092 | 0.030 |
| Kappa with 30 s causal smoothing | | 0.642 |
| Parameters | 30,757 | 30,757 |
| CPU inference | 0.026 ms/s | 0.026 ms/s |

Predicting every second independently makes the raw output far more fragmented than a scored
hypnogram: 181 stage changes an hour against 13 for the technician. A trailing-window mode
filter, which uses only past predictions and so stays causal, cuts that to 26 an hour and adds
0.009 kappa at no training cost. See [RESULTS.md](RESULTS.md).

### What causality costs

The same model with the causal constraint removed, holding parameters, data and folds fixed:

| Subjects | Causal | Non-causal | Difference | p | Folds causal loses |
|---|---|---|---|---|---|
| 20 | 0.6625 | 0.6482 | +0.014 | 0.276 | 2 of 5 |
| 78 | 0.6335 | 0.6570 | **-0.024** | **0.013** | **5 of 5** |

On 78 subjects, giving the model access to future signal improves kappa by 0.024, consistently
across every fold (paired t(4) = -4.24, p = 0.013). That is the measurable price of running in
real time.

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
`fold_N_test_report.txt`, and one row per fold in `test_metrics_summary.csv`. Checkpoints and the
subject splits used for each fold go to `checkpoints/`.

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

Two decisions affect the numbers a lot.

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
  data/preprocessing.py   EDF loading, wake trimming, per-epoch normalisation
  data/dataset.py         windowing, subject-wise CV splits, weighted sampler
  model/                  causal conv, MRCNN, TCN, attention, classifier
  train/                  focal and weighted-CE losses, cross-validation loop
  eval/evaluate.py        metrics and CPU latency benchmark
configs/                  default plus the ablation configs
results/                  archived runs, see RESULTS.md
smoke_test.py             fast pre-run sanity check
```

## Config notes

- `sequence_length` (60) is the context in seconds. Longer gives the attention more history and
  is the most useful knob if kappa is low. Must be a multiple of 30 for the 30-second reporting.
- `causal` (true). Set to false only for the causality ablation.
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
