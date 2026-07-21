# Causal Sleep Staging

Sleep stage classification from a single EEG channel that predicts a stage every second using
only past signal.

Most sleep staging models score 30-second epochs and read the whole night at once, so they
cannot run live. This one is strictly causal: the prediction at time *t* depends only on input
up to *t*. It has 30,757 parameters and runs at 0.045 ms per second of EEG on a CPU, so it can
keep up with a live stream on a wearable.

## Results

Subject-wise 5-fold cross-validation on Sleep-EDF-20 (20 subjects, 39 recordings), single
channel Fpz-Cz at 100 Hz, five classes (W, N1, N2, N3, REM).

| Metric | Value |
|---|---|
| Accuracy | 72.8% |
| Cohen's kappa | 0.643 |
| Macro F1 | 0.679 |
| Parameters | 30,757 |
| CPU inference | 0.045 ms per second of EEG |

Per-fold numbers, the preprocessing ablation, and a comparison against published baselines
are in [RESULTS.md](RESULTS.md).

## Model

```
raw EEG, 100 Hz
  -> MRCNN        two parallel causal convolutions: kernel 50 (0.5 s, spindles and
                  K-complexes) and kernel 400 (4 s, slow waves), max-pooled to 1 Hz
  -> TCN          3 residual blocks of causal dilated convolutions (dilations 1, 2, 4)
  -> attention    causal masked multi-head self-attention, 4 heads
  -> classifier   linear, 5 classes, one prediction per second
```

Causality is enforced by left-padding every convolution by `(kernel - 1) * dilation` and
masking the attention above the diagonal. Perturbing the signal at second *t* leaves every
output before *t* bit-identical.

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

## Running

```bash
python -m src.data.preprocessing --all     # EDF -> data/processed/subject_XX.npz
python -m src.train.train --fold -1        # all 5 folds
python -m src.eval.evaluate --fold 0       # repeat for folds 0-4
python -m src.eval.evaluate --benchmark    # CPU latency
```

Output lands in `logs/`: per-epoch curves in `fold_N_metrics.csv`, held-out test reports in
`fold_N_test_report.txt`, and one row per fold in `test_metrics_summary.csv`. Checkpoints and
the subject splits used for each fold go to `checkpoints/`.

## Preprocessing notes

Two decisions affect the numbers a lot.

Sleep-EDF cassette recordings run about 20 hours per night, most of it awake and out of bed.
Keeping all of it makes Wake 68% of the data and inflates accuracy, and it is not what the
published baselines do. `wake_trim_minutes: 30` keeps 30 minutes of wake either side of each
night's sleep period and drops the rest, which brings Wake down to roughly 15%. Trimming runs
per night, before a subject's two nights are concatenated, so the daytime gap between the two
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
configs/default.yaml      hyperparameters
results/                  archived runs, see RESULTS.md
```

## Config notes

- `sequence_length` (60) is the context in seconds. Longer gives the attention more history
  and is the most useful knob if kappa is low.
- `wake_trim_minutes` (30). Set to `null` to keep the full recordings.
- `use_weighted_sampler` (false). Focal loss already handles the class imbalance; enabling
  both makes the model over-predict N1 badly (precision drops to 0.12-0.20).
- Write `weight_decay` as `0.0001`, not `1e-4`. YAML parses the latter as a string and the
  optimiser then fails.

## Credits

PES University capstone project PW25_BJD_21. Initial codebase and project scaffolding by
Jahnvi R.
