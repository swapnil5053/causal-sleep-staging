# Reproducing the experiments

This guide maps the repository's configurations, commands, and archived outputs. Run commands
from the repository root. Raw Sleep-EDF data is not distributed with this repository.

> **Causality scope:** the archived experiments use per-epoch z-score normalization calculated
> over each complete 30-second epoch. The neural network is causal with respect to its normalized
> input, but this preprocessing step uses later samples from the epoch. Preserve it only when
> reproducing the archived numbers exactly. The streaming configurations below replace it with a
> past-only statistic and retrain every compared model.

### End-to-end causal preprocessing pilot

The streaming configurations are isolated from the archived epoch-normalized data: they read and
write their own `processed_dir`, log directory and checkpoint directory, so nothing already
archived is touched. They apply a trailing 30-second z-score at every raw sample; the statistic
at time `t` contains only samples at or before `t` and resets at each recording.

| Config | Seed | Causal | Normalization |
|---|---:|:---:|---|
| `configs/sleep78_streaming_causal.yaml` | 42 | yes | trailing 30 s |
| `configs/sleep78_streaming_noncausal.yaml` | 42 | no | trailing 30 s |
| `configs/sleep78_streaming_causal_s43.yaml` | 43 | yes | trailing 30 s |
| `configs/sleep78_streaming_noncausal_s43.yaml` | 43 | no | trailing 30 s |
| `configs/sleep78_streaming_causal_s44.yaml` | 44 | yes | trailing 30 s |
| `configs/sleep78_streaming_noncausal_s44.yaml` | 44 | no | trailing 30 s |
| `configs/sleep78_streaming_causal_w120.yaml` | 42 | yes | trailing 120 s (window robustness, optional) |

Order of work:

```bash
# 0. proofs and sanity checks, seconds, no data required
python -m unittest discover -s tests -v
python smoke_test.py configs/sleep78_streaming_causal.yaml
python scripts/verify_causality.py --config configs/sleep78_streaming_causal.yaml

# 1. preprocessing, once, shared by both arms
python -m src.data.preprocessing --config configs/sleep78_streaming_causal.yaml --all

# 2. go/no-go on one fold per arm before committing to the full sweep
python -m src.train.train --config configs/sleep78_streaming_causal.yaml --fold 0
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --fold 0
```

Compare that fold-0 kappa against fold 0 of `results/sleep78_causal/test_metrics_summary.csv`,
which is the same architecture, seed and split under the old epoch normalization. A drop of a
few thousandths is expected and reportable; a collapse means the window length needs revisiting
before spending the rest of the compute.

Each fold's checkpoint directory records the subject split, so folds may be run individually or
with `--fold -1`; training re-seeds per fold, and both give identical results.

`scripts/verify_causality.py` has been run against the trained causal checkpoint and its report
is archived at `results/causality_verification.md`. Re-run it after any change to normalization,
padding or masking, and archive the new report with the run; the end-to-end causal claim rests
on it.

### Preprocessing manifest

Every preprocessing run writes `preprocessing_manifest.json` into its `processed_dir`. It records
the config, normalization method and window, the number of PSG files found, paired, written and
failed, the class distribution, and per-subject second counts. Archive it with the run: it is the
evidence for the data-side half of the run record in section 9, and it is how a reviewer confirms
which normalization produced a given result set.

The dataset loader reads the same metadata from each `.npz` and refuses to build a dataset from a
directory that mixes normalization regimes, so a partially reprocessed directory fails loudly at
the start of training rather than silently producing an unreportable number.

### Separating the partition from the initialization

`train.seed` seeds the weight initialization and, historically, also the subject shuffle that
decides the folds. Seeds 42, 43 and 44 therefore produced three different partitions rather than
three initializations of one partition, and the two effects cannot be separated in the archived
runs: they are three repeats of five-fold cross-validation over fifteen distinct partitions.
Analyses of that data should account for the train-set overlap between repeats rather than
treating the fifteen cells as fifteen independent folds.

`train.split_seed` pins the partition independently of `train.seed`:

```yaml
train:
  seed: 43           # weight initialization
  split_seed: 42     # fold layout; hold fixed to vary only the initialization
```

The two sit together so it is visible that they are separate knobs. `data.split_seed` is
accepted as an alias, but `train.split_seed` wins if both are present, so do not set both.

Omitting `split_seed` uses `train.seed`, so every configuration written before this option
existed produces exactly the splits it always did. Each fold's `split_fold_N.yaml` now records
both seeds alongside the subject lists, so the partition behind a result is recoverable from the
run itself. Setting one `split_seed` across a seed sweep gives the matched design the archived
runs do not have.

### Continuity of the stored signal

Preprocessing drops unscored epochs and concatenates a subject's two nights, so a stored array is
a sequence of discontinuous stretches rather than one recording. Each `.npz` now carries
`segment_starts`, the second indices where a fresh stretch begins, and the manifest reports how
many segments each subject has.

Two loader options use it, both off by default so the archived windowing is unchanged:

| Option | Effect |
|---|---|
| `data.respect_boundaries` | Drop context windows that span a night join or an unscored-epoch gap |
| `data.cover_tail` | Add a final window flush with the end of the recording instead of discarding the trailing `(len - seq_len) % stride` seconds |

`respect_boundaries` needs `segment_starts`, so it raises rather than silently doing nothing on
directories processed before that metadata existed. Re-run preprocessing into a clean directory
to use it.

### The normalizer resets at every recording

`StreamingZScore` is reset at the start of each recording, including between a subject's two
nights. This is deliberate — a device powering on has no history either — but it means the
first seconds of every night are normalized against a partial window rather than a full 30 s
one, and this is worth one sentence in the methods rather than leaving a reviewer to find it.

It is safe rather than merely tolerable. Within a trailing window holding `n` samples the
largest attainable magnitude is `sqrt(n - 1)`, so the cold start cannot produce an infinity
or a NaN however few samples have been seen; `causal_rolling_zscore` returns 0 while the
standard deviation is below `eps`. `scripts/verify_causality.py` checks that bound on every
run and reports the measured maximum against it, and
`tests/test_end_to_end_causality.py::test_normalized_output_is_finite_including_warmup`
asserts it directly.

## 1. Environment

Python 3.10 or newer is required. Create an isolated environment and install the pinned minimum
dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for the machine before installing the
remaining requirements. Record the Python, PyTorch, CUDA, GPU, operating-system, and dependency
versions alongside a new experiment; these details can affect numerical reproducibility.

## 2. Data acquisition

The experiments use the Sleep-EDF Expanded sleep-cassette subset from PhysioNet. For the
20-subject subset used by the configuration ablations, download subjects 00 through 19:

```bash
aws s3 sync --no-sign-request --exclude "*" --include "SC40*" --include "SC41*" \
  s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/
```

Remove both `--include` filters to retrieve the complete 78-subject sleep-cassette set. A valid
download contains paired `*-PSG.edf` and `*-Hypnogram.edf` files. The loader matches recordings
and annotations by their subject prefix; scorer letters in hypnogram filenames need not be
renamed.

The dataset has its own Open Data Commons Attribution License. Follow PhysioNet's current access,
license, and citation requirements when redistributing derived material or publishing results.

## 3. Preprocessing

Process the 20-subject data with the default configuration:

```bash
python -m src.data.preprocessing --config configs/default.yaml --all
```

Process the full 78-subject data with its dedicated configuration:

```bash
python -m src.data.preprocessing --config configs/sleep78_causal.yaml --all
```

Both the causal and non-causal 78-subject experiments point to `data/processed78`, so preprocessing
is performed once and the same arrays are used in the controlled comparison. Each output file is
named `subject_<ID>.npz` and contains:

- `x`: normalized EEG, `float32`, shape `(number_of_seconds, 100)`;
- `y`: integer labels, `int64`, shape `(number_of_seconds,)`;
- `normalization_method`, `normalization_window_seconds`, `normalization_eps`,
  `wake_trim_minutes`, `resample_rate`: the preprocessing that produced the file.

Before training, record the number of raw PSG files, paired hypnograms, processed subjects, and
any skipped recordings. Never combine processed files created with different preprocessing
methods in one directory.

## 4. Fast pre-run checks

Run the smoke test before committing resources to a complete fold:

```bash
python smoke_test.py configs/sleep78_causal.yaml
```

Run the unit tests with the Python standard library test runner:

```bash
python -m unittest discover -s tests -v
```

The tests check tensor dimensions, parameter count, model-layer causality, subject split
integrity, and — in `tests/test_end_to_end_causality.py` — causality of the composed
normalization-plus-model path, including that the offline arrays equal the online
sample-at-a-time filter and that the non-causal arm genuinely does leak.

For a run-specific, archivable version of the same proof against real weights:

```bash
python scripts/verify_causality.py \
  --config configs/sleep78_streaming_causal.yaml \
  --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth \
  --subject data/processed78_streaming/subject_00.npz \
  --out results/causality_verification.md
```

## 5. Training

Train one fold first to verify memory use, output locations, and approximate runtime:

```bash
python -m src.train.train --config configs/sleep78_causal.yaml --fold 0
```

Train all five folds sequentially:

```bash
python -m src.train.train --config configs/sleep78_causal.yaml --fold -1
```

For every fold, training saves:

- the subject split in `split_fold_<N>.yaml`;
- the best checkpoint in `best_model_fold_<N>.pth`;
- the per-epoch training and validation history in `fold_<N>_metrics.csv`.

The validation fold selects the checkpoint using the configured metric (kappa by default). The
test fold is not used for parameter updates, early stopping, or checkpoint selection.

To reproduce another experiment, change only the configuration path. Do not manually edit a
shared configuration during a run.

## 6. Held-out evaluation

Evaluation is run once per fold. On Bash-compatible shells:

```bash
for fold in 0 1 2 3 4; do
  python -m src.eval.evaluate --config configs/sleep78_causal.yaml --fold "$fold"
done
```

On Windows PowerShell:

```powershell
0..4 | ForEach-Object {
    python -m src.eval.evaluate --config configs/sleep78_causal.yaml --fold $_
}
```

The evaluator produces a classification report for each fold and appends one row per fold to
`test_metrics_summary.csv`. Start with a new log directory or remove a disposable duplicate
summary before repeating evaluation; the evaluator appends rather than replacing existing rows.
Do not delete or overwrite curated files under `results/`.

The per-second metrics compare predictions against 30-second expert labels replicated to each
second. The 30-second metrics majority-vote each group of 30 predictions for comparison with
conventional epoch-level systems.

### Saved predictions

Each evaluation also writes `fold_N_predictions.npz` holding the per-second subject id, expert
label, prediction and logits. Smoothing sweeps, per-subject confidence intervals, calibration and
class-prior correction all read from it, so none of them require another pass over a checkpoint.
Archive it with the run; `--no_save_predictions` skips it.

### Streaming evaluation

The default evaluation tiles each recording with non-overlapping windows, so the opening seconds
of every window are predicted from a partly zero-padded history. A deployment does not restart
its buffer that way. `--stream_stride N` recomputes every `N` seconds against a full context
window and keeps only the freshest `N` predictions, so every second after the first window is
decided with at least `sequence_length - N` seconds of real history:

```bash
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --fold 0 --stream_stride 30
```

Cost scales as `sequence_length / N` forward passes. Results are written under a `_streamingN`
suffix (`test_metrics_summary_streaming30.csv` and so on) so the two modes sit side by side and
neither overwrites the archived artifacts. With `--stream_stride` equal to `sequence_length` the
two modes coincide, apart from the trailing seconds that tiling drops.

## 7. Latency, smoothing, statistics, and figures

Run the CPU throughput benchmark for a configuration:

```bash
python -m src.eval.evaluate --config configs/sleep78_causal.yaml --benchmark
```

The reported `ms/sec` value is the time for a complete synthetic sequence divided by its length.
Record total window time as well when describing latency, and distinguish this throughput measure
from end-to-end streaming latency.

Evaluate trailing-window smoothing using the trained causal checkpoints:

```bash
python sweep_smoothing.py --config configs/sleep78_causal.yaml \
  --out results/smoothing.md
```

Generate the archived statistical summary:

```bash
python analysis_stats.py --out results/statistics.md
```

Run the same paired tests on any other run pair with `--pair NAME=causal_dir,noncausal_dir`
(repeatable). For the streaming pilot:

```bash
python analysis_stats.py \
  --pair "Sleep-EDF-78 streaming=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42" \
  --out results/statistics_streaming.md
```

When the same comparison has been repeated under several seeds, report the pooled paired test
over all seeds x folds rather than three separate five-fold tests:

```bash
python scripts/pool_seeds.py \
  --seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42 \
  --seed 43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43 \
  --seed 44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44 \
  --out results/statistics_streaming_pooled.md
```

It refuses to run if a summary CSV lists a fold twice or if the two arms cover different folds,
so a half-finished sweep cannot be reported as a complete one.

Generate figures (requires the processed data and referenced checkpoint for the hypnogram panel):

```bash
python make_figures.py \
  --config configs/sleep78_causal.yaml \
  --run results/sleep78_causal \
  --checkpoint checkpoints_78causal/best_model_fold_0.pth \
  --out figures
```

Validate archived CSV structure without changing any files:

```bash
python scripts/validate_results.py results
```

## 8. Experiment provenance

The table links reproducible configurations to their curated results. Generated log and
checkpoint directories are intentionally ignored by Git; selected reports and, where present,
checkpoints are archived under `results/`.

| Experiment | Configuration | Seed | Context | Causal | Curated results |
|---|---|---:|---:|:---:|---|
| Sleep-EDF-20 baseline | `configs/default.yaml` | 42 | 60 s | yes | `results/run_a_baseline/` |
| Sleep-EDF-20 longer context | `configs/run_b_context.yaml` | 42 | 120 s | yes | `results/run_b_context/` |
| Sleep-EDF-20 extra TCN depth | `configs/run_c_depth.yaml` | 42 | 120 s | yes | `results/run_c_depth/` |
| Sleep-EDF-20 non-causal, baseline depth | `configs/run_d2_noncausal.yaml` | 42 | 120 s | no | `results/run_d2_noncausal/` |
| Sleep-EDF-20 non-causal, extra depth | `configs/run_d_noncausal.yaml` | 42 | 120 s | no | `results/run_d_noncausal/` |
| Sleep-EDF-78 causal | `configs/sleep78_causal.yaml` | 42 | 120 s | yes | `results/sleep78_causal/` |
| Sleep-EDF-78 causal | `configs/sleep78_causal_s43.yaml` | 43 | 120 s | yes | `results/sleep78_causal_s43/` |
| Sleep-EDF-78 causal | `configs/sleep78_causal_s44.yaml` | 44 | 120 s | yes | `results/sleep78_causal_s44/` |
| Sleep-EDF-78 non-causal | `configs/sleep78_noncausal.yaml` | 42 | 120 s | no | `results/sleep78_noncausal/` |
| Sleep-EDF-78 non-causal | `configs/sleep78_noncausal_s43.yaml` | 43 | 120 s | no | `results/sleep78_noncausal_s43/` |
| Sleep-EDF-78 non-causal | `configs/sleep78_noncausal_s44.yaml` | 44 | 120 s | no | `results/sleep78_noncausal_s44/` |
| Sleep-EDF-78 streaming causal | `configs/sleep78_streaming_causal.yaml` | 42 | 120 s | yes | `results/sleep78_streaming_causal_s42/` |
| Sleep-EDF-78 streaming causal | `configs/sleep78_streaming_causal_s43.yaml` | 43 | 120 s | yes | `results/sleep78_streaming_causal_s43/` |
| Sleep-EDF-78 streaming causal | `configs/sleep78_streaming_causal_s44.yaml` | 44 | 120 s | yes | `results/sleep78_streaming_causal_s44/` |
| Sleep-EDF-78 streaming non-causal | `configs/sleep78_streaming_noncausal.yaml` | 42 | 120 s | no | `results/sleep78_streaming_noncausal_s42/` |
| Sleep-EDF-78 streaming non-causal | `configs/sleep78_streaming_noncausal_s43.yaml` | 43 | 120 s | no | `results/sleep78_streaming_noncausal_s43/` |
| Sleep-EDF-78 streaming non-causal | `configs/sleep78_streaming_noncausal_s44.yaml` | 44 | 120 s | no | `results/sleep78_streaming_noncausal_s44/` |

The streaming rows use trailing-window normalization and `data/processed78_streaming`. Every
other row uses the epoch z-score and `data/processed78`. The two regimes are never mixed in one
processed directory.

`configs/sleep78_ctx300.yaml` defines a 300-second causal experiment, but this repository does
not contain a matching curated result directory. `results/trimmed/` and
`results/baseline_untrimmed/` are legacy preprocessing-ablation archives and are not mapped to
complete dedicated configuration files in the current tree; treat them as supporting artifacts,
not one-command reproductions.

## 9. Run record checklist

For every new experiment, preserve the following together:

- Git commit hash and configuration file;
- random seed and subject split YAML files;
- Python, dependency, CUDA, and hardware versions;
- preprocessing method and processed-data manifest;
- per-epoch training history and selected checkpoint epoch;
- held-out per-fold predictions or reports;
- per-fold metrics and their aggregation procedure;
- exact commands used for post-processing and figures;
- any failed, interrupted, or excluded runs and the reason for exclusion.

Keeping this record makes it possible to distinguish an exact reproduction from a method change
that requires a new result set.
