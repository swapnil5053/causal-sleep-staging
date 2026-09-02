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

| Config | Seed | Split seed | Causal | Normalization |
|---|---:|---:|:---:|---|
| `configs/sleep78_streaming_causal.yaml` | 42 | 42 | yes | trailing 30 s |
| `configs/sleep78_streaming_noncausal.yaml` | 42 | 42 | no | trailing 30 s |
| `configs/sleep78_streaming_causal_s43.yaml` | 43 | 42 | yes | trailing 30 s |
| `configs/sleep78_streaming_noncausal_s43.yaml` | 43 | 42 | no | trailing 30 s |
| `configs/sleep78_streaming_causal_s44.yaml` | 44 | 42 | yes | trailing 30 s |
| `configs/sleep78_streaming_noncausal_s44.yaml` | 44 | 42 | no | trailing 30 s |
| `configs/sleep78_streaming_causal_w120.yaml` | 42 | 42 | yes | trailing 120 s (window robustness, optional) |

The split seed is now pinned across every row, so the seed column varies initialisation alone.
The archived runs predate that separation and used the seed for both; see section 5.

Order of work:

```bash
# 0. proofs and sanity checks, seconds, no data required
python -m unittest discover -s tests -v
python smoke_test.py configs/sleep78_streaming_causal.yaml
python scripts/verify_causality.py --config configs/sleep78_streaming_causal.yaml
python scripts/streaming_demo.py --synthetic --verify

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

Training writes `split_fold_<N>.yaml` to `checkpoint_dir`, not `log_dir`, so the curated
directories under `results/` did not originally receive them and the checkpoints they sat
beside are gitignored. They have been recovered: every `fold_<N>_test_report.txt` records its
held-out subjects, and `get_cv_splits` is deterministic, so the full partition regenerates from
the run's seed.

```bash
python scripts/recover_splits.py --check     # verify against the archived reports
python scripts/recover_splits.py             # write the yaml files
```

All six archived streaming runs reproduce every fold's test set exactly, and the recovered
files are committed alongside their reports. Nothing is written unless the reconstruction
matches, and an existing file is never overwritten without `--force`: a plausible-looking split
that is not the one the model was trained on would be worse than no file at all. Each file
records the seed that produced the partition, which for these runs is the training seed because
they predate `train.split_seed`.

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
  `wake_trim_minutes`, `resample_rate`: the preprocessing that produced the file;
- `segment_starts`: second offsets at which the recording is discontinuous.

Unscored epochs are dropped and a subject's two nights are concatenated with the daytime
between them removed, so consecutive rows of `x` are not always consecutive in time.
`segment_starts` records every such join, and `SleepDataset` refuses windows that span one - a
window covering a join asks the model to read across a jump that never happened. Files written
before this metadata existed carry none and are read as a single continuous recording, so
archived runs reproduce exactly; the dataset reports how many such subjects it saw. Pass
`respect_segments=False` to restore the old behaviour deliberately.

Normalization resets at the start of every recording, including between a subject's two nights.
That is intentional - it is what a device does when it is switched on - and it means the first
window of each night runs on a partial trailing statistic. The output stays bounded:
`scripts/verify_causality.py` checks that no normalized value exceeds `sqrt(window - 1)`, so the
cold start cannot produce an outlier, only a slightly noisier statistic.

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

`tests/test_segment_boundaries.py` covers discontinuity handling, including the guarantee that
a file without segment metadata loads exactly as it did before. `tests/test_warm_start_eval.py`
pins the second-level bookkeeping that makes warm and cold comparable, and
`tests/test_analysis_scripts.py` checks the calibration and response-time arithmetic against
hand-constructed cases.

`tests/test_recover_splits.py` additionally asserts that every archived streaming run still
reproduces its own partition, so a change to `get_cv_splits` cannot silently invalidate the
recovered split files.

Three further suites cover the review-facing claims: `tests/test_streaming_demo.py` (the
sample-at-a-time front end reproduces both the offline arrays and `evaluate.py`'s labels, and
a future sample cannot change an emitted second), `tests/test_seed_separation.py` (every
config pins one subject partition, and `split_seed` still falls back to `seed`), and
`tests/test_subject_metrics.py` (per-subject attribution is exact, and the paired test refuses
rows that are not actually paired).

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

### Seeds: initialisation and partition are separate

`train.seed` seeds weight initialisation, shuffling and dropout. `train.split_seed` seeds the
subject partition used for cross-validation. Every configuration in this repository pins
`split_seed: 42`, so a run under a different `seed` is a genuine replication: the same folds,
the same held-out subjects, a different initialisation. Both values are recorded in each fold's
`split_fold_<N>.yaml` alongside the subject lists.

`split_seed` falls back to `seed` when a configuration omits it, which is the behaviour every
archived run was produced under. That matters when reproducing one:

| Run | Reproduce with |
|---|---|
| any seed-42 run | the config unchanged; `split_seed` and `seed` are both 42 |
| `*_s43` / `*_s44` archived runs | `--split_seed 43` / `--split_seed 44`, restoring the partition those runs actually used |

```bash
# reproduce the archived seed-43 streaming run exactly, partition included
python -m src.train.train --config configs/sleep78_streaming_causal_s43.yaml \
  --fold -1 --split_seed 43
```

Without that flag the seed-43 and seed-44 configurations now produce the seed-42 partition,
which is the intended behaviour going forward and *not* what the archived results were computed
on. This was a real defect found by code review: because one seed drove both jobs, the
three-seed sweeps varied the subject partition as well as the initialisation, so the fifteen
per-fold measurements are not fifteen replications of one experiment. Reporting them pooled
requires a correction for the reused subject pool; see section 7.

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

It also writes one row per held-out subject to `fold_<N>_subject_metrics.csv` and appends the
same rows to a run-level `test_subject_metrics.csv`: accuracy, kappa, macro F1 and per-class F1
for each night, with the number of scored seconds. A subject appears in the test set of exactly
one fold, so a five-fold sweep over 78 subjects yields 78 per-subject scores. A night the
technician scored as a single stage has no defined kappa; that field is left empty rather than
written as zero.

The per-second metrics compare predictions against 30-second expert labels replicated to each
second. The 30-second metrics majority-vote each group of 30 predictions for comparison with
conventional epoch-level systems.

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

Pooling folds across seeds is not the same as having independent measurements: the same 78
subjects are reused each time. Pairing by subject avoids the problem instead of correcting for
it, because each subject is held out in exactly one fold:

```bash
python scripts/subject_paired_test.py \
  --pair "Streaming-78 seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42" \
  --out results/statistics_streaming_by_subject.md
```

It refuses to run when the two arms cover different subjects, when a subject is listed twice,
or when a subject sits in different folds in the two arms — the last of which means the arms
were trained on different partitions and the comparison is not controlled. One seed at a time:
two seeds sharing a partition score the same subject twice, and those differences are not
independent of each other.

### Warm-started evaluation

Held-out evaluation scores non-overlapping windows, so at position 0 of each window the causal
model has one second of history while the non-causal model attends over the rest of the window.
Roughly the first quarter of every window is context-starved for one arm only, and never would
be in a streaming deployment holding a rolling buffer. Part of the measured causality cost is
therefore an artefact of the measurement.

```bash
python scripts/warm_start_eval.py   --config configs/sleep78_streaming_causal.yaml   --checkpoint_dir checkpoints_78streaming_causal_s42 --fold 0   --save_predictions logs_warm/fold_0_predictions.npz   --out results/warm_start_fold_0.md
```

It scores the same seconds twice - cold, exactly as `evaluate.py` does, and warm, keeping only
the last `--stride` predictions of overlapping windows so every scored second carries at least
`sequence_length - stride` seconds of context. The two are compared over an identical set of
seconds, so the difference is context and nothing else, and the archived cold number over all
seconds is reported alongside so nothing is quietly restated. `src/eval/evaluate.py` is not
modified, so the archived numbers cannot move.

`--save_predictions` writes per-second probabilities for both scorings, which the two analyses
below read without another forward pass:

```bash
python scripts/calibration.py --predictions logs_warm/fold_0_predictions.npz   --out results/calibration_fold_0.md --figure figures/fig_calibration.png
python scripts/transition_response.py --predictions logs_warm/fold_0_predictions.npz   --out results/transition_response_fold_0.md
```

`calibration.py` reports expected and maximum calibration error, a reliability table and the
Brier score, per stage as well as overall - what a triage system needs in order to know when it
is unsure. `transition_response.py` measures how many seconds after a scored stage change the
model follows, against the same predictions read at 30-second epoch resolution, which is the
measured answer to what per-second output buys. Its comparison is paired over transitions both
series detected, because the detection rates differ.

Demonstrate the deployed path, and check it matches the evaluation it is reported against:

```bash
python scripts/streaming_demo.py --synthetic --verify        # no data needed
python scripts/streaming_demo.py \
  --edf data/raw/SC4001E0-PSG.edf --hypnogram data/raw/SC4001EC-Hypnogram.edf \
  --config configs/sleep78_streaming_causal.yaml \
  --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth \
  --minutes 6 --verify
```

`--verify` asserts that labels emitted sample-at-a-time are identical to those
`src/eval/evaluate.py` produces over the same recording, and exits non-zero otherwise. The
equivalence does not depend on the weights, so it can be checked before a checkpoint exists.
`--mode rolling` emits from a trailing context instead of a fixed window; it deliberately does
*not* match `evaluate.py`, and the difference is the window-boundary artefact noted in the
limitations.

Generate figures (requires the processed data and referenced checkpoint for the hypnogram panel):

```bash
python make_figures.py \
  --config configs/sleep78_streaming_causal.yaml \
  --run results/sleep78_streaming_causal \
  --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth \
  --out figures
```

Those are also the defaults, so bare `python make_figures.py` reproduces the archived figures.
The confusion and ablation panels render without a checkpoint; only the hypnogram needs one,
and it is skipped with a message rather than written blank. Passing
`--run results/sleep78_causal --config configs/sleep78_causal.yaml` regenerates the older
epoch-normalised figures, which no longer match the reported numbers.

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

Every archived run in this table was produced before `train.split_seed` existed, so each used
its own seed for the subject partition as well as for initialisation. Reproducing a `_s43` or
`_s44` row exactly therefore needs `--split_seed 43` or `--split_seed 44`; see section 5.

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
