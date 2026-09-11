# Reproducing the experiments

This guide maps the repository's configurations, commands and archived outputs. Run commands from
the repository root. Neither dataset is distributed with this repository.

Two preprocessing regimes exist here and they are never mixed. The current line of work uses a
trailing 30-second z-score, computable one sample at a time, which is end to end causal. An
earlier set of runs used per-epoch z-scoring, which normalises an early sample with statistics
over its whole epoch and therefore reads that sample's future. Those runs are kept so previously
published numbers stay verifiable, and they are marked as such throughout. Closing the leak raised
kappa on both arms, so it was a modelling gain rather than a trade.

For the fastest path to every reported number without retraining anything, see section 8.

## Contents

1. [Environment](#1-environment)
2. [Data acquisition](#2-data-acquisition)
3. [Preprocessing](#3-preprocessing)
4. [Fast pre-run checks](#4-fast-pre-run-checks)
5. [Training](#5-training)
6. [Held-out evaluation, both protocols](#6-held-out-evaluation-both-protocols)
7. [Analysis, statistics and figures](#7-analysis-statistics-and-figures)
8. [Whole experiments in one command](#8-whole-experiments-in-one-command)
9. [Experiment provenance](#9-experiment-provenance)
10. [Run record checklist](#10-run-record-checklist)

Design decisions that affect every run are documented after section 10: the preprocessing
manifest, the separation of the partition seed from the initialisation seed, signal continuity,
and the normaliser cold start.

## 1. Environment

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA training install the PyTorch build appropriate for the machine before the remaining
requirements. Record the Python, PyTorch, CUDA, GPU, operating-system and dependency versions
alongside a new experiment; these affect numerical reproducibility.

A full fold takes about 16 minutes on an RTX 4060 laptop GPU and hours on CPU.

## 2. Data acquisition

### Sleep-EDF Expanded

The sleep-cassette subset from PhysioNet. For the complete 78-subject set:

```bash
aws s3 sync --no-sign-request \
  s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/
```

For the 20-subject subset used by the configuration ablations, restrict to subjects 00 to 19:

```bash
aws s3 sync --no-sign-request --exclude "*" --include "SC40*" --include "SC41*" \
  s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/
```

A valid download contains paired `*-PSG.edf` and `*-Hypnogram.edf` files. The loader matches
recordings to annotations by subject prefix, so the scorer letters in hypnogram filenames need no
renaming.

Sleep-EDF Expanded carries its own Open Data Commons Attribution License.

### DOD-H

Dreem Open Datasets, 25 subjects, channel F3-M2 at 250 Hz. `scripts/dod_preprocessing.py`
documents acquisition and handles the whole path: the h5 reader, resampling to 100 Hz with a
polyphase anti-aliasing filter, the not-scored code mapped onto the existing dropped-epoch logic,
and the five-scorer consensus.

```bash
python scripts/dod_preprocessing.py --help
```

Validate on two or three records before processing the cohort. Records are streamed to `.npz` one
at a time rather than held in memory; the cohort does not fit in 16 GB.

DOD-H is not wake-trimmed, because Wake is already 12.3% of that corpus against 68% in untrimmed
Sleep-EDF.

## 3. Preprocessing

Each configuration names its own `processed_dir`, so the regimes cannot collide.

```bash
# current line of work, trailing-window normalisation, 78 subjects
python -m src.data.preprocessing --config configs/sleep78_streaming_causal.yaml --all

# earlier epoch-normalised runs, 78 subjects
python -m src.data.preprocessing --config configs/sleep78_causal.yaml --all

# 20-subject configuration ablations
python -m src.data.preprocessing --config configs/default.yaml --all
```

Preprocessing is performed once per regime and both arms of a comparison read the same arrays.

Each output file is named `subject_<ID>.npz` and contains:

- `x`: normalised EEG, `float32`, shape `(number_of_seconds, 100)`
- `y`: integer labels, `int64`, shape `(number_of_seconds,)`
- `segment_starts`: the second indices where a fresh continuous stretch begins
- `normalization_method`, `normalization_window_seconds`, `normalization_eps`,
  `wake_trim_minutes`, `resample_rate`: the preprocessing that produced the file

The loader reads that metadata from every file and refuses to build a dataset from a directory
mixing regimes, so a partially reprocessed directory fails at the start of training rather than
producing an unreportable number.

## 4. Fast pre-run checks

Seconds, and no data required:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py configs/sleep78_streaming_causal.yaml
python scripts/verify_causality.py --config configs/sleep78_streaming_causal.yaml
```

The tests check tensor dimensions, parameter counts, layer-level causality for both
architectures, subject split integrity, and in `tests/test_end_to_end_causality.py` the causality
of the composed normalisation and model path, including that the offline arrays equal the online
sample-at-a-time filter and that the non-causal arm genuinely does leak.

For a run-specific, archivable proof against real weights:

```bash
python scripts/verify_causality.py \
  --config configs/sleep78_streaming_causal.yaml \
  --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth \
  --subject data/processed78_streaming/subject_00.npz \
  --out results/generated/causality_verification.md
```

Re-run it after any change to normalisation, padding or masking, and archive the report with the
run. The end to end causal claim rests on it. Reports for both architectures are at
`results/causality_verification.md` and `results/causality_verification_gru.md`.

Before committing to a full sweep, train and score one fold and compare it against the archived
fold 0 for the same configuration. A drop of a few thousandths is expected and reportable; a
collapse means something needs revisiting before spending the rest of the compute.

## 5. Training

```bash
python -m src.train.train --config configs/sleep78_streaming_causal.yaml --fold 0    # one fold
python -m src.train.train --config configs/sleep78_streaming_causal.yaml --fold -1   # all five
```

Training re-seeds per fold, so running folds individually and running `--fold -1` give identical
results. Every fold saves the subject split in `split_fold_<N>.yaml`, the best checkpoint in
`best_model_fold_<N>.pth`, and the per-epoch history in `fold_<N>_metrics.csv`.

The validation fold selects the checkpoint on the configured metric, kappa by default. The test
fold is used for none of parameter updates, early stopping or checkpoint selection.

To reproduce another experiment, change only the configuration path. Do not edit a shared
configuration during a run.

### The recurrent architecture

`configs/sleep78_streaming_gru*.yaml` are generated rather than hand-written, so that everything
outside the model block is copied from the matching convolutional config and the two architectures
cannot silently diverge on data, folds, split seed or schedule:

```bash
python scripts/make_gru_configs.py
```

It prints the parameter count of each arm and refuses to write anything if the causal arm and its
bidirectional control drift more than one percent apart. A bidirectional GRU at equal hidden size
would carry 1.94 times the parameters, so the control runs at a reduced per-direction width.

### The capacity check

```bash
python scripts/make_capacity_configs.py
```

Generates a wider convolutional causal arm, 95,237 parameters against the baseline's 30,757, used
to test whether the protocol effect depends on capacity. No control arm is needed, because the
protocol gain is a within-arm quantity.

## 6. Held-out evaluation, both protocols

Every experiment is scored twice from the same weights.

**Tiled** is the default. The recording is covered by non-overlapping context windows and every
position of every window contributes one scored prediction, so the opening seconds of each window
are predicted from a partly empty history.

**Streaming** recomputes every `N` seconds against a full context buffer and keeps only the
freshest `N` predictions, so every second after the first window is decided with at least
`sequence_length - N` seconds of real history. This is what a deployment does.

```bash
for fold in 0 1 2 3 4; do
  python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --fold "$fold"
  python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --fold "$fold" --stream_stride 30
done
```

On PowerShell:

```powershell
0..4 | ForEach-Object {
    python -m src.eval.evaluate --config configs\sleep78_streaming_causal.yaml --fold $_
    python -m src.eval.evaluate --config configs\sleep78_streaming_causal.yaml --fold $_ --stream_stride 30
}
```

Streaming costs `sequence_length / N` forward passes, so four times a tiled pass at 120 s context
and stride 30. Results are written under a `_streamingN` suffix, so the two protocols sit side by
side and neither overwrites the other. With `--stream_stride` equal to `sequence_length` the two
coincide, apart from the trailing seconds tiling drops.

The evaluator appends one row per fold to `test_metrics_summary.csv`. Start with a new log
directory or delete a disposable duplicate before repeating evaluation; it appends rather than
replacing. Never delete or overwrite curated files under `results/`.

Per-second metrics compare predictions against 30-second expert labels replicated to each second.
The 30-second metrics majority-vote each group of 30 predictions, which is what compares like for
like against epoch-level systems.

### Saved predictions

Each evaluation also writes `fold_N_predictions.npz` holding the per-second subject id, expert
label, prediction and logits. Smoothing sweeps, per-subject confidence intervals and the per-class
breakdown all read from it, so none of them needs another pass over a checkpoint. Archive it with
the run. `--no_save_predictions` skips it, at the cost of having to re-evaluate later.

## 7. Analysis, statistics and figures

### The protocol excess, the decisive statistic

How much each arm gains from the protocol change, and how much larger that gain is for the causal
arm. A within-arm quantity, so it does not require the two arms to be comparable.

```bash
python scripts/protocol_excess.py \
  --seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42 \
  --seed 43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43 \
  --seed 44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44 \
  --label "Sleep-EDF-78, TCN" \
  --out results/generated/protocol_excess_sleep78.md
```

It also reports the between-arm causality cost under each protocol, computed directly from the
same folds rather than by subtracting rounded means. On the published directories it must return
+0.0314 with corrected t(14) = 12.69 for Sleep-EDF-78 and +0.0292 with t(14) = 6.48 for DOD-H;
anything else means the script disagrees with the papers and must be fixed before it is used.

### The between-arm cost, pooled across seeds

```bash
python scripts/pool_seeds.py \
  --seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42 \
  --seed 43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43 \
  --seed 44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44 \
  --out results/generated/statistics_streaming_pooled.md
```

Per-seed results use the naive paired test; the pooled figure uses the Nadeau-Bengio correction,
which inflates the variance estimate by `1/n + 1/(k-1)`, here `1/15 + 1/4`, because folds pooled
across seeds still share training subjects. Both scripts refuse to run if a summary CSV lists a
fold twice or if the two arms cover different folds, so a half-finished sweep cannot be reported
as a complete one.

`scripts/analysis_stats.py --pair NAME=causal_dir,noncausal_dir` reports any single run pair
separately.

### Accuracy against permitted latency

One sweep yields accuracy at every buffer position without retraining, which is the mechanism
evidence behind the protocol result.

```bash
python scripts/latency_sweep.py --config configs/sleep78_streaming_causal.yaml \
  --stride 5 --out results/generated/latency_causal_s5.md
```

Use stride 5 for Sleep-EDF-78. The unsuffixed archived files are stride 30 and mixing strides
across datasets is not a fair comparison.

### Reporting delay, per-second against epoch-level

```bash
python scripts/boundary_latency.py logs_78streaming_causal_s42 \
  --hold 10 --out results/generated/boundary_latency.md
```

Paths are positional. Both systems are charged for the evidence they need: a change counts as
reported once the new stage has been held for `--hold` consecutive seconds, and an epoch-level
system cannot emit the label for an epoch until that epoch has ended.

### Per-class and gain-against-capacity

```bash
python scripts/per_class_breakdown.py
python scripts/gain_vs_strength.py \
  --arm "TCN causal, Sleep-EDF-78=logs_78streaming_causal_s42" \
  --control "TCN control, Sleep-EDF-78=logs_78streaming_noncausal_s42" \
  --out results/generated/gain_vs_strength.md
```

`gain_vs_strength.py` takes causal arms and control arms separately and correlates only the causal
ones. Pooling both kinds would be circular: a control scores higher and gains less by
construction.

### Smoothing, throughput and figures

```bash
python scripts/sweep_smoothing.py --config configs/sleep78_streaming_causal.yaml \
  --out results/generated/smoothing.md
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml --benchmark
python scripts/make_paper_figures.py
python scripts/validate_results.py results
```

The benchmark's `ms/sec` is the time for a complete synthetic sequence divided by its length.
Record total window time as well, and distinguish this throughput measure from end to end
streaming latency.

### Every other script in the repository

Nothing is tracked that does not produce something reported. The remaining scripts, and what each
one is for:

| Script | Produces |
|---|---|
| `scripts/streaming_demo.py` | The sample-at-a-time device path, and the proof that it reproduces the offline labels exactly over a held-out recording. This is the evidence behind the streaming-equivalence claim |
| `scripts/analyze_predictions.py` | `results/predictions.md`: subject-level bootstrap, per-subject table, per-class breakdown by protocol |
| `scripts/seed_overlap_simulation.py` | `results/seed_overlap.md`: the train-set overlap between seed repeats that motivates the corrected test |
| `scripts/validate_results.py` | A structural check over every archived CSV, changing nothing |
| `scripts/make_paper_figures.py` | `figures/fig_latency.png` and `fig_perclass.png`, the two-panel manuscript figures |
| `scripts/run_sweep.py`, `scripts/eval_archived.py` | The matched sweep runner and archived-checkpoint evaluation that produced the three-seed streaming runs |
| `scripts/sweep_smoothing.py` | The trailing-window smoothing sweep |
| `scripts/analysis_stats.py` | `results/statistics.md`: the paired and pooled causality statistics |
| `scripts/smoke_test.py` | Model, losses, optimiser, checkpoint round-trip and metrics on random batches, in seconds |

## 8. Whole experiments in one command

The PowerShell drivers under `runs/` chain training, both evaluation protocols, output checks and
statistics for an entire experiment. Each waits for the GPU, skips anything already finished, and
refuses to compute a statistic from an incomplete run, so they are safe to leave unattended and
safe to restart.

| Driver | What it runs |
|---|---|
| `runs/run_gru_s42.ps1` | Recurrent arms, seed 42, both protocols, the excess |
| `runs/run_gru_s4344.ps1` | Seeds 43 and 44, then the three-seed pooled statistics and the buffer sweeps |
| `runs/run_capacity_check.ps1` | The wider causal arm and the gain-against-capacity table |
| `runs/reproduce.ps1` | **Every reported number, from the committed CSVs, without retraining** |
| `runs/check_repo.ps1` | Repository hygiene gate: tests, lint, links, stray artifacts |

```powershell
powershell -ExecutionPolicy Bypass -File runs\reproduce.ps1
```

That writes into `results/generated/`. Diff against the committed copies in `results/`; every
script here is deterministic given the same inputs, so a disagreement is a bug rather than a
rounding difference.

## 9. Experiment provenance

Generated log and checkpoint directories are gitignored. Curated per-fold outputs are archived
under `results/`.

### Current line of work, trailing-window normalisation

| Experiment | Configuration | Seed | Curated results |
|---|---|---:|---|
| Sleep-EDF-78, conv., causal | `configs/sleep78_streaming_causal.yaml` | 42 | `results/sleep78_streaming_causal/` |
| Sleep-EDF-78, conv., causal | `configs/sleep78_streaming_causal_s43.yaml` | 43 | `results/sleep78_streaming_causal_s43/` |
| Sleep-EDF-78, conv., causal | `configs/sleep78_streaming_causal_s44.yaml` | 44 | `results/sleep78_streaming_causal_s44/` |
| Sleep-EDF-78, conv., control | `configs/sleep78_streaming_noncausal.yaml` | 42 | `results/sleep78_streaming_noncausal/` |
| Sleep-EDF-78, conv., control | `configs/sleep78_streaming_noncausal_s43.yaml` | 43 | `results/sleep78_streaming_noncausal_s43/` |
| Sleep-EDF-78, conv., control | `configs/sleep78_streaming_noncausal_s44.yaml` | 44 | `results/sleep78_streaming_noncausal_s44/` |
| Sleep-EDF-78, recurrent, causal | `configs/sleep78_streaming_gru.yaml` | 42 | `results/gru_s42/` |
| Sleep-EDF-78, recurrent, causal | `configs/sleep78_streaming_gru_s43.yaml` | 43 | `results/gru_s43/` |
| Sleep-EDF-78, recurrent, causal | `configs/sleep78_streaming_gru_s44.yaml` | 44 | `results/gru_s44/` |
| Sleep-EDF-78, recurrent, control | `configs/sleep78_streaming_gru_noncausal.yaml` | 42 | `results/gru_noncausal_s42/` |
| Sleep-EDF-78, recurrent, control | `configs/sleep78_streaming_gru_noncausal_s43.yaml` | 43 | `results/gru_noncausal_s43/` |
| Sleep-EDF-78, recurrent, control | `configs/sleep78_streaming_gru_noncausal_s44.yaml` | 44 | `results/gru_noncausal_s44/` |
| Sleep-EDF-78, conv. wide, causal | `configs/sleep78_streaming_causal_wide.yaml` | 42 | `results/causal_wide_s42/` |
| DOD-H, conv., causal | `configs/dodh/causal_s42.yaml` | 42 | `results/dodh_causal_s42/` |
| DOD-H, conv., causal | `configs/dodh/causal_s43.yaml` | 43 | `results/dodh_causal_s43/` |
| DOD-H, conv., causal | `configs/dodh/causal_s44.yaml` | 44 | `results/dodh_causal_s44/` |
| DOD-H, conv., control | `configs/dodh/noncausal_s42.yaml` | 42 | `results/dodh_noncausal_s42/` |
| DOD-H, conv., control | `configs/dodh/noncausal_s43.yaml` | 43 | `results/dodh_noncausal_s43/` |
| DOD-H, conv., control | `configs/dodh/noncausal_s44.yaml` | 44 | `results/dodh_noncausal_s44/` |

Every row above uses 120 s of context and is scored under both protocols. The Sleep-EDF rows read
`data/processed78_streaming`.

### Earlier runs, per-epoch z-scoring, kept for verifiability

| Experiment | Configuration | Seed | Curated results |
|---|---|---:|---|
| Sleep-EDF-78 causal | `configs/sleep78_causal{,_s43,_s44}.yaml` | 42, 43, 44 | `results/sleep78_causal{,_s43,_s44}/` |
| Sleep-EDF-78 non-causal | `configs/sleep78_noncausal{,_s43,_s44}.yaml` | 42, 43, 44 | `results/sleep78_noncausal{,_s43,_s44}/` |
| Sleep-EDF-20 baseline, 60 s | `configs/default.yaml` | 42 | `results/run_a_baseline/` |
| Sleep-EDF-20 longer context | `configs/run_b_context.yaml` | 42 | `results/run_b_context/` |
| Sleep-EDF-20 extra depth | `configs/run_c_depth.yaml` | 42 | `results/run_c_depth/` |
| Sleep-EDF-20 non-causal, baseline depth | `configs/run_d2_noncausal.yaml` | 42 | `results/run_d2_noncausal/` |
| Sleep-EDF-20 non-causal, extra depth | `configs/run_d_noncausal.yaml` | 42 | `results/run_d_noncausal/` |

These read `data/processed78` and `data/processed`. They are not mixed with the streaming runs in
any directory or any table.

### Not one-command reproductions

`configs/sleep78_ctx300.yaml` and `configs/sleep78_streaming_causal_w15.yaml` and `_w120.yaml`
define context-length and normalisation-window variants with no matching curated result
directory. `results/trimmed/` and `results/baseline_untrimmed/` are preprocessing-ablation
archives predating the current configuration layout. Treat all of these as supporting artifacts.

Baseline reproductions of three published models are documented separately in
[`results/baselines/README.md`](results/baselines/README.md), including the defects that bound
what each run can be cited for.

## 10. Run record checklist

For every new experiment, preserve together:

- the git commit hash and the configuration file
- the random seed, the split seed, and the subject split YAML files
- Python, dependency, CUDA and hardware versions
- the preprocessing method and the processed-data manifest
- the per-epoch training history and the selected checkpoint epoch
- held-out per-fold predictions or reports, under both protocols
- per-fold metrics and their aggregation procedure
- the exact commands used for post-processing and figures
- any failed, interrupted or excluded run, and why it was excluded

Keeping this record is what distinguishes an exact reproduction from a method change that requires
a new result set.

---

## Design decisions that affect every run

### The preprocessing manifest

Every preprocessing run writes `preprocessing_manifest.json` into its `processed_dir`, recording
the config, the normalisation method and window, the number of PSG files found, paired, written
and failed, the class distribution, and per-subject second counts. Archive it with the run. It is
the evidence for the data half of the run record, and it is how a reviewer confirms which
normalisation produced a given result set. The DOD-H equivalent is
`results/dodh_preprocessing_manifest.json`.

### Separating the partition from the initialisation

`train.seed` seeds the weight initialisation and, historically, also the subject shuffle that
decides the folds. Seeds 42, 43 and 44 therefore produced three different partitions rather than
three initialisations of one partition, and in those runs the two effects cannot be separated:
they are three repeats of five-fold cross-validation over fifteen distinct partitions. Analyses of
that data must account for train-set overlap between repeats rather than treating the fifteen
cells as independent, which is exactly what the Nadeau-Bengio correction does.

`train.split_seed` pins the partition independently:

```yaml
train:
  seed: 43           # weight initialisation
  split_seed: 42     # fold layout; hold fixed to vary only the initialisation
```

The two sit together so it is visible that they are separate knobs. `data.split_seed` is accepted
as an alias and `train.split_seed` wins if both are present, so do not set both. Omitting
`split_seed` uses `train.seed`, so every configuration written before this option existed produces
exactly the splits it always did. Each `split_fold_N.yaml` records both seeds alongside the subject
lists, so the partition behind a result is recoverable from the run itself.

### Continuity of the stored signal

Preprocessing drops unscored epochs and concatenates a subject's two nights, so a stored array is a
sequence of discontinuous stretches rather than one recording. Each `.npz` carries
`segment_starts`, and the manifest reports how many segments each subject has.

| Option | Effect |
|---|---|
| `data.respect_boundaries` | Drop context windows spanning a night join or an unscored-epoch gap |
| `data.cover_tail` | Add a final window flush with the end of the recording rather than discarding the trailing `(len - seq_len) % stride` seconds |

Both are off by default so the archived windowing is unchanged. `respect_boundaries` needs
`segment_starts` and raises rather than silently doing nothing on directories processed before that
metadata existed.

### The normaliser resets at every recording

`StreamingZScore` is reset at the start of each recording, including between a subject's two
nights. This is deliberate, since a device powering on has no history either, but it means the
first seconds of every night are normalised against a partial window rather than a full 30 s one,
and that is worth one sentence in a methods section rather than leaving a reviewer to find it.

It is safe rather than merely tolerable. Within a trailing window holding `n` samples the largest
attainable magnitude is `sqrt(n - 1)`, so the cold start cannot produce an infinity or a NaN
however few samples have been seen, and `causal_rolling_zscore` returns 0 while the standard
deviation is below `eps`. `scripts/verify_causality.py` checks that bound on every run and reports
the measured maximum against it, and
`tests/test_end_to_end_causality.py::test_normalized_output_is_finite_including_warmup` asserts it
directly.

### BatchNorm during training

The convolutional front end uses batch normalisation, which pools over the time axis within a
window during training. Inference uses running statistics, so the deployed path is causal and the
perturbation test passes in evaluation mode, but the learned weights are not independent of
within-window future samples. Both arms share this property, so it does not affect any comparison
reported here. It is stated because a careful reviewer will look for it.
