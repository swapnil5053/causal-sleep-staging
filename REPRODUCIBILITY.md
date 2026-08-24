# Reproducing the experiments

This guide maps the repository's configurations, commands, and archived outputs. Run commands
from the repository root. Raw Sleep-EDF data is not distributed with this repository.

> **Causality scope:** the archived experiments use per-epoch z-score normalization calculated
> over each complete 30-second epoch. The neural network is causal with respect to its normalized
> input, but this preprocessing step uses later samples from the epoch. Preserve it only when
> reproducing the archived numbers exactly. A new end-to-end causal study should introduce a
> past-only normalization strategy and retrain all compared models.

### End-to-end causal preprocessing pilot

New experiments should use the isolated streaming configurations rather than overwrite the
archived epoch-normalized data. They apply a trailing 30-second z-score at every raw sample;
the statistic at time `t` contains only samples at or before `t` and resets at each recording.

```bash
python -m src.data.preprocessing --config configs/sleep78_streaming_causal.yaml --all
python -m src.train.train --config configs/sleep78_streaming_causal.yaml --fold 0
python -m src.train.train --config configs/sleep78_streaming_noncausal.yaml --fold 0
```

Both pilot arms read `data/processed78_streaming`, but write to separate log and checkpoint
directories. Run fold 0 first as a go/no-go check. If performance remains viable, run folds 0-4
for seed 42, then copy both configurations for seeds 43 and 44 with distinct log/checkpoint
directories. Do not describe the pipeline as end-to-end causal until the new causal checkpoints
have been evaluated and the future-perturbation test passes for the complete preprocessing-model
path.

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

- `x`: normalized EEG with shape `(number_of_seconds, 100)`;
- `y`: integer labels with shape `(number_of_seconds,)`.

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

The tests check tensor dimensions, parameter count, model-layer causality, and subject split
integrity. They do not prove end-to-end causality through the current normalization step.

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
