# Results

Archived output for every experiment reported in the paper. Each run directory holds
per-fold metrics and a classification report; the Markdown files are the analyses built
from them. Nothing here is written by hand — every file is produced by a script in
`scripts/` or at the repo root, so the numbers in the paper can be regenerated from the
checkpoints. Commands are in `REPRODUCIBILITY.md`.

## Run directories

| Directory | What it is |
|---|---|
| `run_a_baseline`, `run_b_context`, `run_c_depth` | Sleep-EDF-20 configuration ablations: 60 s, 120 s, and 120 s with a fourth TCN block |
| `run_d_noncausal`, `run_d2_noncausal` | Sleep-EDF-20 non-causal controls |
| `trimmed`, `baseline_untrimmed` | Preprocessing ablation — wake trimming and the weighted sampler |
| `sleep78_causal{,_s43,_s44}` | Sleep-EDF-78, epoch-normalised pipeline, three seeds |
| `sleep78_noncausal{,_s43,_s44}` | Matched non-causal arm, same pipeline and seeds |
| `sleep78_streaming_causal{,_s43,_s44}` | Sleep-EDF-78, causal trailing-window normalisation |
| `sleep78_streaming_noncausal{,_s43,_s44}` | Matched non-causal arm, leak-free pipeline |

Each contains `test_metrics_summary.csv` (one row per fold), `fold_N_metrics.csv`,
and `fold_N_test_report.txt`. The streaming directories also carry `split_fold_N.yaml`
recording which subjects were held out and under which seed.

## Analyses

| File | Written by | Contents |
|---|---|---|
| `statistics.md` | `analysis_stats.py` | The causality ablation on the epoch-normalised pipeline, Sleep-EDF-20 and -78 |
| `statistics_streaming.md` | `analysis_stats.py` | The same ablation on the leak-free pipeline, seed 42 |
| `statistics_streaming_pooled.md` | `scripts/pool_seeds.py` | Pooled over three seeds with the Nadeau–Bengio correction — **the figure reported in the paper** |
| `per_class_breakdown.md` / `.csv` | `scripts/per_class_breakdown.py` | Where the causal cost falls, by sleep stage, with Holm correction |
| `seed_overlap.md` | `scripts/seed_overlap_simulation.py` | How much the subject partition moved between seeds, which is why the pooled test needs correcting |
| `smoothing.md` | `sweep_smoothing.py` | Trailing-window mode filter sweep and the resulting hypnogram fragmentation |
| `causality_verification.md` | `scripts/verify_causality.py` | Perturbation test on the full path from raw sample to logit, with a non-causal negative control |
| `latency_remeasurement.md` | `scripts/latency_remeasurement.py` | CPU inference latency, median over 1000 timed calls with the thread count pinned |

## Which numbers are in the paper

Paper v9 reports the epoch-normalised pipeline: `statistics.md` and `sleep78_causal*`.
The leak-free re-run — `statistics_streaming_pooled.md` and `sleep78_streaming_*` — was
completed after v9 and supersedes it. Both are kept so that every published number stays
verifiable while the corrected numbers move forward. `BRANCH_NOTES.md` covers what changed
and why.

`smoothing.md` is still measured on the epoch-normalised pipeline; the leak-free re-sweep
needs the saved prediction files and is outstanding.
