# Results

Archived output for every experiment reported in the paper. Each run directory holds
per-fold metrics and a classification report; the two Markdown files are the analyses built
from them. Every number traces back to a committed CSV — nothing here is typed by hand.
Commands to regenerate any of it are in `REPRODUCIBILITY.md`.

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

Each contains `test_metrics_summary.csv` (one row per fold), `fold_N_metrics.csv` and
`fold_N_test_report.txt`. The streaming directories also carry `split_fold_N.yaml`
recording which subjects were held out and under which seed.

## Analyses

| File | Contents |
|---|---|
| `statistics.md` | The causality ablation on both preprocessing regimes, the per-class decomposition, the seed/split overlap that motivates the corrected test, and the smoothing sweep |
| `verification.md` | The causality perturbation test with its negative control, and CPU inference latency |
| `per_class_breakdown.csv` | Per-fold per-class F1 behind the decomposition table |

Raw script output, when regenerated, is written to `results/generated/`.

## Which numbers are in the paper

Paper v9 reports the epoch-normalised pipeline: the Sleep-EDF-78 rows in `statistics.md`
and the `sleep78_causal*` directories. The leak-free re-run — the pooled three-seed table
and `sleep78_streaming_*` — was completed after v9 and supersedes it. Both are kept so that
every published number stays verifiable while the corrected numbers move forward.
`BRANCH_NOTES.md` covers what changed and why.
