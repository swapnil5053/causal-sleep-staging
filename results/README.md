# Results index

Every number reported anywhere in this repository traces back to a file in this directory, and
every file here was written by a script rather than by hand. `runs/reproduce.ps1` regenerates all
of it from the committed per-fold CSVs without retraining.

Start from [`../RESULTS.md`](../RESULTS.md), which states each result and links to the file that
produced it. This page is the reverse index: given a file, what it is.

## Reports

| File | What it holds |
|---|---|
| [`protocol_excess_sleep78.md`](protocol_excess_sleep78.md) | The protocol excess on Sleep-EDF-78 with the convolutional model, and the cost of causality under each protocol |
| [`protocol_excess_dodh.md`](protocol_excess_dodh.md) | The same on DOD-H |
| [`protocol_excess_gru_pooled.md`](protocol_excess_gru_pooled.md) | The same on Sleep-EDF-78 with the recurrent model |
| [`statistics.md`](statistics.md) | The causality ablation on both preprocessing regimes, the per-class decomposition, the seed and split overlap that motivates the corrected test, and the smoothing sweep |
| [`statistics_dodh.md`](statistics_dodh.md) | Pooled causality statistics for DOD-H |
| [`statistics_gru_tiled_pooled.md`](statistics_gru_tiled_pooled.md) | Pooled causality statistics for the recurrent arms |
| [`verification.md`](verification.md) | The causality perturbation test with its negative control, and CPU inference latency |
| [`causality_verification.md`](causality_verification.md) | The layer-by-layer causality checks for the convolutional architecture, from `scripts/verify_causality.py` |
| [`causality_verification_gru.md`](causality_verification_gru.md) | The same checks for the recurrent architecture |
| [`predictions.md`](predictions.md) | Per-class breakdown by protocol, subject-level bootstrap |
| [`per_class_breakdown.md`](per_class_breakdown.md) | Per-class kappa and F1 for the two arms, with the paired CSV beside it |
| [`seed_overlap.md`](seed_overlap.md) | Train-set overlap between seed repeats, which is why the corrected test is used |
| [`gain_vs_strength.md`](gain_vs_strength.md) | Protocol gain against absolute score across every arm on disk, causal and control reported apart |
| [`statistics_streaming.md`](statistics_streaming.md), [`statistics_streaming_pooled.md`](statistics_streaming_pooled.md) | Causality statistics under the streaming protocol, per seed and pooled |
| [`smoothing.md`](smoothing.md) | The trailing-window smoothing sweep on the earlier pipeline, superseded by the section of the same name in `statistics.md` |
| [`boundary_latency.md`](boundary_latency.md) | How quickly each system reports a stage change, Sleep-EDF-78 |
| [`boundary_latency_dodh.md`](boundary_latency_dodh.md) | The same on DOD-H |

## Accuracy against permitted latency

One sweep yields accuracy at every buffer position, which is the mechanism evidence. Each report
has a `.csv` beside it holding the full curve.

| File | Arm, dataset, stride |
|---|---|
| [`latency_causal_s5.md`](latency_causal_s5.md) | Causal, Sleep-EDF-78, stride 5 |
| [`latency_noncausal_s5.md`](latency_noncausal_s5.md) | Non-causal, Sleep-EDF-78, stride 5 |
| [`latency_causal_wide_s5.md`](latency_causal_wide_s5.md) | Causal at 3.1x parameters, Sleep-EDF-78, stride 5 |
| [`latency_dodh_causal.md`](latency_dodh_causal.md) | Causal, DOD-H |
| [`latency_dodh_noncausal.md`](latency_dodh_noncausal.md) | Non-causal, DOD-H |
| [`latency_causal.md`](latency_causal.md), [`latency_noncausal.md`](latency_noncausal.md) | Sleep-EDF-78 at stride 30, superseded by the stride-5 pair |
| `lat_*_b240.md`, `lat_*_b360.md` | Buffers longer than the trained context, a deliberate train and test mismatch, reported as one |

Use the stride-5 files for Sleep-EDF-78. The unsuffixed files are stride 30, and mixing strides
across datasets is not a fair comparison.

## Run directories

Each holds `test_metrics_summary.csv` with one row per fold, `fold_N_metrics.csv`,
`fold_N_test_report.txt`, and where the run predates neither, `split_fold_N.yaml` recording which
subjects were held out under which seed. Directories with a `_streaming30` suffixed summary were
scored under both protocols.

| Directory | What it is |
|---|---|
| `sleep78_streaming_causal{,_s43,_s44}` | Sleep-EDF-78, convolutional, causal, trailing-window normalisation, three seeds |
| `sleep78_streaming_noncausal{,_s43,_s44}` | The matched non-causal arm |
| `dodh_causal_s4{2,3,4}` | DOD-H, convolutional, causal, three seeds |
| `dodh_noncausal_s4{2,3,4}` | The matched non-causal arm |
| `gru_s4{2,3,4}`, `gru_noncausal_s4{2,3,4}` | Sleep-EDF-78, recurrent, both arms, three seeds |
| `causal_wide_s42` | Sleep-EDF-78, convolutional, causal, 3.1x parameters, one seed: the capacity check |
| `sleep78_causal{,_s43,_s44}`, `sleep78_noncausal{,_s43,_s44}` | The earlier epoch-normalised pipeline, kept so published numbers stay verifiable |
| `run_a_baseline`, `run_b_context`, `run_c_depth` | Sleep-EDF-20 configuration ablations: 60 s context, 120 s, and 120 s with a fourth block |
| `run_d_noncausal`, `run_d2_noncausal` | Sleep-EDF-20 non-causal controls |
| `trimmed`, `baseline_untrimmed` | Preprocessing ablation: wake trimming and the weighted sampler |

`dodh_preprocessing_manifest.json` records how the DOD-H recordings were read, resampled and
scored to consensus.

## Baselines

Three published models reproduced in-house, with their settings, per-fold numbers and the defects
that bound what each run can be cited for. See [`baselines/README.md`](baselines/README.md).

## Which numbers supersede which

The `sleep78_causal*` and `sleep78_noncausal*` directories use per-epoch z-scoring, which reads an
early sample's own future when normalising it. That pipeline was replaced by the trailing-window
normaliser in the `sleep78_streaming_*` directories, which raised kappa on both arms rather than
lowering it.

Both are kept. The earlier directories make every previously published number verifiable; the
streaming directories are what current results report.

`results/generated/` is where the scripts write when rerun. It is not tracked, so a regenerated
report can be compared against the committed copy here without touching it.
