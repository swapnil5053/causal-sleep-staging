# Alternative Architecture: Causal MRCNN-GRU

## Purpose

The repository's original model uses a multi-resolution CNN, a dilated TCN, and causal
self-attention. This branch adds a structurally different sequence model so the causal result
can be tested for architectural dependence rather than presented as evidence from one model
family only.

The alternative keeps the data pipeline, input channel, one-second output rate, class set,
normalization, loss, folds, and optimizer settings unchanged. Only the temporal sequence encoder
changes:

```text
100 Hz EEG
  -> two-branch causal MRCNN
  -> one-Hz feature sequence
  -> two-layer unidirectional GRU
  -> layer normalization
  -> five-class classifier
```

The MRCNN front end remains shared in spirit with the baseline: a 0.5-second branch captures
short events and a 4-second branch captures slower waves. After pooling to one feature vector per
second, the GRU processes the sequence recurrently from left to right.

## Causality design

For the causal architecture, the GRU is unidirectional. Its hidden state at second `t` depends
only on the input sequence through second `t`; it has no future-state pathway and requires no
attention mask. The MRCNN remains left-padded through the existing `CausalConv1d` implementation.

The matched control uses the same GRU hidden size, layer count, dropout, front end, classifier,
data, and training settings, but changes the recurrent encoder to a bidirectional GRU. This
explicitly exposes future context and makes the alternative architecture's causality cost
measurable in the same way as the original model's control.

## Configuration

The causal run is defined in `configs/sleep78_streaming_gru.yaml` and the control in
`configs/sleep78_streaming_gru_noncausal.yaml`. Both use:

| Setting | Value |
|---|---:|
| Dataset | Sleep-EDF-78 |
| Input | Fpz-Cz, 100 Hz |
| Normalization | trailing 30-second z-score |
| Context | 120 seconds |
| MRCNN channels | 16 + 16 |
| GRU layers | 2 |
| GRU hidden size | 32 |
| Output | 1 prediction per second |
| Classes | W, N1, N2, N3, REM |
| Cross-validation | subject-wise 5-fold |

The alternative has no reported accuracy or kappa in this branch because training was not run.
Those values must be generated from the paired configurations, not inferred from the original
architecture's results.

## Reproducible experiment

After installing `requirements.txt` and preparing the processed streaming data, run a smoke test
and then train/evaluate the same folds for both arms:

```bash
python smoke_test.py configs/sleep78_streaming_gru.yaml
python -m src.train.train --config configs/sleep78_streaming_gru.yaml --fold -1
python -m src.eval.evaluate --config configs/sleep78_streaming_gru.yaml --fold 0

python smoke_test.py configs/sleep78_streaming_gru_noncausal.yaml
python -m src.train.train --config configs/sleep78_streaming_gru_noncausal.yaml --fold -1
python -m src.eval.evaluate --config configs/sleep78_streaming_gru_noncausal.yaml --fold 0
```

Repeat evaluation for folds 1 through 4, or use the branch's streaming evaluation options when
running from the provenance-aware branch. Keep the split seed fixed between the causal and
non-causal arms. Report per-second metrics, 30-second majority-vote metrics, per-class F1, model
parameter count, and latency under the same timing protocol.

## Publication value

This addition supports a stronger claim only after the experiments are completed and reported:

> The measured cost of end-to-end causal inference is not specific to the original TCN-attention
> encoder if the same direction and approximate effect are reproduced by a recurrent encoder.

The comparison should report both positive and negative findings. If the GRU changes the size or
class distribution of the causality penalty, that is scientifically useful: it shows which part
of the cost is architecture-dependent. The branch therefore adds a complementary model and a
matched control, not a preselected result.

## Validation coverage

`tests/test_gru_architecture.py` checks:

- Output shape and a compact parameter budget.
- No future-input influence on earlier outputs for the causal GRU.
- Future-input influence on earlier outputs for the bidirectional control.

The test suite requires PyTorch. The model wiring also rejects unknown architecture names rather
than silently falling back to the baseline.