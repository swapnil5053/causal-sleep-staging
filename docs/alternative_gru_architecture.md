# Second architecture: causal MRCNN-GRU

## Purpose

The headline result is measured on one sequence encoder: a dilated causal TCN followed by
masked self-attention. The mechanism behind it, that tiling a recording with independent
context windows repeatedly destroys accumulated history and scores the model on the
wreckage, predicts the effect for any model whose predictions depend on that history. One
architecture cannot demonstrate that. This adds a structurally different encoder so the
claim can be tested rather than asserted.

Everything outside the sequence encoder is unchanged: the data, the input channel, the
one-second output rate, the class set, the trailing 30-second normalizer, the loss, the
folds, the split seed and the optimizer settings.

```text
100 Hz EEG
  -> two-branch causal MRCNN, kernels 50 and 400, left-aligned pooling to 1 Hz
  -> two-layer GRU
  -> layer normalization
  -> five-class linear classifier, one prediction per second
```

## Causality

The causal arm uses a unidirectional GRU. Its hidden state at second t is a function of the
input sequence through t only, so it has no future-state pathway and needs no mask. The
MRCNN front end stays left-padded through `CausalConv1d`, as in the baseline.

`scripts/verify_causality.py` covers the whole path for this architecture, not just the
network. Run against `configs/sleep78_streaming_gru.yaml` it perturbs the raw signal from
second 120 onward and requires every earlier logit to be bit-identical, checks the offline
normalizer against the sample-at-a-time streaming filter, and confirms the control does
leak so the ablation is a real contrast. All seven checks pass; the report is written to
`results/causality_verification_gru.md`.

## The matched control

The non-causal twin is a bidirectional GRU. This is where the recurrent case differs from
the convolutional one and the difference has to be stated rather than glossed.

For the TCN encoder, causality is a padding flip and a mask flip, so the two arms are
literally the same network with the same parameters. A recurrent encoder has no such
switch: the non-causal twin necessarily adds a second direction, which at equal hidden
size doubles the recurrent weights and doubles the classifier input width.

| Model | Parameters |
|---|---:|
| TCN plus attention, either arm | 30,757 |
| Causal GRU, hidden 32 | 20,197 |
| Bidirectional GRU, hidden 32 | 39,237 |
| Bidirectional GRU, hidden 19, used as the control | 20,335 |

Running the control at hidden 32 would confound the causality comparison with a factor of
1.94 in capacity. The control therefore runs at hidden 19 per direction, which matches the
causal arm to 0.68%. The two arms are matched on parameter count rather than being
identical, and results should say so.

This affects the between-arm causality cost only. The decisive statistic, the excess
protocol gain to the causal arm, is a within-arm quantity: the same trained weights scored
on the same held-out seconds under two protocols. It does not depend on the two arms being
comparable to each other.

## Configuration

The six configs are generated, not written by hand:

```powershell
python scripts\make_gru_configs.py
```

Each GRU config is derived from the matching TCN streaming config, copying every key
outside the model block, so both architectures share data, folds, split seed, schedule and
optimizer by construction. The script refuses to write anything if the two arms drift more
than one percent apart on parameter count.

| Setting | Value |
|---|---|
| Dataset | Sleep-EDF-78, Fpz-Cz at 100 Hz |
| Normalization | trailing 30-second z-score |
| Context | 120 seconds, one prediction per second |
| MRCNN channels | 16 plus 16 |
| GRU | 2 layers, hidden 32 causal, hidden 19 bidirectional control |
| Cross-validation | subject-wise 5-fold, seeds 42, 43 and 44 |

## Running it

Both arms are trained on the same folds and then scored under both evaluation protocols
from the same weights.

```powershell
python -m src.train.train --config configs\sleep78_streaming_gru.yaml --fold -1
python -m src.eval.evaluate --config configs\sleep78_streaming_gru.yaml --fold 0
python -m src.eval.evaluate --config configs\sleep78_streaming_gru.yaml --fold 0 --stream_stride 30
```

Repeat for folds 1 through 4 and for the control config, then compute the excess with
`scripts/protocol_excess.py`. That script reproduces the published TCN numbers on both
datasets, so it can be trusted on a new one.

## What counts as a result

The prediction is that the excess protocol gain to the causal arm reproduces at roughly
the size measured for the TCN encoder, +0.0314 on Sleep-EDF-78, with every fold positive.
A recurrent encoder has unbounded rather than fixed-width memory, so if anything it should
be penalised at least as heavily at buffer positions carrying little history.

A null or much smaller result is a finding and is reported as one. It would narrow the
mechanism claim from any model that accumulates history to something closer to models with
a fixed finite receptive field, and the papers would have to say so. The per-quartile
kappa table is the diagnostic that separates those two cases.

Absolute kappa for this architecture should not be placed beside the TCN's as an
architecture comparison. At 20,197 against 30,757 parameters they are not a matched pair,
and no claim in the papers needs that comparison.
