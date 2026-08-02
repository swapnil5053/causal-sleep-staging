# Model architecture and tensor flow

This document describes how a batch of single-channel EEG is transformed into one sleep-stage
prediction per second. The default model predicts five classes: Wake, N1, N2, N3, and REM.

## End-to-end tensor flow

Let:

- `B` be the batch size;
- `L` be the sequence length in seconds; and
- `C` be the number of output classes (`5`).

The configured sampling rate is 100 Hz, so each second contains 100 EEG samples.

| Stage | Operation | Output shape |
|---|---|---|
| Dataset | Load a window of `L` one-second segments | `(B, L, 100)` |
| Continuous input | Flatten the time segments into one signal | `(B, 1, L * 100)` |
| MRCNN | Extract short- and long-scale features | `(B, 32, L)` |
| TCN | Model temporal patterns with residual blocks | `(B, 32, L)` |
| Transpose | Put time before the feature dimension | `(B, L, 32)` |
| Self-attention | Combine relevant temporal context | `(B, L, 32)` |
| Classifier | Produce five unnormalised class scores | `(B, L, C)` |

The highest classifier score at each of the `L` positions is the predicted sleep stage for that
second. Softmax is not part of the model because the training loss accepts logits directly.

```text
EEG window (B, L, 100)
        |
        v
continuous EEG (B, 1, L * 100)
        |
        v
multi-resolution CNN (B, 32, L)
        |
        v
temporal convolution network (B, 32, L)
        |
        v
masked self-attention (B, L, 32)
        |
        v
linear classifier (B, L, 5)
```

## Multi-resolution feature extraction

`MultiResolutionCNN` applies two convolution branches to the same continuous EEG signal:

- A kernel of 50 samples covers 0.5 seconds at 100 Hz. It is intended to capture relatively
  short events such as sleep spindles and components of K-complexes.
- A kernel of 400 samples covers 4 seconds. It is intended to capture slower activity such as
  the waves associated with deep sleep.

Each branch produces 16 channels. Concatenating them gives 32 features at every EEG sample.
Max-pooling with a kernel and stride of 100 then reduces the representation from 100 Hz to one
feature vector per second.

## Temporal convolution network

The default TCN contains three residual blocks with dilation factors `1`, `2`, and `4`. Each
block contains two convolutions, batch normalisation, ReLU activations, dropout, and a residual
connection. Dilation lets later blocks cover a wider temporal area without using a large kernel.

The channel width remains 32 throughout the default TCN. A 1x1 projection is available in a
residual block when its input and output channel counts differ.

## Self-attention and classification

The attention layer has four heads operating on the 32-dimensional per-second features. In
causal mode it applies an upper-triangular mask: output position `t` may attend to positions up
to and including `t`, but not to positions after `t`. A residual connection and layer
normalisation follow attention.

A linear layer maps each 32-dimensional result to five logits. Training compares all per-second
logits with their corresponding labels. Evaluation selects the largest logit at every second.

## Where causality is enforced

When `model.causal: true`, the neural network prevents future model input from reaching an
earlier output in two places:

1. Every `CausalConv1d` pads only the left side by `(kernel_size - 1) * dilation`.
2. Self-attention masks entries above the main diagonal.

When `model.causal: false`, convolutions use symmetric padding and attention uses no mask. The
layer types, channel widths, and trainable parameter count otherwise remain the same, enabling
a controlled architectural comparison.

### Preprocessing boundary

The causality described above applies to the neural network given its input tensor. The current
preprocessing pipeline z-score normalises each complete 30-second epoch using that epoch's mean
and standard deviation. Consequently, a normalized value near the start of an epoch depends on
later raw samples in the same epoch. An end-to-end streaming system must replace this operation
with fixed training-set statistics or another past-only normalization method before claiming
strict causality from raw EEG through prediction.

Wake trimming and label expansion are offline dataset-construction operations. The expert label
for each 30-second epoch is repeated across its 30 one-second training targets; the dataset does
not contain independently scored one-second annotations.

## Default parameterization

The main architecture uses:

| Component | Setting |
|---|---|
| MRCNN channels | 16 short-scale + 16 long-scale |
| MRCNN kernels | 50 and 400 samples |
| Pooling factor | 100 samples (100 Hz to 1 Hz) |
| TCN channels | `[32, 32, 32]` |
| TCN kernel | 3 |
| TCN dilations | `[1, 2, 4]` |
| Attention heads | 4 |
| Output classes | 5 |
| Trainable parameters | 30,757 |

Sequence length changes the amount of context and computation but does not change the number of
trainable parameters. The default configuration uses 60 seconds, while the primary
Sleep-EDF-78 configurations use 120 seconds.

## Source map

| Responsibility | File |
|---|---|
| Integrated forward pass | `src/model/full_model.py` |
| Multi-resolution convolutions | `src/model/mrcnn.py` |
| Causal/symmetric convolution padding | `src/model/causal_conv.py` |
| Temporal residual blocks | `src/model/temporal.py` |
| Masked/unmasked self-attention | `src/model/attention.py` |
| Per-second linear classifier | `src/model/classifier.py` |
| Window construction and subject splits | `src/data/dataset.py` |
| Raw EDF preprocessing | `src/data/preprocessing.py` |
| Training loop | `src/train/train.py` |
| Held-out evaluation | `src/eval/evaluate.py` |
