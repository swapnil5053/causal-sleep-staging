# How released implementations score a context window

The central claim of this work is about how sleep-staging models are scored, not about how they are
built. That claim rests on a premise: that covering a recording with non-overlapping context
windows, and scoring every position of every window, is the normal thing to do. This file records
how that premise was checked, so a reader can repeat the check rather than take it on trust.

## What to look for

A sequence model reads a context buffer of `L_buf` seconds and emits a prediction at every position
in it. Evaluation code has to decide which of those positions count. There are three patterns:

1. **Tiled.** The recording is cut into non-overlapping windows and every position of every window
   contributes one scored prediction. Each window starts from an empty history, so a fixed fraction
   of the scored predictions carry almost no accumulated past.
2. **Streaming.** The buffer advances by a stride smaller than its length and only the freshest
   predictions of each pass are kept, so every scored second after the first window carries a full
   buffer of genuine history.
3. **Trimmed.** Positions below a history threshold are discarded rather than scored.

The question for any repository is which of the three its evaluation loop implements. In practice
it is settled by two things: how the test set is windowed, and whether the loop keeps all
`L_buf` predictions per window or a subset.

## How to check one repository

1. Find the evaluation entry point, not the training loop. Scoring decisions live there.
2. Find where the test recording is turned into windows. A stride equal to the window length is
   tiling; a stride smaller than the window length is overlap, and then step 3 decides.
3. Find where predictions are collected for the metric. If every element of the model output is
   appended, the protocol is tiled. If the loop slices the output before appending, read the slice
   bounds: that is either streaming or trimming.
4. Record the commit hash you read, because the answer can change between versions.

## What was checked

| Implementation | Protocol found | Where | Commit / version | Date |
|---|---|---|---|---|
| This work, pre-September 2026 | Tiled | `src/eval/evaluate.py`, all `L_buf` positions appended per window | see repository history before `v1.0.0` | 2026-09 |

Our own earlier implementation tiled, and every number this paper reports as a tiled score was
produced by it. That much we can state from our own history.

**Rows for other published implementations are to be added by direct inspection before this file is
cited as evidence about the field.** A claim about someone else's released code has to be verified
against that code at a named commit. Until a row is filled in from an actual reading, this file
supports a statement about our own pipeline and nothing wider.

## Why it matters

If tiling is the norm, then reported causality penalties across this literature are inflated by the
mechanism this paper measures, and the correction is free: no retraining, only a second pass over
existing weights. If tiling is not the norm, the finding still holds for the implementations that
do it, including our own, and the prescription in the paper still applies. The survey changes the
scope of the claim, not its validity.
