# Reframing the contribution against U-Sleep and U-Time

Deliverable for the Friday review: the novelty claim as currently written does not survive
contact with U-Sleep, and the two paragraphs below are the replacement. The finding is ours
and it is better raised by us than by an examiner.

## 1. What the reading found

**Per-second sleep staging is not novel.** U-Sleep (Perslev et al., *npj Digital Medicine*
4:72, 2021) evaluated its output at **14 frequencies spanning 2 to 7,680 stages per minute**
— the top of that range is 128 Hz, well past our 1 Hz — and reports mean F1 rising from 0.60
at 2 stages/minute to a maximum of **0.94 at 1,280 stages/minute**. It was trained and
evaluated on **15,660 participants across 16 clinical studies**. Its predecessor U-Time
(Perslev et al., NeurIPS 2019) established the same dense-output formulation.

So the sentence currently opening [README.md](../README.md) — "Most sleep staging models
score 30-second epochs and read the whole night at once, so they cannot run live" — is half
wrong in a way a reviewer will find immediately. The second clause is true. The first is not:
high-frequency output was solved five years ago, at a scale we cannot approach.

**What U-Sleep does not have is causality.** It is a fully convolutional U-Net: an
encoder-decoder with skip connections, symmetric convolutions and upsampling, consuming a
long input segment in one pass. Every one of those choices lets an output at time *t* depend
on signal after *t*. The paper makes no causal, online, streaming or latency claim, because
that is not what it is for — it is a batch scoring system for completed recordings.

> **Verify before Friday.** The output-frequency and cohort figures above are confirmed from
> the published record. The architectural reading is from the methods description and could
> not be re-checked against the full text here (Nature's site is paywalled to automated
> access). Read the Methods section — specifically the decoder and the segment classifier —
> and confirm two things: that convolutions are not causally padded, and that the paper
> nowhere claims online operation. If either is wrong, the paragraphs below need revising
> before they go in the deck.

## 2. The claim, rewritten

Two paragraphs. Neither says anything we cannot demonstrate.

### Abstract

> Sleep staging systems that score a completed recording can afford to read the whole night
> at once, and the strongest of them do: U-Sleep reaches its best accuracy with a fully
> convolutional encoder-decoder whose prediction at any instant draws on signal from both
> directions in time, and demonstrates that dense output at up to 7,680 stages per minute is
> attainable at that scale. High-frequency staging is therefore established. What is not
> established is what such a system costs once it is constrained to run as a recording
> arrives. We present a 30,757-parameter single-channel model that emits one sleep stage per
> second under end-to-end causality — the prediction at second *t* is a function of the raw
> signal up to second *t* alone, through normalisation as well as through the network — and
> we measure the price of that constraint rather than assuming it. Holding architecture,
> parameters, data and subject folds fixed and varying only access to future signal, the
> causal model loses 0.025 Cohen's kappa (0.665 against 0.690) on Sleep-EDF-78. The result is
> replicated under two independent preprocessing regimes: per-epoch z-scoring, which is not
> itself causal, gives 0.029, and a trailing-window statistic, which is, gives 0.025. Across
> thirty paired measurements the sign never changes. Causality is verified rather than
> asserted: perturbing the input at a future sample leaves every earlier output
> bit-identical, checked over the whole raw-sample-to-logits path by a script that ships with
> the code and exits non-zero on failure.

### Contributions

> This work makes three contributions, none of which is the per-second output rate. First, a
> measurement: the cost of end-to-end causal operation for a fixed sleep-staging
> architecture, isolated by an ablation that changes only convolution padding and the
> attention mask and holds every parameter, every training subject and every fold constant.
> That cost is 0.025 kappa, and it is replicated across three seeds, five folds and two
> preprocessing regimes — thirty paired measurements agreeing on sign and closely on
> magnitude. Second, a verification method: causality is a property of the whole pipeline,
> not of the network alone, and we show that a model can satisfy it layer by layer while its
> preprocessing quietly violates it. Our own archived runs did exactly that, normalising each
> 30-second epoch against statistics drawn from the entire epoch. Replacing that with a
> trailing-window statistic, and proving the offline arrays identical to what a
> sample-at-a-time online filter produces, made the pipeline causal end to end — and improved
> kappa from 0.641 to 0.665, so the correction cost nothing. Third, the artefact itself: a
> streaming implementation that ingests raw samples one at a time and reproduces the
> evaluated predictions exactly, so the reported kappa is the number the deployable path
> produces and not a number obtained by batching a night.

## 3. Why this version survives an examiner

| Objection | The answer in the text above |
|---|---|
| "Per-second staging already exists — see U-Sleep." | Conceded in the first sentence, with the number, before it is asked. |
| "So what is new?" | The measured cost of causality, not the output rate. |
| "Isn't your model just worse than U-Sleep?" | Different problem. U-Sleep scores completed recordings; it cannot run as the signal arrives, and we quantify what running that way costs. |
| "How do you know your pipeline is causal?" | `scripts/verify_causality.py`, run live: seven checks, maximum logit difference 0.000e+00, with a non-causal control that leaks. |
| "Is the effect real or a fluke of one split?" | Two preprocessing regimes, three seeds, five folds, thirty paired measurements, consistent sign. |
| "Isn't the gain just from a better normaliser?" | Yes, partly, and we say so: the non-causal arm improved by a comparable amount, so the improvement is the normaliser, not causality. Causality still costs 0.025. |

## 4. Numbers used above, and where they come from

| Figure | Source |
|---|---|
| causal 0.6649, non-causal 0.6898, cost -0.0249 | `results/statistics_streaming_pooled.md` |
| epoch-normalised 0.6406 / 0.6693, cost -0.0287 | `results/statistics.md`, `RESULTS.md` |
| 30 paired measurements, consistent sign | 2 regimes x 3 seeds x 5 folds |
| 30,757 parameters | `README.md`, `smoke_test.py` |
| bit-identical outputs before the perturbation | `results/causality_verification.md` |
| 2 to 7,680 stages/minute, F1 0.60 to 0.94, 15,660 participants, 16 studies | Perslev et al. 2021 |

**One dependency.** The paragraphs above deliberately quote effect sizes and consistency
rather than a p-value, because the pooled p is being recomputed: the archived
`t(14) = -7.33, p = 3.75e-06` treats fifteen measurements over one 78-subject pool as
independent, and the Nadeau-Bengio correction gives roughly `t = -3.36, p = 0.005`. If a
p-value is wanted in the abstract, take it from the regenerated
`results/statistics_streaming_pooled.md`, not from the current file. The subject-level test
in `scripts/subject_paired_test.py` is the stronger alternative once checkpoints allow it:
pairing by subject gives 78 genuinely independent differences and needs no correction at all.
