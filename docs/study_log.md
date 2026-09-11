# Study log

How this project arrived at the result it reports, including the claim it abandoned. The papers
say that an earlier version of this work reported a finding now believed to be largely an
artifact. This is that history in one page.

## 1. A per-second causal model, on 20 subjects

The starting question was what a strictly causal sleep stager costs. The model emits one stage per
second from a single EEG channel using only past signal: a two-branch causal convolution pooled to
1 Hz, dilated causal convolutions, causally masked attention, a linear head. 30,757 parameters.

Measured on the 20-subject Sleep-EDF cassette subset, the causal and non-causal arms came out
within 0.014 kappa of each other and the sign was unstable across folds. Fold-to-fold standard
deviation there is 0.092. With four held-out subjects per fold, nothing was resolvable.

**Change:** move to the 78-subject set, which raises held-out subjects per fold from four to
sixteen and drops fold variance to 0.030.

## 2. The normalisation leak

Per-epoch z-scoring is near-universal in this literature and it reads the future: normalising an
early sample with statistics computed over its whole 30-second epoch uses samples that arrive
later. The network was causal and the pipeline was not.

It was replaced with a trailing 30-second statistic computable one sample at a time, and
`src/data/normalization.py` ships the online filter so the equivalence is proved rather than
asserted.

Kappa rose from 0.6406 to 0.6649 on the causal arm and 0.6693 to 0.6898 on the non-causal arm.
Closing the leak was a modelling gain, not a trade.

## 3. The first result, now superseded

With three seeds by five folds on the leak-free pipeline, the causal constraint cost 0.0249 kappa,
the causal arm losing in 14 of 15 paired folds, and the cost appeared concentrated in REM, which
paid 0.060 F1 against 0.025 for the next-largest class.

That was the reported finding. It is reproducible under the protocol that produced it.

## 4. The doubt

The evaluation covered each recording with non-overlapping 120-second windows and scored every
position of every window. Each window starts from an empty history. A causal model at position 0
of a window has seen nothing; a non-causal model at the same position can still read forward.

The protocol was therefore not neutral between the two arms, and no deployment would ever score a
model that way.

## 5. The measurement that settled it, 8 September 2026

No retraining was needed. The same trained weights were rescored under a streaming protocol,
recomputing every 30 seconds against a full buffer and keeping only the freshest 30 predictions,
so every scored second carries at least 90 seconds of genuine history.

The causal arm gained 0.0350 kappa. The non-causal arm gained 0.0036. Under the deployment
protocol the causality cost became +0.0065, a tie.

## 6. The reframing, and why the obvious claim was dropped

The first instinct was to claim that latency-matched evaluation removes the causality penalty.
That claim was dropped, for two reasons.

The streaming protocol strips the non-causal arm of the lookahead that defines it, so it compares
deployment options rather than architectures. And on DOD-H the claim fails at the margin: the
non-causal arm's best operating point beats the causal arm's zero-latency score by 0.0136 kappa in
exchange for 46 seconds of delay.

What replaced it is a within-arm quantity: how much does each arm gain from the protocol change,
and how much larger is that gain for the causal arm. The same weights, the same held-out seconds,
two protocols. It does not require the arms to be comparable, which is why it is measurable on
corpora where the between-arm ablation is not.

The mechanism was then verified arithmetically rather than argued: the tiled score is the mean of
the accuracy curve across all 120 buffer positions, reproduced to within 0.003 kappa in all four
model and dataset combinations, and the penalty is generated entirely by positions carrying under
30 seconds of history.

The apparent REM concentration went with it. Under the deployment protocol REM's penalty falls
from 0.060 F1 to 0.002, and N1, N2 and N3 invert with the causal arm ahead.

## 7. A second dataset, 9 September 2026

DOD-H: 25 subjects, a different montage, a different sampling rate, a different scoring team, a
different class balance, and no wake trimming. Three seeds by five folds per arm, both protocols.

The protocol effect replicated at +0.0292 against Sleep-EDF-78's +0.0314, positive in all 15
folds. The between-arm causality cost did not resolve at all, p = 0.22, because five held-out
subjects per fold give a fold standard deviation of 0.098.

That asymmetry became a reported result in its own right. The same fifteen runs cannot resolve the
between-arm difference and do resolve the within-arm one, which is the practical argument for the
estimand.

## 8. A second architecture, 10 to 11 September 2026

The remaining objection was that the effect was demonstrated on one architecture. The project plan
had cut a second architecture as a three to four week job. It was instead ported from a recurrent
variant written on a separate branch, at a cost of one file and two configuration changes.

One defect had to be fixed first. The bidirectional control carried 1.94 times the parameters of
the causal arm, which would have confounded capacity with causality. Reducing the per-direction
hidden size brought the two within 0.68%.

The protocol excess reproduced at +0.0298, positive in all 15 folds. The gain to the causal arm
itself was +0.0353 against the convolutional model's +0.0350, an agreement to 0.0003 kappa between
two architecture families sharing only the front end.

## 9. What is still open

Both models are small and their absolute accuracy sits below published offline systems. A capacity
check retrained the convolutional causal arm at 3.1 times the parameters. Its protocol gain came
out at +0.0365, in line with the other three arms, so the penalty is not a function of parameter
count. But the wider model scored lower rather than higher, 0.6535 tiled against 0.6634, with its
best checkpoints arriving at epochs 4 to 8 while training accuracy kept climbing. It overfits this
corpus. So capacity is settled and strength is not, and that is recorded as open rather than
answered.

The streaming protocol still denies the non-causal arm its lookahead. At matched latency the two
datasets disagree, and both outcomes are reported.

## Reading on

- [`../RESULTS.md`](../RESULTS.md) for every number and the file that produced it.
- [`architecture.md`](architecture.md) and [`alternative_gru_architecture.md`](alternative_gru_architecture.md)
  for the two models.
- [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the commands.
