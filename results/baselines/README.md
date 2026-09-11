# Baseline reproductions

Three published models were run in-house so that the comparison in `../../RESULTS.md` rests on
runs we control rather than on numbers copied from papers. Each report below records the
settings used, the per-fold results, and the defects that limit what the run can be cited for.

All three were run on Sleep-EDF, 39 recordings, single channel Fpz-Cz at 100 Hz. That is the
20-subject cassette subset, not the 78-subject set used for the main experiments, so these
numbers are comparable to each other and to our Sleep-EDF-20 rows, and are not comparable to any
Sleep-EDF-78 figure.

| Report | Model | Epoch | Folds | Headline |
|---|---|---|---|---|
| [`deepsleepnet.md`](deepsleepnet.md) | DeepSleepNet, CNN plus Bi-LSTM, about 22M parameters | 30 s | 5, subject-wise | 82.23% accuracy, kappa 0.7535, mean of 5 folds |
| [`attnsleep.md`](attnsleep.md) | AttnSleep, multi-scale residual CNN plus attention, about 48k parameters | 30 s and 5 s | 10 | 80.90% accuracy, kappa 0.7088 at 30 s |
| [`caresleepnet.md`](caresleepnet.md) | CareSleepNet, CNN plus Transformer, 22.7M parameters | 30 s | 5, subject-wise | 78.07% accuracy, kappa 0.694, best fold only |

## What these runs can and cannot support

They support a statement about the accuracy and parameter cost of established offline models on
this corpus, run by us under settings we record.

They do not support a protocol comparison. All three are epoch-level models that emit one label
per epoch, so the tiled and streaming protocols do not apply to them: there is no context buffer
whose positions can be scored separately. The protocol result in `../../RESULTS.md` is measured
only on models that emit a label per second.

Two of the three carry defects that bound what they can be cited for, both recorded in their own
reports rather than buried: CareSleepNet lost one fold to a logging failure and produced a 38%
outlier on another, so only its best fold is verified; and the AttnSleep 5-second variants were
trained on a smaller sample than the 30-second baseline, so the comparison between its own rows
is not like for like.
