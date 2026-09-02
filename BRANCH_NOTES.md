# Branch notes: `run-provenance-and-streaming-eval`

**What this branch is for, in one sentence:** it fixes a defect that makes the archived
seed sweep mean something different from what the paper says it means, and it stops the
pipeline from throwing away the evidence every follow-up analysis needs.

Nothing in `results/` changes. No archived number changes. Every behavioural change is
opt-in, and the default path was verified byte-for-byte against `main`.

---

## Contents

1. [Why this branch exists](#1-why-this-branch-exists)
2. [What changed, file by file](#2-what-changed-file-by-file)
3. [What is new that you can run today](#3-what-is-new-that-you-can-run-today)
4. [Proof that nothing broke](#4-proof-that-nothing-broke)
5. [What this does to publication chances](#5-what-this-does-to-publication-chances)
6. [What work is left](#6-what-work-is-left)
7. [Quick reference](#7-quick-reference)

---

## 1. Why this branch exists

### Problem 1: the seed was secretly also the fold seed

In `main`, `train.py` passed `config['train']['seed']` into `get_cv_splits()`. That single
value controlled **two different things at once**: which random weights the model started
from, *and* which subjects landed in which fold.

So when the paper says "three seeds, 42, 43 and 44, on the same five folds", that is not
what happened. Each seed produced a **completely different split of the 78 subjects**. You
can see it in the archived run records:

```
results/sleep78_causal/fold_0_test_report.txt       (seed 42)
  ['48','49','53','08','52','71','33','15','62','54','70','51','41','77','57','23']

results/sleep78_causal_s43/fold_0_test_report.txt   (seed 43)
  ['75','74','00','49','67','30','21','52','15','13','34','28','35','57','26','01']
```

Fold 0 shares **3 of 16 subjects** between the two seeds. They are not the same fold.

**What this means for the paper.** The archived 78-subject experiment is three repeats of
5-fold cross-validation over **fifteen distinct partitions** — not five partitions measured
three times. Section V-C of the draft says the opposite and builds its whole statistical
argument on it. Two consequences:

- The "average the three seeds per fold, then test at n = 5" procedure averages three
  *unrelated* subject sets that happen to share an index. It is not a fold-level average.
- The variance decomposition that reported "a fold variance estimate of zero" was
  decomposing a factor that does not exist across seeds. That paragraph has to go.

**The good news:** the *pairing* was always sound. Causal and non-causal within one seed
used byte-identical subject lists, so all 15 differences are properly matched pairs. Only
the degrees-of-freedom accounting was wrong, and correcting it makes the result *stronger*,
not weaker (see [section 5](#5-what-this-does-to-publication-chances)).

**The fix.** `data.split_seed` now sets the partition independently of `train.seed`. Leave
it out and you get the old behaviour exactly, so every archived config still reproduces its
own splits. Set it to one value across a seed sweep and you finally get the matched design
the paper claims to have.

### Problem 2: evaluation threw away its predictions

`evaluate.py` computed metrics and then discarded the predictions. That one omission is
why the smoothing sweep, per-subject confidence intervals, calibration, prior correction
and any transition-timing analysis each cost a **complete retraining run**. The trained
weights from the archived runs no longer exist, so those analyses are currently impossible
at any price.

Evaluation now writes `fold_N_predictions.npz` with the subject, expert label, prediction
and logits for every second. Three analyses that were previously out of reach now run in
seconds from that file.

### Problem 3: the evaluation was not actually streaming

The default evaluation tiles each recording with **non-overlapping** 120-second windows.
Every window starts with an empty, zero-padded history, so the opening seconds of each
window are decided from context a deployed system would never lack. The paper claims a
real-time streaming model while evaluating it in a mode no deployment would use — which is
the same category of criticism the paper levels at everyone else's preprocessing.

`--stream_stride N` adds the honest mode: recompute every N seconds against a full buffer,
keep only the freshest N predictions.

### Problem 4: windows silently spanned splices

Preprocessing drops unscored epochs and concatenates a subject's two nights, so a stored
array is a chain of discontinuous stretches. A 120-second window could span a join that
never existed in time. Preprocessing now records where those joins are.

---

## 2. What changed, file by file

| File | Change | Default behaviour |
|---|---|---|
| `src/data/dataset.py` | `resolve_split_seed()`, `segment_starts()`, `respect_boundaries`, `cover_tail`, `window_subjects`, `window_starts` | unchanged |
| `src/train/train.py` | uses the resolved split seed; records both seeds in `split_fold_N.yaml`; records training class counts and loss type in the checkpoint | unchanged |
| `src/eval/evaluate.py` | `predict_windowed()` / `predict_streaming()`; saves predictions; `--stream_stride`; `--no_save_predictions` | unchanged |
| `src/data/preprocessing.py` | records `segment_starts` in each `.npz` and a segment count in the manifest | unchanged |
| `sweep_smoothing.py` | `--predictions` mode: sweeps from saved predictions instead of checkpoints | unchanged |
| `scripts/analyze_predictions.py` | **new** — subject-level bootstrap, per-subject table, class-prior correction | new |
| `scripts/boundary_latency.py` | **new** — stage-change reporting latency, 1 Hz vs 30 s | new |
| `tests/test_windowing.py` | **new** — 20 tests for windowing and streaming evaluation | new |
| `tests/test_analysis_scripts.py` | **new** — 27 tests for the analysis scripts | new |
| `tests/test_subject_splits.py` | 5 new tests, including the regression that would have caught the seed defect | new |
| `REPRODUCIBILITY.md` | documents split seeds, continuity, saved predictions, streaming evaluation | docs |

Test count goes from **19 to 67**.

### New configuration keys

```yaml
data:
  split_seed: 42            # fold layout. Omit -> uses train.seed (archived behaviour)
  respect_boundaries: false # drop windows spanning a night join or dropped-epoch gap
  cover_tail: false         # add a final window flush with the end of the recording
```

All three default to the archived behaviour. `respect_boundaries` **raises** rather than
silently doing nothing if the processed data predates the segment metadata, so you cannot
believe you enabled it when you did not.

`cover_tail` applies to **training only**. Evaluation deliberately ignores it: its extra
window overlaps the previous one, and those seconds would be counted twice in accuracy,
kappa and the confusion matrix. Use `--stream_stride` to score the trailing seconds
without double-counting.

---

## 3. What is new that you can run today

Everything below runs from a `logs_*` directory after evaluation. No checkpoint, no
processed data, no GPU.

### Subject-level confidence intervals

```bash
python scripts/analyze_predictions.py logs_78streaming_causal_s42 --bootstrap --per-subject
```

Resamples **whole subjects**, not seconds. Seconds inside one recording are heavily
autocorrelated, so a second-level interval would be far too narrow to mean anything. This
replaces the fold-level interval at n = 5 that the paper currently reports, and is both
narrower and easier to defend to a reviewer.

Implemented by summing per-subject 5×5 confusion matrices rather than re-scoring millions
of seconds per draw, so 5,000 draws take well under a second even on a full fold.

### Class-prior correction for the N3 over-prediction

```bash
python scripts/analyze_predictions.py logs_78streaming_causal_s42 --prior-correction
```

Pooled over the streaming runs, N3 currently has recall 0.85 but precision 0.55, and 12% of
all true N2 seconds are called N3 — the class-weighted focal loss pays more for missing N3
than for over-calling it. This shifts the logits by the log ratio of the target and training
priors at inference. No retraining, no added parameters, same family as the mode filter.

The training class balance is read from the prediction file (newly recorded by the trainer).
If it is absent the script says so and names the assumption it fell back on rather than
printing an authoritative-looking table built on a guess.

### The "why one label per second?" experiment

```bash
python scripts/boundary_latency.py logs_78streaming_causal_s42 --hold 10
```

This is the experiment the paper does not have and most needs. Its own limitations section
concedes that per-second output is shown to be *attainable*, not *useful*.

The comparison is set up carefully, because the expert labels are themselves only known to
30 seconds. Asking which system lands closer to the scored transition instant is **not** a
fair question: the scored transition sits on the epoch grid by construction, so an
epoch-level system that is merely correct scores a perfect zero. That measures the grid,
not the model.

What *is* fair is when each system can report the change at all, under the causal
constraint both face in deployment:

- **per-second:** the label for second *t* is emitted at second *t*
- **30-second:** the label for epoch *k* cannot be emitted until epoch *k* has finished,
  at second 30(*k*+1)

So the epoch system carries a **structural 30-second floor** on any change that begins an
epoch. The per-second system has no such floor. Whether it actually beats the floor, and
how reliably, is the empirical question.

Two safeguards, both of which materially change the answer:

- **Both systems are charged for their evidence.** A detection requires the new stage to
  be held for `--hold` seconds, so the per-second system reports at the *last* second of
  that run, not the first. Crediting the first second while charging the epoch system its
  full window would understate every delay by `hold - 1` seconds and inflate the headline.
- **Changes the model was already calling are flagged.** If the model spent most of the
  run-up emitting the incoming stage, that is a false positive during the *outgoing* stage,
  not an early detection. The report gives a clean subset excluding them, and says to quote
  that one.

Splices are skipped: a label change at a night join is not a transition in time.

### Smoothing without a checkpoint

```bash
python sweep_smoothing.py --predictions logs_78streaming_causal_s42
```

The smoothing sweep in `results/smoothing.md` was run on the **epoch-z-scored** pipeline —
the leaky one. If the paper's headline moves to the streaming runs, Table VII and Figure 2
are on a different pipeline from everything around them, which a reviewer will notice.
Re-running it still needs streaming checkpoints (they do not exist), but from the *next*
sweep onward this command does it in seconds.

### Streaming evaluation

```bash
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml \
    --fold 0 --stream_stride 30
```

Costs `sequence_length / N` forward passes — 4× a normal pass at stride 30 and 120 s
context. Writes to `*_streaming30` files, so it never overwrites an archived artifact and
the two modes sit side by side.

---

## 4. Proof that nothing broke

The critical invariant is that the default path is **byte-identical** to `main`. Verified
by running the same config on the same data before and after every change:

| Artifact | Result |
|---|---|
| Subject splits, all 19 configs (SHA-256 of every fold assignment) | identical |
| `test_metrics_summary.csv` | identical |
| `fold_0_metrics.csv` (per-epoch training history) | identical |
| `fold_0_test_report.txt` (metrics + confusion matrix) | identical |
| Checkpoint weights (all 64 tensors) | identical |
| `split_fold_0.yaml` | + `split_seed`, `train_seed` only |
| Checkpoint dict | + `train_class_counts`, `loss_type` only |

Also confirmed still working: full test suite (67 tests), `smoke_test.py`,
`scripts/verify_causality.py` (exit 0 on the causal config, exit 1 correctly refusing the
non-causal one), `scripts/validate_results.py`, `analysis_stats.py`, `scripts/pool_seeds.py`
(reproduces −0.0249, *t*(14) = −7.33, *p* = 3.75e−06 exactly), `make_figures.py`.

### Bugs found and fixed during review

An independent adversarial review of this branch found twelve issues, all fixed before this
document was written. The three that would have put wrong numbers in a paper:

1. **The latency metric was biased in our own favour.** The per-second system was credited
   at the *first* second of its confirmation window while the 30-second system paid its
   full window. Every delay was understated by `hold - 1` seconds — 9 s at the defaults.
   That is the number the script exists to produce.
2. **`cover_tail` double-counted seconds during evaluation.** Its extra window overlaps the
   previous one, so overlapping seconds entered accuracy, kappa and the confusion matrix
   twice, moving accuracy by ~0.8 points on identical weights. Evaluation now ignores it.
3. **Subject ids collided across log directories**, which is the documented usage
   (`logs_*/`). Two runs each containing subject "00" merged into one bootstrap unit,
   silently halving the number of independent units and giving the wrong interval width.

The rest: `float16` logits did not reproduce the saved predictions (now `float32`); a
uniform training prior was hard-coded and unverifiable (now recorded by the trainer); the
"already calling it" guard inspected only the first few seconds of the run-up; night joins
counted as stage changes; macro-F1 used two different label sets for the estimate and its
interval; a 1.8 GB memory blow-up when pooling 15 folds; `--stream_stride 0` silently ran
the wrong path; a quoted or blank `split_seed` would change the folds while recording the
wrong value; duplicate files could be loaded twice.

---

## 5. What this does to publication chances

Two things need saying plainly before the numbers.

**This branch alone does not raise the odds much.** It is engineering groundwork. What it
does is make the *next* sweep produce evidence that is currently unobtainable, and remove a
defect that a careful reviewer could find on their own — which is the difference between a
methods question and a credibility problem.

**The single biggest win available to you is not in this branch at all.** It is that the
repository already contains complete leak-free streaming results (κ = 0.665 pooled over
three seeds, against the draft's 0.634) that the paper still describes as unfinished
future work. Swapping the headline costs zero compute. That is worth more than everything
here.

### Odds by stage

Percentages are my read of reviewer behaviour at these venues, not a calibrated forecast.
Treat them as relative ordering, not precision.

| Stage | IEEE EMBC | IEEE BHI | IEEE JBHI | IEEE TNSRE | Biomed. Signal Process. Control | IEEE Sensors J. |
|---|---:|---:|---:|---:|---:|---:|
| Draft v8 as it stands | 70% | 65% | 30% | 25% | 45% | 40% |
| + rewrite on the leak-free results, corrected statistics (**no compute**) | 85% | 80% | 38% | 33% | 60% | 55% |
| + this branch, then one clean retrain with its analyses | 93% | 88% | 50% | 45% | 72% | 68% |
| + a second dataset and the ablation on a second architecture | — | — | 67% | 62% | 85% | 75% |
| + real embedded measurement and a protocol-matched baseline | — | — | 75% | 72% | 90% | 85% |

Venue notes:

- **IEEE EMBC** (conference, ~4 pages) is the natural home for the work as it stands after
  the rewrite. The measurement and the per-class decomposition are a coherent short paper.
- **IEEE BHI** is similar in bar, slightly more receptive to the wearable framing.
- **IEEE JBHI** and **IEEE TNSRE** are the "good journal" targets. Both will hold you to
  multi-dataset evidence; TNSRE is marginally harder for a paper whose absolute accuracy
  sits below published baselines, which is why it trails JBHI throughout.
- **Biomedical Signal Processing and Control** (Elsevier) is the realistic strong-odds
  journal and is materially more attainable than JBHI at every stage. If the goal is a
  solid indexed journal publication rather than a specific IEEE title, this is the highest
  expected value.
- **IEEE Sensors Journal** becomes attractive specifically once there is an on-device
  measurement, because that is what it rewards.

### What this branch contributes to those numbers

| Contribution | Effect |
|---|---|
| Removes a defect a reviewer could find themselves | Protects the whole submission; a reviewer who discovers the seed problem unaided will doubt everything else |
| Lets the statistics be described correctly | The corrected framing survives every level of conservatism (see below) and is *more* defensible than the draft's |
| Makes the streaming claim testable at the evaluation level | Answers the strongest methodological objection available to a reviewer |
| Provides the "why 1 Hz" experiment | Answers the paper's own central unanswered question |
| Makes every future re-analysis free | Turns "we would need to retrain" into "we ran it" during revision, which is when it matters most |

### The statistics, corrected

The corrected framing is three repeats of 5-fold cross-validation over fifteen distinct
partitions. The right primary test is the **Nadeau–Bengio corrected resampled *t*-test**,
which accounts for the train-set overlap between repeats that a naive paired test ignores.

| Test | Streaming (leak-free) | Epoch z-score |
|---|---|---|
| Mean Δκ (causal − non-causal) | −0.0249 (loses 14/15) | −0.0287 (loses 15/15) |
| Naive paired *t*, n = 15 | *t*(14) = −7.33, *p* = 3.8e−06 | *t*(14) = −10.37, *p* = 5.9e−08 |
| **Nadeau–Bengio corrected** | **−3.36, *p* = 0.0047** | **−4.76, *p* = 0.0003** |
| Seed-level, n = 3 | *t*(2) = −15.62, *p* = 0.0041 | *t*(2) = −11.10, *p* = 0.0080 |

The effect holds at every level of conservatism, so correcting the description costs you
nothing rhetorically and buys a statistics section that will not be attacked.

---

## 6. What work is left

### Immediate, no compute (do this first — biggest single gain)

1. **Move the headline to the streaming runs.** They are already in `results/`, pooled over
   three seeds: accuracy 0.7505, κ 0.6649, macro-F1 0.6894, N1 F1 0.4212, 30 s κ 0.6847.
   The draft reports the leaky epoch-z-scored seed-42 run (κ 0.634) instead.
2. **Rewrite §III-D, VII, VIII, IX.** The preprocessing leak is *closed*, not future work.
   It improves both arms (+0.024 κ causal, +0.021 κ non-causal), so it is a modelling gain,
   not the removal of a crutch.
3. **Rewrite §V-C** using the corrected framing above. Delete the variance-decomposition
   paragraph and the √3 argument.
4. **Replace the per-class table** with the leak-free version. It is cleaner: N2 and N3 come
   out free, and REM alone survives Holm correction at −0.060 F1, 2.6× the macro average.
5. **Fix the reproducibility claim.** The draft says per-fold checkpoints are in the
   repository. They are not — `checkpoints/` holds only `.gitkeep`. At 30,757 parameters
   they are ~130 KB each; commit them or delete the claim.
6. **Reconcile the AttnSleep baseline.** The draft says "the same 5 folds";
   `AttnSleep_Results.md` records 10-fold. Both cannot be true.
7. **Add one sentence on BatchNorm** — it pools over the time axis during training, though
   inference uses frozen running statistics and is genuinely causal. Pre-empt the question.

### The retrain (this is the main outstanding compute)

One sweep = 5 folds × 2 arms × 3 seeds. Run it with:

```yaml
data:
  split_seed: 42        # same value in all three seed configs
```

so that seeds vary the initialisation only and you finally have the matched design. Then,
all free from the saved predictions:

- windowed vs streaming evaluation side by side (the gap is itself a result)
- smoothing sweep redone leak-free, with the window selected on **validation** folds rather
  than the test folds it is reported on
- boundary-latency analysis
- N3 prior correction
- subject-level bootstrap intervals

Budget note: `--stream_stride 30` at 120 s context costs 4× a normal evaluation pass.

Also worth folding into the same sweep: `respect_boundaries: true` so no training window
spans a splice, and a deployment table (MACs, peak memory, p50/p95/p99 latency) — the
current evidence is CPU throughput on a desktop i9 against a 3 ms/s target the paper itself
calls arbitrary.

### For a good journal (JBHI, TNSRE, BSPC)

1. **A second dataset.** The single highest-value addition. **DOD-H / DOD-O** is the best
   value: small, openly available, and multi-scorer — so it also supplies the inter-scorer
   agreement ceiling the paper currently lacks, which is what would let you position
   κ = 0.665 against what two humans achieve rather than only against DeepSleepNet.
   SHHS or MASS if you want scale instead.
2. **The causality ablation on a second architecture.** Until then, "causality costs
   0.025 κ" is a property of *this* network, not of the constraint.
3. **One protocol-matched baseline.** TinySleepNet (~1.3 M parameters) through your exact
   preprocessing, splits, seeds and wake trimming. One genuinely matched baseline is worth
   more than three unmatched ones.
4. **A measurement on target-class hardware** (Raspberry Pi or ARM Cortex-M). Very few
   papers in this space have one, and it is what turns "feasible on a desktop CPU" into the
   on-device claim the framing promises.
5. **Cross-dataset transfer** — train on Sleep-EDF, test on the second dataset without
   fine-tuning. Reviewers ask for this by reflex and it is nearly free once dataset two is
   in the pipeline.

### Suggested repositioning

Lead with the leak, not the model. The strongest available framing is: **causality is a
property of the pipeline, not of the network, and this literature measures it in the wrong
place.** Then:

1. A near-universal preprocessing step reads the future; the causal replacement is free.
2. Given a genuinely causal pipeline, the constraint itself costs 0.025 κ, concentrated
   almost entirely in REM.
3. A verification harness that makes the claim checkable rather than asserted.

The per-second model then becomes the vehicle that makes points 1–3 measurable, rather than
a competitor to DeepSleepNet that loses on accuracy. That removes the most obvious rejection
reason, which is reporting below-baseline numbers and asking to be published anyway.

---

## 7. Quick reference

```bash
# Everything still works the way it always did
python -m src.train.train    --config configs/sleep78_streaming_causal.yaml --fold -1
python -m src.eval.evaluate  --config configs/sleep78_streaming_causal.yaml --fold 0

# Pin the fold layout so seeds vary the initialisation only
#   data:
#     split_seed: 42

# Streaming evaluation (4x cost at stride 30, 120 s context)
python -m src.eval.evaluate --config configs/sleep78_streaming_causal.yaml \
    --fold 0 --stream_stride 30

# Analyses, all from saved predictions, no checkpoint needed
python scripts/analyze_predictions.py logs_78streaming_causal_s42 \
    --bootstrap --per-subject --prior-correction --out results/predictions.md
python scripts/boundary_latency.py logs_78streaming_causal_s42 \
    --hold 10 --out results/boundary_latency.md
python sweep_smoothing.py --predictions logs_78streaming_causal_s42

# Checks
python -m unittest discover -s tests          # 67 tests
python smoke_test.py configs/sleep78_streaming_causal.yaml
python scripts/verify_causality.py --config configs/sleep78_streaming_causal.yaml
```

### Files written by evaluation

| File | Contents |
|---|---|
| `fold_N_test_report.txt` | metrics, classification report, confusion matrix |
| `test_metrics_summary.csv` | one row per fold |
| `fold_N_predictions.npz` | subject, second index, expert label, prediction, logits, segment flags, training class counts |
| `*_streamingN.*` | the same three, from streaming evaluation |
