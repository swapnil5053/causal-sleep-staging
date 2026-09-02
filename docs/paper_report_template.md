# Template: paper-summary report

Every method change and every piece of related-work reading gets one of these before any code
is written. One to two pages.

The point is to force the justification to exist before the implementation does, so a change
arrives in the paper with a reason attached rather than as a fait accompli. The failure mode
for this project at its current stage is that architecture work displaces the measurement,
which is the actual contribution — and that contribution is now strong enough to have something
to lose.

**The rule: no method change enters the codebase before its report exists.** If the paragraph
in section 5 is hard to write, drop the idea rather than forcing it in.

Copy this file to `docs/reports/<short-name>.md` and fill it in.

---

## 1. Citation and summary

Full citation. One paragraph on what the paper actually does — not what its abstract claims,
what its method section describes.

## 2. The mechanism to import

The specific mechanism, described precisely enough to implement from this page alone:
equations, hyperparameters, computational complexity, and what state it carries at inference.

## 3. What it changes about our system

- Which module it replaces or sits beside.
- What the parameter count becomes.
- **Whether end-to-end causality is preserved, and by what argument.** For anything touching
  preprocessing or the model, state explicitly whether `python scripts/verify_causality.py`
  would still pass, and which of its seven checks are at risk.
- Whether `scripts/streaming_demo.py --verify` would still show the streaming path reproducing
  the evaluated predictions.
- What it does to inference cost per second of EEG.

## 4. Expected effect, written down before running it

The numbers we expect to move and by roughly how much, recorded *before* the experiment. Also:

> **Abandonment threshold.** The result at which we stop and report a negative finding rather
> than tuning until it works: ______

## 5. The paragraph as it would appear in the paper

Write it out, positioned against the existing result — an addition to the measurement, or a
replacement for part of the method. If this paragraph cannot be written convincingly, that is
the answer, and the correct next step is to drop the idea.

## 6. Cost and risk

Implementation time, compute, and the honest risk that this displaces the paper's actual
contribution.

---

## For related-work reports, replace section 5

Reading whose purpose is to constrain the novelty claim answers a different closing question.
Not "should we import this" but:

> **Does this invalidate our claim, and if so, what is the narrower claim we can still make?**

Write the narrowed claim out in full, as it would appear in the abstract. `docs/framing_usleep.md`
is a worked example of this variant.
