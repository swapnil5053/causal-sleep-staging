"""How quickly a stage change is reported, per-second output against 30-second output.

This is the measurement that answers "what does one label per second actually buy?".

The comparison has to be set up carefully, because the expert labels are themselves only
known at 30-second resolution. Asking which system lands closer to the scored transition
instant is not a fair question: the scored transition sits on the epoch grid by
construction, so an epoch-level system that is merely correct scores a perfect zero. That
measures the grid, not the model.

What can be compared fairly is *when each system is able to report the change at all*,
under the causal constraint both would face in deployment:

  per-second   the label for second t is emitted at second t
  30-second    the label for epoch k covers seconds [30k, 30k+30) and cannot be emitted
               until that epoch has finished, at second 30(k+1)

So an epoch-level system carries a structural floor: a stage change at the start of epoch
k cannot be reported before second 30(k+1), a full 30 s later. A per-second system has no
such floor and may report inside the epoch. Whether it actually does, and how reliably, is
an empirical question, and this script answers it from archived predictions.

A detection requires the new stage to be held for `--hold` consecutive seconds, so that a
single flickering second does not count as detecting the transition.

    python scripts/boundary_latency.py logs_78streaming_causal_s42
    python scripts/boundary_latency.py logs_*/ --hold 15 --out results/boundary_latency.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_predictions import load_folds  # noqa: E402

STAGES = ["W", "N1", "N2", "N3", "REM"]
SECONDS_PER_EPOCH = 30


def epoch_majority(predictions):
    """Majority vote inside each 30-second epoch: the epoch-level system's output."""
    usable = (len(predictions) // SECONDS_PER_EPOCH) * SECONDS_PER_EPOCH
    blocks = predictions[:usable].reshape(-1, SECONDS_PER_EPOCH)
    return np.array([np.bincount(row, minlength=len(STAGES)).argmax() for row in blocks])


def first_sustained(stream, target, start, stop, hold):
    """First index in [start, stop) where `stream` equals `target` for `hold` seconds."""
    start = max(start, 0)
    stop = min(stop, len(stream) - hold + 1)
    for index in range(start, stop):
        if np.all(stream[index:index + hold] == target):
            return index
    return None


# A change is treated as already being called when the model spent at least this much of
# the pre-boundary window emitting the incoming stage. Chosen so that a genuine detection a
# few seconds early is not flagged, while a model sitting on the stage for most of the
# preceding epoch is.
ALREADY_ACTIVE_SHARE = 0.5


def subject_transitions(y_true, y_pred, hold, max_early, max_late, segment_start=None):
    """Reporting delay for every scored stage change in one subject record.

    Both systems are charged for the evidence they need. The per-second system requires
    `hold` consecutive seconds of the new stage, so it can only report at the *last* second
    of that run, not the first; the epoch system requires a whole epoch and reports at the
    end of it. Crediting the per-second system at the start of its hold window while
    charging the epoch system its full window would understate every delay by `hold - 1`
    seconds and inflate the headline comparison.

    Searching backwards from the scored change has a second trap: if the model was already
    emitting the new stage through most of the preceding window, that is a false positive
    during the outgoing stage rather than an early detection. Those are flagged.
    """
    epochs = len(y_true) // SECONDS_PER_EPOCH
    if epochs < 3:
        return []
    true_epochs = y_true[::SECONDS_PER_EPOCH][:epochs]
    pred_epochs = epoch_majority(y_pred)

    rows = []
    for k in range(1, epochs):
        if true_epochs[k] == true_epochs[k - 1]:
            continue
        new_stage = int(true_epochs[k])
        instant = k * SECONDS_PER_EPOCH
        # Skip changes too close to either end for the search window to fit.
        if instant < max(hold, max_early) or instant + max_late > len(y_true):
            continue

        search_from = instant - max_early
        # A label change at a splice is not a stage transition in time: it is where an
        # unscored epoch was dropped or where the subject's two nights were joined.
        if segment_start is not None and \
                np.any(segment_start[search_from:instant + max_late]):
            continue

        # Was the model already sitting on the new stage through the run-up?
        prior_share = float(np.mean(y_pred[search_from:instant] == new_stage)) \
            if instant > search_from else 0.0
        already_active = prior_share >= ALREADY_ACTIVE_SHARE

        # Per-second system: confirmed at the last second of the sustained run.
        second = first_sustained(y_pred, new_stage, search_from, instant + max_late, hold)
        per_second_delay = None if second is None else int(second + hold - 1 - instant)

        # Epoch system: epoch j's label is only available once epoch j has elapsed.
        epoch_delay = None
        first_epoch = max(k - max_early // SECONDS_PER_EPOCH, 0)
        last_epoch = min(k + max_late // SECONDS_PER_EPOCH, len(pred_epochs))
        for j in range(first_epoch, last_epoch):
            if pred_epochs[j] == new_stage:
                epoch_delay = int((j + 1) * SECONDS_PER_EPOCH - instant)
                break

        rows.append({
            "stage": new_stage,
            "per_second": per_second_delay,
            "epoch": epoch_delay,
            "already_active": already_active,
            "prior_share": prior_share,
        })
    return rows


def summarise(values):
    finite = np.array([v for v in values if v is not None], dtype=float)
    if len(finite) == 0:
        return None
    return {
        "n": len(finite),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
    }


def render(folds, args):
    lines = []
    write = lines.append

    rows = []
    spliced = 0
    for fold in folds:
        codes = fold["subject_code"]
        flags = fold.get("segment_start")
        for code in np.unique(codes):
            mask = codes == code
            before = len(rows)
            rows.extend(subject_transitions(
                fold["y_true"][mask], fold["y_pred"][mask],
                args.hold, args.max_early, args.max_late,
                segment_start=None if flags is None else flags[mask]))
            spliced += 0 if flags is None else max(0, before - len(rows))

    write("# Stage-change reporting latency\n")
    write(f"Read from {len(folds)} archived prediction file(s). A stage change counts as "
          f"reported once the new stage has been held for {args.hold} consecutive seconds, "
          f"so the report lands on the last second of that run. Search window: "
          f"{args.max_early} s before to {args.max_late} s after the scored change.\n")
    write("Both systems are charged for the evidence they need. The per-second system "
          "reports at the end of its hold window; the 30-second system cannot report epoch "
          "*k* until epoch *k* has elapsed, so it carries a structural floor of 30 s on any "
          "change that begins an epoch. Negative values mean the change was reported before "
          "the scored boundary.\n")
    if any(f.get("segment_start") is None for f in folds):
        write("> Note: at least one prediction file predates the recording of segment "
              "boundaries, so label changes at a night join or a dropped-epoch splice "
              "cannot be excluded from it. Re-run evaluation to record them.\n")

    if not rows:
        write("No scored stage changes survived the window filters, so there is nothing to "
              "report. Check that the prediction files cover whole recordings.\n")
        return "\n".join(lines) + "\n"

    flagged = sum(1 for r in rows if r["already_active"])
    write(f"**{len(rows)} scored stage changes.**\n")
    if flagged:
        write(f"In {flagged} of them ({flagged / len(rows):.1%}) the per-second model was "
              f"already emitting the incoming stage when the search window opened, "
              f"{args.max_early} s before the scored change. Those are more likely false "
              f"positives during the outgoing stage than genuine early detections, so the "
              f"clean subset below excludes them.\n")

    clean = [r for r in rows if not r["already_active"]]
    per_second = summarise([r["per_second"] for r in rows])
    epoch = summarise([r["epoch"] for r in rows])

    write("## Overall\n")
    write("| System | Detected | Median delay | Mean delay | IQR |")
    write("|---|---:|---:|---:|---|")
    for label, stats in (("Per-second (1 Hz)", per_second), ("30-second majority", epoch)):
        if stats is None:
            write(f"| {label} | 0 / {len(rows)} | n/a | n/a | n/a |")
            continue
        write(f"| {label} | {stats['n']} / {len(rows)} | {stats['median']:.0f} s | "
              f"{stats['mean']:.1f} s | {stats['p25']:.0f} to {stats['p75']:.0f} s |")
    write("")

    if flagged and clean:
        write("### Excluding changes the model was already calling\n")
        write("| System | Detected | Median delay | Mean delay | IQR |")
        write("|---|---:|---:|---:|---|")
        for label, stats in (("Per-second (1 Hz)", summarise([r["per_second"] for r in clean])),
                             ("30-second majority", summarise([r["epoch"] for r in clean]))):
            if stats is None:
                write(f"| {label} | 0 / {len(clean)} | n/a | n/a | n/a |")
                continue
            write(f"| {label} | {stats['n']} / {len(clean)} | {stats['median']:.0f} s | "
                  f"{stats['mean']:.1f} s | {stats['p25']:.0f} to {stats['p75']:.0f} s |")
        write("")
        write("This is the conservative reading, and the one to quote.\n")

    both = [r for r in clean if r["per_second"] is not None and r["epoch"] is not None]
    if both:
        earlier = sum(1 for r in both if r["per_second"] < r["epoch"])
        same = sum(1 for r in both if r["per_second"] == r["epoch"])
        gain = np.array([r["epoch"] - r["per_second"] for r in both], dtype=float)
        write("## Head to head\n")
        write(f"Restricted to the {len(both)} changes both systems reported, excluding any "
              f"the model was already calling before the window opened.\n")
        write("| Outcome | Count | Share |")
        write("|---|---:|---:|")
        write(f"| Per-second reports earlier | {earlier} | {earlier / len(both):.1%} |")
        write(f"| Same second | {same} | {same / len(both):.1%} |")
        write(f"| 30-second reports earlier | {len(both) - earlier - same} | "
              f"{(len(both) - earlier - same) / len(both):.1%} |")
        write("")
        write(f"Median time saved by the per-second output: **{np.median(gain):.0f} s** "
              f"(mean {gain.mean():.1f} s).\n")

    write("## By stage entered\n")
    write("| Stage | Changes | Per-second median | 30-second median | Difference |")
    write("|---|---:|---:|---:|---:|")
    for index, stage in enumerate(STAGES):
        subset = [r for r in (clean or rows) if r["stage"] == index]
        if not subset:
            continue
        a = summarise([r["per_second"] for r in subset])
        b = summarise([r["epoch"] for r in subset])
        a_text = f"{a['median']:.0f} s" if a else "n/a"
        b_text = f"{b['median']:.0f} s" if b else "n/a"
        diff = f"{b['median'] - a['median']:+.0f} s" if (a and b) else "n/a"
        write(f"| {stage} | {len(subset)} | {a_text} | {b_text} | {diff} |")
    write("")

    write("## Reading this\n")
    write("A negative or small positive per-second median means the model commits to the "
          "new stage inside the epoch the technician assigned it to, which an epoch-level "
          "system cannot do at all. That is the concrete argument for the finer output "
          "rate, expressed in seconds rather than in agreement.\n")
    write("Two caveats belong with any figure quoted from this table. The scored boundary "
          "is itself only located to the nearest 30 s, so these numbers describe latency "
          "against the *scored* transition and not against the physiological one. And a "
          "hold requirement trades detection speed against false alarms; sweep `--hold` "
          "before quoting a single number.\n")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="Prediction .npz files, or log directories containing them.")
    parser.add_argument("--pattern", default="fold_*_predictions.npz",
                        help="Glob used inside a directory.")
    parser.add_argument("--hold", type=int, default=10,
                        help="Seconds the new stage must persist to count as reported.")
    parser.add_argument("--max-early", type=int, default=30,
                        help="How far before the scored change to look. One epoch by "
                             "default, matching the resolution of the scored boundary "
                             "itself; a wider window mostly collects false positives.")
    parser.add_argument("--max-late", type=int, default=300,
                        help="How far after the scored change to look before giving up.")
    parser.add_argument("--out", default=None, help="Write the report here.")
    parser.add_argument("--quiet", action="store_true", help="File only, no console output.")
    args = parser.parse_args()

    if args.hold < 1:
        raise SystemExit("--hold must be at least 1 second")
    if args.max_late < args.hold:
        raise SystemExit("--max-late must be at least --hold seconds")

    folds = load_folds(args.paths, args.pattern)
    report = render(folds, args)

    if not args.quiet:
        print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
