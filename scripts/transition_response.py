"""How quickly the model follows a stage change - what per-second output actually buys.

The paper claims one prediction per second matters, but never says what that gets you. This
answers it directly: at every scored stage transition, how many seconds pass before the model
switches to the new stage. A 30-second epoch system cannot answer faster than the next epoch
boundary, so the comparison against a majority-voted 30 s reading of the same predictions is
the number that makes the case.

    python scripts/transition_response.py --predictions logs_warm/fold_0_predictions.npz \
        --out results/transition_response.md

Reads the dump from `scripts/warm_start_eval.py`; no forward pass required. Transitions are
taken from the technician's labels, which are 30-second epoch labels replicated per second, so
a transition is located at the epoch boundary and the resolution of the ground truth is 30 s
even though the model's is 1 s. That limits what this can show and is stated in the output.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGES = ["W", "N1", "N2", "N3", "REM"]
NOT_DETECTED = -1


def find_transitions(targets, subjects=None):
    """Indices where the scored stage changes, never spanning two subjects."""
    targets = np.asarray(targets)
    if len(targets) < 2:
        return np.empty(0, dtype=int)
    changed = targets[1:] != targets[:-1]
    if subjects is not None:
        subjects = np.asarray(subjects)
        changed &= subjects[1:] == subjects[:-1]
    return np.flatnonzero(changed) + 1


def response_latency(preds, transitions, targets, horizon):
    """Seconds from each transition until the prediction first equals the new stage.

    Returns one entry per transition: the lag, or ``NOT_DETECTED`` if the model never reaches
    the new stage within ``horizon`` seconds. Never reads past the following transition, so a
    short stage cannot borrow credit from the one after it.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    latencies = []
    for i, start in enumerate(transitions):
        new_stage = targets[start]
        limit = min(len(preds), start + horizon)
        if i + 1 < len(transitions):
            limit = min(limit, transitions[i + 1])
        window = preds[start:limit]
        hit = np.flatnonzero(window == new_stage)
        latencies.append(int(hit[0]) if len(hit) else NOT_DETECTED)
    return np.asarray(latencies, dtype=int)


def epoch_majority(preds, epoch=30):
    """Collapse per-second predictions to one per 30 s epoch, then re-expand.

    This is what the same model would emit if it reported at conventional epoch resolution, so
    the difference between the two is what per-second output buys and nothing else.
    """
    preds = np.asarray(preds)
    usable = len(preds) - len(preds) % epoch
    if usable == 0:
        return preds.copy()
    blocks = preds[:usable].reshape(-1, epoch)
    voted = np.array([np.bincount(b, minlength=5).argmax() for b in blocks])
    out = np.repeat(voted, epoch)
    return np.concatenate([out, preds[usable:]])


def summarise(latencies):
    found = latencies[latencies != NOT_DETECTED]
    return {
        "transitions": int(len(latencies)),
        "detected": int(len(found)),
        "detected_fraction": float(len(found) / len(latencies)) if len(latencies) else float("nan"),
        "median": float(np.median(found)) if len(found) else float("nan"),
        "mean": float(found.mean()) if len(found) else float("nan"),
        "p90": float(np.percentile(found, 90)) if len(found) else float("nan"),
        "within_5s": float((found <= 5).mean()) if len(found) else float("nan"),
        "within_30s": float((found <= 30).mean()) if len(found) else float("nan"),
    }


def paired_gain(per_second, epoch):
    """Compare two latency arrays only on transitions both of them detected.

    Comparing the median of each series separately would be comparing different subsets of
    transitions whenever the detection rates differ, which they do. The paired difference is
    the only honest version of "how much earlier".
    """
    per_second = np.asarray(per_second)
    epoch = np.asarray(epoch)
    if per_second.shape != epoch.shape:
        raise ValueError("latency arrays must be aligned transition for transition")
    both = (per_second != NOT_DETECTED) & (epoch != NOT_DETECTED)
    if not both.any():
        return {"n": 0, "median_gain": float("nan"), "mean_gain": float("nan"),
                "earlier_fraction": float("nan")}
    difference = epoch[both] - per_second[both]          # positive = per-second is earlier
    return {
        "n": int(both.sum()),
        "median_gain": float(np.median(difference)),
        "mean_gain": float(difference.mean()),
        "earlier_fraction": float((difference > 0).mean()),
    }


def render(results, source, horizon, pairings=()):
    lines = ["# Response time at stage transitions\n",
             f"From `{os.path.basename(source)}`, horizon {horizon} s. Latency is the number of "
             "seconds after a scored stage change before the model first outputs the new "
             "stage.\n",
             "> The technician's labels are 30-second epoch labels replicated to every second, "
             "so a transition is only ever located to the nearest epoch boundary. A latency below "
             "30 s therefore means the model responds within the epoch in which the change was "
             "scored; it does not establish sub-epoch accuracy against a sub-epoch truth, "
             "because no such truth exists in this dataset.\n",
             "| Series | Transitions | Detected | Median | Mean | p90 | <=5 s | <=30 s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in results:
        lines.append(
            f"| {name} | {s['transitions']:,} | {s['detected_fraction']:.1%} | "
            f"{s['median']:.0f} s | {s['mean']:.1f} s | {s['p90']:.0f} s | "
            f"{s['within_5s']:.1%} | {s['within_30s']:.1%} |")
    lines.append("")
    lines.append("Detection rates differ between the series, so the medians above are taken over "
                 "different subsets of transitions and must not be subtracted from one another. "
                 "The comparison below is paired: it uses only transitions that both series "
                 "detected.\n")
    lines.append("| Comparison | Transitions both detected | Median gain | Mean gain | "
                 "Per-second earlier |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, gain in pairings:
        if gain["n"] == 0:
            lines.append(f"| {label} | 0 | n/a | n/a | n/a |")
            continue
        lines.append(f"| {label} | {gain['n']:,} | {gain['median_gain']:+.0f} s | "
                     f"{gain['mean_gain']:+.1f} s | {gain['earlier_fraction']:.1%} |")
    lines.append("")
    lines.append("A positive gain means per-second output reaches the new stage that many "
                 "seconds before the same predictions read at 30-second epoch resolution. That "
                 "is what the output rate buys, measured rather than asserted.\n")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="an .npz from scripts/warm_start_eval.py --save_predictions")
    ap.add_argument("--horizon", type=int, default=120,
                    help="give up this many seconds after a transition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.predictions, allow_pickle=True)
    targets = data["targets"]
    subjects = data["subjects"] if "subjects" in data.files else None
    transitions = find_transitions(targets, subjects)
    if len(transitions) == 0:
        raise SystemExit("no stage transitions in this dump")

    results, pairings = [], []
    for key, label in (("cold_probabilities", "cold"), ("warm_probabilities", "warm")):
        if key not in data.files:
            continue
        preds = data[key].argmax(axis=1)
        per_second = response_latency(preds, transitions, targets, args.horizon)
        epoch = response_latency(epoch_majority(preds), transitions, targets, args.horizon)
        results.append((f"per-second, {label}", summarise(per_second)))
        results.append((f"30 s majority vote, {label}", summarise(epoch)))
        pairings.append((f"per-second vs 30 s epochs, {label}", paired_gain(per_second, epoch)))
    if not results:
        raise SystemExit(f"{args.predictions} holds no probability arrays")

    text = render(results, args.predictions, args.horizon, pairings)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(text)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
