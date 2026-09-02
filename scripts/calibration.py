"""How well the per-second probabilities know when they are wrong.

A real-time triage system has to be able to say "I am unsure", and a model that reports 0.9
confidence on predictions that are right 60% of the time cannot. This measures that gap:
expected and maximum calibration error, a reliability table, the Brier score, and the same
broken down per stage, from the prediction dump written by `scripts/warm_start_eval.py`.

    python scripts/calibration.py --predictions logs_warm/fold_0_predictions.npz \
        --out results/calibration_fold_0.md --figure figures/fig_calibration.png

Both the cold and warm scorings in the dump are reported, because context changes confidence
as well as accuracy. Nothing here needs a forward pass.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGES = ["W", "N1", "N2", "N3", "REM"]


def reliability(confidence, correct, n_bins=10):
    """Bin predictions by confidence and report accuracy within each bin.

    Returns one row per non-empty bin: ``(low, high, count, mean_confidence, accuracy)``.
    Equal-width bins on [0, 1]; a bin nobody landed in is omitted rather than reported as zero.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    if len(confidence) == 0:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so that a confidence of exactly 1.0 lands in the top bin, not outside it
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append((float(edges[b]), float(edges[b + 1]), count,
                     float(confidence[mask].mean()), float(correct[mask].mean())))
    return rows


def calibration_error(rows, total):
    """Expected and maximum calibration error from reliability rows."""
    if not rows or total == 0:
        return float("nan"), float("nan")
    ece = sum(count / total * abs(accuracy - mean_conf)
              for _, _, count, mean_conf, accuracy in rows)
    mce = max(abs(accuracy - mean_conf) for _, _, _, mean_conf, accuracy in rows)
    return float(ece), float(mce)


def brier_score(probabilities, targets, num_classes=5):
    """Multi-class Brier score: mean squared error against the one-hot truth."""
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(targets)), targets] = 1.0
    return float(((probabilities - onehot) ** 2).sum(axis=1).mean())


def analyse(probabilities, targets, n_bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets)
    preds = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = preds == targets

    rows = reliability(confidence, correct, n_bins)
    ece, mce = calibration_error(rows, len(targets))

    per_stage = []
    for stage in range(probabilities.shape[1]):
        mask = preds == stage
        if not mask.any():
            per_stage.append((stage, 0, float("nan"), float("nan"), float("nan")))
            continue
        stage_rows = reliability(confidence[mask], correct[mask], n_bins)
        stage_ece, _ = calibration_error(stage_rows, int(mask.sum()))
        per_stage.append((stage, int(mask.sum()), float(confidence[mask].mean()),
                          float(correct[mask].mean()), stage_ece))

    return {
        "n": len(targets),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "ece": ece,
        "mce": mce,
        "brier": brier_score(probabilities, targets, probabilities.shape[1]),
        "rows": rows,
        "per_stage": per_stage,
    }


def render(results, source):
    lines = ["# Calibration of the per-second probabilities\n",
             f"From `{os.path.basename(source)}`. Expected calibration error is the "
             "count-weighted gap between confidence and accuracy; a well-calibrated model "
             "predicting 0.8 is right 80% of the time.\n",
             "| Scoring | Seconds | Accuracy | Mean confidence | ECE | MCE | Brier |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results:
        lines.append(f"| {name} | {r['n']:,} | {r['accuracy']:.4f} | {r['mean_confidence']:.4f} | "
                     f"{r['ece']:.4f} | {r['mce']:.4f} | {r['brier']:.4f} |")
    lines.append("")
    for name, r in results:
        gap = r["mean_confidence"] - r["accuracy"]
        direction = "over-confident" if gap > 0 else "under-confident"
        lines.append(f"## {name}\n")
        lines.append(f"Mean confidence exceeds accuracy by {gap:+.4f}, so the model is "
                     f"{direction} overall.\n")
        lines.append("| Confidence bin | Seconds | Mean confidence | Accuracy | Gap |")
        lines.append("|---|---:|---:|---:|---:|")
        for low, high, count, mean_conf, accuracy in r["rows"]:
            lines.append(f"| {low:.1f}-{high:.1f} | {count:,} | {mean_conf:.4f} | "
                         f"{accuracy:.4f} | {accuracy - mean_conf:+.4f} |")
        lines.append("")
        lines.append("| Predicted stage | Seconds | Mean confidence | Accuracy | ECE |")
        lines.append("|---|---:|---:|---:|---:|")
        for stage, count, mean_conf, accuracy, ece in r["per_stage"]:
            if count == 0:
                lines.append(f"| {STAGES[stage]} | 0 | n/a | n/a | n/a |")
                continue
            lines.append(f"| {STAGES[stage]} | {count:,} | {mean_conf:.4f} | {accuracy:.4f} | "
                         f"{ece:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect")
    for (name, r), colour in zip(results, ["#0072B2", "#D55E00"]):
        if not r["rows"]:
            continue
        x = [mean_conf for _, _, _, mean_conf, _ in r["rows"]]
        y = [accuracy for _, _, _, _, accuracy in r["rows"]]
        ax.plot(x, y, marker="o", color=colour, label=f"{name} (ECE {r['ece']:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability, per-second predictions")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="an .npz from scripts/warm_start_eval.py --save_predictions")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    data = np.load(args.predictions, allow_pickle=True)
    targets = data["targets"]
    results = []
    for key, name in (("cold_probabilities", "cold"), ("warm_probabilities", "warm")):
        if key in data.files:
            results.append((name, analyse(data[key], targets, args.bins)))
    if not results:
        raise SystemExit(f"{args.predictions} holds no probability arrays")

    text = render(results, args.predictions)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(text)
        print(f"written to {args.out}")
    if args.figure:
        figure(results, args.figure)
        print(f"figure written to {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
