"""Does the tiling penalty shrink as the model gets stronger?

The obvious objection to the protocol result is that these are small models which lean
heavily on accumulated context, so a stronger model would not pay the cold-start cost.
This answers it with the runs already on disk: for every arm, on every dataset, at every
architecture and capacity, it reports that arm's own protocol gain alongside its absolute
score, and tests whether the two are related.

The gain is a within-arm quantity, the same weights scored on the same held-out seconds
under two protocols, so arms that are not comparable to each other still contribute
comparable gains.

    python scripts/gain_vs_strength.py \\
        --arm "TCN causal, Sleep-EDF-78=logs_78streaming_causal_s42" \\
        --arm "TCN control, Sleep-EDF-78=logs_78streaming_noncausal_s42" \\
        --arm "TCN causal, DOD-H=results/dodh_causal_s42" \\
        --arm "TCN control, DOD-H=results/dodh_noncausal_s42" \\
        --arm "GRU causal, Sleep-EDF-78=logs_78streaming_gru_s42" \\
        --arm "GRU control, Sleep-EDF-78=logs_78streaming_gru_noncausal_s42" \\
        --out results/generated/gain_vs_strength.md

Each directory must hold both `test_metrics_summary.csv` and
`test_metrics_summary_streaming<N>.csv`.

Read the result conservatively. A handful of arms is a small sample and the absence of a
relationship is not proof that one does not exist. What it does support is the narrower,
honest sentence: across the models measured here, spanning a range of absolute scores and
two architecture families, the protocol gain shows no tendency to shrink with model
strength.
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats


def read_kappas(directory, filename):
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {filename} in {directory}")
    kappas = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["fold"])
            if fold in kappas:
                raise ValueError(f"{path} lists fold {fold} twice; delete it and re-evaluate")
            kappas[fold] = float(row["kappa"])
    if not kappas:
        raise ValueError(f"{path} contains no rows")
    return kappas


def parse_arm(argument):
    label, _, directory = argument.partition("=")
    if not (label and directory):
        raise argparse.ArgumentTypeError(f"Expected LABEL=DIR, got {argument!r}")
    return label.strip(), directory.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Protocol gain against absolute score, across arms.")
    parser.add_argument("--arm", action="append", required=True, type=parse_arm,
                        metavar="LABEL=DIR", help="One run directory. Repeatable.")
    parser.add_argument("--stream_stride", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tiled_name = "test_metrics_summary.csv"
    streaming_name = f"test_metrics_summary_streaming{args.stream_stride}.csv"

    rows = []
    fold_strength = []
    fold_gain = []

    for label, directory in args.arm:
        tiled = read_kappas(directory, tiled_name)
        streaming = read_kappas(directory, streaming_name)
        if set(tiled) != set(streaming):
            raise SystemExit(f"{directory}: folds differ between protocols")
        folds = sorted(tiled)
        gains = [streaming[f] - tiled[f] for f in folds]
        rows.append({
            "label": label,
            "folds": len(folds),
            "tiled": float(np.mean([tiled[f] for f in folds])),
            "streaming": float(np.mean([streaming[f] for f in folds])),
            "gain": float(np.mean(gains)),
        })
        fold_strength.extend(streaming[f] for f in folds)
        fold_gain.extend(gains)

    if len(rows) < 3:
        raise SystemExit("need at least 3 arms for a correlation to mean anything")

    arm_strength = np.array([r["streaming"] for r in rows])
    arm_gain = np.array([r["gain"] for r in rows])
    r_arm, p_arm = stats.pearsonr(arm_strength, arm_gain)
    rho_arm, p_rho = stats.spearmanr(arm_strength, arm_gain)
    r_fold, p_fold = stats.pearsonr(np.array(fold_strength), np.array(fold_gain))

    lines = ["# Protocol gain against absolute score", ""]
    lines.append("Each row is one arm: one set of trained weights, scored on its own held-out")
    lines.append("seconds under both protocols. The gain is within-arm, so arms that are not")
    lines.append("comparable to each other still contribute comparable gains.")
    lines.append("")
    lines.append("| Arm | Folds | Tiled | Streaming | Gain |")
    lines.append("|---|---:|---|---|---|")
    for r in sorted(rows, key=lambda x: x["streaming"]):
        lines.append(f"| {r['label']} | {r['folds']} | {r['tiled']:.4f} | "
                     f"{r['streaming']:.4f} | {r['gain']:+.4f} |")
    lines.append("")
    lines.append(f"- Absolute score spans {arm_strength.min():.4f} to {arm_strength.max():.4f} kappa "
                 f"across {len(rows)} arms")
    lines.append(f"- Gain spans {arm_gain.min():+.4f} to {arm_gain.max():+.4f}")
    lines.append(f"- Across arms: Pearson r = {r_arm:+.3f} (p = {p_arm:.3g}), "
                 f"Spearman rho = {rho_arm:+.3f} (p = {p_rho:.3g})")
    lines.append(f"- Across all {len(fold_gain)} individual folds: Pearson r = {r_fold:+.3f} "
                 f"(p = {p_fold:.3g}); folds within an arm are not independent, so this is "
                 f"descriptive only")
    lines.append("")
    if p_arm > 0.05:
        lines.append("No relationship is resolved at this sample size. That is not evidence of")
        lines.append("absence: with this few arms the test has little power. The defensible")
        lines.append("statement is that across the models measured here the protocol gain shows no")
        lines.append("tendency to shrink as absolute score rises.")
    else:
        lines.append("A relationship is resolved. Report it: the size of the tiling penalty is not")
        lines.append("independent of how strong the model is, which bounds the claim.")

    report = "\n".join(lines) + "\n"
    if not args.quiet:
        print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
