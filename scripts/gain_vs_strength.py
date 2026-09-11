"""Protocol gain against absolute score, across arms.

The predictable objection to the protocol result is that these models are small and lean on
accumulated context, so a stronger model would not pay the cold-start penalty. This tabulates
every arm on disk: its absolute score, and its own protocol gain.

The gain is within-arm, the same weights scored on the same held-out seconds under two
protocols, so arms that are not comparable to each other still contribute comparable gains.

Causal and control arms are reported separately and only causal arms are correlated. Mixing them
would be circular: a control arm scores higher and gains less by construction, so a correlation
across the pooled set restates the causal-control difference rather than measuring anything about
model strength.

    python scripts/gain_vs_strength.py \\
        --arm "TCN causal, Sleep-EDF-78=logs_78streaming_causal_s42" \\
        --arm "TCN causal wide, Sleep-EDF-78=logs_78streaming_causal_wide_s42" \\
        --arm "GRU causal, Sleep-EDF-78=logs_78streaming_gru_s42" \\
        --arm "TCN causal, DOD-H=results/dodh_causal_s42" \\
        --control "TCN control, Sleep-EDF-78=logs_78streaming_noncausal_s42" \\
        --control "GRU control, Sleep-EDF-78=logs_78streaming_gru_noncausal_s42" \\
        --control "TCN control, DOD-H=results/dodh_noncausal_s42" \\
        --out results/generated/gain_vs_strength.md

Each directory must hold both `test_metrics_summary.csv` and
`test_metrics_summary_streaming<N>.csv`.

A correlation here only means something if the causal arms actually differ in absolute score. The
report states the span so that a flat one cannot be read as evidence of independence. At the time
of writing the four causal arms span 0.008 kappa, which is no variation at all, so the table is
descriptive and the correlation is reported only to show it resolves nothing.
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


def summarise(label, directory, kind, tiled_name, streaming_name):
    tiled = read_kappas(directory, tiled_name)
    streaming = read_kappas(directory, streaming_name)
    if set(tiled) != set(streaming):
        raise SystemExit(f"{directory}: folds differ between protocols")
    folds = sorted(tiled)
    return {
        "label": label,
        "kind": kind,
        "folds": len(folds),
        "tiled": float(np.mean([tiled[f] for f in folds])),
        "streaming": float(np.mean([streaming[f] for f in folds])),
        "gain": float(np.mean([streaming[f] - tiled[f] for f in folds])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Protocol gain against absolute score, causal and control reported apart.")
    parser.add_argument("--arm", action="append", default=[], type=parse_arm,
                        metavar="LABEL=DIR", help="A causal arm. Repeatable.")
    parser.add_argument("--control", action="append", default=[], type=parse_arm,
                        metavar="LABEL=DIR", help="A non-causal control arm. Repeatable.")
    parser.add_argument("--stream_stride", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.arm:
        raise SystemExit("at least one --arm is required")

    tiled_name = "test_metrics_summary.csv"
    streaming_name = f"test_metrics_summary_streaming{args.stream_stride}.csv"

    causal = [summarise(lbl, d, "causal", tiled_name, streaming_name) for lbl, d in args.arm]
    control = [summarise(lbl, d, "control", tiled_name, streaming_name) for lbl, d in args.control]

    lines = ["# Protocol gain against absolute score", ""]
    lines.append("Each row is one arm: one set of trained weights, scored on its own held-out")
    lines.append("seconds under both protocols. The gain is within-arm, so arms that are not")
    lines.append("comparable to each other still contribute comparable gains.")
    lines.append("")
    lines.append("| Arm | Kind | Folds | Tiled | Streaming | Gain |")
    lines.append("|---|---|---:|---|---|---|")
    for r in sorted(causal, key=lambda x: x["gain"]) + sorted(control, key=lambda x: x["gain"]):
        lines.append(f"| {r['label']} | {r['kind']} | {r['folds']} | {r['tiled']:.4f} | "
                     f"{r['streaming']:.4f} | {r['gain']:+.4f} |")
    lines.append("")

    gains = np.array([r["gain"] for r in causal])
    strength = np.array([r["streaming"] for r in causal])
    span = float(strength.max() - strength.min())

    lines.append(f"- {len(causal)} causal arms, gains {gains.min():+.4f} to {gains.max():+.4f}, "
                 f"spread {gains.max() - gains.min():.4f}")
    if control:
        cg = np.array([r["gain"] for r in control])
        lines.append(f"- {len(control)} control arms, gains {cg.min():+.4f} to {cg.max():+.4f}")
    lines.append(f"- Absolute score across the causal arms spans {span:.4f} kappa, "
                 f"{strength.min():.4f} to {strength.max():.4f}")
    lines.append("")

    if len(causal) < 3:
        lines.append("Fewer than three causal arms, so no correlation is reported.")
    else:
        r_val, p_val = stats.pearsonr(strength, gains)
        rho, p_rho = stats.spearmanr(strength, gains)
        lines.append(f"- Across causal arms: Pearson r = {r_val:+.3f} (p = {p_val:.3g}), "
                     f"Spearman rho = {rho:+.3f} (p = {p_rho:.3g})")
        lines.append("")
        if span < 0.02:
            lines.append("**The correlation above resolves nothing and should not be quoted.** The")
            lines.append("causal arms differ by less than 0.02 kappa in absolute score, so there is")
            lines.append("no variation in strength to correlate a gain against. What the table does")
            lines.append("support is the narrower statement: across these arms, spanning two")
            lines.append("datasets, two architecture families and a range of capacities, the")
            lines.append("protocol gain is nearly constant.")
        else:
            lines.append("The causal arms differ enough in absolute score for the correlation to")
            lines.append("carry information. Read it with the sample size in mind.")

    lines.append("")
    lines.append("Control arms are excluded from the correlation deliberately. A control scores")
    lines.append("higher and gains less by construction, so pooling the two kinds would restate the")
    lines.append("causal and control difference as though it were a relationship with strength.")

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
