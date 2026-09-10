"""Within-arm protocol gain, and the excess of that gain to the causal arm.

`pool_seeds.py` answers a between-arm question: how far apart are the two arms under one
protocol. This answers the within-arm one: how much does each arm gain when the same
weights are scored on the same held-out seconds under the streaming protocol instead of
the tiled one, and how much larger is that gain for the causal arm.

    gain(arm, fold)  = kappa_streaming(arm, fold) - kappa_tiled(arm, fold)
    excess(fold)     = gain(causal, fold) - gain(non-causal, fold)

The excess is the paper's decisive statistic. It does not require the two arms to be
comparable, because each gain is measured on one set of weights against itself, so model
variance drops out of the comparison entirely.

    python scripts/protocol_excess.py \\
        --seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42 \\
        --seed 43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43 \\
        --seed 44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44 \\
        --out results/generated/protocol_excess_sleep78.md

Each run directory must hold both `test_metrics_summary.csv` (tiled) and
`test_metrics_summary_streaming<N>.csv`, written by `src/eval/evaluate.py` with and
without `--stream_stride N`.

Before trusting this on a new experiment, run it on the published Sleep-EDF-78 and DOD-H
directories. It must return +0.0314 with corrected t(14) = 12.69 and +0.0292 with
corrected t(14) = 6.48. Anything else means the reader disagrees with the numbers already
in the papers and must be fixed before it is used.

Exits non-zero if a fold is missing from any of the four CSVs a seed needs, so a
half-finished run cannot be reported as a complete one.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pool_seeds import nadeau_bengio_t_test  # noqa: E402


def read_kappas(directory, filename):
    """Return {fold_index: kappa} from one summary CSV in a run directory."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {filename} in {directory}")

    kappas = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["fold"])
            if fold in kappas:
                raise ValueError(
                    f"{path} lists fold {fold} more than once. evaluate.py appends, so a "
                    f"re-run duplicates rows; delete the file and evaluate every fold again.")
            kappas[fold] = float(row["kappa"])
    if not kappas:
        raise ValueError(f"{path} contains no rows")
    return kappas


def parse_seed(argument):
    name, _, dirs = argument.partition("=")
    causal_dir, _, noncausal_dir = dirs.partition(",")
    if not (name and causal_dir and noncausal_dir):
        raise argparse.ArgumentTypeError(
            f"Expected NAME=CAUSAL_DIR,NONCAUSAL_DIR, got {argument!r}")
    return name, causal_dir, noncausal_dir


def gains(directory, tiled_name, streaming_name):
    """Return {fold: kappa_streaming - kappa_tiled} for one run directory."""
    tiled = read_kappas(directory, tiled_name)
    streaming = read_kappas(directory, streaming_name)
    if set(tiled) != set(streaming):
        raise ValueError(
            f"{directory}: folds differ between protocols "
            f"(tiled {sorted(tiled)}, streaming {sorted(streaming)}). Evaluate every fold "
            f"under both protocols before pooling.")
    return {f: streaming[f] - tiled[f] for f in tiled}


def main():
    parser = argparse.ArgumentParser(
        description="Within-arm protocol gain and the excess to the causal arm.")
    parser.add_argument("--seed", action="append", required=True, type=parse_seed,
                        metavar="NAME=CAUSAL_DIR,NONCAUSAL_DIR",
                        help="One seed's run pair. Repeatable.")
    parser.add_argument("--stream_stride", type=int, default=30,
                        help="The N used at evaluation time, which names the streaming CSV.")
    parser.add_argument("--label", default="",
                        help="Name for the comparison, used in the report heading.")
    parser.add_argument("--out", default=None, help="Write a markdown report here.")
    parser.add_argument("--quiet", action="store_true", help="File only, no console output.")
    args = parser.parse_args()

    tiled_name = "test_metrics_summary.csv"
    streaming_name = f"test_metrics_summary_streaming{args.stream_stride}.csv"

    per_seed = []
    causal_gains = []
    noncausal_gains = []
    excesses = []

    for name, causal_dir, noncausal_dir in args.seed:
        causal = gains(causal_dir, tiled_name, streaming_name)
        noncausal = gains(noncausal_dir, tiled_name, streaming_name)
        if set(causal) != set(noncausal):
            raise ValueError(
                f"Seed {name}: folds differ between arms "
                f"(causal {sorted(causal)}, non-causal {sorted(noncausal)}).")

        folds = sorted(causal)
        seed_excess = [causal[f] - noncausal[f] for f in folds]
        causal_gains.extend(causal[f] for f in folds)
        noncausal_gains.extend(noncausal[f] for f in folds)
        excesses.extend(seed_excess)
        per_seed.append({
            "name": name,
            "folds": folds,
            "causal": float(np.mean([causal[f] for f in folds])),
            "noncausal": float(np.mean([noncausal[f] for f in folds])),
            "excess": float(np.mean(seed_excess)),
            "positive": sum(1 for d in seed_excess if d > 0),
        })

    fold_counts = {len(s["folds"]) for s in per_seed}
    if len(fold_counts) != 1:
        raise SystemExit(
            f"seeds contribute different fold counts {sorted(fold_counts)}; the corrected "
            f"test assumes one k-fold design")
    k_folds = fold_counts.pop()

    excess = np.asarray(excesses, dtype=float)
    n = excess.size
    mean_excess = float(excess.mean())
    positive = int((excess > 0).sum())
    smallest = float(excess.min())

    t_stat, p_value, corrected_se = nadeau_bengio_t_test(excess, k_folds=k_folds)

    rng = np.random.default_rng(0)
    boot = rng.choice(excess, size=(20000, n), replace=True).mean(axis=1)
    ci_low, ci_high = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    heading = "Protocol gain by arm"
    if args.label:
        heading = f"{heading}: {args.label}"

    lines = [f"# {heading}", ""]
    lines.append(f"Streaming stride {args.stream_stride} s against the tiled protocol. "
                 f"{len(per_seed)} seeds x {k_folds} folds = **{n} paired measurements**. "
                 f"A positive excess means the protocol change is worth more to the causal arm.")
    lines.append("")
    lines.append("| Seed | Causal arm gains | Non-causal arm gains | Excess | Folds positive |")
    lines.append("|---|---|---|---|---|")
    for s in per_seed:
        lines.append(f"| {s['name']} | {s['causal']:+.4f} | {s['noncausal']:+.4f} | "
                     f"{s['excess']:+.4f} | {s['positive']}/{len(s['folds'])} |")
    lines.append(f"| **Pooled** | **{np.mean(causal_gains):+.4f}** | "
                 f"**{np.mean(noncausal_gains):+.4f}** | **{mean_excess:+.4f}** | "
                 f"**{positive}/{n}** |")
    lines.append("")
    lines.append(f"- Nadeau-Bengio corrected: t({n - 1}) = {t_stat:.2f}, p = {p_value:.3g} "
                 f"(corrected SE {corrected_se:.5f}, factor 1/{n} + 1/{k_folds - 1})")
    lines.append(f"- Bootstrap 95% CI on the mean excess: [{ci_low:+.4f}, {ci_high:+.4f}]")
    lines.append(f"- Smallest single-fold excess: {smallest:+.4f}")
    seed_means = [s["excess"] for s in per_seed]
    if len(seed_means) > 1:
        lines.append(f"- Per-seed range: {min(seed_means):+.4f} to {max(seed_means):+.4f}")
    lines.append("")
    lines.append("Each gain is a within-arm quantity: the same trained weights scored on the "
                 "same held-out seconds under two protocols. The excess therefore does not "
                 "depend on the two arms being comparable to each other.")

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
