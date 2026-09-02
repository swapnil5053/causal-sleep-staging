"""Pool the per-fold causality differences across seeds into one paired test.

`analysis_stats.py --pair` reports each run pair separately. When the same comparison has
been repeated under several seeds, the honest headline is the pooled test over all
seeds x folds paired measurements, which is what this produces.

Per-seed results use the standard naive paired t-test: each seed's 5 folds are one
resampling and behave close enough to independent for a within-seed test. The pooled
figure across seeds uses the Nadeau-Bengio correction, because pooling folds across
seeds means those folds still share overlapping training subjects, which understates
variance under the naive assumption of independence. See Nadeau & Bengio (2003) and
the repeated k-fold extension in Bouckaert & Frank (2004).

    python scripts/pool_seeds.py \
        --seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42 \
        --seed 43=logs_78streaming_causal_s43,logs_78streaming_noncausal_s43 \
        --seed 44=logs_78streaming_causal_s44,logs_78streaming_noncausal_s44 \
        --out results/statistics_streaming_pooled.md

Each directory must contain a `test_metrics_summary.csv` written by `src/eval/evaluate.py`.
Exits non-zero if a pair is missing folds or the two arms disagree on which folds are present,
so a half-finished run cannot be reported as a complete one.
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats


def read_kappas(directory):
    """Return {fold_index: kappa} from a run directory's summary CSV."""
    path = os.path.join(directory, "test_metrics_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No test_metrics_summary.csv in {directory}")

    kappas = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["fold"])
            if fold in kappas:
                raise ValueError(
                    f"{path} lists fold {fold} more than once. evaluate.py appends, so a "
                    f"re-run duplicates rows; delete the file and evaluate every fold again."
                )
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


def naive_t_test(diffs):
    """Standard paired t-test treating all measurements as independent."""
    diffs = np.asarray(diffs, dtype=float)
    t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
    return float(t_stat), float(p_value)


def nadeau_bengio_t_test(diffs, k_folds):
    """Corrected paired t-test for POOLING across repeated k-fold CV runs.

    Folds pooled across seeds still share overlapping training subjects within each
    seed's resampling, so treating all n pooled measurements as fully independent
    understates the standard error. This inflates the variance estimate by 1/(k-1) -
    the train/test size ratio for leave-one-fold-out - on top of the usual 1/n term.

    Returns (t_stat, p_value, corrected_se).
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n < 2:
        raise ValueError("need at least 2 measurements for a t-test")
    mean_diff = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    correction = 1.0 / n + 1.0 / (k_folds - 1)
    se = sd * np.sqrt(correction)
    if se == 0:
        raise ValueError("zero variance in differences, cannot compute t-statistic")
    t_stat = mean_diff / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    return t_stat, p_value, se


def main():
    parser = argparse.ArgumentParser(
        description="Pool causal vs non-causal per-fold kappa across seeds.")
    parser.add_argument("--seed", action="append", required=True, type=parse_seed,
                        metavar="NAME=CAUSAL_DIR,NONCAUSAL_DIR",
                        help="One seed's run pair. Repeatable.")
    parser.add_argument("--out", default=None, help="Write a markdown report here.")
    parser.add_argument("--quiet", action="store_true", help="File only, no console output.")
    args = parser.parse_args()

    per_seed = []
    differences = []

    for name, causal_dir, noncausal_dir in args.seed:
        causal = read_kappas(causal_dir)
        noncausal = read_kappas(noncausal_dir)

        if set(causal) != set(noncausal):
            raise ValueError(
                f"Seed {name}: folds differ between arms "
                f"(causal {sorted(causal)}, non-causal {sorted(noncausal)}). "
                f"Both arms must be evaluated on the same folds before pooling.")

        folds = sorted(causal)
        seed_diffs = [causal[f] - noncausal[f] for f in folds]
        differences.extend(seed_diffs)

        seed_t, seed_p = naive_t_test(seed_diffs)

        per_seed.append({
            "name": name,
            "folds": folds,
            "causal": float(np.mean([causal[f] for f in folds])),
            "noncausal": float(np.mean([noncausal[f] for f in folds])),
            "difference": float(np.mean(seed_diffs)),
            "losses": sum(1 for d in seed_diffs if d < 0),
            "t": seed_t,
            "p": seed_p,
        })

    fold_counts = {len(s["folds"]) for s in per_seed}
    if len(fold_counts) != 1:
        print(f"warning: seeds contribute different fold counts {sorted(fold_counts)}; "
              f"the pooled mean is therefore unevenly weighted.", file=sys.stderr)
    k_folds = sorted(fold_counts)[0]

    diffs = np.asarray(differences, dtype=float)
    n = diffs.size
    mean_difference = float(diffs.mean())

    # Pooled test uses the Nadeau-Bengio correction (see module docstring).
    t_stat, p_value, corrected_se = nadeau_bengio_t_test(diffs, k_folds=k_folds)

    # Naive pooled test kept for comparison, so the correction's effect is visible.
    naive_pooled_t, naive_pooled_p = naive_t_test(diffs)

    try:
        _, wilcoxon_p = stats.wilcoxon(diffs)
    except ValueError:
        wilcoxon_p = float("nan")

    cohens_d = mean_difference / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else float("nan")
    wilcoxon_floor = 2.0 / (2 ** n)

    rng = np.random.default_rng(0)
    boot = rng.choice(diffs, size=(20000, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    losses = int((diffs < 0).sum())

    lines = ["# Pooled causality statistics", ""]
    lines.append(f"{len(per_seed)} seeds x {k_folds} folds = **{n} paired measurements**. "
                 f"A negative difference means the causal model scores lower.")
    lines.append("")
    lines.append("Per-seed results use the naive paired t-test (5 folds within one seed). "
                 "The pooled figure across seeds uses the Nadeau-Bengio correction, since "
                 "folds pooled across seeds still share overlapping training subjects and "
                 "the naive pooled test overstates significance.")
    lines.append("")
    lines.append("## Per-seed results (lead with these)\n")
    lines.append("| Seed | Folds | Causal | Non-causal | Difference | Causal loses | t | p |")
    lines.append("|---|---:|---|---|---|---|---|---|")
    for s in per_seed:
        lines.append(f"| {s['name']} | {len(s['folds'])} | {s['causal']:.4f} | "
                     f"{s['noncausal']:.4f} | {s['difference']:+.4f} | "
                     f"{s['losses']}/{len(s['folds'])} | {s['t']:.2f} | {s['p']:.4f} |")
    lines.append("")
    pooled_causal = float(np.mean([s["causal"] for s in per_seed]))
    pooled_noncausal = float(np.mean([s["noncausal"] for s in per_seed]))
    lines.append("## Pooled\n")
    lines.append(f"- Causal: {pooled_causal:.4f}, Non-causal: {pooled_noncausal:.4f}, "
                 f"Difference: {mean_difference:+.4f}, Causal loses: {losses}/{n}")
    lines.append(f"- **Nadeau-Bengio corrected: t({n - 1}) = {t_stat:.2f}, "
                 f"p = {p_value:.4g}**")
    lines.append(f"- Naive pooled t-test (folds treated as fully independent, for "
                 f"comparison): t({n - 1}) = {naive_pooled_t:.2f}, p = {naive_pooled_p:.3g}")
    lines.append(f"- Wilcoxon signed-rank: p = {wilcoxon_p:.4f}"
                 + (f" (floor at n = {n} is {wilcoxon_floor:.4f} when every difference "
                    f"shares a sign)" if losses in (0, n) else ""))
    lines.append(f"- Bootstrap 95% CI on the mean difference: "
                 f"[{ci_low:+.4f}, {ci_high:+.4f}]")
    lines.append(f"- Cohen's d = {cohens_d:.2f}")
    lines.append("")
    seed_means = [s["difference"] for s in per_seed]
    if len(seed_means) > 1:
        lines.append(f"- Seed-to-seed spread of the effect: "
                     f"{min(seed_means):+.4f} to {max(seed_means):+.4f} "
                     f"(sd {np.std(seed_means, ddof=1):.4f})")
        lines.append("")
    lines.append("The corrected pooled test is more conservative because it accounts for "
                 "shared training subjects across folds. The bootstrap interval and "
                 "per-seed results are the more reliable summary; Cohen's d is inflated "
                 "at small paired sample sizes.")

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
