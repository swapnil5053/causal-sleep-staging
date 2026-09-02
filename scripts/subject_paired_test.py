"""Pair the causality comparison by subject instead of by fold.

The fold-level test has five paired measurements per seed, and pooling three seeds does not
honestly give fifteen: the same 78 subjects are reused, so the measurements share variance and
need a correction. Pairing by subject avoids the problem rather than correcting for it. Each
subject sits in the test set of exactly one fold, the causal and non-causal arms are trained on
the same partition, and so subject *s* yields one difference that is independent of every other
subject's. Seventy-eight of them, not five.

    python scripts/subject_paired_test.py \
        --pair "Streaming-78 seed 42=logs_78streaming_causal_s42,logs_78streaming_noncausal_s42" \
        --out results/statistics_streaming_by_subject.md

Each directory needs the `test_subject_metrics.csv` written by `src/eval/evaluate.py`. The
script refuses to run when the two arms cover different subjects, when a subject appears twice,
or when the arms were evaluated on different partitions, because in each of those cases the
rows are not paired and the test would be meaningless.

One seed at a time. Two seeds sharing a partition stage the same subject twice, and those two
differences are not independent of each other.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRIC_CHOICES = ("kappa", "accuracy", "macro_f1")


def read_subject_metrics(directory, metric):
    """Return ``{subject: value}`` and ``{subject: fold}`` from a run directory."""
    path = os.path.join(directory, "test_subject_metrics.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"No test_subject_metrics.csv in {directory}. Re-run src/eval/evaluate.py for "
            f"every fold; it writes one row per held-out subject.")

    values, folds = {}, {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            subject = row["subject"]
            if subject in values:
                raise SystemExit(
                    f"{path} lists subject {subject} more than once. evaluate.py appends, so a "
                    f"re-evaluated fold duplicates rows; delete the file and evaluate each "
                    f"fold once.")
            raw = row[metric].strip()
            if raw == "":
                continue           # a night with a single scored stage has no kappa
            values[subject] = float(raw)
            folds[subject] = int(row["fold"])
    if not values:
        raise SystemExit(f"{path} has no scoreable rows for metric '{metric}'")
    return values, folds


def bootstrap_ci(differences, n_boot=20000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean paired difference."""
    rng = np.random.default_rng(seed)
    d = np.asarray(differences)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def analyse(name, causal_dir, noncausal_dir, metric):
    causal, causal_folds = read_subject_metrics(causal_dir, metric)
    noncausal, noncausal_folds = read_subject_metrics(noncausal_dir, metric)

    only_causal = sorted(set(causal) - set(noncausal))
    only_noncausal = sorted(set(noncausal) - set(causal))
    if only_causal or only_noncausal:
        raise SystemExit(
            f"{name}: the two arms do not cover the same subjects, so the rows are not paired. "
            f"Only in {causal_dir}: {only_causal or 'none'}. "
            f"Only in {noncausal_dir}: {only_noncausal or 'none'}.")

    moved = sorted(s for s in causal if causal_folds[s] != noncausal_folds[s])
    if moved:
        raise SystemExit(
            f"{name}: subjects {moved} are in different folds in the two arms, so the arms were "
            f"trained on different partitions and the comparison is not controlled. Check that "
            f"both configs name the same train.split_seed.")

    subjects = sorted(causal)
    differences = [causal[s] - noncausal[s] for s in subjects]
    n = len(differences)
    if n < 2:
        raise SystemExit(f"{name}: need at least 2 paired subjects, found {n}")

    mean, sd = st.mean(differences), st.stdev(differences)
    if sd == 0:
        raise SystemExit(f"{name}: every paired difference is identical, no variance to test")
    t = mean / (sd / math.sqrt(n))
    lo, hi = bootstrap_ci(differences)

    p_t = p_w = None
    try:
        from scipy import stats
        p_t = float(stats.ttest_rel([causal[s] for s in subjects],
                                    [noncausal[s] for s in subjects]).pvalue)
        p_w = float(stats.wilcoxon([causal[s] for s in subjects],
                                   [noncausal[s] for s in subjects]).pvalue)
    except Exception:
        pass

    return dict(name=name, metric=metric, n=n, subjects=subjects,
                causal=causal, noncausal=noncausal, folds=causal_folds,
                differences=differences,
                mean_c=st.mean(causal[s] for s in subjects),
                mean_n=st.mean(noncausal[s] for s in subjects),
                mean=mean, sd=sd, t=t, p_t=p_t, p_w=p_w, d=mean / sd, lo=lo, hi=hi,
                loses=sum(1 for x in differences if x < 0),
                causal_dir=causal_dir, noncausal_dir=noncausal_dir)


def render(rows):
    lines = []
    w = lines.append
    w("# Causality cost, paired by subject\n")
    w("Each held-out subject gives one paired difference between the causal model and the same")
    w("architecture with the causal constraint removed, trained on the same partition. A subject")
    w("is in the test set of exactly one fold, so these differences are independent of one")
    w("another and need no correction for reused data - which is the reason to prefer this test")
    w("to a pooled fold-level one.\n")
    w("## Summary\n")
    w("| Comparison | Metric | Subjects | Causal | Non-causal | Difference | 95% CI | t | p | "
      "Cohen's d | Subjects causal loses |")
    w("|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|")
    for r in rows:
        p_t = f"{r['p_t']:.2e}" if r["p_t"] is not None and r["p_t"] < 1e-4 else (
            f"{r['p_t']:.4f}" if r["p_t"] is not None else "n/a")
        w(f"| {r['name']} | {r['metric']} | {r['n']} | {r['mean_c']:.4f} | {r['mean_n']:.4f} | "
          f"{r['mean']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['t']:.2f} | {p_t} | "
          f"{r['d']:.2f} | {r['loses']}/{r['n']} |")
    w("")

    for r in rows:
        w(f"## {r['name']}\n")
        w(f"- Causal arm: `{r['causal_dir']}`")
        w(f"- Non-causal arm: `{r['noncausal_dir']}`")
        p_t = f"{r['p_t']:.2e}" if r["p_t"] is not None and r["p_t"] < 1e-4 else (
            f"{r['p_t']:.4f}" if r["p_t"] is not None else "not computed")
        p_w = f"{r['p_w']:.2e}" if r["p_w"] is not None and r["p_w"] < 1e-4 else (
            f"{r['p_w']:.4f}" if r["p_w"] is not None else "not computed")
        w(f"- Paired t-test over subjects: t({r['n'] - 1}) = {r['t']:.2f}, p = {p_t}")
        w(f"- Wilcoxon signed-rank: p = {p_w}")
        w(f"- Bootstrap 95% CI on the mean difference: [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        w(f"- Cohen's d = {r['d']:.2f}")
        w(f"- Between-subject sd of the difference: {r['sd']:.4f}")
        w(f"- The causal model is worse on {r['loses']} of {r['n']} subjects\n")
        w(f"| Subject | Fold | Causal {r['metric']} | Non-causal {r['metric']} | Difference |")
        w("|---|---:|---:|---:|---:|")
        for s in r["subjects"]:
            w(f"| {s} | {r['folds'][s]} | {r['causal'][s]:.4f} | {r['noncausal'][s]:.4f} | "
              f"{r['causal'][s] - r['noncausal'][s]:+.4f} |")
        w(f"| **Mean** | | **{r['mean_c']:.4f}** | **{r['mean_n']:.4f}** | "
          f"**{r['mean']:+.4f}** |\n")
    return "\n".join(lines) + "\n"


def parse_pair(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--pair must look like NAME=causal_dir,noncausal_dir (got {spec!r})")
    name, _, dirs = spec.partition("=")
    parts = [p.strip() for p in dirs.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--pair needs exactly two comma-separated directories (got {spec!r})")
    return (name.strip(), parts[0], parts[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", type=parse_pair, action="append", required=True,
                    metavar="NAME=CAUSAL_DIR,NONCAUSAL_DIR",
                    help="a run pair to compare; repeatable")
    ap.add_argument("--metric", choices=METRIC_CHOICES, default="kappa")
    ap.add_argument("--out", default="results/statistics_by_subject.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows = [analyse(name, causal, noncausal, args.metric)
            for name, causal, noncausal in args.pair]

    text = render(rows)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open(args.out, "w").write(text)
    if not args.quiet:
        print(text)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
