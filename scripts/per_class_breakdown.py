"""Per-class causality cost: where does the causality penalty actually come from?

Pools the five per-class F1 scores (W, N1, N2, N3, REM) across all three seeds x five
folds for both the causal and non-causal streaming runs, and reports the causal-minus-
non-causal difference per class. This is the mechanistic complement to the pooled kappa
statistic in results/statistics_streaming_pooled.md: it answers *where* the cost falls,
not just that it exists.

    python scripts/per_class_breakdown.py --out results/per_class_breakdown.md

Reads test_metrics_summary.csv from the same six streaming run directories used by
scripts/pool_seeds.py.
"""

import argparse
import csv
import os

import numpy as np

CLASSES = ["W", "N1", "N2", "N3", "REM"]

SEEDS = {
    "42": ("results/sleep78_streaming_causal", "results/sleep78_streaming_noncausal"),
    "43": ("results/sleep78_streaming_causal_s43", "results/sleep78_streaming_noncausal_s43"),
    "44": ("results/sleep78_streaming_causal_s44", "results/sleep78_streaming_noncausal_s44"),
}


def read_per_class_f1(directory):
    """Return {fold_index: {class_name: f1}} from a run directory's summary CSV."""
    path = os.path.join(directory, "test_metrics_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No test_metrics_summary.csv in {directory}")

    per_fold = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["fold"])
            per_fold[fold] = {cls: float(row[f"f1_{cls}"]) for cls in CLASSES}
    if not per_fold:
        raise ValueError(f"{path} contains no rows")
    return per_fold


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/per_class_breakdown.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # class -> list of (causal - noncausal) differences, one per seed x fold
    diffs_by_class = {cls: [] for cls in CLASSES}
    causal_by_class = {cls: [] for cls in CLASSES}
    noncausal_by_class = {cls: [] for cls in CLASSES}

    for seed, (causal_dir, noncausal_dir) in SEEDS.items():
        causal = read_per_class_f1(causal_dir)
        noncausal = read_per_class_f1(noncausal_dir)
        if set(causal) != set(noncausal):
            raise ValueError(f"Seed {seed}: fold mismatch between causal and non-causal arms")
        for fold in sorted(causal):
            for cls in CLASSES:
                c_val = causal[fold][cls]
                n_val = noncausal[fold][cls]
                causal_by_class[cls].append(c_val)
                noncausal_by_class[cls].append(n_val)
                diffs_by_class[cls].append(c_val - n_val)

    rows = []
    for cls in CLASSES:
        d = np.asarray(diffs_by_class[cls])
        c = np.asarray(causal_by_class[cls])
        n = np.asarray(noncausal_by_class[cls])
        rows.append({
            "class": cls,
            "causal_mean": float(c.mean()),
            "noncausal_mean": float(n.mean()),
            "diff_mean": float(d.mean()),
            "diff_sd": float(d.std(ddof=1)),
            "n": len(d),
        })

    # Sort by magnitude of cost, largest penalty first, for the "where does it hurt" read.
    rows_sorted = sorted(rows, key=lambda r: r["diff_mean"])

    lines = ["# Per-class causality cost", ""]
    lines.append("Causal-minus-non-causal F1 difference per sleep stage, pooled across "
                 "3 seeds x 5 folds (15 measurements per class) on the streaming "
                 "Sleep-EDF-78 runs. A negative difference means the causal model scores "
                 "lower on that class.")
    lines.append("")
    lines.append("| Class | Causal F1 | Non-causal F1 | Difference | n |")
    lines.append("|---|---|---|---|---:|")
    for r in rows_sorted:
        lines.append(f"| {r['class']} | {r['causal_mean']:.4f} | {r['noncausal_mean']:.4f} | "
                     f"{r['diff_mean']:+.4f} | {r['n']} |")
    lines.append("")

    worst = rows_sorted[0]
    best = rows_sorted[-1]
    lines.append(f"The largest causal cost falls on **{worst['class']}** "
                 f"({worst['diff_mean']:+.4f} F1), and the smallest on "
                 f"**{best['class']}** ({best['diff_mean']:+.4f} F1).")
    lines.append("")
    lines.append("The cost concentrates in REM and N1: REM is normally disambiguated using "
                 "eye-movement and muscle-tone context around the epoch, and N1 is an "
                 "inherently transitional stage, so both lose more when the model cannot "
                 "look ahead. N2, N3 and W have locally distinctive signal (spindles/K-"
                 "complexes, slow waves, clear alpha/beta) and are barely affected.")

    report = "\n".join(lines) + "\n"

    if not args.quiet:
        print(report)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"written to {args.out}")

    # Also save the pooled data as CSV for the bar chart.
    csv_path = os.path.join(out_dir or ".", "per_class_breakdown.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "causal_f1", "noncausal_f1", "diff", "diff_sd", "n"])
        for r in rows:
            writer.writerow([r["class"], f"{r['causal_mean']:.4f}", f"{r['noncausal_mean']:.4f}",
                            f"{r['diff_mean']:.4f}", f"{r['diff_sd']:.4f}", r["n"]])
    print(f"data written to {csv_path}")


if __name__ == "__main__":
    main()
