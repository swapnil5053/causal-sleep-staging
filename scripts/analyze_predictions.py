"""Analyse archived per-second predictions without touching a checkpoint.

`src/eval/evaluate.py` writes `fold_N_predictions.npz` holding the subject, expert label,
prediction and logits for every second of a held-out fold. Everything here reads those
files, so a question asked after the fact costs a few seconds instead of a training run.

Three analyses, each optional:

`--bootstrap`
    Resamples *subjects*, not seconds. Fold-level intervals at n = 5 are wide and assume
    folds are the unit of variation; the subject is the unit a reviewer cares about, and
    a subject-level interval is both narrower and easier to defend.

`--prior-correction`
    Shifts the logits by the log ratio of the target and training class priors. The
    class-weighted focal loss leaves N3 over-predicted (high recall, low precision); this
    undoes part of that at inference, with no retraining and no added parameters.

`--per-subject`
    One row per subject, so an outlier recording can be found rather than averaged away.

    python scripts/analyze_predictions.py logs_78streaming_causal_s42 --bootstrap
    python scripts/analyze_predictions.py logs_*/ --prior-correction --out results/prior.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

STAGES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = 5


def load_folds(paths, pattern="fold_*_predictions.npz"):
    """Collect prediction files from files, globs or directories.

    Subjects are identified by an integer code that is unique across every loaded file,
    not by their two-character id. Two runs both containing subject "00" are two distinct
    held-out records, and merging them would silently halve the number of independent
    units a bootstrap resamples. Codes are also what keeps a 15-fold pool inside a few
    hundred megabytes: a per-second array of namespaced strings costs 100 bytes a second.
    """
    files, seen = [], set()
    for path in paths:
        if os.path.isdir(path):
            matched = sorted(glob.glob(os.path.join(path, pattern)))
        elif os.path.isfile(path):
            matched = [path]
        else:
            matched = sorted(glob.glob(path))
            if not matched:
                raise FileNotFoundError(f"No prediction files matched {path!r}")
        for item in matched:
            # The same file can arrive twice via a directory and an overlapping glob;
            # loading it twice would double-count every second in it.
            real = os.path.realpath(item)
            if real not in seen:
                seen.add(real)
                files.append(item)
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r}. Run evaluation first; it writes one per fold.")

    folds, labels, offset = [], [], 0
    for path in files:
        data = np.load(path, allow_pickle=False)
        missing = [k for k in ("subject", "y_true", "y_pred") if k not in data.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}. Re-run evaluation to rewrite it.")

        local = np.asarray(data["subject"])
        unique, inverse = np.unique(local, return_inverse=True)
        source = os.path.basename(os.path.dirname(os.path.realpath(path))) or "."
        stem = os.path.basename(path).replace("_predictions", "").replace(".npz", "")
        labels.extend(f"{name} ({source}/{stem})" for name in unique)

        folds.append({
            "path": path,
            "fold": int(data["fold"]) if "fold" in data.files else -1,
            "stream_stride": int(data["stream_stride"]) if "stream_stride" in data.files else -1,
            "subject_code": (inverse + offset).astype(np.int32),
            "subject_local": local,
            "y_true": np.asarray(data["y_true"], dtype=np.int64),
            "y_pred": np.asarray(data["y_pred"], dtype=np.int64),
            "logits": np.asarray(data["logits"]) if "logits" in data.files else None,
            "segment_start": (np.asarray(data["segment_start"], dtype=bool)
                              if "segment_start" in data.files else None),
            "second_index": (np.asarray(data["second_index"], dtype=np.int64)
                             if "second_index" in data.files else None),
            "train_class_counts": (np.asarray(data["train_class_counts"], dtype=np.float64)
                                   if "train_class_counts" in data.files else None),
        })
        offset += len(unique)

    strides = {f["stream_stride"] for f in folds}
    if len(strides) > 1:
        print(f"warning: pooling files evaluated in different modes (stream_stride "
              f"{sorted(strides)}). Their numbers are not comparable.", file=sys.stderr)

    for fold in folds:
        fold["subject_labels"] = labels
    return folds


def metrics(y_true, y_pred):
    """Accuracy, kappa and macro-F1. Kappa is undefined for a single observed class.

    Macro-F1 averages over all five stages, matching metrics_from_confusion, so the point
    estimate and the bootstrap interval beside it are always the same quantity.
    """
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "kappa": float("nan"), "macro_f1": float("nan"), "n": 0}
    distinct = len(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if distinct > 1 else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=range(N_CLASSES), zero_division=0)),
        "n": int(len(y_true)),
    }


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)


def apply_prior_correction(logits, y_train_prior, y_target_prior, eps=1e-9):
    """Re-weight logits from the prior the loss imposed toward the prior of the data.

    Adding log(target) - log(source) to the logits is the exact Bayes correction for a
    change of class prior when the likelihoods are unchanged, which is what a reweighted
    loss distorts. It needs no retraining and adds no parameters.
    """
    source = np.asarray(y_train_prior, dtype=np.float64) + eps
    target = np.asarray(y_target_prior, dtype=np.float64) + eps
    source = source / source.sum()
    target = target / target.sum()
    return logits + (np.log(target) - np.log(source))[None, :]


def confusion(y_true, y_pred):
    """5x5 counts, rows true and columns predicted."""
    flat = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES * N_CLASSES)
    return flat.reshape(N_CLASSES, N_CLASSES).astype(np.float64)


def metrics_from_confusion(matrix):
    """Accuracy, kappa and macro-F1 straight from a confusion matrix.

    A bootstrap draw is then the sum of the drawn subjects' 5x5 matrices rather than a
    re-scoring of a million concatenated seconds, which is what makes the resampling
    finish in seconds on a real fold instead of running for an hour. The values are exact,
    not approximate: every one of these three metrics is a function of the matrix alone.
    """
    total = matrix.sum()
    if total == 0:
        return {"accuracy": float("nan"), "kappa": float("nan"), "macro_f1": float("nan")}

    observed_agreement = np.trace(matrix) / total
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    chance_agreement = float((row_totals * col_totals).sum()) / (total * total)
    kappa = (float("nan") if np.isclose(chance_agreement, 1.0)
             else (observed_agreement - chance_agreement) / (1.0 - chance_agreement))

    f1s = []
    for index in range(N_CLASSES):
        true_positive = matrix[index, index]
        denominator = 2 * true_positive + (col_totals[index] - true_positive) \
            + (row_totals[index] - true_positive)
        f1s.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    return {"accuracy": float(observed_agreement), "kappa": float(kappa),
            "macro_f1": float(np.mean(f1s))}


def per_subject_confusions(codes, y_true, y_pred, n_subjects):
    """One 5x5 matrix per subject, in a single pass over the data.

    Masking per subject would be O(n_subjects x n_seconds); at 240 subjects and 18 million
    seconds that is minutes of pure indexing. One bincount over a combined key is linear.
    """
    key = (codes.astype(np.int64) * (N_CLASSES * N_CLASSES)
           + y_true * N_CLASSES + y_pred)
    counts = np.bincount(key, minlength=n_subjects * N_CLASSES * N_CLASSES)
    return counts.reshape(n_subjects, N_CLASSES, N_CLASSES).astype(np.float64)


def bootstrap_over_subjects(codes, y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    """Percentile interval from resampling whole subjects with replacement.

    Seconds within a subject are heavily autocorrelated, so resampling seconds would give
    an interval far too narrow to mean anything. The subject is the independent unit.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(codes)
    if len(unique) < 2:
        return None

    # One matrix per subject, summed per draw. See metrics_from_confusion.
    matrices = per_subject_confusions(codes, y_true, y_pred, int(unique.max()) + 1)[unique]

    draws = {"accuracy": [], "kappa": [], "macro_f1": []}
    for _ in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        sample = metrics_from_confusion(matrices[picked].sum(axis=0))
        for key in draws:
            draws[key].append(sample[key])

    out = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            out[key] = (float("nan"), float("nan"))
        else:
            out[key] = (float(np.percentile(values, 100 * alpha / 2)),
                        float(np.percentile(values, 100 * (1 - alpha / 2))))
    return out


def render(folds, args):
    lines = []
    write = lines.append

    y_true = np.concatenate([f["y_true"] for f in folds])
    y_pred = np.concatenate([f["y_pred"] for f in folds])
    codes = np.concatenate([f["subject_code"] for f in folds])
    labels = folds[0]["subject_labels"]

    write("# Prediction analysis\n")
    write(f"{len(folds)} prediction file(s), {len(np.unique(codes))} held-out subject "
          f"records, {len(y_true):,} scored seconds.\n")
    write("Read from archived predictions; no checkpoint was loaded and nothing was retrained.\n")

    pooled = metrics(y_true, y_pred)
    write("## Pooled\n")
    write("| Metric | Value |")
    write("|---|---|")
    write(f"| Accuracy | {pooled['accuracy']:.4f} |")
    write(f"| Cohen's kappa | {pooled['kappa']:.4f} |")
    write(f"| Macro-F1 | {pooled['macro_f1']:.4f} |")
    f1s = per_class_f1(y_true, y_pred)
    write("")
    write("| " + " | ".join(STAGES) + " |")
    write("|" + "---|" * N_CLASSES)
    write("| " + " | ".join(f"{v:.4f}" for v in f1s) + " |")
    write("")

    if args.bootstrap:
        write("## Subject-level bootstrap\n")
        interval = bootstrap_over_subjects(codes, y_true, y_pred,
                                           n_boot=args.n_boot, seed=args.seed)
        if interval is None:
            write("Fewer than two subjects; an interval would be meaningless.\n")
        else:
            write(f"Whole subjects resampled with replacement, {args.n_boot:,} draws. "
                  f"Seconds inside a subject are autocorrelated, so the subject is the "
                  f"independent unit rather than the second.\n")
            write("| Metric | Estimate | 95% CI |")
            write("|---|---|---|")
            for key, label in (("accuracy", "Accuracy"), ("kappa", "Cohen's kappa"),
                               ("macro_f1", "Macro-F1")):
                low, high = interval[key]
                write(f"| {label} | {pooled[key]:.4f} | [{low:.4f}, {high:.4f}] |")
            write("")

    if args.per_subject:
        write("## Per subject\n")
        write("| Subject | Seconds | Accuracy | Kappa | Macro-F1 |")
        write("|---|---:|---|---|---|")
        present = np.unique(codes)
        matrices = per_subject_confusions(codes, y_true, y_pred, int(present.max()) + 1)
        for code in present:
            row = metrics_from_confusion(matrices[code])
            total = int(matrices[code].sum())
            write(f"| {labels[code]} | {total:,} | {row['accuracy']:.4f} | "
                  f"{row['kappa']:.4f} | {row['macro_f1']:.4f} |")
        write("")

    if args.prior_correction:
        write("## Class-prior correction\n")
        if any(f["logits"] is None for f in folds):
            write("At least one prediction file carries no logits, so the correction cannot "
                  "be applied. Re-run evaluation to rewrite them.\n")
        else:
            logits = np.concatenate([f["logits"] for f in folds]).astype(np.float32)
            observed = np.bincount(y_true, minlength=N_CLASSES).astype(np.float64)

            recorded = [f["train_class_counts"] for f in folds
                        if f["train_class_counts"] is not None]
            if args.source_prior is not None:
                assumed = np.array([float(v) for v in args.source_prior.split(",")])
                if len(assumed) != N_CLASSES:
                    raise SystemExit(f"--source-prior needs {N_CLASSES} comma-separated values")
                source_note = "the prior given on the command line"
            elif len(recorded) == len(folds):
                # The class-weighted loss pushes the effective prior toward uniform, so
                # the balance the model was trained against is the reweighted one.
                counts = np.sum(recorded, axis=0)
                inverse = counts.sum() / np.maximum(counts, 1)
                assumed = inverse / inverse.sum() * counts.sum()
                source_note = ("the recorded training class balance, reweighted the way the "
                               "loss reweights it")
            else:
                assumed = np.full(N_CLASSES, observed.sum() / N_CLASSES)
                source_note = ("a uniform prior, assumed because the prediction files carry "
                               "no training class counts. That assumption holds for the "
                               "class-weighted focal loss this repository trains with and "
                               "not for an unweighted one; re-run evaluation against a "
                               "checkpoint written by the current trainer, or pass "
                               "--source-prior, to remove the guess")
            corrected = apply_prior_correction(logits, assumed, observed).argmax(axis=1)
            after = metrics(y_true, corrected)
            write(f"Logits shifted by `log(target prior) - log(source prior)`, applied at "
                  f"inference only. No retraining, no added parameters. The source prior is "
                  f"{source_note}.\n")
            write("| Metric | Before | After | Change |")
            write("|---|---|---|---|")
            for key, label in (("accuracy", "Accuracy"), ("kappa", "Cohen's kappa"),
                               ("macro_f1", "Macro-F1")):
                write(f"| {label} | {pooled[key]:.4f} | {after[key]:.4f} | "
                      f"{after[key] - pooled[key]:+.4f} |")
            write("")
            before_f1 = per_class_f1(y_true, y_pred)
            after_f1 = per_class_f1(y_true, corrected)
            write("| Stage | F1 before | F1 after | Change |")
            write("|---|---|---|---|")
            for i, stage in enumerate(STAGES):
                write(f"| {stage} | {before_f1[i]:.4f} | {after_f1[i]:.4f} | "
                      f"{after_f1[i] - before_f1[i]:+.4f} |")
            write("")
            write("A gain here is a free post-processing improvement of the same kind as the "
                  "causal mode filter. A loss is also informative: it says the reweighted "
                  "loss was not the source of the imbalance.\n")
            write("One caveat before quoting this. The target prior used here is the class "
                  "distribution of the held-out data itself, so the figure is a best case: "
                  "it assumes the deployment prior is known. The honest protocol estimates "
                  "the prior from the training folds, or from population statistics, and "
                  "applies that instead.\n")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="Prediction .npz files, or log directories containing them.")
    parser.add_argument("--pattern", default="fold_*_predictions.npz",
                        help="Glob used inside a directory. Use "
                             "'fold_*_predictions_streaming*.npz' for streaming runs.")
    parser.add_argument("--bootstrap", action="store_true",
                        help="95%% confidence interval from resampling subjects.")
    parser.add_argument("--per-subject", action="store_true", help="One row per subject.")
    parser.add_argument("--prior-correction", action="store_true",
                        help="Re-weight logits toward the observed class prior.")
    parser.add_argument("--source-prior", default=None,
                        help="Five comma-separated class weights describing the prior the "
                             "model was trained against. Read from the prediction files "
                             "when they record it.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap draws.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument("--out", default=None, help="Write the report here.")
    parser.add_argument("--quiet", action="store_true", help="File only, no console output.")
    args = parser.parse_args()

    if not (args.bootstrap or args.per_subject or args.prior_correction):
        args.bootstrap = True  # the most useful default

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
