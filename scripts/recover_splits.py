"""Rebuild the `split_fold_<N>.yaml` files for the archived streaming runs.

Training writes the subject split to `checkpoint_dir`, not `log_dir`, so the curated
directories under `results/` never received them and the checkpoints they sit beside are
gitignored. The splits are not lost, though: every `fold_<N>_test_report.txt` records its
held-out subjects, and `get_cv_splits` is deterministic, so the full train/val/test partition
can be regenerated from the run's seed and checked against what the report says.

    python scripts/recover_splits.py --check          # verify only, write nothing
    python scripts/recover_splits.py                  # verify, then write the yaml files

Nothing is written unless the reconstruction reproduces every archived test set exactly, and
an existing file is never overwritten without `--force`. A run whose reports disagree with the
reconstruction is reported and skipped, because the alternative - writing a plausible-looking
split that is not the one the model was trained on - is worse than having no file.

The `split_seed` recorded in each file is the seed that produced the partition, which for these
archived runs is the training seed: they predate `train.split_seed`. See REPRODUCIBILITY.md
section 5.
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_cv_splits  # noqa: E402

# The archived streaming sweep. Each of these runs took its partition from its training seed.
DEFAULT_RUNS = [
    ("results/sleep78_streaming_causal", 42),
    ("results/sleep78_streaming_noncausal", 42),
    ("results/sleep78_streaming_causal_s43", 43),
    ("results/sleep78_streaming_noncausal_s43", 43),
    ("results/sleep78_streaming_causal_s44", 44),
    ("results/sleep78_streaming_noncausal_s44", 44),
]


def archived_test_subjects(run_dir):
    """Read `{fold: [subject, ...]}` out of a run's fold reports."""
    found = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "fold_*_test_report.txt"))):
        fold = int(re.search(r"fold_(\d+)_", os.path.basename(path)).group(1))
        with open(path) as handle:
            line = next((l for l in handle if l.startswith("Test subjects:")), None)
        if line is None:
            raise ValueError(f"{path} has no 'Test subjects:' line")
        found[fold] = ast.literal_eval(line.split(":", 1)[1].strip())
    if not found:
        raise ValueError(f"no fold_*_test_report.txt in {run_dir}")
    return found


def reconstruct(run_dir, seed):
    """Regenerate the partition and check it against the reports.

    Returns ``(splits, subjects, mismatches)``. ``splits`` maps fold to the
    ``(train, val, test)`` triple; ``mismatches`` lists folds where the regenerated test set
    differs from the archived one, and must be empty before anything is written.
    """
    archived = archived_test_subjects(run_dir)
    # Every subject is held out in exactly one fold, so the union is the full cohort - which
    # is what get_cv_splits shuffles. Sorting matches what get_all_subject_ids returns.
    subjects = sorted({s for group in archived.values() for s in group})

    splits, mismatches = {}, []
    for fold in sorted(archived):
        train, val, test = get_cv_splits(subjects, num_folds=len(archived),
                                         fold_idx=fold, seed=seed)
        splits[fold] = (train, val, test)
        if test != archived[fold]:
            mismatches.append(fold)
    return splits, subjects, mismatches


class QuotedDumper(yaml.SafeDumper):
    """Emit every string quoted.

    Subject IDs carry a leading zero (`08`, `09`). PyYAML happens to read those back as
    strings, but an unquoted `08` is one YAML 1.2 loader away from becoming the integer 8,
    and `subject_8.npz` does not exist. Quoting removes the question.
    """


QuotedDumper.add_representer(
    str, lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'"))


def write_split(run_dir, fold, train, val, test, seed, force):
    path = os.path.join(run_dir, f"split_fold_{fold}.yaml")
    if os.path.exists(path) and not force:
        return path, False
    with open(path, "w") as handle:
        yaml.dump({
            "train_subjects": train,
            "val_subjects": val,
            "test_subjects": test,
            "split_seed": seed,
            "recovered_by": "scripts/recover_splits.py",
            "recovered_note": (
                "Regenerated from the archived fold report's test-subject list, which this "
                "partition reproduces exactly. This run predates train.split_seed, so the "
                "partition seed is the training seed."),
        }, handle, Dumper=QuotedDumper, sort_keys=False)
    return path, True


def parse_run(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--run must look like DIR=SEED (got {spec!r})")
    directory, _, seed = spec.partition("=")
    if not seed.strip().lstrip("-").isdigit():
        raise argparse.ArgumentTypeError(f"seed must be an integer (got {spec!r})")
    return (directory.strip(), int(seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=parse_run, action="append", default=None,
                    metavar="DIR=SEED", help="a run directory and the seed it used; repeatable")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    runs = args.run if args.run else DEFAULT_RUNS
    failures = 0
    written = 0
    skipped = 0

    for run_dir, seed in runs:
        if not os.path.isdir(run_dir):
            print(f"[skip] {run_dir}: not a directory")
            continue
        try:
            splits, subjects, mismatches = reconstruct(run_dir, seed)
        except ValueError as exc:
            print(f"[FAIL] {run_dir}: {exc}")
            failures += 1
            continue

        if mismatches:
            print(f"[FAIL] {run_dir}: seed {seed} does not reproduce folds {mismatches}. "
                  f"Not writing anything for this run.")
            failures += 1
            continue

        print(f"[ok]   {os.path.basename(run_dir):40s} seed {seed}, "
              f"{len(splits)} folds, {len(subjects)} subjects, every test set reproduced")
        if args.check:
            continue

        for fold, (train, val, test) in sorted(splits.items()):
            path, did_write = write_split(run_dir, fold, train, val, test, seed, args.force)
            if did_write:
                written += 1
            else:
                skipped += 1
                print(f"       exists, not overwritten: {path}  (use --force)")

    print()
    if args.check:
        print(f"verified {len(runs) - failures} of {len(runs)} runs, wrote nothing")
    else:
        print(f"wrote {written} split files, skipped {skipped} existing, {failures} runs failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
