"""Seed/split overlap simulation: does changing the seed also change the CV folds?

get_cv_splits() takes a single `seed` argument that controls both weight initialisation
(when passed as init_seed elsewhere) and the subject shuffle used to build folds. Running
the "same" fold index under seeds 42, 43, 44 therefore does not give the same subjects in
that fold each time -- it gives a different partition. This script quantifies that overlap
directly, using the real get_cv_splits() function from src/data/dataset.py.

Uses a synthetic 78-subject ID list ('00' .. '77', zero-padded, matching the convention
seen in results/trimmed/checkpoints/split_fold_0.yaml) since the full processed dataset's
.npz files live only on the GPU laptop. get_cv_splits() only shuffles and partitions IDs --
it never reads subject data -- so the fold membership and overlap counts are identical to
what the real 78-subject run would produce.

    python scripts/seed_overlap_simulation.py --out results/generated/seed_overlap.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_cv_splits

NUM_SUBJECTS = 78
NUM_FOLDS = 5
SEEDS = [42, 43, 44]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/generated/seed_overlap.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    subject_ids = [f"{i:02d}" for i in range(NUM_SUBJECTS)]

    # test_subjects per (seed, fold_idx)
    test_subs_by_seed = {}
    for seed in SEEDS:
        test_subs_by_seed[seed] = {}
        for fold_idx in range(NUM_FOLDS):
            _, _, test_subs = get_cv_splits(
                subject_ids, num_folds=NUM_FOLDS, fold_idx=fold_idx, seed=seed)
            test_subs_by_seed[seed][fold_idx] = set(test_subs)

    seed_pairs = [(SEEDS[i], SEEDS[j]) for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]

    lines = ["# Seed / split overlap simulation", ""]
    lines.append(f"`get_cv_splits()` takes a single `seed` argument that controls both the "
                 f"subject shuffle and (elsewhere) weight initialisation. This simulates "
                 f"{NUM_FOLDS}-fold splits over {NUM_SUBJECTS} synthetic subjects at seeds "
                 f"{', '.join(str(s) for s in SEEDS)} and measures how much the *test set* "
                 f"of each fold index changes across seeds. If seeds only changed "
                 f"initialisation, overlap would be 100%. It is not.")
    lines.append("")
    lines.append("## Test-subject overlap per fold, across seed pairs\n")
    header = "| Fold | " + " | ".join(f"{a}-{b}" for a, b in seed_pairs) + " | Fold size |"
    sep = "|---|" + "|".join(["---"] * len(seed_pairs)) + "|---:|"
    lines.append(header)
    lines.append(sep)

    all_overlaps = []
    for fold_idx in range(NUM_FOLDS):
        row = [f"| {fold_idx} "]
        fold_size = None
        for a, b in seed_pairs:
            set_a = test_subs_by_seed[a][fold_idx]
            set_b = test_subs_by_seed[b][fold_idx]
            overlap = len(set_a & set_b)
            fold_size = len(set_a)
            all_overlaps.append(overlap)
            row.append(f"| {overlap}/{fold_size} ")
        row.append(f"| {fold_size} |")
        lines.append("".join(row))

    lines.append("")
    avg_overlap = sum(all_overlaps) / len(all_overlaps)
    fold_sizes = [len(test_subs_by_seed[SEEDS[0]][f]) for f in range(NUM_FOLDS)]
    avg_fold_size = sum(fold_sizes) / len(fold_sizes)
    lines.append(f"Average test-subject overlap across all folds and seed pairs: "
                 f"**{avg_overlap:.1f} subjects** out of an average fold size of "
                 f"{avg_fold_size:.1f} (fold sizes: {fold_sizes}) "
                 f"-- about {100 * avg_overlap / avg_fold_size:.0f}% overlap.")
    lines.append("")
    lines.append("This confirms that `seed` conflates weight initialisation with the subject "
                 "split: results reported for \"fold 0\" under different seeds are not the "
                 "same held-out subjects, so per-seed \"fold 0\" numbers are not directly "
                 "comparable across seeds. The fix (already applied going forward: see the "
                 "separated `split_seed` / `init_seed` config change) is to fix the split "
                 "seed independently of the initialisation seed, so repeated-seed runs share "
                 "folds and only initialisation varies.")

    report = "\n".join(lines) + "\n"

    if not args.quiet:
        print(report)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
