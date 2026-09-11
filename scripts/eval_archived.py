"""Re-evaluate the archived streaming checkpoints to recover their per-second predictions.

The six streaming runs were trained in August and their weights still exist. Evaluation at
the time did not save predictions, which is why the smoothing sweep, subject-level intervals,
prior correction and boundary-latency analyses were all described as needing a retrain. They
do not. Running evaluation again against the same weights and the same processed data
regenerates the metrics and, with this branch's evaluate.py, writes fold_N_predictions.npz
alongside them.

This is 30 evaluation passes, no training. On a laptop GPU it is hours, not days.

What this does NOT give you: the archived runs let train.seed set the fold partition, so the
three seeds are fifteen distinct partitions rather than five folds measured three times. That
is what the matched sweep in run_sweep.py fixes. Every analysis that needs predictions but
not matched folds can run today from these weights; only the matched-design statistics need
the sweep.

Preconditions:
  data/processed78_streaming/       78 .npz files, copied from the archive
  checkpoints_78streaming_<arm>_s<seed>/   best_model_fold_*.pth, copied from the archive

Do NOT copy the archived logs_78streaming_* directories in. evaluate.py APPENDS a row to
test_metrics_summary.csv, so evaluating into a directory that already has rows leaves two
rows per fold and every mean computed from that file is silently wrong. This script refuses
to run when it finds that situation.

Usage:
    python scripts/eval_archived.py --check
    python scripts/eval_archived.py --dry-run
    python scripts/eval_archived.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sweep import (REPO, STREAM_STRIDE, load_config, run_stage,  # noqa: E402
                       stage_done, summary_has_fold)

# The archived configs, not the sweep ones: these checkpoints belong to these splits.
ARCHIVED = [
    ("causal",    42, "configs/sleep78_streaming_causal.yaml"),
    ("causal",    43, "configs/sleep78_streaming_causal_s43.yaml"),
    ("causal",    44, "configs/sleep78_streaming_causal_s44.yaml"),
    ("noncausal", 42, "configs/sleep78_streaming_noncausal.yaml"),
    ("noncausal", 43, "configs/sleep78_streaming_noncausal_s43.yaml"),
    ("noncausal", 44, "configs/sleep78_streaming_noncausal_s44.yaml"),
]
FOLDS = (0, 1, 2, 3, 4)


def check(entries):
    problems, notes = [], []
    try:
        import torch
        notes.append(f"torch {torch.__version__}"
                     + (", CUDA on" if torch.cuda.is_available()
                        else ", CPU only (evaluation is slow but workable)"))
    except ImportError:
        problems.append("torch is not installed in the active environment.")

    pdir = os.path.join(REPO, "data", "processed78_streaming")
    n = len([f for f in os.listdir(pdir) if f.endswith(".npz")]) if os.path.isdir(pdir) else 0
    if n < 78:
        problems.append(f"expected 78 .npz files in data/processed78_streaming, found {n}. "
                        f"Copy them from the archived checkout.")
    else:
        notes.append(f"{n} preprocessed subjects present")

    for arm, seed, cfg_rel in entries:
        cfg_path = os.path.join(REPO, cfg_rel)
        if not os.path.exists(cfg_path):
            problems.append(f"missing config {cfg_rel}")
            continue
        cfg = load_config(cfg_path)
        ckpt_dir = os.path.join(REPO, cfg["train"]["checkpoint_dir"])
        log_dir = os.path.join(REPO, cfg["train"]["log_dir"])

        found = sum(os.path.exists(os.path.join(ckpt_dir, f"best_model_fold_{f}.pth"))
                    for f in FOLDS)
        if found != 5:
            problems.append(f"{os.path.basename(ckpt_dir)}: {found}/5 checkpoints. "
                            f"Copy the folder from the archived checkout.")
        else:
            notes.append(f"{os.path.basename(ckpt_dir)}: 5/5 checkpoints")

        # The duplicate-row trap.
        pre_existing = [f for f in FOLDS if summary_has_fold(log_dir, f)]
        if pre_existing:
            problems.append(
                f"{cfg['train']['log_dir']}/test_metrics_summary.csv already has rows for "
                f"folds {pre_existing}. evaluate.py appends, so re-running would duplicate "
                f"them. Move that log directory aside (rename it *_archived) and let this "
                f"run write a fresh one.")

    print("Check")
    print("-" * 72)
    for x in notes:
        print(f"  ok    {x}")
    for x in problems:
        print(f"  FAIL  {x}")
    print("-" * 72)
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-streaming", action="store_true",
                    help="Skip the --stream_stride pass (it costs 4x a windowed evaluation).")
    args = ap.parse_args()

    if not check(ARCHIVED):
        sys.exit("\nCheck failed. Nothing was run.")
    if args.check:
        return

    stages = ["eval"] + ([] if args.skip_streaming else ["eval_stream"])
    todo = []
    for arm, seed, cfg_rel in ARCHIVED:
        cfg = load_config(os.path.join(REPO, cfg_rel))
        log_dir = os.path.join(REPO, cfg["train"]["log_dir"])
        ckpt_dir = os.path.join(REPO, cfg["train"]["checkpoint_dir"])
        tag = f"{arm}_s{seed}"
        for fold in FOLDS:
            for stage in stages:
                if stage_done(stage, log_dir, ckpt_dir, fold):
                    continue
                todo.append((os.path.join(REPO, cfg_rel), tag, fold, stage))

    print(f"\n{len(todo)} evaluation passes to run.")
    if args.dry_run or not todo:
        for _, tag, fold, stage in todo:
            print(f"  {tag:16s} fold {fold}  {stage}")
        return

    failures = []
    for i, (cfg_path, tag, fold, stage) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {tag}  fold {fold}", flush=True)
        if not run_stage(stage, cfg_path, fold, f"archived_{tag}"):
            failures.append((tag, fold, stage))

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} passes failed:")
        for tag, fold, stage in failures:
            print(f"  {tag:16s} fold {fold}  {stage}")
        sys.exit(1)
    print("All predictions recovered. These now run in minutes and need no GPU:\n")
    print("  python scripts/boundary_latency.py logs_78streaming_causal_s42 --hold 10 \\")
    print("      --out results/boundary_latency.md")
    print("  python scripts/analyze_predictions.py logs_78streaming_causal_s42 \\")
    print("      --bootstrap --per-subject --prior-correction --out results/predictions.md")
    print("  python scripts/sweep_smoothing.py --predictions logs_78streaming_causal_s42")
    print("  python scripts/analysis_stats.py --pair")


if __name__ == "__main__":
    main()
