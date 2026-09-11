"""Drive the matched causal/non-causal sweep and its two evaluation passes.

One sweep is 6 configurations (2 arms x 3 initialisation seeds) x 5 folds = 30 training
runs, each followed by a windowed evaluation and a streaming evaluation. The point of
running it through here rather than by hand is that the work is resumable and that no
result file is ever written twice.

Resumability matters because the whole sweep is days of wall clock on a laptop GPU. Every
stage records a file when it finishes; a stage whose file already exists is skipped, so a
machine that sleeps, crashes or is interrupted resumes where it stopped instead of starting
over or, worse, silently appending a second copy of a fold's metrics.

That second failure is the one worth naming. ``evaluate.py`` *appends* a row to
``test_metrics_summary.csv``; re-evaluating a fold that was already scored leaves two rows
for the same fold and every mean computed from that file is then wrong in a way nothing
downstream can detect. The skip logic below is what prevents it.

Usage:
    python scripts/run_sweep.py --preflight        # check the environment, run nothing
    python scripts/run_sweep.py --dry-run          # print the plan, run nothing
    python scripts/run_sweep.py                    # run it
    python scripts/run_sweep.py --arm causal       # one arm only
    python scripts/run_sweep.py --seeds 42         # one seed only
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "configs", "sweep")   # overridable with --config_dir
TAG = "sweep"                                        # marker required in output dir names
RUN_LOG_DIR = os.path.join(REPO, "logs_sweep_runs")
STREAM_STRIDE = 30

ARMS = ("causal", "noncausal")
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- config

def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def config_path(arm, seed):
    return os.path.join(CONFIG_DIR, f"{arm}_s{seed}.yaml")


def dirs_for(cfg):
    return (os.path.join(REPO, cfg["train"]["log_dir"]),
            os.path.join(REPO, cfg["train"]["checkpoint_dir"]))


# --------------------------------------------------------------------------- preflight

def preflight(configs):
    """Fail loudly now rather than three hours into fold 2."""
    problems, notes = [], []

    try:
        import torch
        notes.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            notes.append(f"CUDA on: {torch.cuda.get_device_name(0)}")
        else:
            problems.append("CUDA is not available. On CPU this sweep is weeks, not days. "
                            "Install the CUDA build of torch before starting.")
    except ImportError:
        problems.append("torch is not installed in the active environment.")

    for mod in ("numpy", "pandas", "scipy", "sklearn", "yaml", "mne", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"missing dependency: {mod} (pip install -r requirements.txt)")

    # Every sweep config reads the same processed directory.
    processed = {c["data"]["processed_dir"] for _, c in configs}
    if len(processed) != 1:
        problems.append(f"configs disagree about processed_dir: {sorted(processed)}")
    pdir = os.path.join(REPO, sorted(processed)[0])

    npz = sorted(glob.glob(os.path.join(pdir, "*.npz")))
    if not npz:
        problems.append(
            f"no preprocessed data in {pdir}. Run preprocessing first:\n"
            + (f"      python scripts/dod_preprocessing.py --h5_dir data/dod/dodh "
               f"--processed_dir {os.path.relpath(pdir, REPO)} --all"
               if "dod" in TAG else
               f"      python -m src.data.preprocessing "
               f"--config {os.path.relpath(configs[0][0], REPO)} --all"))
    else:
        notes.append(f"{len(npz)} preprocessed subjects in {os.path.relpath(pdir, REPO)}")
        folds = {c["train"].get("num_folds", 5) for _, c in configs}
        if len(folds) != 1:
            problems.append(f"configs disagree about num_folds: {sorted(folds)}")
        elif len(npz) % sorted(folds)[0]:
            problems.append(f"{len(npz)} subjects does not divide evenly into "
                            f"{sorted(folds)[0]} folds; some folds will be uneven.")
        # respect_boundaries raises at dataset construction if this metadata is absent.
        # Better to find out here than after the first fold has trained.
        import numpy as np
        with np.load(npz[0]) as d:
            if "segment_starts" not in d.files:
                problems.append(
                    f"{os.path.basename(npz[0])} has no 'segment_starts' array, so it was "
                    f"written by an older preprocessing pass. Every sweep config sets "
                    f"respect_boundaries: true and will refuse to run. Delete "
                    f"{os.path.relpath(pdir, REPO)} and re-run preprocessing on this branch.")

    # Predictions dominate the footprint: ~20 MB per fold per evaluation mode,
    # 60 files over the sweep, plus checkpoints and logs.
    free_gb = shutil.disk_usage(REPO).free / 1e9
    notes.append(f"{free_gb:.0f} GB free on the repository volume")
    if free_gb < 10:
        problems.append(f"only {free_gb:.1f} GB free; the sweep writes roughly 3 GB "
                        f"and preprocessing needs more.")

    # Refuse to start on top of half-written archived output.
    for path, cfg in configs:
        log_dir, ckpt_dir = dirs_for(cfg)
        for d in (log_dir, ckpt_dir):
            if os.path.exists(d) and TAG not in os.path.basename(d):
                problems.append(f"{path} writes to {d}, whose name does not contain "
                                f"'{TAG}'. Refusing, in case that is an archived run.")

    print("Preflight")
    print("-" * 72)
    for n in notes:
        print(f"  ok    {n}")
    for p in problems:
        print(f"  FAIL  {p}")
    print("-" * 72)
    return not problems


# --------------------------------------------------------------------------- stages

def stage_done(stage, log_dir, ckpt_dir, fold):
    """Has this stage already produced its output file?"""
    if stage == "train":
        return os.path.exists(os.path.join(ckpt_dir, f"best_model_fold_{fold}.pth"))
    if stage == "eval":
        return os.path.exists(os.path.join(log_dir, f"fold_{fold}_test_report.txt"))
    if stage == "eval_stream":
        return os.path.exists(
            os.path.join(log_dir, f"fold_{fold}_test_report_streaming{STREAM_STRIDE}.txt"))
    raise ValueError(stage)


def summary_has_fold(log_dir, fold, suffix=""):
    """Guard against a duplicate row in the appended summary CSV."""
    path = os.path.join(log_dir, f"test_metrics_summary{suffix}.csv")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        next(f, None)
        return any(line.split(",")[0].strip() == str(fold) for line in f if line.strip())


def command_for(stage, cfg_path, fold):
    rel = os.path.relpath(cfg_path, REPO)
    if stage == "train":
        return [sys.executable, "-m", "src.train.train", "--config", rel, "--fold", str(fold)]
    cmd = [sys.executable, "-m", "src.eval.evaluate", "--config", rel, "--fold", str(fold)]
    if stage == "eval_stream":
        cmd += ["--stream_stride", str(STREAM_STRIDE)]
    return cmd


def run_stage(stage, cfg_path, fold, tag):
    """Run one stage, tee its output to a log file, return True on success."""
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    log_path = os.path.join(RUN_LOG_DIR, f"{tag}_fold{fold}_{stage}.log")
    cmd = command_for(stage, cfg_path, fold)
    started = time.time()
    print(f"    {stage:12s} -> {os.path.relpath(log_path, REPO)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
        code = proc.wait()
    mins = (time.time() - started) / 60
    if code != 0:
        print(f"    FAILED ({code}) after {mins:.1f} min. Last lines of {log_path}:")
        with open(log_path, encoding="utf-8") as lf:
            for line in lf.readlines()[-15:]:
                print("      " + line.rstrip())
        return False
    print(f"    done in {mins:.1f} min", flush=True)
    return True


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, action="append",
                    help="Restrict to one arm. Repeatable. Default: both.")
    ap.add_argument("--config_dir", default=None,
                    help="Directory of {arm}_s{seed}.yaml configs. Default configs/sweep. "
                         "Use configs/dodh for the DOD-H replication. Output directory names "
                         "must contain this directory's name, so one sweep cannot overwrite "
                         "another.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and stop.")
    ap.add_argument("--preflight", action="store_true", help="Check the environment and stop.")
    ap.add_argument("--skip-streaming", action="store_true",
                    help="Skip the --stream_stride pass (it costs 4x a windowed evaluation).")
    args = ap.parse_args()

    if args.config_dir:
        global CONFIG_DIR, TAG
        CONFIG_DIR = os.path.join(REPO, args.config_dir)
        TAG = os.path.basename(os.path.normpath(args.config_dir))
        if not os.path.isdir(CONFIG_DIR):
            sys.exit(f"no such config directory: {CONFIG_DIR}")

    arms = tuple(args.arm) if args.arm else ARMS
    configs = []
    for arm in arms:
        for seed in args.seeds:
            p = config_path(arm, seed)
            if not os.path.exists(p):
                sys.exit(f"missing config: {p}")
            configs.append((p, load_config(p)))

    if not preflight(configs):
        sys.exit("\nPreflight failed. Nothing was run.")
    if args.preflight:
        return

    stages = ["train", "eval"] + ([] if args.skip_streaming else ["eval_stream"])

    # Build the work list first so the plan is visible before anything starts.
    todo, skipped = [], 0
    for cfg_path, cfg in configs:
        log_dir, ckpt_dir = dirs_for(cfg)
        tag = os.path.splitext(os.path.basename(cfg_path))[0]
        for fold in args.folds:
            for stage in stages:
                if stage_done(stage, log_dir, ckpt_dir, fold):
                    skipped += 1
                    continue
                # A report can be missing while its summary row is present if a previous
                # run was killed between the two writes. Re-running would duplicate the row.
                suffix = f"_streaming{STREAM_STRIDE}" if stage == "eval_stream" else ""
                if stage.startswith("eval") and summary_has_fold(log_dir, fold, suffix):
                    print(f"  note  {tag} fold {fold} {stage}: summary row already present, "
                          f"skipping to avoid a duplicate. Delete the row and the report "
                          f"together if you want to redo it.")
                    skipped += 1
                    continue
                todo.append((cfg_path, tag, fold, stage))

    print(f"\n{len(todo)} stages to run, {skipped} already complete.")
    if args.dry_run or not todo:
        for cfg_path, tag, fold, stage in todo:
            print(f"  {tag:18s} fold {fold}  {stage}")
        return

    started = time.time()
    failures = []
    current = None
    for i, (cfg_path, tag, fold, stage) in enumerate(todo, 1):
        if (tag, fold) != current:
            current = (tag, fold)
            print(f"\n[{i}/{len(todo)}] {tag}  fold {fold}", flush=True)
        # Do not evaluate a fold whose training failed in this session.
        if stage.startswith("eval") and (tag, fold, "train") in failures:
            print(f"    {stage:12s} skipped, training failed")
            continue
        try:
            if not run_stage(stage, cfg_path, fold, tag):
                failures.append((tag, fold, stage))
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run the same command to resume from here.")
            sys.exit(130)

    hours = (time.time() - started) / 3600
    print(f"\n{'=' * 72}")
    print(f"Sweep finished in {hours:.1f} h. {len(todo) - len(failures)} of {len(todo)} "
          f"stages succeeded.")
    if failures:
        print("\nFailed stages (re-run the same command to retry only these):")
        for tag, fold, stage in failures:
            print(f"  {tag:18s} fold {fold}  {stage}")
        sys.exit(1)
    print("\nNext:")
    print("  python scripts/pool_seeds.py")
    print("  python scripts/analysis_stats.py --pair")
    for arm in arms:
        print(f"  python scripts/analyze_predictions.py logs_{TAG}_{arm}_s{args.seeds[0]} "
              f"--bootstrap --per-subject --prior-correction")
    print(f"  python scripts/boundary_latency.py logs_{TAG}_causal_s{args.seeds[0]} --hold 10")
    print(f"  python scripts/sweep_smoothing.py --predictions logs_{TAG}_causal_s{args.seeds[0]}")


if __name__ == "__main__":
    main()
