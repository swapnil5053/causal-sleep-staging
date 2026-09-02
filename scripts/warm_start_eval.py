"""Evaluate with a warm-started context, and report the causality cost both ways.

`src/eval/evaluate.py` scores non-overlapping windows: at position 0 of each window the
causal model has one second of history while the non-causal model attends over the following
119. Roughly the first quarter of every window is context-starved, for one arm only, and never
would be in a streaming deployment holding a rolling buffer. Part of the measured causality
cost is therefore an artefact of how it is measured, not of causality.

This script measures how much. It scores the same seconds twice:

``cold``
    exactly what `evaluate.py` does - windows of `sequence_length`, stride equal to the window,
    every position scored. Reproduces the archived numbers.

``warm``
    overlapping windows at `--stride`, keeping only the last `stride` predictions of each. Every
    scored second then carries at least ``sequence_length - stride`` seconds of context, which
    is what `scripts/streaming_demo.py --mode rolling` shows the deployed path actually has.

The two are compared over the identical set of seconds, so the difference is context and
nothing else. The archived cold number over all seconds is reported alongside, so nothing is
quietly restated.

    python scripts/warm_start_eval.py --config configs/sleep78_streaming_causal.yaml \
        --checkpoint_dir checkpoints_78streaming_causal_s42 --fold 0 \
        --save_predictions logs_warm/fold_0_predictions.npz

`--save_predictions` writes per-second probabilities, predictions, targets and subject ids for
both modes, so `scripts/calibration.py` and `scripts/transition_response.py` can run without
another forward pass.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SleepDataset  # noqa: E402
from src.model.full_model import SleepStagingModel  # noqa: E402

STAGES = ["W", "N1", "N2", "N3", "REM"]


# ------------------------------------------------------------------- assembling seconds

def cold_seconds(window_subjects, window_values, seq_len):
    """Every position of every non-overlapping window, grouped by subject.

    ``window_values`` is one array per window, shaped ``(seq_len, ...)``. Returns
    ``{subject: array}`` indexed from second 0 of that subject's retained recording.
    """
    if len(window_subjects) != len(window_values):
        raise ValueError("one value array per window is required")
    out = {}
    for subject, values in zip(window_subjects, window_values):
        if len(values) != seq_len:
            raise ValueError(f"expected {seq_len} positions per window, got {len(values)}")
        out.setdefault(subject, []).append(np.asarray(values))
    return {s: np.concatenate(v) for s, v in out.items()}


def warm_seconds(window_subjects, window_values, seq_len, stride):
    """The last ``stride`` positions of each overlapping window, grouped by subject.

    Window ``i`` of a subject spans seconds ``[i*stride, i*stride + seq_len)``, so keeping its
    final ``stride`` positions yields seconds ``[i*stride + seq_len - stride, i*stride +
    seq_len)``. Those are contiguous across ``i`` and start at ``seq_len - stride``, which is
    the first second that can carry a full warm context. Returns ``(offset, {subject: array})``.
    """
    if not 0 < stride <= seq_len:
        raise ValueError(f"stride must be in (0, seq_len], got {stride}")
    out = {}
    for subject, values in zip(window_subjects, window_values):
        values = np.asarray(values)
        if len(values) != seq_len:
            raise ValueError(f"expected {seq_len} positions per window, got {len(values)}")
        out.setdefault(subject, []).append(values[seq_len - stride:])
    return seq_len - stride, {s: np.concatenate(v) for s, v in out.items()}


def common_span(cold, warm, offset):
    """Restrict both modes to the seconds they both scored, per subject.

    Cold starts at second 0; warm starts at ``offset``. Returns ``(cold_slice, warm_slice)``
    dictionaries covering exactly the same absolute seconds for every subject.
    """
    cold_out, warm_out = {}, {}
    for subject, warm_values in warm.items():
        cold_values = cold[subject]
        end = min(len(cold_values), offset + len(warm_values))
        if end <= offset:
            continue
        cold_out[subject] = cold_values[offset:end]
        warm_out[subject] = warm_values[: end - offset]
    return cold_out, warm_out


def score(targets, preds):
    return {
        "n_seconds": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "kappa": float(cohen_kappa_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "per_class_f1": f1_score(targets, preds, average=None, labels=[0, 1, 2, 3, 4],
                                 zero_division=0),
    }


# ------------------------------------------------------------------------- the forward pass

def run(model, dataset, device, batch_size=1):
    """Return one probability array per window, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    windows = []
    with torch.no_grad():
        for x, _ in loader:
            probabilities = torch.softmax(model(x.to(device)), dim=-1).cpu().numpy()
            windows.extend(probabilities)
    return windows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--stride", type=int, default=None,
                    help="warm stride in seconds; default sequence_length // 4, so every scored "
                         "second carries at least three quarters of a window of context")
    ap.add_argument("--save_predictions", default=None,
                    help="write an .npz of per-second probabilities for both modes")
    ap.add_argument("--out", default=None, help="write a markdown report here")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seq_len = int(config["data"]["sequence_length"])
    stride = args.stride if args.stride else max(1, seq_len // 4)
    if not 0 < stride <= seq_len:
        raise SystemExit(f"--stride must be in (0, {seq_len}]")

    checkpoint_dir = args.checkpoint_dir or config["train"]["checkpoint_dir"]
    processed_dir = args.processed_dir or config["data"]["processed_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{args.fold}.pth")
    split_path = os.path.join(checkpoint_dir, f"split_fold_{args.fold}.yaml")
    for path in (checkpoint_path, split_path):
        if not os.path.exists(path):
            raise SystemExit(f"not found: {path}")

    test_subjects = yaml.safe_load(open(split_path))["test_subjects"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SleepStagingModel(config=config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("=" * 74)
    print(f"WARM-STARTED EVALUATION - fold {args.fold}")
    print("=" * 74)
    print(f"Context {seq_len} s, warm stride {stride} s "
          f"(minimum warm context {seq_len - stride} s)")
    print(f"Test subjects: {len(test_subjects)}")

    cold_dataset = SleepDataset(processed_dir, test_subjects, seq_len=seq_len, stride=seq_len)
    warm_dataset = SleepDataset(processed_dir, test_subjects, seq_len=seq_len, stride=stride)

    cold_probabilities = cold_seconds(cold_dataset.window_subjects,
                                      run(model, cold_dataset, device), seq_len)
    cold_labels = cold_seconds(cold_dataset.window_subjects, cold_dataset.labels, seq_len)
    offset, warm_probabilities = warm_seconds(warm_dataset.window_subjects,
                                              run(model, warm_dataset, device), seq_len, stride)

    # Cold over everything it scores: the number comparable to the archived reports.
    full_targets = np.concatenate([cold_labels[s] for s in sorted(cold_labels)])
    full_preds = np.concatenate([cold_probabilities[s].argmax(-1) for s in sorted(cold_labels)])
    full = score(full_targets, full_preds)

    # Then the paired comparison, over the seconds both modes scored.
    cold_common, warm_common = common_span(cold_probabilities, warm_probabilities, offset)
    label_common, _ = common_span(cold_labels, warm_probabilities, offset)
    subjects = sorted(cold_common)
    targets = np.concatenate([label_common[s] for s in subjects])
    cold_restricted = score(targets, np.concatenate([cold_common[s].argmax(-1) for s in subjects]))
    warm_restricted = score(targets, np.concatenate([warm_common[s].argmax(-1) for s in subjects]))

    rows = [("cold, all scored seconds (archived)", full),
            ("cold, restricted to the common span", cold_restricted),
            ("warm, restricted to the common span", warm_restricted)]
    print()
    print(f"{'mode':38s} {'seconds':>10s} {'acc':>8s} {'kappa':>8s} {'macroF1':>8s}")
    for name, m in rows:
        print(f"{name:38s} {m['n_seconds']:10,d} {m['accuracy']:8.4f} {m['kappa']:8.4f} "
              f"{m['macro_f1']:8.4f}")
    delta = warm_restricted["kappa"] - cold_restricted["kappa"]
    print()
    print(f"Warm start changes kappa by {delta:+.4f} on identical seconds.")
    print("A positive value means part of the archived causality cost was a window-boundary")
    print("artefact rather than a cost of causality.")

    if args.save_predictions:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_predictions,
            subjects=np.array([s for s in subjects for _ in range(len(cold_common[s]))]),
            targets=targets,
            cold_probabilities=np.concatenate([cold_common[s] for s in subjects]).astype(np.float32),
            warm_probabilities=np.concatenate([warm_common[s] for s in subjects]).astype(np.float32),
            seq_len=np.array(seq_len),
            stride=np.array(stride),
            offset=np.array(offset),
            fold=np.array(args.fold),
        )
        print(f"\npredictions written to {args.save_predictions}")

    if args.out:
        lines = ["# Warm-started evaluation\n",
                 f"Fold {args.fold}, context {seq_len} s, warm stride {stride} s "
                 f"(minimum warm context {seq_len - stride} s).\n",
                 "Cold scoring uses non-overlapping windows, so the first seconds of every window "
                 "carry almost no context for the causal arm. Warm scoring keeps only the last "
                 "`stride` predictions of overlapping windows, which is the context a streaming "
                 "deployment actually has. Both are scored over identical seconds.\n",
                 "| Mode | Seconds | Accuracy | Kappa | Macro F1 |", "|---|---:|---:|---:|---:|"]
        for name, m in rows:
            lines.append(f"| {name} | {m['n_seconds']:,} | {m['accuracy']:.4f} | "
                         f"{m['kappa']:.4f} | {m['macro_f1']:.4f} |")
        lines += ["", f"Warm start changes kappa by **{delta:+.4f}** on identical seconds.", ""]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"report written to {args.out}")

    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
