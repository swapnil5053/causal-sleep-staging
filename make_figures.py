"""Generate the paper/report figures from artifacts already in results/.

    python make_figures.py                          # defaults to the Sleep-EDF-78 causal run
    python make_figures.py --run results/run_b_context
    python make_figures.py --checkpoint checkpoints_78causal/best_model_fold_0.pth \
                           --subject data/processed78/subject_19.npz

Produces, in figures/:
  fig_confusion.png   normalised confusion matrix, summed over folds
  fig_ablation.png    kappa per run with per-fold points
  fig_hypnogram.png   predicted vs scored stages across one night (needs torch + a checkpoint)

The hypnogram is skipped with a message if the checkpoint or subject file is missing, so the
other two always render.
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["W", "N1", "N2", "N3", "REM"]
# colour-blind safe, dark = deeper sleep
STAGE_COLOURS = ["#D55E00", "#E69F00", "#56B4E9", "#0072B2", "#009E73"]


def parse_confusion(path):
    """Pull the 5x5 confusion matrix out of a fold_N_test_report.txt."""
    text = open(path).read()
    block = text.split("Confusion Matrix")[-1].splitlines()[2:]  # skip header + column names
    rows = []
    for line in block:
        nums = re.findall(r"\d+", line)
        if len(nums) >= 5:
            rows.append([int(x) for x in nums[-5:]])
        if len(rows) == 5:
            break
    if len(rows) != 5:
        raise ValueError(f"could not parse a 5x5 matrix from {path}")
    return np.array(rows)


def figure_confusion(run_dir, out_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "fold_*_test_report.txt")))
    if not files:
        print(f"  skip confusion: no fold reports in {run_dir}")
        return
    total = sum(parse_confusion(f) for f in files)
    norm = total / total.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), STAGES)
    ax.set_yticks(range(5), STAGES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Scored by technician")
    ax.set_title(f"Confusion matrix, {len(files)} folds pooled\n(row-normalised)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_confusion.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p}")


def figure_ablation(out_dir):
    """Kappa per run, with the five fold values scattered over each bar."""
    import csv
    runs = [
        ("60 s\n20 subj", "results/run_a_baseline"),
        ("120 s\n20 subj", "results/run_b_context"),
        ("120 s +4blk\n20 subj", "results/run_c_depth"),
        ("120 s\n78 subj", "results/sleep78_causal"),
        ("120 s non-causal\n78 subj", "results/sleep78_noncausal"),
        # end-to-end causal preprocessing pilot; skipped automatically until archived
        ("streaming norm\n78 subj", "results/sleep78_streaming_causal"),
        ("streaming norm\nnon-causal", "results/sleep78_streaming_noncausal"),
    ]
    labels, means, folds, noncausal = [], [], [], []
    for label, d in runs:
        p = None
        for fn in ("test_metrics_summary.csv", "summary.csv"):
            if os.path.exists(os.path.join(d, fn)):
                p = os.path.join(d, fn)
                break
        if p is None:
            continue
        k = [float(r["kappa"]) for r in csv.DictReader(open(p))]
        labels.append(label)
        means.append(np.mean(k))
        folds.append(k)
        noncausal.append("noncausal" in d)
    if not labels:
        print("  skip ablation: no run summaries found")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # blue = causal (the deployable model), red = non-causal ablation
    colours = ["#c47f7f" if nc else "#7f9fc4" for nc in noncausal]
    bars = ax.bar(labels, means, color=colours, edgecolor="black", linewidth=0.6)
    for i, k in enumerate(folds):
        ax.scatter([i] * len(k), k, color="black", zorder=3, s=18, alpha=0.75)
    for b, mu, k in zip(bars, means, folds):
        ax.text(b.get_x() + b.get_width() / 2, max(k) + 0.018, f"{mu:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(0, max(max(max(f) for f in folds) * 1.18, 0.05))  # stay valid if a run collapsed
    ax.set_title("Cross-validated kappa by configuration\n(black points are individual folds)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_ablation.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p}")


def figure_hypnogram(checkpoint, subject_npz, config_path, out_dir, hours=8):
    if not (checkpoint and os.path.exists(checkpoint)):
        print(f"  skip hypnogram: checkpoint not found ({checkpoint})")
        return
    if not (subject_npz and os.path.exists(subject_npz)):
        print(f"  skip hypnogram: subject file not found ({subject_npz})")
        return
    try:
        import torch
        import yaml
        from src.model.full_model import SleepStagingModel
    except Exception as e:
        print(f"  skip hypnogram: {type(e).__name__}: {e}")
        return

    cfg = yaml.safe_load(open(config_path))
    L = cfg["data"]["sequence_length"]

    data = np.load(subject_npz)
    x, y = data["x"], data["y"]
    n = min(len(x), int(hours * 3600))
    n -= n % L                                   # whole windows only
    x, y = x[:n], y[:n]

    model = SleepStagingModel(config=cfg)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for s in range(0, n, L):
            win = torch.from_numpy(x[s:s + L]).float().unsqueeze(0)
            preds.append(torch.argmax(model(win), dim=-1)[0].numpy())
    preds = np.concatenate(preds)

    hrs = np.arange(n) / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True)
    for ax, series, title in ((axes[0], y, "Scored by technician"),
                              (axes[1], preds, "Model, causal, one prediction per second")):
        for s in range(5):
            mask = series == s
            ax.fill_between(hrs, 0, 1, where=mask, color=STAGE_COLOURS[s],
                            step="mid", linewidth=0)
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xlim(0, hrs[-1])
    axes[1].set_xlabel("Hours from start of recording")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLOURS]
    axes[0].legend(handles, STAGES, ncol=5, loc="lower center",
                   bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)
    agree = (preds == y).mean()
    fig.suptitle(f"{os.path.basename(subject_npz)}  |  per-second agreement {agree:.1%}",
                 y=0.02, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_hypnogram.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}  (agreement {agree:.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/sleep78_causal",
                    help="run directory holding fold_*_test_report.txt")
    ap.add_argument("--checkpoint", default="checkpoints_78causal/best_model_fold_0.pth")
    ap.add_argument("--subject", default=None,
                    help="a .npz from the run's held-out test set")
    ap.add_argument("--config", default="configs/sleep78_causal.yaml")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--hours", type=float, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"run: {args.run}")

    subject = args.subject
    if subject is None:                          # pick a held-out subject automatically
        import yaml
        split = os.path.join(os.path.dirname(args.checkpoint), "split_fold_0.yaml")
        proc = yaml.safe_load(open(args.config))["data"]["processed_dir"]
        if os.path.exists(split):
            test = yaml.safe_load(open(split))["test_subjects"]
            cand = os.path.join(proc, f"subject_{test[0]}.npz")
            subject = cand if os.path.exists(cand) else None

    figure_confusion(args.run, args.out)
    figure_ablation(args.out)
    figure_hypnogram(args.checkpoint, subject, args.config, args.out, args.hours)
    print("done")


if __name__ == "__main__":
    main()
