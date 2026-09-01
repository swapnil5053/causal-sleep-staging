"""Causal temporal smoothing of the per-second prediction stream.

The model emits an independent label every second, so its output is far more fragmented than
a human hypnogram (a technician changes stage roughly 20 times an hour). This sweeps a
trailing-window mode filter over the prediction stream and reports what it does to agreement
and to output stability.

The filter is causal: the label at second t is the mode of predictions over [t-w+1, t], so it
uses only past predictions and preserves the real-time property.

    python sweep_smoothing.py --config configs/sleep78_causal.yaml
    python sweep_smoothing.py --config configs/run_b_context.yaml \
        --checkpoint_dir checkpoints_runB --folds 0 1 2 3 4

Inference runs once per fold; the sweep is then almost free. Writes results/smoothing.md.
"""
import argparse
import os

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

STAGES = ["W", "N1", "N2", "N3", "REM"]


def causal_mode_filter(pred, w, n_classes=5):
    """Label at t = most common prediction in [t-w+1, t]. Ties go to the most recent.

    Implemented with a running count so it is O(n) rather than O(n*w).
    """
    if w <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    counts = np.zeros(n_classes, dtype=np.int64)
    for i, p in enumerate(pred):
        counts[p] += 1
        if i >= w:
            counts[pred[i - w]] -= 1
        best = counts.argmax()
        # tie-break toward the current prediction, which keeps genuine transitions responsive
        if counts[p] == counts[best]:
            best = p
        out[i] = best
    return out


def transitions_per_hour(seq):
    if len(seq) < 2:
        return 0.0
    return float((np.diff(seq) != 0).sum() / (len(seq) / 3600.0))


def predict_subject(model, torch, x, L):
    """Per-second predictions for one recording, in order, no cross-subject bleed."""
    n = (len(x) // L) * L
    dev = next(model.parameters()).device      # keep the batch on the model's device
    preds = []
    with torch.no_grad():
        for s in range(0, n, L):
            win = torch.from_numpy(x[s:s + L]).float().unsqueeze(0).to(dev)
            preds.append(torch.argmax(model(win), dim=-1)[0].cpu().numpy())
    return (np.concatenate(preds) if preds else np.array([], dtype=int)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sleep78_causal.yaml")
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[1, 5, 10, 15, 30, 45, 60, 90, 120])
    ap.add_argument("--out", default="results/smoothing.md")
    args = ap.parse_args()

    import torch
    from src.model.full_model import SleepStagingModel

    cfg = yaml.safe_load(open(args.config))
    L = cfg["data"]["sequence_length"]
    proc = cfg["data"]["processed_dir"]
    ckpt_dir = args.checkpoint_dir or cfg["train"]["checkpoint_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {args.config} | checkpoints {ckpt_dir} | device {device}")

    # raw predictions and truth, kept per subject so smoothing never crosses a recording
    per_fold = {}
    for fold in args.folds:
        ck_path = os.path.join(ckpt_dir, f"best_model_fold_{fold}.pth")
        split_path = os.path.join(ckpt_dir, f"split_fold_{fold}.yaml")
        if not (os.path.exists(ck_path) and os.path.exists(split_path)):
            print(f"  fold {fold}: checkpoint or split missing, skipping")
            continue
        model = SleepStagingModel(config=cfg)
        model.load_state_dict(torch.load(ck_path, map_location=device,
                                         weights_only=False)["model_state_dict"])
        model = model.to(device).eval()
        subs = yaml.safe_load(open(split_path))["test_subjects"]

        pairs = []
        for s in subs:
            p = os.path.join(proc, f"subject_{s}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            pred, n = predict_subject(model, torch, d["x"], L)
            if n:
                pairs.append((pred, d["y"][:n]))
        per_fold[fold] = pairs
        print(f"  fold {fold}: {len(pairs)} test recordings, "
              f"{sum(len(p) for p, _ in pairs):,} seconds")

    if not per_fold:
        print("nothing to evaluate")
        return

    # sweep
    rows = []
    for w in args.windows:
        accs, kaps, f1s, n1s, trans = [], [], [], [], []
        for fold, pairs in per_fold.items():
            P = np.concatenate([causal_mode_filter(p, w) for p, _ in pairs])
            Y = np.concatenate([y for _, y in pairs])
            accs.append(accuracy_score(Y, P))
            kaps.append(cohen_kappa_score(Y, P))
            f1s.append(f1_score(Y, P, average="macro", zero_division=0))
            n1s.append(f1_score(Y, P, average=None, zero_division=0, labels=[0, 1, 2, 3, 4])[1])
            trans.append(np.mean([transitions_per_hour(causal_mode_filter(p, w))
                                  for p, _ in pairs]))
        rows.append(dict(w=w, acc=np.mean(accs), kappa=np.mean(kaps), f1=np.mean(f1s),
                         n1=np.mean(n1s), trans=np.mean(trans), kappa_sd=np.std(kaps, ddof=1)))

    ref_trans = np.mean([transitions_per_hour(y) for pairs in per_fold.values()
                         for _, y in pairs])

    best = max(rows, key=lambda r: r["kappa"])
    base = rows[0]

    L_ = []
    a = L_.append
    a("# Causal temporal smoothing\n")
    a("The model labels each second independently, so its output changes stage far more often")
    a("than a scored hypnogram. A trailing-window mode filter is applied to the prediction")
    a("stream: the label at second t is the most common prediction over [t-w+1, t]. It uses only")
    a("past predictions, so the real-time property is preserved, and it costs nothing to run.\n")
    a(f"Config: `{args.config}`, {len(per_fold)} folds, smoothing applied within each recording.\n")
    a("| Window (s) | Accuracy | Kappa | Macro F1 | N1 F1 | Stage changes/hour |")
    a("|---|---|---|---|---|---|")
    for r in rows:
        mark = " **" if r["w"] == best["w"] else ""
        a(f"| {r['w']}{mark} | {r['acc']:.4f} | {r['kappa']:.4f} | {r['f1']:.4f} | "
          f"{r['n1']:.3f} | {r['trans']:.0f} |")
    a("")
    a(f"Human scoring in the same recordings changes stage **{ref_trans:.0f} times an hour**.\n")
    a(f"Unsmoothed output changes **{base['trans']:.0f} times an hour**, "
      f"{base['trans']/max(ref_trans,1e-9):.0f}x more often than a technician.\n")
    a(f"Best window is **{best['w']} s**: kappa {best['kappa']:.4f} against "
      f"{base['kappa']:.4f} unsmoothed ({best['kappa']-base['kappa']:+.4f}), "
      f"with stage changes falling from {base['trans']:.0f} to {best['trans']:.0f} per hour.\n")
    a("Smoothing is applied at inference only. No retraining is involved and the model is")
    a("unchanged, so this is a free post-processing gain available to any per-second model.\n")
    text = "\n".join(L_) + "\n"
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open(args.out, "w").write(text)
    print("\n" + text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
