"""Accuracy as a function of permitted latency, from a fixed context buffer.

The question this answers: given a buffer of `sequence_length` seconds, where in that buffer
should a system place the second it is labelling? Placing it at the very end means zero
latency and no lookahead. Placing it L seconds earlier means the model sees L seconds of
future, at the price of reporting L seconds late, and of L seconds less past history.

That trade is the whole real-time argument, and it is measurable without retraining:

  - The causal model cannot read future signal, so its curve should be flat or decline
    as L grows (it only loses history). Its correct operating point is L = 0.
  - The non-causal model gains lookahead as L grows and should improve, recovering toward
    its offline score.

Where the two curves cross, and how much latency the non-causal model needs before it is
worth its extra capability, is the paper's central figure.

## Why this is cheap

A single forward pass over a window produces a prediction at *every* offset in that window,
and offset p carries exactly (sequence_length - 1 - p) seconds of lookahead. So one sweep of
windows yields every latency at once, rather than one evaluation pass per latency.

The cost is that each latency is scored on a different 1-in-`stride` subsample of seconds.
All subsamples are uniform over the same recordings and each still contains hundreds of
thousands of seconds, so the curve is stable; treat small jitter between adjacent L as
sampling noise and read the shape, not individual points.

Usage:
    python scripts/latency_sweep.py --config configs/sleep78_streaming_causal.yaml \
        --out results/latency_causal.md
    python scripts/latency_sweep.py --config configs/sleep78_streaming_noncausal.yaml \
        --out results/latency_noncausal.md
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.full_model import SleepStagingModel  # noqa: E402

CLASSES = ("W", "N1", "N2", "N3", "REM")


def window_starts(n, seq_len, stride):
    """Window start positions covering the recording, including one flush with the end."""
    if n < seq_len:
        return []
    starts = list(range(0, n - seq_len + 1, stride))
    if starts[-1] != n - seq_len:
        starts.append(n - seq_len)
    return starts


def collect_by_offset(model, processed_dir, subject_ids, seq_len, stride, device):
    """Return per-offset arrays of (true, predicted) over every window in every subject.

    Offset p means the label was produced at position p of a seq_len window, so it saw
    p + 1 seconds of past and seq_len - 1 - p seconds of future.
    """
    true_by_off = [[] for _ in range(seq_len)]
    pred_by_off = [[] for _ in range(seq_len)]

    with torch.no_grad():
        for sub_id in subject_ids:
            path = os.path.join(processed_dir, f"subject_{sub_id}.npz")
            if not os.path.exists(path):
                print(f"  warning: no processed file for subject {sub_id}, skipping")
                continue
            data = np.load(path)
            x, y = data["x"], data["y"]
            n = len(x)
            starts = window_starts(n, seq_len, stride)
            if not starts:
                print(f"  warning: subject {sub_id} shorter than the buffer, skipping")
                continue

            # Slice each batch as it is needed rather than materialising every window at
            # once. At stride 1 a single night is ~59,000 windows, and stacking them all
            # would be ~2.8 GB of float32 for one subject.
            # Attention memory grows with the square of the buffer, so shrink the batch
            # as the buffer grows rather than discovering the limit on an 8 GB card.
            bs = max(8, int(64 * 120 / seq_len))
            preds = np.empty((len(starts), seq_len), dtype=np.int8)
            for i in range(0, len(starts), bs):
                sl = starts[i:i + bs]
                chunk = torch.from_numpy(
                    np.stack([x[s:s + seq_len] for s in sl])).float().to(device)
                preds[i:i + bs] = model(chunk).argmax(dim=-1).cpu().numpy().astype(np.int8)

            truth = np.empty((len(starts), seq_len), dtype=np.int8)
            for i, s in enumerate(starts):
                truth[i] = y[s:s + seq_len]
            for p in range(seq_len):
                true_by_off[p].append(truth[:, p])
                pred_by_off[p].append(preds[:, p])

    return ([np.concatenate(a) for a in true_by_off],
            [np.concatenate(a) for a in pred_by_off])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--buffer", type=int, default=None,
                    help="Context buffer in seconds. Defaults to the trained "
                         "sequence_length. Setting it larger evaluates the same weights on "
                         "a longer window than they were trained on, which is a train/test "
                         "mismatch and must be reported as one. The causal arm at offset "
                         "(trained_len - 1) is the control: its history and its mask are "
                         "unchanged, so any change there is the length effect alone.")
    ap.add_argument("--stride", type=int, default=30,
                    help="window spacing. Smaller means more seconds scored per latency "
                         "and proportionally more compute.")
    ap.add_argument("--report_at", type=int, nargs="+", default=None,
                    help="latencies in seconds to tabulate. The full curve is always "
                         "written to the CSV.")
    ap.add_argument("--out", default=None, help="markdown report path")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    trained_len = config["data"]["sequence_length"]
    seq_len = args.buffer or trained_len
    if args.report_at is None:
        pts = [0, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 119, 120, 150, 180, 240, 300]
        args.report_at = sorted({p for p in pts if p < seq_len} | {seq_len - 1})
    processed_dir = args.processed_dir or config["data"]["processed_dir"]
    checkpoint_dir = args.checkpoint_dir or config["train"]["checkpoint_dir"]
    causal = config["model"]["causal"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {args.config}\ncausal={causal}  buffer={seq_len}s "
          f"(trained at {trained_len}s)  stride={args.stride}s  device={device}")
    if seq_len != trained_len:
        print(f"  NOTE: buffer != trained length. Control point is offset "
              f"{trained_len - 1} (latency {seq_len - trained_len}), where the causal arm "
              f"has exactly {trained_len}s of history as in training.")

    # Accumulate across folds so each latency is scored on the whole held-out population.
    agg_true = [[] for _ in range(seq_len)]
    agg_pred = [[] for _ in range(seq_len)]

    for fold in args.folds:
        ckpt_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")
        split_path = os.path.join(checkpoint_dir, f"split_fold_{fold}.yaml")
        if not os.path.exists(ckpt_path):
            sys.exit(f"missing checkpoint {ckpt_path}")
        test_subs = yaml.safe_load(open(split_path))["test_subjects"]

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SleepStagingModel(config=config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        print(f"fold {fold}: {len(test_subs)} test subjects")
        t, p = collect_by_offset(model, processed_dir, test_subs, seq_len,
                                 args.stride, device)
        for off in range(seq_len):
            agg_true[off].append(t[off])
            agg_pred[off].append(p[off])

    rows = []
    for off in range(seq_len):
        yt = np.concatenate(agg_true[off])
        yp = np.concatenate(agg_pred[off])
        lookahead = seq_len - 1 - off
        # Per-class F1 over the fixed 0-4 label set, so a class absent from a subsample
        # scores 0 rather than shifting the column order.
        per_class = f1_score(yt, yp, average=None, labels=list(range(len(CLASSES))),
                             zero_division=0)
        rows.append(dict(
            latency=lookahead, history=off + 1, n=len(yt),
            accuracy=accuracy_score(yt, yp),
            kappa=cohen_kappa_score(yt, yp),
            macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
            per_class=per_class,
        ))
    rows.sort(key=lambda r: r["latency"])

    csv_path = (args.out or "latency_sweep.md").rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("latency_s,history_s,n_seconds,accuracy,kappa,macro_f1,"
                + ",".join(f"f1_{c}" for c in CLASSES) + "\n")
        for r in rows:
            f.write(f"{r['latency']},{r['history']},{r['n']},{r['accuracy']:.4f},"
                    f"{r['kappa']:.4f},{r['macro_f1']:.4f},"
                    + ",".join(f"{v:.4f}" for v in r["per_class"]) + "\n")

    by_lat = {r["latency"]: r for r in rows}
    zero = by_lat[0]
    best = max(rows, key=lambda r: r["kappa"])

    lines = [
        "# Accuracy versus permitted latency",
        "",
        f"Model: `{args.config}` (`causal: {causal}`). Buffer {seq_len} s "
        f"(trained at {trained_len} s), window stride {args.stride} s, folds {args.folds}.",
        "",
        ("Buffer differs from the trained sequence length, so this is a train/test length "
         f"mismatch. The control is latency {seq_len - trained_len} s, at which the causal "
         f"arm sees exactly {trained_len} s of history with its future masked, as in "
         "training: if its score there matches the trained-length run, length "
         "extrapolation is not affecting the comparison."
         if seq_len != trained_len else
         "Buffer equals the trained sequence length."),
        "",
        "Latency L means the label for second *t* is emitted at *t + L*, so the model sees "
        "L seconds of future and `buffer - L` seconds of past. L = 0 is real time.",
        "",
        "| Latency (s) | History (s) | Seconds scored | Accuracy | Kappa | Macro-F1 |",
        "|---:|---:|---:|---|---|---|",
    ]
    for lat in args.report_at:
        if lat in by_lat:
            r = by_lat[lat]
            mark = " **" if lat == best["latency"] else ""
            lines.append(f"| {lat}{mark} | {r['history']} | {r['n']:,} | {r['accuracy']:.4f} "
                         f"| {r['kappa']:.4f} | {r['macro_f1']:.4f} |")
    lines += [
        "",
        f"At zero latency kappa is **{zero['kappa']:.4f}**. The best latency is "
        f"**{best['latency']} s** at kappa **{best['kappa']:.4f}** "
        f"({best['kappa'] - zero['kappa']:+.4f}).",
        "",
        "## Per-class F1 by latency",
        "",
        "The aggregate penalty under batch tiling is dominated by short-history positions. "
        "This table is what decides whether a per-class claim measured under tiling still "
        "holds at the latency a deployment actually runs at.",
        "",
        "| Latency (s) | " + " | ".join(CLASSES) + " |",
        "|---:|" + "---|" * len(CLASSES),
    ]
    for lat in args.report_at:
        if lat in by_lat:
            lines.append(f"| {lat} | "
                         + " | ".join(f"{v:.4f}" for v in by_lat[lat]["per_class"]) + " |")
    lines += [
        "",
        "For a causal model this curve should be flat or falling: it cannot read future "
        "signal, so a larger L only costs it history. For a non-causal model it should rise, "
        "and the rise is what lookahead is actually worth in seconds of delay.",
        "",
        "Each latency is scored on a different 1-in-stride subsample of seconds, all uniform "
        "over the same recordings. Read the shape of the curve; treat small differences "
        "between adjacent latencies as sampling noise.",
        "",
        f"Full curve for every latency 0-{seq_len - 1}: `{os.path.basename(csv_path)}`.",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(text + "\n")
        print(f"\nwritten to {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
