import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix

from src.data.dataset import SleepDataset, segment_starts
from src.model.full_model import SleepStagingModel

def get_config(config_path):
    """Load config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _empty_result():
    empty = np.empty(0, dtype=np.int64)
    return {"y_pred": empty, "y_true": empty, "logits": np.empty((0, 5), dtype=np.float32),
            "subject": np.empty(0, dtype=object), "second_index": empty,
            "segment_start": np.empty(0, dtype=bool)}


def _segment_flags(second_index, subjects, segment_starts_by_subject):
    """Mark each scored second that begins a fresh, discontinuous stretch of signal.

    Carried through to the saved predictions so a later analysis can tell a real stage
    change from the splice left by a dropped epoch or the join between two nights.
    """
    flags = np.zeros(len(second_index), dtype=bool)
    for sub_id, starts in segment_starts_by_subject.items():
        if starts is None or len(starts) == 0:
            continue
        mask = subjects == sub_id
        if mask.any():
            flags[mask] = np.isin(second_index[mask], starts)
    return flags


def predict_windowed(model, dataset, device):
    """Predict over the dataset's windows in order, one window at a time.

    This is the archived evaluation: windows tile the recording without overlap, so the
    first seconds of each window are predicted from a partly zero-padded history.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    preds, targets, logits = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(torch.argmax(out, dim=-1).cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
            logits.append(out.cpu().numpy().reshape(-1, out.shape[-1]))
    if not preds:
        return _empty_result()

    seq_len = dataset.windows.shape[1]
    subjects = np.repeat(np.asarray(dataset.window_subjects, dtype=object), seq_len)
    second_index = (np.asarray(dataset.window_starts, dtype=np.int64)[:, None]
                    + np.arange(seq_len, dtype=np.int64)[None, :]).reshape(-1)
    return {
        "y_pred": np.concatenate(preds),
        "y_true": np.concatenate(targets),
        "logits": np.concatenate(logits),
        "subject": subjects,
        "second_index": second_index,
        "segment_start": _segment_flags(second_index, subjects,
                                        dataset.segment_starts_by_subject),
    }

def predict_streaming(model, processed_dir, subject_ids, seq_len, stream_stride, device):
    """Predict with a rolling context, keeping only the freshest seconds of each window.

    A deployment recomputes every ``stream_stride`` seconds against a full buffer rather
    than restarting from an empty one every ``seq_len`` seconds. Each window contributes
    only its last ``stream_stride`` outputs, so every second after the opening window is
    decided with at least ``seq_len - stream_stride`` seconds of genuine history. The
    opening window is the unavoidable warm-up and is kept as-is.

    With ``stream_stride == seq_len`` this reduces to the non-overlapping tiling of
    :func:`predict_windowed`, plus the trailing seconds that tiling drops.
    """
    if not 1 <= stream_stride <= seq_len:
        raise ValueError(f"stream_stride must be in [1, {seq_len}], got {stream_stride}")

    preds, targets, logits, subjects = [], [], [], []
    indices, flags = [], []
    with torch.no_grad():
        for sub_id in subject_ids:
            path = os.path.join(processed_dir, f"subject_{sub_id}.npz")
            if not os.path.exists(path):
                print(f"Warning: no processed file for subject {sub_id}, skipping.")
                continue
            data = np.load(path)
            x, y = data['x'], data['y']
            n = len(x)
            starts = segment_starts(data, n)
            if n < seq_len:
                print(f"Warning: subject {sub_id} is shorter than the context window, skipping.")
                continue

            # (window start, first second to keep) pairs covering [0, n) exactly once.
            plan = [(0, 0)]
            covered = seq_len
            while covered < n:
                start = min(covered + stream_stride - seq_len, n - seq_len)
                plan.append((start, covered))
                covered = start + seq_len

            sub_pred = np.empty(n, dtype=np.int64)
            sub_logit = np.empty((n, 5), dtype=np.float32)
            for start, keep_from in plan:
                window = torch.from_numpy(x[start:start + seq_len]).float().unsqueeze(0).to(device)
                out = model(window)[0].cpu().numpy()
                offset = keep_from - start
                sub_pred[keep_from:start + seq_len] = out[offset:].argmax(axis=-1)
                sub_logit[keep_from:start + seq_len] = out[offset:]

            within = np.arange(n, dtype=np.int64)
            preds.append(sub_pred)
            targets.append(y[:n].astype(np.int64))
            logits.append(sub_logit)
            subjects.append(np.full(n, sub_id, dtype=object))
            indices.append(within)
            flags.append(np.isin(within, starts) if starts is not None
                         else np.zeros(n, dtype=bool))

    if not preds:
        return _empty_result()
    return {
        "y_pred": np.concatenate(preds),
        "y_true": np.concatenate(targets),
        "logits": np.concatenate(logits),
        "subject": np.concatenate(subjects),
        "second_index": np.concatenate(indices),
        "segment_start": np.concatenate(flags),
    }

def run_latency_benchmark(model_class, config, num_runs=200):
    """
    Measure inference latency on CPU to verify compliance with the <= 3ms/sec target.
    Simulates a sequence of EEG inputs and benchmarks CPU processing wall-clock time.
    """
    print("\n==================================================")
    print("           INFERENCE LATENCY BENCHMARK")
    print("==================================================")
    
    # Initialize model on CPU
    model = model_class(config=config)
    model.eval()
    
    # Context sequence length (L seconds)
    L = config['data']['sequence_length']
    # Dummy input representing L seconds of 100 Hz raw EEG (batch_size=1, sequence_len=L, sample_rate=100)
    dummy_input = torch.randn(1, L, 100)
    
    # Warm-up runs
    print("Warming up model...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    # Benchmark runs
    print(f"Profiling over {num_runs} runs on CPU...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    # Calculation
    total_elapsed_ms = (end_time - start_time) * 1000.0
    avg_per_run_ms = total_elapsed_ms / num_runs
    
    # Each run processes a sequence of length L seconds of EEG.
    # Latency is defined as: ms of computation / second of EEG
    latency_per_sec_ms = avg_per_run_ms / L
    
    print(f"\nBenchmark Results (CPU):")
    print(f"  Total time for {num_runs} runs: {total_elapsed_ms:.2f} ms")
    print(f"  Average time per {L}-second sequence: {avg_per_run_ms:.3f} ms")
    print(f"  Inference Latency per second of EEG: {latency_per_sec_ms:.4f} ms/sec")
    
    target_threshold = 3.0 # ms/sec target
    if latency_per_sec_ms <= target_threshold:
        print(f"  Status: PASSED (Under the target threshold of {target_threshold} ms/sec)")
    else:
        print(f"  Status: FAILED/SLOW (Exceeded the target threshold of {target_threshold} ms/sec)")
    print("==================================================")

def evaluate_fold(fold_idx, config, device, args):
    """Load fold model, run predictions on held-out test subjects, and report metrics."""
    checkpoint_dir = args.checkpoint_dir or config['train']['checkpoint_dir']
    processed_dir = args.processed_dir or config['data']['processed_dir']
    
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx}.pth")
    split_meta_path = os.path.join(checkpoint_dir, f"split_fold_{fold_idx}.yaml")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint for fold {fold_idx} not found at {checkpoint_path}. Train the model first.")
    if not os.path.exists(split_meta_path):
        raise FileNotFoundError(f"Split metadata for fold {fold_idx} not found at {split_meta_path}.")
        
    # Load splits
    with open(split_meta_path, "r") as f:
        splits = yaml.safe_load(f)
    test_subs = splits["test_subjects"]
    
    print(f"\n==================================================")
    print(f"             EVALUATING FOLD {fold_idx}")
    print(f"==================================================")
    print(f"Test Subjects: {test_subs}")
    
    seq_len = config['data']['sequence_length']

    # Load Model
    # weights_only=False: the checkpoint stores config/metrics alongside the weights,
    # and newer torch defaults this to True which would reject them.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SleepStagingModel(config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (saved Val F1: {checkpoint['val_f1']:.4f})")
    
    stream_stride = getattr(args, "stream_stride", None)
    if stream_stride is not None:
        print(f"Evaluation mode: streaming, recomputing every {stream_stride} s against a "
              f"{seq_len} s buffer")
        result = predict_streaming(model, processed_dir, test_subs, seq_len,
                                   stream_stride, device)
    else:
        # Non-overlapping windows: the archived evaluation. cover_tail is deliberately not
        # honoured here: its extra window overlaps the previous one, and every second in
        # the overlap would then be counted twice in accuracy, kappa and the confusion
        # matrix. It is a training-time option only. Use --stream_stride to score the
        # trailing seconds without double-counting anything.
        test_dataset = SleepDataset(
            processed_dir, test_subs, seq_len=seq_len, stride=seq_len,
            respect_boundaries=config['data'].get('respect_boundaries', False))
        result = predict_windowed(model, test_dataset, device)

    all_preds = result["y_pred"]
    all_targets = result["y_true"]
    all_logits = result["logits"]
    all_subjects = result["subject"]

    # Calculate performance metrics
    accuracy = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Also score at 30 s granularity by majority-voting each epoch's 30 per-second predictions.
    # Ground truth is constant inside an epoch (labels are replicated), so this is the number
    # that is directly comparable to published 30 s-epoch results.
    epoch30 = {}
    if seq_len % 30 == 0:
        p30 = np.asarray(all_preds).reshape(-1, 30)
        t30 = np.asarray(all_targets).reshape(-1, 30)
        preds_e = np.array([np.bincount(r, minlength=5).argmax() for r in p30])
        targets_e = t30[:, 0]
        epoch30 = {
            "accuracy": accuracy_score(targets_e, preds_e),
            "kappa": cohen_kappa_score(targets_e, preds_e),
            "macro_f1": f1_score(targets_e, preds_e, average='macro', zero_division=0),
            "n_epochs": len(targets_e),
        }

    print(f"\nHeld-Out Test Set Metrics:")
    print(f"  Overall Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Cohen's Kappa    : {kappa:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    
    if epoch30:
        print(f"\nAggregated to 30s epochs (comparable to published 30s results):")
        print(f"  Overall Accuracy : {epoch30['accuracy']:.4f} ({epoch30['accuracy']*100:.2f}%)")
        print(f"  Cohen's Kappa    : {epoch30['kappa']:.4f}")
        print(f"  Macro F1-Score   : {epoch30['macro_f1']:.4f}   over {epoch30['n_epochs']:,} epochs")

    # Print target benchmark comparison
    print(f"\nTarget Benchmark Comparison:")
    print(f"  Accuracy (Target >= 78%): {'PASSED' if accuracy >= 0.78 else 'BELOW TARGET'}")
    print(f"  Kappa (Target 0.60-0.70+): {'PASSED' if kappa >= 0.60 else 'BELOW TARGET'}")
    
    # Class report
    target_names = ["Wake (W)", "N1", "N2", "N3", "REM"]
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=target_names, 
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )
    print(f"\nClassification Report:")
    print(report)
    
    # Highlight N1 Bottleneck F1-Score
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    n1_f1 = per_class_f1[1]
    print(f"==================================================")
    print(f"---> Bottleneck Detection - N1 Staging F1-Score: {n1_f1:.4f}")
    if n1_f1 < 0.40:
        print("     ALERT: N1 performance is below normal thresholds. Adjust focal loss alpha/gamma weight.")
    else:
        print("     N1 performance is in acceptable/high range for second-by-second EEG models.")
    print(f"==================================================")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])
    print(f"\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print("      W    N1    N2    N3   REM")
    for name, row in zip(target_names, cm):
        print(f"{name:5s} " + " ".join(f"{val:5d}" for val in row))
    print("==================================================")

    # save metrics to disk. Streaming runs write beside the archived artifacts under their
    # own suffix so the two evaluation modes can be compared without overwriting anything.
    log_dir = args.log_dir or config['train'].get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    suffix = f"_streaming{stream_stride}" if stream_stride else ""
    report_path = os.path.join(log_dir, f"fold_{fold_idx}_test_report{suffix}.txt")
    with open(report_path, "w") as rf:
        rf.write(f"Fold {fold_idx} held-out test results\n")
        rf.write(f"Test subjects: {test_subs}\n")
        if stream_stride:
            rf.write(f"Evaluation: streaming, {stream_stride} s stride over a {seq_len} s buffer\n")
        rf.write(f"Overall Accuracy: {accuracy:.4f}\n")
        rf.write(f"Cohen's Kappa:    {kappa:.4f}\n")
        rf.write(f"Macro F1-Score:   {macro_f1:.4f}\n")
        if epoch30:
            rf.write(f"\nAggregated to 30s epochs ({epoch30['n_epochs']:,} epochs):\n")
            rf.write(f"Accuracy: {epoch30['accuracy']:.4f}  Kappa: {epoch30['kappa']:.4f}  "
                     f"Macro F1: {epoch30['macro_f1']:.4f}\n")
        rf.write("\n")
        rf.write("Per-class F1: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(target_names, per_class_f1)) + "\n\n")
        rf.write("Classification Report:\n" + report + "\n")
        rf.write("Confusion Matrix (Rows=True, Cols=Predicted):\n")
        rf.write("      W    N1    N2    N3   REM\n")
        for name, row in zip(target_names, cm):
            rf.write(f"{name:5s} " + " ".join(f"{val:5d}" for val in row) + "\n")

    # append headline metrics, one row per fold
    summary_path = os.path.join(log_dir, f"test_metrics_summary{suffix}.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a") as sf:
        if write_header:
            sf.write("fold,accuracy,kappa,macro_f1,f1_W,f1_N1,f1_N2,f1_N3,f1_REM,"
                     "acc_30s,kappa_30s,macro_f1_30s\n")
        sf.write(f"{fold_idx},{accuracy:.4f},{kappa:.4f},{macro_f1:.4f}," +
                 ",".join(f"{v:.4f}" for v in per_class_f1) +
                 (f",{epoch30['accuracy']:.4f},{epoch30['kappa']:.4f},{epoch30['macro_f1']:.4f}"
                  if epoch30 else ",,,") + "\n")
    print(f"Saved test report to {report_path} and appended headline metrics to {summary_path}")

    # Keep the raw predictions. Smoothing sweeps, per-subject confidence intervals,
    # calibration and prior correction all need them, and regenerating them from a
    # checkpoint costs a full evaluation pass every time they are wanted.
    if not getattr(args, "no_save_predictions", False):
        predictions_path = os.path.join(log_dir, f"fold_{fold_idx}_predictions{suffix}.npz")
        payload = dict(
            subject=np.asarray(all_subjects, dtype=str),
            y_true=np.asarray(all_targets, dtype=np.int8),
            y_pred=np.asarray(all_preds, dtype=np.int8),
            # float32, not float16: argmax over half precision disagrees with the saved
            # predictions on a few hundred seconds per fold, which would make any metric
            # recomputed from the logits differ from the report.
            logits=np.asarray(all_logits, dtype=np.float32),
            second_index=np.asarray(result["second_index"], dtype=np.int32),
            segment_start=np.asarray(result["segment_start"], dtype=bool),
            seq_len=np.array(seq_len),
            stream_stride=np.array(-1 if stream_stride is None else stream_stride),
            fold=np.array(fold_idx),
        )
        # The class balance the loss was trained against, so a later prior correction does
        # not have to guess it.
        train_counts = checkpoint.get('train_class_counts')
        if train_counts is not None:
            payload["train_class_counts"] = np.asarray(train_counts, dtype=np.int64)
        np.savez_compressed(predictions_path, **payload)
        print(f"Saved per-second predictions to {predictions_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sleep Staging Causal Network.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (0 to K-1).")
    parser.add_argument("--processed_dir", type=str, default=None, help="Override processed data directory.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")
    parser.add_argument("--log_dir", type=str, default=None, help="Override directory for metric outputs.")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU inference latency benchmark instead of evaluation.")
    parser.add_argument("--stream_stride", type=int, default=None,
                        help="Evaluate in streaming mode: recompute every N seconds against a "
                             "full context buffer and keep only the freshest N predictions, "
                             "instead of tiling the recording with independent windows. "
                             "Must be between 1 and the context length. "
                             "Results are written under a _streamingN suffix.")
    parser.add_argument("--no_save_predictions", action="store_true",
                        help="Skip writing fold_N_predictions.npz.")
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    if args.benchmark:
        run_latency_benchmark(SleepStagingModel, config)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        evaluate_fold(args.fold, config, device, args)

if __name__ == "__main__":
    main()
