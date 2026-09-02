import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix

from src.data.dataset import SleepDataset
from src.model.full_model import SleepStagingModel

def get_config(config_path):
    """Load config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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

def per_subject_metrics(subjects, targets, preds):
    """Score every test subject separately.

    A fold pools ~16 nights into one kappa, which is why the causality comparison only ever
    had five paired measurements per seed. Scoring each subject turns one fold into as many
    paired differences as it holds test subjects, and unlike folds those really are
    independent: a subject appears in the test set of exactly one fold.

    Returns one dict per subject, in the order the subjects appear in the test set.
    """
    subjects = np.asarray(subjects)
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    rows = []
    for subject in dict.fromkeys(subjects.tolist()):     # preserves order, drops repeats
        mask = subjects == subject
        y_true, y_pred = targets[mask], preds[mask]
        # A night the technician scored as a single stage has no kappa to report. It is left
        # empty rather than written as zero, which would quietly drag any paired mean.
        kappa = float("nan") if len(set(y_true.tolist())) < 2 else cohen_kappa_score(y_true, y_pred)
        rows.append({
            "subject": subject,
            "n_seconds": int(mask.sum()),
            "accuracy": accuracy_score(y_true, y_pred),
            "kappa": kappa,
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "per_class_f1": f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4],
                                     zero_division=0),
        })
    return rows


SUBJECT_CSV_HEADER = "fold,subject,n_seconds,accuracy,kappa,macro_f1,f1_W,f1_N1,f1_N2,f1_N3,f1_REM\n"


def format_subject_row(fold_idx, row):
    """One CSV line. An unscoreable kappa is written as an empty field, not a zero."""
    kappa = "" if np.isnan(row["kappa"]) else f"{row['kappa']:.4f}"
    return (f"{fold_idx},{row['subject']},{row['n_seconds']},{row['accuracy']:.4f},"
            f"{kappa},{row['macro_f1']:.4f},"
            + ",".join(f"{v:.4f}" for v in row["per_class_f1"]) + "\n")


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
    
    # Load Dataset for testing (stride matches seq_len to evaluate sequential non-overlapping staging)
    seq_len = config['data']['sequence_length']
    test_dataset = SleepDataset(processed_dir, test_subs, seq_len=seq_len, stride=seq_len)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # process subject streams sequence-by-sequence
        shuffle=False,
        num_workers=0
    )
    
    # Load Model
    # weights_only=False: the checkpoint stores config/metrics alongside the weights,
    # and newer torch defaults this to True which would reject them.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SleepStagingModel(config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (saved Val F1: {checkpoint['val_f1']:.4f})")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # batch_size is 1 and shuffle is off, so predictions come back in dataset order
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            
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

    # save metrics to disk
    log_dir = args.log_dir or config['train'].get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, f"fold_{fold_idx}_test_report.txt")
    with open(report_path, "w") as rf:
        rf.write(f"Fold {fold_idx} held-out test results\n")
        rf.write(f"Test subjects: {test_subs}\n")
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
    summary_path = os.path.join(log_dir, "test_metrics_summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a") as sf:
        if write_header:
            sf.write("fold,accuracy,kappa,macro_f1,f1_W,f1_N1,f1_N2,f1_N3,f1_REM,"
                     "acc_30s,kappa_30s,macro_f1_30s\n")
        sf.write(f"{fold_idx},{accuracy:.4f},{kappa:.4f},{macro_f1:.4f}," +
                 ",".join(f"{v:.4f}" for v in per_class_f1) +
                 (f",{epoch30['accuracy']:.4f},{epoch30['kappa']:.4f},{epoch30['macro_f1']:.4f}"
                  if epoch30 else ",,,") + "\n")
    # Per-subject breakdown. The dataset records which subject each window came from, and
    # evaluation uses stride == seq_len, so every window contributes exactly seq_len
    # consecutive predictions in order.
    subjects_per_second = np.repeat(test_dataset.window_subjects, seq_len)
    if len(subjects_per_second) != len(all_preds):
        raise RuntimeError(
            f"Fold {fold_idx}: {len(subjects_per_second)} subject labels for "
            f"{len(all_preds)} predictions. Per-subject attribution would be wrong, so it "
            f"is not written.")

    subject_rows = per_subject_metrics(subjects_per_second, all_targets, all_preds)
    subject_path = os.path.join(log_dir, f"fold_{fold_idx}_subject_metrics.csv")
    with open(subject_path, "w") as pf:
        pf.write(SUBJECT_CSV_HEADER)
        for row in subject_rows:
            pf.write(format_subject_row(fold_idx, row))

    # A run-level copy as well, so a whole sweep is one read for the subject-level paired test.
    run_subject_path = os.path.join(log_dir, "test_subject_metrics.csv")
    write_subject_header = not os.path.exists(run_subject_path)
    with open(run_subject_path, "a") as pf:
        if write_subject_header:
            pf.write(SUBJECT_CSV_HEADER)
        for row in subject_rows:
            pf.write(format_subject_row(fold_idx, row))

    print(f"\nPer-subject results ({len(subject_rows)} held-out subjects):")
    for row in subject_rows:
        kappa = "  n/a" if np.isnan(row["kappa"]) else f"{row['kappa']:.4f}"
        print(f"  subject {row['subject']:>4}: kappa {kappa}  acc {row['accuracy']:.4f}  "
              f"macro-F1 {row['macro_f1']:.4f}  ({row['n_seconds']:,} s)")

    print(f"\nSaved test report to {report_path}, per-subject metrics to {subject_path}, "
          f"and appended headline metrics to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sleep Staging Causal Network.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (0 to K-1).")
    parser.add_argument("--processed_dir", type=str, default=None, help="Override processed data directory.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")
    parser.add_argument("--log_dir", type=str, default=None, help="Override directory for metric outputs.")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU inference latency benchmark instead of evaluation.")
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
