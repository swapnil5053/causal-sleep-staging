import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SleepStagingModel(config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (saved Val F1: {checkpoint['val_f1']:.4f})")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
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
    
    print(f"\nHeld-Out Test Set Metrics:")
    print(f"  Overall Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Cohen's Kappa    : {kappa:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sleep Staging Causal Network.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (0 to K-1).")
    parser.add_argument("--processed_dir", type=str, default=None, help="Override processed data directory.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")
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
