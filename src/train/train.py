import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.data.dataset import SleepDataset, get_all_subject_ids, get_cv_splits, get_weighted_sampler
from src.model.full_model import SleepStagingModel
from src.train.losses import FocalLoss, WeightedCrossEntropyLoss

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config(config_path):
    """Load config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_idx, (x, y) in enumerate(dataloader):
        # x: (batch_size, L, 100), y: (batch_size, L)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: outputs shape (batch_size, L, 5)
        logits = model(x)
        
        # Compute loss
        # Loss functions flatten inputs inside their forward methods
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1) # (batch_size, L)
        
        all_preds.extend(preds.detach().cpu().numpy().flatten())
        all_targets.extend(y.detach().cpu().numpy().flatten())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            running_loss += loss.item() * x.size(0)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_kappa = cohen_kappa_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Calculate per-class F1-scores to monitor N1 bottleneck specifically
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    return val_loss, val_acc, val_kappa, val_f1, per_class_f1

def run_fold(fold_idx, config, device, args):
    """Run training and validation for a single fold."""
    print(f"\n==================================================")
    print(f"               STARTING FOLD {fold_idx}")
    print(f"==================================================")

    # Re-seed per fold so that `--fold 3` on its own reproduces fold 3 of a `--fold -1`
    # sweep. Without this, a fold's initialisation depends on how many folds ran before it.
    set_seed(config['train']['seed'] + fold_idx)

    processed_dir = args.processed_dir or config['data']['processed_dir']
    checkpoint_dir = args.checkpoint_dir or config['train']['checkpoint_dir']
    log_dir = args.log_dir or config['train'].get('log_dir', 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get subject splits for this fold
    subject_ids = get_all_subject_ids(processed_dir)
    if not subject_ids:
        raise ValueError(f"No processed subject files found in {processed_dir}. Make sure you run preprocessing first!")
        
    train_subs, val_subs, test_subs = get_cv_splits(
        subject_ids, 
        num_folds=config['train']['num_folds'], 
        fold_idx=fold_idx, 
        seed=config['train']['seed']
    )
    
    print(f"Train subjects: {train_subs}")
    print(f"Val subjects: {val_subs}")
    print(f"Test subjects (held-out for evaluate.py): {test_subs}")
    
    # Save test subject split metadata for the evaluate script
    split_meta_path = os.path.join(checkpoint_dir, f"split_fold_{fold_idx}.yaml")
    with open(split_meta_path, "w") as sf:
        yaml.safe_dump({
            "train_subjects": train_subs,
            "val_subjects": val_subs,
            "test_subjects": test_subs
        }, sf)
    
    # Create datasets
    seq_len = config['data']['sequence_length']
    stride = config['data']['sequence_stride']
    
    train_dataset = SleepDataset(processed_dir, train_subs, seq_len=seq_len, stride=stride)
    val_dataset = SleepDataset(processed_dir, val_subs, seq_len=seq_len, stride=stride)
    
    # Setup dataloaders
    batch_size = args.batch_size or config['train']['batch_size']
    
    # Optional weighted sampler for oversampling N1/N3
    sampler = None
    if config['train']['use_weighted_sampler']:
        print("Using weighted sampler to handle class imbalance (oversampling N1/N3).")
        sampler = get_weighted_sampler(train_dataset)
        
    # pinning host memory only helps (and is only supported) when copying to a GPU
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0, # set to 0 to prevent windows multiprocessing issues
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    # Initialize Model
    model = SleepStagingModel(config=config)
    model = model.to(device)
    
    # Setup loss criterion
    loss_type = args.loss_type or config['train']['loss_type']
    if loss_type == "focal":
        # Calculate standard class counts inside dataset to formulate optional alpha weights
        flat_labels = train_dataset.labels.flatten()
        counts = np.bincount(flat_labels, minlength=5)
        # Handle zero frequency cases
        counts = np.maximum(counts, 1)
        alpha = len(flat_labels) / (5.0 * counts)
        # Normalize weights
        alpha = alpha / np.sum(alpha)
        
        gamma = args.gamma or config['train']['focal_gamma']
        criterion = FocalLoss(gamma=gamma, alpha=alpha)
        print(f"Using Focal Loss (gamma={gamma}, alpha={alpha.round(3)})")
    else:
        # Inverse class frequency weighted CE
        flat_labels = train_dataset.labels.flatten()
        counts = np.bincount(flat_labels, minlength=5)
        counts = np.maximum(counts, 1)
        weights = len(flat_labels) / (5.0 * counts)
        criterion = WeightedCrossEntropyLoss(weights=weights)
        print(f"Using Weighted Cross Entropy Loss (weights={weights.round(3)})")

    # the loss holds class weights as a buffer, so it has to sit on the same device as the batch
    criterion = criterion.to(device)

    # Optimizer and Scheduler
    lr = args.lr or config['train']['lr']
    weight_decay = config['train']['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = args.epochs or config['train']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Checkpoint on the metric we actually report. Validation macro-F1 and kappa peak at
    # different epochs, so selecting on kappa keeps the saved model consistent with the
    # headline number.
    metric_name = config['train'].get('checkpoint_metric', 'kappa')
    patience = config['train'].get('early_stopping_patience', 0)

    best_score = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_kappa, val_f1, per_class_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log stats
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_kappa": val_kappa,
            "val_macro_f1": val_f1,
            "val_f1_W": per_class_f1[0],
            "val_f1_N1": per_class_f1[1],
            "val_f1_N2": per_class_f1[2],
            "val_f1_N3": per_class_f1[3],
            "val_f1_REM": per_class_f1[4]
        })
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.3f} - Kappa: {val_kappa:.3f} - MacroF1: {val_f1:.3f} - N1F1: {per_class_f1[1]:.3f}")
              
        score = val_kappa if metric_name == "kappa" else val_f1
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_since_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_kappa': val_kappa,
                'selection_metric': metric_name,
                'config': config
            }, checkpoint_path)
            print(f"----> Saved best model checkpoint to {checkpoint_path}")
        else:
            epochs_since_improve += 1
            if patience and epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch}: no improvement in val {metric_name} "
                      f"for {patience} epochs (best {best_score:.4f} at epoch {best_epoch}).")
                break

    # Save metric history to CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(log_dir, f"fold_{fold_idx}_metrics.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Fold {fold_idx} finished. CSV history saved to {history_path}")
    print(f"Best validation {metric_name}: {best_score:.4f} (epoch {best_epoch})")

    return best_score

def main():
    parser = argparse.ArgumentParser(description="Train Sleep Staging Causal Network.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to run (0 to K-1). Run -1 to run all folds sequentially.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--loss_type", type=str, default=None, help="Override loss type (focal/weighted_ce).")
    parser.add_argument("--gamma", type=float, default=None, help="Override focal loss gamma.")
    parser.add_argument("--processed_dir", type=str, default=None, help="Override processed data directory.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")
    parser.add_argument("--log_dir", type=str, default=None, help="Override directory for metric CSVs.")
    args = parser.parse_args()
    
    config = get_config(args.config)
    set_seed(config['train']['seed'])
    
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_folds = config['train']['num_folds']
    
    if args.fold == -1:
        print(f"Running all {num_folds} cross-validation folds sequentially...")
        fold_scores = []
        for f in range(num_folds):
            score = run_fold(f, config, device, args)
            fold_scores.append(score)
        print(f"\n==================================================")
        metric_name = config['train'].get('checkpoint_metric', 'kappa')
        print(f"Cross-Validation Summary:")
        for idx, score in enumerate(fold_scores):
            print(f"  Fold {idx}: best validation {metric_name} = {score:.4f}")
        print(f"Average validation {metric_name}: {np.mean(fold_scores):.4f}")
        print(f"==================================================")
    else:
        if args.fold < 0 or args.fold >= num_folds:
            raise ValueError(f"Fold index {args.fold} is out of bounds for {num_folds}-fold cross validation.")
        run_fold(args.fold, config, device, args)

if __name__ == "__main__":
    main()
