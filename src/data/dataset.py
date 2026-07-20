import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

class SleepDataset(Dataset):
    """
    PyTorch Dataset that loads processed subject .npz files and yields
    fixed-length sequence windows (context length L) for training sequence models.
    """
    def __init__(self, processed_dir, subject_ids, seq_len=30, stride=15):
        """
        Args:
            processed_dir (str): Directory containing preprocessed subject .npz files.
            subject_ids (list of str): List of subject IDs to load (e.g. ['00', '01']).
            seq_len (int): Sequence window length in seconds. Default: 30.
            stride (int): Stride for sliding window sequence extraction. Default: 15.
        """
        super(SleepDataset, self).__init__()
        
        self.windows = []
        self.labels = []
        
        # Load and segment data subject-by-subject
        for sub_id in subject_ids:
            file_path = os.path.join(processed_dir, f"subject_{sub_id}.npz")
            if not os.path.exists(file_path):
                print(f"Warning: Processed file for subject {sub_id} not found at {file_path}. Skipping.")
                continue
                
            data = np.load(file_path)
            x, y = data['x'], data['y'] # x: (N_seconds, 100), y: (N_seconds,)
            
            # Extract sliding windows
            num_seconds = len(x)
            if num_seconds < seq_len:
                print(f"Warning: Subject {sub_id} recording has fewer seconds ({num_seconds}) than seq_len ({seq_len}). Skipping.")
                continue
                
            for start in range(0, num_seconds - seq_len + 1, stride):
                end = start + seq_len
                self.windows.append(x[start:end])
                self.labels.append(y[start:end])
                
        # Convert lists to arrays or tensors
        if len(self.windows) > 0:
            self.windows = np.array(self.windows, dtype=np.float32) # (N_windows, L, 100)
            self.labels = np.array(self.labels, dtype=np.int64)     # (N_windows, L)
            print(f"Dataset initialized with {len(self.windows)} windows from {len(subject_ids)} subjects.")
        else:
            self.windows = np.empty((0, seq_len, 100), dtype=np.float32)
            self.labels = np.empty((0, seq_len), dtype=np.int64)
            print(f"Warning: Dataset is empty.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Returns:
            x: float tensor of shape (L, 100)
            y: long tensor of shape (L,)
        """
        x = torch.from_numpy(self.windows[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


def get_all_subject_ids(processed_dir):
    """Scan the processed directory and extract all available subject IDs."""
    files = glob.glob(os.path.join(processed_dir, "subject_*.npz"))
    subject_ids = []
    for f in files:
        basename = os.path.basename(f)
        # Match subject_<id>.npz
        match = re.search(r'subject_([A-Za-z0-9_]+)\.npz', basename)
        if match:
            subject_ids.append(match.group(1))
    return sorted(list(set(subject_ids)))

import re # needed in get_all_subject_ids


def get_subject_splits(subject_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split subject IDs into train/val/test sets to prevent subject leak.
    
    Args:
        subject_ids (list of str): Full list of subject IDs.
        train_ratio, val_ratio, test_ratio (float): Ratios for splits.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_subs, val_subs, test_subs)
    """
    import random
    random.seed(seed)
    shuffled = sorted(list(subject_ids))
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_subs = shuffled[:n_train]
    val_subs = shuffled[n_train:n_train + n_val]
    test_subs = shuffled[n_train + n_val:]
    
    return train_subs, val_subs, test_subs


def get_cv_splits(subject_ids, num_folds=5, fold_idx=0, seed=42):
    """
    Generate subject-wise cross-validation splits for a specific fold index.
    Matches standard leave-some-subjects-out evaluations.
    
    Args:
        subject_ids (list of str): Full list of subject IDs.
        num_folds (int): Total number of folds.
        fold_idx (int): Current fold index (0 to num_folds - 1).
        seed (int): Random seed.
        
    Returns:
        tuple: (train_subs, val_subs, test_subs)
    """
    import random
    random.seed(seed)
    shuffled = sorted(list(subject_ids))
    random.shuffle(shuffled)
    
    # Split into K folds
    # np.array_split returns numpy strings; cast back to plain str so the split
    # metadata stays yaml-serializable.
    folds = np.array_split(shuffled, num_folds)
    folds = [[str(s) for s in f] for f in folds]
    
    # Fold at fold_idx is the test set
    test_subs = folds[fold_idx]
    
    # Next fold is the validation set
    val_idx = (fold_idx + 1) % num_folds
    val_subs = folds[val_idx]
    
    # Remaining folds form the training set
    train_subs = []
    for idx, f in enumerate(folds):
        if idx != fold_idx and idx != val_idx:
            train_subs.extend(f)
            
    return train_subs, val_subs, test_subs


def get_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler that oversamples minority stages N1 and N3.
    Re-balances sequences based on class frequency inside each window.
    
    Args:
        dataset (SleepDataset): Training dataset.
        
    Returns:
        WeightedRandomSampler: Sampler for DataLoader.
    """
    # Calculate global frequency of each class in this dataset
    labels = dataset.labels # shape: (N_windows, L)
    if len(labels) == 0:
        return None
        
    # Flatten to calculate class occurrences
    flat_labels = labels.flatten()
    class_counts = np.bincount(flat_labels, minlength=5)
    
    # Compute inverse class frequencies
    class_weights = np.zeros(5)
    for c in range(5):
        if class_counts[c] > 0:
            class_weights[c] = len(flat_labels) / (5.0 * class_counts[c])
        else:
            class_weights[c] = 0.0
            
    # Compute weights for each window sequence as the average label weight in that window
    window_weights = []
    for window_lbls in labels:
        w = np.mean([class_weights[int(lbl)] for lbl in window_lbls])
        window_weights.append(w)
        
    # Convert to tensor and return sampler
    sampler_weights = torch.DoubleTensor(window_weights)
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True
    )
    
    return sampler
