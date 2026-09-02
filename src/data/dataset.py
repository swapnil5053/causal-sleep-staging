import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

def resolve_split_seed(config):
    """Return the seed that decides which subjects land in which fold.

    The training seed used to double as the split seed, so changing it moved the fold
    boundaries as well as the weight initialisation and the two effects could not be
    told apart afterwards. ``data.split_seed`` pins the partition independently: hold it
    fixed across runs to vary only the initialisation, or vary it to sample partitions.

    When ``data.split_seed`` is absent the training seed is used, so every configuration
    written before this option existed still produces exactly the splits it always did.
    """
    data_cfg = (config or {}).get('data') or {}
    train_cfg = (config or {}).get('train') or {}
    if 'split_seed' in data_cfg:
        seed = data_cfg['split_seed']
        source = "data.split_seed"
    else:
        seed = train_cfg.get('seed', data_cfg.get('seed', 42))
        source = "train.seed"

    # A seed that is None or a string would silently reshuffle the folds: random.seed(None)
    # draws a fresh partition every run, and random.seed("42") is a different partition
    # from random.seed(42). Both would be recorded in the split file as though they were 42.
    try:
        return int(seed)
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} must be an integer, got {seed!r}. A blank or quoted value in the "
            f"config would change the fold layout without changing what is recorded."
        )


def segment_starts(npz, num_seconds):
    """Second indices at which a fresh, discontinuous stretch of signal begins.

    Preprocessing drops unscored epochs and concatenates a subject's two nights, so the
    stored array is not one continuous recording. Files written before this metadata
    existed report a single segment, which is what the loader assumed implicitly.
    """
    if 'segment_starts' not in npz.files:
        return None
    starts = np.asarray(npz['segment_starts'], dtype=np.int64).ravel()
    starts = starts[(starts >= 0) & (starts < num_seconds)]
    return np.unique(np.concatenate(([0], starts)))


def describe_normalization(npz):
    """Return a short identifier for how a processed .npz was normalized.

    Files written before the streaming pipeline existed carry no metadata; they are
    reported as ``epoch_zscore (legacy, no metadata)`` so a mixed directory is still
    detected rather than being read as a match.
    """
    if 'normalization_method' not in npz.files:
        return "epoch_zscore (legacy, no metadata)"
    method = str(npz['normalization_method'])
    if method == "causal_rolling" and 'normalization_window_seconds' in npz.files:
        return f"causal_rolling@{float(npz['normalization_window_seconds']):g}s"
    return method


class SleepDataset(Dataset):
    """
    PyTorch Dataset that loads processed subject .npz files and yields
    fixed-length sequence windows (context length L) for training sequence models.
    """
    def __init__(self, processed_dir, subject_ids, seq_len=30, stride=15,
                 respect_boundaries=False, cover_tail=False):
        """
        Args:
            processed_dir (str): Directory containing preprocessed subject .npz files.
            subject_ids (list of str): List of subject IDs to load (e.g. ['00', '01']).
            seq_len (int): Sequence window length in seconds. Default: 30.
            stride (int): Stride for sliding window sequence extraction. Default: 15.
            respect_boundaries (bool): Drop windows that span a night join or a gap left
                by an unscored epoch, so no window mixes two discontinuous stretches of
                signal. Requires ``segment_starts`` metadata from preprocessing.
            cover_tail (bool): Append a final window ending at the last second when the
                stride does not divide the recording evenly. Without it the trailing
                ``(len - seq_len) % stride`` seconds are never seen.

        Both flags default to False, which reproduces the windowing used for every
        archived run.
        """
        super(SleepDataset, self).__init__()

        self.windows = []
        self.labels = []
        self.window_subjects = []
        self.window_starts = []
        self.normalization_method = None
        self.respect_boundaries = respect_boundaries
        self.cover_tail = cover_tail
        self.windows_dropped_at_boundaries = 0
        self.segment_starts_by_subject = {}

        # Load and segment data subject-by-subject
        for sub_id in subject_ids:
            file_path = os.path.join(processed_dir, f"subject_{sub_id}.npz")
            if not os.path.exists(file_path):
                print(f"Warning: Processed file for subject {sub_id} not found at {file_path}. Skipping.")
                continue

            data = np.load(file_path)
            x, y = data['x'], data['y'] # x: (N_seconds, 100), y: (N_seconds,)

            # Refuse to silently mix preprocessing regimes inside one directory: a run that
            # blends causal and epoch-normalized subjects would be unreportable, and the
            # failure is otherwise invisible. Files written before normalization metadata
            # existed are treated as the legacy epoch z-score.
            method = describe_normalization(data)
            if self.normalization_method is None:
                self.normalization_method = method
            elif method != self.normalization_method:
                raise ValueError(
                    f"Inconsistent preprocessing in {processed_dir}: subject {sub_id} was "
                    f"normalized with '{method}' but earlier subjects used "
                    f"'{self.normalization_method}'. Re-run preprocessing into a clean "
                    f"directory before training."
                )

            # Extract sliding windows
            num_seconds = len(x)
            if num_seconds < seq_len:
                print(f"Warning: Subject {sub_id} recording has fewer seconds ({num_seconds}) than seq_len ({seq_len}). Skipping.")
                continue
                
            boundaries = None
            if respect_boundaries:
                boundaries = segment_starts(data, num_seconds)
                if boundaries is None:
                    raise ValueError(
                        f"respect_boundaries=True needs segment metadata, but "
                        f"{os.path.basename(file_path)} was written before preprocessing "
                        f"recorded it. Re-run preprocessing into a clean directory, or "
                        f"leave respect_boundaries off to keep the archived windowing."
                    )

            starts = list(range(0, num_seconds - seq_len + 1, stride))
            if cover_tail and starts and starts[-1] + seq_len < num_seconds:
                starts.append(num_seconds - seq_len)

            for start in starts:
                end = start + seq_len
                if boundaries is not None:
                    # A window is clean when no segment begins strictly inside it.
                    crossings = boundaries[(boundaries > start) & (boundaries < end)]
                    if len(crossings) > 0:
                        self.windows_dropped_at_boundaries += 1
                        continue
                self.windows.append(x[start:end])
                self.labels.append(y[start:end])
                self.window_subjects.append(sub_id)
                self.window_starts.append(start)
                if boundaries is not None:
                    self.segment_starts_by_subject[sub_id] = boundaries
                elif sub_id not in self.segment_starts_by_subject:
                    stored = segment_starts(data, num_seconds)
                    if stored is not None:
                        self.segment_starts_by_subject[sub_id] = stored

        # Convert lists to arrays or tensors
        self.window_subjects = np.array(self.window_subjects, dtype=object)
        self.window_starts = np.array(self.window_starts, dtype=np.int64)
        if len(self.windows) > 0:
            self.windows = np.array(self.windows, dtype=np.float32) # (N_windows, L, 100)
            self.labels = np.array(self.labels, dtype=np.int64)     # (N_windows, L)
            print(f"Dataset initialized with {len(self.windows)} windows from {len(subject_ids)} "
                  f"subjects (normalization: {self.normalization_method}).")
            if self.windows_dropped_at_boundaries:
                print(f"  dropped {self.windows_dropped_at_boundaries} window(s) spanning a "
                      f"night join or an unscored-epoch gap.")
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
