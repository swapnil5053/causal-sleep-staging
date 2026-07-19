# Continuous Sleep Staging from Raw EEG using Causal Deep Learning

This repository implements a causal deep learning pipeline for continuous, second-by-second sleep stage classification from raw single-channel EEG signals (designed for Sleep-EDF Expanded Fpz-Cz channel at 100 Hz). The system integrates a Multi-Resolution CNN (MRCNN) for capture of high-frequency spindles/K-complexes and low-frequency slow waves, a dilated Causal Temporal Convolutional Network (TCN) for context modeling, a causal (masked) Multi-Head Self-Attention layer, and a stage classifier. The entire architecture is causal (left-padded/upper-masked) to ensure zero future information leakage, targeting a budget of ~48K parameters and inference latency of <= 3ms/sec.

For a comprehensive walkthrough of dataset setup, configuration tuning, and troubleshooting, please refer to [HANDOFF.md](HANDOFF.md).

---

## Codebase Structure

```
src/
  data/
    preprocessing.py  # EDF loader, epoch segmentation, Z-score normalization
    dataset.py        # PyTorch Dataset, splits, and oversampling sampler
  model/
    causal_conv.py    # Left-padded Causal Conv1D
    mrcnn.py          # Parallel multi-resolution receptive field convolution
    temporal.py       # Stack of residual causal dilated convolutions (TCN)
    attention.py      # Masked causal Multi-Head Self-Attention
    classifier.py     # Stage prediction logit projection
    full_model.py     # End-to-end network wiring and parameter budget check
  train/
    losses.py         # Focal Loss and Weighted Cross Entropy
    train.py          # Subject-wise cross-validation training loop
  eval/
    evaluate.py       # Metrics calculator & CPU latency benchmark
configs/
  default.yaml        # Global hyperparameters config
```

---

## Quick Start Commands

Detailed execution instructions are provided in `HANDOFF.md`. Below are the primary commands:

### 1. Preprocess Dataset
```bash
# Process a single subject
python -m src.data.preprocessing --subject 01

# Process all subjects found in data/raw/
python -m src.data.preprocessing --all
```

### 2. Train Model
```bash
# Run training on fold 0 of subject-wise cross-validation
python -m src.train.train --fold 0

# Run all 5 folds of subject-wise cross-validation sequentially
python -m src.train.train --fold -1
```

### 3. Evaluate Performance & Profile Latency
```bash
# Evaluate best checkpoint for fold 0 on held-out test subjects
python -m src.eval.evaluate --fold 0

# Profile CPU inference latency (ms per second of EEG)
python -m src.eval.evaluate --benchmark
```
