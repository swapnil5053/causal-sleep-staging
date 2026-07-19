# Project Handoff: Continuous Sleep Staging from Raw EEG

Hey! Welcome to the repository. The codebase for our capstone project, **"Continuous Sleep Staging from Raw EEG using Causal Deep Learning"** (PES University, PW25_BJD_21), has been fully written and structured.

---

## 1. Project Status & Stage
All core code, models, preprocessing pipelines, custom loss functions, training harnesses, and evaluation suites are **fully written, verified for import/syntax, and ready to go**. 

Nothing has been trained or run yet, as this machine lacks the dataset and GPU resources. The pipeline is structured so you can download the data, drop it in the folder, and run the training/evaluation scripts directly.

---

## 2. Dataset Setup Instructions
You need to download the **Sleep-EDF Database Expanded v1.0.0** from PhysioNet. 

* **Source URL:** [PhysioNet Sleep-EDF Expanded v1.0.0](https://physionet.org/content/sleep-edfv1/1.0.0/)
* **Subset:** We are evaluating on the **Sleep Cassette** subset (which represents the standard Sleep-EDF-20 protocol).
* **Required Files:** You will need pairs of files for each subject:
  * `*PSG.edf` (contains the raw polysomnography EEG signal).
  * `*Hypnogram.edf` (contains the AASM hypnogram annotations, labeled every 30 seconds).
  
### Data Directory Structure
Place the downloaded `.edf` files directly inside `data/raw/`. The directory structure should look exactly like this:
```
data/
  raw/
    SC4001E0-PSG.edf
    SC4001EC-Hypnogram.edf
    SC4002E0-PSG.edf
    SC4002EC-Hypnogram.edf
    ...
  processed/
```

---

## 3. Step-by-Step Execution Guide

Run these commands in order from the repository root directory.

### Step A: Environment Setup
Set up a virtual environment and install the required physiological and deep learning libraries:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step B: Run Data Preprocessing
The preprocessing script parses the raw EDFs, extracts the `EEG Fpz-Cz` channel, resamples it to 100 Hz, performs Z-score normalization per 30-second epoch, maps sleep stages, and saves the output in 1-second segments.

```bash
# OPTION 1: Process a single subject (e.g. subject 00)
python -m src.data.preprocessing --subject 00

# OPTION 2: Process ALL subjects found in data/raw/
python -m src.data.preprocessing --all
```

### Step C: Run Subject-Wise Cross-Validation Training
We train the model using a subject-wise cross-validation protocol to prevent data leakage. You can train a specific fold or run all of them sequentially:

```bash
# OPTION 1: Train a single fold (e.g., Fold 0) - Recommended for parallel GPU training
python -m src.train.train --fold 0

# OPTION 2: Train all 5 folds sequentially
python -m src.train.train --fold -1
```

### Step D: Run Evaluation & Benchmark Latency
Once training completes, evaluate the models on the held-out subject test set, and verify inference latency on CPU:

```bash
# Run metrics evaluation for Fold 0
python -m src.eval.evaluate --fold 0

# Profile CPU latency to verify <= 3ms/sec inference target
python -m src.eval.evaluate --benchmark
```

---

## 4. Expected Outputs and File Locations
Here is where the generated outputs will land:
* **Processed Data:** `data/processed/subject_{id}.npz` files containing arrays `x` (shape `[N_seconds, 100]`) and `y` (shape `[N_seconds,]`).
* **Checkpoints:** `checkpoints/best_model_fold_{fold}.pth` (saves the model weights, optimizer state, and training configs when validation Macro F1 peaks).
* **Fold Metadata:** `checkpoints/split_fold_{fold}.yaml` (saves the subject split assignments for train/val/test to ensure the evaluation script tests the correct subject set).
* **Metrics logs:** `logs/fold_{fold}_metrics.csv` containing train/validation losses and accuracies mapped per epoch.

---

## 5. Hyperparameter Tuning Guide

If results deviate from our target benchmarks, adjust parameters inside `configs/default.yaml` directly:

### Target Benchmarks:
* **Overall Accuracy:** $\ge 78\%$
* **Cohen's Kappa:** $0.60 \text{ to } 0.70+$
* **Model Size:** $\approx 48\text{K}$ parameters
* **Inference Latency:** $\le 3\text{ ms per second of EEG}$ on CPU

### Tuning Scenarios:
1. **If Cohen's Kappa is < 0.60 or Accuracy is low:**
   * Try increasing the sequence length: Set `data.sequence_length` to `60` or `100` to give the causal self-attention layer a larger temporal context.
   * Fine-tune learning rate `train.lr` (try `5e-4` or `1e-3`).
   * Increase TCN channels from `[32, 32, 32]` to `[48, 48, 48]` (keep an eye on the parameter budget!).
2. **If N1 F1-Score is too low (bottleneck stage):**
   * Change `train.loss_type` to `"focal"` to force the network to focus on hard samples.
   * Increase `train.focal_gamma` to `2.5` or `3.0`.
   * Ensure `train.use_weighted_sampler` is set to `true` to oversample windows containing N1 stages.
3. **If Parameter Count is over budget (48K):**
   * Tweak model size in `configs/default.yaml`.
   * Reduce MRCNN channels: Set `model.mrcnn_channels_1` and `model.mrcnn_channels_2` to `12` or `8`.
   * Reduce TCN depth/channels: Set `model.tcn_channels` to `[24, 24, 24]`.
4. **If CPU Inference Latency is > 3 ms/sec:**
   * Reduce the attention heads: Set `model.attn_num_heads` to `2` or `1` (less projection projection overhead).
   * Shrink `model.tcn_channels` to lower the dimensionality of features passed to the attention layer.

---

## 6. Troubleshooting "If Something Breaks"

* **"Could not find EEG channel matching target..."**
  * **Cause:** EDF files from other versions of Sleep-EDF or different databases name channels differently (e.g. `Fpz-Cz` or `EEG Fpz-Cz`).
  * **Fix:** Check raw channel names printed in terminal, then update `data.target_channel` in `configs/default.yaml` to match.
* **"Mismatched sample rates"**
  * **Cause:** If raw signal sample rate is not 100 Hz, resampling might fail if the file is corrupted.
  * **Fix:** The preprocessing script automatically runs `raw.resample(100)`. If it errors, check if your MNE library is up to date (`pip install -U mne`).
* **"CUDA not available"**
  * **Cause:** CUDA drivers are missing or PyTorch was installed without CUDA support.
  * **Fix:** The scripts are designed to automatically fall back to CPU. However, training will be slow. To run on GPU, install the CUDA version of PyTorch from the [official PyTorch website](https://pytorch.org/).
* **"Subject files not found"**
  * **Cause:** The raw files do not match standard naming conventions.
  * **Fix:** Verify raw filenames end with `PSG.edf` and `Hypnogram.edf` and that they match. For example, if you have `SC4001E0-PSG.edf`, you must have `SC4001EC-Hypnogram.edf` (with the digit `0` in PSG mapped to `C` in Hypnogram).
