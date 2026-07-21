import os
import re
import glob
import argparse
import yaml
import numpy as np
import mne

def get_config(config_path="configs/default.yaml"):
    """Load config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def find_eeg_channel(channel_names, target="EEG Fpz-Cz"):
    """
    Search for a matching channel name among the list of channel names in the EDF.
    Handles slight naming variations (e.g. whitespace, case sensitivity).
    """
    # Try exact match first
    if target in channel_names:
        return target
        
    # Case-insensitive check
    for ch in channel_names:
        if ch.strip().lower() == target.strip().lower():
            return ch
            
    # Substring check
    clean_target = target.replace(" ", "").lower()
    for ch in channel_names:
        clean_ch = ch.replace(" ", "").lower()
        if clean_target in clean_ch or clean_ch in clean_target:
            return ch
            
    raise ValueError(f"Could not find EEG channel matching target '{target}' in channels: {channel_names}")

def process_subject(psg_path, hypno_path, target_channel="EEG Fpz-Cz", resample_rate=100,
                    wake_trim_minutes=30):
    """
    Load raw PSG and Hypnogram, extract target channel, resample,
    perform Z-score normalization per 30-second epoch, and segment into 1-second windows.

    Args:
        psg_path (str): Path to PSG.edf file.
        hypno_path (str): Path to Hypnogram.edf file.
        target_channel (str): Name of the channel to extract.
        resample_rate (float): Target sampling rate (Hz).
        wake_trim_minutes (float): How many minutes of Wake to keep either side of the
            sleep period. Sleep-EDF cassette records ~20h per subject, so most of the
            recording is the subject awake and out of bed; keeping all of it makes Wake
            ~68% of the data and inflates accuracy. Standard protocol (DeepSleepNet,
            AttnSleep, TinySleepNet) keeps 30 min either side. Set to None to keep everything.

    Returns:
        tuple: (signal_windows, labels)
            signal_windows: np.ndarray of shape (N_seconds, 100)
            labels: np.ndarray of shape (N_seconds,)
    """
    print(f"Loading PSG: {os.path.basename(psg_path)}")
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    print(f"Loading Hypnogram: {os.path.basename(hypno_path)}")
    annotations = mne.read_annotations(hypno_path)
    raw.set_annotations(annotations, emit_warning=False)
    
    # Identify target channel
    matching_channel = find_eeg_channel(raw.ch_names, target_channel)
    print(f"Selected channel: {matching_channel}")
    raw.pick_channels([matching_channel])
    
    # Resample if frequency is different
    sfreq = raw.info['sfreq']
    if sfreq != resample_rate:
        print(f"Resampling from {sfreq} Hz to {resample_rate} Hz...")
        raw.resample(resample_rate, verbose=False)
        
    # Get raw continuous data
    # raw_data shape: (1, n_samples)
    raw_data = raw.get_data()
    n_samples = raw_data.shape[1]
    total_duration = n_samples / resample_rate
    
    print(f"Total duration: {total_duration:.1f} seconds ({n_samples} samples)")
    
    # Map sleep stages per second
    second_labels = np.full(int(total_duration), -1, dtype=int)
    for annot in annotations:
        onset = int(annot['onset'])
        duration = int(annot['duration'])
        desc = annot['description']
        
        # Mapping to Sleep Stages: W=0, N1=1, N2=2, N3=3 (merge N3 and N4), REM=4
        label = -1
        if "Sleep stage W" in desc:
            label = 0
        elif "Sleep stage 1" in desc:
            label = 1
        elif "Sleep stage 2" in desc:
            label = 2
        elif "Sleep stage 3" in desc or "Sleep stage 4" in desc:
            label = 3
        elif "Sleep stage R" in desc:
            label = 4
            
        if label != -1:
            start_sec = max(0, onset)
            end_sec = min(len(second_labels), onset + duration)
            second_labels[start_sec:end_sec] = label
            
    # Process in 30-second blocks
    num_30s_blocks = int(total_duration // 30)
    samples_per_30s = 30 * resample_rate # 3000
    samples_per_1s = resample_rate # 100
    
    processed_signals = []
    processed_labels = []

    skipped_blocks = 0

    # Label of each 30s epoch (-1 = unscored/movement)
    epoch_labels_all = np.array([second_labels[i * 30] for i in range(num_30s_blocks)])

    # Trim the long awake stretches either side of the sleep period.
    first_block, last_block = 0, num_30s_blocks
    if wake_trim_minutes is not None:
        sleep_blocks = np.where((epoch_labels_all != -1) & (epoch_labels_all != 0))[0]
        if len(sleep_blocks) > 0:
            margin = int(round(wake_trim_minutes * 2))  # 30s epochs -> 2 per minute
            first_block = max(0, sleep_blocks[0] - margin)
            last_block = min(num_30s_blocks, sleep_blocks[-1] + margin + 1)
            trimmed = num_30s_blocks - (last_block - first_block)
            print(f"Wake trimming: keeping epochs {first_block}-{last_block - 1} "
                  f"({trimmed} epochs / {trimmed * 30 / 3600:.1f} h of surrounding Wake removed)")
        else:
            print("Wake trimming: no sleep epochs found, keeping the full recording.")

    for i in range(first_block, last_block):
        start_sec = i * 30
        end_sec = (i + 1) * 30
        
        # Check label for this epoch (we check start of epoch, standard AASM labels map to 30s epochs)
        label = second_labels[start_sec]
        if label == -1:
            skipped_blocks += 1
            continue # Skip invalid/unknown labels (e.g. "?", movement, or out of range)
            
        start_sample = start_sec * resample_rate
        end_sample = end_sec * resample_rate
        
        epoch_signal = raw_data[0, start_sample:end_sample]
        
        # Z-score normalization per 30-second epoch
        mean = np.mean(epoch_signal)
        std = np.std(epoch_signal)
        if std == 0:
            std = 1e-8
        normalized_epoch = (epoch_signal - mean) / std
        
        # Reshape to 30 1-second windows (each of size 100 samples)
        # shape: (30, 100)
        epoch_windows = normalized_epoch.reshape(30, samples_per_1s)
        
        # Replicate label for all 30 seconds
        epoch_labels = np.full(30, label, dtype=int)
        
        processed_signals.append(epoch_windows)
        processed_labels.append(epoch_labels)
        
    if len(processed_signals) == 0:
        raise ValueError("No valid annotated sleep epochs were processed.")
        
    x = np.vstack(processed_signals) # Shape: (N_seconds, 100)
    y = np.concatenate(processed_labels) # Shape: (N_seconds,)
    
    print(f"Processed: {len(x)} seconds of data (skipped {skipped_blocks * 30} seconds of invalid/noise stages)")
    return x, y

def main():
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF EDF files.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--subject", type=str, default=None, help="Process specific subject ID (e.g. '01').")
    parser.add_argument("--all", action="store_true", help="Process all available subject files.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Override raw data directory.")
    parser.add_argument("--processed_dir", type=str, default=None, help="Override processed data directory.")
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    raw_dir = args.raw_dir or config['data']['raw_dir']
    processed_dir = args.processed_dir or config['data']['processed_dir']
    target_channel = config['data']['target_channel']
    resample_rate = config['data']['resample_rate']
    wake_trim_minutes = config['data'].get('wake_trim_minutes', 30)
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all PSG files
    psg_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
    if not psg_files:
        print(f"No PSG files (*PSG.edf) found in raw directory: {raw_dir}")
        print("Please check your file paths and match the Sleep-EDF Expanded naming structure.")
        return
        
    print(f"Found {len(psg_files)} PSG file(s) in raw directory.")
    
    # Match each PSG file with its corresponding Hypnogram file
    pairs = []
    for psg_file in psg_files:
        psg_name = os.path.basename(psg_file)
        
        # Standard Sleep-EDF Expanded naming:
        # PSG: SC4xx1E0-PSG.edf -> Hypnogram: SC4xx1EC-Hypnogram.edf
        # Let's try replacing 0-PSG with C-Hypnogram
        hypno_name = psg_name.replace("0-PSG.edf", "C-Hypnogram.edf")
        hypno_path = os.path.join(raw_dir, hypno_name)
        
        if not os.path.exists(hypno_path):
            # Fallback search by subject prefix
            prefix = psg_name.split("-")[0][:-2] # Extract SC4xxx prefix
            hypno_matches = glob.glob(os.path.join(raw_dir, f"{prefix}*Hypnogram.edf"))
            if hypno_matches:
                hypno_path = hypno_matches[0]
            else:
                print(f"Warning: Could not find matching hypnogram for PSG {psg_name}. Skipping.")
                continue
                
        # Extract subject ID
        # e.g., SC4001E0-PSG.edf -> subject "00"
        match = re.search(r'(SC|SN)4(\d{2})', psg_name, re.IGNORECASE)
        sub_id = match.group(2) if match else psg_name.split("-")[0]
        
        pairs.append((sub_id, psg_file, hypno_path))
        
    # Filter by subject ID if specified
    if args.subject:
        pairs = [p for p in pairs if p[0] == args.subject]
        if not pairs:
            print(f"No files found for subject ID: {args.subject}")
            return
    elif not args.all:
        print("Please specify a subject with --subject <id> or run on all files with --all.")
        print(f"Available subjects in directory: {sorted(list(set(p[0] for p in pairs)))}")
        return
        
    # Group by subject so both nights (e.g. SC4001/SC4002 -> subject 00) go into one
    # file instead of the second night overwriting the first.
    from collections import defaultdict
    grouped = defaultdict(list)
    for sub_id, psg_path, hypno_path in pairs:
        grouped[sub_id].append((psg_path, hypno_path))

    for sub_id, recordings in sorted(grouped.items()):
        print(f"\n=========================================")
        print(f"Processing Subject: {sub_id}  ({len(recordings)} recording(s)/night(s))")
        print(f"=========================================")
        xs, ys = [], []
        for psg_path, hypno_path in sorted(recordings):  # sorted -> deterministic night order
            try:
                x, y = process_subject(psg_path, hypno_path, target_channel, resample_rate,
                                       wake_trim_minutes=wake_trim_minutes)
                xs.append(x)
                ys.append(y)
            except Exception as e:
                print(f"Error processing recording {os.path.basename(psg_path)}: {str(e)}")
                import traceback
                traceback.print_exc()

        if not xs:
            print(f"No valid recordings processed for subject {sub_id}. Skipping.")
            continue

        x = np.concatenate(xs, axis=0)  # (total_seconds, 100)
        y = np.concatenate(ys, axis=0)  # (total_seconds,)
        out_path = os.path.join(processed_dir, f"subject_{sub_id}.npz")
        np.savez_compressed(out_path, x=x, y=y)
        print(f"Successfully saved subject {sub_id} ({len(x)} seconds from {len(xs)} night(s)) to {out_path}")

if __name__ == "__main__":
    main()
